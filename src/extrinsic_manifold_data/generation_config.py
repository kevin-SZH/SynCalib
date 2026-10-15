"""Parse and validate self-contained dataset generation tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from .config_sections import config_section_fingerprint, load_config_mapping
from .sample_generation import NominalExtrinsicInput, load_nominal_extrinsic_config
from .sample_generation import (
    DIRECT_LEFT_EULER_BOX,
    DIRECT_RESIDUAL_SAMPLER_MODES,
    FIXED_IDENTITY,
    RESIDUAL_SAMPLER_MODES,
    direct_euler_box_max_geodesic_deg,
)
from .sensors import (
    RECTIFIED_1600_PROFILE_NAME,
    RECTIFIED_1600_WIDE_PROFILE_NAME,
    load_training_sensor_config,
    training_sensor_config_fingerprint,
)
from .pair_pool import SPLIT_NAMES, SourceManifestConfig, load_source_manifest_config
from .sample_generation import (
    JOINT_CELL_DISTRIBUTION,
    JOINT_CELL_SAMPLING,
    TRANSLATION_INTERVAL_SAMPLING,
    TRANSLATION_RADIUS_DISTRIBUTION,
)


STOCHASTIC_TASK_SCHEMA = "stochastic_generation_task_v5"
LEGACY_STOCHASTIC_TASK_SCHEMA = "stochastic_generation_task_v4"
STOCHASTIC_TASK_SCHEMAS = (
    LEGACY_STOCHASTIC_TASK_SCHEMA,
    STOCHASTIC_TASK_SCHEMA,
)

NETWORK_TARGETS = ("calibnet",)
CALIBNET_LEVELS = ("single",)
CALIBNET_FIXED_LEVELS = ("fixed_baseline",)
CALIBNET_DIRECT_LEVELS = ("perturbed_t0",)
COVERAGE_SCOPE = "per_curriculum_level_and_split_across_scenes"

FIXED_IDENTITY_COVERAGE = "fixed_identity"
DIRECT_EULER_BOX_COVERAGE = "direct_euler_component_box"
GEODESIC_PRODUCT_BALL_COVERAGE = "geodesic_product_ball"

SCALE_DISTRIBUTION = "solved_from_target_translation"
VIEW_ROTATION_DISTRIBUTION = "target_translation_conditioned_uniform_roll"
STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION = (
    "target_translation_conditioned_strict_level_rig"
)
VIEW_ROTATION_DISTRIBUTIONS = (
    VIEW_ROTATION_DISTRIBUTION,
    STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION,
)

REPORT_ONLY = "report_only"
TRANSLATION_RADIUS_BALANCED = "translation_radius_balanced"
JOINT_BALANCED = "joint_balanced"
RETRY_SAME_RESIDUAL = "retry_same_residual"
EXHAUST_SCALE_FEASIBLE_PAIRS_THEN_RESAMPLE = (
    "exhaust_scale_feasible_pairs_then_resample"
)
ACCEPTED_DISTRIBUTION_CONTROLS = (
    REPORT_ONLY,
    TRANSLATION_RADIUS_BALANCED,
    JOINT_BALANCED,
)


DEFAULT_MAX_CONSECUTIVE_ATTEMPTS_WITHOUT_ACCEPTANCE = 500


PHYSICAL_COPY_REUSE = "physical_copy"


SOURCE_PRIMARY_ACCEPTANCE = "source_primary_acceptance"


@dataclass(frozen=True)
class PairedLidarExportConfig:
    """Represent paired lidar export config."""

    output_suffix: str
    sensor_profile: str
    vertical_fov_deg: Tuple[float, float]
    image_and_text_reuse: str
    acceptance_policy: str
    regenerate_from_raw_cubemap_depth: bool

    def to_metadata(self) -> dict:
        """Return a metadata representation."""
        return {
            "output_suffix": self.output_suffix,
            "sensor_profile": self.sensor_profile,
            "vertical_fov_deg": list(self.vertical_fov_deg),
            "image_and_text_reuse": self.image_and_text_reuse,
            "acceptance_policy": self.acceptance_policy,
            "regenerate_from_raw_cubemap_depth": (
                self.regenerate_from_raw_cubemap_depth
            ),
        }


@dataclass(frozen=True)
class ResidualSamplerConfig:
    """Represent residual sampler config."""

    mode: str
    rejection_policy: str = RETRY_SAME_RESIDUAL
    max_samples_per_residual: int = 1
    max_realization_attempts_per_residual: Optional[int] = None

    def to_metadata(self) -> dict:
        """Return a metadata representation."""
        value = {"mode": self.mode}
        if self.mode == FIXED_IDENTITY:
            value.update({
                "source_implementation": (
                    "ExtrinsicManifoldData fixed physical extrinsic sampler"
                ),
                "true_extrinsic_policy": "Ti_equals_T0",
                "generation_time_perturbation": False,
                "rejection_policy": self.rejection_policy,
                "translation_scale_policy": (
                    "fixed_extrinsic_online_baseline_exception"
                ),
            })
        elif self.mode == DIRECT_LEFT_EULER_BOX:
            value.update({
                "source_implementation": (
                    "ExtrinsicManifoldData direct left Euler component box"
                ),
                "supervised_transform": "delta_T",
                "composition": "Ti_equals_delta_T_times_T0",
                "rotation_parameterization": (
                    "Rz_yaw_times_Ry_pitch_times_Rx_roll"
                ),
                "rejection_policy": self.rejection_policy,
                "max_samples_per_residual": self.max_samples_per_residual,
                "max_realization_attempts_per_residual": (
                    self.max_realization_attempts_per_residual
                ),
                "translation_scale_policy": (
                    "fixed_absolute_per_component_user_exception"
                ),
            })
        else:
            value.update({
                "rotation_parameterization": "so3_axis_angle_geodesic_ball",
                "translation_scale_policy": "proportional_to_norm_t0",
            })
        return value


@dataclass(frozen=True)
class RandomPairSamplingConfig:
    """Represent random pair sampling config."""

    max_pose_translation_distance_m: float
    ordered_frame_roles: bool


@dataclass(frozen=True)
class GenerationRuntimeConfig:
    """Represent generation runtime config."""

    depth_cache_max_mb: int
    rgb_cache_max_mb: int
    opencv_threads_per_worker: int
    numeric_threads_per_worker: int
    progress_interval_seconds: float
    max_consecutive_attempts_without_acceptance: Optional[int]
    incremental_persistence: bool
    direct_output_persistence: bool

    @property
    def effective_max_consecutive_attempts_without_acceptance(self) -> int:
        """Return effective max consecutive attempts without acceptance."""
        configured = self.max_consecutive_attempts_without_acceptance
        if configured is None:
            return DEFAULT_MAX_CONSECUTIVE_ATTEMPTS_WITHOUT_ACCEPTANCE
        return configured


@dataclass(frozen=True)
class GenerationLevel:
    """Represent generation level."""

    name: str
    max_rotation_deg: float
    perturbation_mode: str
    alpha_t: float | None
    t_min_floor_m: float | None
    max_translation_component_m: float | None
    accepted_samples_per_sequence: int

    def effective_max_translation(self, nominal: np.ndarray) -> float:
        """Return effective max translation."""
        if self.perturbation_mode in DIRECT_RESIDUAL_SAMPLER_MODES:
            if self.max_translation_component_m is None:
                raise ValueError("component-box level lacks translation limit")
            return self.max_translation_component_m
        if self.alpha_t is None or self.t_min_floor_m is None:
            raise ValueError("geodesic level lacks proportional translation")
        return max(
            self.alpha_t * float(np.linalg.norm(nominal[:3, 3])),
            self.t_min_floor_m,
        )

    def residual_translation_normalization_m(
        self, nominal: np.ndarray
    ) -> float:
        """Return residual translation normalization m."""
        limit = self.effective_max_translation(nominal)
        if self.perturbation_mode in DIRECT_RESIDUAL_SAMPLER_MODES:

            return float(np.sqrt(3.0) * limit)
        return limit

    @property
    def max_geodesic_rotation_deg(self) -> float:
        """Return max geodesic rotation deg."""
        if self.perturbation_mode == FIXED_IDENTITY:
            maximum = float(np.sqrt(3.0) * self.max_rotation_deg)
            if maximum >= 180.0:
                raise ValueError(
                    "fixed online rotation-vector box exceeds SO(3) limit"
                )
            return maximum
        if self.perturbation_mode == DIRECT_LEFT_EULER_BOX:
            return direct_euler_box_max_geodesic_deg(self.max_rotation_deg)
        return self.max_rotation_deg

    @property
    def limit_metadata(self) -> dict:
        """Return limit metadata."""
        if self.perturbation_mode == FIXED_IDENTITY:
            return {
                "true_extrinsic_policy": "Ti_equals_T0",
                "generation_time_perturbation": False,
                "online_max_rotation_vector_component_deg": (
                    self.max_rotation_deg
                ),
                "online_max_geodesic_rotation_deg": (
                    self.max_geodesic_rotation_deg
                ),
                "online_max_translation_component_m": (
                    self.max_translation_component_m
                ),
                "online_perturbation_semantics": (
                    "CalibNet UniformTransformSE3 with shared random "
                    "rotation and translation magnitudes"
                ),
            }
        if self.perturbation_mode == DIRECT_LEFT_EULER_BOX:
            return {
                "max_euler_component_deg": self.max_rotation_deg,
                "max_geodesic_rotation_deg": self.max_geodesic_rotation_deg,
                "rotation_limit_semantics": "per_euler_xyz_component_box",
                "max_translation_component_m": (
                    self.max_translation_component_m
                ),
                "translation_limit_semantics": (
                    "fixed_absolute_per_named_component_box"
                ),
            }
        return {
            "max_geodesic_rotation_deg": self.max_rotation_deg,
            "rotation_limit_semantics": "axis_angle_geodesic_ball",
            "alpha_t": self.alpha_t,
            "t_min_floor_m": self.t_min_floor_m,
            "translation_limit_semantics": (
                "max(alpha_t * norm(T0.translation), t_min_floor_m)"
            ),
        }


@dataclass(frozen=True)
class GenerationTask:
    """Represent generation task."""

    target_network: str
    nominal: NominalExtrinsicInput
    nominal_config: Path
    source: SourceManifestConfig
    source_config: Path
    pair_quality_config: Path
    sensor_config: Path
    quality_config: Path
    coverage_config: Path
    seed: int
    pair_sampling: RandomPairSamplingConfig
    runtime: GenerationRuntimeConfig
    max_attempts_per_accepted: Optional[int]
    max_accepted_per_pair: int
    scale_min: float
    scale_max: float
    scale_distribution: str
    scale_by_scene: Dict[str, Tuple[float, float]]
    residual_sampler: ResidualSamplerConfig
    accepted_distribution_control: str
    translation_distribution: str
    translation_interval_sampling: str
    rotation_distribution: str
    joint_cell_sampling: str
    joint_cell_max_consecutive_failures: int
    joint_radius_subintervals_per_interval: int
    view_rotation_distribution: str
    max_level_rig_tilt_deg: Optional[float]
    camera_optical_axis_latitude_band_enabled: bool
    max_camera_optical_axis_latitude_deg: Optional[float]
    paired_lidar_export: Optional[PairedLidarExportConfig]
    all_configured_scenes_per_level: bool
    coverage_scope: str
    config_fingerprints: Dict[str, str]
    levels: Tuple[GenerationLevel, ...]
    schema_version: str = STOCHASTIC_TASK_SCHEMA

    def scale_range_for_scene(self, scene: str) -> Tuple[float, float]:
        """Return scale range for scene."""
        if scene not in self.source.scenes:
            raise KeyError(f"scene is absent from task source: {scene}")
        return self.scale_by_scene.get(
            scene, (self.scale_min, self.scale_max)
        )

    def jobs(self) -> List[dict]:
        """Return jobs."""
        return [
            {
                "target_network": self.target_network,
                "level": level.name,
                "split": split,
                "scene": scene,
                "trajectory": self.source.trajectory,
                "accepted_sample_target": level.accepted_samples_per_sequence,
                "scale_range": list(self.scale_range_for_scene(scene)),
                "effective_max_translation_m": (
                    level.effective_max_translation(self.nominal.matrix)
                ),
                **level.limit_metadata,
            }
            for level in self.levels
            for split in SPLIT_NAMES
            for scene in self.source.scene_splits[split]
        ]

    def to_metadata(self) -> dict:
        """Return a metadata representation."""
        return {
            "schema_version": self.schema_version,
            "target_network": self.target_network,
            "nominal_extrinsic": self.nominal.to_metadata(),
            "seed": self.seed,
            "pair_sampling": {
                "max_pose_translation_distance_m": (
                    self.pair_sampling.max_pose_translation_distance_m
                ),
                "ordered_frame_roles": self.pair_sampling.ordered_frame_roles,
                "distribution": (
                    "all_ordered_pairs_within_raw_pose_translation_radius"
                ),
            },
            "runtime": {
                "depth_cache_max_mb": self.runtime.depth_cache_max_mb,
                "rgb_cache_max_mb": self.runtime.rgb_cache_max_mb,
                "opencv_threads_per_worker": (
                    self.runtime.opencv_threads_per_worker
                ),
                "numeric_threads_per_worker": (
                    self.runtime.numeric_threads_per_worker
                ),
                "progress_interval_seconds": (
                    self.runtime.progress_interval_seconds
                ),
                "max_consecutive_attempts_without_acceptance": (
                    self.runtime.max_consecutive_attempts_without_acceptance
                ),
                "incremental_persistence": (
                    self.runtime.incremental_persistence
                ),
                "direct_output_persistence": (
                    self.runtime.direct_output_persistence
                ),
                "pair_pool_cache": "output_root/.cache/qualified_pairs",
                "stage_timing": True,
            },
            "scale_sampling": {
                "minimum": self.scale_min,
                "maximum": self.scale_max,
                "distribution": self.scale_distribution,
                "by_scene": {
                    scene: {"minimum": bounds[0], "maximum": bounds[1]}
                    for scene, bounds in sorted(self.scale_by_scene.items())
                },
            },
            "residual_sampler": self.residual_sampler.to_metadata(),
            "accepted_distribution_control": (
                self.accepted_distribution_control
            ),
            "coverage_enforcement": (
                "report_only"
                if self.accepted_distribution_control == REPORT_ONLY
                else "strict"
            ),
            "translation_distribution": self.translation_distribution,
            "translation_interval_sampling": self.translation_interval_sampling,
            "max_attempts_per_accepted": self.max_attempts_per_accepted,
            "max_accepted_per_pair": self.max_accepted_per_pair,
            "rotation_distribution": self.rotation_distribution,
            "joint_cell_sampling": self.joint_cell_sampling,
            "joint_cell_max_consecutive_failures": (
                self.joint_cell_max_consecutive_failures
            ),
            "joint_radius_subintervals_per_interval": (
                self.joint_radius_subintervals_per_interval
            ),
            "view_rotation_distribution": self.view_rotation_distribution,
            "max_level_rig_tilt_deg": self.max_level_rig_tilt_deg,
            "camera_optical_axis_latitude_band": {
                "enabled": self.camera_optical_axis_latitude_band_enabled,
                "max_abs_latitude_deg": (
                    self.max_camera_optical_axis_latitude_deg
                ),
            },
            "paired_lidar_export": (
                self.paired_lidar_export.to_metadata()
                if self.paired_lidar_export is not None else None
            ),
            "all_configured_scenes_per_level": (
                self.all_configured_scenes_per_level
            ),
            "coverage_scope": self.coverage_scope,
            "config_fingerprints": dict(self.config_fingerprints),
            "levels": [
                {
                    "name": level.name,
                    **level.limit_metadata,
                    "effective_max_translation_m": (
                        level.effective_max_translation(self.nominal.matrix)
                    ),
                    "residual_translation_normalization_m": (
                        level.residual_translation_normalization_m(
                            self.nominal.matrix
                        )
                    ),
                    "accepted_samples_per_sequence": (
                        level.accepted_samples_per_sequence
                    ),
                }
                for level in self.levels
            ],
        }


def _resolve_existing(config_path: Path, raw: dict, field: str) -> Path:
    """Resolve existing."""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (config_path.parent / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _resolve_component(
    config_path: Path,
    raw: dict,
    *,
    section: str,
    reference_field: str,
) -> Path:
    """Resolve component."""
    has_section = section in raw
    has_reference = reference_field in raw
    if has_section == has_reference:
        raise ValueError(
            f"configure exactly one of {section} or {reference_field}"
        )
    if has_section:
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping")
        return config_path
    return _resolve_existing(config_path, raw, reference_field)


def _fingerprint(path: Path) -> str:
    """Return fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pair_sampling(raw: dict) -> RandomPairSamplingConfig:
    """Load pair sampling."""
    value = raw.get("pair_sampling")
    if not isinstance(value, dict):
        raise ValueError("pair_sampling must be a mapping")
    expected_fields = {
        "max_pose_translation_distance_m",
        "ordered_frame_roles",
    }
    if set(value) != expected_fields:
        raise ValueError(
            "pair_sampling fields must be exactly "
            f"{sorted(expected_fields)}; gap-based enumeration is unsupported"
        )
    max_distance_m = float(value["max_pose_translation_distance_m"])
    if not np.isfinite(max_distance_m) or max_distance_m <= 0.0:
        raise ValueError(
            "pair_sampling.max_pose_translation_distance_m must be finite "
            "and positive"
        )
    if value.get("ordered_frame_roles") is not True:
        raise ValueError(
            "pair_sampling.ordered_frame_roles must be true so depth/RGB roles "
            "are sampled explicitly in both temporal directions"
        )
    return RandomPairSamplingConfig(
        max_pose_translation_distance_m=max_distance_m,
        ordered_frame_roles=True,
    )


def _validate_coverage_mode(path: Path, residual_mode: str) -> None:
    """Validate coverage mode."""
    raw, _, _ = load_config_mapping(
        path, section="coverage", description="coverage config"
    )
    expected = {
        FIXED_IDENTITY: FIXED_IDENTITY_COVERAGE,
        DIRECT_LEFT_EULER_BOX: DIRECT_EULER_BOX_COVERAGE,
    }.get(residual_mode, GEODESIC_PRODUCT_BALL_COVERAGE)
    actual = str(raw.get("mode", "")).strip()
    if actual != expected:
        raise ValueError(
            f"coverage.mode must be {expected!r} for residual mode "
            f"{residual_mode!r}, got {actual!r}"
        )


def _load_runtime(raw: dict) -> GenerationRuntimeConfig:
    """Load runtime."""
    value = raw.get("runtime") or {}
    if not isinstance(value, dict):
        raise ValueError("runtime must be a mapping")
    config = GenerationRuntimeConfig(
        depth_cache_max_mb=int(value.get("depth_cache_max_mb", 6144)),
        rgb_cache_max_mb=int(value.get("rgb_cache_max_mb", 2048)),
        opencv_threads_per_worker=int(
            value.get("opencv_threads_per_worker", 1)
        ),
        numeric_threads_per_worker=int(
            value.get("numeric_threads_per_worker", 1)
        ),
        progress_interval_seconds=float(
            value.get("progress_interval_seconds", 10.0)
        ),
        max_consecutive_attempts_without_acceptance=(
            None
            if value.get("max_consecutive_attempts_without_acceptance") is None
            else int(value["max_consecutive_attempts_without_acceptance"])
        ),
        incremental_persistence=value.get(
            "incremental_persistence", False
        ),
        direct_output_persistence=value.get(
            "direct_output_persistence", False
        ),
    )
    if config.depth_cache_max_mb < 1 or config.rgb_cache_max_mb < 1:
        raise ValueError("runtime cache budgets must be positive MiB values")
    if config.opencv_threads_per_worker < 1:
        raise ValueError("runtime.opencv_threads_per_worker must be positive")
    if config.numeric_threads_per_worker < 1:
        raise ValueError("runtime.numeric_threads_per_worker must be positive")
    if config.progress_interval_seconds <= 0.0:
        raise ValueError("runtime.progress_interval_seconds must be positive")
    if (
        config.max_consecutive_attempts_without_acceptance is not None
        and config.max_consecutive_attempts_without_acceptance < 1
    ):
        raise ValueError(
            "runtime.max_consecutive_attempts_without_acceptance must be "
            "positive when configured"
        )
    if not isinstance(config.incremental_persistence, bool):
        raise ValueError("runtime.incremental_persistence must be boolean")
    if not isinstance(config.direct_output_persistence, bool):
        raise ValueError("runtime.direct_output_persistence must be boolean")
    if (
        config.direct_output_persistence
        and not config.incremental_persistence
    ):
        raise ValueError(
            "runtime.direct_output_persistence requires "
            "incremental_persistence=true"
        )
    return config


def _load_residual_sampler(
    sampling: dict,
    target_network: str,
) -> ResidualSamplerConfig:
    """Load residual sampler."""
    value = sampling.get("residual_sampler")
    if not isinstance(value, dict):
        raise ValueError("sampling.residual_sampler must be a mapping")
    mode = str(value.get("mode", "")).strip()
    if mode not in RESIDUAL_SAMPLER_MODES:
        raise ValueError(
            "sampling.residual_sampler.mode must be one of "
            f"{RESIDUAL_SAMPLER_MODES}"
        )
    if mode == FIXED_IDENTITY:
        if target_network != "calibnet":
            raise ValueError(
                "fixed_identity is only valid for CalibNet baseline tasks"
            )
        unexpected = set(value) - {"mode"}
        if unexpected:
            raise ValueError(
                "fixed identity sampler has unsupported fields: "
                f"{unexpected}"
            )
        return ResidualSamplerConfig(mode=mode)
    if mode == DIRECT_LEFT_EULER_BOX:
        allowed = {
            "mode",
            "rejection_policy",
            "max_samples_per_residual",
            "max_realization_attempts_per_residual",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(
                "direct Euler sampler has unsupported fields: "
                f"{sorted(unexpected)}"
            )
        rejection_policy = str(value.get("rejection_policy", "")).strip()
        if rejection_policy != EXHAUST_SCALE_FEASIBLE_PAIRS_THEN_RESAMPLE:
            raise ValueError(
                "direct Euler sampler rejection_policy must be "
                f"{EXHAUST_SCALE_FEASIBLE_PAIRS_THEN_RESAMPLE}"
            )
        max_samples = int(value.get("max_samples_per_residual", 0))
        max_attempts = int(value.get(
            "max_realization_attempts_per_residual", 0
        ))
        if max_samples < 1 or max_attempts < 1:
            raise ValueError(
                "direct Euler residual reuse and attempt budgets must be positive"
            )
        return ResidualSamplerConfig(
            mode=mode,
            rejection_policy=rejection_policy,
            max_samples_per_residual=max_samples,
            max_realization_attempts_per_residual=max_attempts,
        )
    unexpected = set(value) - {"mode"}
    if unexpected:
        raise ValueError(
            f"geodesic residual sampler has unsupported fields: {unexpected}"
        )
    return ResidualSamplerConfig(mode=mode)


def _load_paired_lidar_export(
    raw: dict,
    schema_version: str,
    residual_mode: str,
) -> Optional[PairedLidarExportConfig]:
    """Load paired LiDAR export."""
    value = raw.get("paired_lidar_export")
    if value is None:
        return None
    if schema_version != STOCHASTIC_TASK_SCHEMA:
        raise ValueError(
            "paired_lidar_export requires stochastic_generation_task_v5"
        )
    if residual_mode != DIRECT_LEFT_EULER_BOX:
        raise ValueError(
            "paired_lidar_export is only valid for direct_left_euler_box tasks"
        )
    if not isinstance(value, dict):
        raise ValueError("paired_lidar_export must be a mapping")
    expected_fields = {
        "output_suffix",
        "sensor_profile",
        "vertical_fov_deg",
        "image_and_text_reuse",
        "acceptance_policy",
        "regenerate_from_raw_cubemap_depth",
    }
    unexpected = set(value) - expected_fields
    missing = expected_fields - set(value)
    if unexpected or missing:
        raise ValueError(
            "paired_lidar_export fields mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    output_suffix = str(value["output_suffix"])
    if (
        not output_suffix.startswith("_")
        or output_suffix in {"_", "__"}
        or "/" in output_suffix
        or "\\" in output_suffix
        or any(part == ".." for part in output_suffix.split("_"))
    ):
        raise ValueError(
            "paired_lidar_export.output_suffix must be a safe underscore suffix"
        )
    vertical_fov = tuple(
        float(item) for item in value["vertical_fov_deg"]
    )
    if len(vertical_fov) != 2 or not np.allclose(
        vertical_fov, (-25.0, 25.0), rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "paired_lidar_export.vertical_fov_deg must be [-25.0, 25.0]"
        )
    config = PairedLidarExportConfig(
        output_suffix=output_suffix,
        sensor_profile=str(value["sensor_profile"]),
        vertical_fov_deg=vertical_fov,
        image_and_text_reuse=str(value["image_and_text_reuse"]),
        acceptance_policy=str(value["acceptance_policy"]),
        regenerate_from_raw_cubemap_depth=value[
            "regenerate_from_raw_cubemap_depth"
        ],
    )
    if config.sensor_profile != RECTIFIED_1600_WIDE_PROFILE_NAME:
        raise ValueError(
            "paired_lidar_export.sensor_profile must be "
            f"{RECTIFIED_1600_WIDE_PROFILE_NAME}"
        )
    if config.image_and_text_reuse != PHYSICAL_COPY_REUSE:
        raise ValueError(
            f"paired image_and_text_reuse must be {PHYSICAL_COPY_REUSE}"
        )
    if config.acceptance_policy != SOURCE_PRIMARY_ACCEPTANCE:
        raise ValueError(
            f"paired acceptance_policy must be {SOURCE_PRIMARY_ACCEPTANCE}"
        )
    if config.regenerate_from_raw_cubemap_depth is not True:
        raise ValueError(
            "paired LiDAR must be regenerated from raw cubemap depth"
        )
    return config


def _load_levels(
    raw: dict,
    target_network: str,
    perturbation_mode: str,
) -> Tuple[GenerationLevel, ...]:
    """Load levels."""
    curriculum = raw.get("curriculum")
    if not isinstance(curriculum, dict):
        raise ValueError("curriculum must be a mapping")
    values = curriculum.get("levels")
    if not isinstance(values, list) or not values:
        raise ValueError("curriculum.levels must be a non-empty list")
    levels = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"curriculum.levels[{index}] must be a mapping")
        if "max_rotation_deg" in value:
            raise ValueError(
                "max_rotation_deg is ambiguous; use max_euler_component_deg, "
                "max_rotation_vector_component_deg, or "
                "max_geodesic_rotation_deg"
            )
        fixed_identity_mode = perturbation_mode == FIXED_IDENTITY
        direct_residual_mode = (
            perturbation_mode in DIRECT_RESIDUAL_SAMPLER_MODES
        )
        quality_overrides = value.get("quality_overrides") or {}
        if not isinstance(quality_overrides, dict):
            raise ValueError(
                f"curriculum.levels[{index}].quality_overrides must be a mapping"
            )
        if quality_overrides:
            raise ValueError(
                "unsupported quality overrides: "
                f"{sorted(quality_overrides)}"
            )
        if fixed_identity_mode:
            rotation_field = "online_max_rotation_vector_component_deg"
        elif perturbation_mode == DIRECT_LEFT_EULER_BOX:
            rotation_field = "max_euler_component_deg"
        else:
            rotation_field = "max_geodesic_rotation_deg"
        if direct_residual_mode:
            forbidden = {"alpha_t", "t_min_floor_m"} & set(value)
            if forbidden:
                raise ValueError(
                    "component-box perturbation forbids proportional "
                    f"fields: {sorted(forbidden)}"
                )
            translation_field = (
                "online_max_translation_component_m"
                if fixed_identity_mode
                else "max_translation_component_m"
            )
            max_translation_component = float(
                value.get(translation_field, 0.0)
            )
            alpha_t = None
            t_min_floor = None
        else:
            if "max_translation_component_m" in value:
                raise ValueError(
                    "geodesic mode forbids max_translation_component_m"
                )
            max_translation_component = None
            alpha_t = float(value.get("alpha_t", 0.0))
            t_min_floor = float(value.get("t_min_floor_m", -1.0))
        level = GenerationLevel(
            name=str(value.get("name", "")).strip().lower(),
            max_rotation_deg=float(value.get(rotation_field, 0.0)),
            perturbation_mode=perturbation_mode,
            alpha_t=alpha_t,
            t_min_floor_m=t_min_floor,
            max_translation_component_m=max_translation_component,
            accepted_samples_per_sequence=int(
                value.get("accepted_samples_per_sequence", 0)
            ),
        )
        if not level.name:
            raise ValueError(f"curriculum.levels[{index}].name is required")
        if not 0.0 < level.max_rotation_deg < 180.0:
            raise ValueError(f"{level.name}: {rotation_field} must be in (0, 180)")
        if direct_residual_mode:
            if max_translation_component <= 0.0:
                raise ValueError(
                    f"{level.name}: {translation_field} must be positive"
                )
        else:
            if alpha_t is None or alpha_t <= 0.0:
                raise ValueError(f"{level.name}: alpha_t must be positive")
            if t_min_floor is None or t_min_floor < 0.0:
                raise ValueError(
                    f"{level.name}: t_min_floor_m must be non-negative"
                )
        if level.accepted_samples_per_sequence < 1:
            raise ValueError(
                f"{level.name}: accepted_samples_per_sequence must be positive"
            )
        levels.append(level)

    if perturbation_mode == FIXED_IDENTITY:
        expected = CALIBNET_FIXED_LEVELS
    elif perturbation_mode == DIRECT_LEFT_EULER_BOX:
        expected = CALIBNET_DIRECT_LEVELS
    else:
        expected = CALIBNET_LEVELS
    names = tuple(level.name for level in levels)
    if names != expected:
        raise ValueError(
            f"{target_network} curriculum levels must be exactly {expected}, got {names}"
        )
    return tuple(levels)


def load_generation_task(
    path: Union[str, Path],
) -> GenerationTask:
    """Load generation task."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("stochastic task config must be a YAML mapping")
    schema_version = str(raw.get("schema_version", ""))
    if schema_version not in STOCHASTIC_TASK_SCHEMAS:
        raise ValueError(
            f"schema_version must be one of {STOCHASTIC_TASK_SCHEMAS}"
        )
    target_network = str(raw.get("target_network", "")).strip().lower()
    if target_network not in NETWORK_TARGETS:
        raise ValueError(f"target_network must be one of {NETWORK_TARGETS}")

    nominal_config = _resolve_component(
        config_path,
        raw,
        section="nominal_extrinsic",
        reference_field="nominal_extrinsic_config",
    )
    source_config = _resolve_component(
        config_path, raw, section="source", reference_field="source_config"
    )
    pair_quality_config = _resolve_component(
        config_path,
        raw,
        section="pair_quality",
        reference_field="pair_quality_config",
    )
    sensor_config = _resolve_component(
        config_path, raw, section="sensor", reference_field="sensor_config"
    )
    quality_config = _resolve_component(
        config_path, raw, section="quality", reference_field="quality_config"
    )
    coverage_config = _resolve_component(
        config_path,
        raw,
        section="coverage",
        reference_field="coverage_config",
    )
    nominal = load_nominal_extrinsic_config(nominal_config)
    source = load_source_manifest_config(source_config)
    pair_sampling = _load_pair_sampling(raw)
    runtime = _load_runtime(raw)

    sampling = raw.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be a mapping")
    seed = int(sampling.get("seed", -1))
    if seed < 0:
        raise ValueError("sampling.seed must be non-negative")
    configured_max_attempts = sampling.get("max_attempts_per_accepted")
    max_attempts = (
        None
        if configured_max_attempts is None
        else int(configured_max_attempts)
    )
    if max_attempts is not None and max_attempts < 1:
        raise ValueError(
            "sampling.max_attempts_per_accepted must be null or positive"
        )
    max_accepted_per_pair = int(sampling.get("max_accepted_per_pair", 0))
    if max_accepted_per_pair < 1:
        raise ValueError("sampling.max_accepted_per_pair must be positive")
    scale = sampling.get("scale")
    if not isinstance(scale, dict):
        raise ValueError("sampling.scale must be a mapping")
    scale_min = float(scale.get("min", 0.0))
    scale_max = float(scale.get("max", 0.0))
    if not (
        np.isfinite(scale_min)
        and np.isfinite(scale_max)
        and 0.0 < scale_min < scale_max
    ):
        raise ValueError("sampling.scale requires 0 < min < max")
    configured_scene_scales = scale.get("by_scene", {})
    if not isinstance(configured_scene_scales, dict):
        raise ValueError("sampling.scale.by_scene must be a mapping")
    configured_scenes = set(source.scenes)
    unknown_scale_scenes = sorted(
        set(str(value) for value in configured_scene_scales) - configured_scenes
    )
    if unknown_scale_scenes:
        raise ValueError(
            "sampling.scale.by_scene contains scenes absent from source: "
            f"{unknown_scale_scenes}"
        )
    scale_by_scene: Dict[str, Tuple[float, float]] = {}
    for scene, bounds in configured_scene_scales.items():
        if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
            raise ValueError(
                f"sampling.scale.by_scene.{scene} must contain exactly min and max"
            )
        scene_min = float(bounds["min"])
        scene_max = float(bounds["max"])
        if not (
            np.isfinite(scene_min)
            and np.isfinite(scene_max)
            and 0.0 < scene_min < scene_max
        ):
            raise ValueError(
                f"sampling.scale.by_scene.{scene} requires finite 0 < min < max"
            )
        scale_by_scene[str(scene)] = (scene_min, scene_max)
    scale_distribution = str(scale.get("distribution", "")).strip()
    if scale_distribution != SCALE_DISTRIBUTION:
        raise ValueError(
            f"sampling.scale.distribution must be {SCALE_DISTRIBUTION}"
        )
    residual_sampler = _load_residual_sampler(sampling, target_network)
    _validate_coverage_mode(coverage_config, residual_sampler.mode)
    base_sensor = load_training_sensor_config(sensor_config)
    paired_lidar_export = _load_paired_lidar_export(
        raw, schema_version, residual_sampler.mode
    )
    if paired_lidar_export is not None and (
        base_sensor.profile != RECTIFIED_1600_PROFILE_NAME
        or tuple(base_sensor.lidar.vertical_fov_deg) != (-24.9, 2.0)
    ):
        raise ValueError(
            "paired LiDAR export requires the 1600 camera and standard "
            "[-24.9, 2.0] base profile"
        )
    if paired_lidar_export is not None and (
        not runtime.incremental_persistence
        or runtime.direct_output_persistence
    ):
        raise ValueError(
            "paired LiDAR export requires incremental_persistence=true and "
            "direct_output_persistence=false"
        )
    accepted_distribution_control = str(
        sampling.get("accepted_distribution_control", "")
    ).strip()
    if accepted_distribution_control not in ACCEPTED_DISTRIBUTION_CONTROLS:
        raise ValueError(
            "sampling.accepted_distribution_control must be one of "
            f"{ACCEPTED_DISTRIBUTION_CONTROLS}"
        )
    legacy_distribution_fields = {
        "translation_distribution",
        "translation_interval_sampling",
        "rotation_distribution",
    } & set(sampling)
    if legacy_distribution_fields:
        raise ValueError(
            "distribution behavior is selected by residual_sampler and "
            "accepted_distribution_control; remove legacy fields "
            f"{sorted(legacy_distribution_fields)}"
        )
    if residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES:
        if accepted_distribution_control != REPORT_ONLY:
            raise ValueError(
                "direct residual perturbations require report_only accepted "
                "distribution control"
            )
        translation_distribution = (
            "fixed_identity_true_extrinsic"
            if residual_sampler.mode == FIXED_IDENTITY
            else "direct_left_euler_translation_component_box"
        )
        translation_interval_sampling = ""
        rotation_distribution = residual_sampler.mode
        joint_cell_sampling = ""
        joint_max_failures = 0
        joint_radius_subintervals = 0
    else:
        translation_distribution = TRANSLATION_RADIUS_DISTRIBUTION
        translation_interval_sampling = TRANSLATION_INTERVAL_SAMPLING
        if accepted_distribution_control == REPORT_ONLY:
            rotation_distribution = "uniform_so3_geodesic_ball"
            joint_cell_sampling = ""
            joint_max_failures = 0
            joint_radius_subintervals = 0
        elif accepted_distribution_control == TRANSLATION_RADIUS_BALANCED:
            rotation_distribution = "uniform_so3_geodesic_ball"
            joint_cell_sampling = ""
            joint_max_failures = 0
            joint_radius_subintervals = 0
        else:
            rotation_distribution = JOINT_CELL_DISTRIBUTION
            joint_cell_sampling = str(
                sampling.get("joint_cell_sampling", "")
            ).strip()
            joint_max_failures = int(
                sampling.get("joint_cell_max_consecutive_failures", 0)
            )
            joint_radius_subintervals = int(
                sampling.get("joint_radius_subintervals_per_interval", 1)
            )
            if joint_cell_sampling != JOINT_CELL_SAMPLING:
                raise ValueError(
                    f"sampling.joint_cell_sampling must be {JOINT_CELL_SAMPLING}"
                )
            if joint_max_failures < 1:
                raise ValueError(
                    "sampling.joint_cell_max_consecutive_failures must be positive"
                )
            if joint_radius_subintervals not in (1, 4):
                raise ValueError(
                    "sampling.joint_radius_subintervals_per_interval must be "
                    "one for legacy sampling or four for fine joint quotas"
                )
            configured_scene_count = sum(
                len(scenes) for scenes in source.scene_splits.values()
            )
            if (
                joint_radius_subintervals == 4
                and configured_scene_count > joint_radius_subintervals
            ):
                raise ValueError(
                    "fine joint quotas support at most four scene partitions"
                )
    view_rotation_distribution = str(
        sampling.get("view_rotation_distribution", "")
    ).strip()
    if view_rotation_distribution not in VIEW_ROTATION_DISTRIBUTIONS:
        raise ValueError(
            "sampling.view_rotation_distribution must be one of "
            f"{VIEW_ROTATION_DISTRIBUTIONS}"
        )
    if "max_camera_optical_axis_latitude_deg" in sampling:
        raise ValueError(
            "use sampling.camera_optical_axis_latitude_band instead of the "
            "legacy flat latitude field"
        )
    configured_latitude_band = sampling.get(
        "camera_optical_axis_latitude_band"
    )
    if configured_latitude_band is None:
        camera_optical_axis_latitude_band_enabled = False
        max_camera_optical_axis_latitude_deg = None
    else:
        if not isinstance(configured_latitude_band, dict):
            raise ValueError(
                "sampling.camera_optical_axis_latitude_band must be a mapping"
            )
        expected_latitude_fields = {"enabled", "max_abs_latitude_deg"}
        if set(configured_latitude_band) != expected_latitude_fields:
            raise ValueError(
                "sampling.camera_optical_axis_latitude_band fields must be "
                f"exactly {sorted(expected_latitude_fields)}"
            )
        camera_optical_axis_latitude_band_enabled = (
            configured_latitude_band["enabled"]
        )
        if type(camera_optical_axis_latitude_band_enabled) is not bool:
            raise ValueError(
                "sampling.camera_optical_axis_latitude_band.enabled must be bool"
            )
        max_camera_optical_axis_latitude_deg = float(
            configured_latitude_band["max_abs_latitude_deg"]
        )
        if (
            not np.isfinite(max_camera_optical_axis_latitude_deg)
            or not 0.0 < max_camera_optical_axis_latitude_deg < 90.0
        ):
            raise ValueError(
                "camera optical-axis latitude limit must lie within (0, 90)"
            )
        if (
            camera_optical_axis_latitude_band_enabled
            and view_rotation_distribution != VIEW_ROTATION_DISTRIBUTION
        ):
            raise ValueError(
                "camera optical-axis latitude band requires uniform-roll views"
            )
    configured_level_rig_tilt = sampling.get("max_level_rig_tilt_deg")
    if view_rotation_distribution == STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION:
        if configured_level_rig_tilt is None:
            raise ValueError(
                "sampling.max_level_rig_tilt_deg is required for the "
                "level-rig constrained view distribution"
            )
        max_level_rig_tilt_deg = float(configured_level_rig_tilt)
        if (
            not np.isfinite(max_level_rig_tilt_deg)
            or not 0.0 < max_level_rig_tilt_deg <= 45.0
        ):
            raise ValueError(
                "sampling.max_level_rig_tilt_deg must lie within (0, 45]"
            )
    else:
        if configured_level_rig_tilt is not None:
            raise ValueError(
                "sampling.max_level_rig_tilt_deg is only valid for the "
                "level-rig constrained view distribution"
            )
        max_level_rig_tilt_deg = None
    if "view_orientation" in sampling:
        raise ValueError(
            "sampling.view_orientation is forbidden; generated views are "
            "filtered by structural quality instead of Euler-angle limits"
        )
    all_scenes = raw.get("all_configured_scenes_per_level")
    if all_scenes is not True:
        raise ValueError("all_configured_scenes_per_level must be true")
    coverage_scope = str(raw.get("coverage_scope", "")).strip()
    if coverage_scope != COVERAGE_SCOPE:
        raise ValueError(f"coverage_scope must be {COVERAGE_SCOPE}")

    levels = _load_levels(raw, target_network, residual_sampler.mode)
    if "required_reference_extrinsics" in raw:
        raise ValueError(
            "required_reference_extrinsics is not part of the two formal modes"
        )
    if "domain_matching" in raw:
        raise ValueError("domain_matching is not part of the two formal modes")
    config_fingerprints = {
        "task": _fingerprint(config_path),
        "nominal_extrinsic": config_section_fingerprint(
            nominal_config, "nominal_extrinsic"
        ),
        "source": config_section_fingerprint(source_config, "source"),
        "pair_quality": config_section_fingerprint(
            pair_quality_config, "pair_quality"
        ),
        "sensor": training_sensor_config_fingerprint(sensor_config),
        "generated_quality": config_section_fingerprint(
            quality_config, "quality"
        ),
        "coverage": config_section_fingerprint(
            coverage_config, "coverage"
        ),
    }
    return GenerationTask(
        target_network=target_network,
        nominal=nominal,
        nominal_config=nominal_config,
        source=source,
        source_config=source_config,
        pair_quality_config=pair_quality_config,
        sensor_config=sensor_config,
        quality_config=quality_config,
        coverage_config=coverage_config,
        seed=seed,
        pair_sampling=pair_sampling,
        runtime=runtime,
        max_attempts_per_accepted=max_attempts,
        max_accepted_per_pair=max_accepted_per_pair,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_distribution=scale_distribution,
        scale_by_scene=scale_by_scene,
        residual_sampler=residual_sampler,
        accepted_distribution_control=accepted_distribution_control,
        translation_distribution=translation_distribution,
        translation_interval_sampling=translation_interval_sampling,
        rotation_distribution=rotation_distribution,
        joint_cell_sampling=joint_cell_sampling,
        joint_cell_max_consecutive_failures=(
            joint_max_failures
        ),
        joint_radius_subintervals_per_interval=(
            joint_radius_subintervals
        ),
        view_rotation_distribution=view_rotation_distribution,
        max_level_rig_tilt_deg=max_level_rig_tilt_deg,
        camera_optical_axis_latitude_band_enabled=(
            camera_optical_axis_latitude_band_enabled
        ),
        max_camera_optical_axis_latitude_deg=(
            max_camera_optical_axis_latitude_deg
        ),
        paired_lidar_export=paired_lidar_export,
        all_configured_scenes_per_level=True,
        coverage_scope=coverage_scope,
        config_fingerprints=config_fingerprints,
        levels=levels,
        schema_version=schema_version,
    )
