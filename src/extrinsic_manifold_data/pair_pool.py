"""Build and restore qualified directed frame-pair pools."""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config_sections import load_config_mapping


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SourceManifestConfig:
    """Represent source manifest config."""

    dataset: str
    trajectory: str
    scene_splits: Dict[str, List[str]]

    def split_for_scene(self, scene: str) -> str:
        """Return split for scene."""
        for split in SPLIT_NAMES:
            if scene in self.scene_splits[split]:
                return split
        raise KeyError(f"scene is not assigned to a split: {scene}")

    @property
    def scenes(self) -> List[str]:
        """Return scenes."""
        return [
            scene
            for split in SPLIT_NAMES
            for scene in self.scene_splits[split]
        ]


def load_source_manifest_config(path: str | Path) -> SourceManifestConfig:
    """Load source manifest config."""
    raw, _, _ = load_config_mapping(
        path,
        section="source",
        description="source manifest config",
    )

    dataset = str(raw.get("dataset", "")).strip().lower()
    if dataset != "tartanground":
        raise ValueError("dataset must currently be 'tartanground'")
    trajectory = str(raw.get("trajectory", "")).strip()
    if not trajectory:
        raise ValueError("trajectory must be a non-empty string")

    values = raw.get("scene_splits")
    if not isinstance(values, dict) or set(values) != set(SPLIT_NAMES):
        raise ValueError("scene_splits must contain exactly train, val, and test")
    splits: Dict[str, List[str]] = {}
    owners: Dict[str, str] = {}
    for split in SPLIT_NAMES:
        scenes = values[split]
        if not isinstance(scenes, list):
            raise ValueError(f"scene_splits.{split} must be a list")
        normalized = []
        for value in scenes:
            scene = str(value).strip()
            if not scene:
                raise ValueError(f"scene_splits.{split} contains an empty scene")
            if scene in owners:
                raise ValueError(
                    f"scene {scene} appears in {owners[scene]} and {split}"
                )
            owners[scene] = split
            normalized.append(scene)
        splits[split] = normalized
    if not owners:
        raise ValueError("scene_splits must assign at least one scene")
    return SourceManifestConfig(
        dataset=dataset,
        trajectory=trajectory,
        scene_splits=splits,
    )


import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .omni import OmniSequence
from .sources.tartanground import scan_tartanground_sequence


SOURCE_MANIFEST_VERSION = "raw_omni_source_manifest_v2_xyz_flu"


def _relative(path: Path, data_root: Path) -> str:
    """Return relative."""
    return path.resolve().relative_to(data_root).as_posix()


def _sequence_payload(
    sequence: OmniSequence,
    split: str,
    data_root: Path,
) -> dict[str, Any]:
    """Return sequence payload."""
    return {
        "sequence_id": sequence.sequence_id,
        "scene": sequence.scene,
        "trajectory": sequence.trajectory,
        "split": split,
        "coordinate_system": {
            "axes": "X-forward, Y-left, Z-up",
            "pose": "world_from_pano",
            "point_convention": "column_vector",
            "azimuth": "atan2(y, x), positive from +X toward +Y",
            "camera_axes": "X-right, Y-down, Z-forward",
        },
        "pano_from_face": {
            face: rotation.tolist()
            for face, rotation in sequence.pano_from_face.items()
        },
        "source_report": sequence.source_report,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "world_from_pano": frame.world_from_pano.tolist(),
                "rgb_faces": {
                    face: _relative(path, data_root)
                    for face, path in frame.rgb_paths.items()
                },
                "depth_faces": {
                    face: _relative(path, data_root)
                    for face, path in frame.depth_paths.items()
                },
            }
            for frame in sequence.frames
        ],
    }


def build_source_manifest(
    data_root: str | Path,
    config_path: str | Path,
    output: str | Path,
    scenes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build source manifest."""
    data_root = Path(data_root).resolve()
    config = load_source_manifest_config(config_path)
    selected_scenes = config.scenes
    if scenes is not None:
        selected_scenes = [str(scene).strip() for scene in scenes]
        if not selected_scenes or any(not scene for scene in selected_scenes):
            raise ValueError("scene filter must contain non-empty scene names")
        if len(set(selected_scenes)) != len(selected_scenes):
            raise ValueError("scene filter contains duplicate scene names")
        unknown = sorted(set(selected_scenes) - set(config.scenes))
        if unknown:
            raise ValueError(
                f"scene filter contains scenes absent from split config: {unknown}"
            )
    selected_scene_set = set(selected_scenes)
    selected_splits = {
        split: [
            scene for scene in split_scenes if scene in selected_scene_set
        ]
        for split, split_scenes in config.scene_splits.items()
    }
    sequences = []
    for scene in selected_scenes:
        sequence = scan_tartanground_sequence(
            data_root, scene, config.trajectory
        )
        sequences.append(_sequence_payload(
            sequence, config.split_for_scene(scene), data_root
        ))

    combined = hashlib.sha256()
    for sequence in sorted(sequences, key=lambda value: value["sequence_id"]):
        combined.update(sequence["sequence_id"].encode("utf-8"))
        combined.update(
            sequence["source_report"]["input_fingerprint"].encode("ascii")
        )
    payload = {
        "version": SOURCE_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset,
        "data_root": str(data_root),
        "trajectory": config.trajectory,
        "scene_splits": selected_splits,
        "configured_scene_splits": config.scene_splits,
        "input_fingerprint": combined.hexdigest(),
        "sequence_count": len(sequences),
        "frame_count": sum(len(sequence["frames"]) for sequence in sequences),
        "sequences": sequences,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(output)
    return payload


import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .geometry import validate_transform
from .omni import OmniFrame, OmniSequence


def _source_path(data_root: Path, relative: str) -> Path:
    """Return source path."""
    path = (data_root / relative).resolve()
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"source path escapes data_root: {relative}") from exc
    return path


def _sequence_fingerprint(payload: dict, sequence_id: str) -> str:
    """Return sequence fingerprint."""
    matches = [
        value for value in payload.get("sequences", [])
        if value.get("sequence_id") == sequence_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one sequence {sequence_id!r}, found {len(matches)}"
        )
    fingerprint = (
        matches[0].get("source_report", {}).get("input_fingerprint")
    )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"source sequence has no input fingerprint: {sequence_id}")
    return fingerprint


def validate_report_source(
    report_source: dict,
    current_source: dict,
    sequence_id: str,
) -> None:
    """Validate report source."""
    current_fingerprint = current_source.get("sequence_fingerprint")
    if not isinstance(current_fingerprint, str) or not current_fingerprint:
        raise ValueError("current source metadata has no sequence fingerprint")

    report_fingerprint = report_source.get("sequence_fingerprint")
    if report_fingerprint is None:
        if (
            report_source.get("manifest_fingerprint")
            == current_source.get("manifest_fingerprint")
        ):
            return
        legacy_manifest = report_source.get("manifest")
        if not isinstance(legacy_manifest, str) or not legacy_manifest:
            raise ValueError("legacy report source has no manifest path")
        try:
            with Path(legacy_manifest).open("r", encoding="utf-8") as handle:
                legacy_payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot validate legacy report manifest: {legacy_manifest}"
            ) from exc
        if legacy_payload.get("version") != SOURCE_MANIFEST_VERSION:
            raise ValueError("legacy report source manifest version is unsupported")
        if (
            legacy_payload.get("input_fingerprint")
            != report_source.get("manifest_fingerprint")
        ):
            raise ValueError("legacy report manifest fingerprint mismatch")
        report_fingerprint = _sequence_fingerprint(
            legacy_payload, sequence_id
        )

    if report_fingerprint != current_fingerprint:
        raise ValueError(f"source sequence fingerprint mismatch: {sequence_id}")


def load_manifest_sequence(
    manifest_path: str | Path,
    sequence_id: str,
    data_root_override: Optional[str | Path] = None,
) -> Tuple[OmniSequence, str, dict]:
    """Load manifest sequence."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != SOURCE_MANIFEST_VERSION:
        raise ValueError(f"unsupported source manifest: {payload.get('version')}")
    data_root = Path(
        data_root_override if data_root_override is not None else payload["data_root"]
    ).resolve()

    matches = [
        value for value in payload.get("sequences", [])
        if value.get("sequence_id") == sequence_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one sequence {sequence_id!r}, found {len(matches)}"
        )
    value = matches[0]
    pano_from_face = {
        face: np.asarray(rotation, dtype=np.float64)
        for face, rotation in value["pano_from_face"].items()
    }
    for face, rotation in pano_from_face.items():
        if rotation.shape != (3, 3):
            raise ValueError(f"{sequence_id}: {face} pano rotation is not 3x3")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        validate_transform(transform)

    frames = []
    for frame in value.get("frames", []):
        frames.append(OmniFrame(
            frame_id=str(frame["frame_id"]),
            frame_index=int(frame["frame_index"]),
            world_from_pano=validate_transform(
                np.asarray(frame["world_from_pano"], dtype=np.float64)
            ),
            rgb_paths={
                face: _source_path(data_root, relative)
                for face, relative in frame["rgb_faces"].items()
            },
            depth_paths={
                face: _source_path(data_root, relative)
                for face, relative in frame["depth_faces"].items()
            },
        ))
    if not frames:
        raise ValueError(f"source manifest sequence has no frames: {sequence_id}")
    indices = [frame.frame_index for frame in frames]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise ValueError(f"source manifest frame indices are not strictly ordered: {sequence_id}")

    sequence = OmniSequence(
        dataset=str(payload["dataset"]),
        scene=str(value["scene"]),
        trajectory=str(value["trajectory"]),
        sequence_id=sequence_id,
        source_dir=data_root / value["scene"] / "Data_omni" / value["trajectory"],
        pano_from_face=pano_from_face,
        frames=tuple(frames),
        source_report=value.get("source_report") or {},
    )
    sequence_fingerprint = _sequence_fingerprint(payload, sequence_id)
    metadata = {
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": payload["input_fingerprint"],
        "sequence_fingerprint": sequence_fingerprint,
        "data_root": str(data_root),
    }
    return sequence, str(value["split"]), metadata


from dataclasses import asdict, dataclass
from pathlib import Path

from .config_sections import load_config_mapping


@dataclass(frozen=True)
class PairGeometryConfig:
    """Represent pair geometry config."""

    pano_height: int
    surface_relative_tolerance: float
    depth_cache_size: int


@dataclass(frozen=True)
class PairGateConfig:
    """Represent pair gate config."""

    min_symmetric_surface_overlap_ratio: float


@dataclass(frozen=True)
class PairSelectionConfig:
    """Represent pair selection config."""

    geometry: PairGeometryConfig
    gate: PairGateConfig

    def to_dict(self) -> dict:
        """Return a dictionary representation."""
        return asdict(self)


def _required(mapping: dict, key: str):
    """Return required."""
    if key not in mapping:
        raise ValueError(f"missing required pair config key: {key}")
    return mapping[key]


def load_pair_selection_config(path: str | Path) -> PairSelectionConfig:
    """Load pair selection config."""
    raw, _, _ = load_config_mapping(
        path, section="pair_quality", description="pair selection config"
    )
    geometry_raw = _required(raw, "geometry")
    expected_geometry = {
        "pano_height", "surface_relative_tolerance", "depth_cache_size"
    }
    unexpected_geometry = set(geometry_raw) - expected_geometry
    if unexpected_geometry:
        raise ValueError(
            "pair_quality.geometry has unused fields: "
            f"{sorted(unexpected_geometry)}"
        )
    geometry = PairGeometryConfig(
        pano_height=int(_required(geometry_raw, "pano_height")),
        surface_relative_tolerance=float(
            _required(geometry_raw, "surface_relative_tolerance")
        ),
        depth_cache_size=int(geometry_raw.get("depth_cache_size", 64)),
    )
    if geometry.pano_height < 8:
        raise ValueError("geometry.pano_height must be at least 8")
    if not 0.0 < geometry.surface_relative_tolerance <= 1.0:
        raise ValueError("surface_relative_tolerance must be within (0, 1]")
    if geometry.depth_cache_size < 1:
        raise ValueError("depth_cache_size must be positive")

    gate_raw = _required(raw, "gate")
    expected_gate = {"min_symmetric_surface_overlap_ratio"}
    unexpected_gate = set(gate_raw) - expected_gate
    if unexpected_gate:
        raise ValueError(
            "pair_quality.gate has unused fields: "
            f"{sorted(unexpected_gate)}"
        )
    gate = PairGateConfig(
        min_symmetric_surface_overlap_ratio=float(
            _required(gate_raw, "min_symmetric_surface_overlap_ratio")
        )
    )
    if not 0.0 <= gate.min_symmetric_surface_overlap_ratio <= 1.0:
        raise ValueError(
            "min_symmetric_surface_overlap_ratio must be within [0, 1]"
        )
    return PairSelectionConfig(geometry=geometry, gate=gate)


from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import invert_transform, validate_transform
from .omni import OmniFrame, OmniSequence
from .sensors import (
    PanoDepthProjection,
    build_pano_depth_projection,
    load_frame_radial_depth,
    pano_directions,
)


PAIR_SCORE_VERSION = "raw_omni_surface_overlap_score_v3_xyz_flu"


@dataclass(frozen=True)
class FramePairCandidate:
    """Represent frame pair candidate."""

    source: OmniFrame
    target: OmniFrame
    gap: int
    pair_descriptor: Optional[dict] = None

    @property
    def pair_id(self) -> str:
        """Return pair ID."""
        return f"{self.source.frame_id}:{self.target.frame_id}"


@dataclass(frozen=True)
class DirectedOverlapMetrics:
    """Represent directed overlap metrics."""

    source_valid_solid_angle_sr: float
    target_owner_solid_angle_sr: float
    same_surface_solid_angle_sr: float
    surface_overlap_ratio: float
    owner_surface_match_ratio: float


def _pano_row_solid_angle(height: int, width: int) -> np.ndarray:
    """Return pano row solid angle."""
    if height < 2 or width != 2 * height:
        raise ValueError("pano dimensions must satisfy width=2*height")
    edges = np.pi * (0.5 - np.arange(height + 1, dtype=np.float64) / height)
    row_area = np.sin(edges[:-1]) - np.sin(edges[1:])
    return np.repeat((row_area * (2.0 * np.pi / width))[:, None], width, axis=1)


def _project_to_pano(points: np.ndarray, height: int, width: int):
    """Project to pano."""
    ranges = np.linalg.norm(points, axis=1)
    valid = np.isfinite(ranges) & (ranges > 1e-8)
    safe_ranges = np.where(valid, ranges, 1.0)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    latitude = np.arcsin(np.clip(points[:, 2] / safe_ranges, -1.0, 1.0))
    pixel_x = np.floor(width * (0.5 - azimuth / (2.0 * np.pi)) + 0.5).astype(np.int64) % width
    pixel_y = np.clip(
        np.floor(height * (0.5 - latitude / np.pi) + 0.5).astype(np.int64),
        0, height - 1,
    )
    return ranges, valid, pixel_x, pixel_y


def _zbuffer_owners(
    ranges: np.ndarray,
    valid: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    width: int,
) -> np.ndarray:
    """Return zbuffer owners."""
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.zeros(0, dtype=np.int64)
    pixels = pixel_y[indices] * width + pixel_x[indices]
    selected_ranges = ranges[indices]
    pixel_count = int(np.max(pixels)) + 1
    minimum_range = np.full(pixel_count, np.inf, dtype=selected_ranges.dtype)
    np.minimum.at(minimum_range, pixels, selected_ranges)
    nearest = selected_ranges == minimum_range[pixels]
    nearest_indices = indices[nearest]
    nearest_pixels = pixels[nearest]
    sentinel = np.iinfo(np.int64).max
    owners = np.full(pixel_count, sentinel, dtype=np.int64)
    np.minimum.at(owners, nearest_pixels, nearest_indices)
    return owners[owners != sentinel]


def _directed_overlap(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    world_from_source: np.ndarray,
    world_from_target: np.ndarray,
    directions: np.ndarray,
    config: PairGeometryConfig,
    solid_angle: np.ndarray,
) -> DirectedOverlapMetrics:
    """Return directed overlap."""
    height, width = source_depth.shape
    source_valid = np.isfinite(source_depth) & (source_depth > 0.0)
    source_indices = np.flatnonzero(source_valid)
    source_directions = directions[source_valid]
    source_ranges = source_depth[source_valid].astype(np.float64, copy=False)
    source_points = source_directions * source_ranges[:, None]
    target_from_source = invert_transform(world_from_target) @ validate_transform(world_from_source)
    points_target = source_points @ target_from_source[:3, :3].T + target_from_source[:3, 3]
    ranges, valid_ranges, pixel_x, pixel_y = _project_to_pano(points_target, height, width)
    owners = _zbuffer_owners(ranges, valid_ranges, pixel_x, pixel_y, width)

    source_weights = solid_angle.reshape(-1)[source_indices]
    source_area = float(np.sum(source_weights))
    if not len(owners):
        return DirectedOverlapMetrics(source_area, 0.0, 0.0, 0.0, 0.0)
    sampled = target_depth[pixel_y[owners], pixel_x[owners]].astype(np.float64, copy=False)
    comparable = np.isfinite(sampled) & (sampled > 0.0)
    denominator = np.maximum(ranges[owners], sampled)
    same_surface = comparable & (
        np.abs(ranges[owners] - sampled)
        / np.maximum(denominator, 1e-12)
        <= config.surface_relative_tolerance
    )
    owner_area = float(np.sum(solid_angle[pixel_y[owners], pixel_x[owners]]))
    same_area = float(np.sum(source_weights[owners[same_surface]]))
    return DirectedOverlapMetrics(
        source_valid_solid_angle_sr=source_area,
        target_owner_solid_angle_sr=owner_area,
        same_surface_solid_angle_sr=same_area,
        surface_overlap_ratio=(same_area / source_area if source_area else 0.0),
        owner_surface_match_ratio=(
            float(
                np.sum(
                    solid_angle[
                        pixel_y[owners[same_surface]],
                        pixel_x[owners[same_surface]],
                    ]
                )
            ) / owner_area
            if owner_area else 0.0
        ),
    )


def _gate_reasons(overlap_ratio: float, gate: PairGateConfig) -> list[str]:
    """Return gate reasons."""
    if overlap_ratio < gate.min_symmetric_surface_overlap_ratio:
        return ["insufficient_symmetric_surface_overlap"]
    return []


def score_radial_depth_pair(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    world_from_source: np.ndarray,
    world_from_target: np.ndarray,
    geometry: PairGeometryConfig,
    gate: PairGateConfig,
    directions: Optional[np.ndarray] = None,
) -> dict:
    """Score radial depth pair."""
    source_depth = np.asarray(source_depth, dtype=np.float64)
    target_depth = np.asarray(target_depth, dtype=np.float64)
    if source_depth.ndim != 2 or source_depth.shape != target_depth.shape:
        raise ValueError("source and target radial depth must have the same 2D shape")
    height, width = source_depth.shape
    if width != 2 * height:
        raise ValueError("radial depth must be an equirectangular width=2*height map")
    if directions is None:
        directions = pano_directions(height, width)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape != (height, width, 3):
        raise ValueError("directions shape does not match radial depth")
    world_from_source = validate_transform(world_from_source)
    world_from_target = validate_transform(world_from_target)
    solid_angle = _pano_row_solid_angle(height, width)
    source_to_target = _directed_overlap(
        source_depth, target_depth, world_from_source, world_from_target,
        directions, geometry, solid_angle,
    )
    target_to_source = _directed_overlap(
        target_depth, source_depth, world_from_target, world_from_source,
        directions, geometry, solid_angle,
    )
    symmetric = float(min(
        source_to_target.surface_overlap_ratio,
        target_to_source.surface_overlap_ratio,
    ))
    target_from_source = invert_transform(world_from_target) @ world_from_source
    baseline = float(np.linalg.norm(target_from_source[:3, 3]))
    positive_depth = np.concatenate((
        source_depth[np.isfinite(source_depth) & (source_depth > 0.0)],
        target_depth[np.isfinite(target_depth) & (target_depth > 0.0)],
    ))
    normalized_depth = positive_depth / baseline if baseline > 1e-12 else np.zeros(0)
    normalized_summary = {
        "count": int(len(normalized_depth)),
        "p10": float(np.percentile(normalized_depth, 10.0)) if len(normalized_depth) else None,
        "p50": float(np.percentile(normalized_depth, 50.0)) if len(normalized_depth) else None,
        "p90": float(np.percentile(normalized_depth, 90.0)) if len(normalized_depth) else None,
        "semantic": "raw_radial_depth_divided_by_raw_pair_baseline",
    }
    reasons = _gate_reasons(symmetric, gate)
    return {
        "version": PAIR_SCORE_VERSION,
        "raw_baseline_m_audit_only": baseline,
        "normalized_depth": normalized_summary,
        "raw_rotation_deg_audit_only": float(np.rad2deg(Rotation.from_matrix(target_from_source[:3, :3]).magnitude())),
        "source_to_target": asdict(source_to_target),
        "target_to_source": asdict(target_to_source),
        "symmetric_surface_overlap_ratio": symmetric,
        "overlap_semantic": "zbuffer_same_surface_solid_angle_ratio",
        "surface_relative_tolerance": geometry.surface_relative_tolerance,
        "passed": not reasons,
        "reasons": reasons,
    }


class RawPairScorer:
    """Represent raw pair scorer."""

    def __init__(self, sequence: OmniSequence, config: PairSelectionConfig):
        """Initialize the instance."""
        self.sequence = sequence
        self.config = config
        face_size = sequence.source_report.get("depth_face_size")
        if (
            not isinstance(face_size, list)
            or len(face_size) != 2
            or face_size[0] != face_size[1]
        ):
            raise ValueError("source report must declare a square depth_face_size")
        self.projection: PanoDepthProjection = build_pano_depth_projection(
            sequence.pano_from_face,
            int(face_size[0]),
            config.geometry.pano_height,
        )
        self._depth_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _depth(self, frame: OmniFrame) -> np.ndarray:
        """Return depth."""
        cached = self._depth_cache.get(frame.frame_id)
        if cached is not None:
            self._depth_cache.move_to_end(frame.frame_id)
            return cached
        depth = load_frame_radial_depth(frame, self.projection)
        self._depth_cache[frame.frame_id] = depth
        while len(self._depth_cache) > self.config.geometry.depth_cache_size:
            self._depth_cache.popitem(last=False)
        return depth

    def score(self, candidate: FramePairCandidate) -> dict:
        """Score."""
        score = score_radial_depth_pair(
            self._depth(candidate.source),
            self._depth(candidate.target),
            candidate.source.world_from_pano,
            candidate.target.world_from_pano,
            self.config.geometry,
            self.config.gate,
            directions=self.projection.directions,
        )
        score.update({
            "pair_id": candidate.pair_id,
            "source_frame": candidate.source.frame_id,
            "target_frame": candidate.target.frame_id,
            "gap": candidate.gap,
        })
        return score


import gzip
import hashlib
import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from ._runtime_log import emit_runtime_log
from .omni import OmniSequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation_config import RandomPairSamplingConfig


APPROVED_PAIR_POOL_VERSION = "surface_overlap_approved_ordered_pair_pool_v4_pose_radius"


def build_frame_pair_manifest(
    data_root: Union[str, Path],
    config_path: Union[str, Path],
    output_path: Union[str, Path],
    scenes: Optional[Iterable[str]] = None,
) -> dict:
    """Build frame pair manifest."""
    return build_source_manifest(
        data_root=data_root,
        config_path=config_path,
        output=output_path,
        scenes=scenes,
    )


def enumerate_ordered_pose_radius_candidates(
    sequence: OmniSequence,
    max_pose_translation_distance_m: float,
) -> Iterator[FramePairCandidate]:
    """Return enumerate ordered pose radius candidates."""
    if (
        not np.isfinite(max_pose_translation_distance_m)
        or max_pose_translation_distance_m <= 0.0
    ):
        raise ValueError("max pose translation distance must be finite and positive")
    frames = sequence.frames
    if len(frames) < 2:
        return
    positions = np.asarray(
        [frame.world_from_pano[:3, 3] for frame in frames],
        dtype=np.float64,
    )
    if positions.shape != (len(frames), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("frame pose translations must be finite xyz vectors")
    neighbor_indices = cKDTree(positions).query_ball_point(
        positions,
        r=float(max_pose_translation_distance_m),
    )
    for source_position, source in enumerate(frames):
        for target_position in sorted(neighbor_indices[source_position]):
            if target_position == source_position:
                continue
            distance_m = float(np.linalg.norm(
                positions[target_position] - positions[source_position]
            ))

            if not 1e-12 < distance_m < max_pose_translation_distance_m:
                continue
            target = frames[target_position]
            yield FramePairCandidate(
                source=source,
                target=target,
                gap=target.frame_index - source.frame_index,
            )


class FeasibilityConditionedPairSampler:
    """Represent feasibility conditioned pair sampler."""

    def __init__(
        self,
        sequence: OmniSequence,
        candidates: Tuple[FramePairCandidate, ...],
        feasibility_weights: np.ndarray,
        raw_candidate_count: int,
        seed: int,
        distribution: str = (
            "overlap_approved_pairs_weighted_by_feasible_so3_measure"
        ),
    ):
        """Initialize the instance."""
        weights = np.asarray(feasibility_weights, dtype=np.float64)
        if weights.shape != (len(candidates),):
            raise ValueError("pair feasibility weights have invalid shape")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("pair feasibility weights must be finite and nonnegative")
        positive = weights > 0.0
        if not np.any(positive):
            raise ValueError(
                f"{sequence.sequence_id}: no approved pair has feasible views"
            )
        self.sequence = sequence
        self.seed = int(seed)
        self._distribution = str(distribution)
        self._rng = np.random.default_rng(self.seed)
        self._candidates = tuple(
            candidate
            for candidate, keep in zip(candidates, positive)
            if bool(keep)
        )
        kept_weights = weights[positive]
        self._probabilities = kept_weights / np.sum(kept_weights)
        self._baseline_norms = np.asarray([
            np.linalg.norm(
                candidate.source.world_from_pano[:3, 3]
                - candidate.target.world_from_pano[:3, 3]
            )
            for candidate in self._candidates
        ], dtype=np.float64)
        self._raw_candidate_count = int(raw_candidate_count)
        self._approved_candidate_count = len(candidates)
        self._globally_infeasible_count = int(np.count_nonzero(~positive))
        self._draw_counts = Counter()
        self._weight_minimum = float(np.min(kept_weights))
        self._weight_maximum = float(np.max(kept_weights))
        self._weight_mean = float(np.mean(kept_weights))
        self._scale_conditioned_draw_count = 0
        self._scale_conditioned_empty_support_count = 0
        self._scale_conditioned_feasible_candidate_total = 0

    @property
    def candidate_count(self) -> int:
        """Return candidate count."""
        return len(self._candidates)

    @property
    def candidates(self) -> Tuple[FramePairCandidate, ...]:
        """Return candidates."""
        return self._candidates

    def draw(self) -> FramePairCandidate:
        """Draw."""
        index = int(self._rng.choice(
            len(self._candidates), p=self._probabilities
        ))
        candidate = self._candidates[index]
        self._draw_counts[candidate.pair_id] += 1
        return candidate

    def draw_scale_feasible(
        self,
        desired_true_translation_norm: float,
        scale_min: float,
        scale_max: float,
        excluded_pair_ids=(),
    ) -> Optional[FramePairCandidate]:
        """Draw scale feasible."""
        desired_norm = float(desired_true_translation_norm)
        scale_min = float(scale_min)
        scale_max = float(scale_max)
        if not np.isfinite(desired_norm) or desired_norm <= 0.0:
            raise ValueError("desired translation norm must be positive")
        if not 0.0 < scale_min < scale_max:
            raise ValueError("scale range must satisfy 0 < min < max")
        valid_baseline = self._baseline_norms > 1e-12
        required_scale = np.full_like(self._baseline_norms, np.inf)
        required_scale[valid_baseline] = (
            desired_norm / self._baseline_norms[valid_baseline]
        )
        feasible = (
            valid_baseline
            & (required_scale >= scale_min - 1e-12)
            & (required_scale <= scale_max + 1e-12)
        )
        excluded = {str(value) for value in excluded_pair_ids}
        if excluded:
            feasible &= np.asarray([
                candidate.pair_id not in excluded
                for candidate in self._candidates
            ], dtype=bool)
        feasible_count = int(np.count_nonzero(feasible))
        self._scale_conditioned_feasible_candidate_total += feasible_count
        if not feasible_count:
            self._scale_conditioned_empty_support_count += 1
            return None
        probabilities = np.where(feasible, self._probabilities, 0.0)
        probabilities /= np.sum(probabilities)
        index = int(self._rng.choice(
            len(self._candidates), p=probabilities
        ))
        candidate = self._candidates[index]
        self._draw_counts[candidate.pair_id] += 1
        self._scale_conditioned_draw_count += 1
        return candidate

    def report(self) -> dict:
        """Return report."""
        source_frames = {value.source.frame_id for value in self._candidates}
        target_frames = {value.target.frame_id for value in self._candidates}
        return {
            "sequence_id": self.sequence.sequence_id,
            "seed": self.seed,
            "distribution": self._distribution,
            "raw_candidate_count": self._raw_candidate_count,
            "approved_candidate_count": self._approved_candidate_count,
            "candidate_count": len(self._candidates),
            "globally_infeasible_candidate_count": (
                self._globally_infeasible_count
            ),
            "feasibility_weight": {
                "minimum": self._weight_minimum,
                "maximum": self._weight_maximum,
                "mean": self._weight_mean,
            },
            "eligible_source_frame_count": len(source_frames),
            "eligible_target_frame_count": len(target_frames),
            "draw_count": int(sum(self._draw_counts.values())),
            "scale_conditioned_draw_count": (
                self._scale_conditioned_draw_count
            ),
            "scale_conditioned_empty_support_count": (
                self._scale_conditioned_empty_support_count
            ),
            "mean_scale_feasible_candidate_count": (
                self._scale_conditioned_feasible_candidate_total
                / self._scale_conditioned_draw_count
                if self._scale_conditioned_draw_count else 0.0
            ),
            "unique_drawn_pairs": len(self._draw_counts),
            "draw_counts": dict(sorted(self._draw_counts.items())),
        }


@dataclass(frozen=True)
class ApprovedPairPool:
    """Represent approved pair pool."""

    candidates: Tuple[FramePairCandidate, ...]
    raw_candidate_count: int
    report: dict


def pair_pool_identity(
    sequence: OmniSequence,
    sampling: RandomPairSamplingConfig,
    sequence_fingerprint: str,
    pair_config_fingerprint: str,
) -> dict:
    """Return pair pool identity."""
    return {
        "version": APPROVED_PAIR_POOL_VERSION,
        "pair_score_version": PAIR_SCORE_VERSION,
        "sequence_id": sequence.sequence_id,
        "sequence_fingerprint": str(sequence_fingerprint),
        "pair_config_fingerprint": str(pair_config_fingerprint),
        "max_pose_translation_distance_m": float(
            sampling.max_pose_translation_distance_m
        ),
        "ordered_frame_roles": bool(sampling.ordered_frame_roles),
    }


def pair_pool_cache_key(identity: dict) -> str:
    """Return pair pool cache key."""
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def pair_pool_cache_path(
    cache_root: Union[str, Path],
    identity: dict,
) -> Path:
    """Return pair pool cache path."""
    sequence_id = str(identity["sequence_id"])
    safe_sequence_id = "".join(
        value if value.isalnum() or value in ("-", "_") else "_"
        for value in sequence_id
    )
    return (
        Path(cache_root)
        / "qualified_pairs"
        / f"{safe_sequence_id}-{pair_pool_cache_key(identity)}.json"
    )


def _write_pair_pool_json_atomic(path: Path, payload: dict) -> None:
    """Write pair pool JSON atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_pair_pool_json(path: Path) -> dict:
    """Read pair pool JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"pair pool cache is not a JSON mapping: {path}")
    return payload


def _score_summary(scores: List[dict]) -> dict:
    """Score summary."""
    reasons = Counter()
    by_gap = defaultdict(lambda: {"candidate_count": 0, "passed_count": 0})
    for score in scores:
        gap = str(int(score["gap"]))
        by_gap[gap]["candidate_count"] += 1
        by_gap[gap]["passed_count"] += int(bool(score["passed"]))
        if not score["passed"]:
            reasons.update(str(value) for value in score.get("reasons", ()))
    passed = sum(bool(score["passed"]) for score in scores)
    baselines = np.asarray(
        [float(score["raw_baseline_m_audit_only"]) for score in scores],
        dtype=np.float64,
    )
    baseline_summary = {
        "count": int(len(baselines)),
        "minimum": float(np.min(baselines)) if len(baselines) else None,
        "p05": float(np.percentile(baselines, 5.0)) if len(baselines) else None,
        "p50": float(np.percentile(baselines, 50.0)) if len(baselines) else None,
        "p95": float(np.percentile(baselines, 95.0)) if len(baselines) else None,
        "maximum": float(np.max(baselines)) if len(baselines) else None,
        "semantic": "euclidean_distance_between_raw_world_from_pano_origins_m",
    }
    return {
        "candidate_count": len(scores),
        "passed_count": passed,
        "rejected_count": len(scores) - passed,
        "acceptance_rate": float(passed / len(scores)) if scores else 0.0,
        "rejection_reasons": dict(sorted(reasons.items())),
        "by_gap": dict(sorted(by_gap.items(), key=lambda value: int(value[0]))),
        "raw_pose_translation_distance_m": baseline_summary,
        "temporal_gap_audit": {
            "absolute_gap_greater_than_30_candidate_count": sum(
                abs(int(score["gap"])) > 30 for score in scores
            ),
            "absolute_gap_greater_than_30_passed_count": sum(
                abs(int(score["gap"])) > 30 and bool(score["passed"])
                for score in scores
            ),
            "gap_is_audit_only": True,
        },
    }


def _validate_cached_pool(
    payload: dict,
    identity: dict,
    candidates_by_id: Dict[str, FramePairCandidate],
) -> Tuple[FramePairCandidate, ...]:
    """Validate cached pool."""
    if payload.get("identity") != identity:
        raise ValueError("pair pool cache identity mismatch")
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(candidates_by_id):
        raise ValueError("pair pool cache score count mismatch")
    score_ids = [str(value.get("pair_id", "")) for value in scores]
    if len(set(score_ids)) != len(score_ids) or set(score_ids) != set(candidates_by_id):
        raise ValueError("pair pool cache candidate identities mismatch")
    approved = []
    for score in scores:
        candidate = candidates_by_id[str(score["pair_id"])]
        if (
            str(score.get("source_frame")) != candidate.source.frame_id
            or str(score.get("target_frame")) != candidate.target.frame_id
            or int(score.get("gap")) != candidate.gap
        ):
            raise ValueError("pair pool cache frame identity mismatch")
        if bool(score.get("passed")):
            descriptor = None
            if all(value in score for value in (
                "raw_baseline_m_audit_only",
                "normalized_depth",
                "symmetric_surface_overlap_ratio",
            )):
                descriptor = {
                    "raw_baseline_m": float(score["raw_baseline_m_audit_only"]),
                    "normalized_depth": dict(score["normalized_depth"]),
                    "symmetric_surface_overlap_ratio": float(
                        score["symmetric_surface_overlap_ratio"]
                    ),
                }
            approved.append(FramePairCandidate(
                candidate.source, candidate.target, candidate.gap, descriptor
            ))
    if not approved:
        raise ValueError("pair pool cache contains no approved pairs")
    expected_summary = _score_summary(scores)
    if payload.get("summary") != expected_summary:
        raise ValueError("pair pool cache summary mismatch")
    return tuple(approved)


def build_or_load_approved_pair_pool(
    sequence: OmniSequence,
    sampling: RandomPairSamplingConfig,
    pair_config: PairSelectionConfig,
    sequence_fingerprint: str,
    pair_config_fingerprint: str,
    cache_root: Union[str, Path],
    progress_interval_seconds: float = 10.0,
) -> ApprovedPairPool:
    """Build or load approved pair pool."""
    identity = pair_pool_identity(
        sequence,
        sampling,
        sequence_fingerprint,
        pair_config_fingerprint,
    )
    if progress_interval_seconds <= 0.0:
        raise ValueError("progress interval seconds must be positive")
    cache_path = pair_pool_cache_path(cache_root, identity)
    candidates = tuple(enumerate_ordered_pose_radius_candidates(
        sequence, sampling.max_pose_translation_distance_m
    ))
    candidates_by_id = {value.pair_id: value for value in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("ordered pair enumeration produced duplicate IDs")

    start = perf_counter()
    cache_status = "built"
    payload: Optional[dict] = None
    if cache_path.is_file():
        try:
            payload = _read_pair_pool_json(cache_path)
            approved = _validate_cached_pool(
                payload, identity, candidates_by_id
            )
            cache_status = "loaded"
        except (OSError, ValueError, KeyError, TypeError):
            payload = None

    if payload is None:
        scorer = RawPairScorer(sequence, pair_config)
        scores = []
        passed_count = 0
        last_progress_log = perf_counter()
        for candidate_index, candidate in enumerate(candidates, start=1):
            score = scorer.score(candidate)
            scores.append(score)
            passed_count += int(bool(score["passed"]))
            now = perf_counter()
            if (
                now - last_progress_log >= progress_interval_seconds
                or candidate_index == len(candidates)
            ):
                emit_runtime_log(
                    "stage_progress",
                    "Scoring qualified frame-pair candidates",
                    scene=sequence.scene,
                    sequence_id=sequence.sequence_id,
                    processed_candidates=candidate_index,
                    total_candidates=len(candidates),
                    approved_candidates=passed_count,
                    acceptance_rate=round(
                        passed_count / candidate_index, 6
                    ),
                    remaining_candidates=len(candidates) - candidate_index,
                    elapsed_seconds=round(now - start, 1),
                )
                last_progress_log = now
        payload = {
            "identity": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": _score_summary(scores),
            "scores": scores,
        }
        approved = _validate_cached_pool(payload, identity, candidates_by_id)
        _write_pair_pool_json_atomic(cache_path, payload)

    elapsed = perf_counter() - start
    summary = dict(payload["summary"])
    report = {
        "version": APPROVED_PAIR_POOL_VERSION,
        "pair_score_version": identity["pair_score_version"],
        "pair_descriptor_semantics": (
            "spherical_solid_angle_surface_overlap_and_raw_baseline_for_"
            "pre_render_domain_conditioning"
        ),
        "candidate_enumeration": (
            "all_ordered_distinct_frame_roles_within_raw_pose_translation_radius"
        ),
        "cache_status": cache_status,
        "cache_file": cache_path.name,
        "cache_key": pair_pool_cache_key(identity),
        "load_or_build_seconds": float(elapsed),
        **summary,
    }
    return ApprovedPairPool(
        candidates=approved,
        raw_candidate_count=len(candidates),
        report=report,
    )


__all__ = [
    "ApprovedPairPool",
    "FeasibilityConditionedPairSampler",
    "build_frame_pair_manifest",
    "build_or_load_approved_pair_pool",
]
