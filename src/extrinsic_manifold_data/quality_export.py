"""Score, audit, and transactionally export generated samples."""

from __future__ import annotations


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .config_sections import load_config_mapping

@dataclass(frozen=True)
class CoverageConfig:
    """Represent coverage config."""

    mode: str
    bins_per_axis: int | None = None
    translation_shell_edges: Tuple[float, ...] = ()
    translation_center_radius: float | None = None
    translation_direction_cells: int | None = None
    min_translation_cell_occupancy: float | None = None
    min_candidates_per_translation_cell: int | None = None
    min_pairs_per_translation_cell: int | None = None
    min_rotation_cells_per_translation_cell: int | None = None
    min_rotation_axes_per_translation_cell: int | None = None
    min_rotation_shells_per_translation_cell: int | None = None
    min_translation_cells_per_rotation_cell: int | None = None

    def to_dict(self) -> dict:
        """Return a dictionary representation."""
        values = asdict(self)
        return {
            key: (list(value) if key == "translation_shell_edges" else value)
            for key, value in values.items()
            if value not in (None, ())
        }


def load_coverage_config(path: str | Path) -> CoverageConfig:
    """Load coverage config."""
    raw, _, _ = load_config_mapping(
        path, section="coverage", description="coverage config"
    )
    mode = str(raw.get("mode", "")).strip()
    if mode == FIXED_IDENTITY_COVERAGE:
        expected = {"mode"}
        config = CoverageConfig(mode=mode)
    elif mode == DIRECT_EULER_BOX_COVERAGE:
        expected = {"mode", "bins_per_axis"}
        bins = int(raw.get("bins_per_axis", 0))
        if bins < 2:
            raise ValueError("coverage.bins_per_axis must be at least two")
        config = CoverageConfig(mode=mode, bins_per_axis=bins)
    elif mode == GEODESIC_PRODUCT_BALL_COVERAGE:
        expected = {
            "mode", "translation_shell_edges", "translation_center_radius",
            "translation_direction_cells", "min_translation_cell_occupancy",
            "min_candidates_per_translation_cell",
            "min_pairs_per_translation_cell",
            "min_rotation_cells_per_translation_cell",
            "min_rotation_axes_per_translation_cell",
            "min_rotation_shells_per_translation_cell",
            "min_translation_cells_per_rotation_cell",
        }
        missing = expected - set(raw)
        if missing:
            raise ValueError(f"geodesic coverage is missing fields: {sorted(missing)}")
        edges = tuple(float(value) for value in raw["translation_shell_edges"])
        config = CoverageConfig(
            mode=mode,
            translation_shell_edges=edges,
            translation_center_radius=float(raw["translation_center_radius"]),
            translation_direction_cells=int(raw["translation_direction_cells"]),
            min_translation_cell_occupancy=float(
                raw["min_translation_cell_occupancy"]
            ),
            min_candidates_per_translation_cell=int(
                raw["min_candidates_per_translation_cell"]
            ),
            min_pairs_per_translation_cell=int(raw["min_pairs_per_translation_cell"]),
            min_rotation_cells_per_translation_cell=int(
                raw["min_rotation_cells_per_translation_cell"]
            ),
            min_rotation_axes_per_translation_cell=int(
                raw["min_rotation_axes_per_translation_cell"]
            ),
            min_rotation_shells_per_translation_cell=int(
                raw["min_rotation_shells_per_translation_cell"]
            ),
            min_translation_cells_per_rotation_cell=int(
                raw["min_translation_cells_per_rotation_cell"]
            ),
        )
        if (
            len(edges) < 2 or edges[0] != 0.0 or edges[-1] < 1.0
            or tuple(sorted(set(edges))) != edges
        ):
            raise ValueError(
                "translation_shell_edges must increase uniquely from 0 through >=1"
            )
        if config.translation_direction_cells < 4:
            raise ValueError("translation_direction_cells must be at least four")
        if not np.isclose(
            config.translation_center_radius, edges[1], rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "translation_center_radius must equal the first nonzero shell edge"
            )
        if not 0.0 < config.min_translation_cell_occupancy <= 1.0:
            raise ValueError(
                "min_translation_cell_occupancy must be within (0, 1]"
            )
        integer_thresholds = (
            config.min_candidates_per_translation_cell,
            config.min_pairs_per_translation_cell,
            config.min_rotation_cells_per_translation_cell,
            config.min_rotation_axes_per_translation_cell,
            config.min_rotation_shells_per_translation_cell,
            config.min_translation_cells_per_rotation_cell,
        )
        if min(integer_thresholds) < 1:
            raise ValueError("all geodesic coverage count thresholds must be positive")
    else:
        raise ValueError(f"unsupported coverage.mode: {mode!r}")
    unexpected = set(raw) - expected
    if unexpected:
        raise ValueError(
            f"coverage mode {mode!r} has unused fields: {sorted(unexpected)}"
        )
    return config


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.transform import Rotation

from .config_sections import load_config_mapping
from .omni import OmniFrame, OmniSequence
from .geometry import project_points, validate_transform
from .sensors import LidarModalityDerivationConfig, TrainingSensorConfig


GENERATED_QUALITY_VERSION = (
    "raw_sensor_unscaled_depth_sobel_projection_quality_v10"
)
LEGACY_GENERATED_QUALITY_VERSIONS = (
    "raw_sensor_structural_quality_v3",
    "raw_sensor_structural_quality_v4",
)


@dataclass(frozen=True)
class GeneratedQualityConfig:
    """Represent generated quality config."""

    visibility_abs_tolerance_m: float
    visibility_rel_tolerance: float
    alpha_radius_px: float
    min_true_projection_coverage_ratio: float
    max_true_occluded_area_ratio: float
    min_nominal_projection_coverage_ratio: float
    depth_sobel_gradient_threshold_m_per_px: float
    min_true_sobel_edge_pixel_ratio: float
    min_true_sobel_direction_diversity: float
    sobel_direction_histogram_bin_deg: float
    sobel_direction_mode_support_half_width_deg: float
    min_true_sobel_direction_mode_separation_deg: float
    min_true_sobel_second_mode_ratio: float
    min_true_sobel_third_mode_ratio: float

    def to_dict(self) -> dict:
        """Return a dictionary representation."""
        return asdict(self)


def load_generated_quality_config(
    path: Union[str, Path],
) -> GeneratedQualityConfig:
    """Load generated quality config."""
    raw, _, _ = load_config_mapping(
        path, section="quality", description="generated quality config"
    )
    deprecated_mean_field = "min_true_masked_depth_sobel_mean_m_per_px"
    if deprecated_mean_field in raw:
        raise ValueError(
            f"quality.{deprecated_mean_field} was replaced by "
            "quality.min_true_sobel_edge_pixel_ratio"
        )
    config = GeneratedQualityConfig(
        visibility_abs_tolerance_m=float(raw["visibility_abs_tolerance_m"]),
        visibility_rel_tolerance=float(raw["visibility_rel_tolerance"]),
        alpha_radius_px=float(raw["alpha_radius_px"]),
        min_true_projection_coverage_ratio=float(
            raw["min_true_projection_coverage_ratio"]
        ),
        max_true_occluded_area_ratio=float(
            raw["max_true_occluded_area_ratio"]
        ),
        min_nominal_projection_coverage_ratio=float(
            raw["min_nominal_projection_coverage_ratio"]
        ),
        depth_sobel_gradient_threshold_m_per_px=float(
            raw["depth_sobel_gradient_threshold_m_per_px"]
        ),
        min_true_sobel_edge_pixel_ratio=float(
            raw["min_true_sobel_edge_pixel_ratio"]
        ),
        min_true_sobel_direction_diversity=float(
            raw["min_true_sobel_direction_diversity"]
        ),
        sobel_direction_histogram_bin_deg=float(
            raw["sobel_direction_histogram_bin_deg"]
        ),
        sobel_direction_mode_support_half_width_deg=float(
            raw["sobel_direction_mode_support_half_width_deg"]
        ),
        min_true_sobel_direction_mode_separation_deg=float(
            raw["min_true_sobel_direction_mode_separation_deg"]
        ),
        min_true_sobel_second_mode_ratio=float(
            raw["min_true_sobel_second_mode_ratio"]
        ),
        min_true_sobel_third_mode_ratio=float(
            raw["min_true_sobel_third_mode_ratio"]
        ),
    )
    if (
        config.visibility_abs_tolerance_m < 0.0
        or config.visibility_rel_tolerance < 0.0
        or not np.isfinite(config.depth_sobel_gradient_threshold_m_per_px)
        or config.depth_sobel_gradient_threshold_m_per_px < 0.0
        or not np.isfinite(config.min_true_sobel_edge_pixel_ratio)
        or not np.isfinite(config.min_true_sobel_direction_diversity)
        or not np.isfinite(config.sobel_direction_histogram_bin_deg)
        or config.sobel_direction_histogram_bin_deg <= 0.0
        or not np.isfinite(
            config.sobel_direction_mode_support_half_width_deg
        )
        or config.sobel_direction_mode_support_half_width_deg <= 0.0
        or not np.isfinite(
            config.min_true_sobel_direction_mode_separation_deg
        )
    ):
        raise ValueError(
            "visibility and Sobel thresholds must be finite and nonnegative"
        )
    ratios = (
        config.min_true_projection_coverage_ratio,
        config.max_true_occluded_area_ratio,
        config.min_nominal_projection_coverage_ratio,
        config.min_true_sobel_edge_pixel_ratio,
        config.min_true_sobel_direction_diversity,
        config.min_true_sobel_second_mode_ratio,
        config.min_true_sobel_third_mode_ratio,
    )
    if any(not 0.0 <= value <= 1.0 for value in ratios):
        raise ValueError("quality ratios must lie within [0, 1]")
    if not np.isfinite(config.alpha_radius_px) or config.alpha_radius_px <= 0.0:
        raise ValueError("alpha_radius_px must be finite and positive")
    bin_count = 180.0 / config.sobel_direction_histogram_bin_deg
    if not np.isclose(bin_count, round(bin_count), rtol=0.0, atol=1e-12):
        raise ValueError("180 must be divisible by Sobel direction histogram bin width")
    if (
        config.min_true_sobel_direction_mode_separation_deg
        < 2.0 * config.sobel_direction_mode_support_half_width_deg
        or config.min_true_sobel_direction_mode_separation_deg > 60.0
    ):
        raise ValueError(
            "Sobel direction mode separation must be at least twice the "
            "support half width and at most 60 degrees"
        )
    if (
        config.min_true_sobel_third_mode_ratio
        > config.min_true_sobel_second_mode_ratio
        or config.min_true_sobel_second_mode_ratio
        + config.min_true_sobel_third_mode_ratio > 1.0
    ):
        raise ValueError(
            "Sobel direction mode ratios must satisfy third <= second and "
            "second + third <= 1"
        )
    return config


def _alpha_complex_projection_region(
    uv: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
    alpha_radius_px: float,
    *,
    rasterize: bool = False,
) -> tuple[dict, np.ndarray | None]:
    """Return alpha complex projection region."""
    coordinates = np.asarray(uv, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("uv must be an Nx2 array")
    if selected.shape != (len(coordinates),):
        raise ValueError("projection coverage mask must align with uv")
    if height < 1 or width < 1:
        raise ValueError("projection coverage image dimensions must be positive")
    if not np.isfinite(alpha_radius_px) or alpha_radius_px <= 0.0:
        raise ValueError("alpha radius must be finite and positive")
    points = coordinates[selected]
    if np.any(~np.isfinite(points)):
        raise ValueError("selected projection coordinates must be finite")
    if len(points):
        pixel_keys = np.floor(points).astype(np.int64)
        _, unique_indices = np.unique(pixel_keys, axis=0, return_index=True)
        points = points[np.sort(unique_indices)]
    empty_report = {
        "triangle_count": 0,
        "accepted_triangle_count": 0,
        "area_px2": 0.0,
        "coverage_ratio": 0.0,
    }
    empty_mask = (
        np.zeros((height, width), dtype=bool) if rasterize else None
    )
    if len(points) < 3:
        return empty_report, empty_mask
    try:
        simplices = Delaunay(points).simplices
    except QhullError:
        return empty_report, empty_mask
    triangles = points[simplices]
    edge_a = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    edge_b = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    edge_c = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    vector_a = triangles[:, 1] - triangles[:, 0]
    vector_b = triangles[:, 2] - triangles[:, 0]
    twice_area = np.abs(
        vector_a[:, 0] * vector_b[:, 1]
        - vector_a[:, 1] * vector_b[:, 0]
    )
    circumradius = np.full(len(triangles), np.inf, dtype=np.float64)
    nondegenerate = twice_area != 0.0
    circumradius[nondegenerate] = (
        edge_a[nondegenerate]
        * edge_b[nondegenerate]
        * edge_c[nondegenerate]
        / (2.0 * twice_area[nondegenerate])
    )
    accepted = circumradius <= alpha_radius_px
    area_px2 = float(np.sum(0.5 * twice_area[accepted]))
    report = {
        "triangle_count": int(len(triangles)),
        "accepted_triangle_count": int(np.count_nonzero(accepted)),
        "area_px2": area_px2,
        "coverage_ratio": float(area_px2 / (height * width)),
    }
    if not rasterize:
        return report, None
    alpha_shape = np.zeros((height, width), dtype=np.uint8)
    if np.any(accepted):
        polygons = np.rint(triangles[accepted]).astype(np.int32)
        polygons[..., 0] = np.clip(polygons[..., 0], 0, width - 1)
        polygons[..., 1] = np.clip(polygons[..., 1], 0, height - 1)
        cv2.fillPoly(alpha_shape, polygons, 1, lineType=cv2.LINE_8)
    return report, alpha_shape.astype(bool)


def _alpha_complex_projection_coverage(
    uv: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
    alpha_radius_px: float,
) -> dict:
    """Return alpha complex projection coverage."""
    report, _ = _alpha_complex_projection_region(
        uv, mask, height, width, alpha_radius_px, rasterize=False
    )
    return report


def _alpha_area_ratio(visible: dict, projected: dict) -> float:
    """Return alpha area ratio."""
    denominator = float(projected["area_px2"])
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(visible["area_px2"] / denominator, 0.0, 1.0))


def _unsigned_direction_distance_deg(
    first: np.ndarray | float,
    second: np.ndarray | float,
) -> np.ndarray:
    """Return unsigned direction distance deg."""
    difference = np.abs(
        np.asarray(first, dtype=np.float64)
        - np.asarray(second, dtype=np.float64)
    ) % 180.0
    return np.minimum(difference, 180.0 - difference)


def _unsigned_direction_mean_deg(values: np.ndarray) -> float:
    """Return unsigned direction mean deg."""
    directions = np.asarray(values, dtype=np.float64)
    if not len(directions) or np.any(~np.isfinite(directions)):
        raise ValueError("unsigned direction mean requires finite nonempty values")
    doubled = np.deg2rad(2.0 * directions)
    angle = np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
    return float((0.5 * np.degrees(angle)) % 180.0)


def _sobel_direction_mode_report(
    directions_deg: np.ndarray,
    config: GeneratedQualityConfig,
) -> dict:
    """Return sobel direction mode report."""
    directions = np.asarray(directions_deg, dtype=np.float64)
    if np.any(~np.isfinite(directions)) or np.any(
        (directions < 0.0) | (directions >= 180.0)
    ):
        raise ValueError("Sobel directions must be finite within [0, 180)")
    bin_count = int(round(180.0 / config.sobel_direction_histogram_bin_deg))
    histogram, edges = np.histogram(
        directions, bins=bin_count, range=(0.0, 180.0)
    )

    smoothed = (
        np.roll(histogram, 1) + 2.0 * histogram + np.roll(histogram, -1)
    ) / 4.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    candidates = [
        index for index in range(bin_count)
        if (
            smoothed[index] >= smoothed[(index - 1) % bin_count]
            and smoothed[index] >= smoothed[(index + 1) % bin_count]
            and (
                smoothed[index] > smoothed[(index - 1) % bin_count]
                or smoothed[index] > smoothed[(index + 1) % bin_count]
            )
        )
    ]
    candidates.sort(key=lambda index: (-smoothed[index], index))
    selected_indices: list[int] = []

    for index in candidates:
        if all(
            float(_unsigned_direction_distance_deg(
                centers[index], centers[other]
            )) >= config.min_true_sobel_direction_mode_separation_deg
            for other in selected_indices
        ):
            selected_indices.append(index)
        if len(selected_indices) == 3:
            break
    mode_centers = np.asarray(
        [centers[index] for index in selected_indices], dtype=np.float64
    )
    assignments = np.full(len(directions), -1, dtype=np.int64)

    for _ in range(2):
        if not len(mode_centers) or not len(directions):
            break
        distances = np.stack([
            _unsigned_direction_distance_deg(directions, center)
            for center in mode_centers
        ], axis=1)
        assignments = np.argmin(distances, axis=1)
        minimum_distances = np.min(distances, axis=1)
        assignments[
            minimum_distances
            > config.sobel_direction_mode_support_half_width_deg
        ] = -1
        for mode_index in range(len(mode_centers)):
            supported = directions[assignments == mode_index]
            if len(supported):
                mode_centers[mode_index] = _unsigned_direction_mean_deg(
                    supported
                )
    if len(mode_centers) and len(directions):
        distances = np.stack([
            _unsigned_direction_distance_deg(directions, center)
            for center in mode_centers
        ], axis=1)
        assignments = np.argmin(distances, axis=1)
        minimum_distances = np.min(distances, axis=1)
        assignments[
            minimum_distances
            > config.sobel_direction_mode_support_half_width_deg
        ] = -1
    modes = []
    denominator = len(directions)
    for mode_index, center in enumerate(mode_centers):
        count = int(np.count_nonzero(assignments == mode_index))
        if count:
            modes.append({
                "center_deg": float(center),
                "pixel_count": count,
                "ratio": float(count / denominator) if denominator else 0.0,
            })
    modes.sort(key=lambda mode: (-mode["pixel_count"], mode["center_deg"]))
    separations = [
        float(_unsigned_direction_distance_deg(
            modes[first]["center_deg"], modes[second]["center_deg"]
        ))
        for first in range(len(modes))
        for second in range(first + 1, len(modes))
    ]
    minimum_separation = min(separations) if separations else 0.0
    second_ratio = modes[1]["ratio"] if len(modes) >= 2 else 0.0
    third_ratio = modes[2]["ratio"] if len(modes) >= 3 else 0.0
    passed = bool(
        len(modes) >= 3
        and second_ratio >= config.min_true_sobel_second_mode_ratio
        and third_ratio >= config.min_true_sobel_third_mode_ratio
        and minimum_separation
        >= config.min_true_sobel_direction_mode_separation_deg
    )
    return {
        "histogram_bin_deg": config.sobel_direction_histogram_bin_deg,
        "histogram_counts": histogram.tolist(),
        "smoothed_histogram_counts": smoothed.tolist(),
        "mode_support_half_width_deg": (
            config.sobel_direction_mode_support_half_width_deg
        ),
        "mode_count": int(len(modes)),
        "mode_centers_deg": [mode["center_deg"] for mode in modes],
        "mode_pixel_counts": [mode["pixel_count"] for mode in modes],
        "mode_ratios": [mode["ratio"] for mode in modes],
        "assigned_pixel_ratio": float(
            sum(mode["pixel_count"] for mode in modes) / denominator
        ) if denominator else 0.0,
        "minimum_mode_separation_deg": float(minimum_separation),
        "minimum_required_separation_deg": (
            config.min_true_sobel_direction_mode_separation_deg
        ),
        "minimum_second_mode_ratio": config.min_true_sobel_second_mode_ratio,
        "minimum_third_mode_ratio": config.min_true_sobel_third_mode_ratio,
        "passed": passed,
        "semantic": (
            "equal_weight_unsigned_circular_histogram_three_separated_modes"
        ),
    }


def camera_projection_visibility(
    uv: np.ndarray,
    camera_depth_points: np.ndarray,
    in_image: np.ndarray,
    target_camera_depth: np.ndarray,
    config: GeneratedQualityConfig,
) -> np.ndarray:
    """Return camera projection visibility."""
    coordinates = np.asarray(uv, dtype=np.float64)
    depths = np.asarray(camera_depth_points, dtype=np.float64)
    image_mask = np.asarray(in_image, dtype=bool)
    gt_depth = np.asarray(target_camera_depth, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("uv must be an Nx2 array")
    if depths.shape != (len(coordinates),) or image_mask.shape != (len(coordinates),):
        raise ValueError("projection visibility arrays must align")
    if gt_depth.ndim != 2 or np.any(np.asarray(gt_depth.shape) < 1):
        raise ValueError("target_camera_depth must be a non-empty HxW array")
    pixel_x = np.floor(coordinates[:, 0]).astype(np.int64)
    pixel_y = np.floor(coordinates[:, 1]).astype(np.int64)
    valid_pixel = (
        image_mask
        & np.isfinite(depths)
        & (depths > 0.0)
        & (pixel_x >= 0)
        & (pixel_x < gt_depth.shape[1])
        & (pixel_y >= 0)
        & (pixel_y < gt_depth.shape[0])
    )
    sampled = np.full(len(coordinates), np.nan, dtype=np.float64)
    sampled[valid_pixel] = gt_depth[pixel_y[valid_pixel], pixel_x[valid_pixel]]
    comparable = valid_pixel & np.isfinite(sampled) & (sampled > 0.0)
    tolerance = np.maximum(
        config.visibility_abs_tolerance_m,
        config.visibility_rel_tolerance * sampled,
    )
    return comparable & (depths <= sampled + tolerance)


def _depth_sobel_arrays(
    camera_depth: np.ndarray,
    projection_mask: np.ndarray,
    gradient_threshold_m_per_px: float,
) -> dict[str, np.ndarray]:
    """Return depth sobel arrays."""
    depth = np.asarray(camera_depth, dtype=np.float64)
    mask = np.asarray(projection_mask, dtype=bool)
    threshold = float(gradient_threshold_m_per_px)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(
            "camera depth and projection mask must have the same HxW shape"
        )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Sobel gradient threshold must be finite and nonnegative")
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 65000.0)
    safe_depth = np.where(valid, depth, 0.0)
    gradient_x = cv2.Sobel(
        safe_depth, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    gradient_y = cv2.Sobel(
        safe_depth, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    magnitude = np.hypot(gradient_x, gradient_y)
    neighborhood = cv2.erode(
        valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    magnitude[~neighborhood] = 0.0
    gradient_pixels = mask & neighborhood & np.isfinite(magnitude)
    edge_pixels = gradient_pixels & (magnitude > threshold)
    return {
        "depth": depth,
        "mask": mask,
        "valid": valid,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "magnitude": magnitude,
        "neighborhood": neighborhood,
        "gradient_pixels": gradient_pixels,
        "edge_pixels": edge_pixels,
    }


def _masked_depth_sobel_report(
    camera_depth: np.ndarray,
    projection_mask: np.ndarray,
    config: GeneratedQualityConfig,
    *,
    evaluate_direction: bool = False,
) -> dict:
    """Return masked depth sobel report."""
    arrays = _depth_sobel_arrays(
        camera_depth,
        projection_mask,
        config.depth_sobel_gradient_threshold_m_per_px,
    )
    mask = arrays["mask"]
    valid = arrays["valid"]
    gradient_x = arrays["gradient_x"]
    gradient_y = arrays["gradient_y"]
    magnitude = arrays["magnitude"]
    gradient_pixels = arrays["gradient_pixels"]
    finite_depth_in_mask = mask & valid
    denominator = int(np.count_nonzero(gradient_pixels))
    edge_pixel_count = int(np.count_nonzero(arrays["edge_pixels"]))
    edge_pixel_ratio = (
        float(edge_pixel_count / denominator) if denominator else 0.0
    )

    edge_pixel_log10_ratio = (
        float(np.log10(edge_pixel_ratio)) if edge_pixel_ratio > 0.0 else None
    )
    direction_report = {
        "checked": bool(evaluate_direction),
        "gate_enabled": False,
        "pixel_count": 0,
        "matrix": None,
        "eigenvalues_descending": None,
        "principal_direction_deg": None,
        "diversity": None,
        "minimum_diversity": (
            config.min_true_sobel_direction_diversity
            if evaluate_direction else None
        ),
        "structure_tensor_passed": None,
        "direction_modes": None,
        "three_mode_passed": None,
        "passed_by": None,
        "reference_passed": None,
        "semantic": (
            "audit_only_structure_tensor_or_three_separated_unsigned_"
            "direction_modes"
        ),
    }
    if evaluate_direction:
        direction_mask = arrays["edge_pixels"]
        direction_count = int(np.count_nonzero(direction_mask))
        matrix = np.zeros((2, 2), dtype=np.float64)
        eigenvalues = np.zeros(2, dtype=np.float64)
        principal_direction_deg = None
        if direction_count:
            unit_x = gradient_x[direction_mask] / magnitude[direction_mask]
            unit_y = gradient_y[direction_mask] / magnitude[direction_mask]
            matrix = np.array([
                [np.mean(unit_x * unit_x), np.mean(unit_x * unit_y)],
                [np.mean(unit_x * unit_y), np.mean(unit_y * unit_y)],
            ], dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
            principal = eigenvectors[:, 1]
            principal_direction_deg = float(
                np.degrees(np.arctan2(principal[1], principal[0])) % 180.0
            )
        diversity = float(np.clip(2.0 * eigenvalues[0], 0.0, 1.0))
        structure_tensor_passed = bool(
            diversity >= config.min_true_sobel_direction_diversity
        )
        directions_deg = np.degrees(np.mod(
            np.arctan2(
                gradient_y[direction_mask], gradient_x[direction_mask]
            ),
            np.pi,
        ))
        mode_report = _sobel_direction_mode_report(directions_deg, config)
        three_mode_passed = bool(mode_report["passed"])
        reference_passed = structure_tensor_passed or three_mode_passed
        if structure_tensor_passed and three_mode_passed:
            passed_by = "both"
        elif structure_tensor_passed:
            passed_by = "structure_tensor"
        elif three_mode_passed:
            passed_by = "three_direction_modes"
        else:
            passed_by = "none"
        direction_report.update({
            "pixel_count": direction_count,
            "matrix": matrix.tolist(),
            "eigenvalues_descending": eigenvalues[::-1].tolist(),
            "principal_direction_deg": principal_direction_deg,
            "diversity": diversity,
            "structure_tensor_passed": structure_tensor_passed,
            "direction_modes": mode_report,
            "three_mode_passed": three_mode_passed,
            "passed_by": passed_by,
            "reference_passed": bool(reference_passed),
        })
    return {
        "algorithm": "sobel_3x3_scale_1_over_8_alpha_shape_v1",
        "mask_semantic": (
            "all_in_image_lidar_projection_alpha_shape_accepted_triangles"
        ),
        "mask_pixel_count": int(np.count_nonzero(mask)),
        "mask_area_ratio": float(np.mean(mask)),
        "finite_depth_pixel_count_in_mask": int(
            np.count_nonzero(finite_depth_in_mask)
        ),
        "gradient_magnitude_pixel_count_in_mask": denominator,
        "threshold_m_per_px": (
            config.depth_sobel_gradient_threshold_m_per_px
        ),
        "max_m_per_px": (
            float(np.max(magnitude[gradient_pixels])) if denominator else 0.0
        ),
        "edge_pixel_count": edge_pixel_count,
        "edge_pixel_denominator": denominator,
        "edge_pixel_ratio": edge_pixel_ratio,
        "edge_pixel_log10_ratio": edge_pixel_log10_ratio,
        "edge_pixel_log10_zero_semantic": "null",
        "minimum_edge_pixel_ratio": config.min_true_sobel_edge_pixel_ratio,
        "nan_depth_excluded": True,
        "edge_pixel_denominator_semantic": (
            "all_mask_pixels_with_finite_3x3_sobel_gradient_magnitude"
        ),
        "direction_diversity": direction_report,
        "passed": bool(
            edge_pixel_ratio >= config.min_true_sobel_edge_pixel_ratio
        ),
    }


def evaluate_generated_projection_quality(
    point_cloud: np.ndarray,
    camera_depth: np.ndarray,
    camera_depth_unscaled: np.ndarray,
    candidate: dict,
    nominal_extrinsic: np.ndarray,
    sensor: TrainingSensorConfig,
    config: GeneratedQualityConfig,
    *,
    check_nominal: bool = True,
    stop_on_rejection: bool = False,
) -> Tuple[dict, List[str]]:
    """Evaluate generated projection quality."""
    cloud = np.asarray(point_cloud)
    if (
        cloud.ndim != 2
        or cloud.shape[1] != 4
        or cloud.dtype != np.float32
        or not np.all(np.isfinite(cloud))
    ):
        raise ValueError("generated point cloud must be finite Nx4 float32")
    points = cloud[:, :3]
    true_extrinsic = validate_transform(
        np.asarray(candidate["true_extrinsic"], dtype=np.float64)
    )
    nominal = validate_transform(np.asarray(nominal_extrinsic, dtype=np.float64))
    height = sensor.camera.height
    width = sensor.camera.width

    true_uv, true_depth, true_in = project_points(
        points, true_extrinsic, sensor.camera.matrix, height, width
    )
    true_visible = camera_projection_visibility(
        true_uv, true_depth, true_in, camera_depth, config
    )
    true_projection, true_alpha_shape = _alpha_complex_projection_region(
        true_uv,
        true_in,
        height,
        width,
        config.alpha_radius_px,
        rasterize=True,
    )
    if true_alpha_shape is None:
        raise RuntimeError("Ti Alpha Shape rasterization was not produced")
    true_visible_projection = _alpha_complex_projection_coverage(
        true_uv, true_visible, height, width, config.alpha_radius_px
    )
    true_visible_area_ratio = _alpha_area_ratio(
        true_visible_projection, true_projection
    )
    true_occluded_area_ratio = float(1.0 - true_visible_area_ratio)
    reasons = []
    if true_projection["coverage_ratio"] < config.min_true_projection_coverage_ratio:
        reasons.append("insufficient_true_projection_coverage")
    if true_occluded_area_ratio > config.max_true_occluded_area_ratio:
        reasons.append("excessive_true_occluded_area_ratio")
    true_sobel = _masked_depth_sobel_report(
        camera_depth_unscaled,
        true_alpha_shape,
        config,
        evaluate_direction=True,
    )
    if not true_sobel["passed"]:
        reasons.append("insufficient_true_sobel_edge_pixel_ratio")
    nominal_projection = true_projection
    if check_nominal:
        nominal_uv, _, nominal_in = project_points(
            points, nominal, sensor.camera.matrix, height, width
        )
        nominal_projection = _alpha_complex_projection_coverage(
            nominal_uv, nominal_in, height, width, config.alpha_radius_px
        )
        if (
            nominal_projection["coverage_ratio"]
            < config.min_nominal_projection_coverage_ratio
        ):
            reasons.append("insufficient_nominal_projection_coverage")

    report = {
        "alpha_radius_px": config.alpha_radius_px,
        "true_projection_alpha_complex": true_projection,
        "true_visible_projection_alpha_complex": true_visible_projection,
        "true_projection_coverage_ratio": true_projection["coverage_ratio"],
        "true_visible_projection_area_ratio": true_visible_area_ratio,
        "true_occluded_projection_area_ratio": true_occluded_area_ratio,
        "true_depth_sobel": true_sobel,
        "sobel_depth_semantic": (
            "virtual_pinhole_axial_depth_before_physical_depth_scale"
        ),
        "sobel_depth_scale_applied": False,
        "physical_depth_scale": float(candidate["depth_scale"]),
        "nominal_projection_checked": bool(check_nominal),
        "nominal_projection_alpha_complex": nominal_projection,
        "nominal_projection_coverage_ratio": nominal_projection["coverage_ratio"],
        "nominal_visibility_checked": False,
        "nominal_sobel_checked": False,
        "visibility_semantic": (
            "camera_pixel_depth_comparison_then_alpha_area_visible_over_all_projected"
        ),
        "true_visibility_denominator_semantic": (
            "true_visible_alpha_area_divided_by_true_all_projection_alpha_area"
        ),
        "projection_area_semantic": (
            "all_in_image_lidar_projection_alpha_complex_continuous_area"
        ),
        "sobel_quality_order": (
            "true_projection_coverage_then_true_visibility_then_alpha_shape_mask"
        ),
    }
    if stop_on_rejection and reasons:
        report["early_rejection_stage"] = "projection_visibility_and_depth_sobel"
    return report, reasons


def combine_generated_quality_reports(
    projection_report: dict,
    reasons: List[str],
) -> dict:
    """Return combine generated quality reports."""
    return {
        "version": GENERATED_QUALITY_VERSION,
        "passed": not reasons,
        **projection_report,
    }


def evaluate_generated_quality(
    image_bgr: np.ndarray,
    camera_depth: np.ndarray,
    point_cloud: np.ndarray,
    beam_indices: np.ndarray,
    beam_grid_shape: tuple[int, int],
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    candidate: dict,
    nominal_extrinsic: np.ndarray,
    sensor: TrainingSensorConfig,
    config: GeneratedQualityConfig,
) -> Tuple[dict, List[str]]:
    """Evaluate generated quality."""
    image = np.asarray(image_bgr)
    expected_shape = (sensor.camera.height, sensor.camera.width, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise ValueError("generated image violates mandatory profile")
    del beam_indices, beam_grid_shape, target, pano_from_face
    projection_report, projection_reasons = (
        evaluate_generated_projection_quality(
            point_cloud,
            camera_depth,
            np.asarray(camera_depth, dtype=np.float64)
            / float(candidate["depth_scale"]),
            candidate,
            nominal_extrinsic,
            sensor,
            config,
        )
    )
    return combine_generated_quality_reports(
        projection_report, projection_reasons
    ), projection_reasons


from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .sample_generation import (
    DIRECT_BOX_COORDINATE_ORDER,
    fibonacci_directions,
    theoretical_translation_cells,
    translation_cell_id,
)


GEODESIC_COVERAGE_VERSION = "geodesic_product_ball_coverage_v2"
DIRECT_BOX_COVERAGE_VERSION = "direct_euler_component_box_coverage_v1"


def _rotation_cell_id(
    normalized_rotvec: np.ndarray,
    config: CoverageConfig,
    directions: np.ndarray,
) -> str:
    """Return rotation cell ID."""
    return translation_cell_id(
        normalized_rotvec, config, directions
    ).replace("t_", "r_", 1)


def theoretical_rotation_cells(config: CoverageConfig) -> list[str]:
    """Return theoretical rotation cells."""
    return [
        value.replace("t_", "r_", 1)
        for value in theoretical_translation_cells(config)
    ]


def _rotation_parts(cell: str) -> tuple[int, int]:
    """Return rotation parts."""
    if cell == "r_center":
        return 0, -1
    parts = cell.split("_")
    return int(parts[1][1:]), int(parts[2][1:])


def audit_geodesic_coverage(
    candidates: Iterable[dict],
    config: CoverageConfig,
) -> dict:
    """Audit geodesic coverage."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("stochastic coverage requires accepted candidates")
    directions = fibonacci_directions(config.translation_direction_cells)
    translation_cells = theoretical_translation_cells(config)
    rotation_cells = theoretical_rotation_cells(config)
    translation_counts = Counter()
    rotation_counts = Counter()
    translation_pairs = defaultdict(set)
    translation_rotations = defaultdict(set)
    translation_rotation_axes = defaultdict(set)
    translation_rotation_shells = defaultdict(set)
    rotation_translations = defaultdict(set)
    support = defaultdict(Counter)
    pair_counts = Counter()

    for candidate in candidates:
        coordinate = np.asarray(candidate["normalized"], dtype=np.float64)
        if coordinate.shape != (6,) or not np.all(np.isfinite(coordinate)):
            raise ValueError("candidate normalized coordinate must be finite length six")
        translation_cell = translation_cell_id(
            coordinate[:3], config, directions
        )
        rotation_cell = _rotation_cell_id(
            coordinate[3:], config, directions
        )
        shell, axis = _rotation_parts(rotation_cell)
        pair_id = str(candidate["pair_id"])
        translation_counts[translation_cell] += 1
        rotation_counts[rotation_cell] += 1
        translation_pairs[translation_cell].add(pair_id)
        translation_rotations[translation_cell].add(rotation_cell)
        translation_rotation_shells[translation_cell].add(shell)
        if axis >= 0:
            translation_rotation_axes[translation_cell].add(axis)
        rotation_translations[rotation_cell].add(translation_cell)
        support[translation_cell][rotation_cell] += 1
        pair_counts[pair_id] += 1

    occupied_translation = [
        cell for cell in translation_cells if translation_counts[cell]
    ]
    occupied_rotation = [cell for cell in rotation_cells if rotation_counts[cell]]
    translation_occupancy = len(occupied_translation) / len(translation_cells)
    rotation_occupancy = len(occupied_rotation) / len(rotation_cells)
    failures = []
    for name, value in (
        ("translation", translation_occupancy),
        ("rotation", rotation_occupancy),
    ):
        if value < config.min_translation_cell_occupancy:
            failures.append(
                f"{name} cell occupancy {value:.6f} < "
                f"{config.min_translation_cell_occupancy:.6f}"
            )

    translation_reports = {}
    for cell in translation_cells:
        report = {
            "candidate_count": int(translation_counts[cell]),
            "pair_count": len(translation_pairs[cell]),
            "rotation_cell_count": len(translation_rotations[cell]),
            "rotation_axis_count": len(translation_rotation_axes[cell]),
            "rotation_shell_count": len(translation_rotation_shells[cell]),
        }
        translation_reports[cell] = report
        if not report["candidate_count"]:
            continue
        checks = (
            ("candidate_count", config.min_candidates_per_translation_cell),
            ("pair_count", config.min_pairs_per_translation_cell),
            (
                "rotation_cell_count",
                config.min_rotation_cells_per_translation_cell,
            ),
            ("rotation_axis_count", config.min_rotation_axes_per_translation_cell),
            (
                "rotation_shell_count",
                config.min_rotation_shells_per_translation_cell,
            ),
        )
        for field, minimum in checks:
            if report[field] < minimum:
                failures.append(f"{cell}: {field} {report[field]} < {minimum}")

    rotation_reports = {}
    for cell in rotation_cells:
        translation_count = len(rotation_translations[cell])
        rotation_reports[cell] = {
            "candidate_count": int(rotation_counts[cell]),
            "translation_cell_count": translation_count,
        }
        if rotation_counts[cell] and (
            translation_count < config.min_translation_cells_per_rotation_cell
        ):
            failures.append(
                f"{cell}: translation_cell_count {translation_count} < "
                f"{config.min_translation_cells_per_rotation_cell}"
            )

    occupied_translation_reports = [
        translation_reports[cell] for cell in occupied_translation
    ]
    occupied_rotation_reports = [
        rotation_reports[cell] for cell in occupied_rotation
    ]
    return {
        "version": GEODESIC_COVERAGE_VERSION,
        "mode": GEODESIC_PRODUCT_BALL_COVERAGE,
        "coverage_target_enforced": True,
        "passed": not failures,
        "failures": failures,
        "candidate_count": len(candidates),
        "translation_cell_count": len(translation_cells),
        "occupied_translation_cells": len(occupied_translation),
        "translation_cell_occupancy": translation_occupancy,
        "empty_translation_cells": [
            cell for cell in translation_cells if not translation_counts[cell]
        ],
        "rotation_cell_count": len(rotation_cells),
        "occupied_rotation_cells": len(occupied_rotation),
        "rotation_cell_occupancy": rotation_occupancy,
        "empty_rotation_cells": [
            cell for cell in rotation_cells if not rotation_counts[cell]
        ],
        "unique_pair_count": len(pair_counts),
        "maximum_pair_reuse": max(pair_counts.values()),
        "conditional_minima": {
            "candidates_per_occupied_translation_cell": min(
                value["candidate_count"]
                for value in occupied_translation_reports
            ),
            "pairs_per_occupied_translation_cell": min(
                value["pair_count"] for value in occupied_translation_reports
            ),
            "rotation_cells_per_occupied_translation_cell": min(
                value["rotation_cell_count"]
                for value in occupied_translation_reports
            ),
            "rotation_axes_per_occupied_translation_cell": min(
                value["rotation_axis_count"]
                for value in occupied_translation_reports
            ),
            "rotation_shells_per_occupied_translation_cell": min(
                value["rotation_shell_count"]
                for value in occupied_translation_reports
            ),
            "translation_cells_per_occupied_rotation_cell": min(
                value["translation_cell_count"]
                for value in occupied_rotation_reports
            ),
        },
        "translation_cells": translation_reports,
        "rotation_cells": rotation_reports,
        "support_matrix": {
            cell: dict(sorted(support[cell].items()))
            for cell in translation_cells
        },
    }


def _direct_box_cell_indices(
    normalized: np.ndarray, bins_per_axis: int
) -> tuple[int, ...]:
    """Return direct box cell indices."""
    value = np.asarray(normalized, dtype=np.float64)
    if bins_per_axis < 2:
        raise ValueError("bins_per_axis must be at least two")
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("Direct-Euler box coordinate must be finite length six")
    if np.any(np.abs(value) > 1.0 + 1e-10):
        raise ValueError("Direct-Euler coordinate lies outside [-1, 1]^6")
    clipped = np.clip(value, -1.0, 1.0)
    indices = np.floor((clipped + 1.0) * 0.5 * bins_per_axis).astype(np.int64)
    indices = np.minimum(indices, bins_per_axis - 1)
    return tuple(int(index) for index in indices)


def _direct_box_cell_id(indices: tuple[int, ...]) -> str:
    """Return direct box cell ID."""
    if len(indices) != 6:
        raise ValueError("Direct-Euler cell index must contain six values")
    return ":".join(str(int(index)) for index in indices)


def _direct_box_report_from_counts(
    counts: Counter, bins_per_axis: int
) -> dict:
    """Return direct box report from counts."""
    total_cells = int(bins_per_axis ** 6)
    normalized_counts = Counter()
    decoded = {}
    for key, raw_count in counts.items():
        indices = tuple(int(value) for value in str(key).split(":"))
        if len(indices) != 6 or any(
            index < 0 or index >= bins_per_axis for index in indices
        ):
            raise ValueError(f"invalid Direct-Euler cell ID: {key}")
        count = int(raw_count)
        if count < 1:
            raise ValueError("sparse Direct-Euler counts must be positive")
        cell_id = _direct_box_cell_id(indices)
        normalized_counts[cell_id] += count
        decoded[cell_id] = indices
    sample_count = int(sum(normalized_counts.values()))
    occupied = len(normalized_counts)
    expected_ratio = (
        1.0 - (1.0 - 1.0 / total_cells) ** sample_count
        if sample_count else 0.0
    )
    values = np.asarray(list(normalized_counts.values()), dtype=np.float64)
    marginal = {name: [0] * bins_per_axis for name in DIRECT_BOX_COORDINATE_ORDER}
    pairwise = {}
    for left in range(6):
        for right in range(left + 1, 6):
            pairwise[f"{DIRECT_BOX_COORDINATE_ORDER[left]}__{DIRECT_BOX_COORDINATE_ORDER[right]}"] = [
                [0] * bins_per_axis for _ in range(bins_per_axis)
            ]
    for cell_id, count in normalized_counts.items():
        indices = decoded[cell_id]
        for axis, name in enumerate(DIRECT_BOX_COORDINATE_ORDER):
            marginal[name][indices[axis]] += count
        for left in range(6):
            for right in range(left + 1, 6):
                key = f"{DIRECT_BOX_COORDINATE_ORDER[left]}__{DIRECT_BOX_COORDINATE_ORDER[right]}"
                pairwise[key][indices[left]][indices[right]] += count
    expected_marginal = sample_count / bins_per_axis if sample_count else 0.0
    deviations = {}
    for name, axis_counts in marginal.items():
        deviations[name] = (
            max(abs(value - expected_marginal) for value in axis_counts)
            / expected_marginal
            if expected_marginal else 0.0
        )
    collision_numerator = sum(
        count * (count - 1) for count in normalized_counts.values()
    )
    collision_probability = (
        collision_numerator / (sample_count * (sample_count - 1))
        if sample_count > 1 else 0.0
    )
    bin_edges = [
        round(float(value), 12)
        for value in np.linspace(-1.0, 1.0, bins_per_axis + 1)
    ]
    return {
        "sample_count": sample_count,
        "coordinate_order": list(DIRECT_BOX_COORDINATE_ORDER),
        "bins_per_axis": int(bins_per_axis),
        "normalized_bin_edges": bin_edges,
        "total_cells": total_cells,
        "occupied_cells": occupied,
        "occupancy_ratio": float(occupied / total_cells),
        "expected_occupied_cells_under_iid_uniform": float(
            total_cells * expected_ratio
        ),
        "expected_occupancy_ratio_under_iid_uniform": float(expected_ratio),
        "maximum_cell_count": int(np.max(values)) if len(values) else 0,
        "p95_occupied_cell_count": (
            float(np.percentile(values, 95.0)) if len(values) else 0.0
        ),
        "collision_probability": float(collision_probability),
        "uniform_collision_probability": float(1.0 / total_cells),
        "marginal_bin_counts": marginal,
        "marginal_max_relative_deviation": deviations,
        "maximum_marginal_relative_deviation": (
            float(max(deviations.values())) if deviations else 0.0
        ),
        "pairwise_bin_counts": pairwise,
        "sparse_cell_counts": dict(sorted(normalized_counts.items())),
    }


def audit_direct_box_distribution(
    records: Iterable[dict], bins_per_axis: int
) -> dict:
    """Audit direct box distribution."""
    counts = Counter()
    for record in records:
        indices = _direct_box_cell_indices(
            np.asarray(record["direct_box_normalized"], dtype=np.float64),
            bins_per_axis,
        )
        counts[_direct_box_cell_id(indices)] += 1
    return _direct_box_report_from_counts(counts, bins_per_axis)


def audit_direct_box_coverage(
    proposed: Iterable[dict], accepted: Iterable[dict], bins_per_axis: int
) -> dict:
    """Audit direct box coverage."""
    proposed = list(proposed)
    accepted = list(accepted)
    unique = {}
    for record in accepted:
        identity = str(record["residual_identity"])
        coordinate = list(record["direct_box_normalized"])
        previous = unique.get(identity)
        if previous is not None and not np.allclose(
            previous["direct_box_normalized"], coordinate, rtol=0.0, atol=1e-12
        ):
            raise ValueError("one residual identity maps to multiple box coordinates")
        unique.setdefault(identity, record)
    return {
        "version": DIRECT_BOX_COVERAGE_VERSION,
        "mode": DIRECT_EULER_BOX_COVERAGE,
        "passed": True,
        "failures": [],
        "coverage_target_enforced": False,
        "proposed_residuals": audit_direct_box_distribution(
            proposed, bins_per_axis
        ),
        "accepted_unique_residuals": audit_direct_box_distribution(
            unique.values(), bins_per_axis
        ),
        "accepted_samples": audit_direct_box_distribution(
            accepted, bins_per_axis
        ),
    }


def audit_fixed_identity_coverage(candidates: Iterable[dict]) -> dict:
    """Audit fixed identity coverage."""
    candidates = list(candidates)
    invalid = [
        index for index, value in enumerate(candidates)
        if not np.allclose(value["normalized"], np.zeros(6), rtol=0.0, atol=1e-12)
    ]
    return {
        "version": "fixed_identity_coverage_v1",
        "mode": FIXED_IDENTITY_COVERAGE,
        "passed": not invalid,
        "failures": ([f"nonzero fixed residual indices: {invalid[:20]}"] if invalid else []),
        "coverage_target_enforced": False,
        "sample_count": len(candidates),
        "unique_residual_count": 1 if candidates else 0,
        "residual": [0.0] * 6,
    }


def audit_stochastic_coverage(
    candidates: Iterable[dict],
    config: CoverageConfig,
    proposed_direct_box: Iterable[dict] = (),
) -> dict:
    """Audit stochastic coverage."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("coverage audit requires accepted candidates")
    if config.mode == FIXED_IDENTITY_COVERAGE:
        return audit_fixed_identity_coverage(candidates)
    if config.mode == DIRECT_EULER_BOX_COVERAGE:
        return audit_direct_box_coverage(
            proposed_direct_box, candidates, int(config.bins_per_axis)
        )
    if config.mode == GEODESIC_PRODUCT_BALL_COVERAGE:
        return audit_geodesic_coverage(candidates, config)
    raise ValueError(f"unsupported coverage mode: {config.mode}")


def merge_direct_box_coverage_reports(
    reports: Iterable[dict], bins_per_axis: int
) -> dict:
    """Merge direct box coverage reports."""
    reports = list(reports)
    merged = {}
    for layer in (
        "proposed_residuals", "accepted_unique_residuals", "accepted_samples"
    ):
        counts = Counter()
        for report in reports:
            if report.get("mode") != DIRECT_EULER_BOX_COVERAGE:
                raise ValueError("cannot merge coverage reports from different modes")
            counts.update(report[layer]["sparse_cell_counts"])
        merged[layer] = _direct_box_report_from_counts(counts, bins_per_axis)
    return {
        "version": DIRECT_BOX_COVERAGE_VERSION,
        "mode": DIRECT_EULER_BOX_COVERAGE,
        "passed": True,
        "failures": [],
        "coverage_target_enforced": False,
        **merged,
    }


from collections import Counter
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Iterator


class StageTiming:
    """Represent stage timing."""

    def __init__(self):
        """Initialize the instance."""
        self._seconds = Counter()
        self._counts = Counter()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Return measure."""
        start = perf_counter()
        try:
            yield
        finally:
            self._seconds[str(stage)] += perf_counter() - start
            self._counts[str(stage)] += 1

    def add(self, stage: str, seconds: float, count: int = 1) -> None:
        """Add."""
        self._seconds[str(stage)] += float(seconds)
        self._counts[str(stage)] += int(count)

    def report(self) -> Dict[str, dict]:
        """Return report."""
        result = {}
        for stage in sorted(self._seconds):
            count = int(self._counts[stage])
            seconds = float(self._seconds[stage])
            result[stage] = {
                "count": count,
                "total_seconds": seconds,
                "mean_seconds": float(seconds / count) if count else 0.0,
            }
        return result


import hashlib
import csv
import json
import shutil
import uuid
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional, Union

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from ._runtime_log import emit_runtime_log
from .geometry import residual_transform, validate_transform
from .pair_pool import load_pair_selection_config
from .sample_generation import (
    DIRECT_LEFT_EULER_BOX,
    DIRECT_RESIDUAL_SAMPLER_MODES,
    FIXED_IDENTITY,
)
from .sensors import (
    RECTIFIED_1600_WIDE_PROFILE_NAME,
    TrainingSensorConfig,
    load_training_sensor_config,
    validate_training_sensor_config,
)
from .sensors import (
    clear_depth_cache,
    configure_depth_cache,
    depth_cache_report,
)
from .sample_generation import (
    ExtrinsicSampleGenerator,
    clear_rgb_cache,
    configure_rgb_cache,
    feasible_direction_probability,
    finalize_sensor_sample,
    generate_candidate_camera_depth,
    generate_candidate_lidar,
    rgb_cache_report,
)
from .pair_pool import load_manifest_sequence
from .sample_generation import (
    EqualAcceptedTranslationRadiusQuota,
    JointAcceptedCellQuota,
)
from .pair_pool import (
    FeasibilityConditionedPairSampler,
    build_or_load_approved_pair_pool,
)
from .generation_config import (
    DIRECT_EULER_BOX_COVERAGE,
    FIXED_IDENTITY_COVERAGE,
    GEODESIC_PRODUCT_BALL_COVERAGE,
    JOINT_BALANCED,
    REPORT_ONLY,
    RETRY_SAME_RESIDUAL,
    SOURCE_PRIMARY_ACCEPTANCE,
    GenerationTask,
)


STOCHASTIC_DATASET_VERSION = "stochastic_raw_omni_extrinsic_dataset_v18_single_camera"
SEQUENCE_PROGRESS_VERSION = "stochastic_sequence_progress_v2"


def _direct_pair_distribution(mode: str) -> str:
    """Return direct pair distribution."""
    if mode == FIXED_IDENTITY:
        return "fixed_extrinsic_scale_conditioned_uniform_approved_pairs"
    if mode == DIRECT_LEFT_EULER_BOX:
        return "direct_residual_scale_conditioned_uniform_approved_pairs"
    else:
        raise ValueError(f"unsupported direct residual mode: {mode}")


def _paired_sensor_config(
    task: GenerationTask,
    base_sensor: TrainingSensorConfig,
) -> Optional[TrainingSensorConfig]:
    """Return paired sensor config."""
    paired = task.paired_lidar_export
    if paired is None:
        return None
    sensor = replace(
        base_sensor,
        profile=paired.sensor_profile,
        lidar=replace(
            base_sensor.lidar,
            vertical_fov_deg=paired.vertical_fov_deg,
        ),
    )
    validate_training_sensor_config(sensor)
    return sensor


class StochasticGenerationError(RuntimeError):
    """Report a stochastic generation failure."""

    def __init__(self, message: str, report: dict):
        """Initialize the instance."""
        super().__init__(message)
        self.report = report

    def __reduce__(self):
        """Reduce."""
        return type(self), (str(self), self.report)


def _derived_seed(base_seed: int, level: str, sequence_id: str) -> int:
    """Return derived seed."""
    digest = hashlib.blake2b(
        f"{base_seed}:{level}:{sequence_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def _level(task: GenerationTask, name: str):
    """Return level."""
    matches = [value for value in task.levels if value.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"task {task.target_network} has no unique curriculum level {name!r}"
        )
    return matches[0]


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON atomic."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _transform_row(transform: np.ndarray) -> str:
    """Transform row."""
    return " ".join(
        f"{value:.18e}" for value in transform[:3, :].reshape(12)
    )


def _write_calib(path: Path, camera_matrix: np.ndarray) -> None:
    """Write calib."""
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    row = " ".join(f"{value:.10f}" for value in projection.reshape(-1))
    path.write_text(f"P2: {row}\n", encoding="utf-8")


def _write_transform_rows(path: Path, transforms: list[np.ndarray]) -> None:
    """Write transform rows."""
    rows = np.stack([
        validate_transform(value)[:3, :].reshape(12)
        for value in transforms
    ])
    np.savetxt(str(path), rows, fmt="%.18e")


def _load_transform_rows(path: Path, sample_count: int) -> np.ndarray:
    """Load transform rows."""
    rows = np.loadtxt(str(path), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    expected_shape = (sample_count, 12)
    if rows.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {rows.shape}")
    return rows


def audit_exported_sequence(
    directory: Union[str, Path],
    sensor_config: Union[str, Path, dict],
) -> dict:
    """Audit exported sequence."""
    root = Path(directory)
    sensor = (
        sensor_config
        if isinstance(sensor_config, TrainingSensorConfig)
        else load_training_sensor_config(sensor_config)
    )
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("version") != STOCHASTIC_DATASET_VERSION:
        raise ValueError(
            f"unsupported dataset version: {metadata.get('version')}"
        )
    if metadata.get("training_semantics") != "nominal_to_true_residual_v1":
        raise ValueError("metadata has invalid training semantics")
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("exported sequence contains no sample metadata")
    sample_count = len(samples)
    if int(metadata.get("accepted_sample_target", -1)) != sample_count:
        raise ValueError("metadata sample count does not match accepted target")

    image_paths = sorted((root / "image_2").glob("*.png"))
    cloud_paths = sorted((root / "velodyne").glob("*.bin"))
    if len(image_paths) != sample_count or len(cloud_paths) != sample_count:
        raise ValueError("image, point cloud, and metadata counts differ")
    pose_rows = _load_transform_rows(root / "poses.txt", sample_count)
    residual_rows = _load_transform_rows(
        root / "residuals.txt", sample_count
    )
    nominal = validate_transform(np.loadtxt(
        str(root / "nominal_extrinsic.txt"), dtype=np.float64
    ))

    expected_projection = np.zeros((3, 4), dtype=np.float64)
    expected_projection[:, :3] = sensor.camera.matrix
    with (root / "calib.txt").open("r", encoding="utf-8") as handle:
        calib_lines = [line.strip() for line in handle if line.strip()]
    if len(calib_lines) != 1 or not calib_lines[0].startswith("P2:"):
        raise ValueError("calib.txt must contain exactly one P2 projection line")
    try:
        projection_values = np.asarray(
            [float(token) for token in calib_lines[0].split(":", 1)[1].split()],
            dtype=np.float64,
        )
    except ValueError as error:
        raise ValueError("calib.txt P2 contains a non-numeric value") from error
    if projection_values.shape != (12,) or not np.all(np.isfinite(
        projection_values
    )):
        raise ValueError("calib.txt P2 must contain 12 finite values")
    if not np.allclose(
        projection_values.reshape(3, 4),
        expected_projection,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("calib.txt P2 does not match configured camera")

    derived_export = (
        metadata.get("lidar_modality_derivation")
        or metadata.get("paired_lidar_derivative")
        or {}
    )
    source_primary_acceptance = (
        derived_export.get("acceptance_policy")
        == SOURCE_PRIMARY_ACCEPTANCE
    )
    retained_quality_failure_count = 0
    scene_names = set()
    for index, (image_path, cloud_path, sample) in enumerate(
        zip(image_paths, cloud_paths, samples)
    ):
        frame_id = f"{index:06d}"
        if (
            image_path.stem != frame_id
            or cloud_path.stem != frame_id
            or sample.get("frame_id") != frame_id
        ):
            raise ValueError(f"frame ordering mismatch at sample {index}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        expected_image_shape = (
            sensor.camera.height,
            sensor.camera.width,
            3,
        )
        if image is None or image.shape != expected_image_shape:
            raise ValueError(f"{image_path}: invalid image geometry")

        raw_cloud = np.fromfile(str(cloud_path), dtype=np.float32)
        if raw_cloud.size == 0 or raw_cloud.size % 4:
            raise ValueError(f"{cloud_path}: point cloud is not nonempty Nx4")
        cloud = raw_cloud.reshape(-1, 4)
        if not np.all(np.isfinite(cloud)) or not np.all(cloud[:, 3] == 0.0):
            raise ValueError(f"{cloud_path}: invalid XYZ or dummy intensity")
        ranges_m = np.linalg.norm(cloud[:, :3], axis=1)
        if (
            float(ranges_m.min()) < sensor.lidar.min_range_m - 1e-5
            or float(ranges_m.max()) > sensor.lidar.max_range_m + 1e-4
        ):
            raise ValueError(f"{cloud_path}: points violate LiDAR range")

        true_transform = np.eye(4, dtype=np.float64)
        true_transform[:3, :] = pose_rows[index].reshape(3, 4)
        residual = np.eye(4, dtype=np.float64)
        residual[:3, :] = residual_rows[index].reshape(3, 4)
        true_transform = validate_transform(true_transform)
        residual = validate_transform(residual)
        stored_true = validate_transform(np.asarray(
            sample.get("true_extrinsic"), dtype=np.float64
        ))
        stored_residual = validate_transform(np.asarray(
            sample.get("residual"), dtype=np.float64
        ))
        expected_residual = residual_transform(true_transform, nominal)
        if not np.allclose(true_transform, stored_true, rtol=0.0, atol=1e-8):
            raise ValueError(f"pose metadata mismatch at sample {index}")
        if not np.allclose(residual, stored_residual, rtol=0.0, atol=1e-8):
            raise ValueError(f"residual metadata mismatch at sample {index}")
        if not np.allclose(residual, expected_residual, rtol=0.0, atol=1e-8):
            raise ValueError(f"left residual identity failed at sample {index}")
        normalized = np.asarray(sample.get("normalized"), dtype=np.float64)
        if normalized.shape != (6,) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"normalized residual invalid at sample {index}")
        quality_passed = bool(
            sample.get("generated_quality", {}).get("passed")
        )
        if not quality_passed:
            if not source_primary_acceptance:
                raise ValueError(
                    f"quality-rejected sample exported at {index}"
                )
            reasons = sample.get("quality_rejection_reasons")
            acceptance = sample.get("derived_acceptance") or {}
            if (
                sample.get("auto_keep") is not False
                or not isinstance(reasons, list)
                or not reasons
                or acceptance.get("policy") != SOURCE_PRIMARY_ACCEPTANCE
                or acceptance.get("retained_despite_derived_failure") is not True
            ):
                raise ValueError(
                    f"derived quality failure lacks explicit retention at {index}"
                )
            retained_quality_failure_count += 1
        scene_names.add(str(sample.get("scene", "")))

    if "" in scene_names:
        raise ValueError("sample metadata contains an empty scene name")
    return {
        "valid": True,
        "version": STOCHASTIC_DATASET_VERSION,
        "split": metadata.get("split"),
        "sample_count": sample_count,
        "scenes": sorted(scene_names),
        "sensor_profile": sensor.profile,
        "image_geometry_validated": True,
        "point_cloud_nx4_validated": True,
        "pose_residual_alignment_validated": True,
        "left_residual_identity_validated": True,
        "generated_quality_passed_validated": (
            retained_quality_failure_count == 0
        ),
        "source_primary_acceptance_validated": source_primary_acceptance,
        "retained_quality_failure_count": retained_quality_failure_count,
    }


def audit_paired_sequence_identity(
    primary_directory: Union[str, Path],
    derived_directory: Union[str, Path],
) -> dict:
    """Audit paired sequence identity."""
    primary = Path(primary_directory)
    derived = Path(derived_directory)
    with (primary / "metadata.json").open("r", encoding="utf-8") as handle:
        primary_metadata = json.load(handle)
    with (derived / "metadata.json").open("r", encoding="utf-8") as handle:
        derived_metadata = json.load(handle)
    primary_samples = primary_metadata.get("samples") or []
    derived_samples = derived_metadata.get("samples") or []
    if len(primary_samples) != len(derived_samples) or not primary_samples:
        raise ValueError("paired sequence metadata sample counts differ")
    control_derivative = (
        derived_metadata.get("dataset_role")
        == "lidar_modality_control_derivative"
    )
    legacy_identity_fields = (
        "frame_id", "candidate_id", "sequence_id", "scene", "source_frame",
        "target_frame", "pair_id", "curriculum_level", "depth_scale",
        "true_extrinsic", "residual", "normalized",
    )
    for index, (primary_sample, derived_sample) in enumerate(
        zip(primary_samples, derived_samples)
    ):
        if control_derivative:


            allowed_changed_fields = {
                "generated_sensor",
                "source_primary_generated_sensor",
                "lidar_modality_derivation",
            }
            primary_comparable = {
                key: value for key, value in primary_sample.items()
                if key not in allowed_changed_fields
            }
            derived_comparable = {
                key: value for key, value in derived_sample.items()
                if key not in allowed_changed_fields
            }
            mismatched = sorted(
                set(primary_comparable) ^ set(derived_comparable)
                | {
                    key
                    for key in set(primary_comparable) & set(derived_comparable)
                    if primary_comparable[key] != derived_comparable[key]
                }
            )
        else:
            mismatched = [
                field for field in legacy_identity_fields
                if primary_sample.get(field) != derived_sample.get(field)
            ]
        if mismatched:
            raise ValueError(
                f"paired metadata identity mismatch at {index}: {mismatched}"
            )
        if control_derivative and (
            derived_sample.get("source_primary_generated_sensor")
            != primary_sample.get("generated_sensor")
        ):
            raise ValueError(
                f"source generated sensor backup mismatch at {index}"
            )
        if control_derivative:
            source_sensor_report = dict(
                primary_sample.get("generated_sensor") or {}
            )
            target_sensor_report = dict(
                derived_sample.get("generated_sensor") or {}
            )
            lidar_changed_fields = {
                "version",
                "sensor_profile",
                "point_cloud_shape",
                "lidar",
                "lidar_modality_derivation",
            }
            source_non_lidar_report = {
                key: value for key, value in source_sensor_report.items()
                if key not in lidar_changed_fields
            }
            target_non_lidar_report = {
                key: value for key, value in target_sensor_report.items()
                if key not in lidar_changed_fields
            }
            if source_non_lidar_report != target_non_lidar_report:
                raise ValueError(
                    f"generated non-LiDAR sensor report mismatch at {index}"
                )
            derivation_report = (
                target_sensor_report
                .get("lidar_modality_derivation", {})
            )
            source_cloud = primary / "velodyne" / f"{index:06d}.bin"
            target_cloud = derived / "velodyne" / f"{index:06d}.bin"
            if (
                derivation_report.get("source_point_cloud_sha256")
                != _file_sha256(source_cloud)
                or derivation_report.get("target_point_cloud_sha256")
                != _file_sha256(target_cloud)
            ):
                raise ValueError(
                    f"point cloud fingerprint mismatch at {index}"
                )
    for filename in (
        "calib.txt",
        "poses.txt",
        "residuals.txt",
        "nominal_extrinsic.txt",
    ):
        primary_text = primary / filename
        derived_text = derived / filename
        if primary_text.read_bytes() != derived_text.read_bytes():
            raise ValueError(f"paired supervision text differs: {filename}")
        primary_stat = primary_text.stat()
        derived_stat = derived_text.stat()
        if (
            primary_stat.st_dev == derived_stat.st_dev
            and primary_stat.st_ino == derived_stat.st_ino
        ):
            raise ValueError(
                f"paired supervision must be a physical copy: {filename}"
            )
    primary_images = sorted((primary / "image_2").glob("*.png"))
    derived_images = sorted((derived / "image_2").glob("*.png"))
    if [path.name for path in primary_images] != [
        path.name for path in derived_images
    ]:
        raise ValueError("paired image frame IDs differ")
    for primary_image, derived_image in zip(primary_images, derived_images):
        primary_stat = primary_image.stat()
        derived_stat = derived_image.stat()
        if primary_image.read_bytes() != derived_image.read_bytes():
            raise ValueError(f"paired image bytes differ: {primary_image.name}")
        if (
            primary_stat.st_dev == derived_stat.st_dev
            and primary_stat.st_ino == derived_stat.st_ino
        ):
            raise ValueError(
                f"paired image must be a physical copy: {primary_image.name}"
            )
    return {
        "valid": True,
        "sample_count": len(primary_samples),
        "metadata_identity_validated": True,
        "all_non_lidar_sample_fields_identical": (
            True if control_derivative else None
        ),
        "source_generated_sensor_backup_validated": (
            True if control_derivative else None
        ),
        "generated_non_lidar_sensor_report_identical": (
            True if control_derivative else None
        ),
        "source_and_target_point_cloud_sha256_validated": (
            True if control_derivative else None
        ),
        "supervision_text_identical": True,
        "supervision_physical_copy_validated": True,
        "image_frame_ids_and_bytes_identical": True,
        "image_physical_copy_validated": True,
    }


class _IncrementalSequenceWriter:
    """Represent _ incremental sequence writer."""

    def __init__(self, stage: Path):
        """Initialize the instance."""
        self.stage = stage
        self.samples_path = stage / "accepted_samples.jsonl"
        self.poses_path = stage / "accepted_poses.rows"
        self.residuals_path = stage / "accepted_residuals.rows"
        self._samples = self.samples_path.open("x", encoding="utf-8")
        self._poses = self.poses_path.open("x", encoding="utf-8")
        self._residuals = self.residuals_path.open("x", encoding="utf-8")
        self.closed = False

    def append(
        self,
        sample: dict,
        true_transform: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        """Return append."""
        json.dump(
            sample,
            self._samples,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._samples.write("\n")
        self._poses.write(_transform_row(true_transform) + "\n")
        self._residuals.write(_transform_row(residual) + "\n")
        self._samples.flush()
        self._poses.flush()
        self._residuals.flush()

    def close(self) -> None:
        """Close."""
        if self.closed:
            return
        self._samples.close()
        self._poses.close()
        self._residuals.close()
        self.closed = True

    def promote_transform_rows(self) -> None:
        """Promote transform rows."""
        self.close()
        self.poses_path.replace(self.stage / "poses.txt")
        self.residuals_path.replace(self.stage / "residuals.txt")

    def remove_sample_journal(self) -> None:
        """Remove sample journal."""
        self.samples_path.unlink()


def _write_metadata_with_sample_journal(
    path: Path,
    metadata: dict,
    samples_path: Path,
) -> None:
    """Write metadata with sample journal."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("{")
        first = True
        for key, value in metadata.items():
            if not first:
                handle.write(",")
            json.dump(key, handle, ensure_ascii=False)
            handle.write(":")
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            first = False
        if not first:
            handle.write(",")
        handle.write('"samples":[')
        sample_index = 0
        with samples_path.open("r", encoding="utf-8") as samples:
            for line in samples:
                value = line.strip()
                if not value:
                    continue
                if sample_index:
                    handle.write(",")
                handle.write(value)
                sample_index += 1
        handle.write("]}")
    temporary.replace(path)


REUSE_AUDIT_METRICS = (
    "pair",
    "source_frame",
    "target_frame",
    "target_view",
    "source_frame_unique_pairs",
    "target_frame_unique_pairs",
)


REUSE_AUDIT_PLOT_METRICS = (
    "pair",
    "source_frame",
    "target_frame",
)


def _reuse_metric_summary(
    counts: Counter,
    count_semantic: str = "accepted_samples",
) -> dict:
    """Reuse metric summary."""
    normalized = Counter({str(key): int(value) for key, value in counts.items()})
    if any(value < 0 for value in normalized.values()):
        raise ValueError("reuse counts must be nonnegative")
    values = np.asarray(list(normalized.values()), dtype=np.float64)
    total = int(np.sum(values)) if len(values) else 0
    report = {
        "count_semantic": str(count_semantic),
        "total_count": total,
        "unique_keys": int(len(values)),
        "maximum_reuse": int(np.max(values)) if len(values) else 0,
        "mean_reuse": float(np.mean(values)) if len(values) else 0.0,
        "p95_reuse": float(np.percentile(values, 95.0)) if len(values) else 0.0,
        "counts": dict(sorted(normalized.items())),
    }
    if count_semantic == "accepted_samples":
        report["total_accepted_samples"] = total
    return report


def _reuse_audit_from_counters(counters: Mapping[str, Counter]) -> dict:
    """Reuse audit from counters."""
    return {
        name: _reuse_metric_summary(
            counters[name],
            (
                "unique_directed_pair_associations"
                if name.endswith("_unique_pairs") else "accepted_samples"
            ),
        )
        for name in REUSE_AUDIT_METRICS
    }


def _target_view_key(sequence_id: str, target_frame: str, rotation) -> str:
    """Return target view key."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera view rotation must be a finite 3x3 matrix")
    values = ",".join(f"{float(value):.12g}" for value in matrix.reshape(-1))
    return f"{sequence_id}|{target_frame}|{values}"


def _reuse_audit_from_samples(samples: Iterable[dict]) -> dict:
    """Reuse audit from samples."""
    counters = {name: Counter() for name in REUSE_AUDIT_METRICS}
    source_pairs = defaultdict(set)
    target_pairs = defaultdict(set)
    for sample in samples:
        sequence_id = str(sample["sequence_id"])
        pair_key = f"{sequence_id}|{sample['pair_id']}"
        source_key = f"{sequence_id}|{sample['source_frame']}"
        target_key = f"{sequence_id}|{sample['target_frame']}"
        counters["pair"][pair_key] += 1
        counters["source_frame"][source_key] += 1
        counters["target_frame"][target_key] += 1
        counters["target_view"][_target_view_key(
            sequence_id, str(sample["target_frame"]),
            sample["pano_from_camera_rotation"],
        )] += 1
        source_pairs[source_key].add(pair_key)
        target_pairs[target_key].add(pair_key)
    counters["source_frame_unique_pairs"].update({
        key: len(values) for key, values in source_pairs.items()
    })
    counters["target_frame_unique_pairs"].update({
        key: len(values) for key, values in target_pairs.items()
    })
    return _reuse_audit_from_counters(counters)


def _merge_reuse_audits(audits: Iterable[dict]) -> dict:
    """Merge reuse audits."""
    counters = {name: Counter() for name in REUSE_AUDIT_METRICS}
    for audit in audits:
        for name in counters:
            counters[name].update(audit.get(name, {}).get("counts", {}))
    return _reuse_audit_from_counters(counters)


def _sample_metadata_record(
    frame_id: str,
    sequence_id: str,
    scene: str,
    level_name: str,
    candidate: dict,
    residual: np.ndarray,
    generated,
    quality: dict,
) -> dict:
    """Sample metadata record."""
    return {
        "frame_id": frame_id,
        "candidate_id": candidate["candidate_id"],
        "sequence_id": sequence_id,
        "scene": scene,
        "source_frame": candidate["source_frame"],
        "target_frame": candidate["target_frame"],
        "pair_id": candidate["pair_id"],
        "curriculum_level": level_name,
        "sensor_sampling_seed": candidate["sensor_sampling_seed"],
        "depth_scale": candidate["depth_scale"],
        "scale_interval": candidate["scale_interval"],
        "translation_sampling": candidate["translation_sampling"],
        "rotation_sampling": candidate["rotation_sampling"],
        "rotation_sampling_detail": candidate.get("rotation_sampling_detail"),
        "residual_perturbation_sampling": candidate.get(
            "residual_perturbation_sampling"
        ),
        "residual_draw_index": candidate.get("residual_draw_index"),
        "direct_box_coordinate_order": candidate.get(
            "direct_box_coordinate_order"
        ),
        "direct_box_components": candidate.get("direct_box_components"),
        "direct_box_normalized": candidate.get("direct_box_normalized"),
        "direct_box_cell_id": candidate.get("direct_box_cell_id"),
        "joint_cell_target": candidate.get("joint_cell_target"),
        "translation_radius_interval_index": candidate[
            "translation_radius_interval_index"
        ],
        "target_normalized_translation": candidate[
            "target_normalized_translation"
        ],
        "pano_from_lidar_rotation": candidate["pano_from_lidar_rotation"],
        "pano_from_camera_rotation": candidate["pano_from_camera_rotation"],
        "raw_relative_rotation": candidate["raw_relative_rotation"],
        "raw_relative_translation_m": candidate[
            "raw_relative_translation_m"
        ],
        "view_rotation_sampling": candidate["view_rotation_sampling"],
        "view_rotation_conditioning": candidate[
            "view_rotation_conditioning"
        ],
        "view_anchor": candidate["view_anchor"],
        "camera_rotation_sampled_independently": False,
        "lidar_rotation_sampled_independently": False,
        "camera_view_audit": candidate["camera_view_audit"],
        "lidar_view_audit": candidate["lidar_view_audit"],
        "true_extrinsic": candidate["true_extrinsic"],
        "residual": residual.tolist(),
        "normalized": candidate["normalized"],
        "generated_sensor": generated.report,
        "generated_quality": quality,
        "manual_keep": True,
        "auto_keep": True,
        "quality_level": "A",
        "quality_weight": 1.0,
    }


def _paired_sample_metadata_record(
    primary_sample: dict,
    generated,
    quality: dict,
    rejection_reasons: list[str],
) -> dict:
    """Return paired sample metadata record."""
    failed = bool(rejection_reasons)
    return {
        **primary_sample,
        "generated_sensor": generated.report,
        "generated_quality": quality,
        "auto_keep": not failed,
        "quality_level": "source_primary_retained" if failed else "A",
        "quality_weight": 0.0 if failed else 1.0,
        "quality_rejection_reasons": list(rejection_reasons),
        "derived_acceptance": {
            "policy": SOURCE_PRIMARY_ACCEPTANCE,
            "source_primary_passed": True,
            "derived_quality_passed": not failed,
            "retained_despite_derived_failure": failed,
        },
    }


def _consecutive_attempt_limit_reached(
    consecutive_attempts_without_acceptance: int,
    limit: int,
) -> bool:
    """Return consecutive attempt limit reached."""
    return consecutive_attempts_without_acceptance >= limit


def _sequence_working_directory(
    output: Path,
    direct_output_persistence: bool,
) -> Path:
    """Return sequence working directory."""
    if direct_output_persistence:
        return output
    return output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"


def _restore_worker_runtime(
    previous_opencv_threads: int,
    numeric_thread_limits,
) -> None:
    """Restore worker runtime."""
    clear_depth_cache()
    clear_rgb_cache()
    cv2.setNumThreads(previous_opencv_threads)
    numeric_thread_limits.restore_original_limits()


def filter_and_export_sequence(
    task: GenerationTask,
    source_manifest: Union[str, Path],
    sequence_id: str,
    level_name: str,
    output: Union[str, Path],
    paired_output: Optional[Union[str, Path]] = None,
    accepted_target_override: Optional[int] = None,
    allow_incomplete_smoke: bool = False,
    cache_root: Optional[Union[str, Path]] = None,
    skip_independent_audit: bool = False,
) -> dict:
    """Filter and export sequence."""
    level = _level(task, level_name)
    accepted_target = (
        level.accepted_samples_per_sequence
        if accepted_target_override is None
        else int(accepted_target_override)
    )
    smoke_limited = accepted_target_override is not None
    if accepted_target < 1:
        raise ValueError("accepted sample target must be positive")
    if smoke_limited and not allow_incomplete_smoke:
        raise ValueError(
            "accepted_target_override requires allow_incomplete_smoke=true"
        )
    if allow_incomplete_smoke and not smoke_limited:
        raise ValueError(
            "allow_incomplete_smoke requires accepted_target_override"
        )

    sequence, split, source_metadata = load_manifest_sequence(
        source_manifest, sequence_id
    )
    expected_split = task.source.split_for_scene(sequence.scene)
    if split != expected_split:
        raise ValueError(
            f"source manifest split {split} != task ownership {expected_split}"
        )
    quota_scenes = sorted({
        scene
        for scenes in task.source.scene_splits.values()
        for scene in scenes
    })
    if sequence.scene not in quota_scenes:
        raise ValueError("sequence scene has no joint-quota partition")
    radius_partition_index = quota_scenes.index(sequence.scene)
    radius_partition_count = len(quota_scenes)
    emit_runtime_log(
        "scene_started",
        "Starting scene sequence generation",
        scene=sequence.scene,
        sequence_id=sequence.sequence_id,
        split=split,
        level=level.name,
        accepted_target=accepted_target,
        paired_lidar_export=task.paired_lidar_export is not None,
    )

    scale_min, scale_max = task.scale_range_for_scene(sequence.scene)

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = _sequence_working_directory(
        output,
        task.runtime.direct_output_persistence,
    )
    if (task.paired_lidar_export is None) != (paired_output is None):
        raise ValueError(
            "paired_output must be configured exactly when paired_lidar_export is set"
        )
    resolved_paired_output = (
        Path(paired_output).resolve() if paired_output is not None else None
    )
    paired_stage = None
    if resolved_paired_output is not None:
        if resolved_paired_output == output:
            raise ValueError("paired output must differ from primary output")
        if resolved_paired_output.exists():
            raise FileExistsError(
                f"paired output already exists: {resolved_paired_output}"
            )
        resolved_paired_output.parent.mkdir(parents=True, exist_ok=True)
        paired_stage = _sequence_working_directory(
            resolved_paired_output, False
        )

    derived_seed = _derived_seed(task.seed, level.name, sequence.sequence_id)
    timing = StageTiming()
    total_start = perf_counter()
    previous_opencv_threads = cv2.getNumThreads()
    numeric_thread_limits = threadpool_limits(
        limits=task.runtime.numeric_threads_per_worker
    )
    cv2.setNumThreads(task.runtime.opencv_threads_per_worker)
    configure_depth_cache(task.runtime.depth_cache_max_mb * 1024 * 1024)
    configure_rgb_cache(task.runtime.rgb_cache_max_mb * 1024 * 1024)
    resolved_cache_root = Path(
        cache_root if cache_root is not None else output.parent / ".cache"
    ).resolve()
    pair_config = load_pair_selection_config(task.pair_quality_config)
    try:
        pair_pool_start = perf_counter()
        emit_runtime_log(
            "stage_started",
            "Loading or building the qualified frame-pair pool",
            scene=sequence.scene,
            level=level.name,
            cache_root=str(resolved_cache_root),
        )
        with timing.measure("pair_pool_load_or_build"):
            pair_pool = build_or_load_approved_pair_pool(
                sequence=sequence,
                sampling=task.pair_sampling,
                pair_config=pair_config,
                sequence_fingerprint=source_metadata["sequence_fingerprint"],
                pair_config_fingerprint=(
                    task.config_fingerprints["pair_quality"]
                ),
                cache_root=resolved_cache_root,
                progress_interval_seconds=(
                    task.runtime.progress_interval_seconds
                ),
            )
        emit_runtime_log(
            "stage_completed",
            "Qualified frame-pair pool ready",
            scene=sequence.scene,
            level=level.name,
            approved_pairs=len(pair_pool.candidates),
            raw_candidates=pair_pool.raw_candidate_count,
            elapsed_seconds=round(perf_counter() - pair_pool_start, 3),
        )
    except BaseException:
        _restore_worker_runtime(previous_opencv_threads, numeric_thread_limits)
        raise
    target_norm = float(np.linalg.norm(task.nominal.matrix[:3, 3]))
    translation_limit = level.residual_translation_normalization_m(
        task.nominal.matrix
    )
    try:
        feasibility_start = perf_counter()
        emit_runtime_log(
            "stage_started",
            "Computing physical feasibility weights for frame pairs",
            scene=sequence.scene,
            level=level.name,
            approved_pairs=len(pair_pool.candidates),
        )
        with timing.measure("pair_feasibility_weights"):
            if task.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES:
                feasibility_weights = np.ones(
                    len(pair_pool.candidates), dtype=np.float64
                )
            else:
                feasibility_weights = np.asarray([
                    feasible_direction_probability(
                        float(np.linalg.norm((
                            candidate.source.world_from_pano[:3, 3]
                            - candidate.target.world_from_pano[:3, 3]
                        ))),
                        target_norm,
                        translation_limit,
                        scale_min,
                        scale_max,
                    )
                    for candidate in pair_pool.candidates
                ], dtype=np.float64)
        pair_sampler = FeasibilityConditionedPairSampler(
            sequence=sequence,
            candidates=pair_pool.candidates,
            feasibility_weights=feasibility_weights,
            raw_candidate_count=pair_pool.raw_candidate_count,
            seed=derived_seed,
            distribution=(
                _direct_pair_distribution(task.residual_sampler.mode)
                if task.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES
                else (
                    "overlap_approved_pairs_weighted_by_"
                    "feasible_so3_measure"
                )
            ),
        )
        emit_runtime_log(
            "stage_completed",
            "Frame-pair physical feasibility weights computed",
            scene=sequence.scene,
            level=level.name,
            usable_pairs=pair_sampler.candidate_count,
            elapsed_seconds=round(perf_counter() - feasibility_start, 3),
        )
    except BaseException:
        _restore_worker_runtime(previous_opencv_threads, numeric_thread_limits)
        raise
    if pair_sampler.candidate_count * task.max_accepted_per_pair < accepted_target:
        _restore_worker_runtime(previous_opencv_threads, numeric_thread_limits)
        raise StochasticGenerationError(
            "approved pair pool cannot satisfy the configured reuse cap",
            {
                "status": "insufficient_pair_pool_capacity",
                "sequence_id": sequence.sequence_id,
                "curriculum_level": level.name,
                "accepted_target": accepted_target,
                "maximum_capacity": (
                    pair_sampler.candidate_count * task.max_accepted_per_pair
                ),
                "pair_pool": pair_pool.report,
            },
        )
    try:
        coverage_config = load_coverage_config(task.coverage_config)
        joint_balancing_enabled = (
            task.accepted_distribution_control == JOINT_BALANCED
        )
        translation_radius_balancing_enabled = (
            task.accepted_distribution_control != REPORT_ONLY
        )
        extrinsic_sampler = ExtrinsicSampleGenerator(
            nominal=task.nominal.matrix,
            level=level,
            scale_min=scale_min,
            scale_max=scale_max,
            seed=derived_seed ^ 0xA5A5A5A5,
            residual_sampler=task.residual_sampler,
            accepted_distribution_control=(
                task.accepted_distribution_control
            ),
            view_rotation_distribution=task.view_rotation_distribution,
            max_level_rig_tilt_deg=task.max_level_rig_tilt_deg,
            camera_optical_axis_latitude_band_enabled=(
                task.camera_optical_axis_latitude_band_enabled
            ),
            max_camera_optical_axis_latitude_deg=(
                task.max_camera_optical_axis_latitude_deg
            ),
            joint_direction_count=(
                coverage_config.translation_direction_cells
                if joint_balancing_enabled else 0
            ),
        )
        sensor = load_training_sensor_config(task.sensor_config)
        paired_sensor = _paired_sensor_config(task, sensor)
        quality_gate = load_generated_quality_config(task.quality_config)
        translation_radius_quota = (
            EqualAcceptedTranslationRadiusQuota.create(
                accepted_target,
                coverage_config.translation_shell_edges,
            )
            if translation_radius_balancing_enabled else None
        )
        joint_cell_quota = (
            JointAcceptedCellQuota.create(
                accepted_target=accepted_target,
                edges=coverage_config.translation_shell_edges,
                direction_count=coverage_config.translation_direction_cells,
                max_consecutive_failures=(
                    task.joint_cell_max_consecutive_failures
                ),
                seed=derived_seed ^ 0x5A5A5A5A,
                radius_subinterval_count=(
                    task.joint_radius_subintervals_per_interval
                ),
                radius_partition_index=radius_partition_index,
                radius_partition_count=radius_partition_count,
            )
            if joint_balancing_enabled else None
        )
    except BaseException:
        _restore_worker_runtime(previous_opencv_threads, numeric_thread_limits)
        raise
    maximum_attempts = (
        None
        if task.max_attempts_per_accepted is None
        else accepted_target * task.max_attempts_per_accepted
    )

    max_consecutive_attempts_without_acceptance = (
        task.runtime.effective_max_consecutive_attempts_without_acceptance
    )
    attempt_reasons = Counter()
    accepted_pair_counts = Counter()
    reuse_counters = {name: Counter() for name in REUSE_AUDIT_METRICS}

    accepted_source_pairs = defaultdict(set)
    accepted_target_pairs = defaultdict(set)
    accepted_candidates = []
    true_transforms = []
    residuals = []
    sample_metadata = []
    paired_sample_metadata = []
    paired_quality_failures = Counter()
    attempt_count = 0
    consecutive_attempts_without_acceptance = 0
    termination_reason = "accepted_target_reached"
    current_residual_draw_count = None
    current_residual_reason_baseline = Counter()
    progress_path = stage / "progress.json"
    last_progress_write = 0.0
    incremental_writer = None
    paired_incremental_writer = None

    def current_residual_attempt_reasons() -> dict:
        """Return current residual attempt reasons."""
        return {
            reason: int(count - current_residual_reason_baseline[reason])
            for reason, count in sorted(attempt_reasons.items())
            if count > current_residual_reason_baseline[reason]
        }

    def write_sequence_progress(
        status: str = "in_progress", *, force: bool = False
    ) -> None:
        """Write sequence progress."""
        nonlocal last_progress_write
        now = perf_counter()
        if (
            not force
            and now - last_progress_write
            < task.runtime.progress_interval_seconds
        ):
            return
        residual_state = extrinsic_sampler.residual_distribution_report()
        acceptance_rate = (
            len(accepted_candidates) / attempt_count
            if attempt_count else 0.0
        )
        progress = {
            "version": SEQUENCE_PROGRESS_VERSION,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sequence_id": sequence.sequence_id,
            "scene": sequence.scene,
            "curriculum_level": level.name,
            "split": split,
            "accepted_count": len(accepted_candidates),
            "accepted_target": accepted_target,
            "attempt_count": attempt_count,
            "consecutive_attempts_without_acceptance": (
                consecutive_attempts_without_acceptance
            ),
            "max_consecutive_attempts_without_acceptance": (
                max_consecutive_attempts_without_acceptance
            ),
            "configured_max_consecutive_attempts_without_acceptance": (
                task.runtime.max_consecutive_attempts_without_acceptance
            ),
            "maximum_total_attempts": maximum_attempts,
            "max_attempts_per_accepted": task.max_attempts_per_accepted,
            "acceptance_rate": acceptance_rate,
            "attempt_reasons": dict(sorted(attempt_reasons.items())),
            "current_residual_attempt_reasons": (
                current_residual_attempt_reasons()
            ),
            "residual_distribution_preservation": residual_state,
            "stage_timing": timing.report(),
            "elapsed_seconds": now - total_start,
            "runtime": {
                "opencv_threads_per_worker": (
                    task.runtime.opencv_threads_per_worker
                ),
                "numeric_threads_per_worker": (
                    task.runtime.numeric_threads_per_worker
                ),
                "progress_interval_seconds": (
                    task.runtime.progress_interval_seconds
                ),
                "incremental_persistence": (
                    task.runtime.incremental_persistence
                ),
                "direct_output_persistence": (
                    task.runtime.direct_output_persistence
                ),
            },
        }
        _write_json_atomic(progress_path, progress)
        emit_runtime_log(
            "scene_progress",
            "Generating scene samples",
            scene=sequence.scene,
            split=split,
            level=level.name,
            status=status,
            accepted_count=len(accepted_candidates),
            accepted_target=accepted_target,
            attempt_count=attempt_count,
            acceptance_rate=round(acceptance_rate, 6),
            consecutive_failures=consecutive_attempts_without_acceptance,
            elapsed_seconds=round(now - total_start, 1),
        )
        last_progress_write = now

    def record_joint_rejection(joint_target) -> None:
        """Record joint rejection."""
        if joint_cell_quota is None:
            return
        if joint_target is None:
            raise RuntimeError("joint rejection has no target cell")
        released = joint_cell_quota.record_failure(joint_target.index)
        if released:
            extrinsic_sampler.discard_translation_target(
                joint_target.translation_interval_index,
                joint_target=joint_target,
            )
            attempt_reasons["joint_cell_released_after_failures"] += 1

    try:
        image_dir = stage / "image_2"
        cloud_dir = stage / "velodyne"
        image_dir.mkdir(parents=True)
        cloud_dir.mkdir()
        paired_image_dir = None
        paired_cloud_dir = None
        if paired_stage is not None:
            paired_image_dir = paired_stage / "image_2"
            paired_cloud_dir = paired_stage / "velodyne"
            paired_image_dir.mkdir(parents=True)
            paired_cloud_dir.mkdir()
        if task.runtime.incremental_persistence:
            incremental_writer = _IncrementalSequenceWriter(stage)
            if paired_stage is not None:
                paired_incremental_writer = _IncrementalSequenceWriter(
                    paired_stage
                )
        emit_runtime_log(
            "stage_started",
            "Starting sensor generation and quality filtering",
            scene=sequence.scene,
            split=split,
            level=level.name,
            accepted_target=accepted_target,
        )
        write_sequence_progress(force=True)
        while len(accepted_candidates) < accepted_target:
            write_sequence_progress()
            if _consecutive_attempt_limit_reached(
                consecutive_attempts_without_acceptance,
                max_consecutive_attempts_without_acceptance,
            ):
                attempt_reasons["consecutive_attempt_limit_reached"] += 1
                if not accepted_candidates:
                    write_sequence_progress(status="failed", force=True)
                    raise StochasticGenerationError(
                        "sequence stalled before accepting its first sample",
                        {
                            "status": "stalled_without_accepted_samples",
                            "sequence_id": sequence.sequence_id,
                            "curriculum_level": level.name,
                            "accepted_target": accepted_target,
                            "accepted_count": 0,
                            "attempt_count": attempt_count,
                            "consecutive_attempt_limit": (
                                max_consecutive_attempts_without_acceptance
                            ),
                            "attempt_reasons": dict(
                                sorted(attempt_reasons.items())
                            ),
                        },
                    )
                if task.accepted_distribution_control != REPORT_ONLY:
                    write_sequence_progress(status="failed", force=True)
                    raise StochasticGenerationError(
                        "partial stall finalization is incompatible with "
                        "accepted-sample quotas",
                        {
                            "status": "stall_with_incomplete_quotas",
                            "sequence_id": sequence.sequence_id,
                            "curriculum_level": level.name,
                            "accepted_target": accepted_target,
                            "accepted_count": len(accepted_candidates),
                            "attempt_count": attempt_count,
                            "consecutive_attempt_limit": (
                                max_consecutive_attempts_without_acceptance
                            ),
                            "attempt_reasons": dict(
                                sorted(attempt_reasons.items())
                            ),
                        },
                    )
                termination_reason = "partial_due_to_stall"
                break
            if (
                maximum_attempts is not None
                and attempt_count >= maximum_attempts
            ):
                write_sequence_progress(status="failed", force=True)
                report = {
                    "status": "insufficient_accepted_samples",
                    "sequence_id": sequence.sequence_id,
                    "curriculum_level": level.name,
                    "accepted_target": accepted_target,
                    "accepted_count": len(accepted_candidates),
                    "attempt_count": attempt_count,
                    "maximum_attempts": maximum_attempts,
                    "attempt_reasons": dict(sorted(attempt_reasons.items())),
                    "residual_distribution_preservation": (
                        extrinsic_sampler.residual_distribution_report()
                    ),
                    "translation_radius_balancing": (
                        translation_radius_quota.report()
                        if translation_radius_quota is not None else None
                    ),
                    "joint_cell_balancing": (
                        joint_cell_quota.report()
                        if joint_cell_quota is not None else None
                    ),
                    "pair_sampler": pair_sampler.report(),
                    "pair_pool": pair_pool.report,
                    "stage_timing": timing.report(),
                    "source_cache": {
                        "depth": depth_cache_report(),
                        "rgb": rgb_cache_report(),
                    },
                    "elapsed_seconds": perf_counter() - total_start,
                }
                raise StochasticGenerationError(
                    "stochastic loop exhausted its configured attempt budget",
                    report,
                )
            attempt_count += 1
            consecutive_attempts_without_acceptance += 1
            if task.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES:
                desired_true_translation_norm = (
                    extrinsic_sampler.pending_direct_true_translation_norm()
                )
                residual_state = (
                    extrinsic_sampler.residual_distribution_report()
                )
                residual_draw_count = int(
                    residual_state["residual_draw_count"]
                )
                if residual_draw_count != current_residual_draw_count:
                    current_residual_draw_count = residual_draw_count
                    current_residual_reason_baseline = Counter(attempt_reasons)
                capped_pair_ids = {
                    pair_id
                    for pair_id, count in accepted_pair_counts.items()
                    if count >= task.max_accepted_per_pair
                }
                with timing.measure("residual_conditioned_pair_selection"):
                    pair = pair_sampler.draw_scale_feasible(
                        desired_true_translation_norm,
                        scale_min,
                        scale_max,
                        excluded_pair_ids=capped_pair_ids,
                    )
                if pair is None:
                    if task.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                        attempt_reasons[
                            "direct_residual_without_scale_feasible_pair"
                        ] += 1
                        extrinsic_sampler.discard_pending_direct_residual()
                        attempt_reasons["direct_residual_resampled"] += 1
                        continue
                    report = {
                        "status": "fixed_extrinsic_has_no_scale_feasible_pair",
                        "sequence_id": sequence.sequence_id,
                        "curriculum_level": level.name,
                        "accepted_target": accepted_target,
                        "accepted_count": len(accepted_candidates),
                        "attempt_count": attempt_count,
                        "desired_true_translation_norm_m": (
                            desired_true_translation_norm
                        ),
                        "scale_range": [scale_min, scale_max],
                        "residual_distribution_preservation": (
                            extrinsic_sampler.residual_distribution_report()
                        ),
                        "pair_sampler": pair_sampler.report(),
                        "pair_pool": pair_pool.report,
                        "stage_timing": timing.report(),
                    }
                    raise StochasticGenerationError(
                        "fixed T0 has no scale-feasible approved frame pair",
                        report,
                    )
                attempt_reasons["residual_scale_conditioned_pair_draw"] += 1
            else:
                pair = pair_sampler.draw()
                if (
                    accepted_pair_counts[pair.pair_id]
                    >= task.max_accepted_per_pair
                ):
                    attempt_reasons["accepted_pair_reuse_cap"] += 1
                    continue
            joint_target = (
                joint_cell_quota.next_open_cell()
                if joint_cell_quota is not None else None
            )
            radius_interval = (
                joint_target.translation_interval_index
                if joint_target is not None
                else (
                    translation_radius_quota.next_open_interval()
                    if translation_radius_quota is not None
                    else -1
                )
            )
            radius_interval_bounds = (
                joint_cell_quota.fine_radius_bounds(
                    joint_target.translation_radius_cell_index
                )
                if joint_target is not None
                else (
                    translation_radius_quota.interval_bounds(radius_interval)
                    if translation_radius_quota is not None
                    else None
                )
            )
            rotation_interval_bounds = (
                joint_cell_quota.fine_radius_bounds(
                    joint_target.rotation_radius_cell_index
                )
                if joint_target is not None else None
            )
            with timing.measure("conditional_extrinsic_proposal"):
                candidate, feasibility = extrinsic_sampler.propose(
                    pair.source,
                    pair.target,
                    pair.pair_id,
                    radius_interval_index=radius_interval,
                    radius_interval=radius_interval_bounds,
                    radius_interval_target=(
                        translation_radius_quota.targets[radius_interval]
                        if translation_radius_quota is not None else 0
                    ),
                    accepted_in_interval=(
                        translation_radius_quota.accepted[radius_interval]
                        if translation_radius_quota is not None else 0
                    ),
                    joint_target=joint_target,
                    rotation_interval=rotation_interval_bounds,
                )
            if translation_radius_quota is not None:
                attempt_reasons[
                    f"translation_radius_interval_{radius_interval:02d}_targeted"
                ] += 1
            else:
                attempt_reasons["random_residual_targeted"] += 1
            if joint_target is not None:
                attempt_reasons["joint_cell_targeted"] += 1
            if candidate is None:
                attempt_reasons[str(feasibility["reason"])] += 1
                record_joint_rejection(joint_target)
                if extrinsic_sampler.record_candidate_rejected():
                    attempt_reasons["direct_residual_attempt_budget_exhausted"] += 1
                continue
            candidate.update({
                "sequence_id": sequence.sequence_id,
                "scene": sequence.scene,
                "split": split,
                "sensor_sampling_seed": int(derived_seed),
            })
            if task.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                cell_indices = _direct_box_cell_indices(
                    np.asarray(candidate["direct_box_normalized"], dtype=np.float64),
                    int(coverage_config.bins_per_axis),
                )
                candidate["direct_box_cell_id"] = _direct_box_cell_id(
                    cell_indices
                )
            if translation_radius_quota is not None:
                realized_radius_interval = (
                    translation_radius_quota.interval_index(
                        candidate["normalized"][:3]
                    )
                )
                if realized_radius_interval != radius_interval:
                    raise RuntimeError(
                        "targeted translation realized in a different "
                        "radius interval"
                    )
            if (
                joint_target is not None
                and joint_cell_quota.cell_index(candidate["normalized"])
                != joint_target.index
            ):
                raise RuntimeError(
                    "targeted proposal realized in a different joint cell"
                )
            if joint_target is not None and (
                joint_cell_quota.fine_radius_cell_indices(
                    candidate["normalized"]
                )
                != (
                    joint_target.translation_radius_cell_index,
                    joint_target.rotation_radius_cell_index,
                )
            ):
                raise RuntimeError(
                    "targeted proposal realized in a different fine radius cell"
                )
            with timing.measure("camera_depth_generation"):
                camera_stage = generate_candidate_camera_depth(
                    pair.source,
                    pair.target,
                    sequence.pano_from_face,
                    candidate,
                    sensor,
                )
            with timing.measure("lidar_generation"):
                lidar_stage = generate_candidate_lidar(
                    pair.source,
                    pair.target,
                    sequence.pano_from_face,
                    candidate,
                    sensor,
                )
            with timing.measure("projection_quality"):
                projection_report, reasons = (
                    evaluate_generated_projection_quality(
                        lidar_stage.point_cloud,
                        camera_stage.camera_depth,
                        camera_stage.camera_depth_unscaled,
                        candidate,
                        task.nominal.matrix,
                        sensor,
                        quality_gate,
                        check_nominal=(
                            task.residual_sampler.mode != FIXED_IDENTITY
                        ),
                        stop_on_rejection=True,
                    )
                )
            quality = combine_generated_quality_reports(
                projection_report, reasons
            )
            if reasons:
                attempt_reasons["generated_quality_rejected"] += 1
                attempt_reasons["projection_quality_rejected"] += 1
                for reason in reasons:
                    attempt_reasons[f"quality:{reason}"] += 1
                record_joint_rejection(joint_target)
                if extrinsic_sampler.record_candidate_rejected():
                    attempt_reasons["direct_residual_attempt_budget_exhausted"] += 1
                continue

            paired_lidar_stage = None
            paired_quality = None
            paired_reasons: list[str] = []
            if paired_sensor is not None:
                with timing.measure("paired_wide_lidar_generation"):
                    paired_lidar_stage = generate_candidate_lidar(
                        pair.source,
                        pair.target,
                        sequence.pano_from_face,
                        candidate,
                        paired_sensor,
                    )
                with timing.measure("paired_wide_projection_quality"):
                    paired_projection_report, paired_reasons = (
                        evaluate_generated_projection_quality(
                            paired_lidar_stage.point_cloud,
                            camera_stage.camera_depth,
                            camera_stage.camera_depth_unscaled,
                            candidate,
                            task.nominal.matrix,
                            paired_sensor,
                            quality_gate,
                            check_nominal=(
                                task.residual_sampler.mode != FIXED_IDENTITY
                            ),
                            stop_on_rejection=False,
                        )
                    )
                paired_quality = combine_generated_quality_reports(
                    paired_projection_report,
                    paired_reasons,
                )

            with timing.measure("rgb_generation"):
                generated = finalize_sensor_sample(
                    pair.source,
                    pair.target,
                    sequence.pano_from_face,
                    candidate,
                    sensor,
                    camera_stage,
                    lidar_stage,
                )
            frame_id = f"{len(accepted_candidates):06d}"
            true = validate_transform(np.asarray(
                candidate["true_extrinsic"], dtype=np.float64
            ))
            if (
                task.residual_sampler.mode == FIXED_IDENTITY
                and not np.allclose(
                    true, task.nominal.matrix, rtol=0.0, atol=1e-10
                )
            ):
                raise RuntimeError(
                    "fixed identity sampler produced Ti different from T0"
                )
            residual = residual_transform(true, task.nominal.matrix)
            sample = _sample_metadata_record(
                frame_id=frame_id,
                sequence_id=sequence.sequence_id,
                scene=sequence.scene,
                level_name=level.name,
                candidate=candidate,
                residual=residual,
                generated=generated,
                quality=quality,
            )
            paired_sample = None
            paired_generated = None
            if paired_lidar_stage is not None:
                paired_generated = replace(
                    generated,
                    point_cloud=paired_lidar_stage.point_cloud,
                    ring_ids=paired_lidar_stage.ring_ids,
                    report={
                        **generated.report,
                        "sensor_profile": paired_sensor.profile,
                        "point_cloud_shape": list(
                            paired_lidar_stage.point_cloud.shape
                        ),
                        "lidar": paired_lidar_stage.report,
                        "paired_lidar_derivative": {
                            **task.paired_lidar_export.to_metadata(),
                            "source_primary_frame_id": frame_id,
                            "image_rendered_once": True,
                        },
                    },
                )
                paired_sample = _paired_sample_metadata_record(
                    sample,
                    paired_generated,
                    paired_quality,
                    paired_reasons,
                )
                for reason in paired_reasons:
                    paired_quality_failures[reason] += 1
            image_path = image_dir / f"{frame_id}.png"
            cloud_path = cloud_dir / f"{frame_id}.bin"
            temporary_image_path = image_dir / f".{frame_id}.tmp.png"
            temporary_cloud_path = cloud_dir / f".{frame_id}.tmp.bin"
            paired_image_path = (
                paired_image_dir / f"{frame_id}.png"
                if paired_image_dir is not None else None
            )
            paired_cloud_path = (
                paired_cloud_dir / f"{frame_id}.bin"
                if paired_cloud_dir is not None else None
            )
            with timing.measure("accepted_serialization"):
                if not cv2.imwrite(
                    str(temporary_image_path), generated.image_bgr
                ):
                    raise OSError(
                        f"failed to write image: {temporary_image_path}"
                    )
                generated.point_cloud.tofile(temporary_cloud_path)
                temporary_image_path.replace(image_path)
                temporary_cloud_path.replace(cloud_path)
                if paired_generated is not None:
                    paired_temporary_image_path = (
                        paired_image_dir / f".{frame_id}.tmp.png"
                    )
                    paired_temporary_cloud_path = (
                        paired_cloud_dir / f".{frame_id}.tmp.bin"
                    )
                    shutil.copy2(image_path, paired_temporary_image_path)
                    paired_generated.point_cloud.tofile(
                        paired_temporary_cloud_path
                    )
                    paired_temporary_image_path.replace(paired_image_path)
                    paired_temporary_cloud_path.replace(paired_cloud_path)
                if incremental_writer is not None:
                    incremental_writer.append(sample, true, residual)
                    if paired_incremental_writer is not None:
                        paired_incremental_writer.append(
                            paired_sample, true, residual
                        )
            if translation_radius_quota is not None:
                translation_radius_quota.record_acceptance(radius_interval)
                if joint_cell_quota is not None:
                    joint_cell_quota.record_acceptance(joint_target.index)
            extrinsic_sampler.record_candidate_accepted(
                radius_interval,
                joint_target=joint_target,
            )
            coverage_candidate = {
                "normalized": candidate["normalized"],
                "pair_id": candidate["pair_id"],
            }
            if task.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                coverage_candidate.update({
                    "residual_identity": (
                        f"{sequence.sequence_id}|"
                        f"{candidate['residual_draw_index']}"
                    ),
                    "direct_box_normalized": candidate[
                        "direct_box_normalized"
                    ],
                    "direct_box_cell_id": candidate["direct_box_cell_id"],
                })
            accepted_candidates.append(coverage_candidate)
            if incremental_writer is None:
                true_transforms.append(true)
                residuals.append(residual)
                sample_metadata.append(sample)
                if paired_sample is not None:
                    paired_sample_metadata.append(paired_sample)
            accepted_pair_counts[pair.pair_id] += 1
            sequence_key = str(sequence.sequence_id)
            reuse_counters["pair"][
                f"{sequence_key}|{pair.pair_id}"
            ] += 1
            reuse_counters["source_frame"][
                f"{sequence_key}|{candidate['source_frame']}"
            ] += 1
            reuse_counters["target_frame"][
                f"{sequence_key}|{candidate['target_frame']}"
            ] += 1
            reuse_counters["target_view"][_target_view_key(
                sequence_key,
                str(candidate["target_frame"]),
                candidate["pano_from_camera_rotation"],
            )] += 1
            pair_key = f"{sequence_key}|{pair.pair_id}"
            source_key = f"{sequence_key}|{candidate['source_frame']}"
            target_key = f"{sequence_key}|{candidate['target_frame']}"
            if pair_key not in accepted_source_pairs[source_key]:
                accepted_source_pairs[source_key].add(pair_key)
                reuse_counters["source_frame_unique_pairs"][source_key] += 1
            if pair_key not in accepted_target_pairs[target_key]:
                accepted_target_pairs[target_key].add(pair_key)
                reuse_counters["target_frame_unique_pairs"][target_key] += 1
            attempt_reasons["accepted"] += 1
            consecutive_attempts_without_acceptance = 0
            write_sequence_progress()

        emit_runtime_log(
            "stage_completed",
            "Sensor generation and quality filtering completed",
            scene=sequence.scene,
            split=split,
            level=level.name,
            accepted_count=len(accepted_candidates),
            attempt_count=attempt_count,
            acceptance_rate=round(
                len(accepted_candidates) / attempt_count, 6
            ),
            elapsed_seconds=round(perf_counter() - total_start, 3),
        )
        partial_sequence = len(accepted_candidates) < accepted_target
        write_sequence_progress(status="finalizing", force=True)
        incomplete_quotas = []
        if (
            translation_radius_quota is not None
            and not translation_radius_quota.complete
        ):
            incomplete_quotas.append("translation_radius")
        if joint_cell_quota is not None and not joint_cell_quota.complete:
            incomplete_quotas.append("joint_cell")
        if incomplete_quotas:
            raise StochasticGenerationError(
                "accepted sequence ended with incomplete strict quotas",
                {
                    "status": "incomplete_accepted_quotas",
                    "sequence_id": sequence.sequence_id,
                    "curriculum_level": level.name,
                    "accepted_target": accepted_target,
                    "accepted_count": len(accepted_candidates),
                    "incomplete_quotas": incomplete_quotas,
                    "translation_radius_balancing": (
                        translation_radius_quota.report()
                        if translation_radius_quota is not None else None
                    ),
                    "joint_cell_balancing": (
                        joint_cell_quota.report()
                        if joint_cell_quota is not None else None
                    ),
                    "attempt_reasons": dict(sorted(attempt_reasons.items())),
                },
            )
        extrinsic_sampler.finalize_direct_residual_stream()
        residual_distribution = (
            extrinsic_sampler.residual_distribution_report()
        )
        if residual_distribution["enabled"] and not partial_sequence:
            if task.residual_sampler.mode == FIXED_IDENTITY:
                invalid_distribution = (
                    residual_distribution["rejection_policy"]
                    != RETRY_SAME_RESIDUAL
                    or residual_distribution["accepted_residual_count"]
                    != accepted_target
                    or residual_distribution["residual_draw_count"]
                    != accepted_target
                    or not residual_distribution[
                        "no_sampled_residual_skipped"
                    ]
                )
            else:
                histogram = residual_distribution[
                    "accepted_samples_per_residual_histogram"
                ]
                attempt_histogram = residual_distribution[
                    "realization_attempts_per_residual_histogram"
                ]
                max_residual_attempts = (
                    task.residual_sampler.max_realization_attempts_per_residual
                )
                histogram_samples = sum(
                    int(count) * int(samples)
                    for samples, count in histogram.items()
                )
                invalid_distribution = (
                    residual_distribution["accepted_sample_count"]
                    != accepted_target
                    or histogram_samples != accepted_target
                    or any(
                        int(samples)
                        > task.residual_sampler.max_samples_per_residual
                        for samples in histogram
                    )
                    or (
                        max_residual_attempts
                        is not None
                        and any(
                            int(attempts)
                            > max_residual_attempts
                            for attempts in attempt_histogram
                        )
                    )
                )
            invalid_distribution = invalid_distribution or (
                residual_distribution["pending_residual"]
                or not residual_distribution[
                    "all_drawn_residuals_accounted_for"
                ]
            )
            if invalid_distribution:
                raise RuntimeError(
                    "direct residual stream accounting is inconsistent"
                )
        coverage = audit_stochastic_coverage(
            accepted_candidates,
            coverage_config,
            proposed_direct_box=(
                extrinsic_sampler.direct_box_proposals()
                if task.residual_sampler.mode == DIRECT_LEFT_EULER_BOX
                else ()
            ),
        )
        if task.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
            direct_counts = coverage
            if (
                direct_counts["proposed_residuals"]["sample_count"]
                != residual_distribution["residual_draw_count"]
                or direct_counts["accepted_unique_residuals"]["sample_count"]
                != residual_distribution["accepted_residual_count"]
                or direct_counts["accepted_samples"]["sample_count"]
                != len(accepted_candidates)
            ):
                raise RuntimeError(
                    "Direct-Euler coverage layers disagree with residual accounting"
                )

        _write_calib(stage / "calib.txt", sensor.camera.matrix)
        if paired_stage is not None:
            _write_calib(
                paired_stage / "calib.txt", paired_sensor.camera.matrix
            )
        if incremental_writer is not None:
            incremental_writer.promote_transform_rows()
            if paired_incremental_writer is not None:
                paired_incremental_writer.promote_transform_rows()
        else:
            _write_transform_rows(stage / "poses.txt", true_transforms)
            _write_transform_rows(stage / "residuals.txt", residuals)
            if paired_stage is not None:
                _write_transform_rows(
                    paired_stage / "poses.txt", true_transforms
                )
                _write_transform_rows(
                    paired_stage / "residuals.txt", residuals
                )
        np.savetxt(
            str(stage / "nominal_extrinsic.txt"),
            task.nominal.matrix,
            fmt="%.18e",
        )
        if paired_stage is not None:
            np.savetxt(
                str(paired_stage / "nominal_extrinsic.txt"),
                task.nominal.matrix,
                fmt="%.18e",
            )
        metadata = {
            "version": STOCHASTIC_DATASET_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_semantics": "nominal_to_true_residual_v1",
            "current_cmrnext_loader_compatible": False,
            "dataset_role": "primary_quality_gated",
            "target_network": task.target_network,
            "curriculum_level": level.name,
            "split": split,
            "n_samples": len(accepted_candidates),
            "requested_accepted_sample_target": accepted_target,
            "partial_sequence": partial_sequence,
            "termination_reason": termination_reason,
            "smoke_limited": smoke_limited,
            "formal_continuous_coverage_passed": (
                False if smoke_limited or partial_sequence else None
            ),
            "formal_coverage_authority": (
                "task_manifest.json level_coverage overall and split reports"
            ),
            "coverage_scope": task.coverage_scope,
            "accepted_distribution_control": (
                task.accepted_distribution_control
            ),
            "residual_distribution_preservation": residual_distribution,
            "coverage_enforced": (
                task.accepted_distribution_control != REPORT_ONLY
            ),
            "sequence_continuous_coverage_passed": coverage["passed"],
            "coordinate_convention": {
                "extrinsic": "camera_from_lidar column-vector",
                "residual": "Ti @ inverse(T0)",
            },
            "task": task.to_metadata(),
            "scene_scale_range": {
                "minimum": scale_min,
                "maximum": scale_max,
                "distribution": task.scale_distribution,
            },
            "level": {
                "name": level.name,
                **level.limit_metadata,
                "effective_max_translation_m": (
                    level.effective_max_translation(task.nominal.matrix)
                ),
            },
            "source_manifest": str(Path(source_manifest).resolve()),
            "source": source_metadata,
            "sensor_setup": sensor.to_dict(),
            "quality_config": quality_gate.to_dict(),
            "coverage_config": coverage_config.to_dict(),
            "accepted_sample_target": len(accepted_candidates),
            "attempt_count": attempt_count,
            "acceptance_rate": len(accepted_candidates) / attempt_count,
            "attempt_reasons": dict(sorted(attempt_reasons.items())),
            "translation_radius_balancing": (
                translation_radius_quota.report()
                if translation_radius_quota is not None else None
            ),
            "joint_cell_balancing": (
                joint_cell_quota.report()
                if joint_cell_quota is not None else None
            ),
            "accepted_pair_counts": dict(sorted(accepted_pair_counts.items())),
            "reuse_audit": _reuse_audit_from_counters(reuse_counters),
            "pair_sampler": pair_sampler.report(),
            "pair_pool": pair_pool.report,
            "true_extrinsic_policy": (
                "Ti_equals_T0"
                if task.residual_sampler.mode == FIXED_IDENTITY else "varied"
            ),
            "sampling_optimization": {
                "pair_distribution": (
                    _direct_pair_distribution(task.residual_sampler.mode)
                    if task.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES
                    else "overlap_approved_pairs_weighted_by_"
                    "feasible_so3_measure"
                ),
                "view_distribution": (
                    task.view_rotation_distribution
                ),
                "acceptance_semantics_changed": True,
                "accepted_distribution_control": (
                    task.accepted_distribution_control
                ),
                "acceptance_semantics_change": (
                    "true extrinsic is fixed to T0"
                    if task.residual_sampler.mode == FIXED_IDENTITY
                    else (
                        "random residual proposals are conditioned by physical "
                        "feasibility and quality gates"
                        if task.accepted_distribution_control == REPORT_ONLY
                        else "accepted samples are quota-balanced across "
                        "normalized translation/rotation cells after filtering"
                    )
                ),
                "accepted_translation_distribution": (
                    task.translation_distribution
                ),
                "translation_interval_sampling": (
                    task.translation_interval_sampling
                ),
                "accepted_rotation_distribution": (
                    task.rotation_distribution
                ),
                "joint_cell_sampling": task.joint_cell_sampling,
                "joint_cell_balancing_enabled": joint_balancing_enabled,
                "joint_radius_subintervals_per_interval": (
                    task.joint_radius_subintervals_per_interval
                ),
            },
            "runtime": {
                "opencv_threads_per_worker": (
                    task.runtime.opencv_threads_per_worker
                ),
                "numeric_threads_per_worker": (
                    task.runtime.numeric_threads_per_worker
                ),
                "progress_interval_seconds": (
                    task.runtime.progress_interval_seconds
                ),
                "max_consecutive_attempts_without_acceptance": (
                    task.runtime.max_consecutive_attempts_without_acceptance
                ),
                "incremental_persistence": (
                    task.runtime.incremental_persistence
                ),
                "direct_output_persistence": (
                    task.runtime.direct_output_persistence
                ),
                "stage_timing": timing.report(),
                "elapsed_seconds": perf_counter() - total_start,
                "source_cache": {
                    "depth": depth_cache_report(),
                    "rgb": rgb_cache_report(),
                },
            },
            "continuous_coverage": coverage,
        }
        paired_metadata = None
        if paired_stage is not None:
            paired_metadata = {
                **metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_role": "paired_lidar_fov_derivative",
                "source_manifest": "../../../_source/source_manifest.json",
                "sensor_setup": paired_sensor.to_dict(),
                "paired_lidar_derivative": {
                    **task.paired_lidar_export.to_metadata(),
                    "source_primary_dataset_role": "primary_quality_gated",
                    "source_primary_sequence_identity": (
                        f"{level.name}/{split}/{sequence.sequence_id}"
                    ),
                    "identical_fields": [
                        "frame_id",
                        "image",
                        "source_frame",
                        "target_frame",
                        "pair_id",
                        "true_extrinsic",
                        "residual",
                        "sample_order",
                    ],
                    "derived_quality_affects_primary_acceptance": False,
                },
                "derived_quality_summary": {
                    "passed_count": (
                        len(accepted_candidates)
                        - sum(paired_quality_failures.values())
                    ),
                    "failed_sample_count": sum(
                        1
                        for sample in (
                            paired_sample_metadata
                            if incremental_writer is None else []
                        )
                        if not sample["generated_quality"]["passed"]
                    ) if incremental_writer is None else None,
                    "rejection_reason_counts": dict(
                        sorted(paired_quality_failures.items())
                    ),
                },
            }
        if incremental_writer is None:
            metadata["samples"] = sample_metadata
            _write_json_atomic(stage / "metadata.json", metadata)
            if paired_metadata is not None:
                paired_metadata["samples"] = paired_sample_metadata
                paired_metadata["derived_quality_summary"][
                    "failed_sample_count"
                ] = sum(
                    not sample["generated_quality"]["passed"]
                    for sample in paired_sample_metadata
                )
                paired_metadata["derived_quality_summary"][
                    "passed_count"
                ] = (
                    len(paired_sample_metadata)
                    - paired_metadata["derived_quality_summary"][
                        "failed_sample_count"
                    ]
                )
                _write_json_atomic(
                    paired_stage / "metadata.json", paired_metadata
                )
        else:
            _write_metadata_with_sample_journal(
                stage / "metadata.json",
                metadata,
                incremental_writer.samples_path,
            )
            if paired_metadata is not None:
                retained_failure_count = sum(
                    paired_quality_failures.values()
                )

                failed_sample_count = 0
                with paired_incremental_writer.samples_path.open(
                    "r", encoding="utf-8"
                ) as samples_handle:
                    for line in samples_handle:
                        value = json.loads(line)
                        if not value["generated_quality"]["passed"]:
                            failed_sample_count += 1
                paired_metadata["derived_quality_summary"].update({
                    "passed_count": (
                        len(accepted_candidates) - failed_sample_count
                    ),
                    "failed_sample_count": failed_sample_count,
                    "total_rejection_reason_occurrences": (
                        retained_failure_count
                    ),
                })
                _write_metadata_with_sample_journal(
                    paired_stage / "metadata.json",
                    paired_metadata,
                    paired_incremental_writer.samples_path,
                )
        progress_path.unlink()
        if skip_independent_audit:
            audit = {
                "valid": None,
                "skipped": True,
                "reason": "explicit_skip_independent_audit",
            }
        else:
            audit_start = perf_counter()
            emit_runtime_log(
                "stage_started",
                "Running the independent sequence integrity audit",
                scene=sequence.scene,
                split=split,
                level=level.name,
            )
            with timing.measure("independent_export_audit"):
                audit = audit_exported_sequence(stage, task.sensor_config)
                paired_audit = (
                    audit_exported_sequence(paired_stage, paired_sensor)
                    if paired_stage is not None else None
                )
                paired_identity_audit = (
                    audit_paired_sequence_identity(stage, paired_stage)
                    if paired_stage is not None else None
                )
            emit_runtime_log(
                "stage_completed",
                "Independent sequence integrity audit completed",
                scene=sequence.scene,
                split=split,
                level=level.name,
                primary_valid=audit["valid"],
                paired_valid=(
                    paired_audit["valid"]
                    if paired_audit is not None else None
                ),
                elapsed_seconds=round(perf_counter() - audit_start, 3),
            )
        if skip_independent_audit:
            paired_audit = (
                {
                    "valid": None,
                    "skipped": True,
                    "reason": "explicit_skip_independent_audit",
                }
                if paired_stage is not None else None
            )
            paired_identity_audit = (
                {
                    "valid": None,
                    "skipped": True,
                    "reason": "explicit_skip_independent_audit",
                }
                if paired_stage is not None else None
            )
        timing.add("total", perf_counter() - total_start)
        metadata["runtime"] = {
            "opencv_threads_per_worker": task.runtime.opencv_threads_per_worker,
            "numeric_threads_per_worker": (
                task.runtime.numeric_threads_per_worker
            ),
            "progress_interval_seconds": (
                task.runtime.progress_interval_seconds
            ),
            "max_consecutive_attempts_without_acceptance": (
                max_consecutive_attempts_without_acceptance
            ),
            "configured_max_consecutive_attempts_without_acceptance": (
                task.runtime.max_consecutive_attempts_without_acceptance
            ),
            "incremental_persistence": (
                task.runtime.incremental_persistence
            ),
            "direct_output_persistence": (
                task.runtime.direct_output_persistence
            ),
            "stage_timing": timing.report(),
            "elapsed_seconds": perf_counter() - total_start,
            "source_cache": {
                "depth": depth_cache_report(),
                "rgb": rgb_cache_report(),
            },
        }
        if incremental_writer is None:
            _write_json_atomic(stage / "metadata.json", metadata)
            if paired_metadata is not None:
                paired_metadata["runtime"] = metadata["runtime"]
                _write_json_atomic(
                    paired_stage / "metadata.json", paired_metadata
                )
        else:
            _write_metadata_with_sample_journal(
                stage / "metadata.json",
                metadata,
                incremental_writer.samples_path,
            )
            if paired_metadata is not None:
                paired_metadata["runtime"] = metadata["runtime"]
                _write_metadata_with_sample_journal(
                    paired_stage / "metadata.json",
                    paired_metadata,
                    paired_incremental_writer.samples_path,
                )
        manifest = {
            "version": STOCHASTIC_DATASET_VERSION,
            "target_network": task.target_network,
            "curriculum_level": level.name,
            "split": split,
            "sequence": ".",
            "sample_count": len(accepted_candidates),
            "smoke_limited": smoke_limited,
            "requested_accepted_sample_target": accepted_target,
            "partial_sequence": partial_sequence,
            "termination_reason": termination_reason,
            "sequence_continuous_coverage_passed": coverage["passed"],
            "formal_continuous_coverage_passed": (
                False if smoke_limited or partial_sequence else None
            ),
            "audit": audit,
        }
        with (stage / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if paired_stage is not None:
            paired_manifest = {
                **manifest,
                "dataset_role": "paired_lidar_fov_derivative",
                "paired_lidar_export": task.paired_lidar_export.to_metadata(),
                "audit": paired_audit,
                "paired_identity_audit": paired_identity_audit,
            }
            with (paired_stage / "manifest.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    paired_manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
        if incremental_writer is not None:
            incremental_writer.remove_sample_journal()
            if paired_incremental_writer is not None:
                paired_incremental_writer.remove_sample_journal()
        commit_start = perf_counter()
        emit_runtime_log(
            "stage_started",
            "Transactionally committing the final sequence directory",
            scene=sequence.scene,
            split=split,
            level=level.name,
            output=str(output),
        )
        if paired_stage is not None:
            paired_stage.replace(resolved_paired_output)
        if stage != output:
            stage.replace(output)
        emit_runtime_log(
            "stage_completed",
            "Final sequence directory committed",
            scene=sequence.scene,
            split=split,
            level=level.name,
            output=str(output),
            paired_output=(
                str(resolved_paired_output)
                if resolved_paired_output is not None else None
            ),
            elapsed_seconds=round(perf_counter() - commit_start, 3),
        )
    except BaseException:
        if (
            resolved_paired_output is not None
            and resolved_paired_output.exists()
            and not output.exists()
            and paired_stage is not None
            and not paired_stage.exists()
        ):
            resolved_paired_output.replace(paired_stage)
        if incremental_writer is not None:
            incremental_writer.close()
        if paired_incremental_writer is not None:
            paired_incremental_writer.close()
        if stage.exists() and not task.runtime.incremental_persistence:
            shutil.rmtree(stage)
        if (
            paired_stage is not None
            and paired_stage.exists()
            and not task.runtime.incremental_persistence
        ):
            shutil.rmtree(paired_stage)
        raise
    finally:
        _restore_worker_runtime(previous_opencv_threads, numeric_thread_limits)

    emit_runtime_log(
        "scene_completed",
        "Scene sequence generation completed",
        scene=sequence.scene,
        split=split,
        level=level.name,
        accepted_count=len(accepted_candidates),
        attempt_count=attempt_count,
        acceptance_rate=round(
            len(accepted_candidates) / attempt_count, 6
        ),
        elapsed_seconds=round(perf_counter() - total_start, 3),
    )
    return {
        "status": (
            "partial_due_to_stall" if partial_sequence else "complete"
        ),
        "output": str(output),
        "target_network": task.target_network,
        "curriculum_level": level.name,
        "split": split,
        "scene": sequence.scene,
        "sample_count": len(accepted_candidates),
        "attempt_count": attempt_count,
        "acceptance_rate": len(accepted_candidates) / attempt_count,
        "unique_accepted_pairs": len(accepted_pair_counts),
        "maximum_pair_reuse": max(accepted_pair_counts.values()),
        "reuse_audit": _reuse_audit_from_counters(reuse_counters),
        "sequence_continuous_coverage_passed": coverage["passed"],
        "continuous_coverage": coverage,
        "formal_continuous_coverage_passed": (
            False if smoke_limited or partial_sequence else None
        ),
        "partial_sequence": partial_sequence,
        "smoke_limited": smoke_limited,
        "translation_radius_balancing": (
            translation_radius_quota.report()
            if translation_radius_quota is not None else None
        ),
        "joint_cell_balancing": (
            joint_cell_quota.report()
            if joint_cell_quota is not None else None
        ),
        "pair_pool": pair_pool.report,
        "stage_timing": timing.report(),
        "audit": audit,
        "paired_output": (
            str(resolved_paired_output)
            if resolved_paired_output is not None else None
        ),
        "paired_audit": paired_audit,
        "paired_identity_audit": paired_identity_audit,
    }


def filter_and_export_dataset(
    task: GenerationTask,
    source_manifest: Union[str, Path],
    output_root: Union[str, Path],
    scenes: Optional[Iterable[str]] = None,
    levels: Optional[Iterable[str]] = None,
    accepted_target_override: Optional[int] = None,
    allow_incomplete_smoke: bool = False,
    resume_existing: bool = False,
    continue_on_coverage_failure: bool = False,
    workers: int = 1,
    cache_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Filter and export dataset."""


    return export_generation_task(
        task=task,
        source_manifest=source_manifest,
        output_root=output_root,
        scenes=scenes,
        levels=levels,
        accepted_target_override=accepted_target_override,
        allow_incomplete_smoke=allow_incomplete_smoke,
        resume_existing=resume_existing,
        continue_on_coverage_failure=continue_on_coverage_failure,
        workers=workers,
        cache_root=cache_root,
    )


__all__ = [
    "derive_lidar_modality_dataset",
    "StochasticGenerationError",
    "filter_and_export_dataset",
    "filter_and_export_sequence",
]


import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Tuple, Union

from ._runtime_log import emit_runtime_log
from .pair_pool import SPLIT_NAMES
from .sensors import (
    load_lidar_modality_derivation_config,
    load_training_sensor_config,
)
from .generation_config import GenerationTask, REPORT_ONLY


STOCHASTIC_TASK_EXPORT_VERSION = "stochastic_task_export_v6_statistics"


LEVEL_STATISTICS_VERSION = "level_statistics_v2"


REUSE_AUDIT_TITLES = {
    "pair": "Directed pair sample reuse",
    "source_frame": "Source frame sample reuse",
    "target_frame": "Target frame sample reuse",
    "target_view": "Target view sample reuse",
    "source_frame_unique_pairs": "Unique pairs per source frame",
    "target_frame_unique_pairs": "Unique pairs per target frame",
}


def _matplotlib_pyplot():
    """Return matplotlib pyplot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _write_statistics_figure(fig, path: Path) -> None:
    """Write statistics figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=120,
            bbox_inches="tight",
            metadata={"Software": "extrinsic_manifold_data"},
        )
        image = cv2.imread(str(temporary), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 32:
            raise OSError(f"statistics PNG cannot be decoded: {temporary}")
        if float(np.std(image)) <= 0.0:
            raise ValueError(f"statistics PNG is blank: {temporary}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_level_statistics_samples(
    level: str,
    results: Iterable[dict],
) -> dict[str, dict]:
    """Load level statistics samples."""
    scopes = {
        "overall": {"samples": [], "scenes": set(), "sequences": set()}
    }
    for result in results:
        root = Path(result["output"])
        with (root / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("curriculum_level") != level:
            raise ValueError(f"{root}: statistics level mismatch")
        split = str(metadata.get("split", ""))
        if split not in SPLIT_NAMES:
            raise ValueError(f"{root}: statistics split is invalid: {split!r}")
        samples = metadata.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{root}: statistics require nonempty samples")
        scope = scopes.setdefault(
            split, {"samples": [], "scenes": set(), "sequences": set()}
        )
        for target in (scopes["overall"], scope):
            target["samples"].extend(samples)
            target["scenes"].add(str(samples[0]["scene"]))
            target["sequences"].add(str(samples[0]["sequence_id"]))
    ordered = {}
    for name in ("overall", *SPLIT_NAMES):
        if name not in scopes:
            continue
        value = scopes[name]
        ordered[name] = {
            "samples": value["samples"],
            "scenes": sorted(value["scenes"]),
            "sequences": sorted(value["sequences"]),
        }
    return ordered


def _residual_arrays(samples: Iterable[dict]) -> dict[str, np.ndarray]:
    """Return residual arrays."""
    translations = []
    rotvecs = []
    normalized = []
    for sample in samples:
        residual = validate_transform(np.asarray(sample["residual"], dtype=np.float64))
        coordinate = np.asarray(sample["normalized"], dtype=np.float64)
        if coordinate.shape != (6,) or not np.all(np.isfinite(coordinate)):
            raise ValueError("statistics normalized residual must be finite length six")
        translations.append(residual[:3, 3])
        rotvecs.append(np.rad2deg(
            Rotation.from_matrix(residual[:3, :3]).as_rotvec()
        ))
        normalized.append(coordinate)
    return {
        "translation_m": np.asarray(translations, dtype=np.float64),
        "rotvec_deg": np.asarray(rotvecs, dtype=np.float64),
        "normalized": np.asarray(normalized, dtype=np.float64),
    }


def _histogram_2d_report(
    values: np.ndarray,
    left: int,
    right: int,
    bins: int,
    limits: tuple[float, float],
) -> dict:
    """Return histogram 2D report."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("2D histogram values must be finite Nx3")
    if bins < 2 or not limits[0] < limits[1]:
        raise ValueError("2D histogram bins or limits are invalid")
    counts, x_edges, y_edges = np.histogram2d(
        array[:, left],
        array[:, right],
        bins=bins,
        range=[limits, limits],
    )
    return {
        "left_axis": int(left),
        "right_axis": int(right),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "counts": counts.astype(np.int64).tolist(),
    }


def _spatial_projection_reports(
    values: np.ndarray,
    bins: int,
    limit: float,
) -> dict:
    """Return spatial projection reports."""
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("spatial projection limit must be positive")
    pairs = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    return {
        name: _histogram_2d_report(
            values, left, right, bins, (-float(limit), float(limit))
        )
        for name, (left, right) in pairs.items()
    }


def _direct_component_report(
    samples: list[dict],
    bins_per_axis: int,
    translation_limit_m: float,
    rotation_limit_deg: float,
) -> dict:
    """Return direct component report."""
    components = np.asarray([
        sample["direct_box_components"] for sample in samples
    ], dtype=np.float64)
    if components.shape != (len(samples), 6) or not np.all(np.isfinite(components)):
        raise ValueError("Direct-Euler statistics require finite Nx6 components")
    limits = np.asarray(
        [translation_limit_m] * 3 + [rotation_limit_deg] * 3,
        dtype=np.float64,
    )
    if np.any(limits <= 0.0) or np.any(np.abs(components) > limits + 1e-10):
        raise ValueError("Direct-Euler statistics components exceed configured limits")
    edges = {
        name: np.linspace(-limits[index], limits[index], bins_per_axis + 1)
        for index, name in enumerate(DIRECT_BOX_COORDINATE_ORDER)
    }
    marginal = {}
    pairwise = {}
    for index, name in enumerate(DIRECT_BOX_COORDINATE_ORDER):
        marginal[name] = {
            "edges": edges[name].tolist(),
            "counts": np.histogram(components[:, index], bins=edges[name])[0]
            .astype(np.int64).tolist(),
        }
    for left in range(6):
        for right in range(left + 1, 6):
            left_name = DIRECT_BOX_COORDINATE_ORDER[left]
            right_name = DIRECT_BOX_COORDINATE_ORDER[right]
            counts, _, _ = np.histogram2d(
                components[:, left],
                components[:, right],
                bins=[edges[left_name], edges[right_name]],
            )
            pairwise[f"{left_name}__{right_name}"] = {
                "left": left_name,
                "right": right_name,
                "counts": counts.astype(np.int64).tolist(),
            }
    return {
        "coordinate_order": list(DIRECT_BOX_COORDINATE_ORDER),
        "coordinate_units": ["m", "m", "m", "deg", "deg", "deg"],
        "sample_count": len(samples),
        "marginal": marginal,
        "pairwise": pairwise,
    }


def _build_extrinsic_distribution_report(
    task: GenerationTask,
    level,
    scope: str,
    samples: list[dict],
    coverage: dict,
    dataset_role: str,
) -> dict:
    """Build extrinsic distribution report."""
    arrays = _residual_arrays(samples)
    translation_limit = float(level.residual_translation_normalization_m(
        task.nominal.matrix
    ))
    rotation_limit = float(level.max_geodesic_rotation_deg)
    projection_translation_limit = (
        float(level.max_translation_component_m)
        if coverage["mode"] == DIRECT_EULER_BOX_COVERAGE
        else translation_limit
    )
    report = {
        "version": LEVEL_STATISTICS_VERSION,
        "scope": scope,
        "dataset_role": dataset_role,
        "mode": coverage["mode"],
        "sample_count": len(samples),
        "translation_unit": "m",
        "rotation_vector_unit": "deg",
        "six_dimensional_euclidean_distance_used": False,
        "translation_projections": _spatial_projection_reports(
            arrays["translation_m"], 24,
            max(projection_translation_limit, 1e-9),
        ),
        "rotation_vector_projections": _spatial_projection_reports(
            arrays["rotvec_deg"], 24, max(rotation_limit, 1e-9)
        ),
    }
    if coverage["mode"] == DIRECT_EULER_BOX_COVERAGE:
        report["direct_euler_components"] = _direct_component_report(
            samples,
            int(coverage["accepted_samples"]["bins_per_axis"]),
            float(level.max_translation_component_m),
            float(level.max_rotation_deg),
        )
        report["coverage_layers"] = {
            name: coverage[name]
            for name in (
                "proposed_residuals",
                "accepted_unique_residuals",
                "accepted_samples",
            )
        }
    elif coverage["mode"] == GEODESIC_PRODUCT_BALL_COVERAGE:
        report["geodesic_coverage"] = {
            "translation_cells": coverage["translation_cells"],
            "rotation_cells": coverage["rotation_cells"],
            "support_matrix": coverage["support_matrix"],
            "empty_translation_cells": coverage["empty_translation_cells"],
            "empty_rotation_cells": coverage["empty_rotation_cells"],
        }
    elif coverage["mode"] == FIXED_IDENTITY_COVERAGE:
        report["fixed_identity"] = {
            "passed": bool(coverage["passed"]),
            "sample_count": int(coverage["sample_count"]),
            "unique_residual_count": int(coverage["unique_residual_count"]),
            "residual": list(coverage["residual"]),
        }
    else:
        raise ValueError(f"unsupported statistics coverage mode: {coverage['mode']}")
    return report


def _plot_histogram_report(ax, report: dict, title: str, x_label: str) -> None:
    """Plot histogram report."""
    counts = np.asarray(report["counts"], dtype=np.float64).T
    x_edges = np.asarray(report["x_edges"], dtype=np.float64)
    y_edges = np.asarray(report["y_edges"], dtype=np.float64)
    image = ax.imshow(
        counts,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="viridis",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(x_label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_reuse_audit(report: dict):
    """Plot reuse audit."""
    plt = _matplotlib_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(np.asarray(axes).reshape(-1), REUSE_AUDIT_PLOT_METRICS):
        metric = report["metrics"][name]
        values = np.asarray(list(metric["counts"].values()), dtype=np.float64)
        if not len(values) or np.any(values < 0.0):
            raise ValueError(f"reuse statistics metric is empty or invalid: {name}")
        maximum = int(np.max(values))
        bins = (
            np.arange(0.5, maximum + 1.5, 1.0)
            if maximum <= 60 else np.linspace(0.5, maximum + 0.5, 41)
        )
        ax.hist(values, bins=bins, color="#2f6f8f", alpha=0.82)
        ax.set_title(REUSE_AUDIT_TITLES[name])
        ax.set_xlabel("accepted samples" if "unique_pairs" not in name else "unique pairs")
        ax.set_ylabel("identity count")
        ordered = np.sort(values)
        ecdf = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
        twin = ax.twinx()
        twin.step(ordered, ecdf, where="post", color="#c23b22", linewidth=1.5)
        twin.set_ylim(0.0, 1.02)
        twin.set_ylabel("ECDF")
        ax.text(
            0.98,
            0.95,
            f"keys={metric['unique_keys']}\nmean={metric['mean_reuse']:.2f}\n"
            f"p95={metric['p95_reuse']:.2f}\nmax={metric['maximum_reuse']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )
    fig.suptitle(
        f"Reuse audit | {report['scope']} | n={report['sample_count']}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig


def _plot_direct_extrinsic_distribution(report: dict):
    """Plot direct extrinsic distribution."""
    plt = _matplotlib_pyplot()
    component = report["direct_euler_components"]
    names = component["coordinate_order"]
    units = component["coordinate_units"]
    fig, axes = plt.subplots(7, 6, figsize=(21, 23))
    for row in range(6):
        for column in range(6):
            ax = axes[row, column]
            if row < column:
                ax.axis("off")
                continue
            if row == column:
                marginal = component["marginal"][names[row]]
                edges = np.asarray(marginal["edges"], dtype=np.float64)
                counts = np.asarray(marginal["counts"], dtype=np.float64)
                ax.bar(
                    0.5 * (edges[:-1] + edges[1:]),
                    counts,
                    width=np.diff(edges) * 0.9,
                    color="#2f6f8f",
                )
                ax.set_title(f"{names[row]} marginal")
            else:
                key = f"{names[column]}__{names[row]}"
                pairwise = component["pairwise"][key]
                counts = np.asarray(pairwise["counts"], dtype=np.float64).T
                image = ax.imshow(counts, origin="lower", aspect="auto", cmap="viridis")
                ax.set_title(f"{names[column]} x {names[row]}", fontsize=8)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(units[column], fontsize=8)
            ax.tick_params(labelsize=7)
    for column, name in enumerate(("xy", "xz", "yz")):
        _plot_histogram_report(
            axes[6, column],
            report["rotation_vector_projections"][name],
            f"rotvec {name.upper()}",
            "deg",
        )
    for column in range(3, 6):
        axes[6, column].axis("off")
    fig.suptitle(
        f"Direct-Euler extrinsic distribution | {report['scope']} | "
        f"n={report['sample_count']}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    return fig


def _plot_geodesic_extrinsic_distribution(report: dict):
    """Plot geodesic extrinsic distribution."""
    plt = _matplotlib_pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for column, name in enumerate(("xy", "xz", "yz")):
        _plot_histogram_report(
            axes[0, column], report["translation_projections"][name],
            f"translation {name.upper()}", "m",
        )
        _plot_histogram_report(
            axes[1, column], report["rotation_vector_projections"][name],
            f"rotvec {name.upper()}", "deg",
        )
    coverage = report["geodesic_coverage"]
    translation_cells = list(coverage["translation_cells"])
    rotation_cells = list(coverage["rotation_cells"])
    axes[2, 0].bar(
        np.arange(len(translation_cells)),
        [coverage["translation_cells"][key]["candidate_count"] for key in translation_cells],
        color="#2f6f8f",
    )
    axes[2, 0].set_title("Translation shell/direction cells")
    axes[2, 1].bar(
        np.arange(len(rotation_cells)),
        [coverage["rotation_cells"][key]["candidate_count"] for key in rotation_cells],
        color="#c23b22",
    )
    axes[2, 1].set_title("Rotation shell/direction cells")
    support = np.asarray([
        [coverage["support_matrix"].get(t_key, {}).get(r_key, 0) for r_key in rotation_cells]
        for t_key in translation_cells
    ], dtype=np.float64)
    image = axes[2, 2].imshow(support, origin="lower", aspect="auto", cmap="magma")
    axes[2, 2].set_title("Translation x rotation support")
    fig.colorbar(image, ax=axes[2, 2], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Geodesic extrinsic distribution | {report['scope']} | n={report['sample_count']}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig


def _plot_fixed_extrinsic_distribution(report: dict):
    """Plot fixed extrinsic distribution."""
    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    fixed = report["fixed_identity"]
    ax.bar(DIRECT_BOX_COORDINATE_ORDER, fixed["residual"], color="#2f6f8f")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("normalized residual")
    ax.set_title(
        f"Fixed identity coverage | {report['scope']} | n={report['sample_count']} | "
        f"passed={fixed['passed']}"
    )
    fig.tight_layout()
    return fig


def _plot_extrinsic_distribution(report: dict):
    """Plot extrinsic distribution."""
    if report["mode"] == DIRECT_EULER_BOX_COVERAGE:
        return _plot_direct_extrinsic_distribution(report)
    if report["mode"] == GEODESIC_PRODUCT_BALL_COVERAGE:
        return _plot_geodesic_extrinsic_distribution(report)
    if report["mode"] == FIXED_IDENTITY_COVERAGE:
        return _plot_fixed_extrinsic_distribution(report)
    raise ValueError(f"unsupported extrinsic plot mode: {report['mode']}")


def _plot_coverage_layers(coverage: dict, scope: str):
    """Plot coverage layers."""
    plt = _matplotlib_pyplot()
    mode = coverage["mode"]
    if mode == DIRECT_EULER_BOX_COVERAGE:
        layers = (
            ("proposed_residuals", "Proposed residuals"),
            ("accepted_unique_residuals", "Accepted unique residuals"),
            ("accepted_samples", "Accepted samples"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, (key, title) in zip(axes, layers):
            layer = coverage[key]
            matrix = np.asarray([
                layer["marginal_bin_counts"][name]
                for name in DIRECT_BOX_COORDINATE_ORDER
            ], dtype=np.float64)
            row_sum = np.maximum(np.sum(matrix, axis=1, keepdims=True), 1.0)
            image = ax.imshow(matrix / row_sum, aspect="auto", cmap="viridis")
            ax.set_yticks(np.arange(6), DIRECT_BOX_COORDINATE_ORDER)
            ax.set_xlabel("normalized component bin")
            ax.set_title(
                f"{title}\nn={layer['sample_count']} occupied={layer['occupied_cells']}/"
                f"{layer['total_cells']} collision={layer['collision_probability']:.5f}"
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    elif mode == GEODESIC_PRODUCT_BALL_COVERAGE:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        translation = coverage["translation_cells"]
        rotation = coverage["rotation_cells"]
        t_keys = list(translation)
        r_keys = list(rotation)
        axes[0].bar(
            np.arange(len(t_keys)),
            [translation[key]["candidate_count"] for key in t_keys],
            color="#2f6f8f",
        )
        axes[0].set_title(
            f"Translation cells\noccupancy={coverage['translation_cell_occupancy']:.3f}"
        )
        axes[1].bar(
            np.arange(len(r_keys)),
            [rotation[key]["candidate_count"] for key in r_keys],
            color="#c23b22",
        )
        axes[1].set_title(
            f"Rotation cells\noccupancy={coverage['rotation_cell_occupancy']:.3f}"
        )
        support = np.asarray([
            [coverage["support_matrix"].get(t_key, {}).get(r_key, 0) for r_key in r_keys]
            for t_key in t_keys
        ], dtype=np.float64)
        image = axes[2].imshow(support, origin="lower", aspect="auto", cmap="magma")
        axes[2].set_title("Conditional support matrix")
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    elif mode == FIXED_IDENTITY_COVERAGE:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(DIRECT_BOX_COORDINATE_ORDER, coverage["residual"], color="#2f6f8f")
        ax.set_title(
            f"Fixed identity coverage | {scope} | n={coverage['sample_count']} | "
            f"passed={coverage['passed']}"
        )
        axes = [ax]
    else:
        raise ValueError(f"unsupported coverage plot mode: {mode}")
    fig.suptitle(f"Coverage layers | {scope}", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig


def _write_statistics_scope(
    directory: Path,
    task: GenerationTask,
    level,
    scope: str,
    scope_data: dict,
    coverage: dict,
    dataset_role: str,
) -> dict:
    """Write statistics scope."""
    samples = list(scope_data["samples"])
    reuse = {
        "version": LEVEL_STATISTICS_VERSION,
        "scope": scope,
        "dataset_role": dataset_role,
        "sample_count": len(samples),
        "scene_count": len(scope_data["scenes"]),
        "sequence_count": len(scope_data["sequences"]),
        "scenes": list(scope_data["scenes"]),
        "sequences": list(scope_data["sequences"]),
        "metrics": _reuse_audit_from_samples(samples),
    }
    for metric in REUSE_AUDIT_METRICS[:4]:
        if reuse["metrics"][metric]["total_accepted_samples"] != len(samples):
            raise RuntimeError(f"reuse metric sample count mismatch: {scope}/{metric}")
    extrinsic_distribution_generated = (
        task.residual_sampler.mode != FIXED_IDENTITY
    )
    extrinsic = (
        _build_extrinsic_distribution_report(
            task, level, scope, samples, coverage, dataset_role
        )
        if extrinsic_distribution_generated else None
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(directory / "reuse_audit.json", reuse)
    if extrinsic is not None:
        _write_json_atomic(
            directory / "extrinsic_distribution.json", extrinsic
        )
    plt = _matplotlib_pyplot()
    figures = []
    try:
        figures.append((_plot_reuse_audit(reuse), directory / "reuse_audit.png"))
        if extrinsic is not None:
            figures.append((
                _plot_extrinsic_distribution(extrinsic),
                directory / "extrinsic_distribution.png",
            ))
        figures.append((
            _plot_coverage_layers(coverage, scope),
            directory / "coverage_layers.png",
        ))
        for figure, path in figures:
            _write_statistics_figure(figure, path)
    finally:
        for figure, _ in figures:
            plt.close(figure)
    artifacts = {
        "sample_count": len(samples),
        "scene_count": len(scope_data["scenes"]),
        "sequence_count": len(scope_data["sequences"]),
        "reuse_audit_json": f"statistics/{scope}/reuse_audit.json",
        "reuse_audit_png": f"statistics/{scope}/reuse_audit.png",
        "coverage_layers_png": f"statistics/{scope}/coverage_layers.png",
        "extrinsic_distribution_generated": (
            extrinsic_distribution_generated
        ),
    }
    if extrinsic_distribution_generated:
        artifacts.update({
            "extrinsic_distribution_json": (
                f"statistics/{scope}/extrinsic_distribution.json"
            ),
            "extrinsic_distribution_png": (
                f"statistics/{scope}/extrinsic_distribution.png"
            ),
        })
    else:
        artifacts["extrinsic_distribution_skip_reason"] = (
            "fixed_identity_has_no_extrinsic_distribution"
        )
    return artifacts


def _write_level_statistics(
    level_directory: Path,
    task: GenerationTask,
    level,
    results: Iterable[dict],
    aggregate_coverage: dict,
    dataset_role: str = "primary_quality_gated",
) -> dict:
    """Write level statistics."""
    level_directory.mkdir(parents=True, exist_ok=True)
    results = list(results)
    scopes = _load_level_statistics_samples(level.name, results)
    coverage_by_scope = {"overall": aggregate_coverage["overall"]}
    coverage_by_scope.update(aggregate_coverage["splits"])
    if set(scopes) != set(coverage_by_scope):
        raise RuntimeError("statistics scopes differ from aggregate coverage scopes")
    if len(scopes["overall"]["samples"]) != int(aggregate_coverage["sample_count"]):
        raise RuntimeError("statistics sample count differs from aggregate coverage")
    computed_overall_reuse = _reuse_audit_from_samples(scopes["overall"]["samples"])
    if aggregate_coverage.get("reuse_audit") != computed_overall_reuse:
        raise RuntimeError("statistics reuse audit differs from aggregate coverage")
    temporary = level_directory / f".statistics.{uuid.uuid4().hex}.tmp"
    backup = level_directory / f".statistics.{uuid.uuid4().hex}.backup"
    final = level_directory / "statistics"
    artifacts = {
        "version": LEVEL_STATISTICS_VERSION,
        "dataset_role": dataset_role,
        "root": "statistics",
        "scopes": {},
    }
    try:
        for scope, scope_data in scopes.items():
            artifacts["scopes"][scope] = _write_statistics_scope(
                temporary / scope,
                task,
                level,
                scope,
                scope_data,
                coverage_by_scope[scope],
                dataset_role,
            )
        if final.exists():
            final.replace(backup)
        try:
            temporary.replace(final)
        except BaseException:
            if backup.exists() and not final.exists():
                backup.replace(final)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and final.exists():
            shutil.rmtree(backup)
    return artifacts


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _reuse_existing_sequence(
    task: GenerationTask,
    job: dict,
    output: Path,
    paired_output: Optional[Path],
    source_manifest_fingerprint: str,
    accepted_target_override: Optional[int],
) -> dict:
    """Reuse existing sequence."""
    with (output / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_target = (
        job["accepted_sample_target"]
        if accepted_target_override is None
        else int(accepted_target_override)
    )
    expected_smoke = accepted_target_override is not None
    partial_sequence = metadata.get("partial_sequence") is True
    coverage_enforced = task.accepted_distribution_control != REPORT_ONLY
    expected = {
        "version": STOCHASTIC_DATASET_VERSION,
        "target_network": task.target_network,
        "curriculum_level": job["level"],
        "split": job["split"],
        "smoke_limited": expected_smoke,
        "coverage_scope": task.coverage_scope,
        "accepted_distribution_control": (
            task.accepted_distribution_control
        ),
        "coverage_enforced": coverage_enforced,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if metadata.get("task") != task.to_metadata():
        mismatches["task"] = "task metadata or config fingerprints differ"
    source = metadata.get("source") or {}
    if source.get("manifest_fingerprint") != source_manifest_fingerprint:
        mismatches["source_manifest_fingerprint"] = {
            "expected": source_manifest_fingerprint,
            "actual": source.get("manifest_fingerprint"),
        }
    samples = metadata.get("samples") or []
    computed_reuse_audit = _reuse_audit_from_samples(samples)
    if metadata.get("reuse_audit") != computed_reuse_audit:
        mismatches["reuse_audit"] = {
            "expected": computed_reuse_audit,
            "actual": metadata.get("reuse_audit"),
        }
    if partial_sequence:
        partial_expected = {
            "requested_accepted_sample_target": expected_target,
            "accepted_sample_target": len(samples),
            "termination_reason": "partial_due_to_stall",
            "formal_continuous_coverage_passed": False,
        }
        mismatches.update({
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in partial_expected.items()
            if metadata.get(key) != value
        })
        if not 0 < len(samples) < expected_target:
            mismatches["sample_count"] = {
                "expected": f"between 1 and {expected_target - 1}",
                "actual": len(samples),
            }
    else:
        if metadata.get("accepted_sample_target") != expected_target:
            mismatches["accepted_sample_target"] = {
                "expected": expected_target,
                "actual": metadata.get("accepted_sample_target"),
            }
        if len(samples) != expected_target:
            mismatches["sample_count"] = {
                "expected": expected_target,
                "actual": len(samples),
            }
    if any(sample.get("sequence_id") != job["sequence_id"] for sample in samples):
        mismatches["sequence_id"] = job["sequence_id"]
    if mismatches:
        raise ValueError(f"cannot resume incompatible output {output}: {mismatches}")
    audit = audit_exported_sequence(output, task.sensor_config)
    paired_audit = None
    paired_identity_audit = None
    if task.paired_lidar_export is not None:
        if paired_output is None or not paired_output.is_dir():
            raise ValueError(
                f"cannot resume primary sequence without paired output: {output}"
            )
        base_sensor = load_training_sensor_config(task.sensor_config)
        paired_sensor = _paired_sensor_config(task, base_sensor)
        paired_audit = audit_exported_sequence(
            paired_output, paired_sensor
        )
        paired_identity_audit = audit_paired_sequence_identity(
            output, paired_output
        )
    coverage = metadata["continuous_coverage"]
    pair_counts = metadata["accepted_pair_counts"]
    return {
        "status": "reused",
        "output": str(output),
        "target_network": task.target_network,
        "curriculum_level": job["level"],
        "split": job["split"],
        "scene": job["scene"],
        "sample_count": len(samples),
        "attempt_count": int(metadata["attempt_count"]),
        "acceptance_rate": float(metadata["acceptance_rate"]),
        "unique_accepted_pairs": len(pair_counts),
        "maximum_pair_reuse": max(pair_counts.values()),
        "reuse_audit": computed_reuse_audit,
        "sequence_continuous_coverage_passed": coverage["passed"],
        "continuous_coverage": coverage,
        "formal_continuous_coverage_passed": (
            False if expected_smoke or partial_sequence else None
        ),
        "partial_sequence": partial_sequence,
        "smoke_limited": expected_smoke,
        "audit": audit,
        "paired_output": str(paired_output) if paired_output is not None else None,
        "paired_audit": paired_audit,
        "paired_identity_audit": paired_identity_audit,
    }


def _export_job(
    task: GenerationTask,
    source_manifest: Union[str, Path],
    job: dict,
    output: Path,
    paired_output: Optional[Path],
    accepted_target_override: Optional[int],
    allow_incomplete_smoke: bool,
    cache_root: Union[str, Path],
) -> dict:
    """Export job."""
    try:
        return filter_and_export_sequence(
            task=task,
            source_manifest=source_manifest,
            sequence_id=job["sequence_id"],
            level_name=job["level"],
            output=output,
            paired_output=paired_output,
            accepted_target_override=accepted_target_override,
            allow_incomplete_smoke=allow_incomplete_smoke,
            cache_root=cache_root,
        )
    except BaseException as exc:
        emit_runtime_log(
            "scene_failed",
            "Scene generation failed",
            scene=job["scene"],
            split=job["split"],
            level=job["level"],
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _log_job_completion(
    result: dict,
    completed_job_count: int,
    total_job_count: int,
) -> None:
    """Return log job completion."""
    emit_runtime_log(
        "task_progress",
        "Scene task completed",
        scene=result["scene"],
        split=result["split"],
        level=result["curriculum_level"],
        status=result["status"],
        accepted_count=result["sample_count"],
        attempt_count=result["attempt_count"],
        acceptance_rate=round(result["acceptance_rate"], 6),
        completed_jobs=completed_job_count,
        total_jobs=total_job_count,
        remaining_jobs=total_job_count - completed_job_count,
    )


def _coverage_candidates(
    result: dict, expected_level: str
) -> Tuple[str, List[dict]]:
    """Return coverage candidates."""
    root = Path(result["output"])
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("curriculum_level") != expected_level:
        raise ValueError(
            f"{root}: expected level {expected_level}, got "
            f"{metadata.get('curriculum_level')}"
        )
    split = str(metadata.get("split", ""))
    if split not in SPLIT_NAMES:
        raise ValueError(f"{root}: invalid split {split!r}")
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{root}: no sample metadata for aggregate coverage")
    candidates = []
    for sample in samples:
        sequence_id = str(sample.get("sequence_id", ""))
        pair_id = str(sample.get("pair_id", ""))
        if not sequence_id or not pair_id:
            raise ValueError(f"{root}: sample is missing sequence_id or pair_id")
        candidates.append({
            "normalized": sample["normalized"],
            "pair_id": f"{sequence_id}:{pair_id}",
        })
    return split, candidates


def audit_stochastic_level_coverage(
    level: str,
    results: Iterable[dict],
    config: CoverageConfig,
) -> dict:
    """Audit stochastic level coverage."""
    results = list(results)
    if not results:
        raise ValueError(f"{level}: no sequence results for level coverage")
    if config.mode == DIRECT_EULER_BOX_COVERAGE:
        overall = merge_direct_box_coverage_reports(
            (result["continuous_coverage"] for result in results),
            int(config.bins_per_axis),
        )
        splits = {
            split: merge_direct_box_coverage_reports(
                (
                    result["continuous_coverage"]
                    for result in results
                    if result["split"] == split
                ),
                int(config.bins_per_axis),
            )
            for split in SPLIT_NAMES
            if any(result["split"] == split for result in results)
        }
    else:
        overall_candidates = []
        split_candidates = {split: [] for split in SPLIT_NAMES}
        for result in results:
            split, candidates = _coverage_candidates(result, level)
            overall_candidates.extend(candidates)
            split_candidates[split].extend(candidates)
        overall = audit_stochastic_coverage(overall_candidates, config)
        splits = {
            split: audit_stochastic_coverage(candidates, config)
            for split, candidates in split_candidates.items()
            if candidates
        }
    failures = []
    if not overall["passed"]:
        failures.append("overall continuous coverage failed")
    failures.extend(
        f"{split} continuous coverage failed"
        for split, report in splits.items()
        if not report["passed"]
    )
    return {
        "scope": "per_curriculum_level_and_split_across_scenes",
        "level": level,
        "mode": config.mode,
        "passed": not failures,
        "failures": failures,
        "sequence_count": len(results),
        "scene_count": len({result["scene"] for result in results}),
        "sample_count": sum(int(result["sample_count"]) for result in results),
        "overall": overall,
        "splits": splits,
    }


def select_task_jobs(
    task: GenerationTask,
    source_manifest: Union[str, Path],
    scenes: Optional[Iterable[str]] = None,
    levels: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Select task jobs."""
    with Path(source_manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    available = {
        str(value["scene"]): str(value["sequence_id"])
        for value in manifest.get("sequences", [])
    }
    selected_scenes = (
        set(available)
        if scenes is None
        else {str(value).strip() for value in scenes}
    )
    if not selected_scenes or any(not value for value in selected_scenes):
        raise ValueError("scene selection must contain non-empty names")
    configured = set(task.source.scenes)
    unknown = sorted(selected_scenes - configured)
    if unknown:
        raise ValueError(f"selected scenes are absent from task source config: {unknown}")
    missing = sorted(selected_scenes - set(available))
    if missing:
        raise ValueError(f"selected scenes are absent from source manifest: {missing}")
    configured_levels = [value.name for value in task.levels]
    if levels is None:
        selected_levels = configured_levels
    else:
        requested_levels = [str(value).strip() for value in levels]
        if not requested_levels or any(not value for value in requested_levels):
            raise ValueError("level selection must contain non-empty names")
        duplicates = sorted({
            value for value in requested_levels
            if requested_levels.count(value) > 1
        })
        if duplicates:
            raise ValueError(f"duplicate selected curriculum levels: {duplicates}")
        unknown_levels = sorted(set(requested_levels) - set(configured_levels))
        if unknown_levels:
            raise ValueError(
                "selected levels are absent from task curriculum: "
                f"{unknown_levels}"
            )
        requested = set(requested_levels)
        selected_levels = [
            value for value in configured_levels if value in requested
        ]
    jobs = []
    for job in task.jobs():
        if job["scene"] not in selected_scenes:
            continue
        if job["level"] not in selected_levels:
            continue
        jobs.append({
            **job,
            "sequence_id": available[job["scene"]],
        })
    expected = len(selected_scenes) * len(selected_levels)
    if len(jobs) != expected:
        raise RuntimeError(f"selected {len(jobs)} jobs, expected {expected}")
    return jobs


def export_generation_task(
    task: GenerationTask,
    source_manifest: Union[str, Path],
    output_root: Union[str, Path],
    scenes: Optional[Iterable[str]] = None,
    levels: Optional[Iterable[str]] = None,
    accepted_target_override: Optional[int] = None,
    allow_incomplete_smoke: bool = False,
    resume_existing: bool = False,
    continue_on_coverage_failure: bool = False,
    workers: int = 1,
    cache_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Export generation task."""
    task_start = perf_counter()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paired_output_root = None
    if task.paired_lidar_export is not None:
        paired_output_root = output_root.with_name(
            output_root.name + task.paired_lidar_export.output_suffix
        )
        if paired_output_root == output_root:
            raise ValueError("paired output root must differ from primary root")
        if paired_output_root.exists() and not resume_existing:
            raise FileExistsError(
                f"paired output root already exists: {paired_output_root}"
            )
        paired_output_root.mkdir(parents=True, exist_ok=True)
    resolved_cache_root = Path(
        cache_root if cache_root is not None else output_root / ".cache"
    ).resolve()
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    if scenes is not None and accepted_target_override is None:
        raise ValueError(
            "a scene subset is smoke-only and requires accepted_target_override"
        )
    jobs = select_task_jobs(
        task,
        source_manifest,
        scenes=scenes,
        levels=levels,
    )
    total_job_count = len(jobs)
    completed_job_count = 0
    emit_runtime_log(
        "stage_started",
        "Starting scene task scheduling",
        total_jobs=total_job_count,
        workers=workers,
        levels=sorted({job["level"] for job in jobs}),
        scenes=sorted({job["scene"] for job in jobs}),
    )
    with Path(source_manifest).open("r", encoding="utf-8") as handle:
        source_manifest_metadata = json.load(handle)
    if paired_output_root is not None:
        paired_source_manifest = (
            paired_output_root / "_source" / "source_manifest.json"
        )
        paired_source_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_manifest, paired_source_manifest)
    source_manifest_fingerprint = str(
        source_manifest_metadata["input_fingerprint"]
    )
    coverage_config = load_coverage_config(task.coverage_config)
    results = []
    level_coverage = {}
    coverage_enforced = (
        task.accepted_distribution_control != REPORT_ONLY
    )
    selected_level_names = {job["level"] for job in jobs}
    selected_levels = [
        value for value in task.levels if value.name in selected_level_names
    ]
    for level in selected_levels:
        level_start = perf_counter()
        level_jobs = [job for job in jobs if job["level"] == level.name]
        emit_runtime_log(
            "stage_started",
            "Starting curriculum-level generation",
            level=level.name,
            level_jobs=len(level_jobs),
            remaining_jobs=total_job_count - completed_job_count,
        )
        results_by_sequence = {}
        pending = []
        for job in level_jobs:
            output = (
                output_root
                / job["level"]
                / job["split"]
                / job["sequence_id"]
            )
            paired_output = (
                paired_output_root
                / job["level"]
                / job["split"]
                / job["sequence_id"]
                if paired_output_root is not None else None
            )
            if output.exists() and resume_existing:
                result = _reuse_existing_sequence(
                    task,
                    job,
                    output,
                    paired_output,
                    source_manifest_fingerprint,
                    accepted_target_override,
                )
                results_by_sequence[job["sequence_id"]] = result
                completed_job_count += 1
                _log_job_completion(
                    result, completed_job_count, total_job_count
                )
            else:
                if paired_output is not None and paired_output.exists():
                    raise ValueError(
                        "paired sequence exists without reusable primary output: "
                        f"{paired_output}"
                    )
                pending.append((job, output, paired_output))
        if workers == 1:
            for job, output, paired_output in pending:
                result = _export_job(
                    task,
                    source_manifest,
                    job,
                    output,
                    paired_output,
                    accepted_target_override,
                    allow_incomplete_smoke,
                    resolved_cache_root,
                )
                results_by_sequence[job["sequence_id"]] = result
                completed_job_count += 1
                _log_job_completion(
                    result, completed_job_count, total_job_count
                )
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _export_job,
                        task,
                        source_manifest,
                        job,
                        output,
                        paired_output,
                        accepted_target_override,
                        allow_incomplete_smoke,
                        resolved_cache_root,
                    ): job
                    for job, output, paired_output in pending
                }
                for future in as_completed(futures):
                    job = futures[future]
                    result = future.result()
                    results_by_sequence[job["sequence_id"]] = result
                    completed_job_count += 1
                    _log_job_completion(
                        result, completed_job_count, total_job_count
                    )
        level_results = [
            results_by_sequence[job["sequence_id"]] for job in level_jobs
        ]
        for result in level_results:
            results.append(result)
        coverage = audit_stochastic_level_coverage(
            level.name, level_results, coverage_config
        )
        coverage["reuse_audit"] = _merge_reuse_audits(
            result.get("reuse_audit", {}) for result in level_results
        )
        statistics_artifacts = _write_level_statistics(
            output_root / level.name,
            task,
            level,
            level_results,
            coverage,
        )
        coverage["statistics_artifacts"] = statistics_artifacts
        level_coverage[level.name] = coverage
        _write_json_atomic(
            output_root / level.name / "coverage.json", coverage
        )
        if paired_output_root is not None:
            paired_statistics_artifacts = _write_level_statistics(
                paired_output_root / level.name,
                task,
                level,
                level_results,
                coverage,
                dataset_role="paired_lidar_fov_derivative",
            )
            _write_json_atomic(
                paired_output_root / level.name / "coverage.json",
                {
                    **coverage,
                    "dataset_role": "paired_lidar_fov_derivative",
                    "coverage_source": "identical_primary_residual_samples",
                    "statistics_artifacts": paired_statistics_artifacts,
                },
            )
        emit_runtime_log(
            "stage_completed",
            "Curriculum-level generation and coverage audit completed",
            level=level.name,
            sample_count=coverage["sample_count"],
            coverage_passed=coverage["passed"],
            remaining_jobs=total_job_count - completed_job_count,
            elapsed_seconds=round(perf_counter() - level_start, 3),
        )
        progress = {
            "version": STOCHASTIC_TASK_EXPORT_VERSION,
            "status": "in_progress",
            "target_network": task.target_network,
            "completed_levels": list(level_coverage),
            "result_count": len(results),
            "level_coverage": level_coverage,
        }
        _write_json_atomic(output_root / "task_progress.json", progress)
        if paired_output_root is not None:
            _write_json_atomic(
                paired_output_root / "task_progress.json",
                {
                    **progress,
                    "dataset_role": "paired_lidar_fov_derivative",
                    "paired_lidar_export": (
                        task.paired_lidar_export.to_metadata()
                    ),
                },
            )
        if (
            accepted_target_override is None
            and coverage_enforced
            and not coverage["passed"]
            and not continue_on_coverage_failure
        ):
            raise StochasticGenerationError(
                f"{level.name}: formal aggregate continuous coverage failed",
                {
                    "status": "insufficient_aggregate_continuous_coverage",
                    "curriculum_level": level.name,
                    "coverage_scope": task.coverage_scope,
                    "coverage": coverage,
                },
            )
    formal_coverage_passed = (
        accepted_target_override is None
        and all(report["passed"] for report in level_coverage.values())
    ) if coverage_enforced else None
    task_reuse_audit = _merge_reuse_audits(
        result.get("reuse_audit", {}) for result in results
    )
    report = {
        "version": STOCHASTIC_TASK_EXPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target_network": task.target_network,
        "configured_levels": [value.name for value in task.levels],
        "required_levels": [value.name for value in selected_levels],
        "level_subset": len(selected_levels) != len(task.levels),
        "scene_count_per_level": len({value["scene"] for value in jobs}),
        "job_count": len(jobs),
        "accepted_target_override": accepted_target_override,
        "smoke_limited": accepted_target_override is not None,
        "coverage_scope": task.coverage_scope,
        "accepted_distribution_control": (
            task.accepted_distribution_control
        ),
        "coverage_enforced": coverage_enforced,
        "task_config_fingerprint": task.config_fingerprints["task"],
        "config_fingerprints": dict(task.config_fingerprints),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "generation_workers": workers,
        "continued_after_coverage_failure": bool(
            continue_on_coverage_failure
            and any(not value["passed"] for value in level_coverage.values())
        ),
        "runtime_parallelism": {
            "available_cpu_count": os.cpu_count(),
            "scene_workers": workers,
            "opencv_threads_per_worker": (
                task.runtime.opencv_threads_per_worker
            ),
            "numeric_threads_per_worker": (
                task.runtime.numeric_threads_per_worker
            ),
            "configured_opencv_thread_budget": (
                workers * task.runtime.opencv_threads_per_worker
            ),
            "configured_numeric_thread_budget": (
                workers * task.runtime.numeric_threads_per_worker
            ),
            "oversubscribed_by_configuration": (
                os.cpu_count() is not None
                and workers * max(
                    task.runtime.opencv_threads_per_worker,
                    task.runtime.numeric_threads_per_worker,
                )
                > int(os.cpu_count())
            ),
            "persistent_pair_pool_cache_enabled": True,
        },
        "formal_continuous_coverage_passed": formal_coverage_passed,
        "level_coverage": level_coverage,
        "reuse_audit": task_reuse_audit,
        "jobs": jobs,
        "results": results,
        "paired_output_root": (
            str(paired_output_root)
            if paired_output_root is not None else None
        ),
    }
    _write_json_atomic(output_root / "task_manifest.json", report)
    if paired_output_root is not None:
        paired_report = {
            **report,
            "dataset_role": "paired_lidar_fov_derivative",
            "source_primary_output_root_name": output_root.name,
            "paired_lidar_export": task.paired_lidar_export.to_metadata(),
            "results": [
                {
                    **result,
                    "output": result["paired_output"],
                    "audit": result["paired_audit"],
                    "paired_identity_audit": result[
                        "paired_identity_audit"
                    ],
                }
                for result in results
            ],
        }
        _write_json_atomic(
            paired_output_root / "task_manifest.json", paired_report
        )
    progress_path = output_root / "task_progress.json"
    if progress_path.exists():
        progress_path.unlink()
    if paired_output_root is not None:
        paired_progress_path = paired_output_root / "task_progress.json"
        if paired_progress_path.exists():
            paired_progress_path.unlink()
    emit_runtime_log(
        "stage_completed",
        "Scene task scheduling completed",
        completed_jobs=completed_job_count,
        total_jobs=total_job_count,
        remaining_jobs=total_job_count - completed_job_count,
        elapsed_seconds=round(perf_counter() - task_start, 3),
    )
    return report


def _write_json_compact_atomic(path: Path, value: dict) -> None:
    """Write JSON compact atomic."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    """Return file sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignore_primary_runtime_artifacts(
    directory: str,
    names: list[str],
) -> set[str]:
    """Return ignore primary runtime artifacts."""
    del directory
    ignored = {
        name
        for name in names
        if name in {".cache", "velodyne", "task_progress.json"}
        or name.startswith(".staging-")
    }
    return ignored


def _validate_source_sensor_metadata(
    source_directory: Path,
    metadata: dict,
    base_sensor: TrainingSensorConfig,
) -> None:
    """Validate source sensor metadata."""
    expected = json.loads(json.dumps(
        base_sensor.to_dict(), ensure_ascii=False
    ))
    stored = json.loads(json.dumps(
        metadata.get("sensor_setup"), ensure_ascii=False
    ))
    stored_lidar = stored.get("lidar") if isinstance(stored, dict) else None
    expected_lidar = expected.get("lidar")
    if (
        isinstance(stored_lidar, dict)
        and isinstance(expected_lidar, dict)
        and "accumulation_duration_s" not in stored_lidar
        and expected_lidar.get("accumulation_duration_s") is None
    ):
        stored_lidar["accumulation_duration_s"] = None
    if stored != expected:
        raise ValueError(
            f"{source_directory}: source sensor metadata does not match config"
        )


def _derive_lidar_modality_sequence(
    source_sequence_directory: str,
    staged_sequence_directory: str,
    final_sequence_directory: str,
    source_manifest_path: str,
    data_root: str,
    base_sensor: TrainingSensorConfig,
    target_sensor: TrainingSensorConfig,
    derivation: LidarModalityDerivationConfig,
    depth_cache_max_mb: int,
    progress_interval_seconds: float,
) -> dict:
    """Derive LiDAR modality sequence."""
    source_directory = Path(source_sequence_directory)
    staged_directory = Path(staged_sequence_directory)
    final_directory = Path(final_sequence_directory)
    with (source_directory / "metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)
    if metadata.get("dataset_role") != "primary_quality_gated":
        raise ValueError(
            f"{source_directory}: source dataset is not primary_quality_gated"
        )
    _validate_source_sensor_metadata(
        source_directory, metadata, base_sensor
    )
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{source_directory}: source samples are missing")

    sequence_id = str(samples[0].get("sequence_id", ""))
    if not sequence_id or any(
        sample.get("sequence_id") != sequence_id for sample in samples
    ):
        raise ValueError(f"{source_directory}: inconsistent sequence identity")
    sequence, split, _ = load_manifest_sequence(
        source_manifest_path,
        sequence_id,
        data_root_override=data_root,
    )
    if split != metadata.get("split"):
        raise ValueError(f"{source_directory}: split differs from source manifest")
    frames_by_id = {frame.frame_id: frame for frame in sequence.frames}

    cv2.setNumThreads(1)
    configure_depth_cache(max(1, int(depth_cache_max_mb)) * 1024 * 1024)
    cloud_directory = staged_directory / "velodyne"
    cloud_directory.mkdir(parents=False, exist_ok=False)
    point_counts = []
    sequence_start = perf_counter()
    last_progress = sequence_start
    try:
        for index, sample in enumerate(samples):
            frame_id = f"{index:06d}"
            if sample.get("frame_id") != frame_id:
                raise ValueError(
                    f"{source_directory}: frame ordering mismatch at {index}"
                )
            source_frame_id = str(sample["source_frame"])
            target_frame_id = str(sample["target_frame"])
            try:
                source_frame = frames_by_id[source_frame_id]
                target_frame = frames_by_id[target_frame_id]
            except KeyError as exc:
                raise ValueError(
                    f"{source_directory}: sample references unknown frame {exc}"
                ) from exc
            lidar_stage = generate_candidate_lidar(
                source_frame,
                target_frame,
                sequence.pano_from_face,
                sample,
                target_sensor,
            )
            point_counts.append(int(len(lidar_stage.point_cloud)))
            temporary_cloud = cloud_directory / f".{frame_id}.tmp.bin"
            lidar_stage.point_cloud.tofile(temporary_cloud)
            target_cloud = cloud_directory / f"{frame_id}.bin"
            temporary_cloud.replace(target_cloud)
            source_cloud = source_directory / "velodyne" / f"{frame_id}.bin"

            source_generated_sensor = json.loads(json.dumps(
                sample["generated_sensor"], ensure_ascii=False
            ))
            generated_sensor = {
                **source_generated_sensor,
                "version": "lidar_modality_control_derivative_sensor_v1",
                "sensor_profile": target_sensor.profile,
                "point_cloud_shape": list(lidar_stage.point_cloud.shape),
                "lidar": lidar_stage.report,
                "lidar_modality_derivation": {
                    "config_fingerprint": derivation.config_fingerprint,
                    "source_primary_frame_id": frame_id,
                    "image_reused_without_rerendering": True,
                    "quality_recomputed": False,
                    "source_point_cloud_sha256": _file_sha256(source_cloud),
                    "target_point_cloud_sha256": _file_sha256(target_cloud),
                },
            }
            sample["source_primary_generated_sensor"] = source_generated_sensor
            sample["generated_sensor"] = generated_sensor
            sample["lidar_modality_derivation"] = {
                "only_changed_modality": "lidar_point_cloud",
                "coordinate_frame_policy": derivation.coordinate_frame_policy,
                "sample_policy": derivation.sample_policy,
                "quality_policy": derivation.quality_policy,
                "source_quality_reused": True,
                "derived_quality_filter_executed": False,
            }
            now = perf_counter()
            if (
                now - last_progress >= float(progress_interval_seconds)
                or index + 1 == len(samples)
            ):
                emit_runtime_log(
                    "stage_progress",
                    "Deriving LiDAR control point-cloud sequences",
                    sequence_id=sequence_id,
                    completed_samples=index + 1,
                    total_samples=len(samples),
                    elapsed_seconds=round(now - sequence_start, 3),
                )
                last_progress = now
    finally:
        clear_depth_cache()

    metadata.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "lidar_modality_control_derivative",
        "source_manifest": "../../../_source/source_manifest.json",
        "source_primary_sensor_setup": base_sensor.to_dict(),
        "sensor_setup": target_sensor.to_dict(),
        "quality_config_scope": "source_primary_acceptance_only",
        "lidar_modality_derivation": {
            **derivation.to_metadata(),
            "acceptance_policy": SOURCE_PRIMARY_ACCEPTANCE,
            "source_primary_dataset_role": "primary_quality_gated",
            "source_primary_sequence_identity": (
                f"{metadata['curriculum_level']}/{metadata['split']}/"
                f"{sequence_id}"
            ),
            "source_quality_reused": True,
            "quality_config_scope": "source_primary_acceptance_only",
            "quality_recomputed": False,
            "quality_filter_executed": False,
            "source_primary_quality_metrics_retained": True,
            "derived_quality_affects_primary_acceptance": False,
        },
        "derived_quality_summary": {
            "evaluated": False,
            "filter_executed": False,
            "source_quality_reused": True,
            "retained_sample_count": len(samples),
        },
    })
    _write_json_compact_atomic(staged_directory / "metadata.json", metadata)

    audit = audit_exported_sequence(staged_directory, target_sensor)
    identity_audit = audit_paired_sequence_identity(
        source_directory, staged_directory
    )
    with (staged_directory / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    audit.update({
        "generated_quality_passed_validated": True,
        "source_primary_quality_passed_validated": True,
        "derived_quality_evaluated": False,
        "source_quality_reused_without_regrading": True,
    })
    manifest.update({
        "dataset_role": "lidar_modality_control_derivative",
        "lidar_modality_derivation": metadata["lidar_modality_derivation"],
        "audit": audit,
        "control_identity_audit": identity_audit,
    })
    _write_json_atomic(staged_directory / "manifest.json", manifest)
    return {
        "status": "derived",
        "output": str(final_directory),
        "source_output": str(source_directory),
        "sequence_id": sequence_id,
        "split": split,
        "sample_count": len(samples),
        "minimum_point_count": min(point_counts),
        "maximum_point_count": max(point_counts),
        "audit": audit,
        "control_identity_audit": identity_audit,
    }


def derive_lidar_modality_dataset(
    task: GenerationTask,
    source_output_root: Union[str, Path],
    output_root: Union[str, Path],
    data_root: Union[str, Path],
    derivation_config_path: Union[str, Path],
    workers: int = 1,
) -> dict:
    """Derive LiDAR modality dataset."""
    source_root = Path(source_output_root).resolve()
    destination_root = Path(output_root).resolve()
    resolved_data_root = Path(data_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"source output root does not exist: {source_root}"
        )
    if destination_root.exists():
        raise FileExistsError(
            f"derived output root already exists: {destination_root}"
        )
    if (
        source_root == destination_root
        or source_root in destination_root.parents
        or destination_root in source_root.parents
    ):
        raise ValueError("source and derived output roots must not be nested")
    if int(workers) < 1:
        raise ValueError("workers must be positive")

    base_sensor = load_training_sensor_config(task.sensor_config)
    derivation = load_lidar_modality_derivation_config(
        derivation_config_path, base_sensor
    )
    target_sensor = derivation.target_sensor
    if target_sensor.camera != base_sensor.camera:
        raise ValueError("derived LiDAR task must preserve the source camera")
    if target_sensor.to_dict() == base_sensor.to_dict():
        raise ValueError("target LiDAR sensor must differ from the source sensor")
    source_manifest = source_root / "_source" / "source_manifest.json"
    source_task_manifest = source_root / "task_manifest.json"
    if not source_manifest.is_file() or not source_task_manifest.is_file():
        raise FileNotFoundError(
            "source output lacks _source/source_manifest.json or task_manifest.json"
        )
    with source_task_manifest.open("r", encoding="utf-8") as handle:
        primary_report = json.load(handle)
    if primary_report.get("status") != "complete":
        raise ValueError("source task manifest is not complete")
    with source_manifest.open("r", encoding="utf-8") as handle:
        source_manifest_metadata = json.load(handle)
    stored_manifest_fingerprint = primary_report.get(
        "source_manifest_fingerprint"
    )
    if (
        source_manifest_metadata.get("input_fingerprint")
        != stored_manifest_fingerprint
    ):
        raise ValueError("source manifest fingerprint differs from task report")
    stored_fingerprints = primary_report.get("config_fingerprints") or {}


    required_fingerprints = (
        "task", "nominal_extrinsic", "source", "sensor"
    )
    mismatched_fingerprints = [
        name for name in required_fingerprints
        if stored_fingerprints.get(name) != task.config_fingerprints[name]
    ]
    if mismatched_fingerprints:
        raise ValueError(
            "source derivation fingerprints differ: "
            f"{mismatched_fingerprints}"
        )
    relative_sequences = sorted(
        metadata_path.parent.relative_to(source_root)
        for metadata_path in source_root.rglob("metadata.json")
    )
    if not relative_sequences or any(
        len(relative.parts) != 3 for relative in relative_sequences
    ):
        raise ValueError(
            "source output must contain level/split/sequence metadata directories"
        )


    for relative in relative_sequences:
        source_directory = source_root / relative
        with (source_directory / "metadata.json").open(
            "r", encoding="utf-8"
        ) as handle:
            source_metadata = json.load(handle)
        _validate_source_sensor_metadata(
            source_directory, source_metadata, base_sensor
        )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = destination_root.with_name(
        f".{destination_root.name}.staging-{uuid.uuid4().hex}"
    )
    emit_runtime_log(
        "stage_started",
        "Copying source images, labels, and audit files into staging",
        source_output=str(source_root),
        staging_output=str(staging_root),
        sequence_count=len(relative_sequences),
    )
    try:
        shutil.copytree(
            source_root,
            staging_root,
            copy_function=shutil.copy2,
            ignore=_ignore_primary_runtime_artifacts,
        )
        arguments = [
            (
                str(source_root / relative),
                str(staging_root / relative),
                str(destination_root / relative),
                str(source_manifest),
                str(resolved_data_root),
                base_sensor,
                target_sensor,
                derivation,
                int(task.runtime.depth_cache_max_mb),
                float(task.runtime.progress_interval_seconds),
            )
            for relative in relative_sequences
        ]
        results = []
        if int(workers) == 1:
            for argument in arguments:
                results.append(_derive_lidar_modality_sequence(*argument))
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [
                    executor.submit(_derive_lidar_modality_sequence, *argument)
                    for argument in arguments
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        results.sort(key=lambda value: value["output"])

        derivative = {
            **derivation.to_metadata(),
            "acceptance_policy": SOURCE_PRIMARY_ACCEPTANCE,
            "source_quality_reused": True,
            "quality_recomputed": False,
            "quality_filter_executed": False,
        }
        task_report = {
            **primary_report,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_role": "lidar_modality_control_derivative",
            "source_primary_output_root": str(source_root),
            "source_primary_output_root_name": source_root.name,
            "lidar_modality_derivation": derivative,
            "source_primary_sensor_setup": base_sensor.to_dict(),
            "sensor_setup": target_sensor.to_dict(),
            "sample_set_identity": "identical_to_source_primary",
            "quality_config_scope": "source_primary_acceptance_only",
            "statistics_provenance": (
                "copied_source_sample_identity_residual_and_reuse_statistics"
            ),
            "generation_workers": int(workers),
            "results": results,
        }
        _write_json_atomic(staging_root / "task_manifest.json", task_report)
        for coverage_path in staging_root.glob("*/coverage.json"):
            with coverage_path.open("r", encoding="utf-8") as handle:
                coverage = json.load(handle)
            coverage.update({
                "dataset_role": "lidar_modality_control_derivative",
                "coverage_source": "identical_primary_residual_samples",
                "point_cloud_quality_recomputed": False,
            })
            if isinstance(coverage.get("statistics_artifacts"), dict):
                coverage["statistics_artifacts"]["dataset_role"] = (
                    "lidar_modality_control_derivative"
                )
            _write_json_atomic(coverage_path, coverage)


        for statistics_path in staging_root.glob(
            "*/statistics/**/*.json"
        ):
            with statistics_path.open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            statistics.update({
                "dataset_role": "lidar_modality_control_derivative",
                "statistics_source": "identical_source_primary_sample_set",
                "target_point_cloud_quality_recomputed": False,
            })
            _write_json_atomic(statistics_path, statistics)

        staging_root.replace(destination_root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    report = {
        "version": STOCHASTIC_TASK_EXPORT_VERSION,
        "status": "complete",
        "dataset_role": "lidar_modality_control_derivative",
        "source_primary_output_root": str(source_root),
        "output_root": str(destination_root),
        "sample_count": sum(result["sample_count"] for result in results),
        "sequence_count": len(results),
        "lidar_modality_derivation": derivation.to_metadata(),
        "target_sensor": target_sensor.to_dict(),
        "sample_set_identity": "identical_to_source_primary",
        "source_quality_reused": True,
        "quality_filter_executed": False,
        "results": results,
    }
    emit_runtime_log(
        "stage_completed",
        "LiDAR control dataset committed transactionally",
        output_root=str(destination_root),
        sample_count=report["sample_count"],
        sequence_count=report["sequence_count"],
    )
    return report


def _audit_metric_distribution(values: Iterable[float]) -> dict:
    """Audit metric distribution."""
    array = np.asarray(list(values), dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError("audit metric distribution requires finite values")
    if not len(array):
        return {"count": 0, "mean": None}
    percentiles = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    result = {
        f"p{percentile:02d}": float(np.percentile(array, percentile))
        for percentile in percentiles
    }
    result.update({"count": int(len(array)), "mean": float(np.mean(array))})
    return result


def _audit_metadata_subset_matches(stored: object, current: object) -> bool:
    """Audit metadata subset matches."""
    if isinstance(stored, dict):
        return isinstance(current, dict) and all(
            key in current and _audit_metadata_subset_matches(value, current[key])
            for key, value in stored.items()
        )
    if isinstance(stored, list):
        return (
            isinstance(current, list)
            and len(stored) == len(current)
            and all(
                _audit_metadata_subset_matches(left, right)
                for left, right in zip(stored, current)
            )
        )
    return stored == current


def _load_audit_manifest_sequence(
    manifest_path: Union[str, Path],
    sequence_id: str,
    data_root: Union[str, Path],
) -> tuple[OmniSequence, str]:
    """Load audit manifest sequence."""
    path = Path(manifest_path).resolve()
    root = Path(data_root).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    accepted_versions = {
        "raw_omni_source_manifest_v1",
        "raw_omni_source_manifest_v2_xyz_flu",
    }
    if payload.get("version") not in accepted_versions:
        raise ValueError(
            f"unsupported audit source manifest: {payload.get('version')}"
        )
    matches = [
        value for value in payload.get("sequences", [])
        if value.get("sequence_id") == sequence_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"audit expected one sequence {sequence_id!r}, found {len(matches)}"
        )
    value = matches[0]
    pano_from_face = {
        face: np.asarray(rotation, dtype=np.float64)
        for face, rotation in value["pano_from_face"].items()
    }
    for face, rotation in pano_from_face.items():
        transform = np.eye(4, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"audit {face} pano rotation is not 3x3")
        transform[:3, :3] = rotation
        validate_transform(transform)

    def source_path(relative: str) -> Path:
        """Return source path."""
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError(f"audit source path escapes data root: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(f"audit source file does not exist: {resolved}")
        return resolved

    frames = tuple(
        OmniFrame(
            frame_id=str(frame["frame_id"]),
            frame_index=int(frame["frame_index"]),
            world_from_pano=validate_transform(
                np.asarray(frame["world_from_pano"], dtype=np.float64)
            ),
            rgb_paths={
                face: source_path(relative)
                for face, relative in frame["rgb_faces"].items()
            },
            depth_paths={
                face: source_path(relative)
                for face, relative in frame["depth_faces"].items()
            },
        )
        for frame in value.get("frames", [])
    )
    if not frames:
        raise ValueError(f"audit source sequence has no frames: {sequence_id}")
    return OmniSequence(
        dataset=str(payload["dataset"]),
        scene=str(value["scene"]),
        trajectory=str(value["trajectory"]),
        sequence_id=sequence_id,
        source_dir=root / value["scene"] / "Data_omni" / value["trajectory"],
        pano_from_face=pano_from_face,
        frames=frames,
        source_report=value.get("source_report") or {},
    ), str(value["split"])


def _audit_mask_outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Audit mask outline."""
    result = np.asarray(image).copy()
    binary = np.asarray(mask, dtype=np.uint8)
    if result.ndim != 3 or result.shape[2] != 3 or binary.shape != result.shape[:2]:
        raise ValueError("audit image and mask shapes do not match")
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _audit_scalar_panel(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Audit scalar panel."""
    field = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(field)
    if field.ndim != 2 or selected.shape != field.shape:
        raise ValueError("audit scalar field and mask shapes do not match")
    normalized = np.zeros(field.shape, dtype=np.uint8)
    if np.any(selected):
        finite_values = field[selected]
        lower, upper = np.percentile(finite_values, [1.0, 99.0])
        if upper <= lower:
            upper = lower + 1.0
        scaled = np.clip((field - lower) / (upper - lower), 0.0, 1.0)
        normalized[selected] = np.rint(255.0 * scaled[selected]).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~selected] = 0
    return _audit_mask_outline(color, mask)


def _audit_projection_panel(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depths_m: np.ndarray,
    in_image: np.ndarray,
    alpha_mask: np.ndarray,
    visible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Audit projection panel."""
    image = np.asarray(image_bgr)
    coordinates = np.asarray(uv, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    selected = np.asarray(in_image, dtype=bool) & np.isfinite(depths)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or coordinates.shape != (len(depths), 2)
        or selected.shape != depths.shape
        or np.asarray(alpha_mask).shape != image.shape[:2]
    ):
        raise ValueError("audit projection arrays do not align")
    output = np.rint(image.astype(np.float32) * 0.45).astype(np.uint8)
    if np.any(selected):
        selected_depths = depths[selected]
        lower, upper = np.percentile(selected_depths, [1.0, 99.0])
        if upper <= lower:
            upper = lower + 1.0
        normalized = np.clip((depths - lower) / (upper - lower), 0.0, 1.0)
        depth_u8 = np.rint(255.0 * normalized).astype(np.uint8)
        colors = cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
        colors = colors.reshape(-1, 3)
        pixel_x = np.floor(coordinates[:, 0]).astype(np.int64)
        pixel_y = np.floor(coordinates[:, 1]).astype(np.int64)
        height, width = image.shape[:2]
        selected &= (
            (pixel_x >= 0) & (pixel_x < width)
            & (pixel_y >= 0) & (pixel_y < height)
        )
        output[pixel_y[selected], pixel_x[selected]] = colors[selected]
        if visible is not None:
            visibility = np.asarray(visible, dtype=bool)
            if visibility.shape != depths.shape:
                raise ValueError("audit visibility array does not align")
            occluded = selected & ~visibility
            output[pixel_y[occluded], pixel_x[occluded]] = (0, 0, 255)
    return _audit_mask_outline(output, alpha_mask)


def _audit_direction_panel(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Audit direction panel."""
    edge_pixels = arrays["edge_pixels"]
    direction = np.mod(
        np.arctan2(arrays["gradient_y"], arrays["gradient_x"]), np.pi
    )
    hsv = np.zeros((*edge_pixels.shape, 3), dtype=np.uint8)
    hsv[..., 0][edge_pixels] = np.rint(
        direction[edge_pixels] * (179.0 / np.pi)
    ).astype(np.uint8)
    hsv[..., 1][edge_pixels] = 255
    hsv[..., 2][edge_pixels] = 255
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return _audit_mask_outline(color, arrays["mask"])


def _audit_titled_panel(
    image_bgr: np.ndarray,
    title: str,
    width: int = 800,
    height: int = 648,
) -> np.ndarray:
    """Audit titled panel."""
    resized = cv2.resize(
        np.asarray(image_bgr), (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((height + 34, width, 3), dtype=np.uint8)
    panel[34:] = resized
    cv2.putText(
        panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def _render_quality_audit_image(
    image_bgr: np.ndarray,
    point_cloud: np.ndarray,
    camera_depth: np.ndarray,
    camera_depth_unscaled: np.ndarray,
    sample: dict,
    nominal_extrinsic: np.ndarray,
    sensor: TrainingSensorConfig,
    config: GeneratedQualityConfig,
    quality_report: dict,
    reasons: List[str],
) -> np.ndarray:
    """Render quality audit image."""
    points = np.asarray(point_cloud)[:, :3]
    height, width = sensor.camera.height, sensor.camera.width
    true_extrinsic = validate_transform(
        np.asarray(sample["true_extrinsic"], dtype=np.float64)
    )
    nominal = validate_transform(np.asarray(nominal_extrinsic, dtype=np.float64))
    true_uv, true_depth, true_in = project_points(
        points, true_extrinsic, sensor.camera.matrix, height, width
    )
    true_visible = camera_projection_visibility(
        true_uv, true_depth, true_in, camera_depth, config
    )
    _, alpha_mask = _alpha_complex_projection_region(
        true_uv, true_in, height, width, config.alpha_radius_px, rasterize=True
    )
    if alpha_mask is None:
        raise RuntimeError("audit Ti Alpha Shape rasterization was not produced")
    nominal_uv, nominal_depth, nominal_in = project_points(
        points, nominal, sensor.camera.matrix, height, width
    )
    arrays = _depth_sobel_arrays(
        camera_depth_unscaled,
        alpha_mask,
        config.depth_sobel_gradient_threshold_m_per_px,
    )
    dense_depth = _audit_scalar_panel(
        np.log(np.maximum(camera_depth, 1e-12)),
        alpha_mask & np.isfinite(camera_depth),
    )
    magnitude = _audit_scalar_panel(
        np.log1p(
            arrays["magnitude"]
            / max(config.depth_sobel_gradient_threshold_m_per_px, 1e-12)
        ),
        arrays["gradient_pixels"],
    )
    panels = [
        _audit_titled_panel(image_bgr, "RGB"),
        _audit_titled_panel(dense_depth, "Dense GT depth in Ti Alpha Shape"),
        _audit_titled_panel(
            _audit_projection_panel(
                image_bgr, true_uv, true_depth, true_in, alpha_mask, true_visible
            ),
            "Ti LiDAR projection; occluded=red",
        ),
        _audit_titled_panel(
            _audit_projection_panel(
                image_bgr, nominal_uv, nominal_depth, nominal_in, alpha_mask
            ),
            "T0 LiDAR projection; Ti mask outline",
        ),
        _audit_titled_panel(
            magnitude, "Ti Alpha Shape unscaled-depth Sobel magnitude"
        ),
        _audit_titled_panel(
            _audit_direction_panel(arrays),
            "Ti edge-pixel Sobel direction; audit only",
        ),
    ]
    rows = [np.hstack(panels[:3]), np.hstack(panels[3:])]
    body = np.vstack(rows)
    sobel = quality_report["true_depth_sobel"]
    direction = sobel["direction_diversity"]
    log_value = sobel["edge_pixel_log10_ratio"]
    log_text = "null" if log_value is None else f"{log_value:.6f}"
    reason_text = "accepted" if not reasons else "|".join(reasons)
    lines = [
        (
            f"frame={sample['frame_id']} candidate={sample['candidate_id']} "
            f"pair={sample['pair_id']}"
        ),
        f"quality={reason_text}",
        (
            f"edge pixels={sobel['edge_pixel_count']}/"
            f"{sobel['edge_pixel_denominator']}  "
            f"ratio={100.0 * sobel['edge_pixel_ratio']:.6f}%  "
            f"log10(ratio)={log_text}"
        ),
        (
            f"edge ratio minimum="
            f"{100.0 * sobel['minimum_edge_pixel_ratio']:.6f}%  "
            f"ratio gate passed={sobel['passed']}  "
            f"direction gate=OFF reference={direction['reference_passed']} "
            f"by={direction['passed_by']}"
        ),
    ]
    header = np.zeros((170, body.shape[1], 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index != 1 or not reasons else (0, 100, 255)
        cv2.putText(
            header, line, (14, 36 + 39 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.82, color, 2, cv2.LINE_AA,
        )
    return np.vstack((header, body))


def _audit_sequence_chunk(
    sequence_directory: str,
    source_manifest_path: str,
    data_root: str,
    output_directory: str,
    sample_indices: list[int],
    nominal_extrinsic: np.ndarray,
    sensor: TrainingSensorConfig,
    config: GeneratedQualityConfig,
    check_nominal: bool,
    depth_cache_max_mb: int,
    resume: bool,
) -> list[dict]:
    """Audit sequence chunk."""
    sequence_path = Path(sequence_directory)
    output_path = Path(output_directory)
    with (sequence_path / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    samples = metadata["samples"]
    sequence_id = str(samples[0]["sequence_id"])
    sequence, split = _load_audit_manifest_sequence(
        source_manifest_path, sequence_id, data_root
    )
    if split != metadata.get("split"):
        raise ValueError("audit split differs from source manifest")
    frames_by_id = {frame.frame_id: frame for frame in sequence.frames}
    image_directory = output_path / "images"
    record_directory = output_path / "records"
    cv2.setNumThreads(1)
    configure_depth_cache(max(1, int(depth_cache_max_mb)) * 1024 * 1024)
    records = []
    started = perf_counter()
    try:
        for local_index, sample_index in enumerate(sample_indices):
            sample = samples[sample_index]
            frame_id = f"{sample_index:06d}"
            if sample.get("frame_id") != frame_id:
                raise ValueError(f"audit frame ordering mismatch at {sample_index}")
            image_path = image_directory / f"{frame_id}.jpg"
            record_path = record_directory / f"{frame_id}.json"
            if resume and image_path.is_file() and record_path.is_file():
                with record_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if (
                    record.get("frame_id") == frame_id
                    and record.get("candidate_id") == sample.get("candidate_id")
                    and image_path.stat().st_size > 0
                ):
                    records.append(record)
                    continue
            try:
                source = frames_by_id[str(sample["source_frame"])]
                target = frames_by_id[str(sample["target_frame"])]
            except KeyError as exc:
                raise ValueError(f"audit sample references unknown frame {exc}") from exc
            camera_stage = generate_candidate_camera_depth(
                source, target, sequence.pano_from_face, sample, sensor
            )
            cloud_values = np.fromfile(
                sequence_path / "velodyne" / f"{frame_id}.bin", dtype=np.float32
            )
            if not len(cloud_values) or len(cloud_values) % 4:
                raise ValueError(f"audit point cloud is not nonempty Nx4: {frame_id}")
            point_cloud = cloud_values.reshape(-1, 4)
            image_bgr = cv2.imread(
                str(sequence_path / "image_2" / f"{frame_id}.png"),
                cv2.IMREAD_COLOR,
            )
            if image_bgr is None:
                raise ValueError(f"audit image is unreadable: {frame_id}")
            report, reasons = evaluate_generated_projection_quality(
                point_cloud,
                camera_stage.camera_depth,
                camera_stage.camera_depth_unscaled,
                sample,
                nominal_extrinsic,
                sensor,
                config,
                check_nominal=check_nominal,
                stop_on_rejection=False,
            )
            audit_image = _render_quality_audit_image(
                image_bgr,
                point_cloud,
                camera_stage.camera_depth,
                camera_stage.camera_depth_unscaled,
                sample,
                nominal_extrinsic,
                sensor,
                config,
                report,
                reasons,
            )
            temporary_image = image_directory / f".{frame_id}.tmp.jpg"
            if not cv2.imwrite(
                str(temporary_image), audit_image,
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            ):
                raise OSError(f"failed to write audit image: {temporary_image}")
            temporary_image.replace(image_path)
            sobel = report["true_depth_sobel"]
            direction = sobel["direction_diversity"]
            record = {
                "index": sample_index,
                "frame_id": frame_id,
                "candidate_id": str(sample["candidate_id"]),
                "pair_id": str(sample["pair_id"]),
                "source_frame": str(sample["source_frame"]),
                "target_frame": str(sample["target_frame"]),
                "edge_pixel_count": int(sobel["edge_pixel_count"]),
                "edge_pixel_denominator": int(sobel["edge_pixel_denominator"]),
                "edge_pixel_ratio": float(sobel["edge_pixel_ratio"]),
                "edge_pixel_ratio_percent": float(
                    100.0 * sobel["edge_pixel_ratio"]
                ),
                "edge_pixel_log10_ratio": sobel["edge_pixel_log10_ratio"],
                "minimum_edge_pixel_ratio": float(
                    sobel["minimum_edge_pixel_ratio"]
                ),
                "edge_pixel_ratio_passed": bool(sobel["passed"]),
                "direction_diversity": direction["diversity"],
                "direction_reference_passed": direction["reference_passed"],
                "direction_passed_by": direction["passed_by"],
                "quality_passed": not reasons,
                "reasons": list(reasons),
                "image": f"images/{frame_id}.jpg",
                "image_width": int(audit_image.shape[1]),
                "image_height": int(audit_image.shape[0]),
                "image_bytes": int(image_path.stat().st_size),
            }
            _write_json_atomic(record_path, record)
            records.append(record)
            if (local_index + 1) % 25 == 0 or local_index + 1 == len(sample_indices):
                emit_runtime_log(
                    "stage_progress",
                    "Auditing Sobel quality for the existing sequence",
                    worker_samples_completed=local_index + 1,
                    worker_sample_count=len(sample_indices),
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
    finally:
        clear_depth_cache()
    records.sort(key=lambda value: value["index"])
    return records


def _write_audit_records_csv(path: Path, records: list[dict]) -> None:
    """Write audit records CSV."""
    fields = [
        "index", "frame_id", "candidate_id", "pair_id", "source_frame",
        "target_frame", "edge_pixel_count", "edge_pixel_denominator",
        "edge_pixel_ratio", "edge_pixel_ratio_percent",
        "edge_pixel_log10_ratio", "minimum_edge_pixel_ratio",
        "edge_pixel_ratio_passed",
        "direction_diversity", "direction_reference_passed",
        "direction_passed_by", "quality_passed", "reasons", "image",
        "image_width", "image_height", "image_bytes",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["reasons"] = "|".join(record["reasons"])
            writer.writerow({field: row[field] for field in fields})
    temporary.replace(path)


def _write_audit_quantile_contact_sheet(
    output_directory: Path,
    records: list[dict],
) -> list[dict]:
    """Write audit quantile contact sheet."""
    ratios = np.asarray(
        [record["edge_pixel_ratio"] for record in records], dtype=np.float64
    )
    quantiles = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    selections = []
    thumbnails = []
    for percentile in quantiles:
        target = float(np.percentile(ratios, percentile))
        index = int(np.argmin(np.abs(ratios - target)))
        record = records[index]
        image = cv2.imread(
            str(output_directory / record["image"]), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"unreadable audit image: {record['image']}")
        thumbnail_height = int(round(image.shape[0] * 480.0 / image.shape[1]))
        thumbnail = cv2.resize(
            image, (480, thumbnail_height), interpolation=cv2.INTER_AREA
        )
        cv2.rectangle(thumbnail, (0, 0), (479, 34), (0, 0, 0), -1)
        log_value = record["edge_pixel_log10_ratio"]
        log_text = "null" if log_value is None else f"{log_value:.6f}"
        cv2.putText(
            thumbnail,
            (
                f"P{percentile:02d} frame={record['frame_id']} "
                f"ratio={100.0 * record['edge_pixel_ratio']:.4f}% "
                f"log10={log_text}"
            ),
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        thumbnails.append(thumbnail)
        selections.append({
            "percentile": percentile,
            "frame_id": record["frame_id"],
            "edge_pixel_ratio": record["edge_pixel_ratio"],
            "edge_pixel_log10_ratio": record["edge_pixel_log10_ratio"],
            "image": record["image"],
        })
    blank = np.zeros_like(thumbnails[0])
    while len(thumbnails) % 3:
        thumbnails.append(blank.copy())
    rows = [
        np.hstack(thumbnails[index:index + 3])
        for index in range(0, len(thumbnails), 3)
    ]
    contact_sheet = np.vstack(rows)
    path = output_directory / "quantile_contact_sheet.jpg"
    temporary = output_directory / ".quantile_contact_sheet.tmp.jpg"
    if not cv2.imwrite(
        str(temporary), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]
    ):
        raise OSError(f"failed to write audit contact sheet: {temporary}")
    temporary.replace(path)
    return selections


def audit_existing_sequence_quality(
    task: GenerationTask,
    sequence_directory: Union[str, Path],
    output_directory: Union[str, Path],
    data_root: Union[str, Path],
    workers: int = 1,
    resume: bool = False,
) -> dict:
    """Audit existing sequence quality."""
    source = Path(sequence_directory).resolve()
    output = Path(output_directory).resolve()
    resolved_data_root = Path(data_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"audit sequence does not exist: {source}")
    if not (source / "metadata.json").is_file():
        raise FileNotFoundError(f"audit sequence lacks metadata.json: {source}")
    if int(workers) < 1:
        raise ValueError("audit workers must be positive")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("audit source and output directories must not be nested")
    with (source / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("audit sequence metadata has no samples")
    sample_count = len(samples)
    expected_frame_ids = {f"{index:06d}" for index in range(sample_count)}
    image_frame_ids = {path.stem for path in (source / "image_2").glob("*.png")}
    cloud_frame_ids = {path.stem for path in (source / "velodyne").glob("*.bin")}
    if image_frame_ids != expected_frame_ids or cloud_frame_ids != expected_frame_ids:
        raise ValueError(
            "audit sequence image, point-cloud, and metadata identities differ"
        )
    for filename in ("poses.txt", "residuals.txt"):
        rows = np.loadtxt(source / filename, dtype=np.float64)
        rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
        if rows.ndim != 2 or len(rows) != sample_count or rows.shape[1] != 12:
            raise ValueError(
                f"audit sequence {filename} count or 3x4 row shape differs"
            )
    exported_nominal = validate_transform(
        np.loadtxt(source / "nominal_extrinsic.txt", dtype=np.float64)
    )
    if not np.allclose(
        exported_nominal, task.nominal.matrix, rtol=0.0, atol=1e-12
    ):
        raise ValueError("audit sequence nominal extrinsic differs from task T0")
    sensor = load_training_sensor_config(task.sensor_config)
    normalized_sensor = json.loads(json.dumps(sensor.to_dict(), allow_nan=False))
    if not _audit_metadata_subset_matches(
        metadata.get("sensor_setup"), normalized_sensor
    ):
        raise ValueError("audit sequence sensor metadata differs from task config")
    quality = load_generated_quality_config(task.quality_config)
    source_manifest_value = metadata.get("source_manifest")
    if not isinstance(source_manifest_value, str) or not source_manifest_value:
        raise ValueError("audit sequence metadata lacks source_manifest")
    source_manifest = Path(source_manifest_value)
    if not source_manifest.is_absolute():
        source_manifest = (source / source_manifest).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"audit source manifest does not exist: {source_manifest}")
    metadata_fingerprint = hashlib.sha256(
        (source / "metadata.json").read_bytes()
    ).hexdigest()
    identity = {
        "version": "existing_sequence_sobel_edge_ratio_audit_v2",
        "quality_version": GENERATED_QUALITY_VERSION,
        "source_sequence": str(source),
        "source_metadata_sha256": metadata_fingerprint,
        "sample_count": sample_count,
        "sensor": normalized_sensor,
        "quality_config": quality.to_dict(),
        "nominal_extrinsic": task.nominal.matrix.tolist(),
        "edge_pixel_semantic": (
            "Ti_alpha_shape_finite_3x3_depth_neighborhood_and_"
            "sobel_magnitude_above_threshold"
        ),
        "edge_pixel_log10_zero_semantic": "null",
        "edge_pixel_ratio_gate_enabled": True,
        "minimum_edge_pixel_ratio": quality.min_true_sobel_edge_pixel_ratio,
        "direction_gate_enabled": False,
    }
    identity_path = output / "audit_identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"audit output already exists: {output}")
        if not identity_path.is_file():
            raise ValueError("audit resume output lacks audit_identity.json")
        with identity_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != identity:
                raise ValueError("audit resume identity differs from current inputs")
    else:
        output.mkdir(parents=True)
        (output / "images").mkdir()
        (output / "records").mkdir()
        _write_json_atomic(identity_path, identity)
    if not (output / "images").is_dir() or not (output / "records").is_dir():
        raise ValueError("audit output lacks images or records directory")
    check_nominal = metadata.get("true_extrinsic_policy") != "Ti_equals_T0"
    indices = np.array_split(np.arange(sample_count, dtype=np.int64), int(workers))
    chunks = [chunk.astype(int).tolist() for chunk in indices if len(chunk)]
    cache_per_worker_mb = max(
        1, int(task.runtime.depth_cache_max_mb) // max(1, len(chunks))
    )
    arguments = [
        (
            str(source), str(source_manifest), str(resolved_data_root),
            str(output), chunk, task.nominal.matrix, sensor, quality,
            check_nominal, cache_per_worker_mb, bool(resume),
        )
        for chunk in chunks
    ]
    records = []
    if len(arguments) == 1:
        records.extend(_audit_sequence_chunk(*arguments[0]))
    else:
        with ProcessPoolExecutor(max_workers=len(arguments)) as executor:
            futures = [
                executor.submit(_audit_sequence_chunk, *argument)
                for argument in arguments
            ]
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda value: value["index"])
    if len(records) != sample_count or [
        record["index"] for record in records
    ] != list(range(sample_count)):
        raise RuntimeError("audit records do not cover every exported sample")
    image_files = sorted((output / "images").glob("*.jpg"))
    if len(image_files) != sample_count or any(
        path.stat().st_size <= 0 for path in image_files
    ):
        raise RuntimeError("audit images do not cover every exported sample")
    ratios = [record["edge_pixel_ratio"] for record in records]
    log_values = [
        record["edge_pixel_log10_ratio"] for record in records
        if record["edge_pixel_log10_ratio"] is not None
    ]
    cutoff_values = (
        -4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0,
        -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
    )
    threshold_sweep = [
        {
            "minimum_log10_ratio": cutoff,
            "equivalent_ratio": float(10.0 ** cutoff),
            "equivalent_percent": float(100.0 * 10.0 ** cutoff),
            "rejected_count": int(sum(
                record["edge_pixel_log10_ratio"] is None
                or record["edge_pixel_log10_ratio"] < cutoff
                for record in records
            )),
        }
        for cutoff in cutoff_values
    ]
    _write_audit_records_csv(output / "samples.csv", records)
    _write_json_atomic(output / "samples.json", records)
    quantile_samples = _write_audit_quantile_contact_sheet(output, records)
    summary = {
        **identity,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "image_count": len(image_files),
        "image_total_bytes": int(sum(path.stat().st_size for path in image_files)),
        "zero_edge_sample_count": int(sum(value == 0.0 for value in ratios)),
        "quality_passed_count_with_edge_pixel_ratio_gate": int(sum(
            record["quality_passed"] for record in records
        )),
        "edge_pixel_ratio_distribution": _audit_metric_distribution(ratios),
        "edge_pixel_log10_ratio_distribution": _audit_metric_distribution(log_values),
        "threshold_sweep": threshold_sweep,
        "quantile_samples": quantile_samples,
        "samples_csv": "samples.csv",
        "samples_json": "samples.json",
        "contact_sheet": "quantile_contact_sheet.jpg",
    }
    _write_json_atomic(output / "summary.json", summary)
    return {
        "version": identity["version"],
        "status": "complete",
        "source_sequence": str(source),
        "output_root": str(output),
        "sample_count": len(records),
        "image_count": len(image_files),
        "summary": str(output / "summary.json"),
        "contact_sheet": str(output / "quantile_contact_sheet.jpg"),
    }
