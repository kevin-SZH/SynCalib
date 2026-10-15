"""Generate physically consistent camera-LiDAR sample candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation_config import GenerationLevel, ResidualSamplerConfig
    from .quality_export import CoverageConfig


from collections import Counter, defaultdict
from typing import Iterable

import numpy as np


JOINT_COVERAGE_VERSION = "conditional_t_r_coverage_v2"


def fibonacci_directions(count: int) -> np.ndarray:
    """Return fibonacci directions."""
    if count < 4:
        raise ValueError("direction count must be at least four")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    values = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        angle = index * golden_angle
        values.append([
            radius * np.cos(angle), radius * np.sin(angle), z
        ])
    return np.asarray(values, dtype=np.float64)


def translation_cell_id(
    normalized_translation: np.ndarray,
    config: CoverageConfig,
    directions: np.ndarray | None = None,
) -> str:
    """Return translation cell ID."""
    value = np.asarray(normalized_translation, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("normalized translation must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm > config.translation_shell_edges[-1] + 1e-6:
        raise ValueError(f"normalized translation lies outside configured ball: {norm}")
    if norm <= config.translation_center_radius:
        return "t_center"
    shell = int(np.searchsorted(
        config.translation_shell_edges[1:], norm, side="right"
    ))
    shell = min(shell, len(config.translation_shell_edges) - 2)
    if directions is None:
        directions = fibonacci_directions(config.translation_direction_cells)
    direction = int(np.argmax(directions @ (value / norm)))
    return f"t_s{shell:02d}_d{direction:03d}"


def theoretical_translation_cells(config: CoverageConfig) -> list[str]:
    """Return theoretical translation cells."""
    directional_cells = [
        f"t_s{shell:02d}_d{direction:03d}"
        for shell in range(1, len(config.translation_shell_edges) - 1)
        for direction in range(config.translation_direction_cells)
    ]
    return ["t_center", *directional_cells]


def audit_joint_coverage(
    candidates: Iterable[dict],
    config: CoverageConfig,
    required_rotation_ids: Iterable[str] | None = None,
) -> dict:
    """Audit joint coverage."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("joint coverage requires at least one candidate")
    directions = fibonacci_directions(config.translation_direction_cells)
    cells = theoretical_translation_cells(config)
    candidate_counts = Counter()
    cell_pairs = defaultdict(set)
    cell_rotations = defaultdict(set)
    cell_rotation_axes = defaultdict(set)
    cell_rotation_shells = defaultdict(set)
    rotation_translation_cells = defaultdict(set)
    matrix = defaultdict(Counter)
    center_candidates = 0

    for candidate in candidates:
        normalized = np.asarray(candidate["normalized"], dtype=np.float64)
        if normalized.shape != (6,):
            raise ValueError("candidate normalized coordinate is not length six")
        cell = translation_cell_id(normalized[:3], config, directions)
        if cell == "t_center":
            center_candidates += 1
        rotation_id = str(candidate["rotation_id"])
        candidate_counts[cell] += 1
        cell_pairs[cell].add(str(candidate["pair_id"]))
        cell_rotations[cell].add(rotation_id)
        axis = int(candidate["rotation_axis_index"])
        if axis >= 0:
            cell_rotation_axes[cell].add(axis)
        cell_rotation_shells[cell].add(int(candidate["rotation_shell_index"]))
        rotation_translation_cells[rotation_id].add(cell)
        matrix[cell][rotation_id] += 1

    occupied = [cell for cell in cells if candidate_counts[cell] > 0]
    empty = [cell for cell in cells if candidate_counts[cell] == 0]
    occupancy = float(len(occupied) / len(cells))
    failures = []
    if occupancy < config.min_translation_cell_occupancy:
        failures.append(
            f"translation cell occupancy {occupancy:.6f} < "
            f"{config.min_translation_cell_occupancy:.6f}"
        )
    cell_reports = {}
    for cell in cells:
        report = {
            "candidate_count": int(candidate_counts[cell]),
            "pair_count": len(cell_pairs[cell]),
            "rotation_cell_count": len(cell_rotations[cell]),
            "rotation_axis_count": len(cell_rotation_axes[cell]),
            "rotation_shell_count": len(cell_rotation_shells[cell]),
        }
        cell_reports[cell] = report
        if not report["candidate_count"]:
            continue
        checks = (
            ("candidate_count", config.min_candidates_per_translation_cell),
            ("pair_count", config.min_pairs_per_translation_cell),
            ("rotation_cell_count", config.min_rotation_cells_per_translation_cell),
            ("rotation_axis_count", config.min_rotation_axes_per_translation_cell),
            ("rotation_shell_count", config.min_rotation_shells_per_translation_cell),
        )
        for name, minimum in checks:
            if report[name] < minimum:
                failures.append(f"{cell}: {name} {report[name]} < {minimum}")

    observed_rotation_ids = set(rotation_translation_cells)
    required_rotation_ids = (
        observed_rotation_ids
        if required_rotation_ids is None
        else {str(value) for value in required_rotation_ids}
    )
    missing_rotation_ids = sorted(
        required_rotation_ids - observed_rotation_ids
    )
    rotation_reports = {}
    for rotation_id in sorted(required_rotation_ids):
        count = len(rotation_translation_cells[rotation_id])
        rotation_reports[rotation_id] = {"translation_cell_count": count}
        if rotation_id in missing_rotation_ids:
            failures.append(f"{rotation_id}: required rotation cell is missing")
        if count < config.min_translation_cells_per_rotation_cell:
            failures.append(
                f"{rotation_id}: translation_cell_count {count} < "
                f"{config.min_translation_cells_per_rotation_cell}"
            )
    occupied_reports = [cell_reports[cell] for cell in occupied]
    return {
        "version": JOINT_COVERAGE_VERSION,
        "passed": not failures,
        "failures": failures,
        "candidate_count": len(candidates),
        "translation_cell_count": len(cells),
        "occupied_translation_cells": len(occupied),
        "translation_cell_occupancy": occupancy,
        "empty_translation_cells": empty,
        "center_candidate_count": center_candidates,
        "required_rotation_cell_count": len(required_rotation_ids),
        "observed_rotation_cell_count": len(observed_rotation_ids),
        "missing_rotation_cells": missing_rotation_ids,
        "conditional_minima": {
            "candidates_per_occupied_translation_cell": min(
                value["candidate_count"] for value in occupied_reports
            ) if occupied_reports else 0,
            "pairs_per_occupied_translation_cell": min(
                value["pair_count"] for value in occupied_reports
            ) if occupied_reports else 0,
            "rotation_cells_per_occupied_translation_cell": min(
                value["rotation_cell_count"] for value in occupied_reports
            ) if occupied_reports else 0,
            "rotation_axes_per_occupied_translation_cell": min(
                value["rotation_axis_count"] for value in occupied_reports
            ) if occupied_reports else 0,
            "rotation_shells_per_occupied_translation_cell": min(
                value["rotation_shell_count"] for value in occupied_reports
            ) if occupied_reports else 0,
            "translation_cells_per_rotation_cell": min(
                value["translation_cell_count"] for value in rotation_reports.values()
            ) if rotation_reports else 0,
        },
        "translation_cells": cell_reports,
        "rotation_cells": rotation_reports,
        "support_matrix": {
            cell: dict(sorted(counts.items()))
            for cell, counts in sorted(matrix.items())
        },
    }


from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


TRANSLATION_RADIUS_DISTRIBUTION = (
    "equal_accepted_per_normalized_radius_interval"
)
TRANSLATION_INTERVAL_SAMPLING = "stratified_uniform_radius_uniform_s2"
TRANSLATION_RADIUS_BALANCING_VERSION = (
    "equal_accepted_translation_radius_intervals_v1"
)
JOINT_CELL_DISTRIBUTION = (
    "equal_accepted_per_translation_rotation_radius_pair_deficit_cells"
)
JOINT_CELL_SAMPLING = (
    "stratified_uniform_radius_conditioned_fibonacci_direction_cell"
)
LEGACY_JOINT_CELL_BALANCING_VERSION = (
    "hierarchical_joint_accepted_cell_quota_v1"
)
JOINT_CELL_BALANCING_VERSION = (
    "hierarchical_joint_accepted_cell_quota_v2"
)


def _validate_edges(edges: Sequence[float]) -> tuple[float, ...]:
    """Validate edges."""
    values = tuple(float(value) for value in edges)
    if (
        len(values) < 2
        or values[0] != 0.0
        or tuple(sorted(set(values))) != values
        or values[-1] < 1.0
    ):
        raise ValueError(
            "translation radius edges must increase uniquely from 0 through >=1"
        )
    return values


def equal_interval_targets(total: int, interval_count: int) -> tuple[int, ...]:
    """Return equal interval targets."""
    if total < 1 or interval_count < 1:
        raise ValueError("total and interval_count must be positive")
    base, remainder = divmod(int(total), int(interval_count))
    return tuple(
        base + int(index < remainder) for index in range(interval_count)
    )


@dataclass
class EqualAcceptedTranslationRadiusQuota:
    """Represent equal accepted translation radius quota."""

    edges: tuple[float, ...]
    targets: tuple[int, ...]
    accepted: list[int] = field(default_factory=list)
    quota_full_skips: list[int] = field(default_factory=list)
    _next_interval_cursor: int = field(default=0, repr=False)

    @classmethod
    def create(
        cls,
        accepted_target: int,
        edges: Sequence[float],
    ) -> "EqualAcceptedTranslationRadiusQuota":
        """Create."""
        validated_edges = _validate_edges(edges)
        interval_count = len(validated_edges) - 1
        return cls(
            edges=validated_edges,
            targets=equal_interval_targets(accepted_target, interval_count),
            accepted=[0] * interval_count,
            quota_full_skips=[0] * interval_count,
        )

    def __post_init__(self) -> None:
        """Validate the initialized instance."""
        self.edges = _validate_edges(self.edges)
        interval_count = len(self.edges) - 1
        if len(self.targets) != interval_count or any(
            target < 0 for target in self.targets
        ):
            raise ValueError("translation-radius targets do not match intervals")
        if not self.accepted:
            self.accepted = [0] * interval_count
        if not self.quota_full_skips:
            self.quota_full_skips = [0] * interval_count
        if (
            len(self.accepted) != interval_count
            or len(self.quota_full_skips) != interval_count
        ):
            raise ValueError("translation-radius counters do not match intervals")

    def interval_index(self, normalized_translation: Sequence[float]) -> int:
        """Return interval index."""
        value = np.asarray(normalized_translation, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "normalized translation must be a finite three-vector"
            )
        radius = float(np.linalg.norm(value))
        if radius > self.edges[-1] + 1e-6:
            raise ValueError(
                f"normalized translation lies outside configured ball: {radius}"
            )
        if radius <= self.edges[1]:
            return 0
        index = int(np.searchsorted(
            self.edges[1:], radius, side="right"
        ))
        return min(index, len(self.targets) - 1)

    def is_full(self, index: int) -> bool:
        """Return is full."""
        return self.accepted[index] >= self.targets[index]

    def next_open_interval(self) -> int:
        """Return next open interval."""
        for offset in range(len(self.targets)):
            index = (self._next_interval_cursor + offset) % len(self.targets)
            if not self.is_full(index):
                self._next_interval_cursor = (index + 1) % len(self.targets)
                return index
        raise ValueError("all translation-radius interval quotas are full")

    def interval_bounds(self, index: int) -> tuple[float, float]:
        """Return interval bounds."""
        if not 0 <= index < len(self.targets):
            raise IndexError("translation-radius interval index is out of range")
        return self.edges[index], min(self.edges[index + 1], 1.0)

    def record_quota_full_skip(self, index: int) -> None:
        """Record quota full skip."""
        if not self.is_full(index):
            raise ValueError("cannot skip an interval whose quota is not full")
        self.quota_full_skips[index] += 1

    def record_acceptance(self, index: int) -> None:
        """Record acceptance."""
        if self.is_full(index):
            raise ValueError("translation-radius interval quota overflow")
        self.accepted[index] += 1

    @property
    def complete(self) -> bool:
        """Return complete."""
        return tuple(self.accepted) == self.targets

    def report(self) -> dict:
        """Return report."""
        intervals = []
        for index, (lower, upper, target, accepted, skipped) in enumerate(zip(
            self.edges[:-1],
            self.edges[1:],
            self.targets,
            self.accepted,
            self.quota_full_skips,
        )):
            intervals.append({
                "id": f"translation_radius_{index:02d}",
                "lower": float(lower),
                "upper": float(upper),
                "target_count": int(target),
                "accepted_count": int(accepted),
                "quota_full_skip_count": int(skipped),
            })
        return {
            "version": TRANSLATION_RADIUS_BALANCING_VERSION,
            "distribution": TRANSLATION_RADIUS_DISTRIBUTION,
            "balanced": self.complete,
            "target_count": int(sum(self.targets)),
            "accepted_count": int(sum(self.accepted)),
            "quota_full_skip_count": int(sum(self.quota_full_skips)),
            "maximum_interval_count_difference": int(
                max(self.accepted) - min(self.accepted)
            ),
            "intervals": intervals,
        }


@dataclass(frozen=True)
class JointCellTarget:
    """Represent joint cell target."""

    index: int
    translation_interval_index: int
    translation_direction_index: int
    rotation_interval_index: int
    rotation_direction_index: int
    target_count: int
    accepted_count: int
    cell_id: str
    translation_radius_cell_index: int = -1
    rotation_radius_cell_index: int = -1
    radius_pair_target_count: int = 0
    radius_pair_accepted_count: int = 0


def _balanced_radius_pair_targets(
    total: int,
    interval_count: int,
    seed: int,
) -> np.ndarray:
    """Return balanced radius pair targets."""
    row_targets = equal_interval_targets(total, interval_count)
    column_targets = equal_interval_targets(total, interval_count)
    base = total // (interval_count * interval_count)
    matrix = np.full(
        (interval_count, interval_count), base, dtype=np.int64
    )
    row_remaining = np.asarray(row_targets, dtype=np.int64) - (
        base * interval_count
    )
    column_remaining = np.asarray(column_targets, dtype=np.int64) - (
        base * interval_count
    )
    for row in range(interval_count):
        used_columns = set()
        for offset in range(int(row_remaining[row])):
            candidates = [
                column for column in range(interval_count)
                if column_remaining[column] > 0
                and column not in used_columns
            ]
            if not candidates:
                raise RuntimeError(
                    "cannot construct balanced radius-pair targets"
                )
            origin = (int(seed) + row + offset) % interval_count
            column = min(
                candidates,
                key=lambda value: (
                    -int(column_remaining[value]),
                    (value - origin) % interval_count,
                ),
            )
            matrix[row, column] += 1
            column_remaining[column] -= 1
            used_columns.add(column)
    if (
        np.any(column_remaining)
        or tuple(matrix.sum(axis=1)) != row_targets
        or tuple(matrix.sum(axis=0)) != column_targets
    ):
        raise RuntimeError("balanced radius-pair target construction failed")
    return matrix


def _rotated_equal_targets(
    total: int,
    count: int,
    offset: int,
) -> tuple[int, ...]:
    """Return rotated equal targets."""
    base, remainder = divmod(int(total), int(count))
    values = [base] * count
    for index in range(remainder):
        values[(offset + index) % count] += 1
    return tuple(values)


def _balanced_matrix_with_margins(
    row_targets: Sequence[int],
    column_targets: Sequence[int],
    seed: int,
) -> np.ndarray:
    """Return balanced matrix with margins."""
    rows = tuple(int(value) for value in row_targets)
    columns = tuple(int(value) for value in column_targets)
    if (
        not rows
        or not columns
        or any(value < 0 for value in (*rows, *columns))
        or sum(rows) != sum(columns)
    ):
        raise ValueError("balanced matrix margins are invalid")
    total = sum(rows)
    base = total // (len(rows) * len(columns))
    matrix = np.full((len(rows), len(columns)), base, dtype=np.int64)
    row_remaining = np.asarray(rows, dtype=np.int64) - base * len(columns)
    column_remaining = (
        np.asarray(columns, dtype=np.int64) - base * len(rows)
    )
    if (
        np.any(row_remaining < 0)
        or np.any(row_remaining > len(columns))
        or np.any(column_remaining < 0)
        or np.any(column_remaining > len(rows))
    ):
        raise ValueError("balanced matrix margins require spread above one")
    row_order = sorted(
        range(len(rows)),
        key=lambda row: (
            -int(row_remaining[row]),
            (row - int(seed)) % len(rows),
        ),
    )
    for row in row_order:
        used_columns = set()
        for offset in range(int(row_remaining[row])):
            candidates = [
                column for column in range(len(columns))
                if column_remaining[column] > 0
                and column not in used_columns
            ]
            if not candidates:
                raise RuntimeError("cannot satisfy balanced matrix margins")
            origin = (int(seed) + 17 * row + offset) % len(columns)
            column = min(
                candidates,
                key=lambda value: (
                    -int(column_remaining[value]),
                    (value - origin) % len(columns),
                ),
            )
            matrix[row, column] += 1
            column_remaining[column] -= 1
            used_columns.add(column)
    if (
        np.any(column_remaining)
        or tuple(matrix.sum(axis=1)) != rows
        or tuple(matrix.sum(axis=0)) != columns
    ):
        raise RuntimeError("balanced matrix construction failed")
    return matrix


def _hierarchical_fine_radius_targets(
    total: int,
    interval_count: int,
    subinterval_count: int,
    partition_index: int,
    partition_count: int,
) -> np.ndarray:
    """Return hierarchical fine radius targets."""
    outer = _balanced_radius_pair_targets(total, interval_count, 0)
    block_row_sums = np.zeros(
        (interval_count, interval_count, subinterval_count), dtype=np.int64
    )
    block_column_sums = np.zeros_like(block_row_sums)
    for outer_row in range(interval_count):
        allocation = _balanced_matrix_with_margins(
            equal_interval_targets(
                int(outer[outer_row].sum()), subinterval_count
            ),
            tuple(int(value) for value in outer[outer_row]),
            101 * outer_row,
        )
        for outer_column in range(interval_count):
            block_row_sums[outer_row, outer_column] = allocation[
                :, outer_column
            ]
    for outer_column in range(interval_count):
        allocation = _balanced_matrix_with_margins(
            tuple(int(value) for value in outer[:, outer_column]),
            equal_interval_targets(
                int(outer[:, outer_column].sum()), subinterval_count
            ),
            211 * outer_column,
        )
        for outer_row in range(interval_count):
            block_column_sums[outer_row, outer_column] = allocation[
                outer_row
            ]
    size = interval_count * subinterval_count
    fine = np.zeros((size, size), dtype=np.int64)
    for outer_row in range(interval_count):
        for outer_column in range(interval_count):
            block = _balanced_matrix_with_margins(
                block_row_sums[outer_row, outer_column],
                block_column_sums[outer_row, outer_column],
                307 * outer_row + 401 * outer_column,
            )
            row_start = outer_row * subinterval_count
            column_start = outer_column * subinterval_count
            fine[
                row_start:row_start + subinterval_count,
                column_start:column_start + subinterval_count,
            ] = block
    half_turn = subinterval_count // 2
    if partition_count == 2:
        row_shift = column_shift = partition_index * half_turn
    else:
        row_shift = (partition_index // 2) * half_turn
        column_shift = (partition_index % 2) * half_turn
    grouped_indices = np.arange(size).reshape(
        interval_count, subinterval_count
    )
    row_indices = np.roll(
        grouped_indices, row_shift, axis=1
    ).reshape(-1)
    column_indices = np.roll(
        grouped_indices, column_shift, axis=1
    ).reshape(-1)
    fine = fine[row_indices][:, column_indices]
    if int(fine.sum()) != total or int(fine.max() - fine.min()) > 1:
        raise RuntimeError("fine radius target construction failed")
    return fine


def _direction_indices(interval_index: int, count: int) -> tuple[int, ...]:
    """Return direction indices."""
    return (-1,) if interval_index == 0 else tuple(range(count))


def _coverage_cell_id(
    prefix: str,
    interval_index: int,
    direction_index: int,
) -> str:
    """Return coverage cell ID."""
    if interval_index == 0:
        return f"{prefix}_center"
    return f"{prefix}_s{interval_index:02d}_d{direction_index:03d}"


@dataclass
class JointAcceptedCellQuota:
    """Represent joint accepted cell quota."""

    edges: tuple[float, ...]
    direction_count: int
    targets: tuple[int, ...]
    cells: tuple[tuple[int, int, int, int], ...]
    max_consecutive_failures: int
    seed: int
    radius_subinterval_count: int
    radius_partition_index: int
    radius_partition_count: int
    fine_radius_targets: tuple[tuple[int, ...], ...]
    accepted: list[int] = field(default_factory=list)
    failed_attempts: list[int] = field(default_factory=list)
    releases: list[int] = field(default_factory=list)
    fine_radius_accepted: list[list[int]] = field(default_factory=list)
    _next_cell_cursor: int = field(default=0, repr=False)
    _pending_cell_index: int = field(default=-1, repr=False)
    _pending_failures: int = field(default=0, repr=False)
    _pending_translation_radius_cell: int = field(default=-1, repr=False)
    _pending_rotation_radius_cell: int = field(default=-1, repr=False)

    @classmethod
    def create(
        cls,
        accepted_target: int,
        edges: Sequence[float],
        direction_count: int,
        max_consecutive_failures: int,
        seed: int,
        radius_subinterval_count: int = 1,
        radius_partition_index: int = 0,
        radius_partition_count: int = 1,
    ) -> "JointAcceptedCellQuota":
        """Create."""
        validated_edges = _validate_edges(edges)
        interval_count = len(validated_edges) - 1
        if direction_count < 4:
            raise ValueError("joint direction count must be at least four")
        if max_consecutive_failures < 1:
            raise ValueError(
                "joint max consecutive failures must be positive"
            )
        if radius_subinterval_count < 1:
            raise ValueError(
                "joint radius subinterval count must be positive"
            )
        if (
            radius_partition_count < 1
            or not 0 <= radius_partition_index < radius_partition_count
        ):
            raise ValueError("joint radius partition is invalid")
        if (
            radius_subinterval_count > 1
            and (
                radius_subinterval_count != 4
                or radius_partition_count > radius_subinterval_count
            )
        ):
            raise ValueError(
                "fine joint quotas require four subintervals and at most "
                "four scene partitions"
            )
        if radius_subinterval_count == 1:
            radius_targets = _balanced_radius_pair_targets(
                accepted_target, interval_count, seed
            )
            fine_targets = radius_targets.copy()
        else:
            fine_targets = _hierarchical_fine_radius_targets(
                accepted_target,
                interval_count,
                radius_subinterval_count,
                radius_partition_index,
                radius_partition_count,
            )
            radius_targets = fine_targets.reshape(
                interval_count,
                radius_subinterval_count,
                interval_count,
                radius_subinterval_count,
            ).sum(axis=(1, 3))
        cells = []
        targets = []
        for translation_interval in range(interval_count):
            translation_directions = _direction_indices(
                translation_interval, direction_count
            )
            for rotation_interval in range(interval_count):
                rotation_directions = _direction_indices(
                    rotation_interval, direction_count
                )
                pair_total = int(
                    radius_targets[
                        translation_interval, rotation_interval
                    ]
                )
                if radius_subinterval_count == 1:
                    combinations = [
                        (
                            translation_interval,
                            translation_direction,
                            rotation_interval,
                            rotation_direction,
                        )
                        for translation_direction in translation_directions
                        for rotation_direction in rotation_directions
                    ]
                    offset = (
                        int(seed)
                        + 17 * translation_interval
                        + 31 * rotation_interval
                    ) % len(combinations)
                    local_targets = _rotated_equal_targets(
                        pair_total, len(combinations), offset
                    )
                    target_rows = zip(combinations, local_targets)
                else:
                    direction_targets = _balanced_matrix_with_margins(
                        equal_interval_targets(
                            pair_total, len(translation_directions)
                        ),
                        equal_interval_targets(
                            pair_total, len(rotation_directions)
                        ),
                        seed
                        + 17 * translation_interval
                        + 31 * rotation_interval,
                    )
                    target_rows = (
                        (
                            (
                                translation_interval,
                                translation_direction,
                                rotation_interval,
                                rotation_direction,
                            ),
                            int(direction_targets[t_index, r_index]),
                        )
                        for t_index, translation_direction in enumerate(
                            translation_directions
                        )
                        for r_index, rotation_direction in enumerate(
                            rotation_directions
                        )
                    )
                for combination, target in target_rows:
                    if target:
                        cells.append(combination)
                        targets.append(target)
        fine_size = interval_count * radius_subinterval_count
        return cls(
            edges=validated_edges,
            direction_count=int(direction_count),
            targets=tuple(targets),
            cells=tuple(cells),
            max_consecutive_failures=int(max_consecutive_failures),
            seed=int(seed),
            radius_subinterval_count=int(radius_subinterval_count),
            radius_partition_index=int(radius_partition_index),
            radius_partition_count=int(radius_partition_count),
            fine_radius_targets=tuple(
                tuple(int(value) for value in row) for row in fine_targets
            ),
            accepted=[0] * len(cells),
            failed_attempts=[0] * len(cells),
            releases=[0] * len(cells),
            fine_radius_accepted=[
                [0] * fine_size for _ in range(fine_size)
            ],
        )

    def __post_init__(self) -> None:
        """Validate the initialized instance."""
        self.edges = _validate_edges(self.edges)
        if len(self.cells) != len(self.targets) or not self.cells:
            raise ValueError(
                "joint cells and targets must be nonempty and align"
            )
        if any(target < 1 for target in self.targets):
            raise ValueError("joint cell targets must be positive")
        for values in (self.accepted, self.failed_attempts, self.releases):
            if len(values) != len(self.cells):
                raise ValueError("joint cell counters do not align")
        fine_size = (
            (len(self.edges) - 1) * self.radius_subinterval_count
        )
        fine_targets = np.asarray(self.fine_radius_targets, dtype=np.int64)
        fine_accepted = np.asarray(self.fine_radius_accepted, dtype=np.int64)
        if (
            self.radius_subinterval_count < 1
            or self.radius_partition_count < 1
            or not 0 <= self.radius_partition_index < self.radius_partition_count
            or fine_targets.shape != (fine_size, fine_size)
            or fine_accepted.shape != (fine_size, fine_size)
            or np.any(fine_targets < 0)
            or np.any(fine_accepted < 0)
            or np.any(fine_accepted > fine_targets)
            or int(fine_targets.sum()) != sum(self.targets)
        ):
            raise ValueError("fine radius quota matrices are invalid")
        self._cell_indices = {
            cell: index for index, cell in enumerate(self.cells)
        }
        self._directions = fibonacci_directions(self.direction_count)
        self._fine_cursors = {
            (translation_interval, rotation_interval): 0
            for translation_interval in range(len(self.edges) - 1)
            for rotation_interval in range(len(self.edges) - 1)
        }

    def interval_bounds(self, index: int) -> tuple[float, float]:
        """Return interval bounds."""
        if not 0 <= index < len(self.edges) - 1:
            raise IndexError("joint radius interval index is out of range")
        return self.edges[index], min(self.edges[index + 1], 1.0)

    @property
    def fine_radius_cell_count(self) -> int:
        """Return fine radius cell count."""
        return (len(self.edges) - 1) * self.radius_subinterval_count

    def fine_radius_bounds(self, index: int) -> tuple[float, float]:
        """Return fine radius bounds."""
        if not 0 <= index < self.fine_radius_cell_count:
            raise IndexError("fine radius cell index is out of range")
        outer_index, local_index = divmod(
            index, self.radius_subinterval_count
        )
        lower, upper = self.interval_bounds(outer_index)
        width = (upper - lower) / self.radius_subinterval_count
        return (
            lower + local_index * width,
            lower + (local_index + 1) * width,
        )

    def _target(self, index: int) -> JointCellTarget:
        """Return target."""
        t_interval, t_direction, r_interval, r_direction = self.cells[index]
        translation_cell = _coverage_cell_id(
            "t", t_interval, t_direction
        )
        rotation_cell = _coverage_cell_id(
            "r", r_interval, r_direction
        )
        pending = index == self._pending_cell_index
        translation_radius_cell = (
            self._pending_translation_radius_cell if pending else -1
        )
        rotation_radius_cell = (
            self._pending_rotation_radius_cell if pending else -1
        )
        fine_target = (
            int(self.fine_radius_targets[
                translation_radius_cell
            ][rotation_radius_cell])
            if pending else 0
        )
        fine_accepted = (
            int(self.fine_radius_accepted[
                translation_radius_cell
            ][rotation_radius_cell])
            if pending else 0
        )
        return JointCellTarget(
            index=index,
            translation_interval_index=t_interval,
            translation_direction_index=t_direction,
            rotation_interval_index=r_interval,
            rotation_direction_index=r_direction,
            target_count=self.targets[index],
            accepted_count=self.accepted[index],
            cell_id=f"{translation_cell}|{rotation_cell}",
            translation_radius_cell_index=translation_radius_cell,
            rotation_radius_cell_index=rotation_radius_cell,
            radius_pair_target_count=fine_target,
            radius_pair_accepted_count=fine_accepted,
        )

    def _select_fine_radius_cell(
        self,
        translation_interval: int,
        rotation_interval: int,
    ) -> tuple[int, int]:
        """Select fine radius cell."""
        count = self.radius_subinterval_count
        cursor_key = (translation_interval, rotation_interval)
        cursor = self._fine_cursors[cursor_key]
        for offset in range(count * count):
            local_index = (cursor + offset) % (count * count)
            local_translation, local_rotation = divmod(local_index, count)
            translation_cell = translation_interval * count + local_translation
            rotation_cell = rotation_interval * count + local_rotation
            if self.fine_radius_accepted[translation_cell][rotation_cell] < (
                self.fine_radius_targets[translation_cell][rotation_cell]
            ):
                self._fine_cursors[cursor_key] = (
                    local_index + 1
                ) % (count * count)
                return translation_cell, rotation_cell
        raise RuntimeError("outer radius pair has no open fine cell")

    def next_open_cell(self) -> JointCellTarget:
        """Return next open cell."""
        if self._pending_cell_index >= 0:
            return self._target(self._pending_cell_index)
        for offset in range(len(self.cells)):
            index = (self._next_cell_cursor + offset) % len(self.cells)
            if self.accepted[index] < self.targets[index]:
                t_interval, _, r_interval, _ = self.cells[index]
                fine_translation, fine_rotation = self._select_fine_radius_cell(
                    t_interval, r_interval
                )
                self._pending_cell_index = index
                self._pending_translation_radius_cell = fine_translation
                self._pending_rotation_radius_cell = fine_rotation
                self._pending_failures = 0
                self._next_cell_cursor = (index + 1) % len(self.cells)
                return self._target(index)
        raise ValueError("all joint accepted-cell quotas are full")

    def record_failure(self, index: int) -> bool:
        """Record failure."""
        if index != self._pending_cell_index:
            raise ValueError("joint failure does not match the pending cell")
        self.failed_attempts[index] += 1
        self._pending_failures += 1
        if self._pending_failures < self.max_consecutive_failures:
            return False
        self.releases[index] += 1
        self._clear_pending()
        return True

    def _clear_pending(self) -> None:
        """Clear pending."""
        self._pending_cell_index = -1
        self._pending_failures = 0
        self._pending_translation_radius_cell = -1
        self._pending_rotation_radius_cell = -1

    def record_acceptance(self, index: int) -> None:
        """Record acceptance."""
        if index != self._pending_cell_index:
            raise ValueError("joint acceptance does not match the pending cell")
        if self.accepted[index] >= self.targets[index]:
            raise ValueError("joint accepted-cell quota overflow")
        translation_cell = self._pending_translation_radius_cell
        rotation_cell = self._pending_rotation_radius_cell
        if self.fine_radius_accepted[translation_cell][rotation_cell] >= (
            self.fine_radius_targets[translation_cell][rotation_cell]
        ):
            raise ValueError("fine radius-pair quota overflow")
        self.accepted[index] += 1
        self.fine_radius_accepted[translation_cell][rotation_cell] += 1
        self._clear_pending()

    def _interval_index(self, vector: Sequence[float]) -> int:
        """Return interval index."""
        value = np.asarray(vector, dtype=np.float64)
        radius = float(np.linalg.norm(value))
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("joint coordinate must be a finite three-vector")
        if radius > self.edges[-1] + 1e-6:
            raise ValueError("joint coordinate lies outside configured ball")
        if radius <= self.edges[1]:
            return 0
        return min(
            int(np.searchsorted(self.edges[1:], radius, side="right")),
            len(self.edges) - 2,
        )

    def _direction_index(
        self,
        vector: Sequence[float],
        interval_index: int,
    ) -> int:
        """Return direction index."""
        if interval_index == 0:
            return -1
        value = np.asarray(vector, dtype=np.float64)
        radius = float(np.linalg.norm(value))
        if radius <= 0.0:
            raise ValueError("directional joint coordinate has zero radius")
        return int(np.argmax(self._directions @ (value / radius)))

    def _fine_radius_index(
        self,
        vector: Sequence[float],
        outer_interval_index: int,
    ) -> int:
        """Return fine radius index."""
        value = np.asarray(vector, dtype=np.float64)
        radius = float(np.linalg.norm(value))
        lower, upper = self.interval_bounds(outer_interval_index)
        fraction = min(
            max((radius - lower) / (upper - lower), 0.0),
            1.0 - np.finfo(np.float64).eps,
        )
        local_index = int(fraction * self.radius_subinterval_count)
        return (
            outer_interval_index * self.radius_subinterval_count
            + local_index
        )

    def fine_radius_cell_indices(
        self, normalized_coordinate: Sequence[float]
    ) -> tuple[int, int]:
        """Return fine radius cell indices."""
        value = np.asarray(normalized_coordinate, dtype=np.float64)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "normalized joint coordinate must be finite length six"
            )
        t_interval = self._interval_index(value[:3])
        r_interval = self._interval_index(value[3:])
        return (
            self._fine_radius_index(value[:3], t_interval),
            self._fine_radius_index(value[3:], r_interval),
        )

    def cell_index(self, normalized_coordinate: Sequence[float]) -> int:
        """Return cell index."""
        value = np.asarray(normalized_coordinate, dtype=np.float64)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "normalized joint coordinate must be finite length six"
            )
        t_interval = self._interval_index(value[:3])
        r_interval = self._interval_index(value[3:])
        cell = (
            t_interval,
            self._direction_index(value[:3], t_interval),
            r_interval,
            self._direction_index(value[3:], r_interval),
        )
        if cell not in self._cell_indices:
            raise ValueError("realized joint cell has no configured target")
        return self._cell_indices[cell]

    @property
    def complete(self) -> bool:
        """Return complete."""
        return (
            tuple(self.accepted) == self.targets
            and np.array_equal(
                np.asarray(self.fine_radius_accepted),
                np.asarray(self.fine_radius_targets),
            )
        )

    def report(self) -> dict:
        """Return report."""
        interval_count = len(self.edges) - 1
        radius_targets = np.zeros(
            (interval_count, interval_count), dtype=np.int64
        )
        radius_accepted = np.zeros_like(radius_targets)
        cell_reports = []
        for index, cell in enumerate(self.cells):
            t_interval, _, r_interval, _ = cell
            radius_targets[t_interval, r_interval] += self.targets[index]
            radius_accepted[t_interval, r_interval] += self.accepted[index]
            target = self._target(index)
            cell_reports.append({
                "id": target.cell_id,
                "translation_interval_index": t_interval,
                "translation_direction_index": cell[1],
                "rotation_interval_index": r_interval,
                "rotation_direction_index": cell[3],
                "target_count": self.targets[index],
                "accepted_count": self.accepted[index],
                "failed_attempt_count": self.failed_attempts[index],
                "release_count": self.releases[index],
            })
        fine_targets = np.asarray(self.fine_radius_targets, dtype=np.int64)
        fine_accepted = np.asarray(
            self.fine_radius_accepted, dtype=np.int64
        )
        fine_edges = [
            self.fine_radius_bounds(index)[0]
            for index in range(self.fine_radius_cell_count)
        ]
        fine_edges.append(self.fine_radius_bounds(
            self.fine_radius_cell_count - 1
        )[1])
        return {
            "version": (
                LEGACY_JOINT_CELL_BALANCING_VERSION
                if self.radius_subinterval_count == 1
                else JOINT_CELL_BALANCING_VERSION
            ),
            "distribution": JOINT_CELL_DISTRIBUTION,
            "sampling": JOINT_CELL_SAMPLING,
            "balanced": self.complete,
            "target_count": int(sum(self.targets)),
            "accepted_count": int(sum(self.accepted)),
            "direction_cell_count": self.direction_count,
            "max_consecutive_failures": self.max_consecutive_failures,
            "seed": self.seed,
            "radius_edges": [float(value) for value in self.edges],
            "radius_pair_target_matrix": radius_targets.tolist(),
            "radius_pair_accepted_matrix": radius_accepted.tolist(),
            "radius_subinterval_count": self.radius_subinterval_count,
            "radius_partition_index": self.radius_partition_index,
            "radius_partition_count": self.radius_partition_count,
            "fine_radius_edges": [float(value) for value in fine_edges],
            "fine_radius_pair_target_matrix": fine_targets.tolist(),
            "fine_radius_pair_accepted_matrix": fine_accepted.tolist(),
            "maximum_fine_radius_pair_count_difference": int(
                fine_accepted.max() - fine_accepted.min()
            ),
            "unfilled_fine_radius_cell_count": int(np.count_nonzero(
                fine_accepted < fine_targets
            )),
            "maximum_joint_cell_count_difference": int(
                max(self.accepted) - min(self.accepted)
            ),
            "unfilled_cell_count": int(sum(
                accepted < target
                for accepted, target in zip(self.accepted, self.targets)
            )),
            "cells": cell_reports,
        }


import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_sections import load_config_mapping
from .geometry import validate_transform


NOMINAL_EXTRINSIC_SCHEMA = "nominal_extrinsic_v1"
LIDAR_TO_CAMERA_DIRECTION = "lidar_to_camera"


@dataclass(frozen=True)
class NominalExtrinsicInput:
    """Represent nominal extrinsic input."""

    t0_id: str
    matrix: np.ndarray
    source_fingerprint: str
    source_type: str = "external_yaml"

    def to_metadata(self) -> dict:
        """Return a metadata representation."""
        return {
            "type": self.source_type,
            "schema_version": NOMINAL_EXTRINSIC_SCHEMA,
            "t0_id": self.t0_id,
            "transform_direction": LIDAR_TO_CAMERA_DIRECTION,
            "coordinate_equation": "p_camera = T0 @ p_lidar",
            "source_fingerprint": self.source_fingerprint,
        }


def validate_nominal_extrinsic(value, field_name: str = "matrix") -> np.ndarray:
    """Validate nominal extrinsic."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numeric values") from exc
    try:
        nominal = validate_transform(matrix)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc

    rotation = nominal[:3, :3]
    orthogonality_error = float(np.linalg.norm(
        rotation @ rotation.T - np.eye(3)
    ))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if max(orthogonality_error, determinant_error) > 1e-8:
        raise ValueError(
            f"{field_name} rotation must be an accurate SO(3) matrix; "
            f"orthogonality_error={orthogonality_error}, "
            f"determinant_error={determinant_error}"
        )
    return nominal


def load_nominal_extrinsic_config(
    path: str | Path,
) -> NominalExtrinsicInput:
    """Load nominal extrinsic config."""
    raw, fingerprint_bytes, embedded = load_config_mapping(
        path,
        section="nominal_extrinsic",
        description="T0 config",
    )
    if raw.get("schema_version") != NOMINAL_EXTRINSIC_SCHEMA:
        raise ValueError(
            "T0 config schema_version must be "
            f"{NOMINAL_EXTRINSIC_SCHEMA!r}"
        )
    if raw.get("configured") is not True:
        raise ValueError(
            "T0 config is not enabled; enter the product matrix and set "
            "configured: true"
        )
    t0_id = raw.get("t0_id")
    if not isinstance(t0_id, str) or not t0_id.strip():
        raise ValueError("T0 config t0_id must be a non-empty string")
    if t0_id.strip() == "replace_with_product_calibration_id":
        raise ValueError("T0 config t0_id still contains the template placeholder")
    if raw.get("transform_direction") != LIDAR_TO_CAMERA_DIRECTION:
        raise ValueError(
            "T0 transform_direction must be 'lidar_to_camera' because the "
            "pipeline uses p_camera = T0 @ p_lidar"
        )
    if raw.get("matrix") is None:
        raise ValueError("T0 config matrix is not filled in")

    matrix = validate_nominal_extrinsic(raw["matrix"], "T0 matrix")
    return NominalExtrinsicInput(
        t0_id=t0_id.strip(),
        matrix=matrix,
        source_fingerprint=hashlib.sha256(fingerprint_bytes).hexdigest(),
        source_type="embedded_task_yaml" if embedded else "external_yaml",
    )


from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


GEODESIC_AXIS_ANGLE = "geodesic_axis_angle"


FIXED_IDENTITY = "fixed_identity"


DIRECT_LEFT_EULER_BOX = "direct_left_euler_box"


RESIDUAL_SAMPLER_MODES = (
    GEODESIC_AXIS_ANGLE,
    FIXED_IDENTITY,
    DIRECT_LEFT_EULER_BOX,
)


DIRECT_RESIDUAL_SAMPLER_MODES = (FIXED_IDENTITY, DIRECT_LEFT_EULER_BOX)


DIRECT_BOX_COORDINATE_ORDER = (
    "tx", "ty", "tz", "roll_x", "pitch_y", "yaw_z"
)


def direct_box_coordinates_from_sampling(
    sampling: dict,
) -> tuple[list[float], list[float]]:
    """Return direct box coordinates from sampling."""
    if sampling.get("mode") != DIRECT_LEFT_EULER_BOX:
        raise ValueError("direct box coordinates require Direct-Euler sampling")
    translation = sampling.get("sampled_translation_xyz_m") or {}
    rotation = sampling.get("sampled_euler_xyz_deg") or {}
    components = np.asarray([
        translation.get("x"), translation.get("y"), translation.get("z"),
        rotation.get("roll_x"), rotation.get("pitch_y"), rotation.get("yaw_z"),
    ], dtype=np.float64)
    limits = np.asarray([
        sampling.get("max_translation_component_m"),
        sampling.get("max_translation_component_m"),
        sampling.get("max_translation_component_m"),
        sampling.get("max_euler_component_deg"),
        sampling.get("max_euler_component_deg"),
        sampling.get("max_euler_component_deg"),
    ], dtype=np.float64)
    if (
        components.shape != (6,) or limits.shape != (6,)
        or not np.all(np.isfinite(components))
        or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0)
    ):
        raise ValueError("Direct-Euler component sampling record is invalid")
    normalized = components / limits
    if np.any(np.abs(normalized) > 1.0 + 1e-10):
        raise ValueError("Direct-Euler sample lies outside configured box")
    return components.tolist(), normalized.tolist()


@dataclass(frozen=True)
class ResidualProposal:
    """Represent residual proposal."""

    correction: np.ndarray
    sampling: dict

    @property
    def rotation(self) -> np.ndarray:
        """Return rotation."""
        return self.correction[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """Return translation."""
        return self.correction[:3, 3]


def fixed_identity_correction() -> ResidualProposal:
    """Return fixed identity correction."""
    correction = np.eye(4, dtype=np.float64)
    sampling = {
        "mode": FIXED_IDENTITY,
        "true_extrinsic_policy": "Ti_equals_T0",
        "sampled_pose_semantics": "no_generation_time_perturbation",
        "supervision_semantics": "nominal_to_true_identity_residual",
        "sampled_pose": correction.tolist(),
        "correction": correction.tolist(),
        "correction_translation_m": [0.0, 0.0, 0.0],
        "correction_translation_norm_m": 0.0,
        "correction_rotvec_rad": [0.0, 0.0, 0.0],
        "correction_geodesic_deg": 0.0,
    }
    return ResidualProposal(correction=correction, sampling=sampling)


def direct_euler_box_max_geodesic_deg(max_component_deg: float) -> float:
    """Return direct euler box max geodesic deg."""
    limit = float(max_component_deg)
    if not 0.0 < limit < 180.0:
        raise ValueError("max Euler component must lie within (0, 180) degrees")
    corners = np.array([
        [roll, pitch, yaw]
        for roll in (-limit, limit)
        for pitch in (-limit, limit)
        for yaw in (-limit, limit)
    ], dtype=np.float64)
    return float(np.rad2deg(np.max(
        Rotation.from_euler("xyz", corners, degrees=True).magnitude()
    )))


def sample_direct_left_euler_box_correction(
    rng: np.random.Generator,
    max_euler_component_deg: float,
    max_translation_component_m: float,
) -> ResidualProposal:
    """Sample independent translation (meters) and Euler XYZ (degrees) components.

    The returned left residual uses Rz(yaw) @ Ry(pitch) @ Rx(roll), and the target
    extrinsic is Ti = delta_T @ T0. Coverage uses these original sampled components.
    """
    angle_limit = float(max_euler_component_deg)
    translation_limit = float(max_translation_component_m)
    direct_euler_box_max_geodesic_deg(angle_limit)
    if not np.isfinite(translation_limit) or translation_limit <= 0.0:
        raise ValueError("translation component limit must be positive")

    euler_xyz_deg = rng.uniform(-angle_limit, angle_limit, size=3)
    translation_xyz_m = rng.uniform(
        -translation_limit, translation_limit, size=3
    )
    correction = np.eye(4, dtype=np.float64)

    correction[:3, :3] = Rotation.from_euler(
        "xyz", euler_xyz_deg, degrees=True
    ).as_matrix()
    correction[:3, 3] = translation_xyz_m
    correction_rotvec = Rotation.from_matrix(
        correction[:3, :3]
    ).as_rotvec()
    sampling = {
        "mode": DIRECT_LEFT_EULER_BOX,
        "sampled_pose_semantics": "direct_nominal_to_true_left_residual",
        "supervision_semantics": "delta_T_equals_sampled_correction",
        "composition": "Ti_equals_delta_T_times_T0",
        "euler_sequence": "extrinsic_xyz_equals_Rz_yaw_Ry_pitch_Rx_roll",
        "max_euler_component_deg": angle_limit,
        "sampled_euler_xyz_deg": {
            "roll_x": float(euler_xyz_deg[0]),
            "pitch_y": float(euler_xyz_deg[1]),
            "yaw_z": float(euler_xyz_deg[2]),
        },
        "max_translation_component_m": translation_limit,
        "sampled_translation_xyz_m": {
            "x": float(translation_xyz_m[0]),
            "y": float(translation_xyz_m[1]),
            "z": float(translation_xyz_m[2]),
        },
        "correction": correction.tolist(),
        "correction_translation_m": translation_xyz_m.tolist(),
        "correction_translation_norm_m": float(
            np.linalg.norm(translation_xyz_m)
        ),
        "correction_rotvec_rad": correction_rotvec.tolist(),
        "correction_geodesic_deg": float(np.rad2deg(
            np.linalg.norm(correction_rotvec)
        )),
    }
    return ResidualProposal(correction=correction, sampling=sampling)


__all__ = [
    "DIRECT_RESIDUAL_SAMPLER_MODES",
    "DIRECT_LEFT_EULER_BOX",
    "DIRECT_BOX_COORDINATE_ORDER",
    "FIXED_IDENTITY",
    "GEODESIC_AXIS_ANGLE",
    "RESIDUAL_SAMPLER_MODES",
    "ResidualProposal",
    "direct_box_coordinates_from_sampling",
    "direct_euler_box_max_geodesic_deg",
    "fixed_identity_correction",
    "sample_direct_left_euler_box_correction",
]


from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import invert_transform, residual_coordinates, validate_transform
from .omni import OmniFrame
from .sensors import (
    CameraDepthStage,
    GeneratedSensorSample,
    LidarStage,
    clear_rgb_cache,
    configure_rgb_cache,
    finalize_sensor_sample,
    generate_candidate_camera_depth,
    generate_candidate_lidar,
    rgb_cache_report,
)


OPTICAL_AXIS_LATITUDE_VIEW_ROTATION_DISTRIBUTION = (
    "target_translation_conditioned_optical_axis_latitude_band"
)
REPORT_ONLY = "report_only"
STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION = "target_translation_conditioned_strict_level_rig"
TRANSLATION_RADIUS_BALANCED = "translation_radius_balanced"
VIEW_ROTATION_DISTRIBUTION = "target_translation_conditioned_uniform_roll"
VIEW_ROTATION_DISTRIBUTIONS = (
    VIEW_ROTATION_DISTRIBUTION,
    STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION,
)


STOCHASTIC_CANDIDATE_VERSION = "stochastic_raw_omni_ti_candidate_v7_optical_axis_latitude"


WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def sample_uniform_so3_ball(
    rng: np.random.Generator,
    max_rotation_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample uniform SO(3) ball."""
    if not 0.0 < max_rotation_deg < 180.0:
        raise ValueError("max_rotation_deg must lie within (0, 180)")
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    while norm < 1e-12:
        axis = rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
    axis /= norm
    maximum = np.deg2rad(max_rotation_deg)
    target = float(rng.random()) * (maximum - np.sin(maximum))
    lower = 0.0
    upper = maximum
    for _ in range(64):
        angle = 0.5 * (lower + upper)
        if angle - np.sin(angle) < target:
            lower = angle
        else:
            upper = angle
    rotvec = axis * (0.5 * (lower + upper))
    return Rotation.from_rotvec(rotvec).as_matrix(), rotvec


def sample_uniform_so3(rng: np.random.Generator) -> np.ndarray:
    """Sample uniform SO(3)."""
    quaternion = rng.normal(size=4)
    while float(np.linalg.norm(quaternion)) < 1e-12:
        quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    return Rotation.from_quat(quaternion).as_matrix()


def sample_uniform_s2(rng: np.random.Generator) -> np.ndarray:
    """Sample uniform s2."""
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    while norm < 1e-12:
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
    return direction / norm


def sample_uniform_s2_cell(
    rng: np.random.Generator,
    direction_index: int,
    directions: np.ndarray,
) -> np.ndarray:
    """Sample uniform s2 cell."""
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("direction centers must have shape Nx3")
    if not 0 <= direction_index < len(values):
        raise ValueError("direction cell index is out of range")
    while True:
        direction = sample_uniform_s2(rng)
        if int(np.argmax(values @ direction)) == direction_index:
            return direction


def sample_stratified_translation_target(
    rng: np.random.Generator,
    radius_interval: tuple[float, float],
    stratum_index: int,
    stratum_count: int,
    direction_cell_index: int = -1,
    directions: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Sample stratified translation target."""
    lower, upper = (float(value) for value in radius_interval)
    upper = min(upper, 1.0)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("normalized radius interval must lie within [0, 1]")
    if stratum_count < 1 or not 0 <= stratum_index < stratum_count:
        raise ValueError("translation radius stratum index is invalid")
    width = (upper - lower) / stratum_count
    stratum_lower = lower + stratum_index * width
    stratum_upper = lower + (stratum_index + 1) * width
    radius = stratum_lower + float(rng.random()) * (
        stratum_upper - stratum_lower
    )
    if direction_cell_index < 0:
        direction = sample_uniform_s2(rng)
        direction_sampling = "uniform_s2"
    else:
        if directions is None:
            raise ValueError(
                "target direction cell requires direction centers"
            )
        direction = sample_uniform_s2_cell(
            rng, direction_cell_index, directions
        )
        direction_sampling = "uniform_s2_conditioned_on_fibonacci_cell"
    target = radius * direction
    return target, {
        "distribution": TRANSLATION_INTERVAL_SAMPLING,
        "radius_interval": [lower, upper],
        "stratum_index": int(stratum_index),
        "stratum_count": int(stratum_count),
        "stratum_bounds": [float(stratum_lower), float(stratum_upper)],
        "normalized_radius": float(radius),
        "direction": direction.tolist(),
        "direction_sampling": direction_sampling,
        "direction_cell_index": int(direction_cell_index),
    }


def sample_stratified_rotation_target(
    rng: np.random.Generator,
    max_rotation_deg: float,
    radius_interval: tuple[float, float],
    stratum_index: int,
    stratum_count: int,
    direction_cell_index: int,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample stratified rotation target."""
    lower, upper = (float(value) for value in radius_interval)
    upper = min(upper, 1.0)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("normalized rotation interval must lie within [0, 1]")
    if stratum_count < 1 or not 0 <= stratum_index < stratum_count:
        raise ValueError("rotation radius stratum index is invalid")
    width = (upper - lower) / stratum_count
    stratum_lower = lower + stratum_index * width
    stratum_upper = lower + (stratum_index + 1) * width
    normalized_radius = stratum_lower + float(rng.random()) * (
        stratum_upper - stratum_lower
    )
    if direction_cell_index < 0:
        axis = sample_uniform_s2(rng)
        axis_sampling = "uniform_s2"
    else:
        axis = sample_uniform_s2_cell(
            rng, direction_cell_index, directions
        )
        axis_sampling = "uniform_s2_conditioned_on_fibonacci_cell"
    rotvec = normalized_radius * np.deg2rad(max_rotation_deg) * axis
    return Rotation.from_rotvec(rotvec).as_matrix(), rotvec, {
        "distribution": JOINT_CELL_SAMPLING,
        "radius_interval": [lower, upper],
        "stratum_index": int(stratum_index),
        "stratum_count": int(stratum_count),
        "stratum_bounds": [float(stratum_lower), float(stratum_upper)],
        "normalized_radius": float(normalized_radius),
        "axis": axis.tolist(),
        "axis_sampling": axis_sampling,
        "direction_cell_index": int(direction_cell_index),
    }


def feasible_direction_cosine_minimum(
    baseline_norm: float,
    target_norm: float,
    translation_limit: float,
    scale_min: float,
    scale_max: float,
    tolerance: float = 1e-12,
) -> Optional[float]:
    """Return feasible direction cosine minimum."""
    values = (baseline_norm, target_norm, translation_limit, scale_min, scale_max)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("feasible-direction inputs must be finite")
    if baseline_norm <= tolerance:
        return None
    if target_norm < 0.0 or translation_limit <= 0.0:
        raise ValueError("target norm and translation limit are invalid")
    if not 0.0 < scale_min < scale_max:
        raise ValueError("scale range is invalid")
    radial_min = baseline_norm * scale_min
    radial_max = baseline_norm * scale_max
    if target_norm <= tolerance:
        return -1.0 if radial_min <= translation_limit + tolerance else None
    radial_candidates = [radial_min, radial_max]
    stationary_sq = target_norm ** 2 - translation_limit ** 2
    if stationary_sq > 0.0:
        stationary = float(np.sqrt(stationary_sq))
        if radial_min <= stationary <= radial_max:
            radial_candidates.append(stationary)
    cosine_minimum = min(
        (radial ** 2 + target_norm ** 2 - translation_limit ** 2)
        / (2.0 * radial * target_norm)
        for radial in radial_candidates
    )
    if cosine_minimum > 1.0 + tolerance:
        return None
    return float(np.clip(cosine_minimum, -1.0, 1.0))


def feasible_direction_probability(
    baseline_norm: float,
    target_norm: float,
    translation_limit: float,
    scale_min: float,
    scale_max: float,
) -> float:
    """Return feasible direction probability."""
    minimum_cosine = feasible_direction_cosine_minimum(
        baseline_norm, target_norm, translation_limit, scale_min, scale_max
    )
    if minimum_cosine is None:
        return 0.0
    if target_norm <= 1e-12:
        return 1.0
    return float(0.5 * (1.0 - minimum_cosine))


def _rotation_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return rotation mapping."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine >= 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine <= -1.0 + 1e-12:
        return Rotation.from_rotvec(
            np.pi * _orthogonal_unit(source)
        ).as_matrix()
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine ** 2)


def _strict_level_rig_view_roll(
    base_rotation: np.ndarray,
    rotation_axis: np.ndarray,
    gravity_pano: np.ndarray,
    lidar_up_camera: np.ndarray,
) -> Tuple[float, dict]:
    """Return strict level rig view roll."""
    axis = np.asarray(rotation_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    gravity = np.asarray(gravity_pano, dtype=np.float64)
    gravity /= np.linalg.norm(gravity)
    lidar_up = np.asarray(lidar_up_camera, dtype=np.float64)
    lidar_up /= np.linalg.norm(lidar_up)
    base_up = np.asarray(base_rotation, dtype=np.float64) @ lidar_up

    constant = float(np.dot(gravity, axis) * np.dot(axis, base_up))
    cosine_coefficient = float(np.dot(gravity, base_up) - constant)
    sine_coefficient = float(np.dot(gravity, np.cross(axis, base_up)))
    roll = float(np.arctan2(
        sine_coefficient, cosine_coefficient
    ) % (2.0 * np.pi))
    rotation = (
        Rotation.from_rotvec(roll * axis).as_matrix()
        @ base_rotation
    )
    rig_up_pano = rotation @ lidar_up
    alignment_cosine = float(np.clip(
        np.dot(rig_up_pano, gravity), -1.0, 1.0
    ))
    return roll, {
        "level_rig_tilt_deg": float(np.rad2deg(np.arccos(
            alignment_cosine
        ))),
        "rig_up_in_target_pano": rig_up_pano.tolist(),
        "level_rig_alignment_cosine": alignment_cosine,
    }


def _sample_optical_axis_latitude_roll(
    rng: np.random.Generator,
    base_rotation: np.ndarray,
    rotation_axis: np.ndarray,
    gravity_pano: np.ndarray,
    max_latitude_deg: float,
) -> Tuple[Optional[float], dict]:
    """Sample optical axis latitude roll."""
    axis = np.asarray(rotation_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    gravity = np.asarray(gravity_pano, dtype=np.float64)
    gravity /= np.linalg.norm(gravity)
    optical_axis = np.asarray(base_rotation, dtype=np.float64)[:, 2]
    threshold = float(np.sin(np.deg2rad(max_latitude_deg)))
    constant = float(np.dot(gravity, axis) * np.dot(axis, optical_axis))
    cosine_coefficient = float(np.dot(gravity, optical_axis) - constant)
    sine_coefficient = float(np.dot(gravity, np.cross(axis, optical_axis)))
    amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
    possible_min = constant - amplitude
    possible_max = constant + amplitude
    if possible_max < -threshold or possible_min > threshold:
        return None, {
            "max_camera_optical_axis_latitude_deg": float(max_latitude_deg),
            "latitude_band_coordinate_frame": "target_pano",
            "sampling": "uniform_rejection_over_roll_circle",
            "feasible": False,
        }

    for attempt in range(1, 100001):
        roll = 2.0 * np.pi * float(rng.random())
        rotation = (
            Rotation.from_rotvec(roll * axis).as_matrix() @ base_rotation
        )
        cosine = float(np.clip(
            np.dot(gravity, rotation[:, 2]), -1.0, 1.0
        ))
        latitude_deg = float(np.rad2deg(np.arcsin(abs(cosine))))
        if latitude_deg <= max_latitude_deg + 1e-12:
            return roll, {
                "camera_optical_axis": "opencv_positive_z",
                "camera_optical_axis_latitude_deg": latitude_deg,
                "max_camera_optical_axis_latitude_deg": float(max_latitude_deg),
                "latitude_band_coordinate_frame": "target_pano",
                "sampling": "uniform_rejection_over_roll_circle",
                "feasible": True,
                "weak_gravity_roll_attempts": attempt,
            }
    raise RuntimeError(
        "optical-axis latitude roll sampling exceeded deterministic budget"
    )

def _view_gravity_audit(
    pano_from_camera: np.ndarray,
    target: OmniFrame,
) -> dict:
    """Return view gravity audit."""
    target_rotation = validate_transform(target.world_from_pano)[:3, :3]
    gravity_pano = target_rotation.T @ WORLD_UP
    gravity_camera = np.asarray(pano_from_camera, dtype=np.float64).T @ gravity_pano
    gravity_camera /= np.linalg.norm(gravity_camera)
    image_plane_norm = float(np.hypot(
        gravity_camera[0], gravity_camera[1]
    ))
    horizon_roll_deg = (
        float(np.rad2deg(np.arctan2(
            gravity_camera[0], -gravity_camera[1]
        )))
        if image_plane_norm > 1e-12 else None
    )
    camera_level_tilt_deg = float(np.rad2deg(np.arccos(np.clip(
        -gravity_camera[1], -1.0, 1.0
    ))))
    return {
        "world_gravity_axis": WORLD_UP.tolist(),
        "gravity_in_camera": gravity_camera.tolist(),
        "gravity_image_plane_norm": image_plane_norm,
        "camera_horizon_roll_deg": horizon_roll_deg,
        "camera_level_tilt_deg": camera_level_tilt_deg,
    }


def rotation_matrix_audit(rotation: np.ndarray) -> dict:
    """Return rotation matrix audit."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix = validate_transform(transform)[:3, :3]
    rotvec = Rotation.from_matrix(matrix).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec / angle if angle > 1e-12 else np.zeros(3, dtype=np.float64)
    return {
        "angle_deg": float(np.rad2deg(angle)),
        "axis": axis.tolist(),
    }


def construct_targeted_stochastic_candidate(
    source: OmniFrame,
    target: OmniFrame,
    pair_id: str,
    nominal: np.ndarray,
    level: GenerationLevel,
    delta_rotation: np.ndarray,
    normalized_residual_translation: np.ndarray,
    translation_sampling: dict,
    scale_min: float,
    scale_max: float,
    view_roll: Optional[float],
    proposal_index: int,
    radius_interval_index: int,
    rotation_sampling: Optional[dict] = None,
    joint_cell_target: Optional[JointCellTarget] = None,
    view_rotation_distribution: str = VIEW_ROTATION_DISTRIBUTION,
    max_level_rig_tilt_deg: Optional[float] = None,
    camera_optical_axis_latitude_band_enabled: bool = False,
    max_camera_optical_axis_latitude_deg: Optional[float] = None,
    view_rng: Optional[np.random.Generator] = None,
    gravity_bucket: Optional[str] = None,
) -> Tuple[Optional[dict], dict]:
    """Realize a requested residual using a qualified source/target frame pair.

    ``nominal`` is LiDAR-to-camera T0; normalized translation uses the level's
    meter limit, rotations are matrices, view_roll is radians, and scale bounds
    are dimensionless. Derive the physical scale from the raw relative pose and
    requested Ti. Return a candidate plus its audit report, or None with a rejection
    report when the geometry is infeasible. Invalid parameters raise ValueError.
    """
    nominal = validate_transform(nominal)
    delta_rotation = np.asarray(delta_rotation, dtype=np.float64)
    normalized_target = np.asarray(
        normalized_residual_translation, dtype=np.float64
    )
    if normalized_target.shape != (3,) or not np.all(np.isfinite(normalized_target)):
        raise ValueError("normalized residual translation must be a finite 3-vector")
    if float(np.linalg.norm(normalized_target)) > 1.0 + 1e-10:
        raise ValueError("normalized residual translation lies outside unit ball")
    rotation_transform = np.eye(4, dtype=np.float64)
    rotation_transform[:3, :3] = delta_rotation
    delta_rotation = validate_transform(rotation_transform)[:3, :3]
    if view_rotation_distribution not in VIEW_ROTATION_DISTRIBUTIONS:
        raise ValueError(
            "unsupported view rotation distribution: "
            f"{view_rotation_distribution}"
        )
    if camera_optical_axis_latitude_band_enabled:
        if view_rotation_distribution != VIEW_ROTATION_DISTRIBUTION:
            raise ValueError(
                "optical-axis latitude band requires uniform-roll views"
            )
        if (
            max_camera_optical_axis_latitude_deg is None
            or not np.isfinite(max_camera_optical_axis_latitude_deg)
            or not 0.0 < max_camera_optical_axis_latitude_deg < 90.0
        ):
            raise ValueError(
                "optical-axis latitude limit must lie within (0, 90)"
            )
        if view_rng is None:
            raise ValueError(
                "optical-axis latitude band requires the caller random stream"
            )
        if view_roll is not None:
            raise ValueError(
                "preselected view_roll is invalid with the latitude band"
            )
    elif view_rotation_distribution == VIEW_ROTATION_DISTRIBUTION and (
        view_roll is None or not 0.0 <= view_roll < 2.0 * np.pi
    ):
        raise ValueError("view_roll must lie within [0, 2*pi)")
    if gravity_bucket is not None:
        raise ValueError("gravity bucket sampling is no longer supported")
    if view_rotation_distribution == STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION:
        if (
            max_level_rig_tilt_deg is None
            or not np.isfinite(max_level_rig_tilt_deg)
            or not 0.0 < max_level_rig_tilt_deg <= 45.0
        ):
            raise ValueError(
                "constrained level-rig view requires max tilt within (0, 45]"
            )
    elif max_level_rig_tilt_deg is not None:
        raise ValueError(
            "max_level_rig_tilt_deg is only valid for constrained views"
        )

    target_from_source = (
        invert_transform(target.world_from_pano)
        @ validate_transform(source.world_from_pano)
    )
    relative_rotation = target_from_source[:3, :3]
    relative_translation = target_from_source[:3, 3]
    baseline_norm = float(np.linalg.norm(relative_translation))
    translation_limit = level.residual_translation_normalization_m(nominal)
    desired_residual_translation = normalized_target * translation_limit
    true_rotation = delta_rotation @ nominal[:3, :3]
    residual_target = delta_rotation @ nominal[:3, 3]
    desired_true_translation = desired_residual_translation + residual_target
    desired_true_norm = float(np.linalg.norm(desired_true_translation))
    base_report = {
        "pair_id": pair_id,
        "proposal_index": int(proposal_index),
        "level": level.name,
        "raw_baseline_m_audit_only": baseline_norm,
        "translation_parameter_limit_m": level.effective_max_translation(nominal),
        "residual_translation_normalization_m": translation_limit,
        "max_geodesic_rotation_deg": level.max_geodesic_rotation_deg,
        "target_normalized_translation": normalized_target.tolist(),
        "translation_radius_interval_index": int(radius_interval_index),
    }
    if baseline_norm <= 1e-12 or desired_true_norm <= 1e-12:
        return None, {**base_report, "reason": "target_translation_degenerate"}
    scale = desired_true_norm / baseline_norm
    if scale < scale_min - 1e-12 or scale > scale_max + 1e-12:
        return None, {
            **base_report,
            "reason": "target_translation_scale_out_of_range",
            "required_scale": float(scale),
        }

    desired_true_direction = desired_true_translation / desired_true_norm
    relative_direction = relative_translation / baseline_norm
    base_rotation = _rotation_mapping(
        desired_true_direction, relative_direction
    )
    latitude_conditioning = None
    if camera_optical_axis_latitude_band_enabled:
        target_rotation = validate_transform(target.world_from_pano)[:3, :3]
        gravity_pano = target_rotation.T @ WORLD_UP
        selected_roll, latitude_conditioning = (
            _sample_optical_axis_latitude_roll(
                view_rng,
                base_rotation,
                relative_direction,
                gravity_pano,
                float(max_camera_optical_axis_latitude_deg),
            )
        )
        gravity_alignment = None
        if selected_roll is None:
            return None, {
                **base_report,
                "reason": "camera_optical_axis_latitude_band_infeasible",
                "view_conditioning": latitude_conditioning,
            }
    elif view_rotation_distribution == VIEW_ROTATION_DISTRIBUTION:
        selected_roll = float(view_roll)
        gravity_alignment = None
    else:
        target_rotation = validate_transform(target.world_from_pano)[:3, :3]
        gravity_pano = target_rotation.T @ WORLD_UP
        selected_roll, gravity_alignment = _strict_level_rig_view_roll(
            base_rotation,
            relative_direction,
            gravity_pano,
            true_rotation @ WORLD_UP,
        )
        gravity_alignment["max_level_rig_tilt_deg"] = float(
            max_level_rig_tilt_deg
        )
        if (
            gravity_alignment["level_rig_tilt_deg"]
            > max_level_rig_tilt_deg + 1e-10
        ):
            return None, {
                **base_report,
                "reason": "level_rig_tilt_exceeds_limit",
                "gravity_alignment": gravity_alignment,
            }
    pano_from_camera = (
        Rotation.from_rotvec(selected_roll * relative_direction).as_matrix()
        @ base_rotation
    )
    pano_from_lidar = relative_rotation.T @ pano_from_camera @ true_rotation
    source_rotation = validate_transform(source.world_from_pano)[:3, :3]
    rig_up_world = source_rotation @ pano_from_lidar @ WORLD_UP
    reconstructed_rotation = (
        pano_from_camera.T @ relative_rotation @ pano_from_lidar
    )
    translation_direction = pano_from_camera.T @ relative_translation
    reconstructed_translation = scale * translation_direction
    if not np.allclose(
        reconstructed_rotation, true_rotation, rtol=0.0, atol=1e-10
    ):
        raise RuntimeError("targeted view does not reconstruct true rotation")
    if not np.allclose(
        reconstructed_translation,
        desired_true_translation,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("targeted view does not reconstruct true translation")

    true_extrinsic = np.eye(4, dtype=np.float64)
    true_extrinsic[:3, :3] = reconstructed_rotation
    true_extrinsic[:3, 3] = reconstructed_translation
    residual, translation, rotvec, normalized = residual_coordinates(
        true_extrinsic,
        nominal,
        translation_limit,
        np.deg2rad(level.max_geodesic_rotation_deg),
    )
    if not np.allclose(translation, desired_residual_translation, atol=1e-10):
        raise RuntimeError("realized residual translation differs from target")

    lidar_view = rotation_matrix_audit(pano_from_lidar)
    camera_view = rotation_matrix_audit(pano_from_camera)
    candidate_id = f"{pair_id}:{level.name}:proposal{proposal_index:09d}"
    gravity_audit = _view_gravity_audit(pano_from_camera, target)
    view_conditioning = {
        "conditioning": (
            "target_residual_translation_exact_optical_axis_latitude_band"
            if camera_optical_axis_latitude_band_enabled
            else (
                "target_residual_translation_exact_uniform_roll"
                if view_rotation_distribution == VIEW_ROTATION_DISTRIBUTION
                else "target_residual_translation_exact_strict_level_rig"
            )
        ),
        "selected_roll_rad": float(selected_roll),
        "translation_radius_interval_index": int(radius_interval_index),
        "target_normalized_translation": normalized_target.tolist(),
        **gravity_audit,
    }
    if camera_optical_axis_latitude_band_enabled:
        view_conditioning.update(latitude_conditioning)
        view_conditioning["conditional_roll_rad"] = float(selected_roll)
    elif view_rotation_distribution == VIEW_ROTATION_DISTRIBUTION:
        view_conditioning["conditional_roll_rad"] = float(selected_roll)
    else:
        view_conditioning.update(gravity_alignment)
    if (
        view_rotation_distribution
        == STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION
    ):
        view_conditioning.update({
            "rig_up_axis": "lidar_positive_z",
            "rig_up_in_world": rig_up_world.tolist(),
            "level_rig_within_threshold": True,
        })
    if rotation_sampling is None:
        rotation_sampling_name = "uniform_so3_geodesic_ball"
    elif rotation_sampling.get("mode") in DIRECT_RESIDUAL_SAMPLER_MODES:
        rotation_sampling_name = rotation_sampling["mode"]
    else:
        rotation_sampling_name = JOINT_CELL_SAMPLING
    effective_view_rotation_sampling = (
        OPTICAL_AXIS_LATITUDE_VIEW_ROTATION_DISTRIBUTION
        if camera_optical_axis_latitude_band_enabled
        else view_rotation_distribution
    )
    candidate = {
        "version": STOCHASTIC_CANDIDATE_VERSION,
        "candidate_id": candidate_id,
        "pair_id": pair_id,
        "source_frame": source.frame_id,
        "target_frame": target.frame_id,
        "curriculum_level": level.name,
        "rotation_id": f"continuous_proposal_{proposal_index:09d}",
        "rotation_axis_index": -1,
        "rotation_shell_index": -1,
        "rotation_sampling": rotation_sampling_name,
        "translation_sampling": dict(translation_sampling),
        "translation_radius_interval_index": int(radius_interval_index),
        "target_normalized_translation": normalized_target.tolist(),
        "view_rotation_sampling": effective_view_rotation_sampling,
        "view_rotation_conditioning": view_conditioning,
        "view_anchor": "target_residual_translation",
        "camera_rotation_sampled_independently": False,
        "lidar_rotation_sampled_independently": False,
        "camera_view_audit": camera_view,
        "lidar_view_audit": lidar_view,
        "lidar_orientation_id": f"continuous_view_{proposal_index:09d}",
        "lidar_tilt_axis_index": -1,
        "lidar_tilt_magnitude_index": -1,
        "lidar_tilt_magnitude_deg": lidar_view["angle_deg"],
        "lidar_tilt_axis": lidar_view["axis"],
        "pano_from_lidar_rotation": pano_from_lidar.tolist(),
        "pano_from_camera_rotation": pano_from_camera.tolist(),
        "raw_relative_rotation": relative_rotation.tolist(),
        "raw_relative_translation_m": relative_translation.tolist(),
        "depth_scale": float(scale),
        "scale_interval": [float(scale), float(scale)],
        "scale_fraction": None,
        "true_extrinsic": true_extrinsic.tolist(),
        "residual": residual.tolist(),
        "residual_translation": translation.tolist(),
        "residual_rotvec": rotvec.tolist(),
        "normalized": normalized.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_deg": float(np.rad2deg(np.linalg.norm(rotvec))),
    }
    if rotation_sampling is not None:
        candidate["rotation_sampling_detail"] = dict(rotation_sampling)
        if rotation_sampling.get("mode") in DIRECT_RESIDUAL_SAMPLER_MODES:
            candidate["residual_perturbation_sampling"] = dict(
                rotation_sampling
            )
        if rotation_sampling.get("mode") == DIRECT_LEFT_EULER_BOX:
            components, box_normalized = direct_box_coordinates_from_sampling(
                rotation_sampling
            )
            candidate.update({
                "residual_draw_index": int(
                    rotation_sampling["residual_draw_index"]
                ),
                "direct_box_coordinate_order": list(
                    DIRECT_BOX_COORDINATE_ORDER
                ),
                "direct_box_components": components,
                "direct_box_normalized": box_normalized,
            })
    if joint_cell_target is not None:
        candidate["joint_cell_target"] = {
            **joint_cell_target.__dict__,
        }
    return candidate, {
        **base_report,
        "reason": None,
        "required_scale": float(scale),
    }


@dataclass
class ExtrinsicSampleGenerator:
    """Represent extrinsic sample generator."""

    nominal: np.ndarray
    level: GenerationLevel
    scale_min: float
    scale_max: float
    seed: int
    residual_sampler: "ResidualSamplerConfig"
    joint_direction_count: int = 0
    accepted_distribution_control: str = TRANSLATION_RADIUS_BALANCED
    view_rotation_distribution: str = VIEW_ROTATION_DISTRIBUTION
    max_level_rig_tilt_deg: Optional[float] = None
    camera_optical_axis_latitude_band_enabled: bool = False
    max_camera_optical_axis_latitude_deg: Optional[float] = None

    def __post_init__(self):

        """Validate the initialized instance."""
        self.nominal = validate_transform(self.nominal)
        if self.view_rotation_distribution not in VIEW_ROTATION_DISTRIBUTIONS:
            raise ValueError(
                "unsupported view rotation distribution: "
                f"{self.view_rotation_distribution}"
            )
        if (
            self.view_rotation_distribution
            == STRICT_LEVEL_RIG_VIEW_ROTATION_DISTRIBUTION
        ):
            if (
                self.max_level_rig_tilt_deg is None
                or not np.isfinite(self.max_level_rig_tilt_deg)
                or not 0.0 < self.max_level_rig_tilt_deg <= 45.0
            ):
                raise ValueError(
                    "constrained view sampler requires max tilt within (0, 45]"
                )
        elif self.max_level_rig_tilt_deg is not None:
            raise ValueError(
                "max_level_rig_tilt_deg is only valid for constrained views"
            )
        if self.camera_optical_axis_latitude_band_enabled:
            if self.view_rotation_distribution != VIEW_ROTATION_DISTRIBUTION:
                raise ValueError(
                    "optical-axis latitude band requires uniform-roll views"
                )
            if (
                self.max_camera_optical_axis_latitude_deg is None
                or not np.isfinite(self.max_camera_optical_axis_latitude_deg)
                or not 0.0 < self.max_camera_optical_axis_latitude_deg < 90.0
            ):
                raise ValueError(
                    "optical-axis latitude limit must lie within (0, 90)"
                )
        self._rng = np.random.default_rng(int(self.seed))
        self.proposal_count = 0
        self._pending_direct_residual = None
        self._pending_direct_attempt_count = 0
        self._pending_direct_accepted_count = 0
        self._direct_residual_draw_count = 0
        self._direct_residual_accepted_count = 0
        self._direct_residual_rejected_count = 0
        self._direct_accepted_sample_count = 0
        self._direct_acceptance_histogram: dict[int, int] = {}
        self._direct_attempt_histogram: dict[int, int] = {}
        self._direct_finalization_reasons: dict[str, int] = {}
        self._direct_box_proposals: list[dict] = []
        self._pending_translation_targets: dict[int, tuple[np.ndarray, dict]] = {}
        self._pending_joint_target = None
        self._joint_directions = (
            fibonacci_directions(self.joint_direction_count)
            if self.joint_direction_count else None
        )

    def _ensure_pending_direct_residual(self):
        """Ensure pending direct residual."""
        if self.residual_sampler.mode not in DIRECT_RESIDUAL_SAMPLER_MODES:
            raise ValueError("pending direct residual requested in another mode")
        residual = self._pending_direct_residual
        if residual is None:
            if self.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                residual = sample_direct_left_euler_box_correction(
                    self._rng,
                    self.level.max_rotation_deg,
                    float(self.level.max_translation_component_m),
                )
            else:
                residual = fixed_identity_correction()
            self._pending_direct_residual = residual
            self._pending_direct_attempt_count = 0
            self._pending_direct_accepted_count = 0
            if self.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                components, normalized = direct_box_coordinates_from_sampling(
                    residual.sampling
                )
                self._direct_box_proposals.append({
                    "residual_draw_index": self._direct_residual_draw_count,
                    "direct_box_components": components,
                    "direct_box_normalized": normalized,
                })
            self._direct_residual_draw_count += 1
        return residual

    def _finalize_pending_direct_residual(self, reason: str) -> None:
        """Finalize pending direct residual."""
        if self._pending_direct_residual is None:
            raise ValueError("no pending direct residual to finalize")
        accepted = self._pending_direct_accepted_count
        attempts = self._pending_direct_attempt_count
        if accepted:
            self._direct_residual_accepted_count += 1
            self._direct_acceptance_histogram[accepted] = (
                self._direct_acceptance_histogram.get(accepted, 0) + 1
            )
        else:
            self._direct_residual_rejected_count += 1
        self._direct_attempt_histogram[attempts] = (
            self._direct_attempt_histogram.get(attempts, 0) + 1
        )
        self._direct_finalization_reasons[reason] = (
            self._direct_finalization_reasons.get(reason, 0) + 1
        )
        self._pending_direct_residual = None
        self._pending_direct_attempt_count = 0
        self._pending_direct_accepted_count = 0

    def discard_pending_direct_residual(self) -> None:
        """Return discard pending direct residual."""
        if self.residual_sampler.mode != DIRECT_LEFT_EULER_BOX:
            raise ValueError("only Direct-Euler residuals may be discarded")
        self._finalize_pending_direct_residual("no_scale_feasible_pair")

    def finalize_direct_residual_stream(self) -> None:
        """Finalize direct residual stream."""
        if self._pending_direct_residual is not None:
            self._finalize_pending_direct_residual("sequence_end")

    def pending_direct_true_translation_norm(self) -> float:
        """Return pending direct true translation norm."""
        residual = self._ensure_pending_direct_residual()
        desired_true_translation = (
            residual.translation
            + residual.rotation @ self.nominal[:3, 3]
        )
        norm = float(np.linalg.norm(desired_true_translation))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(
                "sampled direct residual requires degenerate translation"
            )
        return norm

    def propose(
        self,
        source: OmniFrame,
        target: OmniFrame,
        pair_id: str,
        radius_interval_index: int = -1,
        radius_interval: Optional[tuple[float, float]] = None,
        radius_interval_target: int = 0,
        accepted_in_interval: int = 0,
        joint_target: Optional[JointCellTarget] = None,
        rotation_interval: Optional[tuple[float, float]] = None,
        gravity_bucket: Optional[str] = None,
    ) -> Tuple[Optional[dict], dict]:
        """Return propose."""
        proposal_index = self.proposal_count
        self.proposal_count += 1
        rotation_sampling = None
        if self.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES:
            if joint_target is not None:
                raise ValueError("direct residual mode cannot target joint cells")
            residual_proposal = self._ensure_pending_direct_residual()
            self._pending_direct_attempt_count += 1
            delta_rotation = residual_proposal.rotation
            normalized_translation = (
                residual_proposal.translation
                / self.level.residual_translation_normalization_m(self.nominal)
            )
            translation_sampling = {
                "distribution": (
                    "fixed_identity_true_extrinsic"
                    if self.residual_sampler.mode != DIRECT_LEFT_EULER_BOX
                    else "direct_left_translation_component_box"
                ),
                "normalized_target": normalized_translation.tolist(),
            }
            if self.residual_sampler.mode == DIRECT_LEFT_EULER_BOX:
                translation_sampling.update({
                    "sampled_translation_xyz_m": (
                        residual_proposal.sampling[
                            "sampled_translation_xyz_m"
                        ]
                    ),
                    "effective_translation_component_limit_m": (
                        self.level.max_translation_component_m
                    ),
                })
            else:
                translation_sampling.update({
                    "true_extrinsic_policy": "Ti_equals_T0",
                    "generation_time_perturbation": False,
                })
            rotation_sampling = {
                **residual_proposal.sampling,
                "residual_draw_index": self._direct_residual_draw_count - 1,
                "realization_attempt_index": (
                    self._pending_direct_attempt_count
                ),
                "rejection_policy": self.residual_sampler.rejection_policy,
            }
        elif self.accepted_distribution_control == REPORT_ONLY:
            delta_rotation, rotvec = sample_uniform_so3_ball(
                self._rng, self.level.max_rotation_deg
            )
            normalized_translation = (
                float(self._rng.random()) * sample_uniform_s2(self._rng)
            )
            translation_sampling = {
                "distribution": "random_uniform_radius_uniform_s2",
                "normalized_target": normalized_translation.tolist(),
            }
            rotation_sampling = {
                "mode": GEODESIC_AXIS_ANGLE,
                "rotvec_rad": rotvec.tolist(),
            }
        elif joint_target is None:
            delta_rotation, _ = sample_uniform_so3_ball(
                self._rng, self.level.max_rotation_deg
            )
            pending = self._pending_translation_targets.get(
                radius_interval_index
            )
            if pending is None:
                pending = sample_stratified_translation_target(
                    self._rng,
                    radius_interval,
                    accepted_in_interval,
                    radius_interval_target,
                )
                self._pending_translation_targets[
                    radius_interval_index
                ] = pending
            normalized_translation, translation_sampling = pending
        else:
            if self._joint_directions is None or rotation_interval is None:
                raise ValueError(
                    "joint target requires direction centers and "
                    "a rotation interval"
                )
            if (
                joint_target.translation_interval_index
                != radius_interval_index
            ):
                raise ValueError(
                    "joint target translation interval mismatch"
                )
            target_key = (
                joint_target.index,
                joint_target.translation_radius_cell_index,
                joint_target.rotation_radius_cell_index,
            )
            if min(target_key[1:]) < 0:
                raise ValueError(
                    "joint target lacks fine radius-cell indices"
                )
            pending = self._pending_joint_target
            if pending is None:
                rotation_target = sample_stratified_rotation_target(
                    self._rng,
                    self.level.max_rotation_deg,
                    rotation_interval,
                    0,
                    1,
                    joint_target.rotation_direction_index,
                    self._joint_directions,
                )
                translation_target = sample_stratified_translation_target(
                    self._rng,
                    radius_interval,
                    0,
                    1,
                    direction_cell_index=(
                        joint_target.translation_direction_index
                    ),
                    directions=self._joint_directions,
                )
                pending = (target_key, translation_target, rotation_target)
                self._pending_joint_target = pending
            elif pending[0] != target_key:
                raise RuntimeError(
                    "pending joint sample belongs to another quota cell"
                )
            normalized_translation, translation_sampling = pending[1]
            delta_rotation, _, rotation_sampling = pending[2]
        view_roll = (
            2.0 * np.pi * float(self._rng.random())
            if (
                self.view_rotation_distribution == VIEW_ROTATION_DISTRIBUTION
                and not self.camera_optical_axis_latitude_band_enabled
            ) else None
        )
        return construct_targeted_stochastic_candidate(
            source=source,
            target=target,
            pair_id=pair_id,
            nominal=self.nominal,
            level=self.level,
            delta_rotation=delta_rotation,
            camera_optical_axis_latitude_band_enabled=(
                self.camera_optical_axis_latitude_band_enabled
            ),
            max_camera_optical_axis_latitude_deg=(
                self.max_camera_optical_axis_latitude_deg
            ),
            view_rng=self._rng,
            normalized_residual_translation=normalized_translation,
            translation_sampling=translation_sampling,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            view_roll=view_roll,
            proposal_index=proposal_index,
            radius_interval_index=radius_interval_index,
            rotation_sampling=rotation_sampling,
            joint_cell_target=joint_target,
            view_rotation_distribution=self.view_rotation_distribution,
            max_level_rig_tilt_deg=self.max_level_rig_tilt_deg,
            gravity_bucket=gravity_bucket,
        )

    def discard_translation_target(
        self,
        radius_interval_index: int = -1,
        joint_target: Optional[JointCellTarget] = None,
    ) -> None:
        """Return discard translation target."""
        if joint_target is not None:
            if (
                self._pending_joint_target is None
                or self._pending_joint_target[0][0] != joint_target.index
            ):
                raise ValueError(
                    "discarded joint cell has no pending target"
                )
            self._pending_joint_target = None
            return
        if radius_interval_index not in self._pending_translation_targets:
            raise ValueError(
                "discarded interval has no pending translation target"
            )
        del self._pending_translation_targets[radius_interval_index]

    def record_translation_target_accepted(
        self,
        radius_interval_index: int = -1,
        joint_target: Optional[JointCellTarget] = None,
    ) -> None:
        """Record translation target accepted."""
        if joint_target is not None:
            if (
                self._pending_joint_target is None
                or self._pending_joint_target[0][0] != joint_target.index
            ):
                raise ValueError(
                    "accepted joint cell has no pending target"
                )
            self._pending_joint_target = None
            return
        if radius_interval_index not in self._pending_translation_targets:
            raise ValueError(
                "accepted interval has no pending translation target"
            )
        del self._pending_translation_targets[radius_interval_index]

    def record_candidate_accepted(
        self,
        radius_interval_index: int = -1,
        joint_target: Optional[JointCellTarget] = None,
    ) -> None:
        """Record candidate accepted."""
        if self.residual_sampler.mode in DIRECT_RESIDUAL_SAMPLER_MODES:
            if self._pending_direct_residual is None:
                raise ValueError(
                    "accepted direct sample has no pending residual"
                )
            self._pending_direct_accepted_count += 1
            self._direct_accepted_sample_count += 1
            if (
                self._pending_direct_accepted_count
                >= self.residual_sampler.max_samples_per_residual
            ):
                self._finalize_pending_direct_residual("accepted_sample_cap")
            elif (
                self.residual_sampler.mode == DIRECT_LEFT_EULER_BOX
                and self.residual_sampler.max_realization_attempts_per_residual
                is not None
                and self._pending_direct_attempt_count
                >= self.residual_sampler.max_realization_attempts_per_residual
            ):
                self._finalize_pending_direct_residual("realization_attempt_cap")
            return
        if self.accepted_distribution_control != REPORT_ONLY:
            self.record_translation_target_accepted(
                radius_interval_index,
                joint_target=joint_target,
            )

    def record_candidate_rejected(self) -> bool:
        """Record candidate rejected."""
        if self.residual_sampler.mode != DIRECT_LEFT_EULER_BOX:
            return False
        if self._pending_direct_residual is None:
            raise ValueError("rejected direct sample has no pending residual")
        limit = self.residual_sampler.max_realization_attempts_per_residual
        if (
            limit is not None
            and self._pending_direct_attempt_count >= limit
        ):
            self._finalize_pending_direct_residual("realization_attempt_cap")
            return True
        return False

    def direct_box_proposals(self) -> list[dict]:
        """Return direct box proposals."""
        return [dict(value) for value in self._direct_box_proposals]


    def residual_distribution_report(self) -> dict:
        """Return residual distribution report."""
        if self.residual_sampler.mode not in DIRECT_RESIDUAL_SAMPLER_MODES:
            return {"enabled": False}
        pending = self._pending_direct_residual is not None
        target_distribution = (
            "fixed_identity_true_extrinsic"
            if self.residual_sampler.mode != DIRECT_LEFT_EULER_BOX
            else "direct_left_euler_translation_component_box"
        )
        return {
            "enabled": True,
            "target_distribution": target_distribution,
            "rejection_policy": self.residual_sampler.rejection_policy,
            "residual_draw_count": self._direct_residual_draw_count,
            "accepted_residual_count": (
                self._direct_residual_accepted_count
            ),
            "accepted_sample_count": self._direct_accepted_sample_count,
            "accepted_samples_per_residual_histogram": {
                str(key): value
                for key, value in sorted(
                    self._direct_acceptance_histogram.items()
                )
            },
            "rejected_residual_count": self._direct_residual_rejected_count,
            "pending_residual": pending,
            "pending_realization_attempts": (
                self._pending_direct_attempt_count
            ),
            "pending_accepted_sample_count": (
                self._pending_direct_accepted_count
            ),
            "realization_attempts_per_residual_histogram": {
                str(key): value
                for key, value in sorted(self._direct_attempt_histogram.items())
            },
            "residual_finalization_reasons": dict(
                sorted(self._direct_finalization_reasons.items())
            ),
            "no_sampled_residual_skipped": (
                self.residual_sampler.mode != DIRECT_LEFT_EULER_BOX
                and self._direct_residual_rejected_count == 0
            ),
            "all_drawn_residuals_accounted_for": (
                self._direct_residual_draw_count
                == self._direct_residual_accepted_count
                + self._direct_residual_rejected_count
                + int(pending)
            ),
            "pending_sampling": (
                self._pending_direct_residual.sampling if pending else None
            ),
        }


__all__ = [
    "CameraDepthStage",
    "GeneratedSensorSample",
    "LidarStage",
    "ExtrinsicSampleGenerator",
    "clear_rgb_cache",
    "configure_rgb_cache",
    "feasible_direction_probability",
    "finalize_sensor_sample",
    "generate_candidate_camera_depth",
    "generate_candidate_lidar",
    "rgb_cache_report",
]
