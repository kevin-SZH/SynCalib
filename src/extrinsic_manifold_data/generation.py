"""Orchestrate configuration, pair pooling, sample generation, and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Union

from ._runtime_log import emit_runtime_log
from .sample_generation import FIXED_IDENTITY
from .pair_pool import build_frame_pair_manifest
from .generation_config import (
    GenerationTask,
    load_generation_task,
)
from .quality_export import (
    audit_existing_sequence_quality,
    derive_lidar_modality_dataset,
    filter_and_export_dataset,
)


SOURCE_MANIFEST_RELATIVE_PATH = Path("_source") / "source_manifest.json"


FIXED_T0_MODE = "fixed_t0"
NEIGHBORHOOD_TI_MODE = "neighborhood_ti"


@dataclass(frozen=True)
class GenerationOptions:
    """Represent generation options."""

    config_path: Union[str, Path]
    data_root: Union[str, Path]
    output_root: Union[str, Path]
    scenes: Optional[Iterable[str]] = None
    levels: Optional[Iterable[str]] = None
    accepted_target_override: Optional[int] = None
    allow_incomplete_smoke: bool = False
    resume: bool = False
    continue_on_coverage_failure: bool = False
    workers: int = 1
    cache_root: Optional[Union[str, Path]] = None
    manifest_only: bool = False
    validate_only: bool = False
    derive_lidar_from: Optional[Union[str, Path]] = None
    derive_lidar_config: Optional[Union[str, Path]] = None
    audit_existing_sequence: Optional[Union[str, Path]] = None


def _generation_mode(task: GenerationTask) -> str:
    """Return generation mode."""
    if task.residual_sampler.mode == FIXED_IDENTITY:
        return FIXED_T0_MODE
    return NEIGHBORHOOD_TI_MODE


def _validate_options(options: GenerationOptions) -> None:
    """Validate options."""
    if options.validate_only and options.manifest_only:
        raise ValueError("validate_only and manifest_only are mutually exclusive")
    if options.derive_lidar_from is not None and (
        options.validate_only or options.manifest_only or options.resume
        or options.audit_existing_sequence is not None
        or options.scenes is not None
        or options.levels is not None
        or options.accepted_target_override is not None
        or options.allow_incomplete_smoke
        or options.continue_on_coverage_failure
        or options.cache_root is not None
    ):
        raise ValueError(
            "derive_lidar_from cannot be combined with validate_only, "
            "manifest_only, resume, audit, generation subset, smoke, "
            "coverage, or pair-cache options"
        )
    if options.audit_existing_sequence is not None and (
        options.validate_only
        or options.manifest_only
        or options.scenes is not None
        or options.levels is not None
        or options.accepted_target_override is not None
        or options.allow_incomplete_smoke
        or options.continue_on_coverage_failure
        or options.cache_root is not None
    ):
        raise ValueError(
            "audit_existing_sequence cannot be combined with generation, "
            "manifest, smoke, coverage, or pair-cache options"
        )
    if (options.derive_lidar_from is None) != (
        options.derive_lidar_config is None
    ):
        raise ValueError(
            "derive_lidar_from and derive_lidar_config must be provided together"
        )
    if options.workers < 1:
        raise ValueError("workers must be positive")
    if options.accepted_target_override is not None:
        if options.accepted_target_override < 1:
            raise ValueError("accepted_target_override must be positive")
    elif options.scenes is not None:
        raise ValueError(
            "scene subsets are smoke-only and require accepted_target_override"
        )


def _validation_report(
    task: GenerationTask,
    config_path: Path,
) -> dict[str, Any]:
    """Return validation report."""
    return {
        "status": "valid",
        "generation_mode": _generation_mode(task),
        "config": str(config_path),
        "target_network": task.target_network,
        "seed": task.seed,
        "nominal_extrinsic": task.nominal.to_metadata(),
        "levels": [level.name for level in task.levels],
    }


def generate_dataset(options: GenerationOptions) -> dict[str, Any]:
    """Generate dataset."""
    task_start = perf_counter()
    emit_runtime_log(
        "task_started",
        "Starting dataset generation",
        config=str(options.config_path),
        output_root=str(options.output_root),
        workers=options.workers,
        resume=options.resume,
    )
    _validate_options(options)


    config_path = Path(options.config_path).resolve()
    data_root = Path(options.data_root).resolve()
    output_root = Path(options.output_root).resolve()


    phase_start = perf_counter()
    emit_runtime_log(
        "stage_started", "Loading and validating the task configuration", config=str(config_path)
    )
    task = load_generation_task(config_path)
    validation = _validation_report(task, config_path)
    emit_runtime_log(
        "stage_completed",
        "Task configuration validated",
        generation_mode=validation["generation_mode"],
        levels=validation["levels"],
        elapsed_seconds=round(perf_counter() - phase_start, 3),
    )
    if options.validate_only:
        emit_runtime_log(
            "task_completed",
            "Validation-only mode completed",
            elapsed_seconds=round(perf_counter() - task_start, 3),
        )
        return validation

    if not data_root.is_dir():
        raise FileNotFoundError(f"TartanGround data root does not exist: {data_root}")

    if options.audit_existing_sequence is not None:
        phase_start = perf_counter()
        emit_runtime_log(
            "stage_started",
            "Rescoring an existing sequence and rendering per-sample quality audits",
            source_sequence=str(options.audit_existing_sequence),
            output_root=str(output_root),
            workers=options.workers,
        )
        report = audit_existing_sequence_quality(
            task=task,
            sequence_directory=options.audit_existing_sequence,
            output_directory=output_root,
            data_root=data_root,
            workers=options.workers,
            resume=options.resume,
        )
        emit_runtime_log(
            "task_completed",
            "Existing-sequence quality audit completed",
            output_root=str(output_root),
            sample_count=report["sample_count"],
            elapsed_seconds=round(perf_counter() - phase_start, 3),
        )
        return {
            **report,
            "generation_mode": validation["generation_mode"],
            "config": str(config_path),
        }

    if options.derive_lidar_from is not None:
        phase_start = perf_counter()
        emit_runtime_log(
            "stage_started",
            "Deriving a LiDAR control dataset from existing sample identities",
            source_output=str(options.derive_lidar_from),
            output_root=str(output_root),
            workers=options.workers,
        )
        report = derive_lidar_modality_dataset(
            task=task,
            source_output_root=options.derive_lidar_from,
            output_root=output_root,
            data_root=data_root,
            derivation_config_path=options.derive_lidar_config,
            workers=options.workers,
        )
        emit_runtime_log(
            "task_completed",
            "LiDAR control dataset derivation completed",
            output_root=str(output_root),
            sample_count=report["sample_count"],
            elapsed_seconds=round(perf_counter() - phase_start, 3),
        )
        return {
            **report,
            "generation_mode": validation["generation_mode"],
            "config": str(config_path),
        }


    source_manifest_path = output_root / SOURCE_MANIFEST_RELATIVE_PATH
    if source_manifest_path.exists() and not options.resume:
        raise FileExistsError(
            "source manifest already exists; pass resume to rebuild and audit it: "
            f"{source_manifest_path}"
        )
    phase_start = perf_counter()
    emit_runtime_log(
        "stage_started",
        "Scanning source data and building the source manifest",
        data_root=str(data_root),
        source_manifest=str(source_manifest_path),
    )
    source_manifest = build_frame_pair_manifest(
        data_root=data_root,
        config_path=config_path,
        output_path=source_manifest_path,
        scenes=options.scenes,
    )
    emit_runtime_log(
        "stage_completed",
        "Source manifest built",
        sequence_count=source_manifest["sequence_count"],
        frame_count=source_manifest["frame_count"],
        elapsed_seconds=round(perf_counter() - phase_start, 3),
    )
    if options.manifest_only:
        emit_runtime_log(
            "task_completed",
            "Manifest-only mode completed",
            elapsed_seconds=round(perf_counter() - task_start, 3),
        )
        return {
            **validation,
            "status": "manifest_complete",
            "source_manifest": str(source_manifest_path),
            "sequence_count": source_manifest["sequence_count"],
            "frame_count": source_manifest["frame_count"],
            "scene_splits": source_manifest["scene_splits"],
        }


    phase_start = perf_counter()
    emit_runtime_log(
        "stage_started",
        "Generating, filtering, and exporting scene data",
        workers=options.workers,
        paired_lidar_export=task.paired_lidar_export is not None,
    )
    export_report = filter_and_export_dataset(
        task=task,
        source_manifest=source_manifest_path,
        output_root=output_root,
        scenes=options.scenes,
        levels=options.levels,
        accepted_target_override=options.accepted_target_override,
        allow_incomplete_smoke=options.allow_incomplete_smoke,
        resume_existing=options.resume,
        continue_on_coverage_failure=options.continue_on_coverage_failure,
        workers=options.workers,
        cache_root=options.cache_root,
    )
    emit_runtime_log(
        "stage_completed",
        "All scene generation and export tasks completed",
        job_count=export_report["job_count"],
        elapsed_seconds=round(perf_counter() - phase_start, 3),
    )
    emit_runtime_log(
        "task_completed",
        "Dataset generation completed",
        output_root=str(output_root),
        elapsed_seconds=round(perf_counter() - task_start, 3),
    )
    return {
        **export_report,
        "generation_mode": validation["generation_mode"],
        "config": str(config_path),
        "source_manifest": str(source_manifest_path),
    }


__all__ = ["GenerationOptions", "generate_dataset"]
