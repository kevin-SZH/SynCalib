"""Provide the unified ExtrinsicManifoldData command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from ._runtime_log import emit_runtime_log
from .generation import GenerationOptions, generate_dataset
from .quality_export import StochasticGenerationError


def _write_failure(output_root: Path, report: dict[str, Any]) -> None:
    """Write failure."""
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "task_failure.json"
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _print_json(value: dict[str, Any], stream: Any = None) -> None:
    """Print JSON."""
    print(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        file=stream,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate KITTI-style camera-LiDAR extrinsic calibration data from "
            "TartanGround cubemaps. One YAML controls the generation mode and thresholds."
        )
    )
    parser.add_argument("--config", required=True, help="self-contained task YAML")
    parser.add_argument("--data-root", required=True, help="TartanGround root directory")
    parser.add_argument("--output-root", required=True, help="KITTI output directory")
    parser.add_argument(
        "--scene",
        dest="scenes",
        action="append",
        help="generate only this scene; repeatable and restricted to smoke tests",
    )
    parser.add_argument(
        "--level",
        dest="levels",
        action="append",
        help="generate only this YAML curriculum level; repeatable",
    )
    parser.add_argument(
        "--accepted-target-override",
        type=int,
        help="override accepted samples per sequence for smoke tests",
    )
    parser.add_argument(
        "--allow-incomplete-smoke",
        action="store_true",
        help="allow a stalled smoke test to keep fewer samples than requested",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="rebuild the source manifest and reuse only fully audited sequences",
    )
    parser.add_argument(
        "--continue-on-coverage-failure",
        action="store_true",
        help="continue with remaining curriculum levels after a coverage failure",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="scene worker processes per curriculum level (default: 1)",
    )
    parser.add_argument(
        "--cache-root",
        help="qualified-pair cache directory (default: <output-root>/.cache)",
    )
    parser.add_argument(
        "--derive-lidar-from",
        help=(
            "derive a new LiDAR modality while preserving source samples, images, and labels"
        ),
    )
    parser.add_argument(
        "--derive-lidar-config",
        help="LiDAR derivation YAML defining the scanner, range, and accumulation time",
    )
    report_mode = parser.add_mutually_exclusive_group()
    report_mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the YAML and T0 without scanning or generating data",
    )
    report_mode.add_argument(
        "--manifest-only",
        action="store_true",
        help="scan source data and write only the scene-holdout manifest",
    )
    report_mode.add_argument(
        "--audit-existing-sequence",
        help=(
            "rescore an existing KITTI sequence and export per-sample reprojection, "
            "Sobel-ratio, and quantile audits without modifying the sequence"
        ),
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> GenerationOptions:
    """Return options from args."""
    return GenerationOptions(
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        scenes=args.scenes,
        levels=args.levels,
        accepted_target_override=args.accepted_target_override,
        allow_incomplete_smoke=args.allow_incomplete_smoke,
        resume=args.resume,
        continue_on_coverage_failure=args.continue_on_coverage_failure,
        workers=args.workers,
        cache_root=args.cache_root,
        manifest_only=args.manifest_only,
        validate_only=args.validate_only,
        derive_lidar_from=args.derive_lidar_from,
        derive_lidar_config=args.derive_lidar_config,
        audit_existing_sequence=args.audit_existing_sequence,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface and return its exit code."""
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()

    failure_root = (
        output_root
        if args.derive_lidar_from is None
        else output_root.with_name(f".{output_root.name}.failure")
    )
    try:
        report = generate_dataset(_options_from_args(args))
    except StochasticGenerationError as exc:
        report = {"status": "failed", "error": str(exc), **exc.report}
        _write_failure(failure_root, report)
        emit_runtime_log(
            "task_failed",
            "Dataset generation did not satisfy the release requirements",
            error=f"{type(exc).__name__}: {exc}",
            output_root=str(output_root),
        )
        _print_json(report, stream=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, KeyError, ValueError, OSError) as exc:
        report = {"status": "error", "error": str(exc)}
        _write_failure(failure_root, report)
        emit_runtime_log(
            "task_failed",
            "The dataset generation entry point failed",
            error=f"{type(exc).__name__}: {exc}",
            output_root=str(output_root),
        )
        _print_json(report, stream=sys.stderr)
        return 1

    failure_path = failure_root / "task_failure.json"
    if failure_path.exists():
        failure_path.unlink()
    if failure_root != output_root and failure_root.is_dir():
        failure_root.rmdir()
    _print_json(report)
    return 0
