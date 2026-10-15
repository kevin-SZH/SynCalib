#!/usr/bin/env python3
"""Run the unified dataset generation command without installing the console script."""

from extrinsic_manifold_data.cli import main


def run() -> int:
    """Run."""
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
