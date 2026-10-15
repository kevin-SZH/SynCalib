"""Define shared records for omnidirectional frames and sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class OmniFrame:
    """Represent omni frame."""

    frame_id: str
    frame_index: int
    world_from_pano: np.ndarray
    rgb_paths: Dict[str, Path]
    depth_paths: Dict[str, Path]


@dataclass(frozen=True)
class OmniSequence:
    """Represent omni sequence."""

    dataset: str
    scene: str
    trajectory: str
    sequence_id: str
    source_dir: Path
    pano_from_face: Dict[str, np.ndarray]
    frames: Tuple[OmniFrame, ...]
    source_report: Dict[str, Any] = field(default_factory=dict)
