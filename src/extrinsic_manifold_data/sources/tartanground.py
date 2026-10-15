"""Load TartanGround cubemap RGB, packed depth, and ground-truth poses."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import validate_transform
from ..omni import OmniFrame, OmniSequence


# Stable cubemap face order shared by source manifests and projection maps.
FACE_NAMES = ("front", "right", "back", "left", "top", "bottom")


# Rotate native +X forward, +Y right, +Z down into internal +X forward, +Y left, +Z up.
TARTAN_TO_PANO = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

# Inverse rotation for querying the original cubemap coordinates.
PANO_TO_TARTAN = TARTAN_TO_PANO.T


def _load_pose_rows(path: Path) -> np.ndarray:
    """Load pose rows."""
    rows = np.loadtxt(str(path), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.all(np.isfinite(rows)):
        raise ValueError(f"{path}: poses must be finite Nx7 x y z qx qy qz qw rows")
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        index = int(np.argmax(np.abs(norms - 1.0)))
        raise ValueError(f"{path}: quaternion row {index} has norm {norms[index]}")
    rows = rows.copy()
    rows[:, 3:7] /= norms[:, None]
    return rows


def _pose_matrices(rows: np.ndarray) -> np.ndarray:
    """Return pose matrices."""
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(rows), axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    transforms[:, :3, 3] = rows[:, :3]
    return transforms


def _frame_paths(directory: Path, suffix: str) -> Dict[str, Path]:
    """Return frame paths."""
    paths: Dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        frame_id = path.name[: -len(suffix)]
        if not frame_id.isdigit():
            raise ValueError(f"unexpected frame filename: {path}")
        if frame_id in paths:
            raise ValueError(f"duplicate frame {frame_id}: {directory}")
        if path.stat().st_size == 0:
            raise ValueError(f"empty source file: {path}")
        paths[frame_id] = path
    if not paths:
        raise ValueError(f"no frames matching *{suffix}: {directory}")
    return paths


def _validate_frame_ids(path_sets: Iterable[Dict[str, Path]]) -> list[str]:
    """Validate frame IDs."""
    values = list(path_sets)
    expected = set(values[0])
    for paths in values[1:]:
        if set(paths) != expected:
            missing = sorted(expected - set(paths))[:5]
            extra = sorted(set(paths) - expected)[:5]
            raise ValueError(
                f"cubemap RGB/depth frame IDs differ: missing={missing}, extra={extra}"
            )
    ordered = sorted(expected, key=int)
    indices = [int(frame_id) for frame_id in ordered]
    if indices != list(range(indices[-1] + 1)):
        raise ValueError("raw frame IDs must be contiguous and start at zero")
    return ordered


def _validate_sample_images(
    rgb_paths: Dict[str, Dict[str, Path]],
    depth_paths: Dict[str, Dict[str, Path]],
    frame_ids: list[str],
) -> tuple[list[int], list[int]]:
    """Validate sample images."""
    sample_ids = sorted({frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]})
    rgb_shape = None
    depth_shape = None
    for frame_id in sample_ids:
        for face in FACE_NAMES:
            image = cv2.imread(str(rgb_paths[face][frame_id]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable RGB image: {rgb_paths[face][frame_id]}")
            packed_depth = cv2.imread(
                str(depth_paths[face][frame_id]), cv2.IMREAD_UNCHANGED
            )
            if (
                packed_depth is None
                or packed_depth.dtype != np.uint8
                or packed_depth.ndim != 3
                or packed_depth.shape[2] != 4
            ):
                raise ValueError(
                    f"depth is not an RGBA-packed float32 PNG: "
                    f"{depth_paths[face][frame_id]}"
                )
            current_rgb = image.shape[:2]
            current_depth = packed_depth.shape[:2]
            if rgb_shape is None:
                rgb_shape = current_rgb
                depth_shape = current_depth
            if current_rgb != rgb_shape or current_depth != depth_shape:
                raise ValueError("cubemap face dimensions are inconsistent")
            if current_rgb != current_depth:
                raise ValueError("RGB and depth cubemap dimensions differ")
    assert rgb_shape is not None and depth_shape is not None
    return [int(rgb_shape[1]), int(rgb_shape[0])], [
        int(depth_shape[1]), int(depth_shape[0])
    ]


def _input_fingerprint(paths: Iterable[Path], source_dir: Path) -> str:
    """Return input fingerprint."""
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(source_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        stat = path.stat()
        digest.update(str(stat.st_size).encode("ascii"))
        if path.name.startswith("pose_lcam_"):
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def scan_tartanground_sequence(
    data_root: str | Path,
    scene: str,
    trajectory: str = "P0000",
) -> OmniSequence:
    """Scan tartanground sequence."""
    data_root = Path(data_root).resolve()
    source_dir = data_root / scene / "Data_omni" / trajectory
    if not source_dir.is_dir():
        raise FileNotFoundError(f"raw TartanGround sequence does not exist: {source_dir}")

    pose_paths = {
        face: source_dir / f"pose_lcam_{face}.txt" for face in FACE_NAMES
    }
    missing_poses = [str(path) for path in pose_paths.values() if not path.is_file()]
    if missing_poses:
        raise FileNotFoundError(f"missing raw cubemap pose files: {missing_poses}")
    pose_rows = {face: _load_pose_rows(path) for face, path in pose_paths.items()}
    pose_counts = {len(rows) for rows in pose_rows.values()}
    if len(pose_counts) != 1:
        raise ValueError(f"cubemap pose counts differ: {sorted(pose_counts)}")
    transforms = {face: _pose_matrices(rows) for face, rows in pose_rows.items()}

    front = transforms["front"]
    pano_from_face: Dict[str, np.ndarray] = {}
    for face in FACE_NAMES:
        centers = np.linalg.norm(
            transforms[face][:, :3, 3] - front[:, :3, 3], axis=1
        )
        if float(centers.max()) > 1e-4:
            raise ValueError(
                f"{scene}/{trajectory}: {face} center differs from front by "
                f"{float(centers.max())} m"
            )
        front_from_face = np.einsum(
            "nij,njk->nik",
            np.transpose(front[:, :3, :3], (0, 2, 1)),
            transforms[face][:, :3, :3],
        )
        drift = Rotation.from_matrix(
            np.einsum("nij,jk->nik", front_from_face, front_from_face[0].T)
        ).magnitude()
        if float(drift.max()) > 1e-4:
            raise ValueError(
                f"{scene}/{trajectory}: {face} rig rotation is not static; "
                f"max drift={float(drift.max())} rad"
            )
        pano_from_face[face] = TARTAN_TO_PANO @ front_from_face[0]

    rgb_paths = {
        face: _frame_paths(
            source_dir / f"image_lcam_{face}", f"_lcam_{face}.png"
        )
        for face in FACE_NAMES
    }
    depth_paths = {
        face: _frame_paths(
            source_dir / f"depth_lcam_{face}", f"_lcam_{face}_depth.png"
        )
        for face in FACE_NAMES
    }
    frame_ids = _validate_frame_ids([*rgb_paths.values(), *depth_paths.values()])
    max_index = int(frame_ids[-1])
    pose_count = pose_counts.pop()
    if max_index >= pose_count:
        raise ValueError(
            f"{scene}/{trajectory}: frame {max_index} has no pose row; count={pose_count}"
        )
    rgb_size, depth_size = _validate_sample_images(
        rgb_paths, depth_paths, frame_ids
    )

    frames = []
    pano_to_tartan = np.eye(4, dtype=np.float64)
    pano_to_tartan[:3, :3] = PANO_TO_TARTAN
    for frame_id in frame_ids:
        index = int(frame_id)
        world_from_pano = validate_transform(front[index] @ pano_to_tartan)
        frames.append(OmniFrame(
            frame_id=frame_id,
            frame_index=index,
            world_from_pano=world_from_pano,
            rgb_paths={face: rgb_paths[face][frame_id] for face in FACE_NAMES},
            depth_paths={face: depth_paths[face][frame_id] for face in FACE_NAMES},
        ))

    fingerprint_paths = list(pose_paths.values())
    for paths in (*rgb_paths.values(), *depth_paths.values()):
        fingerprint_paths.extend(paths.values())
    return OmniSequence(
        dataset="tartanground",
        scene=scene,
        trajectory=trajectory,
        sequence_id=f"{scene}_{trajectory}",
        source_dir=source_dir,
        pano_from_face=pano_from_face,
        frames=tuple(frames),
        source_report={
            "frame_count": len(frames),
            "pose_row_count": int(pose_count),
            "unused_pose_rows": int(pose_count - len(frames)),
            "rgb_face_size": rgb_size,
            "depth_face_size": depth_size,
            "input_fingerprint": _input_fingerprint(
                fingerprint_paths, source_dir
            ),
        },
    )
