"""Provide SE(3), SO(3), and pinhole projection utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def validate_transform(transform: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Return a finite 4x4 SE(3) matrix or raise ValueError.

    ``atol`` bounds homogeneous-row, orthogonality, and determinant errors.
    """
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("transform has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > atol:
        raise ValueError(f"rotation determinant is {determinant}, expected 1")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Return the rigid inverse of a validated 4x4 transform."""
    transform = validate_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def residual_transform(true_extrinsic: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """Return the complete left residual Ti @ inverse(T0).

    Both inputs map LiDAR to camera coordinates, with translations in meters.
    The residual translation is ti - delta_R @ t0, rather than ti - t0.
    """
    return validate_transform(true_extrinsic) @ invert_transform(nominal)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm of a 3x3 rotation, in radians."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    return Rotation.from_matrix(rotation).as_rotvec()


def residual_coordinates(
    true_extrinsic: np.ndarray,
    nominal: np.ndarray,
    max_translation: float,
    max_rotation_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the residual matrix, translation, rotation vector, and audit coordinates.

    Translation and rotation-vector limits are in meters and radians respectively.
    Normalized coordinates describe coverage; they do not define a physical metric.
    """
    if max_translation <= 0.0 or max_rotation_rad <= 0.0:
        raise ValueError("normalization limits must be positive")
    residual = residual_transform(true_extrinsic, nominal)
    translation = residual[:3, 3].copy()
    rotvec = rotation_vector(residual[:3, :3])
    normalized = np.concatenate(
        [translation / max_translation, rotvec / max_rotation_rad]
    )
    return residual, translation, rotvec, normalized


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a column-vector SE(3) transform to an Nx3 point array stored by rows."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    transform = validate_transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(
    points_lidar: np.ndarray,
    transform: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points using a LiDAR-to-camera transform and pinhole K.

    Return Nx2 pixel coordinates, axial camera depths in meters, and a mask of
    finite points in front of the camera within the width/height pixel bounds.
    """
    points_camera = transform_points(points_lidar, transform)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must be 3x3")
    depth = points_camera[:, 2]
    finite = np.all(np.isfinite(points_camera), axis=1)
    front = finite & (depth > 1e-8)
    safe_depth = np.where(front, depth, 1.0)
    normalized_x = points_camera[:, 0] / safe_depth
    normalized_y = points_camera[:, 1] / safe_depth

    u = (
        intrinsic[0, 0] * normalized_x
        + intrinsic[0, 1] * normalized_y
        + intrinsic[0, 2]
    )
    v = intrinsic[1, 1] * normalized_y + intrinsic[1, 2]
    in_image = (
        front
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    return np.column_stack((u, v)), depth, in_image
