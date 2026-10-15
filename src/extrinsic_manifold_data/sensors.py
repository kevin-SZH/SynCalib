"""Render pinhole cameras and simulate configured LiDAR sensors."""

from __future__ import annotations


from collections import OrderedDict
from typing import Callable, Dict

import numpy as np


class ArrayMemoryCache:
    """Represent array memory cache."""

    def __init__(self, name: str, max_bytes: int):
        """Initialize the instance."""
        self.name = str(name)
        self._max_bytes = 0
        self._values = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self.configure(max_bytes)

    def configure(self, max_bytes: int) -> None:
        """Set the cache capacity in bytes and clear existing entries."""
        max_bytes = int(max_bytes)
        if max_bytes < 1:
            raise ValueError(f"{self.name} cache max_bytes must be positive")
        self.clear()
        self._max_bytes = max_bytes

    def clear(self) -> None:
        """Clear cached arrays and reset usage counters."""
        self._values.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, loader: Callable[[str], np.ndarray]) -> np.ndarray:
        """Return a cached array, loading it on a cache miss."""
        key = str(key)
        cached = self._values.get(key)
        if cached is not None:
            self._hits += 1
            self._values.move_to_end(key)
            return cached

        self._misses += 1
        value = np.asarray(loader(key))
        if value.dtype.hasobject:
            raise ValueError(f"{self.name} cache refuses object arrays")
        value.flags.writeable = False
        value_bytes = int(value.nbytes)
        if value_bytes <= self._max_bytes:
            while self._values and self._bytes + value_bytes > self._max_bytes:
                _, evicted = self._values.popitem(last=False)
                self._bytes -= int(evicted.nbytes)
                self._evictions += 1
            self._values[key] = value
            self._bytes += value_bytes
        return value

    def report(self) -> Dict[str, object]:
        """Return report."""
        requests = self._hits + self._misses
        return {
            "name": self.name,
            "max_bytes": self._max_bytes,
            "resident_bytes": self._bytes,
            "resident_items": len(self._values),
            "requests": requests,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": float(self._hits / requests) if requests else 0.0,
            "evictions": self._evictions,
        }


from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .omni import OmniFrame


_DEFAULT_DEPTH_CACHE_BYTES = 512 * 1024 * 1024

_DEPTH_CACHE = ArrayMemoryCache("cubemap_depth", _DEFAULT_DEPTH_CACHE_BYTES)


@dataclass(frozen=True)
class PanoDepthProjection:
    """Represent pano depth projection."""

    directions: np.ndarray
    face_names: tuple[str, ...]
    face_index: np.ndarray
    map_x: tuple[np.ndarray, ...]
    map_y: tuple[np.ndarray, ...]
    radial_scale: np.ndarray
    height: int
    width: int
    face_size: int


def pano_directions(height: int, width: int) -> np.ndarray:
    """Return pano directions."""
    if height < 2 or width != 2 * height:
        raise ValueError("pano dimensions must satisfy width=2*height and height>=2")
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    azimuth = (0.5 - uu / width) * (2.0 * np.pi)
    latitude = (0.5 - vv / height) * np.pi
    cos_latitude = np.cos(latitude)
    return np.stack((
        cos_latitude * np.cos(azimuth),
        cos_latitude * np.sin(azimuth),
        np.sin(latitude),
    ), axis=-1)


def build_pano_depth_projection(
    pano_from_face: Dict[str, np.ndarray],
    face_size: int,
    pano_height: int,
) -> PanoDepthProjection:
    """Build pano depth projection."""
    if face_size < 2:
        raise ValueError("face_size must be at least two")
    face_names = tuple(pano_from_face)
    if len(face_names) != 6:
        raise ValueError("an omnidirectional cubemap must contain six faces")
    width = 2 * pano_height
    directions = pano_directions(pano_height, width)
    face_directions = []
    for face in face_names:
        rotation = np.asarray(pano_from_face[face], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"{face} pano_from_face rotation must be 3x3")
        face_directions.append(directions @ rotation)
    face_directions_array = np.stack(face_directions, axis=0)
    forward = face_directions_array[..., 0]
    face_index = np.argmax(forward, axis=0).astype(np.uint8)
    selected_forward = np.take_along_axis(
        forward, face_index[None], axis=0
    )[0]
    if np.any(selected_forward <= 0.0):
        raise ValueError("cubemap face rotations do not cover the full sphere")

    focal = face_size / 2.0
    center = (face_size - 1.0) / 2.0
    map_x = []
    map_y = []
    for direction in face_directions_array:
        safe = np.where(direction[..., 0] > 1e-12, direction[..., 0], 1.0)
        map_x.append((focal * direction[..., 1] / safe + center).astype(np.float32))
        map_y.append((focal * direction[..., 2] / safe + center).astype(np.float32))
    return PanoDepthProjection(
        directions=directions,
        face_names=face_names,
        face_index=face_index,
        map_x=tuple(map_x),
        map_y=tuple(map_y),
        radial_scale=(1.0 / selected_forward).astype(np.float32),
        height=pano_height,
        width=width,
        face_size=face_size,
    )


def _decode_packed_float32_depth(path: str) -> np.ndarray:
    """Decode packed float32 depth."""
    packed = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if (
        packed is None
        or packed.dtype != np.uint8
        or packed.ndim != 3
        or packed.shape[2] != 4
    ):
        raise ValueError(f"depth is not an RGBA-packed float32 PNG: {path}")
    depth = packed.view("<f4").reshape(packed.shape[:2]).astype(
        np.float32, copy=True
    )
    depth.flags.writeable = False
    return depth


def configure_depth_cache(max_bytes: int) -> None:
    """Configure depth cache."""
    _DEPTH_CACHE.configure(max_bytes)


def clear_depth_cache() -> None:
    """Clear depth cache."""
    _DEPTH_CACHE.clear()


def depth_cache_report() -> dict:
    """Return depth cache report."""
    return _DEPTH_CACHE.report()


def load_packed_float32_depth(path) -> np.ndarray:
    """Load packed float32 depth."""
    return _DEPTH_CACHE.get(str(path), _decode_packed_float32_depth)


def load_frame_radial_depth(
    frame: OmniFrame,
    projection: PanoDepthProjection,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Load frame radial depth."""
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    depth = np.zeros((projection.height, projection.width), dtype=np.float32)
    for index, face in enumerate(projection.face_names):
        face_depth = load_packed_float32_depth(frame.depth_paths[face])
        if face_depth.shape != (projection.face_size, projection.face_size):
            raise ValueError(
                f"{face} depth shape {face_depth.shape} does not match "
                f"{projection.face_size}x{projection.face_size}"
            )
        remapped = cv2.remap(
            face_depth,
            projection.map_x[index],
            projection.map_y[index],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = projection.face_index == index
        depth[mask] = remapped[mask]
    depth = depth * projection.radial_scale * float(depth_scale)
    valid = np.isfinite(depth) & (depth > 0.0)
    return np.where(valid, depth, 0.0).astype(np.float32)


import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from .config_sections import config_section_fingerprint, load_config_mapping


PROFILE_NAME = "kitti_odometry_04_12_rectified_hdl64e"


RECTIFIED_1600_PROFILE_NAME = "rectified_pinhole_1600x1296_kitti_hdl64e"


RECTIFIED_1600_WIDE_PROFILE_NAME = (
    "rectified_pinhole_1600x1296_kitti_hdl64e_vfov_m25_p25"
)


MID360_PROFILE_NAME = "livox_mid360_nonrepeating"

RS32_PROFILE_NAME = "robosense_rs32_nonuniform_repeated_ring"

HDL64E_MODEL = "velodyne_hdl64e"
MID360_MODEL = "livox_mid360"
RS32_MODEL = "robosense_rs32"


@dataclass(frozen=True)
class CameraSensorConfig:
    """Represent camera sensor config."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    model: str = "pinhole_rectified"

    @property
    def matrix(self) -> np.ndarray:
        """Return matrix."""
        return np.array([
            [self.fx, self.skew, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass(frozen=True)
class LidarSensorConfig:
    """Represent lidar sensor config."""
    scanner_model: str
    num_rings: int
    points_per_ring: int
    vertical_fov_deg: tuple[float, float]
    horizontal_fov_deg: tuple[float, float]
    repeated_pattern: bool
    min_range_m: float
    max_range_m: float
    beam_pattern_file: Optional[str] = None
    beam_pattern_path: Optional[str] = None
    beam_pattern_sha256: Optional[str] = None
    channel_calibration_file: Optional[str] = None
    channel_calibration_path: Optional[str] = None
    channel_calibration_sha256: Optional[str] = None
    pattern_duration_s: Optional[float] = None
    scan_frequency_hz: Optional[float] = None
    beams_per_frame: Optional[int] = None
    accumulation_duration_s: Optional[float] = None
    interleaved_channels: Optional[int] = None
    pattern_row_count: Optional[int] = None


@dataclass(frozen=True)
class TrainingSensorConfig:
    """Represent training sensor config."""
    profile: str
    camera: CameraSensorConfig
    lidar: LidarSensorConfig

    def to_dict(self) -> dict:
        """Return a dictionary representation."""
        value = asdict(self)
        value["lidar"].pop("beam_pattern_path", None)
        value["lidar"].pop("channel_calibration_path", None)
        return value


LIDAR_DERIVATION_SCHEMA = "lidar_modality_derivation_v1"
REUSE_SOURCE_LIDAR_FRAME = "reuse_source_lidar_frame"
PRESERVE_ALL_SOURCE_SAMPLES = "preserve_all_source_samples"
PHYSICAL_COPY_ARTIFACTS = "physical_copy"
PRESERVE_SOURCE_QUALITY = "preserve_source_primary_without_regrading"


@dataclass(frozen=True)
class LidarModalityDerivationConfig:
    """Represent lidar modality derivation config."""

    output_suffix: str
    coordinate_frame_policy: str
    sample_policy: str
    artifact_reuse: str
    quality_policy: str
    target_sensor: TrainingSensorConfig
    config_fingerprint: str

    def to_metadata(self) -> dict:
        """Return a metadata representation."""
        return {
            "schema_version": LIDAR_DERIVATION_SCHEMA,
            "output_suffix": self.output_suffix,
            "coordinate_frame_policy": self.coordinate_frame_policy,
            "sample_policy": self.sample_policy,
            "artifact_reuse": self.artifact_reuse,
            "quality_policy": self.quality_policy,
            "only_changed_modality": "lidar_point_cloud",
            "target_sensor": self.target_sensor.to_dict(),
            "config_fingerprint": self.config_fingerprint,
        }


def _require_exact(actual, expected, name: str) -> None:
    """Require exact."""
    if isinstance(expected, float):
        matches = bool(np.isclose(
            float(actual), expected, rtol=0.0, atol=1e-9
        ))
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"{name} must match mandatory training profile: "
            f"expected {expected}, got {actual}"
        )


def _profile_expectations() -> dict[str, dict]:
    """Return profile expectations."""
    shared_hdl = {
        "lidar.scanner_model": HDL64E_MODEL,
        "lidar.num_rings": 64,
        "lidar.points_per_ring": 2083,
        "lidar.horizontal_fov_deg": (-180.0, 180.0),
        "lidar.repeated_pattern": True,
        "lidar.min_range_m": 0.9,
        "lidar.max_range_m": 120.0,
        "lidar.beam_pattern_file": None,
        "lidar.pattern_duration_s": None,
        "lidar.scan_frequency_hz": None,
        "lidar.beams_per_frame": None,
        "lidar.accumulation_duration_s": None,
        "lidar.interleaved_channels": None,
        "lidar.channel_calibration_file": None,
        "lidar.pattern_row_count": None,
    }
    rectified_1600_camera = {
        "camera.width": 1600,
        "camera.height": 1296,
        "camera.fx": 731.392029,
        "camera.fy": 731.015015,
        "camera.cx": 812.549011,
        "camera.cy": 653.577026,
        "camera.skew": -0.085165,
        "camera.model": "pinhole_rectified",
    }
    return {
        PROFILE_NAME: {
            "camera.width": 1226,
            "camera.height": 370,
            "camera.fx": 707.0912,
            "camera.fy": 707.0912,
            "camera.cx": 601.8873,
            "camera.cy": 183.1104,
            "camera.skew": 0.0,
            "camera.model": "pinhole_rectified",
            **shared_hdl,
            "lidar.vertical_fov_deg": (-24.9, 2.0),
        },
        RECTIFIED_1600_PROFILE_NAME: {
            **rectified_1600_camera,
            **shared_hdl,
            "lidar.vertical_fov_deg": (-24.9, 2.0),
        },
        RECTIFIED_1600_WIDE_PROFILE_NAME: {
            **rectified_1600_camera,
            **shared_hdl,
            "lidar.vertical_fov_deg": (-25.0, 25.0),
        },
    }


def _mid360_expectations() -> dict[str, object]:
    """Return mid360 expectations."""
    return {
        "lidar.scanner_model": MID360_MODEL,
        "lidar.num_rings": 4,
        "lidar.points_per_ring": 5000,
        "lidar.vertical_fov_deg": (-7.2123, 52.164),
        "lidar.horizontal_fov_deg": (0.0, 360.0),
        "lidar.repeated_pattern": False,
        "lidar.beam_pattern_file": "mid360.csv",
        "lidar.pattern_duration_s": 4.0,
        "lidar.scan_frequency_hz": 10.0,
        "lidar.beams_per_frame": 20000,
        "lidar.interleaved_channels": 4,
        "lidar.pattern_row_count": 800000,
        "lidar.channel_calibration_file": None,
    }


def _rs32_expectations() -> dict[str, object]:
    """Return rs32 expectations."""
    return {
        "lidar.scanner_model": RS32_MODEL,
        "lidar.num_rings": 32,
        "lidar.points_per_ring": 1801,
        "lidar.vertical_fov_deg": (-25.0, 15.0167),
        "lidar.horizontal_fov_deg": (-180.0, 180.0),
        "lidar.repeated_pattern": True,
        "lidar.min_range_m": 0.4,
        "lidar.max_range_m": 200.0,
        "lidar.beam_pattern_file": None,
        "lidar.channel_calibration_file": "rs32.csv",
        "lidar.pattern_duration_s": None,
        "lidar.scan_frequency_hz": 10.0,
        "lidar.beams_per_frame": 57632,
        "lidar.accumulation_duration_s": None,
        "lidar.interleaved_channels": None,
        "lidar.pattern_row_count": None,
    }


def _mid360_accumulated_beam_count(config: LidarSensorConfig) -> int:
    """Return mid360 accumulated beam count."""
    if config.scanner_model != MID360_MODEL:
        raise ValueError("accumulated beam count is only defined for MID-360")
    required = (
        config.accumulation_duration_s,
        config.pattern_duration_s,
        config.scan_frequency_hz,
        config.beams_per_frame,
        config.interleaved_channels,
        config.pattern_row_count,
    )
    if any(value is None for value in required):
        raise ValueError("MID-360 accumulation config is incomplete")
    duration = float(config.accumulation_duration_s)
    pattern_duration = float(config.pattern_duration_s)
    if (
        not np.isfinite(duration)
        or not np.isfinite(pattern_duration)
        or not 0.0 < duration <= pattern_duration
    ):
        raise ValueError(
            "MID-360 accumulation_duration_s must satisfy "
            "0 < duration <= pattern_duration_s"
        )
    raw_count = (
        duration
        * float(config.scan_frequency_hz)
        * int(config.beams_per_frame)
    )
    beam_count = int(round(raw_count))
    if (
        beam_count < 1
        or not np.isclose(raw_count, beam_count, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(
            "MID-360 accumulation duration must produce an integer beam count"
        )
    if beam_count > int(config.pattern_row_count):
        raise ValueError("MID-360 accumulated window exceeds the beam pattern")
    if beam_count % int(config.interleaved_channels):
        raise ValueError(
            "MID-360 accumulated beam count must be divisible by "
            "interleaved_channels"
        )
    return beam_count


def _validate_pinhole_camera(config: CameraSensorConfig) -> None:
    """Validate pinhole camera."""
    values = np.asarray(
        [config.fx, config.fy, config.cx, config.cy, config.skew],
        dtype=np.float64,
    )
    if config.model != "pinhole_rectified":
        raise ValueError("task camera.model must be pinhole_rectified")
    if config.width < 2 or config.height < 2:
        raise ValueError("task camera dimensions must be at least 2x2")
    if not np.all(np.isfinite(values)) or min(config.fx, config.fy) <= 0.0:
        raise ValueError("task camera intrinsics must be finite "
            "with positive focal lengths")


def _validate_repeated_ring_lidar(config: LidarSensorConfig) -> None:
    """Validate repeated ring LiDAR."""
    if not config.scanner_model.strip():
        raise ValueError("repeated ring LiDAR scanner_model must be non-empty")
    if not config.repeated_pattern:
        raise ValueError("repeated ring LiDAR requires repeated_pattern=true")
    if config.num_rings < 1 or config.points_per_ring < 1:
        raise ValueError("repeated ring LiDAR grid dimensions must be positive")
    vertical = np.asarray(config.vertical_fov_deg, dtype=np.float64)
    horizontal = np.asarray(config.horizontal_fov_deg, dtype=np.float64)
    if (
        vertical.shape != (2,)
        or not np.all(np.isfinite(vertical))
        or not -90.0 <= vertical[0] < vertical[1] <= 90.0
    ):
        raise ValueError(
            "repeated ring LiDAR vertical_fov_deg must be increasing within "
            "[-90, 90]"
        )
    if (
        horizontal.shape != (2,)
        or not np.all(np.isfinite(horizontal))
        or not 0.0 < horizontal[1] - horizontal[0] <= 360.0
    ):
        raise ValueError(
            "repeated ring LiDAR horizontal FOV span must be within (0, 360]"
        )
    if not (
        np.isfinite(config.min_range_m)
        and np.isfinite(config.max_range_m)
        and 0.0 <= config.min_range_m < config.max_range_m
    ):
        raise ValueError(
            "repeated ring LiDAR range must satisfy "
            "0 <= min_range_m < max_range_m"
        )
    if (
        config.scan_frequency_hz is None
        or not np.isfinite(config.scan_frequency_hz)
        or config.scan_frequency_hz <= 0.0
    ):
        raise ValueError("repeated ring LiDAR scan_frequency_hz must be positive")
    expected_beams = config.num_rings * config.points_per_ring
    if config.beams_per_frame != expected_beams:
        raise ValueError(
            "repeated ring LiDAR beams_per_frame must equal "
            "num_rings * points_per_ring"
        )
    pattern_fields = (
        config.beam_pattern_file,
        config.beam_pattern_path,
        config.beam_pattern_sha256,
        config.pattern_duration_s,
        config.accumulation_duration_s,
        config.interleaved_channels,
        config.pattern_row_count,
    )
    if any(value is not None for value in pattern_fields):
        raise ValueError(
            "repeated ring LiDAR cannot configure a non-repeated beam pattern"
        )


def validate_training_sensor_config(config: TrainingSensorConfig) -> None:
    """Validate training sensor config."""
    profiles = _profile_expectations()
    if config.lidar.scanner_model == MID360_MODEL:
        if config.profile != MID360_PROFILE_NAME:
            raise ValueError(
                f"MID-360 sensor.profile must be {MID360_PROFILE_NAME!r}"
            )
        _validate_pinhole_camera(config.camera)
        if not (
            np.isfinite(config.lidar.min_range_m)
            and np.isfinite(config.lidar.max_range_m)
            and 0.0 <= config.lidar.min_range_m < config.lidar.max_range_m
        ):
            raise ValueError(
                "MID-360 range must satisfy 0 <= min_range_m < max_range_m"
            )
        expected = _mid360_expectations()
    elif config.lidar.scanner_model == RS32_MODEL:
        if config.profile != RS32_PROFILE_NAME:
            raise ValueError(
                f"RS32 sensor.profile must be {RS32_PROFILE_NAME!r}"
            )
        _validate_pinhole_camera(config.camera)
        _validate_repeated_ring_lidar(config.lidar)
        expected = _rs32_expectations()
    elif config.profile in profiles:
        expected = profiles[config.profile]
    elif config.lidar.scanner_model == HDL64E_MODEL:
        raise ValueError(
            "HDL-64E must continue to use one of the fixed whitelist profiles"
        )
    elif config.lidar.repeated_pattern:
        if (
            not config.profile.strip()
            or config.profile in (MID360_PROFILE_NAME, RS32_PROFILE_NAME)
        ):
            raise ValueError(
                "repeated ring LiDAR requires a non-reserved sensor.profile"
            )
        _validate_pinhole_camera(config.camera)
        _validate_repeated_ring_lidar(config.lidar)
        return
    else:
        raise ValueError(
            "unsupported non-repeated LiDAR scanner model: "
            f"{config.lidar.scanner_model!r}"
        )
    actual = {
        "camera.width": config.camera.width,
        "camera.height": config.camera.height,
        "camera.fx": config.camera.fx,
        "camera.fy": config.camera.fy,
        "camera.cx": config.camera.cx,
        "camera.cy": config.camera.cy,
        "camera.skew": config.camera.skew,
        "camera.model": config.camera.model,
        "lidar.scanner_model": config.lidar.scanner_model,
        "lidar.num_rings": config.lidar.num_rings,
        "lidar.points_per_ring": config.lidar.points_per_ring,
        "lidar.vertical_fov_deg": config.lidar.vertical_fov_deg,
        "lidar.horizontal_fov_deg": config.lidar.horizontal_fov_deg,
        "lidar.repeated_pattern": config.lidar.repeated_pattern,
        "lidar.min_range_m": config.lidar.min_range_m,
        "lidar.max_range_m": config.lidar.max_range_m,
        "lidar.beam_pattern_file": config.lidar.beam_pattern_file,
        "lidar.pattern_duration_s": config.lidar.pattern_duration_s,
        "lidar.scan_frequency_hz": config.lidar.scan_frequency_hz,
        "lidar.beams_per_frame": config.lidar.beams_per_frame,
        "lidar.accumulation_duration_s": config.lidar.accumulation_duration_s,
        "lidar.interleaved_channels": config.lidar.interleaved_channels,
        "lidar.pattern_row_count": config.lidar.pattern_row_count,
        "lidar.channel_calibration_file": (
            config.lidar.channel_calibration_file
        ),
    }
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, tuple):
            if len(actual_value) != len(expected_value):
                raise ValueError(f"{name} has invalid length")
            for index, value in enumerate(expected_value):
                _require_exact(actual_value[index], value, f"{name}[{index}]")
        else:
            _require_exact(actual_value, expected_value, name)
    if config.lidar.scanner_model == MID360_MODEL:
        _mid360_accumulated_beam_count(config.lidar)
    elif config.lidar.scanner_model == RS32_MODEL:
        load_rs32_channel_calibration(config.lidar)


def _sha256_file(path: Path) -> str:
    """Return sha256 file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_beam_pattern(
    config_path: Path, lidar: dict, scanner_model: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve beam pattern."""
    raw_reference = lidar.get("beam_pattern_file")
    repeated_pattern = bool(lidar["repeated_pattern"])
    if repeated_pattern:
        if raw_reference is not None:
            raise ValueError(
                "repeated ring LiDAR profile cannot configure beam_pattern_file"
            )
        return None, None, None
    if scanner_model != MID360_MODEL:
        raise ValueError(
            "non-repeated LiDAR profile requires a supported table-driven model"
        )
    reference = Path(str(raw_reference or ""))
    if (
        not str(raw_reference or "").strip()
        or reference.is_absolute()
        or ".." in reference.parts
    ):
        raise ValueError(
            "MID-360 beam_pattern_file must be a safe relative path"
        )
    resolved = (config_path.parent / reference).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MID-360 beam pattern does not exist: {resolved}")
    return reference.as_posix(), str(resolved), _sha256_file(resolved)

def _resolve_channel_calibration(
    config_path: Path, lidar: dict, scanner_model: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve channel calibration."""
    raw_reference = lidar.get("channel_calibration_file")
    if scanner_model != RS32_MODEL:
        if raw_reference is not None:
            raise ValueError(
                "channel_calibration_file is only supported for RS32"
            )
        return None, None, None
    reference = Path(str(raw_reference or ""))
    if (
        not str(raw_reference or "").strip()
        or reference.is_absolute()
        or ".." in reference.parts
    ):
        raise ValueError(
            "RS32 channel_calibration_file must be a safe relative path"
        )
    resolved = (config_path.parent / reference).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"RS32 channel calibration does not exist: {resolved}"
        )
    return reference.as_posix(), str(resolved), _sha256_file(resolved)


def _load_lidar_sensor_config(
    config_path: Path,
    lidar: dict,
) -> LidarSensorConfig:
    """Load LiDAR sensor config."""
    scanner_model = str(lidar["scanner_model"])
    pattern_file, pattern_path, pattern_sha256 = _resolve_beam_pattern(
        config_path.resolve(), lidar, scanner_model
    )
    calibration_file, calibration_path, calibration_sha256 = (
        _resolve_channel_calibration(config_path.resolve(), lidar, scanner_model)
    )
    return LidarSensorConfig(
        scanner_model=scanner_model,
        num_rings=int(lidar["num_rings"]),
        points_per_ring=int(lidar["points_per_ring"]),
        vertical_fov_deg=tuple(
            float(value) for value in lidar["vertical_fov_deg"]
        ),
        horizontal_fov_deg=tuple(
            float(value) for value in lidar["horizontal_fov_deg"]
        ),
        repeated_pattern=bool(lidar["repeated_pattern"]),
        min_range_m=float(lidar["min_range_m"]),
        max_range_m=float(lidar["max_range_m"]),
        beam_pattern_file=pattern_file,
        beam_pattern_path=pattern_path,
        beam_pattern_sha256=pattern_sha256,
        channel_calibration_file=calibration_file,
        channel_calibration_path=calibration_path,
        channel_calibration_sha256=calibration_sha256,
        pattern_duration_s=(
            float(lidar["pattern_duration_s"])
            if "pattern_duration_s" in lidar else None
        ),
        scan_frequency_hz=(
            float(lidar["scan_frequency_hz"])
            if "scan_frequency_hz" in lidar else None
        ),
        beams_per_frame=(
            int(lidar["beams_per_frame"])
            if "beams_per_frame" in lidar else None
        ),
        accumulation_duration_s=(
            float(lidar["accumulation_duration_s"])
            if "accumulation_duration_s" in lidar else None
        ),
        interleaved_channels=(
            int(lidar["interleaved_channels"])
            if "interleaved_channels" in lidar else None
        ),
        pattern_row_count=(
            int(lidar["pattern_row_count"])
            if "pattern_row_count" in lidar else None
        ),
    )


def load_training_sensor_config(path: str | Path) -> TrainingSensorConfig:
    """Load training sensor config."""
    raw, _, _ = load_config_mapping(
        path,
        section="sensor",
        description="sensor config",
    )
    camera = raw["camera"]
    lidar = raw["lidar"]
    config = TrainingSensorConfig(
        profile=str(raw["profile"]),
        camera=CameraSensorConfig(
            width=int(camera["width"]),
            height=int(camera["height"]),
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            skew=float(camera.get("skew", 0.0)),
            model=str(camera.get("model", "pinhole_rectified")),
        ),
        lidar=_load_lidar_sensor_config(Path(path), lidar),
    )
    validate_training_sensor_config(config)
    if config.lidar.scanner_model == MID360_MODEL:
        load_mid360_beam_table(config.lidar)
    return config


def load_lidar_modality_derivation_config(
    path: str | Path,
    source_sensor: TrainingSensorConfig,
) -> LidarModalityDerivationConfig:
    """Load LiDAR modality derivation config."""
    config_path = Path(path).resolve()
    raw, raw_bytes, _ = load_config_mapping(
        config_path,
        section="lidar_derivation",
        description="lidar modality derivation config",
    )
    if raw.get("schema_version") != LIDAR_DERIVATION_SCHEMA:
        raise ValueError(
            f"schema_version must be {LIDAR_DERIVATION_SCHEMA!r}"
        )
    policies = {
        "coordinate_frame_policy": REUSE_SOURCE_LIDAR_FRAME,
        "sample_policy": PRESERVE_ALL_SOURCE_SAMPLES,
        "artifact_reuse": PHYSICAL_COPY_ARTIFACTS,
        "quality_policy": PRESERVE_SOURCE_QUALITY,
    }
    for field, expected in policies.items():
        actual = str(raw.get(field, "")).strip()
        if actual != expected:
            raise ValueError(f"{field} must be {expected!r}, got {actual!r}")
    suffix = str(raw.get("output_suffix", "")).strip()
    if (
        not suffix.startswith("_")
        or suffix in {"_", "__"}
        or "/" in suffix
        or "\\" in suffix
        or not suffix[1:].replace("_", "").isalnum()
    ):
        raise ValueError("output_suffix must be a safe underscore suffix")
    if "camera" in raw or "sensor" in raw:
        raise ValueError(
            "LiDAR derivation config must inherit the source camera"
        )
    target_lidar = raw.get("target_lidar")
    if not isinstance(target_lidar, dict):
        raise ValueError("target_lidar must be a mapping")
    target_profile = str(raw.get("target_profile", "")).strip()
    target_sensor = TrainingSensorConfig(
        profile=target_profile,
        camera=source_sensor.camera,
        lidar=_load_lidar_sensor_config(config_path, target_lidar),
    )
    validate_training_sensor_config(target_sensor)
    if target_sensor.lidar.scanner_model == MID360_MODEL:
        load_mid360_beam_table(target_sensor.lidar)
    digest = hashlib.sha256(raw_bytes)
    if target_sensor.lidar.beam_pattern_sha256 is not None:
        digest.update(target_sensor.lidar.beam_pattern_sha256.encode("ascii"))
    if target_sensor.lidar.channel_calibration_sha256 is not None:
        digest.update(target_sensor.lidar.channel_calibration_sha256.encode("ascii"))
    return LidarModalityDerivationConfig(
        output_suffix=suffix,
        coordinate_frame_policy=policies["coordinate_frame_policy"],
        sample_policy=policies["sample_policy"],
        artifact_reuse=policies["artifact_reuse"],
        quality_policy=policies["quality_policy"],
        target_sensor=target_sensor,
        config_fingerprint=digest.hexdigest(),
    )


def training_sensor_config_fingerprint(path: str | Path) -> str:
    """Return training sensor config fingerprint."""
    config = load_training_sensor_config(path)
    digest = hashlib.sha256(
        config_section_fingerprint(path, "sensor").encode("ascii")
    )
    if config.lidar.beam_pattern_sha256 is not None:
        digest.update(config.lidar.beam_pattern_sha256.encode("ascii"))
    if config.lidar.channel_calibration_sha256 is not None:
        digest.update(config.lidar.channel_calibration_sha256.encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "CameraSensorConfig",
    "LidarSensorConfig",
    "LidarModalityDerivationConfig",
    "LIDAR_DERIVATION_SCHEMA",
    "HDL64E_MODEL",
    "MID360_MODEL",
    "MID360_PROFILE_NAME",
    "RS32_MODEL",
    "RS32_PROFILE_NAME",
    "PROFILE_NAME",
    "RECTIFIED_1600_PROFILE_NAME",
    "RECTIFIED_1600_WIDE_PROFILE_NAME",
    "TrainingSensorConfig",
    "load_training_sensor_config",
    "load_rs32_channel_calibration",
    "load_lidar_modality_derivation_config",
    "training_sensor_config_fingerprint",
    "validate_training_sensor_config",
]


from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CubemapRayProjection:
    """Represent cubemap ray projection."""
    face_names: tuple[str, ...]
    face_index: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    radial_scale: np.ndarray
    ray_shape: tuple[int, ...]
    face_size: int


@lru_cache(maxsize=8)
def camera_rays(config: CameraSensorConfig) -> np.ndarray:
    """Return camera rays."""
    u = np.arange(config.width, dtype=np.float64)
    v = np.arange(config.height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    normalized_y = (vv - config.cy) / config.fy
    normalized_x = (
        uu - config.cx - config.skew * normalized_y
    ) / config.fx
    return np.stack((
        normalized_x,
        normalized_y,
        np.ones_like(uu),
    ), axis=-1)


@dataclass(frozen=True)
class Rs32ChannelCalibration:
    """Represent rs32 channel calibration."""

    vertical_angles_deg: np.ndarray
    horizontal_offsets_deg: np.ndarray


@lru_cache(maxsize=4)
def _load_rs32_channel_calibration(
    path: str, fingerprint: str, expected_channels: int,
) -> Rs32ChannelCalibration:
    """Load rs32 channel calibration."""
    del fingerprint
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    expected_shape = (expected_channels, 2)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(
            "RS32 channel calibration must be finite "
            f"{expected_channels}x2, got {values.shape}"
        )
    vertical = values[:, 0].copy()
    horizontal = values[:, 1].copy()
    if np.any((vertical < -90.0) | (vertical > 90.0)):
        raise ValueError("RS32 vertical angles must lie within [-90, 90] degrees")
    if np.unique(vertical).size != expected_channels:
        raise ValueError("RS32 vertical angles must be unique per channel")
    if np.any((horizontal < -180.0) | (horizontal > 180.0)):
        raise ValueError(
            "RS32 horizontal offsets must lie within [-180, 180] degrees"
        )
    for array in (vertical, horizontal):
        array.flags.writeable = False
    return Rs32ChannelCalibration(vertical, horizontal)


def load_rs32_channel_calibration(
    config: LidarSensorConfig,
) -> Rs32ChannelCalibration:
    """Load rs32 channel calibration."""
    if config.scanner_model != RS32_MODEL:
        raise ValueError("RS32 channel calibration requested for another scanner")
    if (
        config.channel_calibration_path is None
        or config.channel_calibration_sha256 is None
    ):
        raise ValueError("RS32 config lacks channel calibration identity")
    table = _load_rs32_channel_calibration(
        config.channel_calibration_path,
        config.channel_calibration_sha256,
        config.num_rings,
    )
    observed_fov = (
        float(table.vertical_angles_deg.min()),
        float(table.vertical_angles_deg.max()),
    )
    if not np.allclose(
        observed_fov, config.vertical_fov_deg, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "RS32 channel calibration vertical FOV differs from profile: "
            f"{observed_fov}"
        )
    return table


@dataclass(frozen=True)
class Mid360BeamTable:
    """Represent mid360 beam table."""

    sample_ids: np.ndarray
    azimuth_deg: np.ndarray
    zenith_deg: np.ndarray


@dataclass(frozen=True)
class LidarBeamPattern:
    """Represent lidar beam pattern."""

    directions: np.ndarray
    ring_ids: np.ndarray
    beam_indices: np.ndarray
    grid_shape: tuple[int, int]
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    report: dict


@lru_cache(maxsize=4)
def _load_mid360_beam_table(
    path: str, fingerprint: str, expected_rows: int,
) -> Mid360BeamTable:
    """Load mid360 beam table."""
    del fingerprint
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().strip()
    if header != "Time/s,Azimuth/deg,Zenith/deg":
        raise ValueError(f"unexpected MID-360 CSV header: {header!r}")
    values = np.loadtxt(
        str(csv_path), delimiter=",", skiprows=1, dtype=np.float64,
    )
    if values.shape != (expected_rows, 3) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"MID-360 table must be finite {expected_rows}x3, got {values.shape}"
        )
    expected_ids = np.arange(1, expected_rows + 1, dtype=np.float64)
    if not np.array_equal(values[:, 0], expected_ids):
        raise ValueError("MID-360 sample IDs must be exactly 1..pattern_row_count")
    azimuth = values[:, 1].copy()
    zenith = values[:, 2].copy()
    if np.any((azimuth < 0.0) | (azimuth > 360.0)):
        raise ValueError("MID-360 azimuth must lie within [0, 360] degrees")
    if np.any((zenith < 0.0) | (zenith > 180.0)):
        raise ValueError("MID-360 zenith must lie within [0, 180] degrees")
    sample_ids = values[:, 0].astype(np.int64)
    for array in (sample_ids, azimuth, zenith):
        array.flags.writeable = False
    return Mid360BeamTable(sample_ids, azimuth, zenith)


def load_mid360_beam_table(config: LidarSensorConfig) -> Mid360BeamTable:
    """Load mid360 beam table."""
    if config.scanner_model != MID360_MODEL:
        raise ValueError("MID-360 beam table requested for another scanner")
    if (
        config.beam_pattern_path is None
        or config.beam_pattern_sha256 is None
        or config.pattern_row_count is None
    ):
        raise ValueError("MID-360 config lacks beam pattern identity")
    table = _load_mid360_beam_table(
        config.beam_pattern_path,
        config.beam_pattern_sha256,
        config.pattern_row_count,
    )
    elevation = 90.0 - table.zenith_deg
    observed_fov = (float(elevation.min()), float(elevation.max()))
    if not np.allclose(
        observed_fov, config.vertical_fov_deg, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "MID-360 table vertical FOV differs from configured profile: "
            f"{observed_fov}"
        )
    channels = int(config.interleaved_channels or 0)
    beams = int(config.beams_per_frame or 0)
    if channels < 1 or beams < 1 or beams % channels:
        raise ValueError(
            "MID-360 beams_per_frame must be divisible by interleaved_channels"
        )
    expected_rows = (
        float(config.pattern_duration_s or 0.0)
        * float(config.scan_frequency_hz or 0.0)
        * beams
    )
    if not np.isclose(
        expected_rows, config.pattern_row_count, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "MID-360 duration, frequency and beams_per_frame do not "
            "reconstruct pattern_row_count"
        )
    return table


@lru_cache(maxsize=8)
def lidar_beam_directions(
    config: LidarSensorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LiDAR beam directions."""
    if not config.repeated_pattern:
        raise ValueError(
            "lidar_beam_directions requires a repeated ring scanner"
        )
    if config.scanner_model == RS32_MODEL:
        calibration = load_rs32_channel_calibration(config)
        vertical_deg = calibration.vertical_angles_deg
        horizontal_offsets_deg = calibration.horizontal_offsets_deg
    else:
        vertical_deg = np.linspace(
            config.vertical_fov_deg[0],
            config.vertical_fov_deg[1],
            config.num_rings,
            dtype=np.float64,
        )
        horizontal_offsets_deg = np.zeros(config.num_rings, dtype=np.float64)
    base_azimuth_deg = np.linspace(
        config.horizontal_fov_deg[0],
        config.horizontal_fov_deg[1],
        config.points_per_ring,
        endpoint=False,
        dtype=np.float64,
    )

    azimuth_grid_deg = (
        base_azimuth_deg[None, :] - horizontal_offsets_deg[:, None]
    )
    if config.scanner_model == RS32_MODEL:
        azimuth_grid_deg = (azimuth_grid_deg + 180.0) % 360.0 - 180.0
    elevation_grid_deg = np.broadcast_to(
        vertical_deg[:, None], azimuth_grid_deg.shape
    )
    elevation_grid = np.deg2rad(elevation_grid_deg)
    azimuth_grid = np.deg2rad(azimuth_grid_deg)
    cos_elevation = np.cos(elevation_grid)
    directions = np.stack((
        cos_elevation * np.cos(azimuth_grid),
        cos_elevation * np.sin(azimuth_grid),
        np.sin(elevation_grid),
    ), axis=-1)
    ring_ids = np.repeat(
        np.arange(config.num_rings, dtype=np.int32),
        config.points_per_ring,
    )
    return (
        directions.reshape(-1, 3),
        ring_ids,
        azimuth_grid_deg.reshape(-1),
        elevation_grid_deg.reshape(-1),
    )


def lidar_beam_pattern(
    config: LidarSensorConfig,
    sampling_seed: int = 0,
    sampling_identity: str = "",
) -> LidarBeamPattern:
    """Return LiDAR beam pattern."""
    if config.repeated_pattern:
        directions, rings, azimuth, elevation = lidar_beam_directions(config)
        beam_count = len(directions)
        report = {
            "scanner_model": config.scanner_model,
            "pattern_semantic": "deterministic_repeated_ring_azimuth_grid",
            "coordinate_formula": (
                "[cos(elevation)*cos(azimuth), "
                "cos(elevation)*sin(azimuth), sin(elevation)]"
            ),
        }
        if config.scanner_model == RS32_MODEL:
            calibration = load_rs32_channel_calibration(config)
            report.update({
                "pattern_semantic": (
                    "rs32_repeated_ring_official_channel_calibration"
                ),
                "channel_calibration_file": config.channel_calibration_file,
                "channel_calibration_sha256": config.channel_calibration_sha256,
                "horizontal_offset_conversion": (
                    "project_azimuth_deg=base_azimuth_deg-csv_horizontal_offset_deg"
                ),
                "horizontal_offset_deg_min": float(
                    calibration.horizontal_offsets_deg.min()
                ),
                "horizontal_offset_deg_max": float(
                    calibration.horizontal_offsets_deg.max()
                ),
                "reflectance_simulated": False,
                "range_policy": "hard_radial_cutoff_without_reflectivity",
            })
        return LidarBeamPattern(
            directions=directions,
            ring_ids=rings,
            beam_indices=np.arange(beam_count, dtype=np.int64),
            grid_shape=(config.num_rings, config.points_per_ring),
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            report=report,
        )
    if config.scanner_model != MID360_MODEL:
        raise ValueError(f"unsupported LiDAR scanner model: {config.scanner_model}")
    if not sampling_identity:
        raise ValueError("MID-360 generation requires a stable sampling identity")
    if (
        config.beams_per_frame is None
        or config.accumulation_duration_s is None
        or config.interleaved_channels is None
        or config.pattern_row_count is None
        or config.beam_pattern_sha256 is None
    ):
        raise ValueError("MID-360 sampling config is incomplete")
    table = load_mid360_beam_table(config)
    accumulated_beams = _mid360_accumulated_beam_count(config)
    beam_rate_hz = float(config.scan_frequency_hz) * int(config.beams_per_frame)
    maximum_start = config.pattern_row_count - accumulated_beams
    if maximum_start < 0:
        raise ValueError("MID-360 accumulated window exceeds the pattern")
    identity = (
        f"{int(sampling_seed)}|{sampling_identity}|"
        f"{config.beam_pattern_sha256}"
    ).encode("utf-8")
    derived_seed = int.from_bytes(
        hashlib.sha256(identity).digest()[:16], "little"
    )
    rng = np.random.default_rng(derived_seed)
    start = int(rng.integers(0, maximum_start + 1))
    stop = start + accumulated_beams
    global_indices = np.arange(start, stop, dtype=np.int64)
    azimuth = table.azimuth_deg[start:stop]
    zenith = table.zenith_deg[start:stop]
    elevation = 90.0 - zenith
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    cos_elevation = np.cos(elevation_rad)
    directions = np.stack((
        cos_elevation * np.cos(azimuth_rad),
        cos_elevation * np.sin(azimuth_rad),
        np.sin(elevation_rad),
    ), axis=1)
    channels = global_indices % config.interleaved_channels
    global_times = global_indices // config.interleaved_channels
    first_time = int(global_times.min())
    time_columns = int(global_times.max() - first_time + 1)
    beam_indices = (
        channels * time_columns + (global_times - first_time)
    ).astype(np.int64)
    return LidarBeamPattern(
        directions=directions,
        ring_ids=channels.astype(np.int32),
        beam_indices=beam_indices,
        grid_shape=(config.interleaved_channels, time_columns),
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        report={
            "scanner_model": MID360_MODEL,
            "pattern_semantic": "nonrepeating_contiguous_accumulation_window_no_wrap",
            "coordinate_formula": (
                "[sin(zenith)*cos(azimuth), "
                "sin(zenith)*sin(azimuth), cos(zenith)]"
            ),
            "beam_pattern_file": config.beam_pattern_file,
            "beam_pattern_sha256": config.beam_pattern_sha256,
            "pattern_row_count": config.pattern_row_count,
            "window_start_zero_based": start,
            "window_stop_exclusive_zero_based": stop,
            "source_row_first_one_based": start + 1,
            "source_row_last_one_based": stop,
            "maximum_window_start_zero_based": maximum_start,
            "window_start_time_s": float(start / beam_rate_hz),
            "window_stop_time_s": float(stop / beam_rate_hz),
            "maximum_window_start_time_s": float(maximum_start / beam_rate_hz),
            "beams_per_frame": config.beams_per_frame,
            "accumulation_duration_s": config.accumulation_duration_s,
            "accumulated_beam_count": accumulated_beams,
            "beam_rate_hz": beam_rate_hz,
            "single_frame_duration_s": float(1.0 / config.scan_frequency_hz),
            "accumulated_frame_equivalents": float(
                config.accumulation_duration_s * config.scan_frequency_hz
            ),
            "static_cubemap_accumulation": True,
            "motion_distortion_simulated": False,
            "scan_frequency_hz": config.scan_frequency_hz,
            "pattern_duration_s": config.pattern_duration_s,
            "interleaved_channels": config.interleaved_channels,
            "beam_grid_shape": [config.interleaved_channels, time_columns],
            "window_wraparound": False,
            "sampling_seed": int(sampling_seed),
            "sampling_identity": sampling_identity,
        },
    )


def build_cubemap_ray_projection(
    rays_pano: np.ndarray,
    pano_from_face: Mapping[str, np.ndarray],
    face_size: int,
) -> CubemapRayProjection:
    """Build cubemap ray projection."""
    rays = np.asarray(rays_pano, dtype=np.float64)
    if rays.ndim < 2 or rays.shape[-1] != 3:
        raise ValueError("rays_pano must have shape (..., 3)")
    if not np.all(np.isfinite(rays)):
        raise ValueError("rays_pano must be finite")
    if np.any(np.linalg.norm(rays, axis=-1) <= 1e-12):
        raise ValueError("rays_pano cannot contain zero directions")
    if face_size < 2:
        raise ValueError("face_size must be at least two")
    face_names = tuple(pano_from_face)
    if len(face_names) != 6:
        raise ValueError("pano_from_face must contain exactly six faces")

    ray_shape = rays.shape[:-1]
    face_index = np.zeros(ray_shape, dtype=np.uint8)
    selected_forward = np.full(ray_shape, -np.inf, dtype=np.float64)
    selected_lateral = np.zeros(ray_shape, dtype=np.float64)
    selected_vertical = np.zeros(ray_shape, dtype=np.float64)
    for index, face in enumerate(face_names):
        rotation = np.asarray(pano_from_face[face], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"{face}: pano_from_face must be 3x3")
        directions = rays @ rotation
        forward = directions[..., 0]
        update = forward > selected_forward
        selected_forward[update] = forward[update]
        selected_lateral[update] = directions[..., 1][update]
        selected_vertical[update] = directions[..., 2][update]
        face_index[update] = index
    if np.any(selected_forward <= 0.0):
        raise ValueError("cubemap rotations do not cover all supplied rays")

    focal = face_size / 2.0
    center = (face_size - 1.0) / 2.0
    ray_norm = np.linalg.norm(rays, axis=-1)
    return CubemapRayProjection(
        face_names=face_names,
        face_index=face_index,
        map_x=(
            focal * selected_lateral / selected_forward + center
        ).astype(np.float32),
        map_y=(
            focal * selected_vertical / selected_forward + center
        ).astype(np.float32),
        radial_scale=(ray_norm / selected_forward).astype(np.float32),
        ray_shape=ray_shape,
        face_size=face_size,
    )


from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from .geometry import validate_transform
from .omni import OmniFrame


SENSOR_SAMPLE_VERSION = "bilateral_cubemap_multi_lidar_sample_v8_rs32_calibration"

_DEFAULT_RGB_CACHE_BYTES = 512 * 1024 * 1024

_RGB_CACHE = ArrayMemoryCache("cubemap_rgb", _DEFAULT_RGB_CACHE_BYTES)


@dataclass(frozen=True)
class GeneratedSensorSample:
    """Represent generated sensor sample."""
    image_bgr: np.ndarray
    camera_depth: np.ndarray
    point_cloud: np.ndarray
    ring_ids: np.ndarray
    report: dict


@dataclass(frozen=True)
class CameraDepthStage:
    """Represent camera depth stage."""

    camera_depth_unscaled: np.ndarray
    camera_depth: np.ndarray
    projection: CubemapRayProjection
    report: dict


@dataclass(frozen=True)
class LidarStage:
    """Represent lidar stage."""

    point_cloud: np.ndarray
    ring_ids: np.ndarray
    beam_indices: np.ndarray
    beam_grid_shape: tuple[int, int]
    report: dict


def _decode_bgr_image(path: str) -> np.ndarray:
    """Decode bgr image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable cubemap RGB face: {path}")
    image.flags.writeable = False
    return image


def configure_rgb_cache(max_bytes: int) -> None:
    """Configure RGB cache."""
    _RGB_CACHE.configure(max_bytes)


def clear_rgb_cache() -> None:
    """Clear RGB cache."""
    _RGB_CACHE.clear()


def rgb_cache_report() -> dict:
    """Return RGB cache report."""
    return _RGB_CACHE.report()


def _load_bgr_image(path: str) -> np.ndarray:
    """Load bgr image."""
    return _RGB_CACHE.get(path, _decode_bgr_image)


def _face_size(face_values: Mapping[str, np.ndarray]) -> int:
    """Return face size."""
    sizes = set()
    for face, value in face_values.items():
        if value.shape[0] != value.shape[1]:
            raise ValueError(f"{face}: cubemap face is not square: {value.shape}")
        sizes.add(int(value.shape[0]))
    if len(sizes) != 1:
        raise ValueError(f"cubemap face sizes differ: {sorted(sizes)}")
    return sizes.pop()


def _composite_faces(
    face_values: Mapping[str, np.ndarray],
    projection,
    interpolation: int,
) -> np.ndarray:
    """Return composite faces."""
    used_indices = np.unique(projection.face_index).astype(np.int64)
    used_faces = [projection.face_names[index] for index in used_indices]
    missing = [face for face in used_faces if face not in face_values]
    if missing:
        raise ValueError(f"cubemap values lack projected faces: {missing}")
    first = np.asarray(face_values[used_faces[0]])
    output_shape = projection.ray_shape + tuple(
        first.shape[2:]
    )
    output = np.zeros(output_shape, dtype=first.dtype)
    flat_output = output.reshape(
        (projection.face_index.size,) + first.shape[2:]
    )
    flat_face_index = projection.face_index.reshape(-1)
    flat_map_x = projection.map_x.reshape(-1)
    flat_map_y = projection.map_y.reshape(-1)
    for index in used_indices:
        face = projection.face_names[index]
        value = np.asarray(face_values[face])
        expected = (projection.face_size, projection.face_size)
        if value.shape[:2] != expected:
            raise ValueError(
                f"{face}: face shape {value.shape[:2]} != {expected}"
            )
        if value.shape[2:] != first.shape[2:] or value.dtype != first.dtype:
            raise ValueError("cubemap face values differ in channels or dtype")
        flat_indices = np.flatnonzero(flat_face_index == index)
        for start in range(0, len(flat_indices), 30000):
            block = flat_indices[start:start + 30000]
            remapped = cv2.remap(
                value,
                flat_map_x[block].reshape(-1, 1),
                flat_map_y[block].reshape(-1, 1),
                interpolation=interpolation,
                borderMode=cv2.BORDER_REPLICATE,
            )
            flat_output[block] = remapped.reshape(
                (len(block),) + first.shape[2:]
            )
    return output


def _projected_face_values(
    paths: Mapping[str, object],
    projection: CubemapRayProjection,
    loader,
    prefetched: Optional[tuple[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Return projected face values."""
    used_indices = np.unique(projection.face_index).astype(np.int64)
    used_faces = {
        projection.face_names[index] for index in used_indices
    }
    values = {}
    if prefetched is not None and prefetched[0] in used_faces:
        values[prefetched[0]] = prefetched[1]
    for face in used_faces:
        if face not in values:
            values[face] = loader(paths[face])
    return values


def render_kitti_camera(
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    pano_from_camera_rotation: np.ndarray,
    config: TrainingSensorConfig,
    projection: Optional[CubemapRayProjection] = None,
) -> Tuple[np.ndarray, dict]:
    """Render kitti camera."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        pano_from_camera_rotation, dtype=np.float64
    )
    rotation = validate_transform(transform)[:3, :3]
    if projection is None:
        first_face = next(iter(pano_from_face))
        first_value = _load_bgr_image(str(target.rgb_paths[first_face]))
        face_size = _face_size({first_face: first_value})
        rays_pano = camera_rays(config.camera) @ rotation.T
        projection = build_cubemap_ray_projection(
            rays_pano, pano_from_face, face_size
        )
        prefetched = (first_face, first_value)
    else:
        face_size = projection.face_size
        prefetched = None
    faces = _projected_face_values(
        target.rgb_paths,
        projection,
        lambda path: _load_bgr_image(str(path)),
        prefetched=prefetched,
    )
    if _face_size(faces) != face_size:
        raise ValueError("RGB and depth cubemap face sizes differ")
    rendered = _composite_faces(faces, projection, cv2.INTER_CUBIC)
    face_counts = {
        face: int(np.count_nonzero(projection.face_index == index))
        for index, face in enumerate(projection.face_names)
    }
    return rendered, {
        "face_size": face_size,
        "face_pixel_counts": face_counts,
    }


def generate_camera_depth_stage(
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    pano_from_camera_rotation: np.ndarray,
    depth_scale: float,
    config: TrainingSensorConfig,
) -> CameraDepthStage:
    """Generate camera depth stage."""
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        pano_from_camera_rotation, dtype=np.float64
    )
    rotation = validate_transform(transform)[:3, :3]
    first_face = next(iter(pano_from_face))
    first_value = load_packed_float32_depth(target.depth_paths[first_face])
    face_size = _face_size({first_face: first_value})
    rays_camera = camera_rays(config.camera)
    rays_pano = rays_camera @ rotation.T
    projection = build_cubemap_ray_projection(
        rays_pano, pano_from_face, face_size
    )
    depths = _projected_face_values(
        target.depth_paths,
        projection,
        load_packed_float32_depth,
        prefetched=(first_face, first_value),
    )
    forward_depth = _composite_faces(
        depths, projection, cv2.INTER_NEAREST
    ).astype(np.float64)
    raw_valid = (
        np.isfinite(forward_depth)
        & (forward_depth > 0.0)
        & (forward_depth < 65000.0)
    )


    radial_depth_unscaled = forward_depth * projection.radial_scale
    camera_depth_unscaled = (
        radial_depth_unscaled / np.linalg.norm(rays_camera, axis=-1)
    )
    camera_depth = camera_depth_unscaled * float(depth_scale)
    valid = (
        raw_valid
        & np.isfinite(camera_depth)
        & (camera_depth > 0.0)
    )
    camera_depth_unscaled[~valid] = np.nan
    camera_depth[~valid] = np.nan
    report = {
        "depth_scale": float(depth_scale),
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "valid_depth_ratio": float(np.mean(valid)),
        "minimum_unscaled_depth_m": float(
            np.min(camera_depth_unscaled[valid])
        ) if np.any(valid) else None,
        "maximum_unscaled_depth_m": float(
            np.max(camera_depth_unscaled[valid])
        ) if np.any(valid) else None,
        "minimum_depth_m": float(np.min(camera_depth[valid]))
        if np.any(valid) else None,
        "maximum_depth_m": float(np.max(camera_depth[valid]))
        if np.any(valid) else None,
    }
    return CameraDepthStage(
        camera_depth_unscaled=camera_depth_unscaled.astype(np.float32),
        camera_depth=camera_depth.astype(np.float32),
        projection=projection,
        report=report,
    )


def generate_lidar_stage(
    source: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    pano_from_lidar_rotation: np.ndarray,
    depth_scale: float,
    config: TrainingSensorConfig,
    sampling_seed: int = 0,
    sampling_identity: str = "",
) -> LidarStage:
    """Query raw cubemap depth along physically rotated LiDAR beams.

    ``pano_from_lidar_rotation`` maps LiDAR directions into the source panorama;
    ``depth_scale`` is the shared physical scene scale. The seed and identity select
    a reproducible scanner pattern. Return finite Nx4 float32 points in LiDAR meters
    with zero intensity, plus beam audit data; invalid scales raise ValueError.
    """
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        pano_from_lidar_rotation, dtype=np.float64
    )
    pano_from_lidar = validate_transform(transform)[:3, :3]
    first_face = next(iter(pano_from_face))
    first_value = load_packed_float32_depth(source.depth_paths[first_face])
    face_size = _face_size({first_face: first_value})
    beam_pattern = lidar_beam_pattern(
        config.lidar, sampling_seed, sampling_identity
    )
    directions_lidar = beam_pattern.directions
    directions_pano = directions_lidar @ pano_from_lidar.T
    projection = build_cubemap_ray_projection(
        directions_pano, pano_from_face, face_size
    )
    depths = _projected_face_values(
        source.depth_paths,
        projection,
        load_packed_float32_depth,
        prefetched=(first_face, first_value),
    )
    forward_depth = _composite_faces(
        depths, projection, cv2.INTER_NEAREST
    )
    raw_valid_depth = (
        np.isfinite(forward_depth)
        & (forward_depth > 0.0)
        & (forward_depth < 65000.0)
    ).reshape(-1)
    radial_depth = (
        forward_depth.astype(np.float64)
        * projection.radial_scale
        * float(depth_scale)
    )
    radial_flat = radial_depth.reshape(-1)
    valid_depth = (
        raw_valid_depth & np.isfinite(radial_flat) & (radial_flat > 0.0)
    )
    in_range = (
        (radial_flat >= config.lidar.min_range_m)
        & (radial_flat <= config.lidar.max_range_m)
    )
    valid = valid_depth & in_range
    points = directions_lidar[valid] * radial_flat[valid, None]
    point_cloud = np.zeros((len(points), 4), dtype=np.float32)
    point_cloud[:, :3] = points.astype(np.float32)
    face_counts = {
        face: int(np.count_nonzero(projection.face_index == index))
        for index, face in enumerate(projection.face_names)
    }
    report = {
        **beam_pattern.report,
        "face_size": face_size,
        "beam_count": int(len(directions_lidar)),
        "pano_from_lidar_rotation": pano_from_lidar.tolist(),
        "valid_point_count": int(np.count_nonzero(valid)),
        "invalid_depth_count": int(np.count_nonzero(~valid_depth)),
        "out_of_range_count": int(np.count_nonzero(valid_depth & ~in_range)),
        "face_beam_counts": face_counts,
        "azimuth_deg_min": float(beam_pattern.azimuth_deg.min()),
        "azimuth_deg_max": float(beam_pattern.azimuth_deg.max()),
        "elevation_deg_min": float(beam_pattern.elevation_deg.min()),
        "elevation_deg_max": float(beam_pattern.elevation_deg.max()),
    }
    return LidarStage(
        point_cloud=point_cloud,
        ring_ids=beam_pattern.ring_ids[valid],
        beam_indices=beam_pattern.beam_indices[valid],
        beam_grid_shape=beam_pattern.grid_shape,
        report=report,
    )


def validate_sensor_candidate_frames(
    source: OmniFrame,
    target: OmniFrame,
    candidate: dict,
) -> None:
    """Validate sensor candidate frames."""
    if str(candidate["source_frame"]) != source.frame_id:
        raise ValueError("candidate source_frame does not match source frame")
    if str(candidate["target_frame"]) != target.frame_id:
        raise ValueError("candidate target_frame does not match target frame")
    if "pano_from_lidar_rotation" not in candidate:
        raise ValueError(
            "candidate lacks the physical LiDAR orientation matrix"
        )


def generate_candidate_camera_depth(
    source: OmniFrame,
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    candidate: dict,
    config: TrainingSensorConfig,
) -> CameraDepthStage:
    """Generate candidate camera depth."""
    validate_sensor_candidate_frames(source, target, candidate)
    return generate_camera_depth_stage(
        target,
        pano_from_face,
        np.asarray(candidate["pano_from_camera_rotation"], dtype=np.float64),
        float(candidate["depth_scale"]),
        config,
    )


def generate_candidate_lidar(
    source: OmniFrame,
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    candidate: dict,
    config: TrainingSensorConfig,
) -> LidarStage:
    """Generate candidate LiDAR."""
    validate_sensor_candidate_frames(source, target, candidate)
    return generate_lidar_stage(
        source,
        pano_from_face,
        np.asarray(candidate["pano_from_lidar_rotation"], dtype=np.float64),
        float(candidate["depth_scale"]),
        config,
        sampling_seed=int(candidate.get("sensor_sampling_seed", 0)),
        sampling_identity=str(candidate["candidate_id"]),
    )


def finalize_sensor_sample(
    source: OmniFrame,
    target: OmniFrame,
    pano_from_face: Mapping[str, np.ndarray],
    candidate: dict,
    config: TrainingSensorConfig,
    camera_stage: CameraDepthStage,
    lidar_stage: LidarStage,
) -> GeneratedSensorSample:
    """Finalize sensor sample."""
    validate_sensor_candidate_frames(source, target, candidate)
    image, camera_report = render_kitti_camera(
        target,
        pano_from_face,
        np.asarray(candidate["pano_from_camera_rotation"], dtype=np.float64),
        config,
        projection=camera_stage.projection,
    )
    expected_image_shape = (
        config.camera.height, config.camera.width, 3
    )
    if image.shape != expected_image_shape or image.dtype != np.uint8:
        raise RuntimeError(
            f"generated image has invalid shape/dtype: {image.shape}/{image.dtype}"
        )
    point_cloud = lidar_stage.point_cloud
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 4:
        raise RuntimeError("generated point cloud is not Nx4")
    if point_cloud.dtype != np.float32:
        raise RuntimeError("generated point cloud is not float32")
    return GeneratedSensorSample(
        image_bgr=image,
        camera_depth=camera_stage.camera_depth,
        point_cloud=point_cloud,
        ring_ids=lidar_stage.ring_ids,
        report={
            "version": SENSOR_SAMPLE_VERSION,
            "sensor_profile": config.profile,
            "candidate_id": str(candidate["candidate_id"]),
            "pair_id": str(candidate["pair_id"]),
            "source_frame": source.frame_id,
            "target_frame": target.frame_id,
            "depth_scale": float(candidate["depth_scale"]),
            "lidar_orientation_id": str(candidate["lidar_orientation_id"]),
            "lidar_tilt_axis_index": int(candidate["lidar_tilt_axis_index"]),
            "lidar_tilt_magnitude_deg": float(
                candidate["lidar_tilt_magnitude_deg"]
            ),
            "image_shape": list(image.shape),
            "point_cloud_shape": list(point_cloud.shape),
            "camera": camera_report,
            "camera_depth": camera_stage.report,
            "lidar": lidar_stage.report,
        },
    )
