"""Load YAML configuration sections and compute stable content fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml


def load_config_mapping(
    path: str | Path,
    *,
    section: str,
    description: str,
) -> Tuple[dict[str, Any], bytes, bool]:
    """Load config mapping."""
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
        document = yaml.safe_load(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{description} must be UTF-8 YAML: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"invalid {description} YAML: {config_path}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{description} must be a YAML mapping")

    embedded = (
        document.get("schema_version") in {
            "stochastic_generation_task_v4",
            "stochastic_generation_task_v5",
        }
        and section in document
    )
    selected = document[section] if embedded else document
    if not isinstance(selected, Mapping):
        raise ValueError(f"{description} section {section!r} must be a mapping")
    mapping = dict(selected)
    fingerprint_bytes = (
        yaml.safe_dump(mapping, sort_keys=True).encode("utf-8")
        if embedded
        else raw_bytes
    )
    return mapping, fingerprint_bytes, embedded


def config_section_fingerprint(path: str | Path, section: str) -> str:
    """Return config section fingerprint."""
    _, fingerprint_bytes, _ = load_config_mapping(
        path,
        section=section,
        description=f"{section} config",
    )
    return hashlib.sha256(fingerprint_bytes).hexdigest()
