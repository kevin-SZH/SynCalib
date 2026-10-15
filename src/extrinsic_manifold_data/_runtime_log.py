"""Emit structured runtime progress for long-running generation tasks."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any


def emit_runtime_log(event: str, message: str, **fields: Any) -> None:
    """Emit runtime log."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [
        f"[{timestamp}]",
        f"[{event}]",
        message,
        f"pid={os.getpid()}",
    ]
    for key, value in fields.items():
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            encoded = json.dumps(str(value), ensure_ascii=False)
        parts.append(f"{key}={encoded}")
    print(" | ".join(parts), file=sys.stderr, flush=True)


__all__ = ["emit_runtime_log"]
