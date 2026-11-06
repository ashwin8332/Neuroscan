"""Strict JSON helpers (NaN/Infinity are not valid JSON and break browsers)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace NaN/Infinity with None and numpy scalars with Python types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalar
        try:
            value = value.item()
        except (TypeError, ValueError):
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(json_safe(data), indent=indent, allow_nan=False) + "\n")
