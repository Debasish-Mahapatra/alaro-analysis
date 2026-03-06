from __future__ import annotations

from .constants import SANITIZE_RE


def safe_name(name: str) -> str:
    return SANITIZE_RE.sub("_", name).strip("_").lower()
