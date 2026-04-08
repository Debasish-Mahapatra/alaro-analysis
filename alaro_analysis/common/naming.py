from __future__ import annotations

import re

from .constants import SANITIZE_RE

_VAR_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


def safe_name(name: str) -> str:
    """Convert a variable name to a filesystem-safe lowercase token."""
    return SANITIZE_RE.sub("_", name).strip("_").lower()


def normalize_var_token(name: str) -> str:
    """Strip non-alphanumeric characters and uppercase for fuzzy matching."""
    return _VAR_TOKEN_RE.sub("", name).upper()
