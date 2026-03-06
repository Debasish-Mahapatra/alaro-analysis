"""Converter alias utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

REQUESTED_VAR_FALLBACK_ALIASES = {
    "KT273TEMPERATUR": (
        "KT273TEMPERATURE",
        "KT273TEMPERATU",
    ),
    "NC_LIQUID_WA": (
        "NC_LIQUID_WAT",
        "NC_LIQUID_WATER",
        "NC.LIQUID.WA",
        "NC.LIQUID.WAT",
        "NC.LIQUID.WATER",
    ),
}


def var_to_ds_name(name: str) -> str:
    return name.replace(".", "_")


def resolve_requested_vars(sample_file: Path, requested_vars: Sequence[str]):
    from .pipeline import resolve_requested_vars as _resolve_requested_vars

    return _resolve_requested_vars(sample_file, requested_vars)


__all__ = ["REQUESTED_VAR_FALLBACK_ALIASES", "resolve_requested_vars", "var_to_ds_name"]
