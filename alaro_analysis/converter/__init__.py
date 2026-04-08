"""FA-to-NetCDF conversion package."""

from __future__ import annotations

from typing import Any

__all__ = ["main", "run_conversion"]


def main(*args: Any, **kwargs: Any) -> Any:
    """CLI entry point for FA-to-NetCDF conversion."""
    from .cli import main as _main

    return _main(*args, **kwargs)


def run_conversion(*args: Any, **kwargs: Any) -> dict:
    """Programmatic entry point for FA-to-NetCDF conversion.

    See ``alaro_analysis.converter.pipeline.run_conversion`` for full
    signature and documentation.
    """
    from .pipeline import run_conversion as _run

    return _run(*args, **kwargs)
