"""FA-to-NetCDF conversion package."""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def main(*args: Any, **kwargs: Any) -> Any:
    from .cli import main as _main

    return _main(*args, **kwargs)
