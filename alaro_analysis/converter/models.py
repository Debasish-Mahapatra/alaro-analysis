"""Converter model exports."""

from __future__ import annotations

from typing import Any

__all__ = ["CropWindow", "FileTask", "RunConfig", "VariablePlan"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .pipeline import CropWindow, FileTask, RunConfig, VariablePlan

    return {
        "CropWindow": CropWindow,
        "FileTask": FileTask,
        "RunConfig": RunConfig,
        "VariablePlan": VariablePlan,
    }[name]
