from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PeriodSpec:
    key: str
    label: str
    output_subdir: Path
    allowed_months: tuple[int, ...] | None = None


@dataclass(frozen=True)
class VerticalAxis:
    values: np.ndarray
    label: str
    is_height_km: bool


@dataclass(frozen=True)
class AxisSpec:
    values: np.ndarray
    label: str
    is_height_km: bool


@dataclass(frozen=True)
class SpatialWindow:
    y_start: int | None
    y_end: int | None
    x_start: int | None
    x_end: int | None
