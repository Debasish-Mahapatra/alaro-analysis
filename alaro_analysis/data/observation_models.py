"""Observation models. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationSeries:
    times: np.ndarray
    rain_number_m3: np.ndarray
    precip_rate_mm_h: np.ndarray
    median_volume_diameter_mm: np.ndarray


@dataclass(frozen=True)
class ObservationWindow:
    rain_number_m3: float
    precip_rate_mm_h: float
    median_volume_diameter_mm: float
    sample_count: int


@dataclass(frozen=True)
class ModelPoint:
    rain_number_m3: float
    rain_number_kg: float
    rain_mixing_ratio_kgkg: float
    dmean_mm: float
    rho_air_kg_m3: float
    temperature_k: float
    pressure_pa: float
    method: str
