"""Radiosonde thermodynamics. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import EPS


def _specific_humidity_to_vapor_pressure_pa(
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    q = np.asarray(specific_humidity, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    q = np.where(q > 0.0, q, np.nan)
    p = np.where(p > 0.0, p, np.nan)
    return q * p / (EPS + (1.0 - EPS) * q)


def dewpoint_c_from_specific_humidity(
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute dewpoint temperature in degC from q and pressure."""
    e = _specific_humidity_to_vapor_pressure_pa(specific_humidity, pressure_pa)
    log_term = np.log(e / 611.2)
    return (243.5 * log_term) / (17.67 - log_term)


def relative_humidity_percent(
    specific_humidity: np.ndarray,
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute RH in percent from q, T, and pressure."""
    e = _specific_humidity_to_vapor_pressure_pa(specific_humidity, pressure_pa)
    t = np.asarray(temperature_k, dtype=np.float64)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return np.clip(100.0 * e / es, 0.0, 100.0)
