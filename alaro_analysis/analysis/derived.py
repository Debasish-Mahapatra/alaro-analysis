"""
Pure-array derived variable computations.

All functions accept numpy arrays and return numpy arrays -- no I/O,
no file paths, no xarray dependencies.  This makes them easy to test
and compose in new analyses.
"""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import CP_D, EPS, G, LV, P0


# ---------------------------------------------------------------------------
# Pressure helpers
# ---------------------------------------------------------------------------


def maybe_convert_pressure_to_pa(p: np.ndarray) -> np.ndarray:
    """Heuristically convert pressure to Pa if it looks like hPa."""
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return p
    p01 = float(np.nanpercentile(finite, 1))
    p99 = float(np.nanpercentile(finite, 99))
    if 100.0 <= p01 <= 1200.0 and 100.0 <= p99 <= 2000.0:
        return p * 100.0
    return p


def compute_dp_pa(pressure_tlyx: np.ndarray) -> np.ndarray:
    """Compute pressure-layer thickness dp (Pa) from a 4-D pressure field.

    Parameters
    ----------
    pressure_tlyx : ndarray, shape (T, L, Y, X)
        Full pressure field in Pa on model levels.

    Returns
    -------
    dp : ndarray, shape (T, L, Y, X)
        Absolute pressure difference per layer.
    """
    p = np.asarray(pressure_tlyx, dtype=np.float64)
    if p.shape[1] == 1:
        return np.abs(p)
    p_half = np.empty(
        (p.shape[0], p.shape[1] + 1, p.shape[2], p.shape[3]),
        dtype=np.float64,
    )
    p_half[:, 1:-1, :, :] = 0.5 * (p[:, :-1, :, :] + p[:, 1:, :, :])
    p_half[:, 0, :, :] = p[:, 0, :, :] + (p[:, 0, :, :] - p_half[:, 1, :, :])
    p_half[:, -1, :, :] = p[:, -1, :, :] - (p_half[:, -2, :, :] - p[:, -1, :, :])
    dp = np.abs(p_half[:, :-1, :, :] - p_half[:, 1:, :, :])
    return dp


# ---------------------------------------------------------------------------
# Thermodynamic derived fields
# ---------------------------------------------------------------------------


def compute_theta_e_field(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute equivalent potential temperature (Bolton approximation).

    Parameters
    ----------
    temperature_k : ndarray
        Temperature in Kelvin.
    specific_humidity : ndarray
        Specific humidity in kg/kg.
    pressure_pa : ndarray
        Pressure in Pa (auto-converted from hPa if needed).

    Returns
    -------
    theta_e : ndarray
        Equivalent potential temperature in K.
    """
    t = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)

    q = np.where(q < 0.0, np.nan, q)
    p = maybe_convert_pressure_to_pa(p)
    p = np.where(p <= 0.0, np.nan, p)

    e = q * p / (EPS + (1.0 - EPS) * q)
    e = np.where(e > 1.0, e, np.nan)

    ln_e = np.log(e / 611.2)
    td_c = (243.5 * ln_e) / (17.67 - ln_e)
    td_k = td_c + 273.15

    td_k = np.where(np.isfinite(td_k), td_k, np.nan)
    td_k = np.clip(td_k, 180.0, 350.0)
    t = np.clip(t, 180.0, 350.0)

    tl = 1.0 / ((1.0 / (td_k - 56.0)) + (np.log(t / td_k) / 800.0)) + 56.0
    kappa = 0.2854 * (1.0 - 0.28 * q)
    expo = q * (1.0 + 0.81 * q) * ((3376.0 / tl) - 2.54)
    theta_e = t * np.power(P0 / p, kappa) * np.exp(expo)
    return theta_e


def compute_mse_field(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    """Compute moist static energy (MSE = cp*T + Lv*q + g*z).

    Returns MSE in J/kg.
    """
    t = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    z = np.asarray(height_m, dtype=np.float64)
    return CP_D * t + LV * q + G * z


def compute_relative_humidity(
    specific_humidity: np.ndarray,
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute relative humidity from q, T, p.

    Returns values clipped to [0, 1].
    """
    q = np.asarray(specific_humidity, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)

    q = np.where(q < 0.0, np.nan, q)
    e = q * p / (EPS + (1.0 - EPS) * q)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    rh = np.clip(e / es, 0.0, 1.0)
    return rh
