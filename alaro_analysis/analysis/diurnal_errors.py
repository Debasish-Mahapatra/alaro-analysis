"""Errors of a simulated diurnal cycle against an observed one.

* ``mean_cycle``: mean by local hour (24 bins, hour h = the interval [h, h + 1)).
* ``first_harmonic``: amplitude and time of maximum of the 24-h harmonic, fitted at the
  bin centres h + 0.5, so the timing is clock time, not a bin label.
* ``mse_decomposition``: the mean-square difference of two cycles split exactly into
  the offset of their means, the difference of their amplitudes (standard deviations)
  and the remainder from their shapes and timing:
  MSE = (mean_m - mean_o)^2 + (sd_m - sd_o)^2 + 2 sd_m sd_o (1 - r).
* ``wrap_hours``: a time difference brought into [-12, 12) h.
"""

from __future__ import annotations

import numpy as np

HOURS = 24
CENTRES = np.arange(HOURS) + 0.5


def mean_cycle(values: np.ndarray, local_hours: np.ndarray) -> np.ndarray:
    """(24, ...) mean by local hour of (time, ...) values; NaN-aware."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full((HOURS, *values.shape[1:]), np.nan)
    for h in range(HOURS):
        sel = values[local_hours == h]
        if sel.shape[0]:
            with np.errstate(invalid="ignore"):
                out[h] = np.nanmean(sel, axis=0)
    return out


def first_harmonic(cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(amplitude, local time of maximum in hours) of the 24-h harmonic of (24, ...) cycles."""
    cycle = np.asarray(cycle, dtype=np.float64)
    angle = 2.0 * np.pi * CENTRES / HOURS
    shape = (HOURS,) + (1,) * (cycle.ndim - 1)
    a = 2.0 / HOURS * np.sum(cycle * np.cos(angle).reshape(shape), axis=0)
    b = 2.0 / HOURS * np.sum(cycle * np.sin(angle).reshape(shape), axis=0)
    return np.hypot(a, b), np.mod(np.arctan2(b, a) * HOURS / (2.0 * np.pi), HOURS)


def mse_decomposition(model: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Exact split of the mean-square difference of two 24-value cycles."""
    m, o = np.asarray(model, float), np.asarray(observed, float)
    sd_m, sd_o = m.std(), o.std()
    r = float(np.corrcoef(m, o)[0, 1])
    parts = {
        "offset": float((m.mean() - o.mean()) ** 2),
        "amplitude": float((sd_m - sd_o) ** 2),
        "shape": float(2.0 * sd_m * sd_o * (1.0 - r)),
    }
    parts["mse"] = float(np.mean((m - o) ** 2))
    parts["rmse"] = float(np.sqrt(parts["mse"]))
    parts["r"] = r
    return parts


def wrap_hours(delta: np.ndarray) -> np.ndarray:
    """Time differences wrapped into [-12, 12) hours."""
    return np.mod(np.asarray(delta, dtype=np.float64) + HOURS / 2, HOURS) - HOURS / 2
