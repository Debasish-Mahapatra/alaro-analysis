"""ARM radar wind profiler at T3: convective vertical velocity, area fraction and mass flux.

Giangrande et al. (2016, JGR 121, 12891, doi:10.1002/2016JD025303) retrieved the vertical
air motion in precipitation from the 1290 MHz profiler at Manacapuru and combined it with
the convective area fraction into a GCM-type mass flux, following Kumar et al. (2015):

    Mc = rho * sigma_u * v_u + rho * sigma_d * v_d        (their eq. 1, kg m-2 s-1)

sigma is the share of the *whole domain* covered by convective updraft (downdraft) and v
the mean velocity there. The domain is made by time-to-space conversion: at the 5 m s-1
cell propagation speed of Amazon convection, one hour of profiles stands for about 20 km
and three hours for about 60 km. Every convective sample counts, with no core-length or
velocity threshold, and a window enters the ensemble when it holds at least five minutes
of convective echo ("when convection is present").

This module reads the PI product ``maorwpcls.YYYYMMDD.cdf`` (doi:10.5439/1440997) and
rebuilds those profiles. The files carry the retrieval, the echo classification and the
interpolated sounding, so the air density comes from the same file.

Two properties of the data are left as they are, because the paper leaves them as they are:
the retrieval is not performed in the melting layer, so roughly 4-5 km is empty, and the
velocity above the freezing level rests on a reflectivity-based fall-speed relation, which
the paper gives as the main source of its 1-2 m s-1 uncertainty.
"""

from __future__ import annotations

import os

# One thread per worker: OpenBLAS otherwise starts 128 threads in every worker and
# the pool runs the machine out of its 2048 process slots.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

MISSING = -9999.0
CONVECTIVE_CLASSES = (6.0, 7.0)  # 6 = convection, 7 = weak convection
MAX_SPEED_MS = 20.0  # "Observed maximum velocity values did not exceed 20 m/s" (section 4.1)
SAMPLE_SECONDS = 6.0  # the files carry 14400 profiles a day
CONVECTIVE_MINUTES = 5.0  # "when convection is present"
MIN_RETRIEVAL_COVERAGE = 0.5  # heights where most convective samples lack w are left out
R_DRY = 287.058
R_VAPOUR = 461.495
DAY_RE = re.compile(r"maorwpcls\.(\d{8})\.cdf$")


def day_of(path: Path) -> date:
    """The day a profiler file covers, from its name."""
    match = DAY_RE.search(str(path))
    if not match:
        raise ValueError(f"not a profiler file: {path}")
    token = match.group(1)
    return date(int(token[:4]), int(token[4:6]), int(token[6:8]))


def _clean(values) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    out[out == MISSING] = np.nan
    return out


def air_density(temperature_c: np.ndarray, pressure_kpa: np.ndarray, rh_percent: np.ndarray):
    """Moist air density (kg m-3) from the file's interpolated sounding."""
    t_kelvin = temperature_c + 273.15
    pressure = pressure_kpa * 1000.0
    # Saturation vapour pressure over water (Bolton 1980), good to 0.1 % over this range.
    e_sat = 611.2 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))
    e = np.clip(rh_percent, 0.0, 100.0) / 100.0 * e_sat
    return (pressure - e) / (R_DRY * t_kelvin) + e / (R_VAPOUR * t_kelvin)


def read_day(path: Path) -> dict[str, np.ndarray]:
    """Height, classification, retrieved velocity and air density of one day."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Dataset(path) as ds:
            height = np.asarray(ds["height"][:], dtype=np.float64)
            fields = {name: _clean(ds[name][:]) for name in
                      ("EchoClassification", "VerticalVelocity", "ReflectivityUAZR",
                       "Temperature", "RH", "BarometricPressure")}
    classification = fields["EchoClassification"]
    w = fields["VerticalVelocity"]
    w[np.abs(w) > MAX_SPEED_MS] = np.nan
    return {
        "height_km": height,
        "convective": np.isin(classification, CONVECTIVE_CLASSES),
        "w": w,
        "rho": air_density(fields["Temperature"], fields["BarometricPressure"], fields["RH"]),
        "n_times": classification.shape[0],
    }


def window_profiles(day: dict[str, np.ndarray], window_hours: int) -> dict[str, np.ndarray]:
    """Per window: area fractions, mean velocities, mass flux and the sums behind them.

    One row per window of the day. ``sigma_u`` is the share of all the window's samples
    that are convective and rising at that height, so the denominator is the whole
    domain, as in eq. (1); ``v_u`` is the mean velocity over the rising samples alone.
    Windows without five minutes of convective echo are marked ``present = False`` and
    left for the caller to drop.
    """
    n_times = day["n_times"]
    per_window = int(round(window_hours * 3600.0 / SAMPLE_SECONDS))
    n_windows = n_times // per_window
    conv = day["convective"][: n_windows * per_window]
    w = day["w"][: n_windows * per_window]
    rho = day["rho"][: n_windows * per_window]
    shape = (n_windows, per_window, conv.shape[1])
    conv = conv.reshape(shape)
    w = w.reshape(shape)
    rho = rho.reshape(shape)

    valid = conv & np.isfinite(w)
    up = valid & (w > 0)
    down = valid & (w < 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_up = up.sum(axis=1).astype(float)
        n_down = down.sum(axis=1).astype(float)
        sum_up = np.where(up, w, 0.0).sum(axis=1)
        sum_down = np.where(down, w, 0.0).sum(axis=1)
        v_up = np.divide(sum_up, n_up, out=np.full_like(sum_up, np.nan), where=n_up > 0)
        v_down = np.divide(sum_down, n_down, out=np.full_like(sum_down, np.nan), where=n_down > 0)
        rho_mean = np.nanmean(np.where(valid, rho, np.nan), axis=1)
        rho_all = np.nanmean(rho, axis=1)
        rho_mean = np.where(np.isfinite(rho_mean), rho_mean, rho_all)
        sigma_u = n_up / per_window
        sigma_d = n_down / per_window
        # Mass flux of the window: the sums already carry the area weighting, since
        # dividing the summed velocity by the window length is sigma * v.
        flux_up = rho_mean * sum_up / per_window
        flux_down = rho_mean * sum_down / per_window
        # A window is in the ensemble when convection is present for five minutes
        # anywhere in the column.
        column = conv.any(axis=2)
        minutes = column.sum(axis=1) * SAMPLE_SECONDS / 60.0
        # Retrieval coverage per height: the share of convective samples that have w.
        n_conv = conv.sum(axis=1).astype(float)
        coverage = np.divide(n_up + n_down, n_conv, out=np.zeros_like(n_conv), where=n_conv > 0)
    return {
        "sigma_up": sigma_u,
        "sigma_down": sigma_d,
        "v_up": v_up,
        "v_down": v_down,
        "flux_up": flux_up,
        "flux_down": flux_down,
        "sum_up": sum_up,
        "sum_down": sum_down,
        "n_up": n_up,
        "n_down": n_down,
        "n_conv": n_conv,
        "coverage": coverage,
        "rho": rho_mean,
        "minutes": minutes,
        "present": minutes >= CONVECTIVE_MINUTES,
        "n_samples": np.full(n_windows, float(per_window)),
    }


def day_task(task) -> dict | None:
    """Worker: ``(path, window_hours)`` -> the day's window rows, or None if unreadable."""
    path, window_hours = task
    try:
        day = read_day(Path(path))
    except Exception as exc:  # a few days are short or corrupt; skip them
        return {"path": str(path), "error": repr(exc)}
    rows = window_profiles(day, window_hours)
    rows["height_km"] = day["height_km"]
    rows["day"] = str(day_of(Path(path)))
    rows["path"] = str(path)
    return rows
