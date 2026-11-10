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
STRATIFORM_CLASSES = (1.0, 2.0)  # stratiform rain with and without a well-defined bright band
# Velocity histograms, shared with the model side (``data.rwp_model_sample``): 0.1 m s-1
# bins, anything beyond +-30 m s-1 in the end bins.
W_EDGES = np.round(np.arange(-300, 301) * 0.1, 1)
# Averaging times of the time-to-space conversion: at 5 m s-1, 800 s is one 4 km cell and
# 3600 s the 18-20 km of the model's 20 km boxes.
BLOCK_SECONDS = (6, 30, 60, 120, 300, 800, 1800, 3600)
PROPAGATION_MS = 5.0
# A window is averaged over its convective samples only (no echo means no data, not zero
# velocity), and kept where they fill at least half of it. Convective echo passes T3 in
# 10-20 minutes, so a stricter cover leaves a handful of hour-long windows per IOP; the
# windows slide by one sample, and the number of independent ones is n_samples / n.
BLOCK_COVERAGE = 0.5
# An hour counts as observed when at least half of its profiles carry data; the model is
# sampled at those hours only (whole days are missing when the profiler was off).
AVAILABLE_MIN = 0.5
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
        "stratiform": np.isin(classification, STRATIFORM_CLASSES),
        "profile_ok": np.isfinite(classification).any(axis=1),
        "w": w,
        "dbz": fields["ReflectivityUAZR"],
        "rho": air_density(fields["Temperature"], fields["BarometricPressure"], fields["RH"]),
        "n_times": classification.shape[0],
    }


def window_profiles(day: dict[str, np.ndarray], window_hours: int,
                    echo_dbz: float | None = None) -> dict[str, np.ndarray]:
    """Per window: area fractions, mean velocities, mass flux and the sums behind them.

    One row per window of the day. ``sigma_u`` is the share of all the window's samples
    that are convective and rising at that height, so the denominator is the whole
    domain, as in eq. (1); ``v_u`` is the mean velocity over the rising samples alone.
    With ``echo_dbz``, a sample counts only where the reflectivity reaches it (the model is
    sampled at its own echo threshold); which windows count does not depend on it.
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
    echo = conv
    if echo_dbz is not None:
        echo = conv & (np.nan_to_num(day["dbz"][: n_windows * per_window], nan=-99.0).reshape(shape) >= echo_dbz)

    valid = echo & np.isfinite(w)
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
        # Retrieval coverage per height: the share of convective echo samples that have w.
        n_conv = echo.sum(axis=1).astype(float)
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
    """Worker: ``(path, window_hours, echo_dbz)`` -> the day's window rows, or None if unreadable."""
    path, window_hours, echo_dbz = task
    try:
        day = read_day(Path(path))
    except Exception as exc:  # a few days are short or corrupt; skip them
        return {"path": str(path), "error": repr(exc)}
    rows = window_profiles(day, window_hours, echo_dbz)
    rows["height_km"] = day["height_km"]
    rows["day"] = str(day_of(Path(path)))
    rows["path"] = str(path)
    return rows


def gate_histogram(values: np.ndarray, keep: np.ndarray, edges: np.ndarray = W_EDGES) -> np.ndarray:
    """(n_gates, n_bins) counts of the (time, gate) ``values`` where ``keep``.

    Values beyond the outer edges go into the end bins.
    """
    n_bins = edges.size - 1
    n_gates = values.shape[-1]
    index = np.clip(np.digitize(np.nan_to_num(values), edges) - 1, 0, n_bins - 1)
    flat = (np.arange(n_gates) * n_bins + index)[keep]
    return np.bincount(flat, minlength=n_gates * n_bins).reshape(n_gates, n_bins)


def block_means(w: np.ndarray, valid: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Means of every ``n``-sample window of (time, gate) ``w``, sliding by one sample.

    Only the ``valid`` samples enter a mean, and a window is kept where they fill at least
    ``BLOCK_COVERAGE`` of it. Returns (window means, kept), (time - n + 1, gate).
    """
    zero = np.zeros((1, w.shape[1]))
    total = np.concatenate([zero, np.cumsum(np.where(valid, w, 0.0), axis=0)])
    count = np.concatenate([zero, np.cumsum(valid, axis=0, dtype=np.float64)])
    total = total[n:] - total[:-n]
    count = count[n:] - count[:-n]
    return total / np.maximum(count, 1.0), count >= BLOCK_COVERAGE * n - 1e-9


def velocity_statistics(day: dict[str, np.ndarray], echo_dbz: float | None = None) -> dict[str, np.ndarray]:
    """Velocity distributions of one day, for the comparisons of the model's w.

    ``hist_conv`` (block, gate, bin): convective echo, w averaged over each of
    ``BLOCK_SECONDS`` (the first is the raw 6 s sample). ``hist_strat`` (gate, bin): raw w
    in stratiform rain, where the true w is a few tenths of m s-1, so its spread measures
    the retrieval error. ``lag_sq``/``lag_n``: squared differences of consecutive convective
    samples; for white noise on a signal that hardly changes in 6 s, sqrt(lag_sq / lag_n / 2)
    is the noise standard deviation. ``available`` (24,): share of each UTC hour's profiles
    that carry data. With ``echo_dbz``, convective samples count only where the reflectivity
    reaches it, as on the model side.
    """
    w = day["w"]
    conv = day["convective"] & np.isfinite(w)
    if echo_dbz is not None:
        conv = conv & (np.nan_to_num(day["dbz"], nan=-99.0) >= echo_dbz)
    strat = day["stratiform"] & np.isfinite(w)
    hist_conv = np.stack([
        gate_histogram(*block_means(w, conv, int(round(seconds / SAMPLE_SECONDS))))
        for seconds in BLOCK_SECONDS
    ])
    pair = conv[1:] & conv[:-1]
    diff = np.where(pair, np.diff(np.nan_to_num(w), axis=0), 0.0)
    per_hour = day["n_times"] // 24
    return {
        "hist_conv": hist_conv,
        "hist_strat": gate_histogram(w, strat),
        "lag_sq": (diff**2).sum(axis=0),
        "lag_n": pair.sum(axis=0),
        "available": day["profile_ok"][: 24 * per_hour].reshape(24, per_hour).mean(axis=1),
    }


def velocity_task(task) -> dict:
    """Worker: ``(path, echo_dbz)`` -> the file's velocity statistics, or the error."""
    path, echo_dbz = task
    try:
        day = read_day(Path(path))
        out = velocity_statistics(day, echo_dbz)
    except Exception as exc:  # a few days are short or corrupt; skip them
        return {"path": str(path), "error": repr(exc)}
    out["height_km"] = day["height_km"]
    out["day"] = str(day_of(Path(path)))
    return out


def histogram_quantiles(hist: np.ndarray, quantiles, edges: np.ndarray = W_EDGES) -> np.ndarray:
    """(..., n_quantiles) quantiles of (..., bin) histograms, linear within a bin; NaN if empty."""
    hist = np.asarray(hist, dtype=np.float64)
    cdf = np.cumsum(hist, axis=-1)
    total = cdf[..., -1:]
    out = np.full((*hist.shape[:-1], len(quantiles)), np.nan)
    flat_cdf = cdf.reshape(-1, cdf.shape[-1])
    flat_total = total.reshape(-1)
    flat_out = out.reshape(-1, len(quantiles))
    for i, (row, n) in enumerate(zip(flat_cdf, flat_total)):
        if n <= 0:
            continue
        cum = np.concatenate([[0.0], row / n])
        flat_out[i] = np.interp(quantiles, cum, edges)
    return out


def exceedance(hist: np.ndarray, threshold: float, edges: np.ndarray = W_EDGES) -> tuple[np.ndarray, np.ndarray]:
    """(share above +threshold, share below -threshold) of (..., bin) histograms."""
    hist = np.asarray(hist, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(total > 0, hist[..., centres > threshold].sum(axis=-1) / total, np.nan)
        down = np.where(total > 0, hist[..., centres < -threshold].sum(axis=-1) / total, np.nan)
    return up, down


def flux_weighted_updraft(hist: np.ndarray, edges: np.ndarray = W_EDGES) -> np.ndarray:
    """Mass-flux-weighted mean of the rising velocities, sum(w^2) / sum(w) over w > 0."""
    hist = np.asarray(hist, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    up = centres > 0
    first = (hist[..., up] * centres[up]).sum(axis=-1)
    second = (hist[..., up] * centres[up] ** 2).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(first > 0, second / first, np.nan)
