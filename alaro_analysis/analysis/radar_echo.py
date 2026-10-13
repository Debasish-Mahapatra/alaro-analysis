"""Radar-echo statistics shared by the SIPAM observations and the model's simulated reflectivity.

Both datasets arrive on the same 4 km grid (SIPAM averaged in linear Z onto the model
cells) and at the same constant altitudes (2..16 km), so every statistic here is
computed identically for radar and model:

* contoured frequency by altitude (CFAD, Yuter & Houze 1995) of reflectivity;
* echo occurrence profiles above fixed thresholds;
* echo-top heights (highest level with Z above a threshold);
* only precipitating columns count: a column whose reflectivity never reaches 20 dBZ
  (2..16 km) is treated as echo-free, for radar and model alike (clear-air echo);
* convective / stratiform classification at 3 km (Steiner et al. 1995, J. Appl.
  Meteor. 34, 1978), adapted to the 4 km grid: background = linear-Z mean of the echo
  cells within 11 km; convective centre if Z >= 40 dBZ or Z - Zbg exceeds
  10 - Zbg^2/180 dB; convective radius 1-5 km with Zbg (Steiner's "medium" curve: 1 km
  below 25 dBZ, +1 km per 5 dBZ, 5 km from 40 dBZ), which on a 4 km grid means the
  centre alone below Zbg 35 dBZ and the centre plus its four neighbours (4 km away)
  from 35 dBZ; every other echo cell is stratiform. The 11 km background disc holds 21
  cells of the 4 km grid.
"""

from __future__ import annotations

import numpy as np

HEIGHTS_KM = np.arange(2.0, 17.0)
DBZ_EDGES = np.arange(0.0, 71.0, 1.0)
ECHO_DBZ = 10.0  # echo threshold for every statistic (cumulus congestus and deeper, Giangrande 2016)
# A column counts only if it reaches PRECIP_COLUMN_DBZ somewhere. This removes the dry-season
# clear-air (biological) echo of the deep boundary layer: in IOP2 SIPAM has echo >= 10 dBZ over
# 17 % of cells at 2 km but >= 20 dBZ over only 2.5 %, the weak echo fading with height up to
# ~5 km, while in IOP1 the two thresholds differ by a factor ~2 at every level.
PRECIP_COLUMN_DBZ = 20.0
OCCURRENCE_DBZ = (10.0, 20.0, 30.0, 40.0)
TOP_DBZ = (20.0, 30.0, 40.0)
CLASSIFY_HEIGHT_KM = 3.0
CELL_KM = 4.0
BACKGROUND_KM = 11.0
UTC_OFFSET = -4


def interpolate_to_heights(dbz: np.ndarray, height_m: np.ndarray, target_km=HEIGHTS_KM) -> np.ndarray:
    """(level, ...) dBZ on model levels -> (n_heights, ...) dBZ at fixed heights.

    Interpolates linear Z between the two levels bracketing each target height.
    Levels may run top first (FA order). Values outside the column are NaN.
    """
    z = np.asarray(height_m, dtype=np.float64)
    if z[0].mean() > z[-1].mean():  # top first -> bottom first
        z, dbz = z[::-1], np.asarray(dbz)[::-1]
    zlin = 10.0 ** (np.asarray(dbz, dtype=np.float64) / 10.0)
    out = np.full((len(target_km), *z.shape[1:]), np.nan)
    for i, h in enumerate(np.asarray(target_km) * 1000.0):
        below = (z <= h).sum(axis=0) - 1  # last level at or below h
        ok = (below >= 0) & (below < z.shape[0] - 1)
        k0 = np.clip(below, 0, z.shape[0] - 2)
        z0 = np.take_along_axis(z, k0[None], 0)[0]
        z1 = np.take_along_axis(z, (k0 + 1)[None], 0)[0]
        v0 = np.take_along_axis(zlin, k0[None], 0)[0]
        v1 = np.take_along_axis(zlin, (k0 + 1)[None], 0)[0]
        f = np.clip((h - z0) / np.where(z1 > z0, z1 - z0, 1.0), 0.0, 1.0)
        v = v0 + f * (v1 - v0)
        with np.errstate(divide="ignore"):
            out[i] = np.where(ok, 10.0 * np.log10(np.maximum(v, 1e-10)), np.nan)
    return out


def _disk_offsets(radius_km: float, cell_km: float = CELL_KM):
    n = int(radius_km // cell_km) + 1
    return [(dy, dx) for dy in range(-n, n + 1) for dx in range(-n, n + 1)
            if np.hypot(dy, dx) * cell_km <= radius_km]


def _shifted_sum(values: np.ndarray, offsets) -> np.ndarray:
    ny, nx = values.shape
    padded = np.pad(values, 3, constant_values=0.0)
    out = np.zeros_like(values)
    for dy, dx in offsets:
        out += padded[3 + dy:3 + dy + ny, 3 + dx:3 + dx + nx]
    return out


def steiner_classes(dbz: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """0 = no echo / outside, 1 = stratiform, 2 = convective, for a 2-D 3 km reflectivity map."""
    echo = domain & np.isfinite(dbz) & (dbz >= ECHO_DBZ)
    zlin = np.where(echo, 10.0 ** (np.nan_to_num(dbz) / 10.0), 0.0)
    offsets = _disk_offsets(BACKGROUND_KM)
    n_bg = _shifted_sum(echo.astype(float), offsets)
    with np.errstate(invalid="ignore", divide="ignore"):
        zbg = np.where(n_bg > 0, 10.0 * np.log10(_shifted_sum(zlin, offsets) / np.maximum(n_bg, 1)), 0.0)
    zbg = np.maximum(zbg, 0.0)
    peak = np.where(zbg < 42.43, 10.0 - zbg**2 / 180.0, 0.0)
    centre = echo & ((dbz >= 40.0) | (dbz - zbg > peak))
    conv = centre.copy()
    big = centre & (zbg >= 35.0)  # convective radius >= 4 km -> the four neighbours too
    if big.any():
        padded = np.pad(big, 1)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            conv |= padded[1 + dy:1 + dy + big.shape[0], 1 + dx:1 + dx + big.shape[1]]
    conv &= echo
    classes = np.zeros(dbz.shape, dtype=np.int8)
    classes[echo] = 1
    classes[conv] = 2
    return classes


def echo_tops(dbz_h: np.ndarray, thresholds=TOP_DBZ, heights_km=HEIGHTS_KM) -> np.ndarray:
    """(n_thresholds, ...) height (km) of the highest level with Z >= threshold; NaN if none."""
    out = np.full((len(thresholds), *dbz_h.shape[1:]), np.nan)
    for i, t in enumerate(thresholds):
        above = np.isfinite(dbz_h) & (dbz_h >= t)
        any_ = above.any(axis=0)
        top = len(heights_km) - 1 - np.argmax(above[::-1], axis=0)
        out[i] = np.where(any_, np.asarray(heights_km)[top], np.nan)
    return out


# Ice water content: radar from Z and T with the S-band (3 GHz) relation of Hogan et al. (2006, J. Appl.
# Meteor. Climatol. 45, 301, Table 2, expected-value fit): log10 IWC = 0.060 Z - 0.0197 T - 1.70
# (IWC g m-3, Z dBZ, T degC), fitted to stratiform aggregates; model from rho * (snow + graupel + cloud
# ice). Only where the mean temperature is below ICE_MAX_T_C (above the melting layer). The relation does
# not hold for the dense ice of convective cores, so the statistics are also kept for stratiform columns.
LOG_IWC_EDGES = np.linspace(-3.0, 1.0, 81)
IWC_THRESHOLDS = (0.05, 0.1, 0.3)
ICE_MAX_T_C = -5.0
Z_IWC_T = (0.0, 0.060, -0.0197, -1.70)  # (a, b, c, d) in log10 IWC = a Z T + b Z + c T + d


def iwc_from_dbz(dbz: np.ndarray, t_c: np.ndarray, coeffs=None) -> np.ndarray:
    """Ice water content (g m-3) from reflectivity (dBZ) and temperature (degC)."""
    a, b, c, d = coeffs if coeffs is not None else Z_IWC_T
    return 10.0 ** (a * dbz * t_c + b * dbz + c * t_c + d)


def linear_to_heights(values: np.ndarray, height_m: np.ndarray, target_km=HEIGHTS_KM) -> np.ndarray:
    """(level, ...) on model levels -> (n_heights, ...), linear in height (NaN outside the column)."""
    z = np.asarray(height_m, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if z[0].mean() > z[-1].mean():
        z, v = z[::-1], v[::-1]
    out = np.full((len(target_km), *z.shape[1:]), np.nan)
    for i, h in enumerate(np.asarray(target_km) * 1000.0):
        below = (z <= h).sum(axis=0) - 1
        ok = (below >= 0) & (below < z.shape[0] - 1)
        k0 = np.clip(below, 0, z.shape[0] - 2)
        z0 = np.take_along_axis(z, k0[None], 0)[0]
        z1 = np.take_along_axis(z, (k0 + 1)[None], 0)[0]
        v0 = np.take_along_axis(v, k0[None], 0)[0]
        v1 = np.take_along_axis(v, (k0 + 1)[None], 0)[0]
        f = np.clip((h - z0) / np.where(z1 > z0, z1 - z0, 1.0), 0.0, 1.0)
        out[i] = np.where(ok, v0 + f * (v1 - v0), np.nan)
    return out


def empty_stats() -> dict[str, np.ndarray]:
    nh, nb = len(HEIGHTS_KM), len(DBZ_EDGES) - 1
    return {
        "cfad": np.zeros((3, nh, nb)),  # all, convective column, stratiform column
        "occurrence": np.zeros((len(OCCURRENCE_DBZ), nh)),
        "n_cells": np.zeros(1),
        "tops": np.zeros((len(TOP_DBZ), nh)),  # histogram over HEIGHTS_KM
        "class_hour": np.zeros((3, 24)),  # convective cells, stratiform cells, all cells, by local hour
        "zr_rain_hour": np.zeros((3, 24)),  # Z-R rain (mm/h summed): convective, stratiform, all cells
        "sfc_rain_hour": np.zeros((3, 24)),  # model's own rain rate at the lowest level (mm/h summed): same split
        "n_maps": np.zeros(1),
        # [0] all precipitating columns, [1] stratiform columns only
        "iwc_hist": np.zeros((2, len(HEIGHTS_KM), len(LOG_IWC_EDGES) - 1)),
        "iwc_occurrence": np.zeros((2, len(IWC_THRESHOLDS), len(HEIGHTS_KM))),
        # every cell, without the precipitating-column filter (which depends on the rain and snow
        # reflectivity, so on the model's size distributions): anvils and light-rain columns count too
        "iwc_occurrence_all_cells": np.zeros((len(IWC_THRESHOLDS), len(HEIGHTS_KM))),
    }


def zr_rain(dbz: np.ndarray) -> np.ndarray:
    """Rain rate (mm/h) from 2 km reflectivity with the SIPAM Z-R law used for the radar rain."""
    return np.where(np.isfinite(dbz) & (dbz >= ECHO_DBZ), (10.0 ** (np.nan_to_num(dbz) / 10.0) / 174.8) ** (1 / 1.56), 0.0)


def accumulate(stats: dict, dbz_h: np.ndarray, classify_domain: np.ndarray, stats_domain: np.ndarray,
               local_hour: int, surface_rain: np.ndarray | None = None,
               iwc_h: np.ndarray | None = None, ice_levels: np.ndarray | None = None) -> None:
    """Add one (n_heights, y, x) reflectivity map to ``stats``.

    ``classify_domain`` is where the Steiner background may look; ``stats_domain`` is
    where statistics are counted (the radar's well-sampled range ring).
    """
    k3 = int(np.argmin(np.abs(HEIGHTS_KM - CLASSIFY_HEIGHT_KM)))
    with np.errstate(invalid="ignore"):
        precipitating = np.nanmax(np.where(np.isfinite(dbz_h), dbz_h, -99.0), axis=0) >= PRECIP_COLUMN_DBZ
    dbz_h = np.where(precipitating[None], dbz_h, np.nan)
    classes = steiner_classes(dbz_h[k3], classify_domain)
    cells = stats_domain
    z = dbz_h[:, cells]  # (nh, n)
    cls = classes[cells]
    echo = np.isfinite(z) & (z >= ECHO_DBZ)
    for c, sel in enumerate((np.ones_like(cls, bool), cls == 2, cls == 1)):
        for k in range(z.shape[0]):
            v = z[k, sel & echo[k]]
            if v.size:
                stats["cfad"][c, k] += np.histogram(v, DBZ_EDGES)[0]
    for i, t in enumerate(OCCURRENCE_DBZ):
        stats["occurrence"][i] += (np.isfinite(z) & (z >= t)).sum(axis=1)
    stats["n_cells"] += cells.sum()
    tops = echo_tops(z)
    for i in range(len(TOP_DBZ)):
        v = tops[i][np.isfinite(tops[i])]
        stats["tops"][i] += np.histogram(v, np.append(HEIGHTS_KM - 0.5, HEIGHTS_KM[-1] + 0.5))[0]
    stats["class_hour"][0, local_hour] += (cls == 2).sum()
    stats["class_hour"][1, local_hour] += (cls == 1).sum()
    stats["class_hour"][2, local_hour] += cls.size
    rain = zr_rain(z[0])
    stats["zr_rain_hour"][0, local_hour] += rain[cls == 2].sum()
    stats["zr_rain_hour"][1, local_hour] += rain[cls == 1].sum()
    stats["zr_rain_hour"][2, local_hour] += rain.sum()
    if surface_rain is not None:
        r = surface_rain[cells]
        stats["sfc_rain_hour"][0, local_hour] += r[cls == 2].sum()
        stats["sfc_rain_hour"][1, local_hour] += r[cls == 1].sum()
        stats["sfc_rain_hour"][2, local_hour] += r.sum()
    if iwc_h is not None and ice_levels is not None:
        for k in np.flatnonzero(ice_levels):
            v = iwc_h[k][cells]
            v = v[np.isfinite(v)]
            for j, thr in enumerate(IWC_THRESHOLDS):
                stats["iwc_occurrence_all_cells"][j, k] += np.sum(v >= thr)
        iwc = np.where(precipitating[None], iwc_h, np.nan)[:, cells]
        for s_i, subset in enumerate((np.ones(cls.shape, bool), cls == 1)):
            for k in np.flatnonzero(ice_levels):
                v = iwc[k][subset]
                v = v[np.isfinite(v) & (v > 0)]
                if v.size:
                    stats["iwc_hist"][s_i, k] += np.histogram(np.log10(v), LOG_IWC_EDGES)[0]
                for j, thr in enumerate(IWC_THRESHOLDS):
                    stats["iwc_occurrence"][s_i, j, k] += np.sum(v >= thr)
    stats["n_maps"] += 1
