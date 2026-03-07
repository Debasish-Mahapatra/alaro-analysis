from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import FREEZING_K


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def infer_freezing_threshold(temperature_profile: np.ndarray) -> float | None:
    valid = np.asarray(temperature_profile, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None
    return FREEZING_K if float(np.median(valid)) > 150.0 else 0.0


def interpolate_profile_to_target_height(
    source_height_km: np.ndarray,
    source_profile: np.ndarray,
    target_height_km: np.ndarray,
) -> np.ndarray:
    source_height_km = np.asarray(source_height_km, dtype=np.float64)
    source_profile = np.asarray(source_profile, dtype=np.float64)
    target_height_km = np.asarray(target_height_km, dtype=np.float64)
    out = np.full((target_height_km.size, source_profile.shape[1]), np.nan, dtype=np.float64)
    for hour in range(source_profile.shape[1]):
        column = source_profile[:, hour]
        finite = np.isfinite(source_height_km) & np.isfinite(column)
        if np.sum(finite) < 2:
            continue
        z = source_height_km[finite]
        v = column[finite]
        order = np.argsort(z)
        z = z[order]
        v = v[order]
        unique_mask = np.concatenate(([True], np.diff(z) > 0.0))
        z = z[unique_mask]
        v = v[unique_mask]
        if z.size < 2:
            continue
        out[:, hour] = np.interp(target_height_km, z, v, left=np.nan, right=np.nan)
    return out


def _mean_temperature_profile(
    temperature_profiles: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray | None:
    if isinstance(temperature_profiles, np.ndarray):
        profile = np.asarray(temperature_profiles, dtype=np.float64)
        if profile.ndim != 2:
            raise ValueError(
                f"Expected 2D temperature profile, got shape {profile.shape}"
            )
        return profile

    profiles = list(temperature_profiles)
    if not profiles:
        return None

    n_levels = min(profile.shape[0] for profile in profiles)
    stacked = np.stack(
        [np.asarray(profile[:n_levels, :], dtype=np.float64) for profile in profiles],
        axis=0,
    )
    valid = np.isfinite(stacked)
    counts = np.sum(valid, axis=0)
    total = np.nansum(stacked, axis=0)
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    nonzero = counts > 0
    mean[nonzero] = total[nonzero] / counts[nonzero]
    return mean


def compute_freezing_line_km(
    axis,
    temperature_profiles: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray | None:
    if not getattr(axis, "is_height_km", False):
        return None

    mean_temp = _mean_temperature_profile(temperature_profiles)
    if mean_temp is None:
        return None

    n_levels = min(np.asarray(axis.values).size, mean_temp.shape[0])
    if n_levels < 2:
        return None

    temp = np.asarray(mean_temp[:n_levels, :], dtype=np.float64)
    threshold = infer_freezing_threshold(temp)
    if threshold is None:
        return None

    y_km = np.asarray(axis.values[:n_levels], dtype=np.float64)
    order = np.argsort(y_km)
    y_sorted = y_km[order]
    t_sorted = temp[order, :]
    freeze_line = np.full((24,), np.nan, dtype=np.float64)

    for hour in range(24):
        column = t_sorted[:, hour]
        finite = np.isfinite(column) & np.isfinite(y_sorted)
        if np.sum(finite) < 2:
            continue
        yy = y_sorted[finite]
        tt = column[finite]
        for idx in range(yy.size - 1):
            t1 = tt[idx]
            t2 = tt[idx + 1]
            y1 = yy[idx]
            y2 = yy[idx + 1]
            if np.isclose(t1, threshold):
                freeze_line[hour] = y1
                break
            if np.isclose(t2, threshold):
                freeze_line[hour] = y2
                break
            d1 = t1 - threshold
            d2 = t2 - threshold
            if d1 * d2 < 0 and not np.isclose(t1, t2):
                frac = (threshold - t1) / (t2 - t1)
                freeze_line[hour] = y1 + frac * (y2 - y1)
                break

    return freeze_line
