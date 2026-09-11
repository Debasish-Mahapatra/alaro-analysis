"""Pair analysis."""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.common.models import PeriodSpec


def compute_variable_scales(
    variable: str,
    scale_periods: list[PeriodSpec],
    profiles: dict[tuple[str, str, str], np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
    anomaly_percentile: float,
) -> tuple[tuple[float, float], float]:
    abs_chunks: list[np.ndarray] = []
    model_anom_chunks: list[np.ndarray] = []

    for period in scale_periods:
        for exp in EXPERIMENTS:
            data = profiles[(period.key, variable, exp)]
            valid = data[np.isfinite(data) & (data > 0)]
            if valid.size > 0:
                abs_chunks.append(valid)

        c = profiles[(period.key, variable, "control")]
        g = profiles[(period.key, variable, "graupel")]
        t = profiles[(period.key, variable, "2mom")]
        for arr in (g - c, t - g):
            diff_valid = arr[np.isfinite(arr)]
            if diff_valid.size > 0:
                model_anom_chunks.append(diff_valid)

    abs_limits = compute_log_abs_limits(
        abs_chunks=abs_chunks,
        absolute_low_percentile=absolute_low_percentile,
        absolute_high_percentile=absolute_high_percentile,
    )

    def anom_scale(chunks: list[np.ndarray]) -> float:
        if not chunks:
            return 1.0
        merged = np.concatenate(chunks)
        finite = np.abs(merged[np.isfinite(merged)])
        if finite.size == 0:
            return 1.0
        scale = float(np.percentile(finite, anomaly_percentile))
        if scale <= 0:
            scale = float(np.max(finite))
        return scale if scale > 0 else 1.0

    return abs_limits, anom_scale(model_anom_chunks)


def compute_pair_comparison_scales(
    pair: tuple[str, str],
    scale_periods: list[PeriodSpec],
    profiles: dict[tuple[str, str, str], np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
    anomaly_percentile: float,
) -> tuple[tuple[float, float], float]:
    normal_var, rad_var = pair
    abs_chunks: list[np.ndarray] = []
    cross_anom_chunks: list[np.ndarray] = []

    for period in scale_periods:
        for exp in EXPERIMENTS:
            normal = profiles[(period.key, normal_var, exp)]
            rad = profiles[(period.key, rad_var, exp)]

            n_valid = normal[np.isfinite(normal) & (normal > 0)]
            r_valid = rad[np.isfinite(rad) & (rad > 0)]
            if n_valid.size > 0:
                abs_chunks.append(n_valid)
            if r_valid.size > 0:
                abs_chunks.append(r_valid)

            cross = rad - normal
            c_valid = cross[np.isfinite(cross)]
            if c_valid.size > 0:
                cross_anom_chunks.append(c_valid)

    abs_limits = compute_log_abs_limits(
        abs_chunks=abs_chunks,
        absolute_low_percentile=absolute_low_percentile,
        absolute_high_percentile=absolute_high_percentile,
    )

    if cross_anom_chunks:
        merged = np.concatenate(cross_anom_chunks)
        finite = np.abs(merged[np.isfinite(merged)])
        if finite.size > 0:
            cross_scale = float(np.percentile(finite, anomaly_percentile))
            if cross_scale <= 0:
                cross_scale = float(np.max(finite))
            if cross_scale <= 0:
                cross_scale = 1.0
        else:
            cross_scale = 1.0
    else:
        cross_scale = 1.0

    return abs_limits, cross_scale


def compute_log_abs_limits(
    abs_chunks: list[np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
) -> tuple[float, float]:
    if not abs_chunks:
        return (1e-12, 1.0)

    abs_all = np.concatenate(abs_chunks)
    pos = abs_all[np.isfinite(abs_all) & (abs_all > 0)]
    if pos.size == 0:
        return (1e-12, 1.0)

    log_pos = np.log10(pos)
    p_low = float(np.percentile(log_pos, absolute_low_percentile))
    p_high = float(np.percentile(log_pos, absolute_high_percentile))
    if p_high <= p_low:
        p_high = p_low + 1.0

    vmin = 10.0**p_low
    vmax = 10.0**p_high
    if vmax <= vmin:
        vmax = vmin * 10.0
    return (float(vmin), float(vmax))
