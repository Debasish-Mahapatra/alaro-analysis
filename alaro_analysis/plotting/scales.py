from __future__ import annotations

import numpy as np


def robust_anomaly_scale(*arrays: np.ndarray, percentile: float = 98.0) -> float:
    chunks = [arr[np.isfinite(arr)] for arr in arrays]
    chunks = [chunk for chunk in chunks if chunk.size > 0]
    if not chunks:
        return 1.0
    merged = np.concatenate(chunks)
    scale = float(np.percentile(np.abs(merged), percentile))
    return scale if scale > 0 else 1.0


def robust_log_limits(
    data: np.ndarray,
    *,
    low_percentile: float = 5.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        return 1e-12, 1.0
    vmin = float(np.percentile(valid, low_percentile))
    vmax = float(np.percentile(valid, high_percentile))
    vmin = max(vmin, float(np.min(valid)))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def infer_abs_limits(control: np.ndarray, *, linear: bool = True) -> tuple[float, float]:
    valid = control[np.isfinite(control)]
    if valid.size == 0:
        return (0.0, 1.0) if linear else (1e-12, 1.0)
    if linear:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax
    positive = valid[valid > 0]
    if positive.size == 0:
        return 1e-12, 1.0
    vmin = float(np.percentile(positive, 2))
    vmax = float(np.percentile(positive, 98))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def infer_anom_scale(*anomalies: np.ndarray) -> float:
    vals: list[np.ndarray] = []
    for arr in anomalies:
        finite = np.abs(arr[np.isfinite(arr)])
        if finite.size > 0:
            vals.append(finite)
    if not vals:
        return 1.0
    merged = np.concatenate(vals)
    scale = float(np.percentile(merged, 98))
    if scale <= 0:
        scale = float(np.max(merged))
    if scale <= 0:
        scale = 1.0
    return scale
