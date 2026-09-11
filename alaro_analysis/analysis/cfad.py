"""Cfad. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import numpy as np


def add_values_to_hist(hist: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    levels = np.arange(vals.size)
    idx = np.searchsorted(edges, vals, side="right") - 1
    valid = np.isfinite(vals) & (idx >= 0) & (idx < edges.size - 1)
    if np.any(valid):
        np.add.at(hist, (levels[valid], idx[valid]), 1)
    return hist


def frequency_percent(hist: np.ndarray) -> np.ndarray:
    row_total = np.sum(hist, axis=1, keepdims=True)
    out = np.full(hist.shape, np.nan, dtype=np.float64)
    np.divide(100.0 * hist, row_total, out=out, where=row_total > 0)
    return out
