"""Disdrometer. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS

WATER_DENSITY = 1000.0


RDMEANREQ_M = 9.0e-4


SOURCE_RAIN_NUMBER_COEFFICIENT = 0.0019 / RDMEANREQ_M**3


def single_moment_rain_number_per_kg(
    rain_mixing_ratio_kgkg: np.ndarray | float,
    dmean_m: float = RDMEANREQ_M,
) -> np.ndarray:
    """Rain number concentration per kg from the model's fixed diameter logic."""
    qr = np.maximum(np.asarray(rain_mixing_ratio_kgkg, dtype=float), 0.0)
    return (0.0019 / dmean_m**3) * qr


def rain_mean_volume_diameter_mm(
    rain_mixing_ratio_kgkg: np.ndarray | float,
    rain_number_kg: np.ndarray | float,
) -> np.ndarray:
    """Return mean volume diameter implied by q_r and N_r."""
    qr = np.asarray(rain_mixing_ratio_kgkg, dtype=float)
    nr = np.asarray(rain_number_kg, dtype=float)
    out = np.full(np.broadcast(qr, nr).shape, np.nan, dtype=float)
    mask = np.isfinite(qr) & np.isfinite(nr) & (qr > 0.0) & (nr > 0.0)
    out[mask] = (6.0 * qr[mask] / (WATER_DENSITY * math.pi * nr[mask])) ** (1.0 / 3.0) * 1000.0
    return out


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def log10_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(mask.sum()) < 2:
        return np.nan
    return float(np.corrcoef(np.log10(x[mask]), np.log10(y[mask]))[0, 1])


def summarize_experiment(
    experiment: str,
    obs_values: np.ndarray,
    model_values: np.ndarray,
    model_qr: np.ndarray,
) -> dict[str, Any]:
    matched = np.isfinite(obs_values) & np.isfinite(model_values)
    positive = matched & (obs_values > 0.0) & (model_values > 0.0)
    bias = model_values[matched] - obs_values[matched]
    ratio = model_values[positive] / obs_values[positive]
    return {
        "experiment": experiment,
        "label": EXPERIMENT_LABELS.get(experiment, experiment),
        "n_matched": int(matched.sum()),
        "n_positive_pair": int(positive.sum()),
        "obs_mean_m3": finite_mean(obs_values[matched]),
        "obs_median_m3": finite_median(obs_values[matched]),
        "model_mean_m3": finite_mean(model_values[matched]),
        "model_median_m3": finite_median(model_values[matched]),
        "bias_mean_m3": finite_mean(bias),
        "bias_median_m3": finite_median(bias),
        "median_model_obs_ratio": finite_median(ratio),
        "log10_corr": log10_correlation(obs_values, model_values),
        "model_rain_qr_positive_count": int(np.sum(np.isfinite(model_qr) & (model_qr > 0.0))),
    }
