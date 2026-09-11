"""Convection sampling. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.common.dsd_models import StrongConvectionSettings


def masked_level_mean(field: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(field) & mask[np.newaxis, :, :]
    counts = np.sum(valid, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(valid, field, 0.0), axis=(1, 2))
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out, counts


def strong_convection_mask(
    omega: np.ndarray,
    mesh: np.ndarray,
    settings: StrongConvectionSettings,
) -> np.ndarray:
    active = np.isfinite(omega) & np.isfinite(mesh) & (mesh > settings.min_updraft_mesh_frac)
    upward_pa_s = -omega
    if settings.metric == "omega":
        return active & np.isfinite(upward_pa_s) & (upward_pa_s >= settings.min_updraft_pa_s)
    flux = upward_pa_s * mesh / G
    return active & np.isfinite(flux) & (flux >= settings.min_updraft_flux)


def level_mean_in_strong_convection(
    values: np.ndarray,
    base_mask: np.ndarray,
    convection_mask: np.ndarray,
    rainy_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask3d = base_mask[np.newaxis, :, :] & convection_mask & rainy_mask & np.isfinite(values)
    counts = np.sum(mask3d, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(mask3d, values, 0.0), axis=(1, 2))
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out, counts
