"""Parameterized updraft diagnostics on a fixed, valid model domain.

Inputs are omega in Pa/s and updraft area fraction in [0, 1]. Inactive
in-domain cells contribute zero; missing cells remain NaN. Spatial reduction
is a separate operation so callers can choose domain or conditional means.
"""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import G

UPDRAFT_CALCULATION_VERSION = 2


def updraft_mass_flux(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Return signed upward mass flux in kg m-2 s-1, preserving missing data."""
    omega, mesh = np.broadcast_arrays(omega, mesh)
    valid = np.isfinite(omega) & np.isfinite(mesh)
    return np.where(valid, np.where(mesh > 0, -omega * mesh / G, 0.0), np.nan)


def updraft_intensity(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Return absolute active-updraft omega in Pa/s, with the same domain mask."""
    omega, mesh = np.broadcast_arrays(omega, mesh)
    valid = np.isfinite(omega) & np.isfinite(mesh)
    return np.where(valid, np.where(mesh > 0, np.abs(omega), 0.0), np.nan)


def downdraft_mass_flux(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Return signed downward mass flux, retaining the shared valid domain."""
    return updraft_mass_flux(-np.asarray(omega), mesh)
