"""Analytic checks for the moisture-flux-convergence primitives."""
from __future__ import annotations

import numpy as np

from alaro_analysis.analysis.derived import (
    column_integrated_vapor_flux,
    horizontal_divergence_spherical,
)
from alaro_analysis.common.constants import EARTH_RADIUS_M, G


def _equatorial_grid(ny: int, nx: int, step_deg: float = 0.04):
    """Regular lon/lat grid centred on the equator (so cos(phi) ~ 1)."""
    lon1d = (np.arange(nx) - nx / 2.0) * step_deg
    lat1d = (np.arange(ny) - ny / 2.0) * step_deg
    lon, lat = np.meshgrid(lon1d, lat1d)
    return lon, lat


def test_divergence_zonal_gradient_matches_metric():
    lon, lat = _equatorial_grid(8, 9, step_deg=0.04)
    # Fx increases by 1 per i-step; centred difference spans 2 steps.
    fx = np.broadcast_to(np.arange(lon.shape[1], dtype=float), lon.shape).copy()
    fy = np.zeros_like(fx)
    div = horizontal_divergence_spherical(fx, fy, lon, lat)

    d_rad = np.deg2rad(2 * 0.04)
    expected = 2.0 / (EARTH_RADIUS_M * d_rad)
    interior = div[1:-1, 1:-1]
    assert np.all(np.isfinite(interior))
    assert np.allclose(interior, expected, rtol=1e-3)
    # Border ring is undefined.
    assert np.all(np.isnan(div[0, :]))
    assert np.all(np.isnan(div[:, -1]))


def test_divergence_meridional_gradient_matches_metric():
    lon, lat = _equatorial_grid(9, 8, step_deg=0.04)
    fy = np.broadcast_to(np.arange(lon.shape[0], dtype=float)[:, None], lon.shape).copy()
    fx = np.zeros_like(fy)
    div = horizontal_divergence_spherical(fx, fy, lon, lat)

    d_rad = np.deg2rad(2 * 0.04)
    expected = 2.0 / (EARTH_RADIUS_M * d_rad)
    assert np.allclose(div[1:-1, 1:-1], expected, rtol=1e-3)


def test_convergence_sign_inflow_is_positive():
    """Transport that slows down eastward piles moisture up: convergence > 0."""
    lon, lat = _equatorial_grid(7, 7, step_deg=0.04)
    # Eastward flux decreasing with i  ->  d(Fx)/dx < 0  ->  -div > 0.
    fx = np.broadcast_to(-np.arange(lon.shape[1], dtype=float), lon.shape).copy()
    fy = np.zeros_like(fx)
    convergence = -horizontal_divergence_spherical(fx, fy, lon, lat)
    interior = convergence[1:-1, 1:-1]
    assert np.all(interior > 0.0)


def test_column_flux_equals_wind_times_precipitable_water():
    nlev, ny, nx = 20, 4, 5
    q0, u0, dp0 = 0.012, 7.5, 5000.0
    q = np.full((nlev, ny, nx), q0)
    u = np.full((nlev, ny, nx), u0)
    v = np.zeros((nlev, ny, nx))
    dp = np.full((nlev, ny, nx), dp0)

    qx, qy = column_integrated_vapor_flux(q, u, v, dp)
    pw = q0 * (nlev * dp0) / G  # precipitable water, kg/m^2
    assert np.allclose(qx, u0 * pw)
    assert np.allclose(qy, 0.0)


def test_column_flux_clips_negative_humidity():
    nlev, ny, nx = 3, 2, 2
    q = np.full((nlev, ny, nx), 0.01)
    q[0] = -0.005  # spurious negative layer
    u = np.full((nlev, ny, nx), 4.0)
    v = np.zeros_like(u)
    dp = np.full((nlev, ny, nx), 4000.0)

    qx_clip, _ = column_integrated_vapor_flux(q, u, v, dp, clip_negative_q=True)
    qx_raw, _ = column_integrated_vapor_flux(q, u, v, dp, clip_negative_q=False)
    # Clipping removes the negative layer's (negative) contribution.
    expected_clip = 0.01 * 4.0 * (2 * 4000.0) / G
    assert np.allclose(qx_clip, expected_clip)
    assert np.all(qx_raw < qx_clip)
