from __future__ import annotations

import math

import numpy as np

from alaro_analysis.common import dsd


def test_mp_lambda_from_lwc_nt_round_trip():
    # Pick (N0, lambda) and verify recovery.
    n0 = 8000.0   # 1/(m^3 mm) -- classical Marshall-Palmer
    lam = 2.5     # 1/mm
    diag = dsd.mp_diagnostics_from_n0_lambda(np.asarray([n0]), np.asarray([lam]))
    lam_back = dsd.mp_lambda_from_lwc_nt(
        np.asarray([float(diag["lwc_g_m3"][0])]),
        np.asarray([float(diag["nt_m3"][0])]),
    )
    assert math.isclose(float(lam_back[0]), lam, rel_tol=1e-6)


def test_mp_diagnostics_match_textbook_relations():
    n0 = 8000.0
    lam = 4.0
    diag = dsd.mp_diagnostics_from_n0_lambda(np.asarray([n0]), np.asarray([lam]))
    assert math.isclose(float(diag["dm_mm"][0]), 1.0, rel_tol=1e-12)
    assert math.isclose(float(diag["d0_mm"][0]), 3.67 / 4.0, rel_tol=1e-12)
    assert math.isclose(float(diag["sigma_m_mm"][0]), 0.5, rel_tol=1e-12)
    assert math.isclose(float(diag["nt_m3"][0]), 2000.0, rel_tol=1e-12)
    # log Nw == log10(N0) for Marshall-Palmer
    assert math.isclose(float(diag["log_nw"][0]), math.log10(n0), rel_tol=1e-12)


def test_mp_from_q_n_per_kg_consistency():
    # For q_r and n_per_kg with rho_air=1.2: LWC=1.2*1000*qr, Nt=1.2*nk
    qr = 1.0e-4
    nk = 1.0e3
    rho = 1.2
    out = dsd.mp_from_q_n_per_kg(np.asarray([qr]), np.asarray([nk]), np.asarray([rho]))
    assert math.isclose(float(out["lwc_g_m3"][0]), 1.2 * 1000.0 * qr, rel_tol=1e-12)
    assert math.isclose(float(out["nt_m3"][0]), 1.2 * nk, rel_tol=1e-12)
    # log Nw equals log10 N0 by construction
    assert math.isclose(
        float(out["log_nw"][0]), math.log10(float(out["n0_per_m3_mm"][0])), rel_tol=1e-12
    )


def test_mp_from_q_fixed_n0_returns_constant_log_nw():
    qr = np.array([1.0e-5, 1.0e-4, 1.0e-3])
    rho = np.full_like(qr, 1.2)
    out = dsd.mp_from_q_fixed_n0(qr, rho, n0_per_m3_mm=8000.0)
    assert np.allclose(out["log_nw"], math.log10(8000.0))
    # D0 must increase with rain (since lambda decreases with q for fixed N0)
    assert out["d0_mm"][0] < out["d0_mm"][1] < out["d0_mm"][2]


def test_temporal_continuity_drops_short_runs():
    # 1-min isolated rain spike, then 7-min rainy run, then 3-min run.
    rainy = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    out = dsd.temporal_continuity_mask(rainy, min_run_minutes=5)
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert np.array_equal(out, expected)


def test_apply_disdrometer_qc_combines_thresholds_and_continuity():
    rr = np.array([0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 5.0, 5.0])
    z = np.array([10.0, 30.0, 30.0, 30.0, 30.0, 60.0, 30.0, 30.0, 30.0, 30.0])
    mask = dsd.apply_disdrometer_qc(rr, z, min_run_minutes=3)
    # First run: indices 1-4 are above thresholds (Z=60 at 5 fails, R=0.05 at 0 fails).
    # That's 4 contiguous minutes (1..4) which clears the >=3 continuity test.
    # Then index 5 fails Z, 6 passes but isolated, 7 fails R, 8-9 pass but only 2 consecutive.
    expected = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert np.array_equal(mask, expected)


def test_empirical_dsd_parameters_recover_synthetic_moments():
    centers = np.array([0.5, 1.0, 1.5])
    widths = np.array([0.5, 0.5, 0.5])
    nd = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 1.0],
    ])
    m_powers = lambda k: np.sum(nd * (centers ** k)[None, :] * widths[None, :], axis=1)
    m3 = m_powers(3)
    m4 = m_powers(4)
    m5 = m_powers(5)
    out = dsd.empirical_dsd_parameters(
        number_density=nd,
        bin_centers_mm=centers,
        bin_widths_mm=widths,
        moment3=m3,
        moment4=m4,
        moment5=m5,
    )
    assert not np.isfinite(out["dm_mm"][0])
    assert math.isclose(float(out["dm_mm"][1]), m4[1] / m3[1], rel_tol=1e-12)
    assert np.isfinite(out["d0_mm"][1])
    assert out["lwc_g_m3"][1] > 0.0
    assert np.isfinite(out["log_nw"][1])
