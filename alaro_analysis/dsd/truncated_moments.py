"""Scheme-native rain DSD moments truncated at the disdrometer's smallest diameter.

Shared by the surface DSD remake (``recipes.dsd.run_dsd_per_cell_moments_truncated``,
whose docstring gives the formulas) and the IOP column statistics
(``data.microphysics_columns``): the model only "sees" the part of its spectrum the
instrument sees.
"""

from __future__ import annotations

import math

import numpy as np

from alaro_analysis.common.dsd import NW_PREFACTOR
from alaro_analysis.dsd.surface import OBS_PARAMETERS

SAMPLE_KEYS = (*OBS_PARAMETERS, "mu")
AB_LAMBDA_PREFACTOR = math.pi * 220.0  # = 691.15; lambda = (rho*qr/691.15)**(-1/1.8)
MU_MIN, MU_MAX = 0.1, 50.0
FALL_B = 0.7706  # w(D) = CF * D**FALL_B
FALL_CF = 654.5
CM = 1000.0 * math.pi / 6.0


def cut_gamma(
    k: float, mu: np.ndarray | float, dmin_m: float, lamb_per_m: np.ndarray
) -> np.ndarray:
    """Truncated-moment factor, verbatim from the provided post-processing."""
    from scipy.special import gammaincc

    x = np.asarray(lamb_per_m, dtype=float) * dmin_m
    mu_b = np.broadcast_to(np.asarray(mu, dtype=float), x.shape)
    out = gammaincc(k + 1.0, x)
    zero = (k + mu_b) == 0.0
    if np.any(zero):
        out = np.where(zero, np.exp(-x), out)
    return out


def _empty_samples() -> dict[str, np.ndarray]:
    return {key: np.empty(0, dtype=np.float32) for key in SAMPLE_KEYS}


def _diag_from_moments(
    qr: np.ndarray,
    rho: np.ndarray,
    m0: np.ndarray,
    m3: np.ndarray,
    m3b: np.ndarray,
    m4: np.ndarray,
    m6: np.ndarray,
    mu: np.ndarray,
    lamb_per_m: np.ndarray,
    dmin_m: float,
    min_rr_mm_h: float,
    max_z_dbz: float,
) -> dict[str, np.ndarray]:
    """Truncated Dm/LWC/Nt/logNw samples from absolute SI moments.

    Cells are kept only if the truncated spectrum would pass the same QC the
    disdrometer applies (rain rate >= min_rr, reflectivity <= max dBZ), using
    pfplsl and zrefl from the provided post-processing.
    """
    q0 = cut_gamma(0.0, mu, dmin_m, lamb_per_m)
    q3 = cut_gamma(3.0, mu, dmin_m, lamb_per_m)
    q4 = cut_gamma(4.0, mu, dmin_m, lamb_per_m)
    q3b = cut_gamma(3.0 + FALL_B, mu, dmin_m, lamb_per_m)
    q6 = cut_gamma(6.0, mu, dmin_m, lamb_per_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        dm_mm = 1000.0 * (m4 / m3) * (q4 / q3)  # zdmassw
        lwc_g_m3 = 1000.0 * rho * qr * q3  # zqrmp in g/m^3
        nt_m3 = m0 * q0  # pnrmp
        nw = NW_PREFACTOR * lwc_g_m3 / dm_mm**4
        log_nw = np.log10(nw)
        d0_mm = 1000.0 * (3.67 + mu) / lamb_per_m  # untruncated analytic (filter only)
        sigma_m_mm = dm_mm / np.sqrt(mu + 4.0)  # filter only
        rain_rate_mm_h = 3600.0 * m3b * CM * FALL_CF * q3b  # pfplsl
        refl_dbz = 10.0 * np.log10(m6 * q6 * 1.0e18)  # zrefl

    keep = (
        np.isfinite(dm_mm)
        & (dm_mm > 0.0)
        & np.isfinite(log_nw)
        & np.isfinite(lwc_g_m3)
        & (lwc_g_m3 > 0.0)
        & np.isfinite(nt_m3)
        & (nt_m3 > 0.0)
        & np.isfinite(d0_mm)
        & (d0_mm > 0.0)
        & np.isfinite(sigma_m_mm)
        & np.isfinite(rain_rate_mm_h)
        & (rain_rate_mm_h >= min_rr_mm_h)
        & np.isfinite(refl_dbz)
        & (refl_dbz <= max_z_dbz)
    )
    if not keep.any():
        return _empty_samples()
    fields = {
        "dm_mm": dm_mm,
        "d0_mm": d0_mm,
        "sigma_m_mm": sigma_m_mm,
        "log_nw": log_nw,
        "lwc_g_m3": lwc_g_m3,
        "nt_m3": nt_m3,
        "mu": np.broadcast_to(np.asarray(mu, dtype=float), dm_mm.shape),
    }
    return {key: np.asarray(val)[keep].astype(np.float32) for key, val in fields.items()}


def moments_abel_boutle(
    qr: np.ndarray, rho: np.ndarray, dmin_m: float, min_rr: float, max_dbz: float
) -> dict[str, np.ndarray]:
    """1-moment Abel-Boutle moments, mu = 0 (Mk = 0.22*gamma(k+1)*lambda^(2.2-1-k))."""
    rho_qr = rho * qr
    lamb = (rho_qr / AB_LAMBDA_PREFACTOR) ** (-1.0 / 1.8)  # 1/m
    mu = np.zeros_like(lamb)
    m0 = 0.22 * lamb**1.2
    m3 = 6.0 * rho_qr / (1000.0 * math.pi)
    m3b = 0.22 * math.gamma(4.0 + FALL_B) * lamb ** (2.2 - 1.0 - 3.0 - FALL_B)
    m4 = 0.22 * math.gamma(5.0) * lamb ** (2.2 - 1.0 - 4.0)
    m6 = 0.22 * math.gamma(7.0) * lamb ** (2.2 - 1.0 - 6.0)
    return _diag_from_moments(qr, rho, m0, m3, m3b, m4, m6, mu, lamb, dmin_m, min_rr, max_dbz)


def moments_two_moment(
    qr: np.ndarray,
    nr_per_kg: np.ndarray,
    rho: np.ndarray,
    dmin_m: float,
    min_rr: float,
    max_dbz: float,
) -> dict[str, np.ndarray]:
    """2-moment gamma moments with the ALARO variable mu(dmeanr) law."""
    from scipy.special import gammaln

    nt = nr_per_kg * rho  # zmom0
    m3 = 6.0 * qr * rho / (1000.0 * math.pi)  # zmom3
    with np.errstate(divide="ignore", invalid="ignore"):
        dmean_m = (m3 / nt) ** (1.0 / 3.0)
        mu = np.clip(19.0 * np.tanh(0.6 * (dmean_m * 1.0e3 - 1.8)) + 17.0, MU_MIN, MU_MAX)
        lamb = (nt * np.exp(gammaln(mu + 4.0) - gammaln(mu + 1.0)) / m3) ** (1.0 / 3.0)
        m3b = nt * np.exp(gammaln(mu + 4.0 + FALL_B) - gammaln(mu + 1.0)) / lamb ** (3.0 + FALL_B)
        m4 = nt * np.exp(gammaln(mu + 5.0) - gammaln(mu + 1.0)) / lamb**4
        m6 = nt * np.exp(gammaln(mu + 7.0) - gammaln(mu + 1.0)) / lamb**6
    return _diag_from_moments(qr, rho, nt, m3, m3b, m4, m6, mu, lamb, dmin_m, min_rr, max_dbz)
