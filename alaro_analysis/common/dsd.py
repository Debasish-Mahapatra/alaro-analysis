"""Drop size distribution helpers for disdrometer/model comparisons.

Two parallel reductions are supported.

* **Path A — empirical**: Dm, D0, sigma_m, Nw computed straight from the
  raw observed N(D) and its moments (no DSD-shape assumption).  Used only
  for the disdrometer.

* **Path B — Marshall-Palmer projection**: every dataset is reduced to a
  pair ``(N0, lambda)`` of an exponential N(D) = N0 exp(-lambda D), then
  the integrated parameters follow analytic MP relations:
  Dm = 4/lambda, D0 = 3.67/lambda, sigma_m = 2/lambda, Nw = N0,
  Nt = N0/lambda, LWC = pi rho_w N0/lambda^4.
  - Disdrometer (path B): fit (N0, lambda) from QC'd LWC and Nt.
  - 2-moment model: fit (N0, lambda) from prognostic q_r and PNR.
  - 1-moment model: fix N0 = 8e6 m^-4 (Marshall-Palmer 1948), solve
    lambda from q_r alone.

QC follows ``disdro.tex`` Section 1: diameter range 0.312 mm <= D <= 8 mm
(already enforced by Parsivel bins), R > 0.1 mm/h, Z_e < 55 dBZ, and a
temporal continuity filter that drops contiguous rainy runs of fewer
than ``N_min`` minutes.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln


WATER_DENSITY_KG_M3 = 1000.0
NW_PREFACTOR = 256.0e3 / math.pi  # log Nw = log10(NW_PREFACTOR * LWC[g/m^3] / Dm[mm]^4)
MP_FIXED_N0_PER_M4 = 8.0e6        # Marshall & Palmer 1948 N0 (m^-4)
MP_FIXED_N0_PER_M3_MM = MP_FIXED_N0_PER_M4 * 1.0e-3  # 1/(m^3 mm)
ATLAS_WILLIAMS_A = 9.65
ATLAS_WILLIAMS_B = 10.30
ATLAS_WILLIAMS_C = 0.6
DEFAULT_QC_DIAMETER_MIN_MM = 0.312
DEFAULT_QC_DIAMETER_MAX_MM = 8.0
DEFAULT_QC_RAIN_RATE_MIN = 0.1     # mm/h
DEFAULT_QC_REFLECTIVITY_MAX_DBZ = 55.0
DEFAULT_QC_MIN_RUN_MINUTES = 5


# ---------------------------------------------------------------------------
# Marshall-Palmer math (Path B / model side)
# ---------------------------------------------------------------------------


def mp_lambda_from_lwc_nt(
    lwc_g_m3: np.ndarray | float,
    nt_m3: np.ndarray | float,
) -> np.ndarray:
    """Slope ``lambda`` (1/mm) from observed LWC (g/m^3) and Nt (1/m^3).

    For an exponential DSD N(D) = N0 exp(-lambda D):
        M0 = N0/lambda, M3 = 6 N0/lambda^4
        => lambda = (6 M0 / M3)^(1/3)
    With ``M3[mm^3/m^3] = (6/pi) * LWC_volume[mm^3/m^3] = (6/pi) * 1000 *
    LWC[g/m^3]`` and M0 = Nt, this gives::

        lambda[1/mm] = (pi * Nt / (1000 * LWC[g/m^3]))**(1/3)
    """
    lwc = np.asarray(lwc_g_m3, dtype=float)
    nt = np.asarray(nt_m3, dtype=float)
    out = np.full(np.broadcast(lwc, nt).shape, np.nan, dtype=float)
    mask = np.isfinite(lwc) & np.isfinite(nt) & (lwc > 0.0) & (nt > 0.0)
    out[mask] = np.cbrt(math.pi * nt[mask] / (1000.0 * lwc[mask]))
    return out


def mp_n0_from_lwc_nt(
    lwc_g_m3: np.ndarray | float,
    nt_m3: np.ndarray | float,
) -> np.ndarray:
    """Intercept ``N0`` in 1/(m^3 mm) from LWC (g/m^3) and Nt (1/m^3)."""
    lwc = np.asarray(lwc_g_m3, dtype=float)
    nt = np.asarray(nt_m3, dtype=float)
    lam = mp_lambda_from_lwc_nt(lwc, nt)
    return nt * lam


def mp_lambda_from_q_with_fixed_n0(
    q_r_kgkg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
    n0_per_m3_mm: float = MP_FIXED_N0_PER_M3_MM,
) -> np.ndarray:
    """Slope ``lambda`` (1/mm) for an MP DSD with fixed N0 (1-moment closure).

    For exponential, ``LWC[mm^3/m^3] = pi N0 / lambda^4``.
    ``LWC[mm^3/m^3] = 1000 * LWC[g/m^3] = 1e6 * rho_air * q_r``.
    Therefore::

        lambda[1/mm] = (pi * N0[1/(m^3 mm)] / (1e6 * rho_air * q_r))**(1/4)
    """
    qr = np.asarray(q_r_kgkg, dtype=float)
    rho = np.asarray(rho_air_kg_m3, dtype=float)
    out = np.full(np.broadcast(qr, rho).shape, np.nan, dtype=float)
    mask = np.isfinite(qr) & np.isfinite(rho) & (qr > 0.0) & (rho > 0.0)
    denom = 1.0e6 * rho[mask] * qr[mask]
    out[mask] = (math.pi * n0_per_m3_mm / denom) ** 0.25
    return out


def mp_diagnostics_from_n0_lambda(
    n0_per_m3_mm: np.ndarray | float,
    lambda_per_mm: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Return integrated DSD parameters from MP (N0, lambda).

    Output keys: ``dm_mm``, ``d0_mm``, ``sigma_m_mm``, ``nt_m3``,
    ``lwc_g_m3``, ``nw_m3_mm``, ``log_nw``.
    """
    n0 = np.asarray(n0_per_m3_mm, dtype=float)
    lam = np.asarray(lambda_per_mm, dtype=float)
    shape = np.broadcast(n0, lam).shape
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "dm_mm", "d0_mm", "sigma_m_mm", "nt_m3", "lwc_g_m3", "nw_m3_mm", "log_nw",
    )}
    mask = np.isfinite(n0) & np.isfinite(lam) & (n0 > 0.0) & (lam > 0.0)
    if not mask.any():
        return out
    n0_v = n0[mask] if n0.ndim else np.full(mask.sum(), float(n0))
    lam_v = lam[mask] if lam.ndim else np.full(mask.sum(), float(lam))
    out["dm_mm"][mask] = 4.0 / lam_v
    out["d0_mm"][mask] = 3.67 / lam_v
    out["sigma_m_mm"][mask] = 2.0 / lam_v
    out["nt_m3"][mask] = n0_v / lam_v
    # LWC[g/m^3] = pi * N0 / lambda^4 / 1000 (because LWC[mm^3/m^3] = pi N0/lambda^4
    # and LWC[g/m^3] = LWC[mm^3/m^3] / 1000 for water)
    out["lwc_g_m3"][mask] = math.pi * n0_v / (lam_v ** 4) / 1000.0
    out["nw_m3_mm"][mask] = n0_v
    out["log_nw"][mask] = np.log10(n0_v)
    return out


def mp_from_q_n_per_kg(
    q_r_kgkg: np.ndarray | float,
    n_r_per_kg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Marshall-Palmer DSD from prognostic 2-moment ``q_r`` and ``N``.

    ``N0 = Nt * lambda`` and ``lambda = (pi * Nt / (1000 * LWC))^{1/3}``.
    """
    qr = np.asarray(q_r_kgkg, dtype=float)
    nk = np.asarray(n_r_per_kg, dtype=float)
    rho = np.asarray(rho_air_kg_m3, dtype=float)
    shape = np.broadcast(qr, nk, rho).shape
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
        "dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw",
    )}
    mask = (
        np.isfinite(qr) & np.isfinite(nk) & np.isfinite(rho)
        & (qr > 0.0) & (nk > 0.0) & (rho > 0.0)
    )
    if not mask.any():
        return out
    qr_v = qr[mask] if qr.ndim else np.full(mask.sum(), float(qr))
    nk_v = nk[mask] if nk.ndim else np.full(mask.sum(), float(nk))
    rho_v = rho[mask] if rho.ndim else np.full(mask.sum(), float(rho))
    lwc = 1000.0 * rho_v * qr_v   # g/m^3
    nt = rho_v * nk_v             # 1/m^3
    lam = mp_lambda_from_lwc_nt(lwc, nt)
    n0 = nt * lam
    diag = mp_diagnostics_from_n0_lambda(n0, lam)
    out["lwc_g_m3"][mask] = lwc
    out["nt_m3"][mask] = nt
    out["lambda_per_mm"][mask] = lam
    out["n0_per_m3_mm"][mask] = n0
    for key in ("dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw"):
        out[key][mask] = diag[key]
    return out


AB12_X1_PER_M_1POINT8 = 0.22
AB12_X2 = 2.2
AB12_PREFACTOR = math.pi * WATER_DENSITY_KG_M3 * AB12_X1_PER_M_1POINT8  # ~691.15
AB12_LAMBDA_EXPONENT = 1.0 / (4.0 - AB12_X2)  # 0.5556
ALARO_VARIABLE_MU_MIN = 0.1
ALARO_VARIABLE_MU_MAX = 50.0


def gamma_dsd_from_q_n_per_kg(
    q_r_kgkg: np.ndarray | float,
    n_r_per_kg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
    *,
    mu: np.ndarray | float = 1.0,
) -> dict[str, np.ndarray]:
    """ALARO 2-moment rain DSD: gamma ``N(D) = N0 D^mu exp(-lambda D)``.

    The default scalar shape ``mu = 1`` matches ``ZSHAPER = 1._JPRB`` in
    ``arpifs/phys_dmn/aplmphys.F90``.  ``mu`` may also be an array broadcastable
    to the input fields, for closures where the shape varies by cell.

    Closed form for the integrated parameters::

        D_v   = (6 q_r / (pi rho_w n_r))**(1/3)
        lambda = ((mu+1)(mu+2)(mu+3))**(1/3) / D_v
        D_m   = (mu+4) / lambda
        D_0   = (3.67+mu)/(4+mu) * D_m
        sigma_m = D_m / sqrt(mu+4)
        N_w   = 4**4 / (pi rho_w) * LWC / D_m**4    (universal definition)

    Returns the same key set as the MP helpers so downstream code is
    interchangeable.  ``log_nw`` is the universal Bringi/Testud value, not the
    underlying gamma intercept ``N0``.
    """
    qr = np.asarray(q_r_kgkg, dtype=float)
    nk = np.asarray(n_r_per_kg, dtype=float)
    rho = np.asarray(rho_air_kg_m3, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    shape = np.broadcast(qr, nk, rho, mu_arr).shape
    qr_b = np.broadcast_to(qr, shape)
    nk_b = np.broadcast_to(nk, shape)
    rho_b = np.broadcast_to(rho, shape)
    mu_b = np.broadcast_to(mu_arr, shape)
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
        "dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw", "mu",
    )}
    mask = (
        np.isfinite(qr_b) & np.isfinite(nk_b) & np.isfinite(rho_b) & np.isfinite(mu_b)
        & (qr_b > 0.0) & (nk_b > 0.0) & (rho_b > 0.0) & (mu_b > -1.0)
    )
    if not mask.any():
        return out
    qr_v = qr_b[mask]
    nk_v = nk_b[mask]
    rho_v = rho_b[mask]
    mu_v = mu_b[mask]

    dmean_v_m = (6.0 * qr_v / (math.pi * WATER_DENSITY_KG_M3 * nk_v)) ** (1.0 / 3.0)
    dmean_v_mm = dmean_v_m * 1000.0
    factor = ((mu_v + 1.0) * (mu_v + 2.0) * (mu_v + 3.0)) ** (1.0 / 3.0)
    lam_per_mm = factor / dmean_v_mm
    dm_mm = (mu_v + 4.0) / lam_per_mm
    d0_mm = (3.67 + mu_v) / (4.0 + mu_v) * dm_mm
    sigma_m_mm = dm_mm / np.sqrt(mu_v + 4.0)
    lwc_g_m3 = 1000.0 * qr_v * rho_v
    nt_m3 = nk_v * rho_v
    nw_m3_mm = NW_PREFACTOR * lwc_g_m3 / dm_mm ** 4
    log_nw = np.log10(np.where(nw_m3_mm > 0.0, nw_m3_mm, np.nan))
    # underlying gamma N0: integral N(D) dD = N0 Gamma(mu+1) / lambda^(mu+1) = Nt
    n0_per_m3_mm = nt_m3 * lam_per_mm ** (mu_v + 1.0) / np.exp(gammaln(mu_v + 1.0))

    out["lwc_g_m3"][mask] = lwc_g_m3
    out["nt_m3"][mask] = nt_m3
    out["lambda_per_mm"][mask] = lam_per_mm
    out["n0_per_m3_mm"][mask] = n0_per_m3_mm
    out["dm_mm"][mask] = dm_mm
    out["d0_mm"][mask] = d0_mm
    out["sigma_m_mm"][mask] = sigma_m_mm
    out["nw_m3_mm"][mask] = nw_m3_mm
    out["log_nw"][mask] = log_nw
    out["mu"][mask] = mu_v
    return out


def alaro_variable_mu_from_dmean_mm(dmean_mm: np.ndarray | float) -> np.ndarray:
    """ALARO variable rain gamma shape parameter from mean diameter in mm."""
    dmean = np.asarray(dmean_mm, dtype=float)
    mu = 19.0 * np.tanh(0.6 * (dmean - 1.8)) + 17.0
    return np.clip(mu, ALARO_VARIABLE_MU_MIN, ALARO_VARIABLE_MU_MAX)


def gamma_dsd_from_q_n_variable_mu_per_kg(
    q_r_kgkg: np.ndarray | float,
    n_r_per_kg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """ALARO 2-moment gamma DSD using the variable-``mu`` shape law.

    The shape is

        mu = max(0.1, min(50, 19*tanh(0.6*(dmean_mm - 1.8)) + 17))

    with ``dmean_mm = 1000 * (6*q_r/(pi*rho_w*n_r))**(1/3)``.
    """
    qr = np.asarray(q_r_kgkg, dtype=float)
    nk = np.asarray(n_r_per_kg, dtype=float)
    shape = np.broadcast(qr, nk).shape
    qr_b = np.broadcast_to(qr, shape)
    nk_b = np.broadcast_to(nk, shape)
    dmean_mm = np.full(shape, np.nan, dtype=float)
    mask = np.isfinite(qr_b) & np.isfinite(nk_b) & (qr_b > 0.0) & (nk_b > 0.0)
    dmean_mm[mask] = (
        1000.0
        * (6.0 * qr_b[mask] / (math.pi * WATER_DENSITY_KG_M3 * nk_b[mask])) ** (1.0 / 3.0)
    )
    mu = alaro_variable_mu_from_dmean_mm(dmean_mm)
    return gamma_dsd_from_q_n_per_kg(qr, nk, rho_air_kg_m3, mu=mu)


def normalized_gamma_diagnostics_from_lwc_dm_mu(
    lwc_g_m3: np.ndarray | float,
    dm_mm: np.ndarray | float,
    mu: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Diagnostics for the normalized gamma DSD.

    Uses

        N(D) = Nw * f(mu) * (D/Dm)^mu * exp[-(4 + mu) D/Dm]

    with the universal Bringi/Testud definition of ``Nw``.  The median
    volume diameter follows the same analytic approximation used elsewhere in
    this module, which is exact enough for the standard exponential case and
    keeps model and observation reductions consistent.
    """
    lwc = np.asarray(lwc_g_m3, dtype=float)
    dm = np.asarray(dm_mm, dtype=float)
    shape_mu = np.asarray(mu, dtype=float)
    shape = np.broadcast(lwc, dm, shape_mu).shape
    lwc_b = np.broadcast_to(lwc, shape)
    dm_b = np.broadcast_to(dm, shape)
    mu_b = np.broadcast_to(shape_mu, shape)
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "dm_mm", "d0_mm", "sigma_m_mm", "lwc_g_m3", "nt_m3",
        "lambda_per_mm", "n0_per_m3_mm", "nw_m3_mm", "log_nw", "mu",
    )}
    mask = (
        np.isfinite(lwc_b) & np.isfinite(dm_b) & np.isfinite(mu_b)
        & (lwc_b > 0.0) & (dm_b > 0.0) & (mu_b > -1.0)
    )
    if not mask.any():
        return out

    lwc_v = lwc_b[mask]
    dm_v = dm_b[mask]
    mu_v = mu_b[mask]
    nw = NW_PREFACTOR * lwc_v / dm_v ** 4
    valid = np.isfinite(nw) & (nw > 0.0)
    if not valid.all():
        lwc_v = lwc_v[valid]
        dm_v = dm_v[valid]
        mu_v = mu_v[valid]
        nw = nw[valid]
        idx = np.flatnonzero(mask)[valid]
    else:
        idx = np.flatnonzero(mask)

    lam = (mu_v + 4.0) / dm_v
    coeff_log = (
        math.log(6.0 / 256.0)
        + (mu_v + 4.0) * np.log(mu_v + 4.0)
        - gammaln(mu_v + 4.0)
    )
    nt_log = (
        np.log(nw)
        + coeff_log
        + np.log(dm_v)
        + gammaln(mu_v + 1.0)
        - (mu_v + 1.0) * np.log(mu_v + 4.0)
    )
    n0_log = np.log(nw) + coeff_log - mu_v * np.log(dm_v)

    for key, values in {
        "dm_mm": dm_v,
        "d0_mm": (3.67 + mu_v) / (4.0 + mu_v) * dm_v,
        "sigma_m_mm": dm_v / np.sqrt(mu_v + 4.0),
        "lwc_g_m3": lwc_v,
        "nt_m3": np.exp(nt_log),
        "lambda_per_mm": lam,
        "n0_per_m3_mm": np.exp(n0_log),
        "nw_m3_mm": nw,
        "log_nw": np.log10(nw),
        "mu": mu_v,
    }.items():
        out[key].flat[idx] = values
    return out


def normalized_gamma_diagnostics_from_lwc_nt_mu(
    lwc_g_m3: np.ndarray | float,
    nt_m3: np.ndarray | float,
    mu: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Diagnostics for a normalized gamma DSD fitted from LWC, Nt, and mu."""
    lwc = np.asarray(lwc_g_m3, dtype=float)
    nt = np.asarray(nt_m3, dtype=float)
    shape_mu = np.asarray(mu, dtype=float)
    shape = np.broadcast(lwc, nt, shape_mu).shape
    lwc_b = np.broadcast_to(lwc, shape)
    nt_b = np.broadcast_to(nt, shape)
    mu_b = np.broadcast_to(shape_mu, shape)
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "dm_mm", "d0_mm", "sigma_m_mm", "lwc_g_m3", "nt_m3",
        "lambda_per_mm", "n0_per_m3_mm", "nw_m3_mm", "log_nw", "mu",
    )}
    mask = (
        np.isfinite(lwc_b) & np.isfinite(nt_b) & np.isfinite(mu_b)
        & (lwc_b > 0.0) & (nt_b > 0.0) & (mu_b > -1.0)
    )
    if not mask.any():
        return out

    lwc_v = lwc_b[mask]
    nt_v = nt_b[mask]
    mu_v = mu_b[mask]
    ratio_log = gammaln(mu_v + 4.0) - gammaln(mu_v + 1.0)
    lam = np.exp((math.log(math.pi / 6000.0) + np.log(nt_v) - np.log(lwc_v) + ratio_log) / 3.0)
    dm = (mu_v + 4.0) / lam
    nw = NW_PREFACTOR * lwc_v / dm ** 4
    valid = np.isfinite(lam) & (lam > 0.0) & np.isfinite(dm) & (dm > 0.0) & np.isfinite(nw) & (nw > 0.0)
    if not valid.any():
        return out

    idx = np.flatnonzero(mask)[valid]
    lwc_v = lwc_v[valid]
    nt_v = nt_v[valid]
    mu_v = mu_v[valid]
    lam = lam[valid]
    dm = dm[valid]
    nw = nw[valid]
    n0 = nt_v * lam ** (mu_v + 1.0) / np.exp(gammaln(mu_v + 1.0))

    for key, values in {
        "dm_mm": dm,
        "d0_mm": (3.67 + mu_v) / (4.0 + mu_v) * dm,
        "sigma_m_mm": dm / np.sqrt(mu_v + 4.0),
        "lwc_g_m3": lwc_v,
        "nt_m3": nt_v,
        "lambda_per_mm": lam,
        "n0_per_m3_mm": n0,
        "nw_m3_mm": nw,
        "log_nw": np.log10(nw),
        "mu": mu_v,
    }.items():
        out[key].flat[idx] = values
    return out


def normalized_gamma_from_empirical_samples(
    samples: dict[str, np.ndarray],
    *,
    mu_min: float = -0.95,
    mu_max: float = 50.0,
) -> dict[str, np.ndarray]:
    """Fit normalized-gamma diagnostics from empirical Dm, sigma_m, and LWC."""
    dm = np.asarray(samples["dm_mm"], dtype=float)
    sigma = np.asarray(samples["sigma_m_mm"], dtype=float)
    lwc = np.asarray(samples["lwc_g_m3"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (dm / sigma) ** 2 - 4.0
    mu = np.clip(mu, mu_min, mu_max)
    diag = normalized_gamma_diagnostics_from_lwc_dm_mu(lwc, dm, mu)
    keep = (
        np.isfinite(diag["dm_mm"]) & (diag["dm_mm"] > 0.0)
        & np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] > 0.0)
        & np.isfinite(diag["sigma_m_mm"]) & (diag["sigma_m_mm"] > 0.0)
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"]) & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"]) & (diag["nt_m3"] > 0.0)
    )
    keys = ("dm_mm", "d0_mm", "sigma_m_mm", "log_nw", "lwc_g_m3", "nt_m3", "mu")
    return {key: diag[key][keep].astype(np.float32) for key in keys}


def mp_from_q_abel_boutle(
    q_r_kgkg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """1-moment ALARO rain DSD via Abel & Boutle (2012).

    From ``arpifs/adiab/gpprs0d.F90``:

        N(D) = 0.22 * lambda^2.2 * exp(-lambda * D)        # SI units
        lambda = (rho_w * pi * 0.22 / (rho_a * q_r))**(1/(4-2.2))
               = (691.15 / (rho_a * q_r))**0.55             # 1/m

    Therefore N0 itself varies with q_r:
        N0 = 0.22 * lambda^2.2                              # 1/m^4

    Returns the same dict keys as the other MP helpers, with quantities in
    the workflow units (``lambda`` in 1/mm, ``N0`` in 1/(m^3 mm)).
    """
    qr = np.asarray(q_r_kgkg, dtype=float)
    rho = np.asarray(rho_air_kg_m3, dtype=float)
    shape = np.broadcast(qr, rho).shape
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
        "dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw",
    )}
    mask = np.isfinite(qr) & np.isfinite(rho) & (qr > 0.0) & (rho > 0.0)
    if not mask.any():
        return out
    qr_v = qr[mask] if qr.ndim else np.full(mask.sum(), float(qr))
    rho_v = rho[mask] if rho.ndim else np.full(mask.sum(), float(rho))
    rho_qr = rho_v * qr_v                          # kg/m^3
    lam_per_m = (AB12_PREFACTOR / rho_qr) ** AB12_LAMBDA_EXPONENT
    lam_per_mm = lam_per_m / 1000.0
    n0_per_m4 = AB12_X1_PER_M_1POINT8 * lam_per_m ** AB12_X2
    n0_per_m3_mm = n0_per_m4 * 1.0e-3
    diag = mp_diagnostics_from_n0_lambda(n0_per_m3_mm, lam_per_mm)
    out["lwc_g_m3"][mask] = 1000.0 * rho_qr
    out["nt_m3"][mask] = diag["nt_m3"]
    out["lambda_per_mm"][mask] = lam_per_mm
    out["n0_per_m3_mm"][mask] = n0_per_m3_mm
    for key in ("dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw"):
        out[key][mask] = diag[key]
    return out


def mp_from_q_fixed_n0(
    q_r_kgkg: np.ndarray | float,
    rho_air_kg_m3: np.ndarray | float,
    n0_per_m3_mm: float = MP_FIXED_N0_PER_M3_MM,
) -> dict[str, np.ndarray]:
    """1-moment Marshall-Palmer closure with fixed ``N0``."""
    qr = np.asarray(q_r_kgkg, dtype=float)
    rho = np.asarray(rho_air_kg_m3, dtype=float)
    shape = np.broadcast(qr, rho).shape
    nan = np.full(shape, np.nan, dtype=float)
    out = {key: nan.copy() for key in (
        "lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
        "dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw",
    )}
    mask = np.isfinite(qr) & np.isfinite(rho) & (qr > 0.0) & (rho > 0.0)
    if not mask.any():
        return out
    qr_v = qr[mask] if qr.ndim else np.full(mask.sum(), float(qr))
    rho_v = rho[mask] if rho.ndim else np.full(mask.sum(), float(rho))
    lwc = 1000.0 * rho_v * qr_v
    lam = mp_lambda_from_q_with_fixed_n0(qr_v, rho_v, n0_per_m3_mm)
    n0 = np.full(lam.shape, float(n0_per_m3_mm))
    nt = n0 / lam
    diag = mp_diagnostics_from_n0_lambda(n0, lam)
    out["lwc_g_m3"][mask] = lwc
    out["nt_m3"][mask] = nt
    out["lambda_per_mm"][mask] = lam
    out["n0_per_m3_mm"][mask] = n0
    for key in ("dm_mm", "d0_mm", "sigma_m_mm", "nw_m3_mm", "log_nw"):
        out[key][mask] = diag[key]
    return out


# ---------------------------------------------------------------------------
# Empirical observation reduction (Path A)
# ---------------------------------------------------------------------------


def empirical_d0_from_psd(
    number_density: np.ndarray,
    bin_centers_mm: np.ndarray,
    bin_widths_mm: np.ndarray,
) -> np.ndarray:
    """Empirical D0 from observed N(D): diameter splitting cumulative mass at 50%.

    ``number_density`` is (T, B) in 1/(m^3 mm); bin metadata is (B,) or (B, T).
    """
    nd = np.asarray(number_density, dtype=float)
    if nd.ndim != 2:
        raise ValueError("number_density must be (T, B)")
    centers = np.asarray(bin_centers_mm, dtype=float)
    widths = np.asarray(bin_widths_mm, dtype=float)
    if centers.ndim == 1:
        centers = np.broadcast_to(centers, nd.shape)
    elif centers.shape != nd.shape and centers.T.shape == nd.shape:
        centers = centers.T
    if widths.ndim == 1:
        widths = np.broadcast_to(widths, nd.shape)
    elif widths.shape != nd.shape and widths.T.shape == nd.shape:
        widths = widths.T
    nd_clean = np.where(np.isfinite(nd) & (nd >= 0.0), nd, 0.0)
    centers_c = np.where(np.isfinite(centers) & (centers > 0.0), centers, 0.0)
    widths_c = np.where(np.isfinite(widths) & (widths > 0.0), widths, 0.0)
    mass = nd_clean * centers_c ** 3 * widths_c
    cum = np.cumsum(mass, axis=1)
    total = cum[:, -1]
    out = np.full(nd.shape[0], np.nan, dtype=float)
    valid = total > 0.0
    if not valid.any():
        return out
    half = 0.5 * total
    n_bins = nd.shape[1]
    for t in np.flatnonzero(valid):
        cum_t = cum[t]
        idx = int(np.searchsorted(cum_t, half[t]))
        if idx >= n_bins:
            out[t] = centers_c[t, -1] + 0.5 * widths_c[t, -1]
        else:
            cum_left = cum_t[idx - 1] if idx > 0 else 0.0
            cum_right = cum_t[idx]
            span = cum_right - cum_left
            frac = (half[t] - cum_left) / span if span > 0 else 0.0
            left_edge = centers_c[t, idx] - 0.5 * widths_c[t, idx]
            out[t] = left_edge + frac * widths_c[t, idx]
    return out


def empirical_dsd_parameters(
    number_density: np.ndarray,
    bin_centers_mm: np.ndarray,
    bin_widths_mm: np.ndarray,
    moment3: np.ndarray,
    moment4: np.ndarray,
    moment5: np.ndarray,
) -> dict[str, np.ndarray]:
    """Empirical Dm, D0, sigma_m, log Nw, LWC, Nt from raw N(D) and moments.

    Inputs are time-series at 1-minute resolution.  Returns NaN for samples
    with non-positive M3 or no detectable rain.
    """
    m3 = np.asarray(moment3, dtype=float)
    m4 = np.asarray(moment4, dtype=float)
    m5 = np.asarray(moment5, dtype=float)
    safe_m3 = np.where(np.isfinite(m3) & (m3 > 0.0), m3, np.nan)
    dm_mm = m4 / safe_m3
    sigma_mm = np.sqrt(np.maximum(m5 / safe_m3 - dm_mm ** 2, 0.0))
    # LWC[g/m^3] = (pi rho_w / 6) M3, with M3 in mm^3/m^3 and the
    # 1e-3 factor converting mm^3 of water at rho_w=1g/cm^3 to grams.
    lwc_g_m3 = (math.pi / 6.0) * safe_m3 * 1.0e-3
    nd = np.asarray(number_density, dtype=float)
    widths = np.asarray(bin_widths_mm, dtype=float)
    if widths.ndim == 1:
        nt = np.nansum(nd * widths[None, :], axis=1)
    elif widths.shape == nd.shape:
        nt = np.nansum(nd * widths, axis=1)
    elif widths.T.shape == nd.shape:
        nt = np.nansum(nd * widths.T, axis=1)
    else:
        raise ValueError("bin_widths shape mismatch")
    d0 = empirical_d0_from_psd(nd, bin_centers_mm, bin_widths_mm)
    nw = NW_PREFACTOR * lwc_g_m3 / dm_mm ** 4
    log_nw = np.full_like(nw, np.nan)
    pos = np.isfinite(nw) & (nw > 0.0)
    log_nw[pos] = np.log10(nw[pos])
    return {
        "dm_mm": dm_mm,
        "d0_mm": d0,
        "sigma_m_mm": sigma_mm,
        "lwc_g_m3": lwc_g_m3,
        "nt_m3": nt.astype(float),
        "log_nw": log_nw,
    }


# ---------------------------------------------------------------------------
# QC (disdro.tex Section 1)
# ---------------------------------------------------------------------------


def temporal_continuity_mask(
    rainy_mask: np.ndarray,
    min_run_minutes: int = DEFAULT_QC_MIN_RUN_MINUTES,
) -> np.ndarray:
    """Drop contiguous runs of rainy minutes shorter than ``min_run_minutes``.

    Pure NumPy: O(T).  Returns a boolean array of the same shape as input.
    """
    rm = np.asarray(rainy_mask, dtype=bool)
    if rm.size == 0 or min_run_minutes <= 1:
        return rm.copy()
    out = rm.copy()
    n = rm.size
    i = 0
    while i < n:
        if not rm[i]:
            i += 1
            continue
        j = i
        while j < n and rm[j]:
            j += 1
        if (j - i) < min_run_minutes:
            out[i:j] = False
        i = j
    return out


def apply_disdrometer_qc(
    precip_rate_mm_h: np.ndarray,
    reflectivity_dbz: np.ndarray,
    *,
    min_rr_mm_h: float = DEFAULT_QC_RAIN_RATE_MIN,
    max_z_dbz: float = DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    min_run_minutes: int = DEFAULT_QC_MIN_RUN_MINUTES,
) -> np.ndarray:
    """Return the boolean QC mask following disdro.tex Section 1."""
    rr = np.asarray(precip_rate_mm_h, dtype=float)
    z = np.asarray(reflectivity_dbz, dtype=float)
    rr_finite = np.where(np.isfinite(rr), rr, -np.inf)
    z_finite = np.where(np.isfinite(z), z, np.inf)
    cand = (rr_finite > min_rr_mm_h) & (z_finite < max_z_dbz)
    return temporal_continuity_mask(cand, min_run_minutes=min_run_minutes)


# ---------------------------------------------------------------------------
# Backwards-compatible Atlas-Williams helper (kept for tests)
# ---------------------------------------------------------------------------


def mass_weighted_fall_velocity_atlas(
    dm_mm: np.ndarray | float,
    *,
    nu: float = 1.0,
) -> np.ndarray:
    """Mass-weighted fall speed (m/s) under the Atlas-Williams power law.

    Provided for legacy callers; not used by the current MP-based workflow.
    For an exponential DSD with shape parameter equal to ``nu - 1`` in the
    legacy convention used here, the mass-weighted velocity reduces to::

        V_m = a - b * ((nu+3) / (nu+3 + c*Dm))**(nu+3)

    with Atlas-Williams constants (a=9.65, b=10.30, c=0.6).
    """
    dm = np.asarray(dm_mm, dtype=float)
    n = float(nu)
    out = np.full_like(dm, np.nan)
    mask = np.isfinite(dm) & (dm > 0.0)
    ratio = (n + 3.0) / (n + 3.0 + ATLAS_WILLIAMS_C * dm[mask])
    out[mask] = ATLAS_WILLIAMS_A - ATLAS_WILLIAMS_B * np.power(ratio, n + 3.0)
    return out


def rain_rate_from_lwc_velocity(
    lwc_g_m3: np.ndarray | float,
    velocity_m_s: np.ndarray | float,
) -> np.ndarray:
    """Rain rate (mm/h) from LWC (g/m^3) and bulk fall speed (m/s)."""
    lwc = np.asarray(lwc_g_m3, dtype=float)
    v = np.asarray(velocity_m_s, dtype=float)
    return 3.6 * np.maximum(lwc, 0.0) * np.maximum(v, 0.0)
