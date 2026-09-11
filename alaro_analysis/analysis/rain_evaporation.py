"""Rain evaporation box-model equations; units and closure choices follow the ALARO implementation."""

from __future__ import annotations

import numpy as np

RMD, RMV = 28.9644, 18.0153


R_UNIV = 8.314511  # universal gas constant      [J/mol/K]


RD = 1000.0 * R_UNIV / RMD  # dry-air gas const  287.06


RV = 1000.0 * R_UNIV / RMV  # vapour gas const   461.52


RETV = RV / RD - 1.0  # 0.6078


RCPV = 4.0 * RV


RCW = 4218.0


RTT = 273.16


RLVTT = 2.5008e6


RESTT = 611.14


RGAMW = (RCW - RCPV) / RV


RBETW = RLVTT / RV + RGAMW * RTT


RALPW = np.log(RESTT) + RBETW / RTT + RGAMW * np.log(RTT)


RD0R = 7.0e-5  # initial rain diameter after autoconversion [m] (RD0R)


RHO_W = 1000.0  # water density used in the scheme [kg/m3]


COLOR_C1M = "#d62728"


COLOR_G1M = "#1f77b4"


COLOR_G2M = "#2ca02c"


def foew(t):
    """Saturation vapour pressure over water [Pa]."""
    return np.exp(RALPW - RBETW / t - RGAMW * np.log(t))


def folh(t):
    """Latent heat of vaporisation [J/kg]."""
    return RV * (RBETW - RGAMW * t)


def foqs(x):
    """Saturation specific humidity from x = es/p."""
    return x / (1.0 + RETV * np.maximum(0.0, 1.0 - x))


def qsat_w(t, p):
    return foqs(foew(t) / p)


def _ab_denominator(t, p):
    es = foew(t)
    a = (1.0 / (0.0231 * RV)) * (folh(t) / t) ** 2  # ZCONDT
    b = (RV / 2.0) * t * p / es  # ZFACT1/ZESW
    return a + b


def evap_1moment(qr, t, p, rh, rho_weighted=False):
    """1-moment (LTWOMOMLIQ=F, LAB12=T) rain evaporation rate [kg/kg/s].

    The model has two 1-moment branches whose ventilation argument differs:
      * G1M (LGRAPRO=T): argument = qr           (acevmel.F90:522)  -> rho_weighted=False
      * C1M (LGRAPRO=F): argument = rho*qr        (acevmel.F90:645)  -> rho_weighted=True
    G1M is the right baseline against G2M (both LGRAPRO=T), isolating the rain moment.
    Both branches divide the prefactor by rho.
    """
    ssat = max(0.0, 1.0 - rh)
    rho = p / (RD * t)
    denom = _ab_denominator(t, p)
    arg = rho * qr if rho_weighted else qr
    vent = 2.2295 * arg ** (-1.0 / 9.0) + 8.738 * p ** (1.0 / 3.0) * arg**0.3807
    return ssat / (rho * denom) * vent


def evap_2moment(qr, nr, mu, t, p, rh):
    """2-moment (LTWOMOMLIQ=T) rain evaporation rate [kg/kg/s]."""
    from scipy.special import gammaln

    ssat = max(0.0, 1.0 - rh)
    denom = _ab_denominator(t, p)
    # gamma-function shape factors (use lgamma for numerical stability)
    g1 = np.exp(gammaln(2.0 + mu) - (1.0 / 3.0) * gammaln(4.0 + mu) - 0.6667 * gammaln(1.0 + mu))
    g2 = np.exp(
        gammaln(0.5 * (5.7706 + 2.0 * mu)) - 0.372 * gammaln(1.0 + mu) - 0.628 * gammaln(4.0 + mu)
    )
    term1 = g1 * 0.608 * qr ** (1.0 / 3.0) * nr**0.667
    term2 = g2 * 5.232 * p ** (1.0 / 3.0) * qr**0.628 * nr**0.37
    return ssat / denom * (term1 + term2)


def mu_from_dmean(dmean_m):
    """Milbrandt-2005 variable rain shape parameter from mean-volume diameter."""
    dmean_mm = dmean_m * 1.0e3
    return max(0.1, min(19.0 * np.tanh(0.6 * (dmean_mm - 1.8)) + 17.0, 50.0))


def nr_from_qr_dmean(qr, dmean_m):
    """Specific rain number [#/kg] consistent with qr and a mean-volume diameter."""
    dmean_m = max(dmean_m, RD0R)
    return 6.0 * qr / (RHO_W * np.pi * dmean_m**3)
