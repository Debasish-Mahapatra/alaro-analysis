#!/usr/bin/env python3
"""Rain evaporation rate vs water-vapour mass fraction, one- vs two-moment scheme.

Note on terminology: ALARO/ARPEGE prognostic moisture variables (qv, qr, ...) are
SPECIFIC contents (mass fractions, mass per unit mass of moist air), not mixing
ratios. FOQS returns specific humidity and the formulas use these specific values
directly, so the numbers are the model's; only the axis label says "mass fraction".

This is a *theoretical* figure: the curves are evaluated straight from the
rain-evaporation formulas the ALARO runs actually used (ACEVMEL, Lopez block
``LA0MPS .AND. LEVAPLOP``), with the run-confirmed flags

    C1M / G1M  : LTWOMOMLIQ=F, LAB12=T   (one-moment rain)
    G2M        : LTWOMOMLIQ=T            (two-moment rain, variable shape)

so nothing is read from model output -- it is the scheme physics, plotted.

Formulas (acevmel.F90, rain / water phase), expressed as an evaporation *rate*
dqv/dt (the model's PEVAR is this rate times the physics timestep TSPHY, so the
timestep cancels and the curve is timestep-independent):

  common  : Ssat = 1 - qv/qsat = 1 - RH                       (= 0.20 at RH=80%)
            A = (1/(0.0231*RV)) * (L(T)/T)**2                 (heat conduction)
            B = (RV/2) * T * P / es(T)                        (vapour diffusion)

  1-moment (LAB12, Abel-Boutle 2012):
            rate = Ssat / (rho*(A+B)) * (2.2295*qr**(-1/9)
                                         + 8.738*P**(1/3)*qr**0.3807)

  2-moment (variable mu gamma DSD, Milbrandt 2005 shape law):
            rate = Ssat / (A+B) * ( G1*0.608*qr**(1/3)*Nr**0.667
                                  + G2*5.232*P**(1/3)*qr**0.628*Nr**0.37 )
            G1 = G(2+mu) * G(4+mu)**(-1/3) / G(1+mu)**0.6667
            G2 = G((5.7706+2*mu)/2) * G(1+mu)**(-0.372) * G(4+mu)**(-0.628)

with qr the specific rain content (mass fraction, kg/kg), Nr the specific rain number (#/kg) and
mu the rain shape parameter.  The two-moment scheme ties Nr and mu to qr through
the mean-volume diameter Dmean = (6*qr/(1000*pi*Nr))**(1/3):

            mu     = max(0.1, min(19*tanh(0.6*(Dmean_mm - 1.8)) + 17, 50))
            Nr     = 6*qr / (1000*pi*Dmean**3)                # invert for given Dmean

Thermodynamics are the ARPEGE/IFS functions FOEW/FOLH/FOQS with the cycle-46
constants from sucst.F90, so es(T), L(T) and qsat(T,P) match the model exactly.

x-axis: with RH held at 80%, qv = 0.80*qsat(T,P); sweeping temperature at a fixed
pressure sweeps qv along the x-axis (cold/dry -> warm/moist).  Evaporation rises
with qv because the A+B denominator collapses as es(T) grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

# --- ARPEGE / IFS cycle-46 thermodynamic constants (sucst.F90) ----------------
R_UNIV = 8.314511          # universal gas constant      [J/mol/K]
RMD, RMV = 28.9644, 18.0153
RD = 1000.0 * R_UNIV / RMD            # dry-air gas const  287.06
RV = 1000.0 * R_UNIV / RMV            # vapour gas const   461.52
RETV = RV / RD - 1.0                  # 0.6078
RCPV = 4.0 * RV
RCW = 4218.0
RTT = 273.16
RLVTT = 2.5008e6
RESTT = 611.14
RGAMW = (RCW - RCPV) / RV
RBETW = RLVTT / RV + RGAMW * RTT
RALPW = np.log(RESTT) + RBETW / RTT + RGAMW * np.log(RTT)

RD0R = 7.0e-5             # initial rain diameter after autoconversion [m] (RD0R)
RHO_W = 1000.0           # water density used in the scheme [kg/m3]

# Paper experiment palette (EXPERIMENT_COLORS): C1M red, G1M blue, G2M green.
COLOR_C1M = "#d62728"
COLOR_G1M = "#1f77b4"
COLOR_G2M = "#2ca02c"


# --- thermodynamic functions (fcttrm.func.h, water phase) ---------------------
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


# --- evaporation formulas (acevmel.F90, rate = PEVAR/TSPHY) -------------------
def _ab_denominator(t, p):
    es = foew(t)
    a = (1.0 / (0.0231 * RV)) * (folh(t) / t) ** 2          # ZCONDT
    b = (RV / 2.0) * t * p / es                              # ZFACT1/ZESW
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
    vent = 2.2295 * arg ** (-1.0 / 9.0) + 8.738 * p ** (1.0 / 3.0) * arg ** 0.3807
    return ssat / (rho * denom) * vent


def evap_2moment(qr, nr, mu, t, p, rh):
    """2-moment (LTWOMOMLIQ=T) rain evaporation rate [kg/kg/s]."""
    ssat = max(0.0, 1.0 - rh)
    denom = _ab_denominator(t, p)
    # gamma-function shape factors (use lgamma for numerical stability)
    g1 = np.exp(gammaln(2.0 + mu) - (1.0 / 3.0) * gammaln(4.0 + mu)
                - 0.6667 * gammaln(1.0 + mu))
    g2 = np.exp(gammaln(0.5 * (5.7706 + 2.0 * mu)) - 0.372 * gammaln(1.0 + mu)
                - 0.628 * gammaln(4.0 + mu))
    term1 = g1 * 0.608 * qr ** (1.0 / 3.0) * nr ** 0.667
    term2 = g2 * 5.232 * p ** (1.0 / 3.0) * qr ** 0.628 * nr ** 0.37
    return ssat / denom * (term1 + term2)


def mu_from_dmean(dmean_m):
    """Milbrandt-2005 variable rain shape parameter from mean-volume diameter."""
    dmean_mm = dmean_m * 1.0e3
    return max(0.1, min(19.0 * np.tanh(0.6 * (dmean_mm - 1.8)) + 17.0, 50.0))


def nr_from_qr_dmean(qr, dmean_m):
    """Specific rain number [#/kg] consistent with qr and a mean-volume diameter."""
    dmean_m = max(dmean_m, RD0R)
    return 6.0 * qr / (RHO_W * np.pi * dmean_m ** 3)


# ------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--rh", type=float, default=0.80, help="relative humidity (fraction)")
    p.add_argument("--pressure-hpa", type=float, default=850.0)
    p.add_argument("--qr-g-kg", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="specific rain contents (mass fractions) to draw [g/kg]")
    p.add_argument("--dmean-mm", type=float, default=1.0,
                   help="2-moment mean-volume drop diameter used to set Nr and mu [mm]")
    p.add_argument("--tmin-c", type=float, default=1.0, help="min temperature [degC]")
    p.add_argument("--tmax-c", type=float, default=33.0, help="max temperature [degC]")
    p.add_argument("--npts", type=int, default=201)
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument(
        "--plot-root", type=Path,
        default=Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper"),
    )
    return p


def main():
    args = build_parser().parse_args()
    rh = args.rh
    p_pa = args.pressure_hpa * 100.0
    dmean_m = args.dmean_mm * 1.0e-3
    qr_list = sorted(q * 1.0e-3 for q in args.qr_g_kg)        # kg/kg

    # temperature grid -> qv at fixed RH (qv = RH * qsat(T,P))
    t_grid = np.linspace(args.tmin_c + 273.15, args.tmax_c + 273.15, args.npts)
    qv_grid = rh * np.array([qsat_w(t, p_pa) for t in t_grid])   # kg/kg

    to_rate = 1000.0 * 3600.0       # kg/kg/s -> g/kg/h

    mu = mu_from_dmean(dmean_m)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    # qr family: line width / alpha ramp (thin/faint = low qr, bold = high qr)
    n = len(qr_list)
    widths = np.linspace(1.3, 2.7, n)
    alphas = np.linspace(0.55, 1.0, n)

    columns = {"qv_g_kg": qv_grid * 1.0e3, "T_degC": t_grid - 273.15}

    for qr, lw, al in zip(qr_list, widths, alphas):
        nr = nr_from_qr_dmean(qr, dmean_m)
        c1m = np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=True) for t in t_grid]) * to_rate
        g1m = np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=False) for t in t_grid]) * to_rate
        g2m = np.array([evap_2moment(qr, nr, mu, t, p_pa, rh) for t in t_grid]) * to_rate
        qr_g = qr * 1.0e3
        ax.plot(qv_grid * 1.0e3, c1m, color=COLOR_C1M, ls="--", lw=lw, alpha=al)
        ax.plot(qv_grid * 1.0e3, g1m, color=COLOR_G1M, ls="--", lw=lw, alpha=al)
        ax.plot(qv_grid * 1.0e3, g2m, color=COLOR_G2M, ls="-", lw=lw, alpha=al)
        # label each qr cluster in neutral grey, just above its topmost curve
        ax.annotate(f"$q_r$ = {qr_g:g} g kg$^{{-1}}$",
                    xy=(qv_grid[-1] * 1.0e3, max(c1m[-1], g1m[-1], g2m[-1])),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color="0.25")
        columns[f"C1M_qr{qr_g:g}_g_kg_h"] = c1m
        columns[f"G1M_qr{qr_g:g}_g_kg_h"] = g1m
        columns[f"G2M_qr{qr_g:g}_g_kg_h"] = g2m

    # experiment legend (colour + 1-moment dashed / 2-moment solid), qr-independent
    scheme_handles = [
        plt.Line2D([], [], color=COLOR_C1M, ls="--", lw=2.4, label="C1M  (1-moment)"),
        plt.Line2D([], [], color=COLOR_G1M, ls="--", lw=2.4, label="G1M  (1-moment)"),
        plt.Line2D([], [], color=COLOR_G2M, ls="-", lw=2.4, label="G2M  (2-moment)"),
    ]
    leg = ax.legend(handles=scheme_handles, loc="upper left", frameon=True,
                    framealpha=0.9, fontsize=9)
    ax.add_artist(leg)

    ax.set_xlabel("Water-vapour mass fraction  $q_v$  (g kg$^{-1}$)")
    ax.set_ylabel("Rain evaporation rate  (g kg$^{-1}$ h$^{-1}$)")
    ax.set_title("Rain evaporation rate versus water-vapour mass fraction", pad=52)
    ax.set_xlim(qv_grid[0] * 1.0e3, qv_grid[-1] * 1.0e3 * 1.10)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)

    # secondary top axis: temperature (qv is monotone in T at fixed RH)
    qv_mm = qv_grid * 1.0e3
    tc = t_grid - 273.15
    fwd = lambda x: np.interp(x, qv_mm, tc)
    inv = lambda x: np.interp(x, tc, qv_mm)
    secax = ax.secondary_xaxis("top", functions=(fwd, inv))
    secax.set_xlabel("Temperature at RH = 80 %  ($^\\circ$C)")

    fig.tight_layout()
    fig.text(0.5, -0.01,
             f"RH = {rh*100:g} %    $p$ = {args.pressure_hpa:g} hPa    "
             f"2-moment: $D_{{mean}}$ = {args.dmean_mm:g} mm, $\\mu$ = {mu:.1f}",
             ha="center", va="top", fontsize=8.5, color="0.3")

    out_dir = args.plot_root / "13. rain evaporation vs qv"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "13. rain evaporation vs qv_450dpi.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    print(f"rendered {png}")

    # data export, mirroring the other paper figures' _data.txt sidecar
    txt = out_dir / "13. rain evaporation vs qv_data.txt"
    keys = list(columns.keys())
    header = (
        "# Rain evaporation rate (g/kg/h) vs water-vapour mass fraction (specific content).\n"
        f"# RH={rh*100:g}%  p={args.pressure_hpa:g}hPa  Dmean(2M)={args.dmean_mm:g}mm  mu={mu:.3f}\n"
        "# C1M 1-moment LGRAPRO=F vent~rho*qr (acevmel.F90:645); "
        "G1M 1-moment LGRAPRO=T vent~qr (:522); G2M 2-moment LTWOMOMLIQ=T (:517)\n"
        + "\t".join(keys)
    )
    data = np.column_stack([np.asarray(columns[k], dtype=float) for k in keys])
    np.savetxt(txt, data, header=header, comments="", fmt="%.6g", delimiter="\t")
    print(f"wrote   {txt}")


if __name__ == "__main__":
    main()
