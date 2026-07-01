#!/usr/bin/env python3
"""Plot 14 - rain evaporation rate vs water-vapour mass fraction: sensitivity to
the rain NUMBER concentration Nr (two-moment scheme), at fixed rain content.

Companion to plot 13. Same theoretical setup (ACEVMEL Lopez evaporation, RH held
at 80%, qv swept by temperature at fixed pressure), but here the rain specific
content qr is held FIXED and the family variable is the rain number Nr. This
isolates the control the literature identifies (Seifert 2008; Morrison 2009): the
1-moment <-> 2-moment evaporation difference is set by number / mean drop size,
not by qr.

The one-moment schemes (C1M, G1M) carry no Nr, so each is a single reference curve
at the fixed qr. The two-moment scheme (G2M) is drawn for a family of Nr; for each
Nr the mean-volume diameter Dmean = (6 qr/(1000 pi Nr))**(1/3) and the shape mu
follow, and the G2M curve crosses the (Nr-independent) 1-moment baseline as Nr
grows (many small drops -> more evaporation).

Physics functions are imported from make_paper_rain_evaporation_qv (the plot-13
script), so the two figures share one adversarially-verified implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_paper_rain_evaporation_qv import (
    COLOR_C1M,
    COLOR_G1M,
    RHO_W,
    evap_1moment,
    evap_2moment,
    mu_from_dmean,
    qsat_w,
)


def dmean_from_qr_nr(qr, nr):
    """Mean-volume diameter [m] for a specific content qr and specific number nr."""
    return (6.0 * qr / (RHO_W * np.pi * nr)) ** (1.0 / 3.0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--rh", type=float, default=0.80, help="relative humidity (fraction)")
    p.add_argument("--pressure-hpa", type=float, default=850.0)
    p.add_argument("--qr-g-kg", type=float, default=1.0,
                   help="fixed specific rain content (mass fraction) [g/kg]")
    p.add_argument("--nr-per-kg", type=float, nargs="+",
                   default=[250.0, 1000.0, 4000.0, 16000.0],
                   help="2-moment rain number concentrations to draw [#/kg]")
    p.add_argument("--tmin-c", type=float, default=1.0)
    p.add_argument("--tmax-c", type=float, default=33.0)
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
    qr = args.qr_g_kg * 1.0e-3                      # kg/kg
    nr_list = sorted(args.nr_per_kg)

    t_grid = np.linspace(args.tmin_c + 273.15, args.tmax_c + 273.15, args.npts)
    qv_grid = rh * np.array([qsat_w(t, p_pa) for t in t_grid])
    qv_mm = qv_grid * 1.0e3
    tc = t_grid - 273.15
    to_rate = 1000.0 * 3600.0                       # kg/kg/s -> g/kg/h

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    columns = {"qv_g_kg": qv_mm, "T_degC": tc}
    handles = []

    # one-moment reference curves: no Nr dependence (single line each at fixed qr)
    c1m = np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=True) for t in t_grid]) * to_rate
    g1m = np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=False) for t in t_grid]) * to_rate
    ax.plot(qv_mm, c1m, color=COLOR_C1M, ls="--", lw=2.2)
    ax.plot(qv_mm, g1m, color=COLOR_G1M, ls="--", lw=2.2)
    handles += [
        plt.Line2D([], [], color=COLOR_C1M, ls="--", lw=2.2, label="C1M (1-moment, no $N_r$)"),
        plt.Line2D([], [], color=COLOR_G1M, ls="--", lw=2.2, label="G1M (1-moment, no $N_r$)"),
    ]
    columns["C1M_g_kg_h"] = c1m
    columns["G1M_g_kg_h"] = g1m

    # two-moment family over Nr: light green = few large drops, dark green = many small
    greens = plt.cm.Greens(np.linspace(0.45, 0.95, len(nr_list)))
    for nr, col in zip(nr_list, greens):
        dmean = dmean_from_qr_nr(qr, nr)
        mu = mu_from_dmean(dmean)
        g2m = np.array([evap_2moment(qr, nr, mu, t, p_pa, rh) for t in t_grid]) * to_rate
        ax.plot(qv_mm, g2m, color=col, ls="-", lw=2.2)
        handles.append(plt.Line2D(
            [], [], color=col, ls="-", lw=2.2,
            label=f"G2M  $N_r$={nr:g} kg$^{{-1}}$  ($D_m$={dmean*1e3:.2f} mm, $\\mu$={mu:.1f})"))
        columns[f"G2M_Nr{nr:g}_g_kg_h"] = g2m

    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9, fontsize=8)

    ax.set_xlabel("Water-vapour mass fraction  $q_v$  (g kg$^{-1}$)")
    ax.set_ylabel("Rain evaporation rate  (g kg$^{-1}$ h$^{-1}$)")
    ax.set_title("Rain evaporation rate versus water-vapour mass fraction "
                 "(varying rain number $N_r$)", pad=52)
    ax.set_xlim(qv_mm[0], qv_mm[-1])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)

    fwd = lambda x: np.interp(x, qv_mm, tc)
    inv = lambda x: np.interp(x, tc, qv_mm)
    secax = ax.secondary_xaxis("top", functions=(fwd, inv))
    secax.set_xlabel("Temperature at RH = 80 %  ($^\\circ$C)")

    fig.tight_layout()
    fig.text(0.5, -0.01,
             f"RH = {rh*100:g} %    $p$ = {args.pressure_hpa:g} hPa    "
             f"fixed $q_r$ = {args.qr_g_kg:g} g kg$^{{-1}}$",
             ha="center", va="top", fontsize=8.5, color="0.3")

    out_dir = args.plot_root / "14. rain evaporation Nr sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "14. rain evaporation Nr sensitivity_450dpi.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    print(f"rendered {png}")

    txt = out_dir / "14. rain evaporation Nr sensitivity_data.txt"
    keys = list(columns.keys())
    header = (
        "# Rain evaporation rate (g/kg/h) vs water-vapour mass fraction; 2-moment Nr sweep.\n"
        f"# RH={rh*100:g}%  p={args.pressure_hpa:g}hPa  fixed qr={args.qr_g_kg:g}g/kg\n"
        "# C1M/G1M (1-moment) carry no Nr; G2M columns are the 2-moment (LTWOMOMLIQ=T) per Nr.\n"
        + "\t".join(keys)
    )
    data = np.column_stack([np.asarray(columns[k], dtype=float) for k in keys])
    np.savetxt(txt, data, header=header, comments="", fmt="%.6g", delimiter="\t")
    print(f"wrote   {txt}")


if __name__ == "__main__":
    main()
