#!/usr/bin/env python3
"""Plot 15 - rain evaporation rate over the rain content (qr) and number (Nr) plane.

Suggested by David: the two-moment scheme spans the whole qr-Nr plane, while the
one-moment scheme carries a single Nr for a given qr, so it collapses to one line
(its Abel-Boutle Nr(qr) locus). Here the colour field is the two-moment (G2M)
evaporation rate over the plane, and the one-moment (G1M) AB12 locus is drawn on
top, coloured on the same scale by its own evaporation rate. Comparing the line
colour to the surrounding field shows whether, at the number AB12 assigns, the
two-moment evaporates more or less than the one-moment.

Thermodynamics are fixed (RH=80%, p, T): that prefactor only scales the whole
field, it does not change its shape. Physics functions come from the plot-13
script make_paper_rain_evaporation_qv (one verified implementation for all three
figures).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from make_paper_rain_evaporation_qv import (
    RD,
    RHO_W,
    evap_1moment,
    evap_2moment,
    qsat_w,
)

AB_LAMBDA_PREFACTOR = np.pi * 220.0     # Abel-Boutle: lambda=(rho*qr/691.15)**(-1/1.8)


def mu_grid(dmean_m):
    """Vectorised Milbrandt-2005 shape parameter (qr>0, so sign()=+1)."""
    dmm = dmean_m * 1.0e3
    return np.clip(19.0 * np.tanh(0.6 * (dmm - 1.8)) + 17.0, 0.1, 50.0)


def ab12_number_per_kg(qr, rho):
    """One-moment Abel-Boutle diagnostic rain number Nt [#/kg] for content qr."""
    lamb = (rho * qr / AB_LAMBDA_PREFACTOR) ** (-1.0 / 1.8)   # 1/m
    n_per_m3 = 0.22 * lamb ** 1.2                              # M0 (Abel-Boutle)
    return n_per_m3 / rho


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--rh", type=float, default=0.80)
    p.add_argument("--pressure-hpa", type=float, default=850.0)
    p.add_argument("--temp-c", type=float, default=17.0, help="fixed temperature [degC]")
    p.add_argument("--qr-min-g-kg", type=float, default=0.01)
    p.add_argument("--qr-max-g-kg", type=float, default=5.0)
    p.add_argument("--nr-min", type=float, default=10.0, help="min Nr [#/kg]")
    p.add_argument("--nr-max", type=float, default=1.0e5, help="max Nr [#/kg]")
    p.add_argument("--ngrid", type=int, default=240)
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
    t = args.temp_c + 273.15
    rho = p_pa / (RD * t)
    to_rate = 1000.0 * 3600.0          # kg/kg/s -> g/kg/h

    # qr-Nr grid (log-log)
    qr_g = np.logspace(np.log10(args.qr_min_g_kg), np.log10(args.qr_max_g_kg), args.ngrid)
    nr = np.logspace(np.log10(args.nr_min), np.log10(args.nr_max), args.ngrid)
    QRG, NR = np.meshgrid(qr_g, nr)
    QR = QRG * 1.0e-3                   # kg/kg

    DMEAN = (6.0 * QR / (RHO_W * np.pi * NR)) ** (1.0 / 3.0)
    MU = mu_grid(DMEAN)
    evap2 = evap_2moment(QR, NR, MU, t, p_pa, rh) * to_rate

    # restrict to physically meaningful drop sizes
    physical = (DMEAN >= 1.0e-4) & (DMEAN <= 6.0e-3)
    evap2 = np.where(physical, evap2, np.nan)

    vmin = np.nanpercentile(evap2, 1.0)
    vmax = np.nanpercentile(evap2, 99.0)
    norm = LogNorm(vmin=max(vmin, 1.0e-3), vmax=vmax)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    mesh = ax.pcolormesh(qr_g, nr, evap2, norm=norm, cmap=cmap, shading="auto")

    # constant mean-volume-diameter reference lines (Nr = 6 qr/(1000 pi D^3))
    for d_mm in (0.5, 1.0, 2.0, 4.0):
        d_m = d_mm * 1.0e-3
        n_line = 6.0 * (qr_g * 1.0e-3) / (RHO_W * np.pi * d_m ** 3)
        inside = (n_line >= nr[0]) & (n_line <= nr[-1])
        ax.plot(qr_g, n_line, color="white", ls=":", lw=1.1, alpha=0.6)
        if inside.any():
            i = np.where(inside)[0][len(np.where(inside)[0]) // 2]
            ax.annotate(f"$D_m$={d_mm:g} mm", xy=(qr_g[i], n_line[i]),
                        fontsize=7.5, color="white", ha="center", va="center",
                        rotation=32,
                        bbox=dict(boxstyle="round,pad=0.1", fc="0.25", ec="none", alpha=0.55))

    # one-moment AB12 locus, coloured by its own (G1M) evaporation rate
    n_ab = ab12_number_per_kg(qr_g * 1.0e-3, rho)
    evap1 = evap_1moment(qr_g * 1.0e-3, t, p_pa, rh, rho_weighted=False) * to_rate
    pts = np.array([qr_g, n_ab]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.plot(qr_g, n_ab, color="white", lw=6.0, zorder=4)        # white casing
    lc = LineCollection(segs, cmap=cmap, norm=norm, zorder=5)
    lc.set_array(evap1[:-1])
    lc.set_linewidth(3.4)
    ax.add_collection(lc)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(qr_g[0], qr_g[-1])
    ax.set_ylim(nr[0], nr[-1])
    ax.set_xlabel("Rain specific content  $q_r$  (g kg$^{-1}$)")
    ax.set_ylabel("Rain number concentration  $N_r$  (kg$^{-1}$)")
    ax.set_title("Two-moment rain evaporation across the $q_r$ and $N_r$ plane", pad=12)

    proxy = plt.Line2D([], [], color="k", lw=3.0, label="1-moment (G1M) AB12 locus")
    ax.legend(handles=[proxy], loc="upper right", fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Rain evaporation rate  (g kg$^{-1}$ h$^{-1}$)")
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    fig.tight_layout()
    fig.text(0.5, -0.01,
             f"RH = {rh*100:g} %    $p$ = {args.pressure_hpa:g} hPa    "
             f"T = {args.temp_c:g} $^\\circ$C    (dotted: constant mean-volume diameter)",
             ha="center", va="top", fontsize=8.5, color="0.3")

    out_dir = args.plot_root / "15. rain evaporation qN plane"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "15. rain evaporation qN plane_450dpi.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    print(f"rendered {png}")

    # data export: the AB12 locus (the 2-moment field is the rendered image)
    txt = out_dir / "15. rain evaporation qN plane_data.txt"
    header = (
        "# 1-moment AB12 locus: rain number and evaporation rate vs rain content.\n"
        f"# RH={rh*100:g}%  p={args.pressure_hpa:g}hPa  T={args.temp_c:g}C\n"
        "qr_g_kg\tNr_ab12_per_kg\tevap1m_g_kg_h"
    )
    np.savetxt(txt, np.column_stack([qr_g, n_ab, evap1]),
               header=header, comments="", fmt="%.6g", delimiter="\t")
    print(f"wrote   {txt}")


if __name__ == "__main__":
    main()
