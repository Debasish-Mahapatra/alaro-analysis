#!/usr/bin/env python3
"""Offline rain evaporation rate of the one- and two-moment schemes, the paper's figure.

The two panels of the older paper figures 13 and 14 (``paper1.make_paper_rain_evaporation_qv``
and ``_nr``) in one figure, from the same formulas (``analysis.rain_evaporation``: ACEVMEL,
Lopez block ``LA0MPS .AND. LEVAPLOP``). Nothing is read from model output, so the figure does
not depend on the runs, only on the scheme of each configuration:

* C1M: one-moment rain, LGRAPRO=F, ventilation argument rho*qr (acevmel.F90:645);
* G1M: one-moment rain, LGRAPRO=T, ventilation argument qr (:522);
* G2M: two-moment rain (LTWOMOMLIQ=T) with the Milbrandt (2005) shape; G2M-XCU shares it.

Relative humidity and pressure are held fixed and the temperature is swept, so qv = RH qsat
runs along the x-axis (a top axis gives the temperature).

* (a) the three schemes at rain contents of 0.1, 0.5 and 1 g/kg; G2M with a mean-volume
  diameter of 1 mm, which sets its rain number and shape;
* (b) G2M at 1 g/kg for a family of rain numbers Nr (per kg of air, as the model carries
  it), against the one-moment rates, which have no Nr.

Output: ``<NN>_rain_evaporation/rain_evaporation_offline_450dpi.jpg`` with the curves in
``data/rain_evaporation_offline.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.rain_evaporation import (
    COLOR_C1M,
    COLOR_G1M,
    COLOR_G2M,
    RHO_W,
    evap_1moment,
    evap_2moment,
    mu_from_dmean,
    nr_from_qr_dmean,
    qsat_w,
)
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import FIGURES, figure_files, write_figure_index

FIGURE_NAME = "rain_evaporation"
PERIOD = "offline"
TO_RATE = 1000.0 * 3600.0  # kg/kg/s -> g/kg/h
TICK, AXIS = 15, 17


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rh", type=float, default=0.80, help="relative humidity (fraction)")
    p.add_argument("--pressure-hpa", type=float, default=850.0)
    p.add_argument("--qr-g-kg", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                   help="rain contents of panel (a) [g/kg]")
    p.add_argument("--dmean-mm", type=float, default=1.0,
                   help="G2M mean-volume diameter of panel (a), which sets Nr and mu [mm]")
    p.add_argument("--fixed-qr-g-kg", type=float, default=1.0, help="rain content of panel (b) [g/kg]")
    p.add_argument("--nr-per-kg", type=float, nargs="+", default=[250.0, 1000.0, 4000.0, 16000.0],
                   help="G2M rain numbers of panel (b) [#/kg]")
    p.add_argument("--tmin-c", type=float, default=1.0)
    p.add_argument("--tmax-c", type=float, default=33.0)
    p.add_argument("--npts", type=int, default=201)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_rain_evaporation", argv=argv)


def one_moment(qr: float, t_grid: np.ndarray, p_pa: float, rh: float) -> dict[str, np.ndarray]:
    return {
        "C1M": np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=True) for t in t_grid]) * TO_RATE,
        "G1M": np.array([evap_1moment(qr, t, p_pa, rh, rho_weighted=False) for t in t_grid]) * TO_RATE,
    }


def two_moment(qr: float, nr: float, t_grid: np.ndarray, p_pa: float, rh: float) -> np.ndarray:
    mu = mu_from_dmean((6.0 * qr / (RHO_W * np.pi * nr)) ** (1.0 / 3.0))
    return np.array([evap_2moment(qr, nr, mu, t, p_pa, rh) for t in t_grid]) * TO_RATE


def temperature_axis(ax, qv_g: np.ndarray, t_c: np.ndarray, rh: float) -> None:
    top = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, qv_g, t_c),
                                               lambda x: np.interp(x, t_c, qv_g)))
    # ticks only where the sweep has values (np.interp clamps outside it)
    top.set_xticks([t for t in range(0, 45, 5) if t_c[0] <= t <= t_c[-1]])
    top.set_xlabel(f"Temperature at RH = {rh * 100:g} % ($^\\circ$C)", fontsize=AXIS)
    top.tick_params(labelsize=TICK)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rh, p_pa = args.rh, args.pressure_hpa * 100.0
    t_grid = np.linspace(args.tmin_c + 273.15, args.tmax_c + 273.15, args.npts)
    qv_g = args.rh * np.array([qsat_w(t, p_pa) for t in t_grid]) * 1.0e3
    t_c = t_grid - 273.15
    columns = {"qv_g_kg": qv_g, "T_degC": t_c}

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0), sharey=False)
    # (a) the three schemes at several rain contents; thin = little rain, thick = much
    ax = axes[0]
    qr_list = sorted(args.qr_g_kg)
    dmean = args.dmean_mm * 1.0e-3
    mu = mu_from_dmean(dmean)
    for qr_g, width in zip(qr_list, np.linspace(1.4, 3.2, len(qr_list))):
        qr = qr_g * 1.0e-3
        rates = one_moment(qr, t_grid, p_pa, rh)
        nr = nr_from_qr_dmean(qr, dmean)
        rates["G2M"] = np.array([evap_2moment(qr, nr, mu, t, p_pa, rh) for t in t_grid]) * TO_RATE
        for name, colour in (("C1M", COLOR_C1M), ("G1M", COLOR_G1M), ("G2M", COLOR_G2M)):
            ax.plot(qv_g, rates[name], color=colour, lw=width)
            columns[f"a_{name}_qr{qr_g:g}_g_kg_h"] = rates[name]
        ax.annotate(f"$q_r$ = {qr_g:g} g kg$^{{-1}}$", xy=(qv_g[-1], max(r[-1] for r in rates.values())),
                    xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=13,
                    color="0.25")
    ax.set_xlim(qv_g[0], qv_g[-1] * 1.22)

    # (b) G2M at one rain content for several rain numbers; light = few large drops
    ax = axes[1]
    qr = args.fixed_qr_g_kg * 1.0e-3
    rates = one_moment(qr, t_grid, p_pa, rh)
    ax.plot(qv_g, rates["C1M"], color=COLOR_C1M, lw=2.4)
    ax.plot(qv_g, rates["G1M"], color=COLOR_G1M, lw=2.4)
    columns["b_C1M_g_kg_h"], columns["b_G1M_g_kg_h"] = rates["C1M"], rates["G1M"]
    nr_handles = []
    nr_list = sorted(args.nr_per_kg)
    for nr, shade in zip(nr_list, plt.cm.Greens(np.linspace(0.40, 0.95, len(nr_list)))):
        rate = two_moment(qr, nr, t_grid, p_pa, rh)
        d_mean = (6.0 * qr / (RHO_W * np.pi * nr)) ** (1.0 / 3.0)
        ax.plot(qv_g, rate, color=shade, lw=2.4)
        nr_handles.append(plt.Line2D([], [], color=shade, lw=2.4,
                                     label=f"G2M, $N_r$ = {nr:g} kg$^{{-1}}$ ($D$ = {d_mean * 1e3:.2f} mm)"))
        columns[f"b_G2M_Nr{nr:g}_g_kg_h"] = rate
    ax.set_xlim(qv_g[0], qv_g[-1])

    for ax, letter in zip(axes, "ab"):
        ax.set_xticks([q for q in range(5, 45, 5) if q <= qv_g[-1]])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Water-vapour mass fraction $q_v$ (g kg$^{-1}$)", fontsize=AXIS)
        ax.tick_params(labelsize=TICK)
        ax.grid(True, ls=":", alpha=0.5)
        ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=18,
                fontweight="bold")
        temperature_axis(ax, qv_g, t_c, rh)
    axes[0].set_ylabel("Rain evaporation rate (g kg$^{-1}$ h$^{-1}$)", fontsize=AXIS)

    handles = [plt.Line2D([], [], color=COLOR_C1M, lw=2.6, label="C1M (one-moment)"),
               plt.Line2D([], [], color=COLOR_G1M, lw=2.6, label="G1M (one-moment)"),
               plt.Line2D([], [], color=COLOR_G2M, lw=2.6, label=f"G2M (two-moment), $D$ = {args.dmean_mm:g} mm")]
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=0.18)
    legend_below(fig, axes, handles + nr_handles, ncol=2, fontsize=LEGEND_FONTSIZE)

    png, jpg, txt = figure_files(args.figure_root, FIGURE_NAME, FIGURE_NAME, PERIOD)
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    convert_to_jpg(png, jpg, dpi=args.dpi)
    png.unlink()

    keys = list(columns)
    title = f"{FIGURE_NAME}_{PERIOD}"
    header = "\n".join([
        title, "=" * len(title),
        "Offline rain evaporation rate (g/kg/h) from the ACEVMEL formulas (analysis.rain_evaporation);",
        "nothing is read from model output. Water-vapour content as specific content (mass fraction).",
        f"RH = {rh * 100:g} %, p = {args.pressure_hpa:g} hPa; qv = RH qsat(T, p), T from {args.tmin_c:g} to "
        f"{args.tmax_c:g} degC.",
        "C1M: one-moment, LGRAPRO=F, ventilation argument rho*qr (acevmel.F90:645). G1M: one-moment,",
        "LGRAPRO=T, argument qr (:522). G2M: two-moment, LTWOMOMLIQ=T (:517), Milbrandt (2005) shape.",
        f"(a) qr = {', '.join(f'{q:g}' for q in qr_list)} g/kg; G2M with mean-volume diameter "
        f"{args.dmean_mm:g} mm (mu = {mu:.3f}), Nr from qr and that diameter.",
        f"(b) qr = {args.fixed_qr_g_kg:g} g/kg; G2M for Nr = {', '.join(f'{n:g}' for n in nr_list)} per kg "
        "of air, mean-volume diameter and mu from qr and Nr.",
        "",
        "\t".join(keys),
    ])
    np.savetxt(txt, np.column_stack([columns[k] for k in keys]), header=header, comments="", fmt="%.6g",
               delimiter="\t")
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.figure_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
