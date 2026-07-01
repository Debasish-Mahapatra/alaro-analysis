"""Combined 2x2 NIMELIT closure-budget figure: the NIMELIT=2 (baseline) and NIMELIT=1
figures merged into one panel grid (rows = NIMELIT setting, cols = C1M / G1M).

Reuses all the per-panel logic + data loaders from make_sup_closure_budget.py; only the
layout is new (2x2 instead of two separate 1x2 figures), with a bold bottom legend.

Output -> microphysics-paper/SUP/closure_budget_compression_NIMELIT/closure_budget_compression_NIMELIT_2x2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import epygram

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sup_closure_budget as C   # reuse gather(), colors, constants, _base/_pert


# 4 panels: (spec, col title, row label, panel letter); index -> (row=i//2, col=i%2)
SPECS = [
    (C._base("control", "C1M"),             "C1M", "NIMELIT = 2", "(a)"),
    (C._base("graupel", "G1M"),             "G1M", "NIMELIT = 2", "(b)"),
    (C._pert("control_NIMELIT_1", "C1M"),   "C1M", "NIMELIT = 1", "(c)"),
    (C._pert("graupel_NIMELIT_1", "G1M"),   "G1M", "NIMELIT = 1", "(d)"),
]

OUT_DIR = C.SUP_ROOT / "closure_budget_compression_NIMELIT"
OUT_PNG = OUT_DIR / "closure_budget_compression_NIMELIT_2x2.png"


def draw_panel(ax, d, *, col, row, col_title):
    ax.axhspan(0, d["z_peak"], color="#BFC7D5", alpha=0.18, zorder=0)
    ax.plot(d["total"], d["altitude_km"], color=C.TOTAL_COLOR, lw=2.8)
    ax.plot(d["convective"], d["altitude_km"], color=C.CONVECTIVE_COLOR, lw=2.0, ls="-.", alpha=0.9)
    ax.plot(d["resolved"], d["altitude_km"], color=C.RESOLVED_COLOR, lw=2.0, ls="--", alpha=0.95)
    ax.axhline(d["z_peak"], color=C.PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), alpha=0.9)
    ax.axhline(d["z_freeze"], color=C.FREEZING_COLOR, lw=1.2, ls=":", alpha=0.95)

    ax2 = ax.twiny()
    ax2.plot(d["u_flux"], d["u_h"], color=C.UPDRAFT_COLOR, lw=2.1, alpha=0.95)
    ax2.set_xlim(0, C.FLUX_XMAX)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.tick_params(axis="x", colors=C.UPDRAFT_COLOR)
    ax2.spines["top"].set_color(C.UPDRAFT_COLOR)
    if row == 0:
        ax2.set_xlabel(r"updraft mass flux (kg m$^{-2}$ s$^{-1}$)", color=C.UPDRAFT_COLOR, labelpad=8)
    else:
        ax2.tick_params(axis="x", labelcolor=C.UPDRAFT_COLOR)  # ticks only, no label

    ax.set_xlim(0, C.COND_XMAX)
    ax.set_ylim(0, C.Z_MAX)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if row == 0:
        ax.set_title(col_title, pad=34, fontweight="bold")
    if row == 1:
        ax.set_xlabel(r"condensation (g kg$^{-1}$ day$^{-1}$)")
    if col == 0:
        ax.set_ylabel("altitude (km)")
    else:
        ax.tick_params(labelleft=False)
    ax.grid(alpha=0.25)


def main():
    epygram.init_env()
    data = [C.gather(spec) for spec, *_ in SPECS]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17})
    fig = plt.figure(figsize=(11.0, 12.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 3.0, 0.30],
                          hspace=0.42, wspace=0.14,
                          left=0.11, right=0.955, top=0.93, bottom=0.055)

    axes = {}
    for i, (spec, col_title, row_label, letter) in enumerate(SPECS):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        axes[(row, col)] = ax
        draw_panel(ax, data[i], col=col, row=row, col_title=col_title)
        ax.text(0.97, 0.96, letter, transform=ax.transAxes, ha="right", va="top",
                fontsize=15, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          boxstyle="round,pad=0.18"), zorder=20)

    # bold row labels on the left margin (NIMELIT = 2 / = 1)
    for row, label in ((0, "NIMELIT = 2"), (1, "NIMELIT = 1")):
        pos = axes[(row, 0)].get_position()
        fig.text(0.028, 0.5 * (pos.y0 + pos.y1), label, rotation=90,
                 va="center", ha="center", fontsize=16, fontweight="bold")

    style_handles = [
        Line2D([0], [0], color=C.TOTAL_COLOR, lw=2.8, label="total condensation"),
        Line2D([0], [0], color=C.CONVECTIVE_COLOR, lw=2.0, ls="-.", label="convection-scheme part"),
        Line2D([0], [0], color=C.RESOLVED_COLOR, lw=2.0, ls="--", label="resolved-microphysics part"),
        Line2D([0], [0], color=C.UPDRAFT_COLOR, lw=2.1, label="updraft mass flux"),
        Line2D([0], [0], color=C.PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), label="condensation peak height"),
        Line2D([0], [0], color=C.FREEZING_COLOR, lw=1.2, ls=":", label=r"0 $^{\circ}$C isotherm"),
    ]
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=style_handles, loc="center", ncol=3, frameon=False,
                  handlelength=2.6, columnspacing=1.8,
                  prop=FontProperties(weight="bold", size=13.5))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
