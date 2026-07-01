#!/usr/bin/env python3
"""Plot 6: hydrometeor mixing-ratio CFADs (frequency by altitude).

Reproduces the original 5-row x 3-column CFAD figure from the histogram caches
built by build_hydrometeor_cfad_from_netcdf.py, with two changes only:
  * rows reordered to graupel -> snow -> cloud ice -> cloud water -> rain
  * row labels are symbols (q_g, q_s, q_i, q_c, q_r) instead of full names.

Per height level the mixing-ratio distribution is normalised to 100 % (a CFAD).
Columns: absolute frequency (log, WhiteBlueGreenYellowRed) and frequency
differences (linear +/-4 %, RdBu_r); graupel has no C1M (2-ice control) so its
first column is "no data" and the absolute panel shows G1M.  A black median line
(per-level median mixing ratio) is drawn on every panel.
"""
from __future__ import annotations

from pathlib import Path

import cmaps
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.vertical import centers_to_edges

CACHE = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/paper6_hydrometeor_cfad")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/6. hydrometeor mixing-ratio cfad")
FIGURE_NAME = "6. hydrometeor mixing-ratio cfad_450dpi.png"
MAX_HEIGHT_KM = 20.0
FREQ_VMIN, FREQ_VMAX = 1e-2, 30.0   # log frequency colour scale (%)
DIFF_SCALE = 4.0                    # +/- % for difference panels

# Rows: (symbol, counts-key) top -> bottom
ROWS = [
    (r"$q_\mathrm{g}$", "GRAUPEL"),
    (r"$q_\mathrm{s}$", "SNOW"),
    (r"$q_\mathrm{i}$", "SOLID_WATER"),
    (r"$q_\mathrm{c}$", "LIQUID_WATER"),
    (r"$q_\mathrm{r}$", "RAIN"),
]
TICK_FS, AXIS_FS, TITLE_FS, CBAR_FS, SYM_FS, LEG_FS = 12, 13, 15, 12, 26, 10


def load_exp(exp: str) -> dict:
    return dict(np.load(CACHE / f"{exp}_cfad.npz"))


def freq_and_median(counts: np.ndarray, log_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-level frequency (%) summing to 100, and per-level median mixing ratio."""
    counts = counts.astype(np.float64)
    tot = counts.sum(axis=1, keepdims=True)
    freq = np.where(tot > 0, 100.0 * counts / np.maximum(tot, 1.0), np.nan)
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    cum = np.cumsum(counts, axis=1)
    med = np.full(counts.shape[0], np.nan)
    for lev in range(counts.shape[0]):
        if cum[lev, -1] <= 0:
            continue
        frac = cum[lev] / cum[lev, -1]
        med[lev] = centers[min(int(np.searchsorted(frac, 0.5)), centers.size - 1)]
    return freq, med


def main() -> None:
    data = {e: load_exp(e) for e in ("control", "graupel", "2mom")}
    log_edges = data["control"]["log_edges"]
    q_edges = 10.0 ** log_edges

    # Common vertical grid from control geopotential; sort ascending, crop to 20 km.
    h = np.asarray(data["control"]["height_km"], dtype=np.float64)
    order = np.argsort(h)
    h_sorted = h[order]
    keep = np.isfinite(h_sorted) & (h_sorted >= 0.0) & (h_sorted <= MAX_HEIGHT_KM)
    y = h_sorted[keep]
    y_edges = centers_to_edges(y)

    # Pre-compute freq + median for every (experiment, species) present.
    freq, med = {}, {}
    for e in ("control", "graupel", "2mom"):
        for key in (f"counts_{r[1]}" for r in ROWS):
            if key in data[e]:
                sp = key.replace("counts_", "")
                f, m = freq_and_median(data[e][key], log_edges)
                freq[(e, sp)] = f
                med[(e, sp)] = m

    def crop(arr1d_or_2d):
        return arr1d_or_2d[order][keep]

    freq_cmap = cmaps.WhiteBlueGreenYellowRed
    freq_norm = mcolors.LogNorm(vmin=FREQ_VMIN, vmax=FREQ_VMAX)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-DIFF_SCALE, vcenter=0.0, vmax=DIFF_SCALE)

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(15.5, 22.0), constrained_layout=True)

    for r, (sym, sp) in enumerate(ROWS):
        # Column plan: (kind, payload, title, median-experiment)
        if sp == "GRAUPEL":
            plan = [
                ("nodata", None, "C1M", None),
                ("abs", ("graupel", sp), "G1M", "graupel"),
                ("diff", ("2mom", "graupel", sp), "G2M − G1M", "2mom"),
            ]
        else:
            plan = [
                ("abs", ("control", sp), "C1M", "control"),
                ("diff", ("graupel", "control", sp), "G1M − C1M", "graupel"),
                ("diff", ("2mom", "graupel", sp), "G2M − G1M", "2mom"),
            ]

        for c, (kind, payload, title, med_exp) in enumerate(plan):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xlim(q_edges[0], q_edges[-1])
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            ax.set_xlabel(r"Mixing ratio (kg kg$^{-1}$)", fontsize=AXIS_FS)
            ax.set_ylabel("Height (km)", fontsize=AXIS_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)

            if kind == "nodata":
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=15, color="0.5")
                continue

            if kind == "abs":
                e, s = payload
                field = crop(np.ma.masked_invalid(freq[(e, s)]))
                pcm = ax.pcolormesh(q_edges, y_edges, field, cmap=freq_cmap,
                                    norm=freq_norm, shading="flat")
                cb = fig.colorbar(pcm, ax=ax)
                cb.set_label("Frequency (%)", fontsize=CBAR_FS)
            else:
                a, b, s = payload
                diff = crop(np.ma.masked_invalid(freq[(a, s)] - freq[(b, s)]))
                pcm = ax.pcolormesh(q_edges, y_edges, diff, cmap="RdBu_r",
                                    norm=diff_norm, shading="flat")
                cb = fig.colorbar(pcm, ax=ax)
                cb.set_label(r"$\Delta$ Frequency (%)", fontsize=CBAR_FS)
            cb.ax.tick_params(labelsize=TICK_FS)

            m = crop(med[(med_exp, sp)])
            ax.plot(m, y, color="black", lw=2.0, label="Median", zorder=6)
            ax.legend(loc="upper right", fontsize=LEG_FS, framealpha=0.9)

    # Row symbols on the far left (placed after the layout is resolved).
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for r, (sym, _sp) in enumerate(ROWS):
        pos = axes[r, 0].get_position()
        fig.text(pos.x0 - 0.045, 0.5 * (pos.y0 + pos.y1), sym, rotation=90,
                 va="center", ha="center", fontsize=SYM_FS, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(out, dpi=450, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
