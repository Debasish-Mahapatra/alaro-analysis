#!/usr/bin/env python3
"""Plot 6 (per-model layout): hydrometeor MASS-FRACTION CFADs.

Contoured Frequency by Altitude Diagrams of the hydrometeor mass fractions, one
row per species and one column per model run (C1M, G1M, G2M). Each panel shows
the per-height-level frequency (%) distribution of the mass fraction (a CFAD:
every height row sums to 100 %), with the per-level median mass fraction
overplotted.

Two corrections relative to the older "mixing-ratio" figure:
  1. ALARO stores hydrometeor content as MASS FRACTION (NetCDF units "1",
     long_name "Atmospheric <species>"), i.e. kg of hydrometeor per kg of total
     moist air -- NOT a mixing ratio (per kg dry air). Axes/labels say so.
  2. Species ordered graupel -> snow -> cloud ice -> cloud water -> rain.

Reads the per-species histogram caches produced (from the masked-netcdf) by
build_hydrometeor_cfad_from_netcdf.py. C1M is the 2-ice control: graupel is not
a prognostic species there, so its C1M panel is "no data".
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
FIGURE_NAME = "6. hydrometeor mass-fraction cfad_per-model_450dpi.png"
TEXT_NAME = "6. hydrometeor mass-fraction cfad_per-model_data.txt"

MAX_HEIGHT_KM = 20.0
FREQ_VMIN, FREQ_VMAX = 1e-2, 30.0   # log frequency colour scale (%); fixed top
                                    # (as in the established Fig 6) so the 1-30 %
                                    # bulk keeps contrast and sparse single-bin
                                    # levels just saturate instead of dominating.
X_MIN, X_MAX = 1e-9, 1e-2  # mass-fraction x-axis window (kg/kg)

# Rows top -> bottom: (symbol, counts-species, full name)
ROWS = [
    (r"$q_\mathrm{g}$", "GRAUPEL", "Graupel"),
    (r"$q_\mathrm{s}$", "SNOW", "Snow"),
    (r"$q_\mathrm{i}$", "SOLID_WATER", "Cloud ice"),
    (r"$q_\mathrm{c}$", "LIQUID_WATER", "Cloud water"),
    (r"$q_\mathrm{r}$", "RAIN", "Rain"),
]
# Columns left -> right: (label, experiment cache key)
COLS = [("C1M", "control"), ("G1M", "graupel"), ("G2M", "2mom")]

TICK_FS, AXIS_FS, TITLE_FS, CBAR_FS, ROW_FS, LEG_FS = 11, 12, 14, 10, 14, 9


def load_exp(exp: str) -> dict:
    return dict(np.load(CACHE / f"{exp}_cfad.npz"))


def freq_and_median(counts: np.ndarray, log_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-level frequency (%) summing to 100, and per-level median mass fraction."""
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


def sorted_height_grid(height_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ascending heights cropped to MAX_HEIGHT_KM, y-edges, index order+keep)."""
    h = np.asarray(height_km, dtype=np.float64)
    order = np.argsort(h)
    h_sorted = h[order]
    keep = np.isfinite(h_sorted) & (h_sorted >= 0.0) & (h_sorted <= MAX_HEIGHT_KM)
    y = h_sorted[keep]
    return y, centers_to_edges(y), (order, keep)


def main() -> None:
    data = {exp: load_exp(exp) for _, exp in COLS}
    log_edges = data["control"]["log_edges"]
    q_edges = 10.0 ** log_edges

    # Per-experiment vertical grid (each cache carries its own geopotential heights).
    grids = {exp: sorted_height_grid(data[exp]["height_km"]) for _, exp in COLS}

    # Pre-compute freq + median for every present (species, experiment).
    freq, med = {}, {}
    for _, exp in COLS:
        for _, sp, _ in ROWS:
            key = f"counts_{sp}"
            if key in data[exp]:
                f, m = freq_and_median(data[exp][key], log_edges)
                freq[(exp, sp)] = f
                med[(exp, sp)] = m

    freq_cmap = cmaps.WhiteBlueGreenYellowRed
    freq_norm = mcolors.LogNorm(vmin=FREQ_VMIN, vmax=FREQ_VMAX)

    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(13.5, 19.5), constrained_layout=True)

    for c, (col_label, exp) in enumerate(COLS):
        axes[0][c].set_title(col_label, fontsize=TITLE_FS, fontweight="bold")
        y, y_edges, (order, keep) = grids[exp]
        for r, (sym, sp, _name) in enumerate(ROWS):
            ax = axes[r][c]
            ax.set_xscale("log")
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"Mass fraction (kg kg$^{-1}$)", fontsize=AXIS_FS)
            if c == 0:
                ax.set_ylabel("Height (km)", fontsize=AXIS_FS)

            if (exp, sp) not in freq:  # graupel has no C1M (2-ice control)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=13, color="0.5")
                continue

            field = np.ma.masked_invalid(freq[(exp, sp)][order][keep])
            pcm = ax.pcolormesh(q_edges, y_edges, field, cmap=freq_cmap,
                                norm=freq_norm, shading="flat")
            cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("Frequency (%)", fontsize=CBAR_FS)
            cb.ax.tick_params(labelsize=TICK_FS)

            m = med[(exp, sp)][order][keep]
            good = np.isfinite(m) & (m > 0)
            if good.any():
                ax.plot(m[good], y[good], color="black", lw=2.0, label="Median", zorder=6)
                ax.legend(loc="upper right", fontsize=LEG_FS, framealpha=0.9)

    # Species symbol + name on the far-left of each row.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for r, (sym, _sp, name) in enumerate(ROWS):
        pos = axes[r][0].get_position()
        fig.text(pos.x0 - 0.055, 0.5 * (pos.y0 + pos.y1), f"{sym}\n{name}", rotation=90,
                 va="center", ha="center", fontsize=ROW_FS, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(out, dpi=450, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # Provenance .txt (microphysics-paper convention: png + data txt together).
    txt = OUTPUT_DIR / TEXT_NAME
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("Hydrometeor mass-fraction CFADs (per-model layout)\n")
        fh.write("=================================================\n")
        fh.write(f"Figure: {out}\n\n")
        fh.write("Method\n------\n")
        fh.write("Per (experiment, species): 2-D histogram counts[level, log10(mass fraction)]\n")
        fh.write("accumulated over every masked-netcdf grid point and hourly file, then each\n")
        fh.write("height level normalised to 100 % (CFAD). Median = per-level median of the\n")
        fh.write("positive mass fraction read off the cumulative distribution.\n")
        fh.write("Source field: ALARO hydrometeor MASS FRACTION (NetCDF units '1', kg/kg of\n")
        fh.write("total moist air); NOT a mixing ratio. Heights from GEOPOTENTIEL.\n")
        fh.write(f"Mass-fraction bins: {q_edges[0]:.0e}..{q_edges[-1]:.0e} kg/kg, "
                 f"{len(q_edges) - 1} log bins. Height shown 0..{MAX_HEIGHT_KM:g} km.\n")
        fh.write(f"Frequency colour scale (log): {FREQ_VMIN:g}..{FREQ_VMAX:g} % "
                 "(fixed top; sparse single-bin levels saturate).\n\n")
        fh.write("Rows (top->bottom): graupel, snow, cloud ice, cloud water, rain.\n")
        fh.write("Columns: C1M (control), G1M (graupel), G2M (2mom).\n")
        fh.write("C1M graupel panel = no data (2-ice control has no prognostic graupel).\n\n")
        fh.write("Sources / sample sizes\n----------------------\n")
        for _, exp in COLS:
            n = int(data[exp]["n_files"])
            present = [sp for _, sp, _ in ROWS if f"counts_{sp}" in data[exp]]
            fh.write(f"{exp}: n_files={n}, species={present}\n")
    print(f"[saved] {txt}")


if __name__ == "__main__":
    main()
