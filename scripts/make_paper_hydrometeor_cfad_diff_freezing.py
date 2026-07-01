#!/usr/bin/env python3
"""Hydrometeor mass-fraction CFADs: C1M, G1M-C1M, G2M-G1M, with freezing levels.

Columns: C1M (absolute frequency), G1M - C1M (difference), G2M - G1M (difference).
Rows (top -> bottom): graupel, snow, cloud ice, cloud water, rain.

Absolute panels use the WhiteBlueGreenYellowRed log colour scale; difference
panels use a diverging RdBu_r scale (+/-4 %).  A per-level median mass-fraction
line is drawn on the absolute panels.  Each column carries the 0 degC (freezing)
level of its model -- C1M, G1M, G2M respectively -- as a horizontal dashed line
labelled with its height in km, computed from the 2-year mean temperature
profile (same caches the diurnal plots use).

Graupel is not a prognostic species in the 2-ice control, so its C1M panel is
"no data" and its middle column shows the G1M absolute CFAD instead of C1M-G1M.

CFAD counts come from the (from-raw) histogram caches built by
build_hydrometeor_cfad_from_netcdf.py; freezing levels from the 2-year
temperature/geopotential profile caches.
"""
from __future__ import annotations

from pathlib import Path

import cmaps
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from alaro_analysis.common.vertical import centers_to_edges

CACHE = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/paper6_hydrometeor_cfad")
TEMP = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/temperature/2years")
GEO = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/geopotential/2years")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/6. hydrometeor mixing-ratio cfad")
FIGURE_NAME = "6. hydrometeor mass-fraction cfad_c1m_g1m-c1m_g2m-g1m_freezing_450dpi.png"
TEXT_NAME = "6. hydrometeor mass-fraction cfad_c1m_g1m-c1m_g2m-g1m_freezing_data.txt"

MAX_HEIGHT_KM = 20.0
FREQ_VMIN, FREQ_VMAX = 1e-2, 30.0   # absolute log frequency colour scale (%)
DIFF_SCALE = 4.0                    # +/- % for difference panels
FREEZING_K = 273.15

# Rows: (symbol, counts-species, full name) top -> bottom
ROWS = [
    (r"$q_\mathrm{g}$", "GRAUPEL", "Graupel"),
    (r"$q_\mathrm{s}$", "SNOW", "Snow"),
    (r"$q_\mathrm{i}$", "SOLID_WATER", "Cloud ice"),
    (r"$q_\mathrm{c}$", "LIQUID_WATER", "Cloud water"),
    (r"$q_\mathrm{r}$", "RAIN", "Rain"),
]
EXP_LABEL = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
# Column -> model whose freezing (0 degC) level is drawn on it: C1M, G1M, G2M.
COL_FREEZE_EXP = ("control", "graupel", "2mom")

# Fonts (all enlarged per request).
TICK_FS, AXIS_FS, TITLE_FS = 17, 20, 23
CBAR_FS, CBAR_TICK_FS = 18, 15
SYM_FS, NAME_FS, LEG_FS = 34, 19, 15
FREEZE_COLOR = "black"


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


def freezing_level_km(exp: str) -> float:
    """Height (km) of the 0 degC level from the 2-year mean temperature profile."""
    t = dict(np.load(TEMP / f"{exp}_full-domain_diurnal_profile.npz", allow_pickle=True))
    g = dict(np.load(GEO / f"{exp}_full-domain_height_profile_first.npz", allow_pickle=True))
    mean = np.asarray(t["mean"], dtype=np.float64)       # (L, 24) Kelvin
    cnt = np.asarray(t["counts"], dtype=np.float64)      # (L, 24)
    w = cnt.sum(axis=1)
    tprof = np.where(w > 0, (mean * cnt).sum(axis=1) / np.maximum(w, 1.0), np.nan)
    h_km = np.asarray(g["height_m"], dtype=np.float64) / 1000.0
    order = np.argsort(h_km)
    z, tt = h_km[order], tprof[order]
    fin = np.isfinite(z) & np.isfinite(tt)
    z, tt = z[fin], tt[fin]
    # First crossing of 273.15 K scanning upward from the surface.
    for i in range(z.size - 1):
        a, b = tt[i] - FREEZING_K, tt[i + 1] - FREEZING_K
        if a == 0.0:
            return float(z[i])
        if a * b < 0.0:
            return float(z[i] + (z[i + 1] - z[i]) * (FREEZING_K - tt[i]) / (tt[i + 1] - tt[i]))
    return float("nan")


def main() -> None:
    data = {e: load_exp(e) for e in ("control", "graupel", "2mom")}
    log_edges = data["control"]["log_edges"]
    q_edges = 10.0 ** log_edges

    # Common vertical grid from control geopotential; ascending, cropped to 20 km.
    h = np.asarray(data["control"]["height_km"], dtype=np.float64)
    order = np.argsort(h)
    h_sorted = h[order]
    keep = np.isfinite(h_sorted) & (h_sorted >= 0.0) & (h_sorted <= MAX_HEIGHT_KM)
    y = h_sorted[keep]
    y_edges = centers_to_edges(y)

    freq, med = {}, {}
    for e in ("control", "graupel", "2mom"):
        for _, sp, _ in ROWS:
            key = f"counts_{sp}"
            if key in data[e]:
                f, m = freq_and_median(data[e][key], log_edges)
                freq[(e, sp)] = f
                med[(e, sp)] = m

    frz = {e: freezing_level_km(e) for e in ("control", "graupel", "2mom")}

    def crop(a):
        return a[order][keep]

    freq_cmap = cmaps.WhiteBlueGreenYellowRed
    freq_norm = mcolors.LogNorm(vmin=FREQ_VMIN, vmax=FREQ_VMAX)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-DIFF_SCALE, vcenter=0.0, vmax=DIFF_SCALE)

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(19.0, 24.5))
    # Explicit margins: a wide LEFT margin holds the species symbol AND the
    # "Height (km)" label with clear space between them.
    fig.subplots_adjust(left=0.135, right=0.975, top=0.965, bottom=0.045,
                        hspace=0.45, wspace=0.58)

    for r, (sym, sp, name) in enumerate(ROWS):
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
            ax.tick_params(axis="both", labelsize=TICK_FS)
            # Row 1 (graupel) and row 2 (snow) set the two title patterns; rows
            # 3-5 repeat row 2, so titling them again is redundant.
            if r <= 1:
                ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"Mass fraction (kg kg$^{-1}$)", fontsize=AXIS_FS)
            if c == 0:
                ax.set_ylabel("Height (km)", fontsize=AXIS_FS)

            handles = []
            if kind == "nodata":
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=18, color="0.5")
                continue
            elif kind == "abs":
                e, s = payload
                field = crop(np.ma.masked_invalid(freq[(e, s)]))
                pcm = ax.pcolormesh(q_edges, y_edges, field, cmap=freq_cmap,
                                    norm=freq_norm, shading="flat")
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(pcm, cax=cax)
                cb.set_label("Frequency (%)", fontsize=CBAR_FS)
                cb.ax.tick_params(labelsize=CBAR_TICK_FS)
            else:
                a, b, s = payload
                diff = crop(np.ma.masked_invalid(freq[(a, s)] - freq[(b, s)]))
                pcm = ax.pcolormesh(q_edges, y_edges, diff, cmap="RdBu_r",
                                    norm=diff_norm, shading="flat")
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(pcm, cax=cax)
                cb.set_label(r"$\Delta$ Frequency (%)", fontsize=CBAR_FS)
                cb.ax.tick_params(labelsize=CBAR_TICK_FS)
                cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

            if med_exp is not None:
                m = crop(med[(med_exp, sp)])
                line, = ax.plot(m, y, color="black", lw=2.4, label="Median", zorder=6)
                handles.append(line)

            fexp = COL_FREEZE_EXP[c]
            zf = frz[fexp]
            if np.isfinite(zf):
                fl = ax.axhline(zf, color=FREEZE_COLOR, lw=2.4, ls="--", zorder=7,
                                label="0 °C level")
                handles.append(fl)

            if handles:
                ax.legend(handles=handles, loc="upper right", fontsize=LEG_FS, framealpha=0.9)

    # Species symbol only, in the wide left margin, well clear of "Height (km)".
    for r, (sym, _sp, _name) in enumerate(ROWS):
        pos = axes[r, 0].get_position()
        yc = 0.5 * (pos.y0 + pos.y1)
        fig.text(0.045, yc, sym, rotation=90, va="center", ha="center",
                 fontsize=SYM_FS, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(out, dpi=450, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    txt = OUTPUT_DIR / TEXT_NAME
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("Hydrometeor mass-fraction CFADs: C1M, G1M-C1M, G2M-G1M, with freezing levels\n")
        fh.write("=" * 74 + "\n")
        fh.write(f"Figure: {out}\n\n")
        fh.write("Columns: C1M (absolute), G1M - C1M, G2M - G1M.\n")
        fh.write("Rows: graupel, snow, cloud ice, cloud water, rain.\n")
        fh.write("Top row only: C1M (no data), G1M (absolute), G2M - G1M.\n")
        fh.write("Absolute: WhiteBlueGreenYellowRed LogNorm "
                 f"{FREQ_VMIN:g}..{FREQ_VMAX:g} %. Difference: RdBu_r +/-{DIFF_SCALE:g} %.\n")
        fh.write("Each height level normalised to 100 % (CFAD).\n")
        fh.write("Black SOLID line = per-level median of that column's RAW mass fraction\n")
        fh.write("  (C1M / G1M / G2M for columns 1 / 2 / 3) -- NOT a median of the differences.\n")
        fh.write("Black DASHED line = 0 degC (freezing) level of that column's model.\n\n")
        fh.write("Freezing (0 degC) level from the 2-year mean temperature profile (km):\n")
        for e in ("control", "graupel", "2mom"):
            fh.write(f"  {EXP_LABEL[e]} ({e}): {frz[e]:.2f} km\n")
        fh.write(f"\nCFAD n_files per experiment: "
                 f"{ {e: int(data[e]['n_files']) for e in data} }\n")
    print(f"[saved] {txt}")
    print("[freezing km]", {EXP_LABEL[e]: round(frz[e], 2) for e in frz})


if __name__ == "__main__":
    main()
