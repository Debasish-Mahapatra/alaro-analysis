"""SUP figures: evaporation and freezing for the 3 freezing-diagnostic runs.

Radar-masked, 15-day, all-hours mean vertical profiles (control_freezing C1M,
graupel_freezing G1M, 2mom_freezing G2M):

  1. freezing_evaporation_profiles.png  - 6 panels: evaporation (EVAPR rain,
     EVAPS snow, EVAPG graupel) and rain freezing (FRZRS->snow, FRZRG->graupel,
     FRZTOT total), three experiments per panel.
  2. freezing_evaporation_totals.png    - total evaporation (EVAPR+EVAPS+EVAPG)
     vs total rain freezing (FRZTOT), as vertical profiles, three experiments.

Source: processed-data/data/freezing_evap_masked/<exp>_masked_profiles.npz
(per-field (87,) model-level arrays, top-first), built by
extra-experiments/_sup_freezing_evap_masked.py over the FA-grid radar mask.
Model level is mapped to height with the existing geopotential cache.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

MASKED_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/freezing_evap_masked")
HEIGHT_NPZ = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/"
                  "geopotential/2years/control_full-domain_height_profile_first.npz")
OUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/freezing_evaporation_profiles")
DPI = 450
Z_MAX = 15.0
TO_GKGDAY = 1.0e3 * 86400.0      # kg/kg/s -> g/kg/day

EXPS = [
    ("control_freezing", "C1M", "#d62728"),
    ("graupel_freezing", "G1M", "#1f77b4"),
    ("2mom_freezing", "G2M", "#2ca02c"),
]
ROWS = [
    [("EVAPR", "rain evaporation"), ("EVAPS", "snow sublimation"),
     ("EVAPG", "graupel sublimation")],
    [("FRZRS", "rain $\\rightarrow$ snow freezing"),
     ("FRZRG", "rain $\\rightarrow$ graupel freezing"),
     ("FRZTOT", "total rain freezing")],
]
CAPTION = ("radar-masked, 15-day all-hours mean, 1-15 Mar 2014")
FREEZING_COLOR = "0.45"

# 0 C freezing level from the radar-masked raw FA temperature profile (S###TEMPERATURE),
# per freezing run; one line per experiment is drawn in that experiment's colour
# (the three are ~identical at ~4.8 km, so they overlap visually).
_TZ = np.load(MASKED_DIR / "tzero_km.npz")
TZERO = {k: float(_TZ[k]) for k in _TZ.files}
TZ0 = float(np.nanmean(list(TZERO.values())))


def height_top_first():
    h = np.asarray(np.load(HEIGHT_NPZ, allow_pickle=True)["height_m"], float) / 1000.0
    return h[::-1]            # surface-first cache -> top-first (index k = model level k+1)


def load_field(exp, field):
    d = np.load(MASKED_DIR / f"{exp}_masked_profiles.npz")
    return np.asarray(d[field], float) * TO_GKGDAY


def clip_sort(z, v):
    m = (z >= 0) & (z <= Z_MAX)
    z, v = z[m], v[m]
    order = np.argsort(z)
    return z[order], v[order]


def gfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def make_profiles(h, columns):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 9.0), sharey=True)
    for r, row in enumerate(ROWS):
        for c, (field, title) in enumerate(row):
            ax = axes[r, c]
            for exp, label, color in EXPS:
                z, v = clip_sort(h, load_field(exp, field))
                ax.plot(v, z, color=color, lw=2.0, label=label)
                ax.axhline(TZERO[exp], color=color, ls=":", lw=1.1, alpha=0.8)
                columns[f"{field}_{label}_z_km"] = z
                columns[f"{field}_{label}_g_kg_day"] = v
            ax.axvline(0.0, color="0.7", lw=0.6)
            ax.set_title(f"{field}  ({title})")
            ax.set_xlabel("rate (g kg$^{-1}$ day$^{-1}$)")
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel("altitude (km)")
            gfmt(ax)
    axes[0, 0].set_ylim(0, Z_MAX)
    handles = [Line2D([], [], color=col, lw=2.0, label=lab) for _, lab, col in EXPS]
    handles.append(Line2D([], [], color=FREEZING_COLOR, ls=":", lw=1.2, label="0 $^\\circ$C isotherm"))
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    fig.suptitle("Evaporation and rain-freezing rate profiles", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = OUT_DIR / "freezing_evaporation_profiles.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


def make_totals(h, columns):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.2), sharey=True)
    for exp, label, color in EXPS:
        evap = (load_field(exp, "EVAPR") + load_field(exp, "EVAPS")
                + load_field(exp, "EVAPG"))
        frz = load_field(exp, "FRZTOT")
        ze, ve = clip_sort(h, evap)
        zf, vf = clip_sort(h, frz)
        axes[0].plot(ve, ze, color=color, lw=2.2, label=label)
        axes[1].plot(vf, zf, color=color, lw=2.2, label=label)
        axes[0].axhline(TZERO[exp], color=color, ls=":", lw=1.1, alpha=0.8)
        axes[1].axhline(TZERO[exp], color=color, ls=":", lw=1.1, alpha=0.8)
        columns[f"TOTEVAP_{label}_z_km"] = ze
        columns[f"TOTEVAP_{label}_g_kg_day"] = ve
        columns[f"FRZTOT_{label}_z_km"] = zf
        columns[f"FRZTOT_{label}_g_kg_day"] = vf
    axes[0].set_title("total evaporation (rain + snow + graupel)")
    axes[1].set_title("total rain freezing (FRZTOT)")
    for ax in axes:
        ax.set_xlabel("rate (g kg$^{-1}$ day$^{-1}$)")
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)
        gfmt(ax)
    axes[0].set_ylabel("altitude (km)")
    axes[0].set_ylim(0, Z_MAX)
    handles = [Line2D([], [], color=col, lw=2.2, label=lab) for _, lab, col in EXPS]
    handles.append(Line2D([], [], color=FREEZING_COLOR, ls=":", lw=1.2, label="0 $^\\circ$C isotherm"))
    axes[0].legend(handles=handles, loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    fig.suptitle("Total evaporation vs total rain freezing", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = OUT_DIR / "freezing_evaporation_totals.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


def write_data(columns, name):
    (OUT_DIR / "data_txt").mkdir(parents=True, exist_ok=True)
    txt = OUT_DIR / "data_txt" / name
    keys = list(columns.keys())
    n = max((len(columns[k]) for k in keys), default=0)
    with open(txt, "w") as f:
        f.write(f"# Evaporation/freezing rate profiles (g/kg/day) vs altitude (km). {CAPTION}.\n")
        f.write("# C1M=control_freezing, G1M=graupel_freezing, G2M=2mom_freezing.\n")
        f.write("\t".join(keys) + "\n")
        for i in range(n):
            f.write("\t".join(f"{columns[k][i]:.6g}" if i < len(columns[k]) else ""
                              for k in keys) + "\n")
    print(f"wrote {txt}")


def main():
    h = height_top_first()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols_p = {}
    make_profiles(h, cols_p)
    write_data(cols_p, "freezing_evaporation_profiles.txt")
    cols_t = {}
    make_totals(h, cols_t)
    write_data(cols_t, "freezing_evaporation_totals.txt")


if __name__ == "__main__":
    main()
