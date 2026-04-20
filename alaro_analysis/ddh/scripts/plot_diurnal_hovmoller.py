#!/usr/bin/env python3
"""
Diurnal cycle Hovmöller diagrams.

For each variable: 1×3 panels (C1M, G1M, G2M)
  x-axis = hour of day (0–23 UTC)
  y-axis = pressure (hPa)
  color  = mean hourly tendency

Uses pre-computed diurnal cycle data from extract_diurnal_cycle.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "diurnal-cycle-data")
YEARLY_DIR = os.path.join(SCRIPT_DIR, "..", "yearly-average-data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "new-analysis-plots")

EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}

VARIABLES = ["CT", "QV", "QL", "QI", "QR", "QS", "QG", "UU", "VV", "TKE", "TTE"]

VAR_INFO = {
    "CT":  {"name": "Temperature (CT)",        "unit": "K/h"},
    "QG":  {"name": "Graupel (QG)",            "unit": "g/kg/h"},
    "QI":  {"name": "Cloud Ice (QI)",          "unit": "g/kg/h"},
    "QL":  {"name": "Cloud Water (QL)",        "unit": "g/kg/h"},
    "QR":  {"name": "Rain (QR)",               "unit": "g/kg/h"},
    "QS":  {"name": "Snow (QS)",               "unit": "g/kg/h"},
    "QV":  {"name": "Water Vapour (QV)",       "unit": "g/kg/h"},
    "TKE": {"name": "TKE",                     "unit": "J/kg/h"},
    "TTE": {"name": "Total Turb. Energy (TTE)","unit": "J/kg/h"},
    "UU":  {"name": "Zonal Momentum (UU)",     "unit": "m/s/h"},
    "VV":  {"name": "Meridional Momentum (VV)","unit": "m/s/h"},
}

FREEZING_K = 273.15

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def load_freezing_level(exp):
    fpath = os.path.join(YEARLY_DIR, exp, "CT_TEMP", "data",
                         "CT_TEMP.DHFDLABOF+0024.T_MEAN.dta")
    if not os.path.isfile(fpath):
        return None
    data = np.loadtxt(fpath)
    pressure = np.abs(data[:, 0])
    temperature = data[:, 1]
    for i in range(len(temperature) - 1):
        t0, t1 = temperature[i], temperature[i + 1]
        if t0 <= FREEZING_K <= t1 or t1 <= FREEZING_K <= t0:
            frac = (FREEZING_K - t0) / (t1 - t0)
            return pressure[i] + frac * (pressure[i + 1] - pressure[i])
    return None


def plot_diurnal_hovmollers():
    print("Diurnal cycle Hovmoller diagrams ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for var in VARIABLES:
        # Load data for all experiments
        matrices = {}
        pressure = None
        for exp in EXPERIMENTS:
            fpath = os.path.join(DATA_DIR, exp, f"{var}_diurnal.npy")
            ppath = os.path.join(DATA_DIR, exp, "pressure.npy")
            if not os.path.isfile(fpath) or not os.path.isfile(ppath):
                continue
            matrices[exp] = np.load(fpath)   # [24, 87]
            if pressure is None:
                pressure = np.load(ppath)

        if len(matrices) < 3 or pressure is None:
            print(f"  {var}: skipped (missing data)")
            continue

        # Symmetric color limits
        all_vals = np.concatenate([m.ravel() for m in matrices.values()])
        vmax = np.percentile(np.abs(all_vals[np.isfinite(all_vals)]), 98)
        if vmax < 1e-30:
            vmax = 1.0

        info = VAR_INFO.get(var, {"name": var, "unit": ""})

        # Edges for pcolormesh
        hour_edges = np.arange(25)  # 0..24
        p_edges = np.zeros(len(pressure) + 1)
        p_edges[1:-1] = 0.5 * (pressure[:-1] + pressure[1:])
        p_edges[0] = max(0, 2 * pressure[0] - p_edges[1])
        p_edges[-1] = 2 * pressure[-1] - p_edges[-2]

        fig = plt.figure(figsize=(22, 10))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.10)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col, exp in enumerate(EXPERIMENTS):
            ax = axes[col]
            # matrices[exp] is [24, n_lev], need to transpose for
            # pcolormesh(hour_edges, p_edges, data[n_lev, 24])
            im = ax.pcolormesh(
                hour_edges, p_edges, matrices[exp].T,
                cmap='RdBu_r', norm=norm, shading='flat'
            )
            ax.set_title(EXP_LABELS[exp], fontsize=18, fontweight='bold')
            ax.set_xlabel("Hour (UTC)", fontsize=14)
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 3))
            ax.set_ylim(1050, 50)
            ax.set_yticks(np.arange(100, 1100, 100))
            if col == 0:
                ax.set_ylabel("Pressure (hPa)", fontsize=14)
            else:
                ax.set_yticklabels([])

            # Freezing level
            p_freeze = load_freezing_level(exp)
            if p_freeze is not None:
                ax.axhline(p_freeze, color='black', linewidth=2.0,
                           linestyle='--', alpha=0.85, zorder=5)
                ax.text(23.5, p_freeze - 12, '0\u00b0C',
                        fontsize=10, fontweight='bold', color='black',
                        ha='right', va='bottom', alpha=0.85,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none'))

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(info["unit"], fontsize=13)

        fig.suptitle(
            f"{info['name']} — Mean Diurnal Cycle (2-yr average)",
            fontsize=17, fontweight='bold', y=1.01)

        outpath = os.path.join(OUTPUT_DIR, f"{var}_diurnal_hovmoller.png")
        fig.savefig(outpath, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"  {var} -> {os.path.basename(outpath)}")


def main():
    print("=" * 60)
    print("DDH Diurnal Cycle Hovmoller Plots")
    print(f"  Data:   {os.path.abspath(DATA_DIR)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)
    print()
    plot_diurnal_hovmollers()
    print()
    print("Done! Figures saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
