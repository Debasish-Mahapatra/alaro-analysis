#!/usr/bin/env python3
"""
Budget fingerprint heatmaps for DDH 24h budget data (2-year average).
Uses pre-computed yearly-average-data.

Plot types:
  1. Fingerprint heatmaps — process x pressure, 3 panels (C1M, G1M, G2M)
  2. Difference fingerprints — G1M-C1M and G2M-C1M delta heatmaps
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alaro_analysis.common.figio import strip_cbar_zeros
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

# ─── Configuration ────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "yearly-average-data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "new-analysis-plots")

EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}

VARIABLES = ["CT", "QG", "QI", "QL", "QR", "QS", "QV", "TKE", "TTE", "UU", "VV"]

VAR_INFO = {
    "CT":  {"name": "Temperature (CT)",        "unit": "K/day"},
    "QG":  {"name": "Graupel (QG)",            "unit": "kg/kg/day"},
    "QI":  {"name": "Cloud Ice (QI)",          "unit": "kg/kg/day"},
    "QL":  {"name": "Cloud Water (QL)",        "unit": "kg/kg/day"},
    "QR":  {"name": "Rain (QR)",               "unit": "kg/kg/day"},
    "QS":  {"name": "Snow (QS)",               "unit": "kg/kg/day"},
    "QV":  {"name": "Water Vapour (QV)",       "unit": "kg/kg/day"},
    "TKE": {"name": "TKE",                     "unit": "m\u00b2/s\u00b2/day"},
    "TTE": {"name": "Total Turb. Energy (TTE)","unit": "m\u00b2/s\u00b2/day"},
    "UU":  {"name": "Zonal Momentum (UU)",     "unit": "m/s/day"},
    "VV":  {"name": "Meridional Momentum (VV)","unit": "m/s/day"},
}

# ─── Process labels ──────────────────────────────────────────────────────────
PROCESS_LABELS = {
    "auto-cv": "Autoconv. (cv)",   "auto-rs": "Autoconv. (rs)",
    "cond-cv": "Condensation (cv)","cond-rs": "Condensation (rs)",
    "condcv":  "Condensation (cv)","condrs":  "Condensation (rs)",
    "evap-cv": "Evaporation (cv)", "evap-rs": "Evaporation (rs)",
    "evapcv":  "Evaporation (cv)", "evaprs":  "Evaporation (rs)",
    "prec-cv": "Precipitation (cv)","prec-rs":"Precipitation (rs)",
    "micro-cv":"Microphysics (cv)","micro-rs":"Microphysics (rs)",
    "rad-sol": "Radiation: Solar", "rad-ther":"Radiation: IR",
    "turconv": "Turbulence (cv)",  "turdiff": "Turbulence (diff)",
    "dynam":   "Dynamics",         "dyn":     "Dynamics",
    "neg":     "Neg. correction",  "gwd-drag":"GWD drag",
    "advection":"Advection",       "buoyancy":"Buoyancy",
    "diffusion":"Diffusion",       "dissipation":"Dissipation",
    "shear":   "Shear",
}

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

FREEZING_K = 273.15  # 0 degrees C in Kelvin


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load(exp, var, proc):
    """Load a .dta profile.  Returns (pressure_hPa, values)."""
    fpath = os.path.join(BASE, exp, var, "data",
                         f"{var}.DHFDLABOF+0024.{proc}.dta")
    if not os.path.isfile(fpath):
        return None, None
    data = np.loadtxt(fpath)
    return np.abs(data[:, 0]), data[:, 1]


def discover_physics_procs(exp, var):
    """Return list of physics process keys (no residual/compsum/tendency)."""
    dirpath = os.path.join(BASE, exp, var, "data")
    if not os.path.isdir(dirpath):
        return []
    prefix = f"{var}.DHFDLABOF+0024."
    procs = []
    for f in sorted(os.listdir(dirpath)):
        if f.startswith(prefix) and f.endswith(".dta"):
            comp = f[len(prefix):-4]
            upper = comp.upper()
            if "RESIDUAL" in upper or "COMPSUM" in upper:
                continue
            if comp.startswith("V") and comp.endswith("M") and len(comp) <= 5:
                continue
            procs.append(comp)
    return procs


def proc_label(comp):
    return PROCESS_LABELS.get(comp, comp)


def load_freezing_level(exp):
    """Find the pressure of the 0 deg C isotherm from the mean temperature profile."""
    fpath = os.path.join(BASE, exp, "CT_TEMP", "data",
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
            p_freeze = pressure[i] + frac * (pressure[i + 1] - pressure[i])
            return p_freeze
    return None


def load_data_matrices(var, procs):
    """Load [n_lev x n_proc] matrices for all experiments."""
    p_ref, _ = load("control", var, procs[0])
    if p_ref is None:
        return None, None
    n_lev = len(p_ref)
    n_proc = len(procs)
    matrices = {}
    for exp in EXPERIMENTS:
        mat = np.full((n_lev, n_proc), np.nan)
        for j, proc in enumerate(procs):
            p, v = load(exp, var, proc)
            if v is not None:
                mat[:, j] = v
        matrices[exp] = mat
    return p_ref, matrices


def make_pressure_edges(p_ref):
    """Build pressure edge array for pcolormesh."""
    n_lev = len(p_ref)
    p_edges = np.zeros(n_lev + 1)
    p_edges[1:-1] = 0.5 * (p_ref[:-1] + p_ref[1:])
    p_edges[0] = max(0, 2 * p_ref[0] - p_edges[1])
    p_edges[-1] = 2 * p_ref[-1] - p_edges[-2]
    return p_edges


def draw_freezing_level(ax, exp, n_proc):
    """Draw freezing level line and label on an axis."""
    p_freeze = load_freezing_level(exp)
    if p_freeze is not None:
        ax.axhline(p_freeze, color='black', linewidth=2.0,
                   linestyle='--', alpha=0.85, zorder=5)
        ax.text(n_proc - 0.1, p_freeze - 12, '0\u00b0C',
                fontsize=10, fontweight='bold', color='black',
                ha='right', va='bottom', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='white', alpha=0.7,
                          edgecolor='none'))


# ─── Fingerprint heatmaps ───────────────────────────────────────────────────

def plot_fingerprints():
    print("Budget fingerprint heatmaps ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for var in VARIABLES:
        procs = discover_physics_procs("control", var)
        if not procs:
            continue

        p_ref, matrices = load_data_matrices(var, procs)
        if p_ref is None:
            continue
        n_lev = len(p_ref)
        n_proc = len(procs)

        all_vals = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
        if len(all_vals) == 0:
            continue
        vmax = np.percentile(np.abs(all_vals), 98)
        if vmax < 1e-30:
            vmax = 1.0

        labels = [proc_label(p) for p in procs]
        info = VAR_INFO.get(var, {"name": var, "unit": ""})
        p_edges = make_pressure_edges(p_ref)
        x_edges = np.arange(n_proc + 1)

        fig = plt.figure(figsize=(8 + n_proc * 1.2, 10))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col, exp in enumerate(EXPERIMENTS):
            ax = axes[col]
            im = ax.pcolormesh(x_edges, p_edges, matrices[exp],
                               cmap='RdBu_r', norm=norm, shading='flat')
            ax.set_title(EXP_LABELS[exp], fontsize=18, fontweight='bold')
            ax.set_xticks(np.arange(n_proc) + 0.5)
            ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=10)
            ax.set_ylim(1050, 50)
            ax.set_yticks(np.arange(100, 1100, 100))
            if col == 0:
                ax.set_ylabel("Pressure (hPa)", fontsize=14)
            else:
                ax.set_yticklabels([])
            draw_freezing_level(ax, exp, n_proc)

        cbar = fig.colorbar(im, cax=cax)
        strip_cbar_zeros(cbar)
        cbar.set_label(info["unit"], fontsize=13)
        fig.suptitle(f"{info['name']} — Budget Fingerprint (2-yr mean)",
                     fontsize=17, fontweight='bold', y=1.01)

        outpath = os.path.join(OUTPUT_DIR, f"{var}_fingerprint.png")
        fig.savefig(outpath, dpi=450, bbox_inches='tight')
        plt.close(fig)
        print(f"  {var} -> {os.path.basename(outpath)}")


# ─── Difference fingerprints ────────────────────────────────────────────────

def plot_diff_fingerprints():
    """G1M-C1M and G2M-C1M difference heatmaps."""
    print("Difference fingerprint heatmaps ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for var in VARIABLES:
        procs = discover_physics_procs("control", var)
        if not procs:
            continue

        p_ref, matrices = load_data_matrices(var, procs)
        if p_ref is None:
            continue
        n_lev = len(p_ref)
        n_proc = len(procs)

        diff_g1m = matrices["graupel"] - matrices["control"]
        diff_g2m = matrices["2mom"] - matrices["control"]

        all_diffs = np.concatenate([
            diff_g1m[np.isfinite(diff_g1m)],
            diff_g2m[np.isfinite(diff_g2m)]
        ])
        if len(all_diffs) == 0:
            continue
        vmax = np.percentile(np.abs(all_diffs), 98)
        if vmax < 1e-30:
            vmax = 1.0

        labels = [proc_label(p) for p in procs]
        info = VAR_INFO.get(var, {"name": var, "unit": ""})
        p_edges = make_pressure_edges(p_ref)
        x_edges = np.arange(n_proc + 1)

        fig = plt.figure(figsize=(6 + n_proc * 1.0, 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        cax = fig.add_subplot(gs[0, 2])
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col, (diff_mat, title, freeze_exp) in enumerate([
            (diff_g1m, "G1M \u2212 C1M", "graupel"),
            (diff_g2m, "G2M \u2212 C1M", "2mom"),
        ]):
            ax = axes[col]
            im = ax.pcolormesh(x_edges, p_edges, diff_mat,
                               cmap='RdBu_r', norm=norm, shading='flat')
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_xticks(np.arange(n_proc) + 0.5)
            ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=10)
            ax.set_ylim(1050, 50)
            ax.set_yticks(np.arange(100, 1100, 100))
            if col == 0:
                ax.set_ylabel("Pressure (hPa)", fontsize=14)
            else:
                ax.set_yticklabels([])
            draw_freezing_level(ax, freeze_exp, n_proc)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"\u0394 {info['unit']}", fontsize=13)
        fig.suptitle(f"{info['name']} — Difference Fingerprint (2-yr mean)",
                     fontsize=17, fontweight='bold', y=1.01)

        outpath = os.path.join(OUTPUT_DIR, f"{var}_diff_fingerprint.png")
        fig.savefig(outpath, dpi=450, bbox_inches='tight')
        plt.close(fig)
        print(f"  {var} -> {os.path.basename(outpath)}")


def main():
    print("=" * 60)
    print("DDH Budget Fingerprint Plots")
    print(f"  Data:   {os.path.abspath(BASE)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)
    print()
    plot_fingerprints()
    print()
    plot_diff_fingerprints()
    print()
    print("Done! Figures saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
