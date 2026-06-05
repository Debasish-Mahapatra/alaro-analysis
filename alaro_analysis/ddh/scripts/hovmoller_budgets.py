#!/usr/bin/env python3
"""
Hovmöller (time-height) diagrams of DDH 24h budget components.
For each variable and component:
  Col 1: C1M (control)
  Col 2: G1M − C1M
  Col 3: G1M − G2M
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alaro_analysis.common.figio import strip_cbar_zeros
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta

# ─── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = "/mnt/scratch/MANAUS/DDH/alaro-24h-budgets/results"
OUTPUT_DIR = "/mnt/scratch/MANAUS/DDH/hovmoller-plots"
EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
VARIABLES = ["CT", "QG", "QI", "QL", "QR", "QS", "QV", "TKE", "TTE", "UU", "VV"]

VAR_INFO = {
    "CT": {"name": "Temperature Budget (CT)", "unit": "K/day"},
    "QG": {"name": "Graupel (QG)", "unit": "kg/kg/day"},
    "QI": {"name": "Cloud Ice (QI)", "unit": "kg/kg/day"},
    "QL": {"name": "Cloud Water (QL)", "unit": "kg/kg/day"},
    "QR": {"name": "Rain (QR)", "unit": "kg/kg/day"},
    "QS": {"name": "Snow (QS)", "unit": "kg/kg/day"},
    "QV": {"name": "Water Vapour (QV)", "unit": "kg/kg/day"},
    "TKE": {"name": "Turbulence Kinetic Energy (TKE)", "unit": "m²/s²/day"},
    "TTE": {"name": "Total Turb. Energy (TTE)", "unit": "m²/s²/day"},
    "UU": {"name": "Zonal Momentum (UU)", "unit": "m/s/day"},
    "VV": {"name": "Meridional Momentum (VV)", "unit": "m/s/day"},
}


def get_sorted_days():
    """Get sorted list of day directory names from control."""
    exp_dir = os.path.join(RESULTS_DIR, "control")
    return sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))


def day_to_date(day_str):
    """Convert DDH20140101 -> datetime."""
    return datetime.strptime(day_str[3:], "%Y%m%d")


def get_components(variable):
    """Discover budget components for a given variable."""
    sample_dir = os.path.join(RESULTS_DIR, "control", "DDH20140101", variable, "data")
    if not os.path.isdir(sample_dir):
        return []
    prefix = f"{variable}.DHFDLABOF+0024."
    return sorted(
        f[len(prefix):-4]
        for f in os.listdir(sample_dir)
        if f.startswith(prefix) and f.endswith(".dta")
    )


def classify_component(comp):
    upper = comp.upper()
    if "RESIDUAL" in upper:
        return "residual"
    if "COMPSUM" in upper:
        return "compsum"
    if comp.startswith("V") and comp.endswith("M") and len(comp) <= 5:
        return "tendency"
    return "physics"


def get_display_name(comp):
    """Short display name for a component."""
    names = {
        "auto-cv": "Autoconv. (cv)", "auto-rs": "Autoconv. (rs)",
        "cond-cv": "Condensation (cv)", "cond-rs": "Condensation (rs)",
        "condcv": "Condensation (cv)", "condrs": "Condensation (rs)",
        "evap-cv": "Evaporation (cv)", "evap-rs": "Evaporation (rs)",
        "evapcv": "Evaporation (cv)", "evaprs": "Evaporation (rs)",
        "prec-cv": "Precipitation (cv)", "prec-rs": "Precipitation (rs)",
        "micro-cv": "Microphysics (cv)", "micro-rs": "Microphysics (rs)",
        "rad-sol": "Radiation: Solar", "rad-ther": "Radiation: IR",
        "turconv": "Turbulence (cv)", "turdiff": "Turbulence (diff)",
        "dynam": "Dynamics", "dyn": "Dynamics",
        "neg": "Neg. correction", "gwd-drag": "GWD drag",
        "advection": "Advection", "buoyancy": "Buoyancy",
        "diffusion": "Diffusion", "dissipation": "Dissipation",
        "shear": "Shear",
    }
    cat = classify_component(comp)
    if cat == "residual":
        return "Residual"
    if cat == "compsum":
        return "Component sum"
    if cat == "tendency":
        return "Total tendency"
    return names.get(comp, comp)


def load_timeseries(experiment, variable, component, days):
    """Load full time-height array [n_days x n_levels]."""
    prefix = f"{variable}.DHFDLABOF+0024."
    profiles = []
    pressure = None

    for day in days:
        filepath = os.path.join(
            RESULTS_DIR, experiment, day, variable, "data",
            f"{prefix}{component}.dta"
        )
        if os.path.isfile(filepath):
            try:
                data = np.loadtxt(filepath)
                if pressure is None:
                    pressure = -data[:, 0]  # negate to get positive hPa
                profiles.append(data[:, 1])
            except Exception:
                profiles.append(np.full(87, np.nan))
        else:
            profiles.append(np.full(87, np.nan))

    return np.array(profiles), pressure  # shape: [n_days, n_levels]


def plot_hovmoller(variable, component, days, dates):
    """Create one Hovmöller figure: C1M | G1M-C1M | G1M-G2M."""

    info = VAR_INFO.get(variable, {"name": variable, "unit": ""})
    disp_name = get_display_name(component)

    # Load data for all 3 experiments
    data_c1m, pressure = load_timeseries("control", variable, component, days)
    data_g1m, _ = load_timeseries("graupel", variable, component, days)
    data_g2m, _ = load_timeseries("2mom", variable, component, days)

    if pressure is None:
        return

    diff_g1m_c1m = data_g1m - data_c1m
    diff_g1m_g2m = data_g1m - data_g2m

    # Time axis as day indices
    n_days = len(days)

    # Create month tick positions
    month_ticks = []
    month_labels = []
    for i, d in enumerate(dates):
        if d.day == 1:
            month_ticks.append(i)
            month_labels.append(d.strftime("%b\n%Y") if d.month == 1 else d.strftime("%b"))

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey=True)

    panels = [
        (data_c1m, "C1M", None),
        (diff_g1m_c1m, "G1M − C1M", None),
        (diff_g1m_g2m, "G1M − G2M", None),
    ]

    for col, (data, title, _) in enumerate(panels):
        ax = axes[col]

        # Use diverging colormap centered at 0
        vmax = np.nanpercentile(np.abs(data), 99)
        if vmax < 1e-15:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.pcolormesh(
            np.arange(n_days), pressure, data.T,
            cmap='RdBu_r', norm=norm, shading='nearest'
        )

        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.set_xlabel("Time", fontsize=18)
        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=18)
        ax.set_ylim(1050, 0)
        ax.set_yticks(np.arange(0, 1100, 100))
        ax.tick_params(axis='both', labelsize=14)

        # Month ticks on x-axis
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, fontsize=12)
        ax.set_xlim(0, n_days)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        strip_cbar_zeros(cbar)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(info["unit"], fontsize=16)

    fig.suptitle(
        f"{info['name']} — {disp_name}\nHovmöller (time-height), 2014–2015",
        fontsize=24, fontweight='bold', y=1.02
    )
    fig.tight_layout()

    outpath = os.path.join(
        OUTPUT_DIR, variable,
        f"{variable}_{component}_hovmoller.png"
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=450, bbox_inches='tight')
    plt.close(fig)
    return outpath


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("DDH 24h Budget — Hovmöller Diagrams")
    print("  Col 1: C1M  |  Col 2: G1M−C1M  |  Col 3: G1M−G2M")
    print("=" * 60)

    days = get_sorted_days()
    dates = [day_to_date(d) for d in days]
    print(f"  Days: {len(days)} ({dates[0].date()} to {dates[-1].date()})")
    print()

    for variable in VARIABLES:
        components = get_components(variable)
        if not components:
            print(f"  {variable}: no components, skipping")
            continue

        print(f"Processing {variable} ({len(components)} components)...")
        for comp in components:
            outpath = plot_hovmoller(variable, comp, days, dates)
            if outpath:
                print(f"  {comp:20s} -> {os.path.basename(outpath)}")

    print(f"\nDone! All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
