#!/usr/bin/env python3
"""
Compute 2-year total average (2014+2015) of DDH 24h budget components
for each variable and each experiment, and plot them.

Layout: 1 row x 3 columns (C1M, G1M, G2M)
Conventions:
  - Consistent colors per physical PROCESS across all variables
  - Same process = same color (e.g. turbulence always brown)
  - Dashed = convective, Solid = resolved/other
  - Radiation = single color, solar=solid, thermal=dashed
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = "/mnt/scratch/MANAUS/DDH/alaro-24h-budgets/results"
OUTPUT_DIR = "/mnt/scratch/MANAUS/DDH/yearly-average-plots"
EXPERIMENTS = ["control", "graupel", "2mom"]
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
VARIABLES = ["CT", "QG", "QI", "QL", "QR", "QS", "QV", "TKE", "TTE", "UU", "VV"]

# Variable descriptions and units (from DDH documentation section 4.5, 6.1)
# CT = c_p * T thermal energy budget
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

# ─── Consistent color + style mapping ─────────────────────────────────────────
# PRINCIPLE:
#   Same physical process = same color
#   Convective variant = dashed, Resolved variant = solid
#   Radiation: solar = solid, thermal/IR = dashed (same color)
#   Turbulence: diffusion = solid, convection = dashed (same color)

# Process colors
C_AUTO   = "#1f77b4"  # blue       — autoconversion
C_COND   = "#d62728"  # red        — condensation
C_EVAP   = "#2ca02c"  # green      — evaporation
C_PREC   = "#ff7f0e"  # orange     — precipitation
C_MICRO  = "#9467bd"  # purple     — microphysics (CT)
C_RAD    = "#e6ab02"  # gold       — radiation (solar & thermal)
C_TURB   = "#8c564b"  # brown      — turbulence (diffusion & convection)
C_DYN    = "#7f7f7f"  # gray       — dynamics
C_NEG    = "#bcbd22"  # olive      — negativity correction
C_GWD    = "#e377c2"  # pink       — gravity wave drag
C_ADV    = "#17becf"  # cyan       — advection
C_BUOY   = "#2ca02c"  # green      — buoyancy
C_SHEAR  = "#1f77b4"  # blue       — shear
C_DISS   = "#d62728"  # red        — dissipation
C_DIFF   = "#8c564b"  # brown      — diffusion (TKE/TTE)

# (color, linestyle, display_label)
PROCESS_STYLE = {
    # Autoconversion: blue, cv=dashed rs=solid
    "auto-cv":   (C_AUTO,  "--", "Autoconv. (cv)"),
    "auto-rs":   (C_AUTO,  "-",  "Autoconv. (rs)"),

    # Condensation: red, cv=dashed rs=solid
    "cond-cv":   (C_COND,  "--", "Condensation (cv)"),
    "cond-rs":   (C_COND,  "-",  "Condensation (rs)"),
    "condcv":    (C_COND,  "--", "Condensation (cv)"),
    "condrs":    (C_COND,  "-",  "Condensation (rs)"),

    # Evaporation: green, cv=dashed rs=solid
    "evap-cv":   (C_EVAP,  "--", "Evaporation (cv)"),
    "evap-rs":   (C_EVAP,  "-",  "Evaporation (rs)"),
    "evapcv":    (C_EVAP,  "--", "Evaporation (cv)"),
    "evaprs":    (C_EVAP,  "-",  "Evaporation (rs)"),

    # Precipitation: orange, cv=dashed rs=solid
    "prec-cv":   (C_PREC,  "--", "Precipitation (cv)"),
    "prec-rs":   (C_PREC,  "-",  "Precipitation (rs)"),

    # Microphysics (CT only): purple, cv=dashed rs=solid
    "micro-cv":  (C_MICRO, "--", "Microphysics (cv)"),
    "micro-rs":  (C_MICRO, "-",  "Microphysics (rs)"),

    # Radiation: SAME gold color, solar=solid, thermal=dashed
    "rad-sol":   (C_RAD,   "-",  "Radiation: Solar"),
    "rad-ther":  (C_RAD,   "--", "Radiation: IR"),

    # Turbulence: SAME brown color, diffusion=solid, convection=dashed
    "turconv":   (C_TURB,  "--", "Turbulence (cv)"),
    "turdiff":   (C_TURB,  "-",  "Turbulence (diff)"),

    # Dynamics: gray
    "dynam":     (C_DYN,   "-",  "Dynamics"),
    "dyn":       (C_DYN,   "-",  "Dynamics"),

    # Negativity correction: olive
    "neg":       (C_NEG,   "-",  "Neg. correction"),

    # Gravity wave drag: pink
    "gwd-drag":  (C_GWD,   "-",  "GWD drag"),

    # TKE/TTE specific terms
    "advection":   (C_ADV,   "-",  "Advection"),
    "buoyancy":    (C_BUOY,  "-",  "Buoyancy"),
    "diffusion":   (C_DIFF,  "-",  "Diffusion"),
    "dissipation": (C_DISS,  "-",  "Dissipation"),
    "shear":       (C_SHEAR, "-",  "Shear"),
}

RESIDUAL_STYLE = ("#999999", "-.", "Residual")
COMPSUM_STYLE  = ("#999999", ":",  "Component sum")
TENDENCY_STYLE = ("#000000", "-",  "Total tendency")


def read_dta(filepath):
    """Read a .dta file: 2 columns (pressure, value), 87 levels."""
    try:
        data = np.loadtxt(filepath)
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None


def get_components(experiment, variable):
    """Discover budget components for a given experiment/variable."""
    sample_dir = os.path.join(RESULTS_DIR, experiment, "DDH20140101", variable, "data")
    if not os.path.isdir(sample_dir):
        return []
    components = []
    prefix = f"{variable}.DHFDLABOF+0024."
    for f in sorted(os.listdir(sample_dir)):
        if f.startswith(prefix) and f.endswith(".dta"):
            comp = f[len(prefix):-4]
            components.append(comp)
    return components


def get_all_days(experiment):
    """Get all DDH day directories across both years."""
    exp_dir = os.path.join(RESULTS_DIR, experiment)
    return sorted(d for d in os.listdir(exp_dir) if d.startswith("DDH"))


def compute_total_avg(experiment, variable, component):
    """Compute the total average profile across all days (2014+2015)."""
    days = get_all_days(experiment)
    prefix = f"{variable}.DHFDLABOF+0024."
    profiles = []
    pressure = None

    for day in days:
        filepath = os.path.join(
            RESULTS_DIR, experiment, day, variable, "data",
            f"{prefix}{component}.dta"
        )
        if not os.path.isfile(filepath):
            continue
        p, v = read_dta(filepath)
        if p is not None:
            if pressure is None:
                pressure = p
            profiles.append(v)

    if len(profiles) == 0:
        return None, None, 0

    avg = np.mean(profiles, axis=0)
    return pressure, avg, len(profiles)


def classify_component(comp):
    """Classify a component as 'physics', 'residual', 'compsum', or 'tendency'."""
    upper = comp.upper()
    if "RESIDUAL" in upper:
        return "residual"
    if "COMPSUM" in upper:
        return "compsum"
    if comp.startswith("V") and comp.endswith("M") and len(comp) <= 5:
        return "tendency"
    return "physics"


def get_style(comp):
    """Get (color, linestyle, linewidth, label) for a component."""
    cat = classify_component(comp)
    if cat == "residual":
        c, ls, lbl = RESIDUAL_STYLE
        return c, ls, 3.0, lbl
    if cat == "compsum":
        c, ls, lbl = COMPSUM_STYLE
        return c, ls, 3.0, lbl
    if cat == "tendency":
        c, ls, lbl = TENDENCY_STYLE
        return c, ls, 3.5, lbl
    if comp in PROCESS_STYLE:
        c, ls, lbl = PROCESS_STYLE[comp]
        return c, ls, 3.0, lbl
    # Fallback
    return "#333333", "-", 2.5, comp


def plot_variable(variable):
    """Plot total-average budgets: 1 row x 3 columns (C1M, G1M, G2M)."""
    components = get_components("control", variable)
    if not components:
        print(f"  No components found for {variable}, skipping.")
        return

    # Order: physics first, then special
    physics = [c for c in components if classify_component(c) == "physics"]
    specials = [c for c in components if classify_component(c) != "physics"]
    plot_comps = physics + specials

    info = VAR_INFO.get(variable, {"name": variable, "unit": ""})

    # First pass: compute all data and find x-ranges per experiment
    all_data = {}  # (exp, comp) -> (pressure, avg)
    exp_xranges = {}
    for experiment in EXPERIMENTS:
        xmin, xmax = 0.0, 0.0
        for comp in plot_comps:
            pressure, avg, n_days = compute_total_avg(experiment, variable, comp)
            if pressure is not None:
                all_data[(experiment, comp)] = (pressure, avg)
                xmin = min(xmin, avg.min())
                xmax = max(xmax, avg.max())
        exp_xranges[experiment] = (xmin, xmax)

    # Decide whether to share x-axis: if the widest panel range is >5x
    # the narrowest, let each panel have its own x-axis
    ranges = [abs(xmax - xmin) for xmin, xmax in exp_xranges.values()]
    ratio = max(ranges) / max(min(ranges), 1e-30)
    share_x = ratio < 5.0

    fig, axes = plt.subplots(1, 3, figsize=(28, 12), sharey=True,
                              sharex=share_x)

    for col, experiment in enumerate(EXPERIMENTS):
        ax = axes[col]
        label = EXP_LABELS[experiment]
        ax.set_title(label, fontsize=26, fontweight='bold')

        for comp in plot_comps:
            key = (experiment, comp)
            if key not in all_data:
                continue
            pressure, avg = all_data[key]

            color, ls, lw, display_label = get_style(comp)
            # Negate pressure (stored as negative in .dta files)
            ax.plot(avg, -pressure, color=color, linestyle=ls,
                    linewidth=lw, label=display_label)

        ax.axvline(0, color='gray', linewidth=0.8, linestyle='-')
        ax.set_xlabel(info["unit"], fontsize=20)
        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=20)
        # Pressure axis: 1000 at bottom, 0 at top, ticks every 100 hPa
        ax.set_ylim(1050, 0)
        ax.set_yticks(np.arange(0, 1100, 100))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=17)

    # Single legend from first subplot — deduplicated
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        seen = {}
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                unique_h.append(h)
                unique_l.append(l)
        fig.legend(unique_h, unique_l, loc='center right',
                   bbox_to_anchor=(1.13, 0.5), fontsize=18, frameon=True,
                   framealpha=0.9, edgecolor='black')

    fig.suptitle(f"{info['name']} — 2-Year Mean 24h Budget (2014–2015)",
                 fontsize=26, fontweight='bold', y=1.02)
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"{variable}_2yr_avg_budget.png")
    fig.savefig(outpath, dpi=450, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove old yearly plots
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith("_yearly_avg_budget.png"):
            os.remove(os.path.join(OUTPUT_DIR, f))
            print(f"  Removed old: {f}")

    print("=" * 60)
    print("DDH 24h Budget — 2-Year Total Average & Plotting")
    print("=" * 60)
    print(f"Experiments: {list(EXP_LABELS.values())}")
    print(f"Variables:   {VARIABLES}")
    print(f"Averaging over: 2014 + 2015 (730 days)")
    print(f"DPI: 400")
    print()

    for exp in EXPERIMENTS:
        n = len(get_all_days(exp))
        print(f"  {EXP_LABELS[exp]} ({exp}): {n} total days")
    print()

    for variable in VARIABLES:
        print(f"Processing {variable}...")
        plot_variable(variable)

    print("\nDone! All plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
