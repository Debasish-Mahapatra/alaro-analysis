#!/usr/bin/env python3
"""
Plot the time-averaged DDH 24-hour budgets.

For each variable, produces one figure with 3 experiments as sub-panels
(control, graupel, 2mom), showing all budget terms as coloured lines
vs pressure.  All panels share the same x- and y-axis scale.

Output: one PNG per variable at 400 dpi.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alaro_analysis.ddh.plot_style import (
    PROCESS_COLOURS,
    get_line_style,
    get_process_name,
)

# ── Font / rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
})

# ── Variable metadata ────────────────────────────────────────────────────────
VAR_META = {
    "CT": {
        "title": "Temperature Budget",
        "unit": "K / day",
        "terms": [
            ("dynam",         "Dynamics"),
            ("micro-rs",      "Micro (resolved)"),
            ("micro-cv",      "Micro (convective)"),
            ("turconv",       "Turbulence (conv)"),
            ("rad-sol",       "Radiation (solar)"),
            ("rad-ther",      "Radiation (thermal)"),
            ("turdiff",       "Turbulence (diff)"),
            ("TCTRESIDUAL",   "Residual"),
            ("VCTM",          "Tendency"),
        ],
    },
    "QG": {
        "title": "Graupel Budget",
        "unit": "g / kg / day",
        "terms": [
            ("evap-cv",       "Evaporation (conv)"),
            ("evap-rs",       "Evaporation (resolved)"),
            ("auto-cv",       "Autoconversion (conv)"),
            ("auto-rs",       "Autoconversion (resolved)"),
            ("prec-cv",       "Precipitation (conv)"),
            ("prec-rs",       "Precipitation (resolved)"),
            ("neg",           "Negativity correction"),
            ("TQGCOMPSUM",    "Sum of tendencies"),
            ("TQGRESIDUAL",   "Residual"),
            ("VQGM",          "Tendency"),
        ],
    },
    "QI": {
        "title": "Ice Water Budget",
        "unit": "g / kg / day",
        "terms": [
            ("turdiff",       "Turbulence (diff)"),
            ("turconv",       "Turbulence (conv)"),
            ("auto-cv",       "Autoconversion (conv)"),
            ("auto-rs",       "Autoconversion (resolved)"),
            ("cond-cv",       "Condensation (conv)"),
            ("cond-rs",       "Condensation (resolved)"),
            ("neg",           "Negativity correction"),
            ("TQICOMPSUM",    "Sum of tendencies"),
            ("TQIRESIDUAL",   "Residual"),
            ("VQIM",          "Tendency"),
        ],
    },
    "QL": {
        "title": "Liquid Water Budget",
        "unit": "g / kg / day",
        "terms": [
            ("turdiff",       "Turbulence (diff)"),
            ("turconv",       "Turbulence (conv)"),
            ("auto-cv",       "Autoconversion (conv)"),
            ("auto-rs",       "Autoconversion (resolved)"),
            ("cond-cv",       "Condensation (conv)"),
            ("cond-rs",       "Condensation (resolved)"),
            ("neg",           "Negativity correction"),
            ("TQLCOMPSUM",    "Sum of tendencies"),
            ("TQLRESIDUAL",   "Residual"),
            ("VQLM",          "Tendency"),
        ],
    },
    "QR": {
        "title": "Rain Water Budget",
        "unit": "g / kg / day",
        "terms": [
            ("evap-cv",       "Evaporation (conv)"),
            ("evap-rs",       "Evaporation (resolved)"),
            ("auto-cv",       "Autoconversion (conv)"),
            ("auto-rs",       "Autoconversion (resolved)"),
            ("prec-cv",       "Precipitation (conv)"),
            ("prec-rs",       "Precipitation (resolved)"),
            ("neg",           "Negativity correction"),
            ("TQRCOMPSUM",    "Sum of tendencies"),
            ("TQRRESIDUAL",   "Residual"),
            ("VQRM",          "Tendency"),
        ],
    },
    "QS": {
        "title": "Snow Budget",
        "unit": "g / kg / day",
        "terms": [
            ("evap-cv",       "Evaporation (conv)"),
            ("evap-rs",       "Evaporation (resolved)"),
            ("auto-cv",       "Autoconversion (conv)"),
            ("auto-rs",       "Autoconversion (resolved)"),
            ("prec-cv",       "Precipitation (conv)"),
            ("prec-rs",       "Precipitation (resolved)"),
            ("neg",           "Negativity correction"),
            ("TQSCOMPSUM",    "Sum of tendencies"),
            ("TQSRESIDUAL",   "Residual"),
            ("VQSM",          "Tendency"),
        ],
    },
    "QV": {
        "title": "Water Vapour Budget",
        "unit": "g / kg / day",
        "terms": [
            ("dynam",         "Dynamics"),
            ("condcv",        "Condensation (conv)"),
            ("condrs",        "Condensation (resolved)"),
            ("evapcv",        "Evaporation (conv)"),
            ("evaprs",        "Evaporation (resolved)"),
            ("turdiff",       "Turbulence (diff)"),
            ("turconv",       "Turbulence (conv)"),
            ("neg",           "Negativity correction"),
            ("TQVRESIDUAL",   "Residual"),
            ("VQVM",          "Tendency"),
        ],
    },
    "TKE": {
        "title": "TKE Budget",
        "unit": "J / kg / day",
        "terms": [
            ("advection",     "Advection"),
            ("diffusion",     "Diffusion"),
            ("shear",         "Shear"),
            ("buoyancy",      "Buoyancy"),
            ("dissipation",   "Dissipation"),
            ("TTKRESIDUAL",   "Residual"),
            ("VTKM",          "Tendency"),
        ],
    },
    "TTE": {
        "title": "TTE Budget",
        "unit": "J / kg / day",
        "terms": [
            ("advection",     "Advection"),
            ("diffusion",     "Diffusion"),
            ("shear",         "Shear"),
            ("dissipation",   "Dissipation"),
            ("TTTRESIDUAL",   "Residual"),
            ("VTTM",          "Tendency"),
        ],
    },
    "UU": {
        "title": "U-Wind Budget",
        "unit": "m / s / day",
        "terms": [
            ("turdiff",       "Turbulence (diff)"),
            ("turconv",       "Turbulence (conv)"),
            ("gwd-drag",      "GWD drag"),
            ("dyn",           "Dynamics"),
            ("TUUCOMPSUM",    "Sum of tendencies"),
            ("TUURESIDUAL",   "Residual"),
            ("VUUM",          "Tendency"),
        ],
    },
    "VV": {
        "title": "V-Wind Budget",
        "unit": "m / s / day",
        "terms": [
            ("turdiff",       "Turbulence (diff)"),
            ("turconv",       "Turbulence (conv)"),
            ("gwd-drag",      "GWD drag"),
            ("dyn",           "Dynamics"),
            ("TVVCOMPSUM",    "Sum of tendencies"),
            ("TVVRESIDUAL",   "Residual"),
            ("VVVM",          "Tendency"),
        ],
    },
}

EXP_LABELS = {
    "control": "Control (C1M)",
    "graupel": "Graupel (G1M)",
    "2mom":    "2-Moment (G2M)",
}

def read_dta(path: Path):
    if not path.is_file():
        return None, None
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def smart_xlim(all_values):
    """Symmetric x-limit based on the full data range."""
    if not all_values:
        return (-1, 1)
    combined = np.concatenate(all_values)
    combined = combined[np.isfinite(combined)]
    if len(combined) == 0:
        return (-1, 1)
    vmax = np.max(np.abs(combined))
    vmax *= 1.15
    if vmax == 0:
        vmax = 1.0
    return (-vmax, vmax)


def plot_variable(var_name, meta, data_root, experiments, out_dir, dpi):
    """Create a 1×N panel figure for *var_name*."""

    n_exp = len(experiments)
    terms = meta["terms"]
    unit = meta["unit"]
    n_terms = len(terms)

    # ── Determine legend rows to reserve space ───────────────────────────
    legend_ncol = min(3, n_terms)
    legend_rows = int(np.ceil(n_terms / legend_ncol))

    # ── Create figure with explicit room for title + legend ──────────────
    panel_w = 6.0
    panel_h = 9.0
    fig_w = panel_w * n_exp + 1.8          # extra for y-label + right margin
    fig_h = panel_h + 1.4 + legend_rows * 0.6  # title top + legend bottom

    fig, axes = plt.subplots(
        1, n_exp,
        figsize=(fig_w, fig_h),
        sharey=True,
        sharex=True,
    )
    if n_exp == 1:
        axes = [axes]

    # Reserve space: top for suptitle, bottom for legend + xlabel
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.91,
        bottom=0.07 + legend_rows * 0.04,
        wspace=0.07,
    )

    # ── First pass: collect all values for shared x-limits ───────────────
    # Exclude Tendency and Residual from xlim calculation (they can have extreme values)
    all_values = []
    for exp in experiments:
        exp_dir = data_root / exp / var_name
        for term_file, term_label in terms:
            # Skip Tendency and Residual when calculating limits
            if term_label == "Tendency" or "residual" in term_label.lower():
                continue
            _, values = read_dta(exp_dir / f"{term_file}.dta")
            if values is not None:
                all_values.append(values)
    xlims = smart_xlim(all_values)

    # ── Plot each experiment panel ───────────────────────────────────────
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax_idx, exp in enumerate(experiments):
        ax = axes[ax_idx]
        exp_dir = data_root / exp / var_name

        for term_file, term_label in terms:
            dta_path = exp_dir / f"{term_file}.dta"
            pressure, values = read_dta(dta_path)
            if pressure is None:
                continue

            pressure_hpa = np.abs(pressure)
            colour, lw, ls, alpha, zorder = get_line_style(term_label)

            ax.plot(
                values, pressure_hpa,
                color=colour, lw=lw, ls=ls, alpha=alpha,
                label=term_label, zorder=zorder,
            )

        # Zero reference
        ax.axvline(0, color="#AAAAAA", lw=0.6, zorder=1)

        # Axis limits
        ax.set_xlim(xlims)
        ax.set_ylim(1013, 1)

        # Panel title
        pl = panel_labels[ax_idx] if ax_idx < len(panel_labels) else ""
        ax.set_title(
            f"{pl}  {EXP_LABELS.get(exp, exp)}",
            fontsize=17, fontweight="bold", pad=12,
        )

        # Only leftmost panel gets y-label
        if ax_idx == 0:
            ax.set_ylabel("Pressure  (hPa)", fontsize=16, fontweight="bold")

        # Only centre panel gets x-label (shared axis)
        if ax_idx == n_exp // 2:
            ax.set_xlabel(unit, fontsize=16, fontweight="bold", labelpad=12)

        # Tick formatting
        ax.tick_params(axis="both", which="major", labelsize=14,
                       length=6, width=0.7, direction="out")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.tick_params(axis="both", which="minor",
                       length=2, width=0.3, direction="out")
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        # Subtle grid
        ax.grid(True, which="major", lw=0.3, color="#D5D5D5", zorder=0)
        ax.grid(True, which="minor", lw=0.15, color="#EDEDED", zorder=0)

        # Clean spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#999999")

    # ── Legend (below the panels, no overlap) ────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uh, ul = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uh.append(h)
            ul.append(l)

    fig.legend(
        uh, ul,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=13,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        borderpad=0.6,
        columnspacing=2.0,
        handlelength=2.5,
        handletextpad=0.8,
        bbox_to_anchor=(0.52, 0.0),
    )

    # ── Suptitle ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"{meta['title']}  —  2-Year Time Average (2014 – 2015)",
        fontsize=20, fontweight="bold", y=0.97,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = out_dir / f"{var_name}_budget_time_avg.png"
    fig.savefig(out_path, dpi=dpi, facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot time-averaged DDH 24h budgets."
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/Users/dev/ddhtoolbox/data/alaro-24h-budgets/time_average"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/Users/dev/ddhtoolbox/data/alaro-24h-budgets/time_average_plots"),
    )
    parser.add_argument("--experiments", nargs="+",
                        default=["control", "graupel", "2mom"])
    parser.add_argument("--variables", nargs="+",
                        default=list(VAR_META.keys()))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data root  : {args.data_root}")
    print(f"Output dir : {args.output_dir}")
    print(f"DPI        : {args.dpi}")
    print(f"Variables  : {args.variables}\n")

    for var in args.variables:
        if var not in VAR_META:
            print(f"  [SKIP] Unknown variable: {var}")
            continue
        print(f"Plotting {var} …")
        plot_variable(
            var, VAR_META[var],
            args.data_root, args.experiments,
            args.output_dir, args.dpi,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
