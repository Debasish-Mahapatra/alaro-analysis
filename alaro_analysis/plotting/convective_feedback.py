"""Convective feedback."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.analysis.convective_feedback import _compute_freezing_levels
from alaro_analysis.common.constants import EXPERIMENT_COLORS as EXP_COLORS
from alaro_analysis.common.convective_feedback_settings import EXPERIMENTS, THERMO_VARS


def plot_h1(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H1 plot: updraft intensity & flux vs total condensate loading.

    Two rows: (a) mean |omega| vs loading,  (b) mean updraft flux vs loading.
    Height-collapsed (marginal over all heights).
    Left column: regression, Right column: binned means.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)

    loading_coarse = np.array([0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01])

    for row, (target, ylabel) in enumerate(
        [
            ("intensity", r"Mean updraft intensity |$\omega$| (Pa/s)"),
            ("flux", r"Mean updraft flux (kg m$^{-2}$ s$^{-1}$)"),
        ]
    ):
        ax_reg = axes[row, 0]
        ax_bar = axes[row, 1]

        sums_key = f"sums_{target}"
        bar_width = 0.8 / len(experiments)
        n_coarse = len(loading_coarse) - 1
        x_positions = np.arange(n_coarse)

        for j, exp in enumerate(experiments):
            r = results[exp]
            fine_bins = r["x_bins"]
            s = np.nansum(r[sums_key], axis=1)
            c = np.nansum(r["counts"], axis=1)
            fine_centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])

            with np.errstate(invalid="ignore"):
                mean_y = np.where(c > 0, s / c, np.nan)

            # Regression
            sel = np.isfinite(mean_y) & (fine_centers > 0.0002) & (fine_centers < 0.008)
            x, y = fine_centers[sel], mean_y[sel]
            if len(x) >= 3:
                w = c[sel]
                slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
                ax_reg.scatter(x * 1000, y, s=12, color=EXP_COLORS[exp], alpha=0.5)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax_reg.plot(
                    x_fit * 1000,
                    intercept + slope * x_fit,
                    color=EXP_COLORS[exp],
                    linewidth=2.5,
                    label=f"{EXPERIMENTS[exp]}  slope={slope:.1f}",
                )

            # Binned means
            coarse_means = np.full(n_coarse, np.nan)
            for k in range(n_coarse):
                mask = (fine_centers >= loading_coarse[k]) & (fine_centers < loading_coarse[k + 1])
                ts, tc = s[mask].sum(), c[mask].sum()
                if tc > 0:
                    coarse_means[k] = ts / tc
            ax_bar.bar(
                x_positions + j * bar_width,
                coarse_means,
                width=bar_width,
                color=EXP_COLORS[exp],
                label=EXPERIMENTS[exp],
                edgecolor="k",
                linewidth=0.5,
            )

        ax_reg.set_xlabel("Total condensate (g/kg)", fontsize=14)
        ax_reg.set_ylabel(ylabel, fontsize=14)
        ax_reg.legend(fontsize=11, loc="upper left")
        ax_reg.grid(alpha=0.3)
        ax_reg.set_title(
            f"H1: {target.capitalize()} vs loading — regression",
            fontsize=16,
            fontweight="bold",
        )

        ax_bar.set_xticks(x_positions + bar_width * (len(experiments) - 1) / 2)
        ax_bar.set_xticklabels(
            [
                f"{loading_coarse[k] * 1000:.1f}–{loading_coarse[k + 1] * 1000:.1f}"
                for k in range(n_coarse)
            ],
            fontsize=11,
        )
        ax_bar.set_xlabel("Condensate loading bin (g/kg)", fontsize=14)
        ax_bar.set_ylabel(ylabel, fontsize=14)
        ax_bar.legend(fontsize=11)
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_title(
            f"H1: {target.capitalize()} vs loading — binned means",
            fontsize=16,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H1 figure: {output_path}")
    plt.close(fig)


def plot_h2(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H2 plot: downdraft diagnostics comparison.

    3 columns (flux, intensity, extent) × 2 rows:
      Top row: height profile (averaged over all hours)
      Bottom row: diurnal cycle (averaged over 0–6 km, the low-level downdraft layer)
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 11), squeeze=False)

    for col, (var, ylabel_prof, ylabel_diur) in enumerate(
        [
            (
                "flux",
                r"Downdraft flux (kg m$^{-2}$ s$^{-1}$)",
                r"Downdraft flux (kg m$^{-2}$ s$^{-1}$)",
            ),
            ("intensity", r"Downdraft intensity |$\omega$| (Pa/s)", r"|$\omega$| (Pa/s)"),
            ("extent", "Downdraft extent (fraction)", "Extent (fraction)"),
        ]
    ):
        ax_prof = axes[0, col]
        ax_diur = axes[1, col]

        for exp in experiments:
            r = results[exp]
            h_bins = r["h_bins"]
            h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
            s = r[f"{var}_sum"]  # (24, nh)
            c = r[f"{var}_cnt"]

            # Height profile: average over all hours
            s_all = np.nansum(s, axis=0)
            c_all = np.nansum(c, axis=0)
            with np.errstate(invalid="ignore"):
                profile = np.where(c_all > 0, s_all / c_all, np.nan)

            ax_prof.plot(
                profile,
                h_centers,
                color=EXP_COLORS[exp],
                linewidth=2.5,
                label=EXPERIMENTS[exp],
            )

            # Diurnal cycle: average over 0–6 km
            low_mask = h_centers <= 6.0
            s_low = np.nansum(s[:, low_mask], axis=1)
            c_low = np.nansum(c[:, low_mask], axis=1)
            with np.errstate(invalid="ignore"):
                diurnal = np.where(c_low > 0, s_low / c_low, np.nan)

            ax_diur.plot(
                np.arange(24),
                diurnal,
                color=EXP_COLORS[exp],
                linewidth=2.5,
                label=EXPERIMENTS[exp],
            )

        ax_prof.set_ylabel("Height (km)", fontsize=14)
        ax_prof.set_xlabel(ylabel_prof, fontsize=14)
        ax_prof.set_ylim(0, 18)
        ax_prof.legend(fontsize=12)
        ax_prof.grid(alpha=0.3)
        ax_prof.set_title(f"H2: Downdraft {var} profile", fontsize=16, fontweight="bold")

        ax_diur.set_xlabel("Hour (UTC)", fontsize=14)
        ax_diur.set_ylabel(ylabel_diur, fontsize=14)
        ax_diur.set_xlim(0, 23)
        ax_diur.set_xticks(np.arange(0, 24, 3))
        ax_diur.legend(fontsize=12)
        ax_diur.grid(alpha=0.3)
        ax_diur.set_title(
            f"H2: Downdraft {var} diurnal (0–6 km)",
            fontsize=16,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H2 figure: {output_path}")
    plt.close(fig)


def plot_h3(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H3 plot: pre-convection (06–09 LT) thermodynamic profiles.

    Top row: absolute profiles for each variable.
    Bottom row: anomalies relative to C1M.
    """
    variables = list(THERMO_VARS.keys())
    ncols = len(variables)
    fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 12), squeeze=False)

    # Compute profiles
    profiles = {}
    for exp in experiments:
        r = results[exp]
        h_bins = r["h_bins"]
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
        profiles[exp] = {}
        for v in variables:
            s = r["sums"][v]
            c = r["counts"][v]
            with np.errstate(invalid="ignore"):
                prof = np.where(c > 0, s / c, np.nan)
            cfg = THERMO_VARS[v]
            profiles[exp][v] = prof * cfg["scale"]

    ref_exp = experiments[0]  # control

    for col, var in enumerate(variables):
        cfg = THERMO_VARS[var]
        ax_abs = axes[0, col]
        ax_anom = axes[1, col]

        for exp in experiments:
            prof = profiles[exp][var]
            ax_abs.plot(
                prof,
                h_centers,
                color=EXP_COLORS[exp],
                linewidth=2.5,
                label=EXPERIMENTS[exp],
            )

            # Anomaly vs control
            if exp != ref_exp:
                ref = profiles[ref_exp][var]
                anom = prof - ref
                ax_anom.plot(
                    anom,
                    h_centers,
                    color=EXP_COLORS[exp],
                    linewidth=2.5,
                    label=f"{EXPERIMENTS[exp]} − C1M",
                )

        ax_abs.set_ylabel("Height (km)", fontsize=14)
        ax_abs.set_xlabel(f"{cfg['label']} ({cfg['unit']})", fontsize=14)
        ax_abs.set_ylim(0, 18)
        ax_abs.legend(fontsize=12)
        ax_abs.grid(alpha=0.3)
        ax_abs.set_title(
            f"H3: Pre-convection {cfg['label']}",
            fontsize=16,
            fontweight="bold",
        )

        ax_anom.set_ylabel("Height (km)", fontsize=14)
        ax_anom.set_xlabel(f"$\\Delta$ {cfg['label']} ({cfg['unit']})", fontsize=14)
        ax_anom.set_ylim(0, 18)
        ax_anom.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax_anom.legend(fontsize=12)
        ax_anom.grid(alpha=0.3)
        ax_anom.set_title(
            f"H3: {cfg['label']} anomaly vs C1M",
            fontsize=16,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H3 figure: {output_path}")
    plt.close(fig)


def _add_freezing_level(ax, freezing_levels: dict[str, float], experiments: list[str]):
    """Draw a horizontal dashed line for each experiment's freezing level."""
    for exp in experiments:
        if exp not in freezing_levels:
            continue
        fl = freezing_levels[exp]
        ax.axhline(
            fl,
            color=EXP_COLORS[exp],
            linewidth=1.5,
            linestyle=":",
            alpha=0.7,
        )
    # Single label annotation using the mean freezing level
    if freezing_levels:
        mean_fl = np.mean(list(freezing_levels.values()))
        ax.annotate(
            f"0 °C ≈ {mean_fl:.1f} km",
            xy=(1.0, mean_fl),
            xycoords=("axes fraction", "data"),
            fontsize=10,
            ha="right",
            va="bottom",
            color="0.3",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )


def plot_h4(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H4 plot: moisture flux divergence profiles.

    4 panels:
      (a) Wind divergence ∂u/∂x + ∂v/∂y — should be ~same for all schemes
      (b) Moisture profile q — differs between schemes
      (c) q × div — divergence acting on moisture (where schemes differ)
      (d) Full moisture flux divergence ∂(qu)/∂x + ∂(qv)/∂y

    Positive = divergence (moisture export), Negative = convergence (moisture import).
    """
    freezing_levels = _compute_freezing_levels(experiments)

    fig, axes = plt.subplots(1, 4, figsize=(28, 8), squeeze=False)

    panel_configs = [
        ("div", r"Wind divergence (s$^{-1}$)", "Wind divergence profile\n(same dynamics?)"),
        ("q", "Specific humidity (g/kg)", "Moisture profile\n(where is the moisture?)"),
        (
            "qdiv",
            r"q $\times$ div (kg kg$^{-1}$ s$^{-1}$)",
            "Divergence $\\times$ moisture\n(where is moisture exported?)",
        ),
        (
            "mfdiv",
            r"$\nabla \cdot (q\mathbf{v})$ (kg kg$^{-1}$ s$^{-1}$)",
            "Full moisture flux divergence",
        ),
    ]

    for col, (var, xlabel, title) in enumerate(panel_configs):
        ax = axes[0, col]

        for exp in experiments:
            r = results[exp]
            h_bins = r["h_bins"]
            h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

            s = r[f"{var}_sum"]
            c = r[f"{var}_cnt"]
            with np.errstate(invalid="ignore"):
                profile = np.where(c > 0, s / c, np.nan)

            # Scale q to g/kg for readability
            if var == "q":
                profile = profile * 1000.0

            ax.plot(
                profile,
                h_centers,
                color=EXP_COLORS[exp],
                linewidth=2.5,
                label=EXPERIMENTS[exp],
            )

        ax.set_ylabel("Height (km)", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, 18)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        # Vertical zero line for divergence panels
        if var != "q":
            ax.axvline(0, color="k", linewidth=0.8, linestyle="--")

        # Freezing level
        _add_freezing_level(ax, freezing_levels, experiments)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H4 figure: {output_path}")
    plt.close(fig)

    # Also plot anomalies relative to control
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), squeeze=False)
    ref_exp = experiments[0]

    for col, (var, xlabel, title) in enumerate(
        [
            ("q", "$\\Delta$ Specific humidity (g/kg)", "Moisture anomaly vs C1M"),
            (
                "qdiv",
                r"$\Delta$ q $\times$ div (kg kg$^{-1}$ s$^{-1}$)",
                "Divergence $\\times$ moisture\nanomaly vs C1M",
            ),
            (
                "mfdiv",
                r"$\Delta$ $\nabla \cdot (q\mathbf{v})$ (kg kg$^{-1}$ s$^{-1}$)",
                "Moisture flux divergence\nanomaly vs C1M",
            ),
        ]
    ):
        ax = axes2[0, col]
        r_ref = results[ref_exp]
        h_bins = r_ref["h_bins"]
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

        s_ref = r_ref[f"{var}_sum"]
        c_ref = r_ref[f"{var}_cnt"]
        with np.errstate(invalid="ignore"):
            prof_ref = np.where(c_ref > 0, s_ref / c_ref, np.nan)
        if var == "q":
            prof_ref = prof_ref * 1000.0

        for exp in experiments:
            if exp == ref_exp:
                continue
            r = results[exp]
            s = r[f"{var}_sum"]
            c = r[f"{var}_cnt"]
            with np.errstate(invalid="ignore"):
                prof = np.where(c > 0, s / c, np.nan)
            if var == "q":
                prof = prof * 1000.0

            anom = prof - prof_ref
            ax.plot(
                anom,
                h_centers,
                color=EXP_COLORS[exp],
                linewidth=2.5,
                label=f"{EXPERIMENTS[exp]} − C1M",
            )

        ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
        _add_freezing_level(ax, freezing_levels, experiments)
        ax.set_ylabel("Height (km)", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, 18)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    anom_path = output_path.with_name(output_path.stem + "_anomaly.png")
    fig2.tight_layout()
    fig2.savefig(anom_path, dpi=450, bbox_inches="tight")
    print(f"Saved H4 anomaly figure: {anom_path}")
    plt.close(fig2)
