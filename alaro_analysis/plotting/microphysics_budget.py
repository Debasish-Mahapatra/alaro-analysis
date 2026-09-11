"""Microphysics budget."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.microphysics_budget import (
    _mean_over_hours,
    _mean_scalar_over_hours,
)
from alaro_analysis.common.constants import EXPERIMENT_COLORS as EXP_COLORS
from alaro_analysis.common.microphysics_budget_settings import (
    B1_LEGEND_FONTSIZE,
    EXPERIMENTS,
    FLUX_FLOOR,
    HOUR_BAND_DEFAULT,
    HOUR_BAND_LABEL,
    PANEL_LABEL_FONTSIZE,
    PANEL_LABELS,
    PANEL_TITLE_FONTSIZE,
    SPECIES,
    SPECIES_COLOR,
    SPECIES_LABEL,
)


def _draw_freeze_line(ax, finals: dict[str, dict[str, np.ndarray]]):
    """Draw a dashed horizontal line at the per-experiment mean 0 C isotherm.

    Colored the same as each experiment so it's obvious which line belongs to
    which model; a single annotation labels them as "0 C isotherm".
    """
    for exp in finals:
        z0 = _mean_scalar_over_hours(finals[exp]["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(z0 / 1000.0, color=EXP_COLORS[exp], lw=1.0, ls="--", alpha=0.7, zorder=1)
    # Legend hint: add a neutral dashed line entry so readers can identify.
    ax.plot([], [], color="k", lw=1.0, ls="--", alpha=0.7, label=r"0 $^\circ$C isotherm")


def _set_panel_title(ax, title: str) -> None:
    ax.set_title(title, color="black", fontsize=PANEL_TITLE_FONTSIZE)


def _add_panel_label(ax, index: int) -> None:
    label = PANEL_LABELS[index] if index < len(PANEL_LABELS) else f"({index + 1})"
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.6,
            "boxstyle": "round,pad=0.18",
        },
        zorder=10,
    )


def plot_b1_species_fraction(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B1 — fractional storage partition by species."""
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 6.5), sharey=True, squeeze=False
    )
    axes = axes[0]

    for col, (ax, (exp, label)) in enumerate(zip(axes, EXPERIMENTS.items())):
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        total = _mean_over_hours(f["mass_per_level_total"], HOUR_BAND_DEFAULT)
        total_safe = np.where(total > 0, total, np.nan)

        # Build fractional columns per species; stacked-area-style plot.
        sp_fracs: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            prof = _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT)
            sp_fracs[sp] = prof / total_safe

        order = list(SPECIES)
        lower = np.zeros_like(z_km)
        for sp in order:
            upper = lower + np.nan_to_num(sp_fracs[sp], nan=0.0)
            ax.fill_betweenx(
                z_km,
                lower,
                upper,
                color=SPECIES_COLOR[sp],
                alpha=0.85,
                label=SPECIES_LABEL[sp],
            )
            lower = upper

        # 0 C isotherm for this experiment only (panel per experiment).
        z0 = _mean_scalar_over_hours(f["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(
                z0 / 1000.0, color="k", lw=1.2, ls="--", alpha=0.85, label=r"0 $^\circ$C isotherm"
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel("Fraction of total condensate mass")
        _set_panel_title(ax, label)
        _add_panel_label(ax, col)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=B1_LEGEND_FONTSIZE, framealpha=0.9)
    fig.suptitle(
        "Species partition of column condensate mass",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def _b1_global_mass_xlim(finals: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    """Shared x-axis limit for stacked species mass panels."""
    vmax = 0.0
    for exp in finals:
        f = finals[exp]
        stack_total: np.ndarray | None = None
        for sp in SPECIES:
            prof = _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT)
            prof = np.nan_to_num(prof, nan=0.0)
            stack_total = prof if stack_total is None else stack_total + prof
        if stack_total is not None and np.any(np.isfinite(stack_total)):
            vmax = max(vmax, float(np.nanmax(stack_total)))
    if vmax <= 0.0:
        vmax = 1.0
    return (0.0, vmax * 1.05)


def plot_b1_species_amount(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B1 — actual hydrometeor storage amount by species."""
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 6.5), sharey=True, squeeze=False
    )
    axes = axes[0]
    xlim = _b1_global_mass_xlim(finals)

    for col, (ax, (exp, label)) in enumerate(zip(axes, EXPERIMENTS.items())):
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0

        # Build absolute mass columns per species; stacked-area-style plot.
        sp_mass: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            sp_mass[sp] = _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT)

        order = list(SPECIES)
        lower = np.zeros_like(z_km)
        for sp in order:
            upper = lower + np.nan_to_num(sp_mass[sp], nan=0.0)
            ax.fill_betweenx(
                z_km,
                lower,
                upper,
                color=SPECIES_COLOR[sp],
                alpha=0.85,
                edgecolor="none",
                linewidth=0.0,
                label=SPECIES_LABEL[sp],
            )
            lower = upper

        # Draw the total-mass outline explicitly so the outer boundary is not
        # inherited from the top species color.
        ax.plot(lower, z_km, color="black", lw=1.2, alpha=0.95, zorder=4)

        # 0 C isotherm for this experiment only (panel per experiment).
        z0 = _mean_scalar_over_hours(f["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(
                z0 / 1000.0, color="k", lw=1.2, ls="--", alpha=0.85, label=r"0 $^\circ$C isotherm"
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel(r"Hydrometeor mass per model layer (kg m$^{-2}$)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        _set_panel_title(ax, label)
        _add_panel_label(ax, col)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=B1_LEGEND_FONTSIZE, framealpha=0.9)
    fig.suptitle(
        f"Hydrometeor mass by species ({HOUR_BAND_LABEL})",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b2_surface_reach(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B2 — surface-reach fraction vs hour (surface flux / max column flux)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        surf = f["surface_flux"]
        col = f["max_column_flux"]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(col > 0, surf / col, np.nan)
        ax.plot(np.arange(24), frac, color=EXP_COLORS[exp], lw=2, label=label)
    ax.set_xlim(0, 23)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Surface flux / max column flux")
    ax.grid(alpha=0.3)
    ax.legend()
    _set_panel_title(ax, "Fraction of precipitation flux reaching the surface")
    _add_panel_label(ax, 0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b3_residence_time(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B3 — per-level residence time tau(z) = mass_above(z) / flux(z)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    ax_total, ax_frozen = axes

    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        mass_tot = _mean_over_hours(f["mass_above_total"], HOUR_BAND_DEFAULT)
        flux = _mean_over_hours(f["flux_total"], HOUR_BAND_DEFAULT)
        flux_ok = flux > FLUX_FLOOR
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_total = np.where(flux_ok, mass_tot / flux, np.nan)

        # Frozen-species tau: use snow + graupel mass / total flux as proxy.
        mass_snow = _mean_over_hours(
            f.get("mass_above_SNOW", np.zeros_like(flux)), HOUR_BAND_DEFAULT
        )
        mass_graupel = _mean_over_hours(
            f.get("mass_above_GRAUPEL", np.zeros_like(flux)), HOUR_BAND_DEFAULT
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_snow = np.where(flux_ok, mass_snow / flux, np.nan)
            tau_graupel = np.where(flux_ok, mass_graupel / flux, np.nan)

        ax_total.plot(tau_total, z_km, color=EXP_COLORS[exp], lw=2, label=label)
        ax_frozen.plot(tau_snow, z_km, color=EXP_COLORS[exp], lw=2, ls="-", label=f"{label} snow")
        ax_frozen.plot(
            tau_graupel, z_km, color=EXP_COLORS[exp], lw=2, ls="--", label=f"{label} graupel"
        )

    for ax in axes:
        _draw_freeze_line(ax, finals)
        ax.set_xscale("log")
        ax.set_xlim(1e1, 1e6)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel(r"Residence time $\tau$ (s)")
        ax.grid(which="both", alpha=0.3)
    ax_total.set_ylabel("Height (km)")
    _set_panel_title(ax_total, "All condensate")
    _add_panel_label(ax_total, 0)
    _set_panel_title(ax_frozen, "Snow (solid) vs graupel (dashed)")
    _add_panel_label(ax_frozen, 1)
    ax_total.legend(loc="upper right")
    ax_frozen.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        r"Per-level condensate residence time $\tau(z) = m_{>z} / F(z)$",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b4_flux_divergence(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B4 — -dF/dz profile.  Positive = source, negative = sink (evap/sublim)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    ax_raw, ax_norm = axes

    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        fdiv = _mean_over_hours(f["flux_div"], HOUR_BAND_DEFAULT)
        surf = _mean_scalar_over_hours(f["surface_flux"], HOUR_BAND_DEFAULT)

        ax_raw.plot(fdiv * 1e6, z_km, color=EXP_COLORS[exp], lw=2, label=label)
        if np.isfinite(surf) and surf > 0:
            ax_norm.plot(fdiv / surf * 1000.0, z_km, color=EXP_COLORS[exp], lw=2, label=label)

    for ax in axes:
        _draw_freeze_line(ax, finals)
        ax.axvline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_ylim(0.0, 18.0)
        ax.grid(alpha=0.3)
    ax_raw.set_ylabel("Height (km)")
    ax_raw.set_xlabel(r"$-dF/dz$ ($10^{-6}$ kg m$^{-3}$ s$^{-1}$)")
    _set_panel_title(ax_raw, "Raw divergence")
    _add_panel_label(ax_raw, 0)
    ax_norm.set_xlabel(r"$-dF/dz \, / \, F_{\mathrm{surf}}$ ($10^{-3}$ m$^{-1}$)")
    _set_panel_title(ax_norm, "Normalized by surface flux")
    _add_panel_label(ax_norm, 1)
    ax_raw.legend(loc="upper right")
    fig.suptitle(
        "Precipitation flux divergence  |  positive = net source, negative = net sink",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b1b_mass_above_species(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """Companion to B1 — absolute column mass-above per species."""
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 6.5), sharey=True, squeeze=False
    )
    axes = axes[0]

    for col, (ax, (exp, label)) in enumerate(zip(axes, EXPERIMENTS.items())):
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        for sp in SPECIES:
            prof = _mean_over_hours(f[f"mass_above_{sp}"], HOUR_BAND_DEFAULT)
            ax.plot(prof * 1000.0, z_km, color=SPECIES_COLOR[sp], lw=2, label=SPECIES_LABEL[sp])

        z0 = _mean_scalar_over_hours(f["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(
                z0 / 1000.0, color="k", lw=1.2, ls="--", alpha=0.85, label=r"0 $^\circ$C isotherm"
            )

        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e2)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel(r"Mass above level (g m$^{-2}$)")
        _set_panel_title(ax, label)
        _add_panel_label(ax, col)
        ax.grid(which="both", alpha=0.3)

    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=B1_LEGEND_FONTSIZE, framealpha=0.9)
    fig.suptitle(
        "Integrated condensate mass above height, per species",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")
