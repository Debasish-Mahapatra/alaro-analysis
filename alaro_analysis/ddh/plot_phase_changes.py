"""DDH phase-change profiles by hydrometeor species.

This is a hydrometeor-budget companion to the QV condensation partition.  It
keeps the phase-change terms as species-specific profiles:

* QL/QI condensation/deposition sources from ``cond-cv`` and ``cond-rs``.
* QR/QS/QG evaporation/sublimation losses from ``evap-cv`` and ``evap-rs``.

The ALARO Q budgets aggregated here do not expose a separate melting block.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import (
    AGG_DIR,
    EXP_COLORS,
    EXPERIMENTS,
    FIG_DIR,
    Z_MAX_KM,
    freezing_level_km,
    load_temperature,
    tick_formatter,
)
from .plot_condensation_partition import column_integrate, normalize_lead
from .plot_style import (
    EXPERIMENT_COMPARISON_LINEWIDTH,
    EXPERIMENT_PANEL_FIGSIZE,
    FIGURE_TITLE_FONTSIZE,
    FREEZING_ALPHA,
    FREEZING_COLOR,
    FREEZING_LINESTYLE,
    FREEZING_LINEWIDTH,
    PANEL_GRID_ALPHA,
    PANEL_LEGEND_FONTSIZE,
    partition_line_style,
)


PROCESSED_DATA_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/ddh_phase_changes"
)
PLOT_DIR = FIG_DIR / "ddh_phase_changes"
DATA_TXT_SUBDIR = "data_txt"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
PANEL_TITLE_FONTSIZE = 16
PANEL_LABEL_FONTSIZE = 14

CONDENSATION_SPECS = (
    ("QL", "cloud_liquid", "Cloud liquid condensation"),
    ("QI", "cloud_ice", "Cloud ice deposition/condensation"),
)
EVAP_SUBLIM_SPECS = (
    ("QR", "rain", "Rain evaporation"),
    ("QS", "snow", "Snow sublimation"),
    ("QG", "graupel", "Graupel sublimation/evaporation"),
)
EVAPORATION_COMPONENT_SPECIES = ("rain", "snow", "graupel")
SUBLIMATION_COMPONENT_SPECIES = ("snow", "graupel")
TOTAL_EVAPORATION_SPECIES = "total_evaporation"
TOTAL_EVAPORATION_VARIABLE = "QR_QS_QG"
TOTAL_EVAPORATION_TITLE = "Total evaporation terms (rain + snow + graupel)"
TOTAL_SUBLIMATION_SPECIES = "total_sublimation"
TOTAL_SUBLIMATION_VARIABLE = "QS_QG"
TOTAL_SUBLIMATION_TITLE = "Total sublimation (snow + graupel)"


@dataclass(frozen=True)
class PhaseChangeProfile:
    experiment: str
    lead: str
    variable: str
    species: str
    process: str
    title: str
    altitude_km: np.ndarray
    convection_gkgday: np.ndarray
    resolved_gkgday: np.ndarray
    total_gkgday: np.ndarray
    n_days: int


def _positive_source(block: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(block, dtype=np.float64), 0.0)


def _positive_loss(block: np.ndarray) -> np.ndarray:
    return np.maximum(-np.asarray(block, dtype=np.float64), 0.0)


def compute_phase_profile(
    *,
    experiment: str,
    lead: str | int,
    variable: str,
    species: str,
    process: str,
    title: str,
    altitude_km: np.ndarray,
    convection_block: np.ndarray,
    resolved_block: np.ndarray,
    sign: str,
    n_days: int = 0,
) -> PhaseChangeProfile:
    """Compute positive phase-change rates from signed hydrometeor blocks."""
    z = np.asarray(altitude_km, dtype=np.float64)
    if sign == "source":
        convection = _positive_source(convection_block)
        resolved = _positive_source(resolved_block)
    elif sign == "loss":
        convection = _positive_loss(convection_block)
        resolved = _positive_loss(resolved_block)
    else:
        raise ValueError("sign must be 'source' or 'loss'")

    if convection.shape != resolved.shape or convection.shape != z.shape:
        raise ValueError(
            "altitude_km, convection_block, and resolved_block must match"
        )
    return PhaseChangeProfile(
        experiment=experiment,
        lead=normalize_lead(lead),
        variable=variable,
        species=species,
        process=process,
        title=title,
        altitude_km=z,
        convection_gkgday=convection,
        resolved_gkgday=resolved,
        total_gkgday=convection + resolved,
        n_days=int(n_days),
    )


def load_phase_profile(
    experiment: str,
    variable: str,
    species: str,
    process: str,
    title: str,
    lead: str | int = "0024",
    *,
    agg_dir: Path = AGG_DIR,
) -> PhaseChangeProfile | None:
    """Load one hydrometeor phase-change profile from aggregated DDH data."""
    lead_text = normalize_lead(lead)
    path = agg_dir / f"lead{lead_text}_VZ" / f"{experiment}_{variable}.npz"
    if not path.exists():
        return None

    if process == "condensation_deposition":
        cv_key, rs_key, sign = "block__cond-cv", "block__cond-rs", "source"
    elif process == "evaporation_sublimation":
        cv_key, rs_key, sign = "block__evap-cv", "block__evap-rs", "loss"
    else:
        raise ValueError(f"unknown process: {process}")

    with np.load(path, allow_pickle=True) as data:
        if cv_key not in data.files or rs_key not in data.files:
            return None
        n_days = int(data["days"].shape[0]) if "days" in data.files else 0
        return compute_phase_profile(
            experiment=experiment,
            lead=lead_text,
            variable=variable,
            species=species,
            process=process,
            title=title,
            altitude_km=data["altitude_km"],
            convection_block=data[cv_key],
            resolved_block=data[rs_key],
            sign=sign,
            n_days=n_days,
        )


def load_all_phase_profiles(
    lead: str | int = "0024",
    *,
    agg_dir: Path = AGG_DIR,
) -> list[PhaseChangeProfile]:
    """Load all available hydrometeor phase-change profiles."""
    profiles: list[PhaseChangeProfile] = []
    for exp in EXPERIMENTS:
        for variable, species, title in CONDENSATION_SPECS:
            profile = load_phase_profile(
                exp,
                variable,
                species,
                "condensation_deposition",
                title,
                lead,
                agg_dir=agg_dir,
            )
            if profile is not None:
                profiles.append(profile)
        for variable, species, title in EVAP_SUBLIM_SPECS:
            profile = load_phase_profile(
                exp,
                variable,
                species,
                "evaporation_sublimation",
                title,
                lead,
                agg_dir=agg_dir,
            )
            if profile is not None:
                profiles.append(profile)
    return profiles


def combine_total_sublimation_profiles(
    profiles: list[PhaseChangeProfile],
) -> list[PhaseChangeProfile]:
    """Combine snow and graupel sublimation into one profile per experiment."""
    combined: list[PhaseChangeProfile] = []
    for exp in EXPERIMENTS:
        parts = [
            p
            for p in profiles
            if p.experiment == exp
            and p.process == "evaporation_sublimation"
            and p.species in SUBLIMATION_COMPONENT_SPECIES
        ]
        if not parts:
            continue
        altitude = parts[0].altitude_km
        if any(p.altitude_km.shape != altitude.shape for p in parts):
            raise ValueError(f"sublimation profiles for {exp} use different grids")
        if any(not np.allclose(p.altitude_km, altitude) for p in parts):
            raise ValueError(f"sublimation profiles for {exp} use different altitudes")

        convection = np.sum([p.convection_gkgday for p in parts], axis=0)
        resolved = np.sum([p.resolved_gkgday for p in parts], axis=0)
        combined.append(
            PhaseChangeProfile(
                experiment=exp,
                lead=parts[0].lead,
                variable=TOTAL_SUBLIMATION_VARIABLE,
                species=TOTAL_SUBLIMATION_SPECIES,
                process="sublimation",
                title=TOTAL_SUBLIMATION_TITLE,
                altitude_km=altitude,
                convection_gkgday=convection,
                resolved_gkgday=resolved,
                total_gkgday=convection + resolved,
                n_days=parts[0].n_days,
            )
        )
    return combined


def combine_total_evaporation_profiles(
    profiles: list[PhaseChangeProfile],
) -> list[PhaseChangeProfile]:
    """Combine all hydrometeor evaporation terms into one profile per experiment."""
    combined: list[PhaseChangeProfile] = []
    for exp in EXPERIMENTS:
        parts = [
            p
            for p in profiles
            if p.experiment == exp
            and p.process == "evaporation_sublimation"
            and p.species in EVAPORATION_COMPONENT_SPECIES
        ]
        if not parts:
            continue
        altitude = parts[0].altitude_km
        if any(p.altitude_km.shape != altitude.shape for p in parts):
            raise ValueError(f"evaporation profiles for {exp} use different grids")
        if any(not np.allclose(p.altitude_km, altitude) for p in parts):
            raise ValueError(f"evaporation profiles for {exp} use different altitudes")

        convection = np.sum([p.convection_gkgday for p in parts], axis=0)
        resolved = np.sum([p.resolved_gkgday for p in parts], axis=0)
        combined.append(
            PhaseChangeProfile(
                experiment=exp,
                lead=parts[0].lead,
                variable=TOTAL_EVAPORATION_VARIABLE,
                species=TOTAL_EVAPORATION_SPECIES,
                process="evaporation",
                title=TOTAL_EVAPORATION_TITLE,
                altitude_km=altitude,
                convection_gkgday=convection,
                resolved_gkgday=resolved,
                total_gkgday=convection + resolved,
                n_days=parts[0].n_days,
            )
        )
    return combined


def common_positive_xlim(
    profiles: list[PhaseChangeProfile],
    *,
    pad_fraction: float = 0.06,
) -> tuple[float, float]:
    values = []
    for profile in profiles:
        for arr in (
            profile.convection_gkgday,
            profile.resolved_gkgday,
            profile.total_gkgday,
        ):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                values.append(finite)
    if not values:
        return (0.0, 1.0)
    vmax = float(np.max(np.concatenate(values)))
    if not np.isfinite(vmax) or vmax <= 0:
        return (0.0, 1.0)
    return (0.0, vmax * (1.0 + pad_fraction))


def _txt_path_for_plot(path: Path) -> Path:
    return path.parent / DATA_TXT_SUBDIR / f"{path.stem}.txt"


def _format_txt_value(value: object) -> str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.10g}"


def _write_headline(fh, title: str, plot_path: Path, notes: tuple[str, ...] = ()) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'=' * len(title)}\n")
    fh.write(f"Source plot: {plot_path}\n")
    for note in notes:
        fh.write(f"{note}\n")
    fh.write("\n")


def _write_csv_section(
    fh,
    title: str,
    columns: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'-' * len(title)}\n")
    fh.write(",".join(columns) + "\n")
    for row in rows:
        fh.write(",".join(_format_txt_value(value) for value in row) + "\n")
    fh.write("\n")


def _experiment_freezing_rows(experiments: list[str] | None = None) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    exp_names = experiments if experiments is not None else list(EXPERIMENTS)
    for exp in exp_names:
        temp = load_temperature(exp)
        rows.append(
            (
                exp,
                EXPERIMENTS.get(exp, exp),
                freezing_level_km(temp),
            )
        )
    return rows


def _write_freezing_section(fh, experiments: list[str] | None = None) -> None:
    _write_csv_section(
        fh,
        "Freezing-level data",
        ("experiment", "label", "z_freeze_km"),
        _experiment_freezing_rows(experiments),
    )


def write_species_panels_txt(
    profiles: list[PhaseChangeProfile],
    process: str,
    species: str,
    path: Path,
) -> Path | None:
    selected = [p for p in profiles if p.process == process and p.species == species]
    if not selected:
        return None

    by_exp = {p.experiment: p for p in selected}
    profile0 = selected[0]
    xlim = common_positive_xlim(selected)
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_headline(
            fh,
            f"DDH Phase-Change Plot Data: {path.stem}",
            path,
            (
                f"Lead: {profile0.lead}",
                f"Process: {process}",
                f"Species/category: {species}",
                f"Variable: {profile0.variable}",
                f"Profile title from metadata: {profile0.title}",
                "Units: g kg-1 day-1.",
                f"Shared x-axis limits: {_format_txt_value(xlim[0])}, {_format_txt_value(xlim[1])}.",
            ),
        )
        _write_freezing_section(fh)
        for exp, exp_label in EXPERIMENTS.items():
            profile = by_exp.get(exp)
            if profile is None:
                _write_csv_section(
                    fh,
                    f"{exp_label} profile data",
                    ("missing_profile",),
                    [("No budget",)],
                )
                continue
            rows = [
                (
                    profile.altitude_km[i],
                    profile.convection_gkgday[i],
                    profile.resolved_gkgday[i],
                    profile.total_gkgday[i],
                    profile.n_days,
                )
                for i in range(profile.altitude_km.size)
            ]
            _write_csv_section(
                fh,
                f"{exp_label} profile data",
                (
                    "altitude_km",
                    "convection_gkgday",
                    "resolved_gkgday",
                    "total_gkgday",
                    "n_days",
                ),
                rows,
            )
    return txt_path


def write_experiment_comparison_txt(
    profiles: list[PhaseChangeProfile],
    path: Path,
    *,
    title: str,
    x_label: str,
) -> Path | None:
    if not profiles:
        return None

    by_exp = {p.experiment: p for p in profiles}
    xlim = common_positive_xlim(profiles)
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_headline(
            fh,
            f"DDH Phase-Change Plot Data: {path.stem}",
            path,
            (
                title,
                f"X variable: {x_label}",
                "The comparison figure plots the total_gkgday profile for each experiment.",
                "Units: g kg-1 day-1.",
                f"Shared x-axis limits: {_format_txt_value(xlim[0])}, {_format_txt_value(xlim[1])}.",
            ),
        )
        _write_freezing_section(fh, [p.experiment for p in profiles])
        for exp, exp_label in EXPERIMENTS.items():
            profile = by_exp.get(exp)
            if profile is None:
                continue
            rows = [
                (
                    profile.altitude_km[i],
                    profile.total_gkgday[i],
                    profile.convection_gkgday[i],
                    profile.resolved_gkgday[i],
                    profile.n_days,
                )
                for i in range(profile.altitude_km.size)
            ]
            _write_csv_section(
                fh,
                f"{exp_label} comparison data",
                (
                    "altitude_km",
                    "total_gkgday",
                    "convection_gkgday",
                    "resolved_gkgday",
                    "n_days",
                ),
                rows,
            )
    return txt_path


def _draw_freezing_line(ax, experiment: str) -> None:
    temp = load_temperature(experiment)
    z0 = freezing_level_km(temp)
    if np.isfinite(z0):
        ax.axhline(
            z0,
            color=FREEZING_COLOR,
            lw=FREEZING_LINEWIDTH,
            ls=FREEZING_LINESTYLE,
            alpha=FREEZING_ALPHA,
            label=r"0 $^{\circ}$C isotherm",
            zorder=1,
        )


def _legend_if_any(ax) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=PANEL_LEGEND_FONTSIZE)


def plot_species_panels(
    profiles: list[PhaseChangeProfile],
    process: str,
    species: str,
    path: Path,
    *,
    show_titles: bool = False,
) -> Path | None:
    """Plot one phase-change species/category with experiment panels."""
    selected = [p for p in profiles if p.process == process and p.species == species]
    if not selected:
        return None

    title = selected[0].title
    xlim = common_positive_xlim(selected)
    process_name = (
        "Condensation"
        if process == "condensation_deposition"
        else "Evaporation"
    )
    process_label = {
        "condensation_deposition": "condensation/deposition",
        "evaporation_sublimation": "evaporation/sublimation",
        "evaporation": "total evaporation",
        "sublimation": "total sublimation",
    }.get(process, process.replace("_", " "))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(EXPERIMENTS),
        figsize=EXPERIMENT_PANEL_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    if len(EXPERIMENTS) == 1:
        axes = [axes]
    by_exp = {p.experiment: p for p in selected}

    for col, (exp, exp_label) in enumerate(EXPERIMENTS.items()):
        ax = axes[col]
        profile = by_exp.get(exp)
        ax.axvline(0.0, color="k", lw=0.7, alpha=0.65)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, Z_MAX_KM)
        ax.grid(alpha=PANEL_GRID_ALPHA)
        ax.xaxis.set_major_formatter(tick_formatter())
        panel_prefix = PANEL_LABELS[col] if col < len(PANEL_LABELS) else f"({col + 1})"
        ax.set_title(exp_label, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        ax.text(
            0.03,
            0.96,
            panel_prefix,
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
        ax.set_xlabel(r"Positive rate (g kg$^{-1}$ day$^{-1}$)")
        if col == 0:
            ax.set_ylabel("Altitude (km)")

        if profile is None:
            ax.text(
                0.5,
                0.5,
                "No budget",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            _draw_freezing_line(ax, exp)
            _legend_if_any(ax)
            continue
        line_color, lw, ls, alpha, zorder = partition_line_style(
            process_name,
            "total",
        )
        ax.plot(
            profile.total_gkgday,
            profile.altitude_km,
            color=line_color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Total",
        )
        line_color, lw, ls, alpha, zorder = partition_line_style(
            process_name,
            "convection",
        )
        ax.plot(
            profile.convection_gkgday,
            profile.altitude_km,
            color=line_color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Convection",
        )
        line_color, lw, ls, alpha, zorder = partition_line_style(
            process_name,
            "resolved",
        )
        ax.plot(
            profile.resolved_gkgday,
            profile.altitude_km,
            color=line_color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Resolved",
        )
        _draw_freezing_line(ax, exp)
        _legend_if_any(ax)

    if show_titles:
        fig.suptitle(
            f"DDH {process_label} by hydrometeor: {title}",
            fontsize=FIGURE_TITLE_FONTSIZE,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_total_sublimation_experiment_comparison(
    profiles: list[PhaseChangeProfile],
    path: Path,
    *,
    show_titles: bool = False,
) -> Path | None:
    """Plot total sublimation profiles for all experiments on one axis."""
    selected = [
        p
        for p in profiles
        if p.process == "sublimation" and p.species == TOTAL_SUBLIMATION_SPECIES
    ]
    if not selected:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, EXPERIMENT_PANEL_FIGSIZE[1]))
    xlim = common_positive_xlim(selected)

    for exp, exp_label in EXPERIMENTS.items():
        profile = next((p for p in selected if p.experiment == exp), None)
        if profile is None:
            continue
        ax.plot(
            profile.total_gkgday,
            profile.altitude_km,
            color=EXP_COLORS.get(exp, "black"),
            lw=EXPERIMENT_COMPARISON_LINEWIDTH,
            label=exp_label,
        )

    ax.axvline(0.0, color="k", lw=0.7, alpha=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, Z_MAX_KM)
    ax.grid(alpha=PANEL_GRID_ALPHA)
    ax.xaxis.set_major_formatter(tick_formatter())
    ax.set_xlabel(r"Total sublimation rate (g kg$^{-1}$ day$^{-1}$)")
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="best", fontsize=PANEL_LEGEND_FONTSIZE)
    if show_titles:
        fig.suptitle(
            "DDH total sublimation by experiment",
            fontsize=FIGURE_TITLE_FONTSIZE,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_total_evaporation_experiment_comparison(
    profiles: list[PhaseChangeProfile],
    path: Path,
    *,
    show_titles: bool = False,
) -> Path | None:
    """Plot total rain evaporation profiles for all experiments on one axis."""
    selected = [
        p
        for p in profiles
        if p.process == "evaporation" and p.species == TOTAL_EVAPORATION_SPECIES
    ]
    if not selected:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, EXPERIMENT_PANEL_FIGSIZE[1]))
    xlim = common_positive_xlim(selected)

    for exp, exp_label in EXPERIMENTS.items():
        profile = next((p for p in selected if p.experiment == exp), None)
        if profile is None:
            continue
        ax.plot(
            profile.total_gkgday,
            profile.altitude_km,
            color=EXP_COLORS.get(exp, "black"),
            lw=EXPERIMENT_COMPARISON_LINEWIDTH,
            label=exp_label,
        )

    ax.axvline(0.0, color="k", lw=0.7, alpha=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, Z_MAX_KM)
    ax.grid(alpha=PANEL_GRID_ALPHA)
    ax.xaxis.set_major_formatter(tick_formatter())
    ax.set_xlabel(r"Total evaporation rate (g kg$^{-1}$ day$^{-1}$)")
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="best", fontsize=PANEL_LEGEND_FONTSIZE)
    if show_titles:
        fig.suptitle(
            "DDH total evaporation by experiment",
            fontsize=FIGURE_TITLE_FONTSIZE,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def summarize_profile(profile: PhaseChangeProfile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pathway, arr in (
        ("convection", profile.convection_gkgday),
        ("resolved", profile.resolved_gkgday),
        ("total", profile.total_gkgday),
    ):
        if np.any(np.isfinite(arr)):
            peak_idx = int(np.nanargmax(arr))
            peak = float(arr[peak_idx])
            peak_z = float(profile.altitude_km[peak_idx])
        else:
            peak = float("nan")
            peak_z = float("nan")
        rows.append(
            {
                "experiment": profile.experiment,
                "lead": profile.lead,
                "variable": profile.variable,
                "species": profile.species,
                "process": profile.process,
                "pathway": pathway,
                "n_days": profile.n_days,
                "column_gkgday_km": column_integrate(arr, profile.altitude_km),
                "peak_gkgday": peak,
                "peak_altitude_km": peak_z,
            }
        )
    return rows


def write_summary_csv(profiles: list[PhaseChangeProfile], path: Path) -> None:
    fields = [
        "experiment",
        "lead",
        "variable",
        "species",
        "process",
        "pathway",
        "n_days",
        "column_gkgday_km",
        "peak_gkgday",
        "peak_altitude_km",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            for row in summarize_profile(profile):
                writer.writerow({k: _format_csv_value(v) for k, v in row.items()})


def write_by_level_csv(profiles: list[PhaseChangeProfile], path: Path) -> None:
    fields = [
        "experiment",
        "lead",
        "variable",
        "species",
        "process",
        "pathway",
        "altitude_km",
        "rate_gkgday",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            for pathway, arr in (
                ("convection", profile.convection_gkgday),
                ("resolved", profile.resolved_gkgday),
                ("total", profile.total_gkgday),
            ):
                for i, z in enumerate(profile.altitude_km):
                    row = {
                        "experiment": profile.experiment,
                        "lead": profile.lead,
                        "variable": profile.variable,
                        "species": profile.species,
                        "process": profile.process,
                        "pathway": pathway,
                        "altitude_km": float(z),
                        "rate_gkgday": float(arr[i]),
                    }
                    writer.writerow({k: _format_csv_value(v) for k, v in row.items()})


def save_npz(profiles: list[PhaseChangeProfile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for profile in profiles:
        prefix = f"{profile.experiment}_{profile.variable}_{profile.process}"
        payload[f"{prefix}_altitude_km"] = profile.altitude_km
        payload[f"{prefix}_convection_gkgday"] = profile.convection_gkgday
        payload[f"{prefix}_resolved_gkgday"] = profile.resolved_gkgday
        payload[f"{prefix}_total_gkgday"] = profile.total_gkgday
        payload[f"{prefix}_n_days"] = np.array(profile.n_days, dtype=np.int64)
    payload["profile_keys"] = np.array(
        [f"{p.experiment}_{p.variable}_{p.process}" for p in profiles], dtype="U64"
    )
    np.savez_compressed(path, **payload)


def write_availability_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "No explicit melting phase-change block was found in the aggregated "
        "ALARO DDH QV/QL/QI/QR/QS/QG budgets. No explicit hydrometeor "
        "freezing block was found in these processed aggregates either; "
        "the ddhtoolbox conversion table lists temperature freezing "
        "diagnostics such as FCTSFR/FCTCFRZ, but they are not present in "
        "the current lead0024_VZ aggregates. Available phase-change blocks "
        "here are QL/QI cond-cv/cond-rs and QR/QS/QG evap-cv/evap-rs.\n"
    )


def run(
    *,
    lead: str | int = "0024",
    agg_dir: Path = AGG_DIR,
    plot_dir: Path = PLOT_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    show_titles: bool = False,
    write_txt: bool = True,
) -> dict[str, Path]:
    lead_text = normalize_lead(lead)
    profiles = load_all_phase_profiles(lead_text, agg_dir=agg_dir)
    total_evaporation_profiles = combine_total_evaporation_profiles(profiles)
    total_sublimation_profiles = combine_total_sublimation_profiles(profiles)
    output_profiles = profiles + total_evaporation_profiles + total_sublimation_profiles

    outputs: dict[str, Path] = {}
    for variable, species, _title in CONDENSATION_SPECS:
        fig_path = plot_species_panels(
            profiles,
            "condensation_deposition",
            species,
            plot_dir / f"{species}_condensation_deposition_lead{lead_text}.png",
            show_titles=show_titles,
        )
        if fig_path is not None:
            outputs[f"{variable.lower()}_condensation_deposition_figure"] = fig_path
            if write_txt:
                txt_path = write_species_panels_txt(
                    profiles,
                    "condensation_deposition",
                    species,
                    fig_path,
                )
                if txt_path is not None:
                    outputs[f"{variable.lower()}_condensation_deposition_txt"] = txt_path
    for variable, species, _title in EVAP_SUBLIM_SPECS:
        fig_path = plot_species_panels(
            profiles,
            "evaporation_sublimation",
            species,
            plot_dir / f"{species}_evaporation_sublimation_lead{lead_text}.png",
            show_titles=show_titles,
        )
        if fig_path is not None:
            outputs[f"{variable.lower()}_evaporation_sublimation_figure"] = fig_path
            if write_txt:
                txt_path = write_species_panels_txt(
                    profiles,
                    "evaporation_sublimation",
                    species,
                    fig_path,
                )
                if txt_path is not None:
                    outputs[f"{variable.lower()}_evaporation_sublimation_txt"] = txt_path
    total_evaporation_path = plot_species_panels(
        output_profiles,
        "evaporation",
        TOTAL_EVAPORATION_SPECIES,
        plot_dir / f"total_evaporation_partition_lead{lead_text}.png",
        show_titles=show_titles,
    )
    if total_evaporation_path is not None:
        outputs["total_evaporation_figure"] = total_evaporation_path
        if write_txt:
            txt_path = write_species_panels_txt(
                output_profiles,
                "evaporation",
                TOTAL_EVAPORATION_SPECIES,
                total_evaporation_path,
            )
            if txt_path is not None:
                outputs["total_evaporation_txt"] = txt_path
    total_evaporation_comparison_path = plot_total_evaporation_experiment_comparison(
        total_evaporation_profiles,
        plot_dir / f"total_evaporation_by_experiment_lead{lead_text}.png",
        show_titles=show_titles,
    )
    if total_evaporation_comparison_path is not None:
        outputs["total_evaporation_by_experiment_figure"] = (
            total_evaporation_comparison_path
        )
        if write_txt:
            txt_path = write_experiment_comparison_txt(
                total_evaporation_profiles,
                total_evaporation_comparison_path,
                title="DDH total evaporation by experiment",
                x_label="Total evaporation rate (g kg-1 day-1)",
            )
            if txt_path is not None:
                outputs["total_evaporation_by_experiment_txt"] = txt_path
    total_sublimation_path = plot_species_panels(
        output_profiles,
        "sublimation",
        TOTAL_SUBLIMATION_SPECIES,
        plot_dir / f"total_sublimation_partition_lead{lead_text}.png",
        show_titles=show_titles,
    )
    if total_sublimation_path is not None:
        outputs["total_sublimation_figure"] = total_sublimation_path
        if write_txt:
            txt_path = write_species_panels_txt(
                output_profiles,
                "sublimation",
                TOTAL_SUBLIMATION_SPECIES,
                total_sublimation_path,
            )
            if txt_path is not None:
                outputs["total_sublimation_txt"] = txt_path
    total_sublimation_comparison_path = plot_total_sublimation_experiment_comparison(
        total_sublimation_profiles,
        plot_dir / f"total_sublimation_by_experiment_lead{lead_text}.png",
        show_titles=show_titles,
    )
    if total_sublimation_comparison_path is not None:
        outputs["total_sublimation_by_experiment_figure"] = (
            total_sublimation_comparison_path
        )
        if write_txt:
            txt_path = write_experiment_comparison_txt(
                total_sublimation_profiles,
                total_sublimation_comparison_path,
                title="DDH total sublimation by experiment",
                x_label="Total sublimation rate (g kg-1 day-1)",
            )
            if txt_path is not None:
                outputs["total_sublimation_by_experiment_txt"] = txt_path

    npz_path = processed_dir / f"phase_changes_by_hydrometeor_lead{lead_text}.npz"
    by_level_csv = processed_dir / "analytics" / f"phase_changes_by_hydrometeor_lead{lead_text}_by_level.csv"
    summary_csv = processed_dir / "analytics" / f"phase_changes_by_hydrometeor_lead{lead_text}_summary.csv"
    note_path = processed_dir / "analytics" / f"phase_changes_by_hydrometeor_lead{lead_text}_availability.txt"
    save_npz(output_profiles, npz_path)
    write_by_level_csv(output_profiles, by_level_csv)
    write_summary_csv(output_profiles, summary_csv)
    write_availability_note(note_path)
    outputs["npz"] = npz_path
    outputs["by_level_csv"] = by_level_csv
    outputs["summary_csv"] = summary_csv
    outputs["availability_note"] = note_path

    for key, path in outputs.items():
        print(f"{key}: {path}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lead", default="0024", help="Forecast lead, e.g. 0024.")
    parser.add_argument("--agg-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--with-titles",
        action="store_true",
        help="Draw figure and panel titles. By default titles are omitted.",
    )
    parser.add_argument(
        "--no-write-txt",
        action="store_true",
        help="Do not write matching data_txt/*.txt files.",
    )
    args = parser.parse_args()
    run(
        lead=args.lead,
        agg_dir=args.agg_dir,
        plot_dir=args.plot_dir,
        processed_dir=args.processed_dir,
        show_titles=args.with_titles,
        write_txt=not args.no_write_txt,
    )


if __name__ == "__main__":
    main()
