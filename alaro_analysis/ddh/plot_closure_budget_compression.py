"""Closure-budget compression figure for C1M, G1M, and G2M.

The figure combines DDH condensation profiles with resolved updraft mass-flux
profiles to show the low-level condensation/updraft asymmetry after adding
graupel.  It writes both the PNG and the text data used for the plot.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from alaro_analysis.ddh.io import AGG_DIR, EXPERIMENTS
from alaro_analysis.ddh.plot_warm_layer_pathway_summary import compute_layer_metrics


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "DDH-closure_budget_compression"
)
DEFAULT_CONDENSATION_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/ddh_condensation"
)
DEFAULT_UPDRAFT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/updraft_flux/2years"
)
DEFAULT_HEIGHT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/geopotential/2years"
)
DATA_TXT_SUBDIR = "data_txt"
FIGURE_NAME = "closure_budget_compression.png"
TEXT_NAME = "closure_budget_compression.txt"

EXPERIMENT_ORDER = ("control", "graupel", "2mom")
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")

TOTAL_COLOR = "#111111"
CONVECTIVE_COLOR = "#D62728"
RESOLVED_COLOR = "#1F77B4"
UPDRAFT_COLOR = "#009E73"
PEAK_COLOR = "#4D4D4D"
FREEZING_COLOR = "#555555"

BAR_STYLES = {
    "C1M": {"facecolor": "#4C78A8", "edgecolor": "#222222"},
    "G1M": {"facecolor": "#E45756", "edgecolor": "#222222"},
    "G2M": {"facecolor": "#54A24B", "edgecolor": "#222222"},
}


@dataclass(frozen=True)
class CondensationProfile:
    experiment: str
    label: str
    altitude_km: np.ndarray
    convective_gkgday: np.ndarray
    resolved_gkgday: np.ndarray
    total_gkgday: np.ndarray


@dataclass(frozen=True)
class UpdraftProfile:
    experiment: str
    label: str
    height_km: np.ndarray
    mean_flux: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class SummaryMetrics:
    experiment: str
    label: str
    column_total_gkgday_km: float
    condensation_peak_height_km: float
    column_total_divided_by_peak_height_gkgday: float
    warm_reservoir_0_3km_gkg_km: float
    convective_condensation_0_3km_gkgday_km: float
    mean_updraft_flux_0_3km_kg_m2_s: float
    convective_condensation_divided_by_flux: float

    @property
    def condensation_per_flux_x10minus3(self) -> float:
        return self.convective_condensation_divided_by_flux / 1000.0


def normalize_lead(lead: str | int) -> str:
    text = str(lead).strip()
    if text.startswith("+"):
        text = text[1:]
    if not text.isdigit():
        raise ValueError(f"lead must be numeric, got {lead!r}")
    return f"{int(text):04d}"


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _label(experiment: str) -> str:
    return EXPERIMENTS.get(experiment, experiment)


def condensation_npz_path(condensation_dir: Path, lead: str) -> Path:
    return condensation_dir / f"condensation_partition_profile_lead{lead}.npz"


def updraft_profile_path(updraft_dir: Path, experiment: str) -> Path:
    return updraft_dir / f"{experiment}_full-domain_diurnal_profile.npz"


def height_profile_path(height_dir: Path, experiment: str) -> Path:
    return height_dir / f"{experiment}_full-domain_height_profile_first.npz"


def load_condensation_profiles(
    condensation_dir: Path,
    lead: str,
    experiments: Sequence[str] = EXPERIMENT_ORDER,
) -> dict[str, CondensationProfile]:
    path = condensation_npz_path(condensation_dir, lead)
    if not path.exists():
        raise FileNotFoundError(f"Missing processed condensation file: {path}")

    out: dict[str, CondensationProfile] = {}
    with np.load(path, allow_pickle=True) as data:
        for experiment in experiments:
            out[experiment] = CondensationProfile(
                experiment=experiment,
                label=_label(experiment),
                altitude_km=np.asarray(data[f"{experiment}_altitude_km"], dtype=float),
                convective_gkgday=np.asarray(
                    data[f"{experiment}_convective_gkgday"], dtype=float
                ),
                resolved_gkgday=np.asarray(
                    data[f"{experiment}_resolved_gkgday"], dtype=float
                ),
                total_gkgday=np.asarray(data[f"{experiment}_total_gkgday"], dtype=float),
            )
    return out


def load_updraft_profiles(
    updraft_dir: Path,
    height_dir: Path,
    experiments: Sequence[str] = EXPERIMENT_ORDER,
) -> dict[str, UpdraftProfile]:
    out: dict[str, UpdraftProfile] = {}
    for experiment in experiments:
        updraft_path = updraft_profile_path(updraft_dir, experiment)
        height_path = height_profile_path(height_dir, experiment)
        if not updraft_path.exists():
            raise FileNotFoundError(f"Missing processed updraft file: {updraft_path}")
        if not height_path.exists():
            raise FileNotFoundError(f"Missing height profile file: {height_path}")

        with np.load(updraft_path, allow_pickle=True) as data:
            mean = np.asarray(data["mean"], dtype=float)
            counts = np.asarray(data["counts"], dtype=float)
        with np.load(height_path, allow_pickle=True) as data:
            height_km = np.asarray(data["height_m"], dtype=float) / 1000.0

        n_levels = min(mean.shape[0], counts.shape[0], height_km.size)
        mean = mean[:n_levels, :]
        counts = counts[:n_levels, :]
        weighted_sum = np.nansum(mean * counts, axis=1)
        count_sum = np.nansum(counts, axis=1)
        mean_flux = np.divide(
            weighted_sum,
            count_sum,
            out=np.full(n_levels, np.nan, dtype=float),
            where=count_sum > 0,
        )
        order = np.argsort(height_km[:n_levels])
        out[experiment] = UpdraftProfile(
            experiment=experiment,
            label=_label(experiment),
            height_km=height_km[:n_levels][order],
            mean_flux=mean_flux[order],
            count=count_sum[order],
        )
    return out


def freezing_level_from_temperature(path: Path) -> float:
    if not path.exists():
        return float("nan")
    with np.load(path, allow_pickle=True) as data:
        altitude = np.asarray(data["altitude_km"], dtype=float)
        temperature = np.asarray(data["temperature_k"], dtype=float)

    finite = np.isfinite(altitude) & np.isfinite(temperature)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    altitude = altitude[finite]
    temperature = temperature[finite]
    order = np.argsort(altitude)
    altitude = altitude[order]
    temperature = temperature[order]

    diff = temperature - 273.15
    crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossings.size == 0:
        return float("nan")
    idx = int(crossings[0])
    weight = diff[idx] / (diff[idx] - diff[idx + 1])
    return float(altitude[idx] + weight * (altitude[idx + 1] - altitude[idx]))


def column_integral(altitude_km: np.ndarray, values: np.ndarray) -> float:
    altitude = np.asarray(altitude_km, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(altitude) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    altitude = altitude[finite]
    values = values[finite]
    order = np.argsort(altitude)
    return float(np.trapezoid(values[order], altitude[order]))


def layer_integral(
    altitude_km: np.ndarray,
    values: np.ndarray,
    bottom_km: float = 0.0,
    top_km: float = 3.0,
) -> float:
    altitude = np.asarray(altitude_km, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(altitude) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    altitude = altitude[finite]
    values = values[finite]
    order = np.argsort(altitude)
    altitude = altitude[order]
    values = values[order]

    if top_km <= bottom_km or top_km < altitude[0] or bottom_km > altitude[-1]:
        return float("nan")
    inside = (altitude > bottom_km) & (altitude < top_km)
    layer_altitude = np.r_[bottom_km, altitude[inside], top_km]
    layer_values = np.interp(layer_altitude, altitude, values)
    return float(np.trapezoid(layer_values, layer_altitude))


def mean_between_heights(
    height_km: np.ndarray,
    values: np.ndarray,
    bottom_km: float = 0.0,
    top_km: float = 3.0,
) -> float:
    height = np.asarray(height_km, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = (
        np.isfinite(height)
        & np.isfinite(values)
        & (height >= bottom_km)
        & (height <= top_km)
    )
    if not np.any(keep):
        return float("nan")
    return float(np.mean(values[keep]))


def warm_reservoir_by_experiment(agg_dir: Path, lead: str) -> dict[str, float]:
    rows = compute_layer_metrics(agg_dir=agg_dir, lead=lead, experiments=EXPERIMENT_ORDER)
    out: dict[str, float] = {}
    for row in rows:
        if row.layer == "0-3 km":
            out[row.experiment] = row.warm_liquid_rain_amount
    return out


def build_summary_metrics(
    condensation: dict[str, CondensationProfile],
    updraft: dict[str, UpdraftProfile],
    *,
    agg_dir: Path,
    lead: str,
    experiments: Sequence[str] = EXPERIMENT_ORDER,
) -> dict[str, SummaryMetrics]:
    warm = warm_reservoir_by_experiment(agg_dir, lead)
    out: dict[str, SummaryMetrics] = {}
    for experiment in experiments:
        cond = condensation[experiment]
        flux = updraft[experiment]
        column_total = column_integral(cond.altitude_km, cond.total_gkgday)
        peak_idx = int(np.nanargmax(cond.total_gkgday))
        peak_height = float(cond.altitude_km[peak_idx])
        convective_0_3 = layer_integral(cond.altitude_km, cond.convective_gkgday, 0.0, 3.0)
        mean_flux_0_3 = mean_between_heights(flux.height_km, flux.mean_flux, 0.0, 3.0)
        out[experiment] = SummaryMetrics(
            experiment=experiment,
            label=_label(experiment),
            column_total_gkgday_km=column_total,
            condensation_peak_height_km=peak_height,
            column_total_divided_by_peak_height_gkgday=column_total
            / max(peak_height, 1.0e-12),
            warm_reservoir_0_3km_gkg_km=warm.get(experiment, float("nan")),
            convective_condensation_0_3km_gkgday_km=convective_0_3,
            mean_updraft_flux_0_3km_kg_m2_s=mean_flux_0_3,
            convective_condensation_divided_by_flux=convective_0_3 / mean_flux_0_3,
        )
    return out


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.97,
        0.96,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "boxstyle": "round,pad=0.18",
        },
        zorder=20,
    )


def plot_figure(
    *,
    condensation: dict[str, CondensationProfile],
    updraft: dict[str, UpdraftProfile],
    metrics: dict[str, SummaryMetrics],
    agg_dir: Path,
    output_path: Path,
    dpi: int,
    experiments: Sequence[str] = EXPERIMENT_ORDER,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )

    fig = plt.figure(figsize=(14.5, 11.4))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[3.0, 0.34, 1.3],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.30,
        wspace=0.20,
        left=0.07,
        right=0.95,
        top=0.87,
        bottom=0.08,
    )

    z_max = 10.0
    cond_xmax = 3.2
    flux_xmax = 1.2e-2

    for index, experiment in enumerate(experiments):
        ax = fig.add_subplot(gs[0, index])
        cond = condensation[experiment]
        flux = updraft[experiment]
        summary = metrics[experiment]
        keep = cond.altitude_km <= z_max + 0.1

        ax.axhspan(
            0,
            summary.condensation_peak_height_km,
            color="#BFC7D5",
            alpha=0.18,
            zorder=0,
        )
        ax.plot(
            cond.total_gkgday[keep],
            cond.altitude_km[keep],
            color=TOTAL_COLOR,
            lw=2.8,
            label="total condensation",
        )
        ax.plot(
            cond.convective_gkgday[keep],
            cond.altitude_km[keep],
            color=CONVECTIVE_COLOR,
            lw=2.0,
            ls="-.",
            alpha=0.9,
            label="convection-scheme part",
        )
        ax.plot(
            cond.resolved_gkgday[keep],
            cond.altitude_km[keep],
            color=RESOLVED_COLOR,
            lw=2.0,
            ls="--",
            alpha=0.95,
            label="resolved-microphysics part",
        )

        freeze_path = agg_dir / f"temperature_{experiment}.npz"
        freezing_level = freezing_level_from_temperature(freeze_path)
        ax.axhline(
            summary.condensation_peak_height_km,
            color=PEAK_COLOR,
            lw=1.2,
            ls=(0, (6, 3)),
            alpha=0.9,
        )
        if np.isfinite(freezing_level):
            ax.axhline(freezing_level, color=FREEZING_COLOR, lw=1.2, ls=":", alpha=0.95)

        ax2 = ax.twiny()
        ax2.plot(
            flux.mean_flux,
            flux.height_km,
            color=UPDRAFT_COLOR,
            lw=2.1,
            alpha=0.95,
            label="updraft mass flux",
        )

        ax.set_xlim(0, cond_xmax)
        ax2.set_xlim(0, flux_xmax)
        ax.set_ylim(0, z_max)
        ax.set_title(cond.label, color="black", pad=20)
        ax.set_xlabel(r"condensation (g kg$^{-1}$ day$^{-1}$)")
        ax2.set_xlabel(
            r"updraft mass flux (kg m$^{-2}$ s$^{-1}$)",
            color=UPDRAFT_COLOR,
            labelpad=8,
        )
        ax2.tick_params(axis="x", colors=UPDRAFT_COLOR)
        ax2.spines["top"].set_color(UPDRAFT_COLOR)
        if index == 0:
            ax.set_ylabel("altitude (km)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(alpha=0.25)
        add_panel_label(ax, PANEL_LABELS[index])

    style_handles = [
        Line2D([0], [0], color=TOTAL_COLOR, lw=2.8, label="total condensation"),
        Line2D(
            [0],
            [0],
            color=CONVECTIVE_COLOR,
            lw=2.0,
            ls="-.",
            label="convection-scheme part",
        ),
        Line2D(
            [0],
            [0],
            color=RESOLVED_COLOR,
            lw=2.0,
            ls="--",
            label="resolved-microphysics part",
        ),
        Line2D([0], [0], color=UPDRAFT_COLOR, lw=2.1, label="updraft mass flux"),
        Line2D(
            [0],
            [0],
            color=PEAK_COLOR,
            lw=1.2,
            ls=(0, (6, 3)),
            label="condensation peak height",
        ),
        Line2D(
            [0],
            [0],
            color=FREEZING_COLOR,
            lw=1.2,
            ls=":",
            label=r"0 $^{\circ}$C isotherm",
        ),
    ]
    ax_style_legend = fig.add_subplot(gs[1, :])
    ax_style_legend.axis("off")
    ax_style_legend.legend(
        handles=style_handles,
        loc="center",
        ncol=3,
        fontsize=12,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.8,
    )

    ax_b = fig.add_subplot(gs[2, :])
    labels = [metrics[experiment].label for experiment in experiments]
    quantities = [
        (
            r"column total condensation" "\n" r"(g kg$^{-1}$ day$^{-1}$ km)",
            [metrics[experiment].column_total_gkgday_km for experiment in experiments],
        ),
        (
            "condensation peak height\n(km)",
            [metrics[experiment].condensation_peak_height_km for experiment in experiments],
        ),
        (
            r"column total condensation /"
            "\n"
            r"height of maximum condensation"
            "\n"
            r"(g kg$^{-1}$ day$^{-1}$)",
            [
                metrics[experiment].column_total_divided_by_peak_height_gkgday
                for experiment in experiments
            ],
        ),
        (
            r"warm reservoir 0-3 km" "\n" r"($\times 10^{-4}$ g kg$^{-1}$ km)",
            [
                metrics[experiment].warm_reservoir_0_3km_gkg_km * 1.0e4
                for experiment in experiments
            ],
        ),
        (
            r"0-3 km convective condensation /"
            "\n"
            r"0-3 km mean updraft flux"
            "\n"
            r"($\times 10^3$)",
            [
                metrics[experiment].condensation_per_flux_x10minus3
                for experiment in experiments
            ],
        ),
    ]

    x = np.arange(len(quantities))
    width = 0.18
    for run_index, label in enumerate(labels):
        values = [quantity[1][run_index] for quantity in quantities]
        offset = (run_index - (len(labels) - 1) / 2.0) * width
        style = BAR_STYLES[label]
        ax_b.bar(
            x + offset,
            values,
            width=width,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            label=label,
            linewidth=0.8,
        )
        for bar_x, bar_y in zip(x + offset, values):
            ax_b.text(
                bar_x,
                bar_y,
                f"{bar_y:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="0.12",
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([quantity[0] for quantity in quantities], fontsize=11)
    ax_b.set_ylabel("value")
    ax_b.set_title("Summary metrics", fontsize=14, pad=14)
    bar_handles = [
        Patch(
            facecolor=BAR_STYLES[label]["facecolor"],
            edgecolor=BAR_STYLES[label]["edgecolor"],
            label=label,
            linewidth=0.8,
        )
        for label in labels
    ]
    ax_b.legend(
        handles=bar_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
        frameon=False,
    )
    ax_b.grid(axis="y", alpha=0.25)
    ymax = max(max(quantity[1]) for quantity in quantities)
    ax_b.set_ylim(0, ymax * 1.18)
    add_panel_label(ax_b, PANEL_LABELS[3])

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_data_txt(
    *,
    path: Path,
    figure_path: Path,
    dpi: int,
    lead: str,
    condensation_dir: Path,
    updraft_dir: Path,
    height_dir: Path,
    agg_dir: Path,
    condensation: dict[str, CondensationProfile],
    updraft: dict[str, UpdraftProfile],
    metrics: dict[str, SummaryMetrics],
    experiments: Sequence[str] = EXPERIMENT_ORDER,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Closure-budget compression figure data\n")
        fh.write("======================================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"DPI: {dpi}\n")
        fh.write(f"Lead: {lead}\n")
        fh.write("Inputs: upstream processed DDH condensation NPZ, upstream processed two-year updraft flux NPZ caches, and DDH aggregate QL/QR warm-reservoir data.\n")
        fh.write(f"Condensation source: {condensation_npz_path(condensation_dir, lead)}\n")
        fh.write(f"Updraft source directory: {updraft_dir}\n")
        fh.write(f"Updraft height source directory: {height_dir}\n")
        fh.write(f"DDH aggregate source directory: {agg_dir}\n")
        fh.write("Mass flux profile: time mean over all 24 local hours, weighted by per-level sample count.\n")
        fh.write("Low-level updraft flux metric: simple mean of the time-mean mass-flux profile over model levels whose heights are between 0 and 3 km.\n")
        fh.write("Low-level convective condensation metric: vertical integral of the convective condensation profile from 0 to 3 km.\n\n")
        fh.write("Note: peak height is the altitude of maximum total condensation, not a diagnosed active-cloud-top height.\n\n")

        fh.write("Summary bars\n")
        fh.write("------------\n")
        fh.write(
            "label,column_total_gkgday_km,condensation_peak_height_km,"
            "column_total_condensation_divided_by_height_of_maximum_condensation_gkgday,"
            "warm_reservoir_0_3km_gkg_km,"
            "convective_condensation_0_3km_divided_by_mean_updraft_flux_0_3km_x10minus3\n"
        )
        for experiment in experiments:
            row = metrics[experiment]
            fh.write(
                ",".join(
                    (
                        row.label,
                        _fmt(row.column_total_gkgday_km),
                        _fmt(row.condensation_peak_height_km),
                        _fmt(row.column_total_divided_by_peak_height_gkgday),
                        _fmt(row.warm_reservoir_0_3km_gkg_km),
                        _fmt(row.condensation_per_flux_x10minus3),
                    )
                )
                + "\n"
            )

        fh.write("\nLow-level condensation-per-flux data\n")
        fh.write("------------------------------------\n")
        fh.write(
            "label,convective_condensation_0_3km_gkgday_km,"
            "mean_updraft_mass_flux_0_3km_kg_m2_s,"
            "convective_condensation_0_3km_divided_by_mean_updraft_flux_0_3km,"
            "plotted_index_x10minus3\n"
        )
        for experiment in experiments:
            row = metrics[experiment]
            fh.write(
                ",".join(
                    (
                        row.label,
                        _fmt(row.convective_condensation_0_3km_gkgday_km),
                        _fmt(row.mean_updraft_flux_0_3km_kg_m2_s),
                        _fmt(row.convective_condensation_divided_by_flux),
                        _fmt(row.condensation_per_flux_x10minus3),
                    )
                )
                + "\n"
            )

        fh.write("\nMass flux profile (time mean)\n")
        fh.write("-----------------------------\n")
        fh.write("label,height_km,mean_mass_flux_kg_m2_s\n")
        for experiment in experiments:
            profile = updraft[experiment]
            for height, mean_flux in zip(profile.height_km, profile.mean_flux):
                fh.write(f"{profile.label},{_fmt(height)},{_fmt(mean_flux)}\n")

        fh.write(f"\nCondensation profile (lead {lead}, two-year mean)\n")
        fh.write("-----------------------------------------------\n")
        fh.write("label,altitude_km,convective_gkgday,resolved_gkgday,total_gkgday\n")
        for experiment in experiments:
            profile = condensation[experiment]
            for altitude, convective, resolved, total in zip(
                profile.altitude_km,
                profile.convective_gkgday,
                profile.resolved_gkgday,
                profile.total_gkgday,
            ):
                fh.write(
                    f"{profile.label},{_fmt(altitude)},{_fmt(convective)},"
                    f"{_fmt(resolved)},{_fmt(total)}\n"
                )


def make_closure_budget_compression(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    condensation_dir: Path = DEFAULT_CONDENSATION_DIR,
    updraft_dir: Path = DEFAULT_UPDRAFT_DIR,
    height_dir: Path = DEFAULT_HEIGHT_DIR,
    agg_dir: Path = AGG_DIR,
    lead: str = "0024",
    dpi: int = 450,
) -> tuple[Path, Path]:
    lead = normalize_lead(lead)
    condensation = load_condensation_profiles(condensation_dir, lead)
    updraft = load_updraft_profiles(updraft_dir, height_dir)
    metrics = build_summary_metrics(
        condensation,
        updraft,
        agg_dir=agg_dir,
        lead=lead,
    )

    figure_path = output_dir / FIGURE_NAME
    text_path = output_dir / DATA_TXT_SUBDIR / TEXT_NAME
    plot_figure(
        condensation=condensation,
        updraft=updraft,
        metrics=metrics,
        agg_dir=agg_dir,
        output_path=figure_path,
        dpi=dpi,
    )
    write_data_txt(
        path=text_path,
        figure_path=figure_path,
        dpi=dpi,
        lead=lead,
        condensation_dir=condensation_dir,
        updraft_dir=updraft_dir,
        height_dir=height_dir,
        agg_dir=agg_dir,
        condensation=condensation,
        updraft=updraft,
        metrics=metrics,
    )
    return figure_path, text_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make the DDH closure-budget compression diagnostic figure."
    )
    parser.add_argument("--lead", default="0024", help="DDH lead time, e.g. 0024.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the PNG and data_txt output.",
    )
    parser.add_argument(
        "--condensation-dir",
        type=Path,
        default=DEFAULT_CONDENSATION_DIR,
        help="Directory containing condensation_partition_profile_leadXXXX.npz.",
    )
    parser.add_argument(
        "--updraft-dir",
        type=Path,
        default=DEFAULT_UPDRAFT_DIR,
        help="Directory containing *_full-domain_diurnal_profile.npz updraft flux caches.",
    )
    parser.add_argument(
        "--height-dir",
        type=Path,
        default=DEFAULT_HEIGHT_DIR,
        help="Directory containing *_full-domain_height_profile_first.npz height caches.",
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=AGG_DIR,
        help="DDH aggregate directory, used for warm-reservoir and temperature data.",
    )
    parser.add_argument("--dpi", type=int, default=450, help="Output figure DPI.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    figure_path, text_path = make_closure_budget_compression(
        output_dir=args.output_dir,
        condensation_dir=args.condensation_dir,
        updraft_dir=args.updraft_dir,
        height_dir=args.height_dir,
        agg_dir=args.agg_dir,
        lead=args.lead,
        dpi=args.dpi,
    )
    print(f"wrote {figure_path}")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()
