"""DDH condensation partition profile from aggregated QV budgets.

The primary diagnostic is the water-vapour sink split into convective and
resolved condensation:

    convective = max(-QV.condcv, 0)
    resolved   = max(-QV.condrs, 0)

Inputs are the aggregated DDH ``{experiment}_QV.npz`` files produced by
``alaro_analysis.ddh.aggregate_budgets``.
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
    EXPERIMENTS,
    FIG_DIR,
    freezing_level_km,
    load_temperature,
    set_altitude_axis,
    tick_formatter,
)
from .plot_style import (
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
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/ddh_condensation"
)
PLOT_DIR = FIG_DIR / "ddh_condensation"
DATA_TXT_SUBDIR = "data_txt"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
PANEL_TITLE_FONTSIZE = 16
PANEL_LABEL_FONTSIZE = 14


@dataclass(frozen=True)
class CondensationPartition:
    experiment: str
    lead: str
    altitude_km: np.ndarray
    convective_gkgday: np.ndarray
    resolved_gkgday: np.ndarray
    total_gkgday: np.ndarray
    convective_fraction: np.ndarray
    resolved_fraction: np.ndarray
    n_days: int


def normalize_lead(lead: str | int) -> str:
    """Return a four-digit lead string such as ``0024``."""
    lead_text = str(lead).strip()
    if lead_text.startswith("+"):
        lead_text = lead_text[1:]
    if not lead_text.isdigit():
        raise ValueError(f"lead must be numeric, got {lead!r}")
    return f"{int(lead_text):04d}"


def _positive_sink(block: np.ndarray) -> np.ndarray:
    """Convert a negative QV tendency into a positive source rate."""
    return np.maximum(-np.asarray(block, dtype=np.float64), 0.0)


def compute_partition(
    *,
    experiment: str,
    lead: str | int,
    altitude_km: np.ndarray,
    condcv: np.ndarray,
    condrs: np.ndarray,
    n_days: int = 0,
) -> CondensationPartition:
    """Compute convective/resolved/total condensation from QV budget blocks."""
    z = np.asarray(altitude_km, dtype=np.float64)
    convective = _positive_sink(condcv)
    resolved = _positive_sink(condrs)
    if convective.shape != resolved.shape or convective.shape != z.shape:
        raise ValueError(
            "altitude_km, condcv, and condrs must have identical shapes"
        )

    total = convective + resolved
    conv_frac = np.divide(
        convective,
        total,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=total > 0,
    )
    rs_frac = np.divide(
        resolved,
        total,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=total > 0,
    )
    return CondensationPartition(
        experiment=experiment,
        lead=normalize_lead(lead),
        altitude_km=z,
        convective_gkgday=convective,
        resolved_gkgday=resolved,
        total_gkgday=total,
        convective_fraction=conv_frac,
        resolved_fraction=rs_frac,
        n_days=int(n_days),
    )


def load_partition(
    experiment: str,
    lead: str | int = "0024",
    *,
    agg_dir: Path = AGG_DIR,
) -> CondensationPartition:
    """Load one experiment's QV aggregate and compute its partition."""
    lead_text = normalize_lead(lead)
    path = agg_dir / f"lead{lead_text}_VZ" / f"{experiment}_QV.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing DDH QV aggregate: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = ("altitude_km", "block__condcv", "block__condrs")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing required field(s): {', '.join(missing)}")
        n_days = int(data["days"].shape[0]) if "days" in data.files else 0
        return compute_partition(
            experiment=experiment,
            lead=lead_text,
            altitude_km=data["altitude_km"],
            condcv=data["block__condcv"],
            condrs=data["block__condrs"],
            n_days=n_days,
        )


def column_integrate(profile: np.ndarray, altitude_km: np.ndarray) -> float:
    """Integrate a vertical profile over altitude in km."""
    z = np.asarray(altitude_km, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(p)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    z = z[finite]
    p = p[finite]
    order = np.argsort(z)
    z = z[order]
    p = p[order]
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(z)))


def summarize_partition(partition: CondensationPartition) -> dict[str, float | str | int]:
    """Return one summary row for the analytics CSV."""
    conv_col = column_integrate(partition.convective_gkgday, partition.altitude_km)
    rs_col = column_integrate(partition.resolved_gkgday, partition.altitude_km)
    total_col = column_integrate(partition.total_gkgday, partition.altitude_km)
    if np.isfinite(total_col) and total_col > 0:
        conv_frac = conv_col / total_col
        rs_frac = rs_col / total_col
    else:
        conv_frac = float("nan")
        rs_frac = float("nan")

    finite_total = np.isfinite(partition.total_gkgday)
    if np.any(finite_total):
        peak_idx = int(np.nanargmax(partition.total_gkgday))
        peak_total = float(partition.total_gkgday[peak_idx])
        peak_altitude = float(partition.altitude_km[peak_idx])
    else:
        peak_total = float("nan")
        peak_altitude = float("nan")

    return {
        "experiment": partition.experiment,
        "lead": partition.lead,
        "n_days": partition.n_days,
        "column_convective_gkgday_km": conv_col,
        "column_resolved_gkgday_km": rs_col,
        "column_total_gkgday_km": total_col,
        "column_convective_fraction": conv_frac,
        "column_resolved_fraction": rs_frac,
        "peak_total_gkgday": peak_total,
        "peak_total_altitude_km": peak_altitude,
    }


def _format_csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def _format_txt_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _txt_path_for_plot(path: Path) -> Path:
    return path.parent / DATA_TXT_SUBDIR / f"{path.stem}.txt"


def _write_headline(
    fh,
    title: str,
    figure_path: Path,
    metadata: tuple[str, ...],
) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'=' * len(title)}\n")
    fh.write(f"Figure: {figure_path}\n")
    for line in metadata:
        fh.write(f"{line}\n")
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


def write_plot_txt(partitions: list[CondensationPartition], path: Path) -> Path:
    """Write the exact data used by the condensation partition figure."""
    if not partitions:
        raise ValueError("at least one partition is required")

    xlim = common_rate_xlim(partitions)
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_headline(
            fh,
            f"DDH Condensation Partition Plot Data: {path.stem}",
            path,
            (
                f"Lead: {partitions[0].lead}",
                "Variables: total, convective, and resolved condensation from QV budget.",
                "Units: g kg-1 day-1.",
                f"Shared x-axis limits: {_format_txt_value(xlim[0])}, {_format_txt_value(xlim[1])}.",
            ),
        )
        _write_csv_section(
            fh,
            "Freezing-level data",
            ("experiment", "label", "z_freeze_km"),
            [
                (
                    part.experiment,
                    EXPERIMENTS.get(part.experiment, part.experiment),
                    freezing_level_km(load_temperature(part.experiment)),
                )
                for part in partitions
            ],
        )
        summary_rows = [
            tuple(summarize_partition(part).get(field, "") for field in (
                "experiment",
                "lead",
                "n_days",
                "column_convective_gkgday_km",
                "column_resolved_gkgday_km",
                "column_total_gkgday_km",
                "column_convective_fraction",
                "column_resolved_fraction",
                "peak_total_gkgday",
                "peak_total_altitude_km",
            ))
            for part in partitions
        ]
        _write_csv_section(
            fh,
            "Summary data",
            (
                "experiment",
                "lead",
                "n_days",
                "column_convective_gkgday_km",
                "column_resolved_gkgday_km",
                "column_total_gkgday_km",
                "column_convective_fraction",
                "column_resolved_fraction",
                "peak_total_gkgday",
                "peak_total_altitude_km",
            ),
            summary_rows,
        )
        for part in partitions:
            rows = [
                (
                    part.altitude_km[i],
                    part.convective_gkgday[i],
                    part.resolved_gkgday[i],
                    part.total_gkgday[i],
                    part.convective_fraction[i],
                    part.resolved_fraction[i],
                    part.n_days,
                )
                for i in range(part.altitude_km.size)
            ]
            label = EXPERIMENTS.get(part.experiment, part.experiment)
            _write_csv_section(
                fh,
                f"{label} profile data",
                (
                    "altitude_km",
                    "convective_gkgday",
                    "resolved_gkgday",
                    "total_gkgday",
                    "convective_fraction",
                    "resolved_fraction",
                    "n_days",
                ),
                rows,
            )
    return txt_path


def write_by_level_csv(partitions: list[CondensationPartition], path: Path) -> None:
    """Write one row per experiment and altitude level."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "lead",
        "altitude_km",
        "convective_gkgday",
        "resolved_gkgday",
        "total_gkgday",
        "convective_fraction",
        "resolved_fraction",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for part in partitions:
            for i, z in enumerate(part.altitude_km):
                row = {
                    "experiment": part.experiment,
                    "lead": part.lead,
                    "altitude_km": float(z),
                    "convective_gkgday": float(part.convective_gkgday[i]),
                    "resolved_gkgday": float(part.resolved_gkgday[i]),
                    "total_gkgday": float(part.total_gkgday[i]),
                    "convective_fraction": float(part.convective_fraction[i]),
                    "resolved_fraction": float(part.resolved_fraction[i]),
                }
                writer.writerow({k: _format_csv_value(v) for k, v in row.items()})


def write_summary_csv(partitions: list[CondensationPartition], path: Path) -> None:
    """Write one summary row per experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summarize_partition(part) for part in partitions]
    fields = list(rows[0].keys()) if rows else [
        "experiment",
        "lead",
        "n_days",
        "column_convective_gkgday_km",
        "column_resolved_gkgday_km",
        "column_total_gkgday_km",
        "column_convective_fraction",
        "column_resolved_fraction",
        "peak_total_gkgday",
        "peak_total_altitude_km",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_value(v) for k, v in row.items()})


def _altitude_stack(partitions: list[CondensationPartition]) -> np.ndarray:
    if not partitions:
        raise ValueError("at least one partition is required")
    shapes = {part.altitude_km.shape for part in partitions}
    if len(shapes) != 1:
        raise ValueError("all experiments must have the same number of altitude levels")
    return np.stack([part.altitude_km for part in partitions], axis=0)


def save_npz(partitions: list[CondensationPartition], path: Path) -> None:
    """Save plot-ready arrays for the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "altitude_km": _altitude_stack(partitions),
        "experiments": np.array([part.experiment for part in partitions], dtype="U16"),
    }
    for part in partitions:
        prefix = part.experiment
        payload[f"{prefix}_altitude_km"] = part.altitude_km
        payload[f"{prefix}_convective_gkgday"] = part.convective_gkgday
        payload[f"{prefix}_resolved_gkgday"] = part.resolved_gkgday
        payload[f"{prefix}_total_gkgday"] = part.total_gkgday
        payload[f"{prefix}_convective_fraction"] = part.convective_fraction
        payload[f"{prefix}_resolved_fraction"] = part.resolved_fraction
        payload[f"{prefix}_n_days"] = np.array(part.n_days, dtype=np.int64)
    np.savez_compressed(path, **payload)


def common_rate_xlim(
    partitions: list[CondensationPartition],
    *,
    pad_fraction: float = 0.06,
) -> tuple[float, float]:
    """Return a shared positive x-axis range for all rate-profile panels."""
    values = []
    for part in partitions:
        for profile in (
            part.total_gkgday,
            part.convective_gkgday,
            part.resolved_gkgday,
        ):
            finite = profile[np.isfinite(profile)]
            if finite.size:
                values.append(finite)
    if not values:
        return (0.0, 1.0)
    vmax = float(np.max(np.concatenate(values)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return (0.0, 1.0)
    return (0.0, vmax * (1.0 + pad_fraction))


def plot_profile(
    partitions: list[CondensationPartition],
    path: Path,
    *,
    show_titles: bool = False,
) -> None:
    """Plot total, convective, and resolved profiles in one panel per experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_rate_xlim(partitions)
    fig, axes = plt.subplots(
        1,
        len(partitions),
        figsize=EXPERIMENT_PANEL_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    if len(partitions) == 1:
        axes = [axes]

    for col, (ax, part) in enumerate(zip(axes, partitions)):
        color, lw, ls, alpha, zorder = partition_line_style("Condensation", "total")
        ax.plot(
            part.total_gkgday,
            part.altitude_km,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Total",
        )
        color, lw, ls, alpha, zorder = partition_line_style("Condensation", "convective")
        ax.plot(
            part.convective_gkgday,
            part.altitude_km,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Convective",
        )
        color, lw, ls, alpha, zorder = partition_line_style("Condensation", "resolved")
        ax.plot(
            part.resolved_gkgday,
            part.altitude_km,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label="Resolved",
        )

        temp = load_temperature(part.experiment)
        z0 = freezing_level_km(temp)
        if np.isfinite(z0):
            ax.axhline(
                z0,
                color=FREEZING_COLOR,
                lw=FREEZING_LINEWIDTH,
                ls=FREEZING_LINESTYLE,
                alpha=FREEZING_ALPHA,
                label=r"0 $^{\circ}$C isotherm",
            )

        label = EXPERIMENTS.get(part.experiment, part.experiment)
        ax.set_title(label, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        panel_label = PANEL_LABELS[col] if col < len(PANEL_LABELS) else f"({col + 1})"
        ax.text(
            0.03,
            0.96,
            panel_label,
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
        ax.set_xlabel(r"Condensation rate (g kg$^{-1}$ day$^{-1}$)")
        ax.set_xlim(*xlim)
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=PANEL_GRID_ALPHA)
        ax.xaxis.set_major_formatter(tick_formatter())
        ax.legend(loc="best", fontsize=PANEL_LEGEND_FONTSIZE)

    set_altitude_axis(axes[0])
    if show_titles:
        fig.suptitle(
            "DDH condensation partition from QV budget",
            fontsize=FIGURE_TITLE_FONTSIZE,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    lead: str | int = "0024",
    agg_dir: Path = AGG_DIR,
    plot_dir: Path = PLOT_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    show_titles: bool = False,
    write_txt: bool = True,
) -> dict[str, Path]:
    """Run the condensation partition analysis and return output paths."""
    lead_text = normalize_lead(lead)
    partitions = [
        load_partition(exp, lead=lead_text, agg_dir=agg_dir)
        for exp in EXPERIMENTS
    ]

    figure_path = plot_dir / f"condensation_partition_profile_lead{lead_text}.png"
    npz_path = processed_dir / f"condensation_partition_profile_lead{lead_text}.npz"
    analytics_dir = processed_dir / "analytics"
    by_level_csv = analytics_dir / f"condensation_partition_profile_lead{lead_text}_by_level.csv"
    summary_csv = analytics_dir / f"condensation_partition_profile_lead{lead_text}_summary.csv"

    plot_profile(partitions, figure_path, show_titles=show_titles)
    save_npz(partitions, npz_path)
    write_by_level_csv(partitions, by_level_csv)
    write_summary_csv(partitions, summary_csv)

    outputs = {
        "figure": figure_path,
        "npz": npz_path,
        "by_level_csv": by_level_csv,
        "summary_csv": summary_csv,
    }
    if write_txt:
        outputs["txt"] = write_plot_txt(partitions, figure_path)
    for key, path in outputs.items():
        print(f"{key}: {path}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lead", default="0024", help="Forecast lead, e.g. 0024.")
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=AGG_DIR,
        help="Root containing lead<LEAD>_VZ aggregated DDH npz files.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PLOT_DIR,
        help="Directory for DDH condensation figures.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for plot-ready npz and analytics CSV outputs.",
    )
    parser.add_argument(
        "--with-titles",
        action="store_true",
        help="Draw figure-level suptitles. Panel titles are always kept.",
    )
    parser.add_argument(
        "--no-write-txt",
        action="store_true",
        help="Do not write the figure-side data_txt file.",
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
