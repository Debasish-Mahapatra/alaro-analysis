"""Lower-layer DDH condensation partition through the diurnal cycle.

The DDH lead files contain tendencies averaged from the beginning of the
forecast to each lead.  This diagnostic reconstructs hourly increments before
plotting, so the x-axis shows the hour in which the condensation occurred.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.ddh.io import (
    AGG_DIR,
    EXPERIMENTS,
    freezing_level_km,
    load_temperature,
    tick_formatter,
)
from alaro_analysis.ddh.plot_style import (
    CONVECTION_COLOR,
    CONVECTION_LINESTYLE,
    PANEL_GRID_ALPHA,
    PANEL_LEGEND_FONTSIZE,
    RESOLVED_COLOR,
    RESOLVED_LINESTYLE,
    TOTAL_COLOR,
    TOTAL_LINEWIDTH,
)


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "DDH-condensation_diurnal_partition"
)
DATA_TXT_SUBDIR = "data_txt"
EXPERIMENT_ORDER = tuple(EXPERIMENTS.keys())
LEADS = tuple(range(1, 25))
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")
DIURNAL_UTC_OFFSET_HOURS = -4.0


@dataclass(frozen=True)
class HourlyCondensation:
    experiment: str
    layer_bottom_km: float
    layer_top_km: float
    lead: int
    hour_start: int
    hour_end: int
    hour_center: float
    convective_amount_gkg_km: float
    resolved_amount_gkg_km: float
    total_amount_gkg_km: float
    convective_rate_equiv_gkgday_km: float
    resolved_rate_equiv_gkgday_km: float
    total_rate_equiv_gkgday_km: float
    cumulative_convective_amount_gkg_km: float
    cumulative_resolved_amount_gkg_km: float
    cumulative_total_amount_gkg_km: float
    n_days: int


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _positive_sink(profile: np.ndarray) -> np.ndarray:
    return np.maximum(-np.asarray(profile, dtype=np.float64), 0.0)


def integrate_layer(
    altitude_km: np.ndarray,
    profile: np.ndarray,
    *,
    layer_bottom_km: float,
    layer_top_km: float,
) -> float:
    """Integrate one profile over a clipped altitude layer in km."""
    z = np.asarray(altitude_km, dtype=np.float64)
    values = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")

    z = z[finite]
    values = values[finite]
    order = np.argsort(z)
    z = z[order]
    values = values[order]

    bottom = max(float(layer_bottom_km), float(z[0]))
    top = min(float(layer_top_km), float(z[-1]))
    if top <= bottom:
        return float("nan")

    inside = (z > bottom) & (z < top)
    layer_z = np.concatenate(([bottom], z[inside], [top]))
    layer_values = np.interp(layer_z, z, values)
    return float(np.trapezoid(layer_values, layer_z))


def _load_qv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing DDH QV aggregate: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = ("altitude_km", "block__condcv", "block__condrs")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing required fields: {', '.join(missing)}")
        n_days = int(data["days"].shape[0]) if "days" in data.files else 0
        return (
            np.asarray(data["altitude_km"], dtype=np.float64),
            np.asarray(data["block__condcv"], dtype=np.float64),
            np.asarray(data["block__condrs"], dtype=np.float64),
            n_days,
        )


def load_hourly_condensation(
    *,
    experiment: str,
    agg_dir: Path,
    leads: Sequence[int] = LEADS,
    layer_bottom_km: float = 0.0,
    layer_top_km: float = 3.0,
) -> list[HourlyCondensation]:
    """Reconstruct hourly lower-layer condensation from accumulated DDH means."""
    previous_convective_cumulative: np.ndarray | None = None
    previous_resolved_cumulative: np.ndarray | None = None
    cumulative_convective_amount = 0.0
    cumulative_resolved_amount = 0.0
    records: list[HourlyCondensation] = []

    for lead in leads:
        lead_text = f"{int(lead):04d}"
        path = agg_dir / f"lead{lead_text}_VZ" / f"{experiment}_QV.npz"
        altitude, condcv, condrs, n_days = _load_qv(path)

        time_fraction_days = float(lead) / 24.0
        convective_cumulative = condcv * time_fraction_days
        resolved_cumulative = condrs * time_fraction_days

        if previous_convective_cumulative is None:
            convective_increment = convective_cumulative
            resolved_increment = resolved_cumulative
        else:
            convective_increment = convective_cumulative - previous_convective_cumulative
            resolved_increment = resolved_cumulative - previous_resolved_cumulative

        convective_amount = integrate_layer(
            altitude,
            _positive_sink(convective_increment),
            layer_bottom_km=layer_bottom_km,
            layer_top_km=layer_top_km,
        )
        resolved_amount = integrate_layer(
            altitude,
            _positive_sink(resolved_increment),
            layer_bottom_km=layer_bottom_km,
            layer_top_km=layer_top_km,
        )
        total_amount = convective_amount + resolved_amount
        cumulative_convective_amount += convective_amount
        cumulative_resolved_amount += resolved_amount

        records.append(
            HourlyCondensation(
                experiment=experiment,
                layer_bottom_km=float(layer_bottom_km),
                layer_top_km=float(layer_top_km),
                lead=int(lead),
                hour_start=int(lead) - 1,
                hour_end=int(lead),
                hour_center=float(lead) - 0.5,
                convective_amount_gkg_km=convective_amount,
                resolved_amount_gkg_km=resolved_amount,
                total_amount_gkg_km=total_amount,
                convective_rate_equiv_gkgday_km=convective_amount * 24.0,
                resolved_rate_equiv_gkgday_km=resolved_amount * 24.0,
                total_rate_equiv_gkgday_km=total_amount * 24.0,
                cumulative_convective_amount_gkg_km=cumulative_convective_amount,
                cumulative_resolved_amount_gkg_km=cumulative_resolved_amount,
                cumulative_total_amount_gkg_km=(
                    cumulative_convective_amount + cumulative_resolved_amount
                ),
                n_days=n_days,
            )
        )
        previous_convective_cumulative = convective_cumulative
        previous_resolved_cumulative = resolved_cumulative

    return records


def build_dataset(
    *,
    agg_dir: Path = AGG_DIR,
    experiments: Sequence[str] = EXPERIMENT_ORDER,
    leads: Sequence[int] = LEADS,
    layer_bottom_km: float = 0.0,
    layer_top_km: float | None = 3.0,
) -> dict[str, list[HourlyCondensation]]:
    return {
        experiment: load_hourly_condensation(
            experiment=experiment,
            agg_dir=agg_dir,
            leads=leads,
            layer_bottom_km=layer_bottom_km,
            layer_top_km=resolve_layer_top_km(experiment, layer_top_km),
        )
        for experiment in experiments
    }


def resolve_layer_top_km(experiment: str, layer_top_km: float | None) -> float:
    """Return a fixed layer top or the experiment's mean freezing level."""
    if layer_top_km is not None:
        return float(layer_top_km)
    z_freeze = freezing_level_km(load_temperature(experiment))
    if not np.isfinite(z_freeze):
        raise ValueError(
            f"Could not determine freezing level for {experiment}; "
            "pass --layer-top-km with a numeric value."
        )
    return float(z_freeze)


def _series(
    records: Sequence[HourlyCondensation],
    field: str,
) -> np.ndarray:
    return np.asarray([getattr(record, field) for record in records], dtype=np.float64)


def _hours(records: Sequence[HourlyCondensation]) -> np.ndarray:
    return _series(records, "hour_center")


def _shifted_hour(hours: np.ndarray | float) -> np.ndarray | float:
    shifted = (np.asarray(hours, dtype=np.float64) + DIURNAL_UTC_OFFSET_HOURS) % 24.0
    if np.isscalar(hours):
        return float(shifted)
    return shifted


def _plot_xy(
    records: Sequence[HourlyCondensation],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    hours = np.asarray(_shifted_hour(_hours(records)), dtype=np.float64)
    values = _series(records, field)
    order = np.argsort(hours)
    return hours[order], values[order]


def _daily_rows(
    data: dict[str, list[HourlyCondensation]],
) -> list[tuple[object, ...]]:
    rows = []
    for experiment, records in data.items():
        rows.append(
            (
                experiment,
                EXPERIMENT_LABELS.get(experiment, experiment),
                records[-1].layer_bottom_km if records else "",
                records[-1].layer_top_km if records else "",
                sum(record.convective_amount_gkg_km for record in records),
                sum(record.resolved_amount_gkg_km for record in records),
                sum(record.total_amount_gkg_km for record in records),
                records[-1].n_days if records else 0,
            )
        )
    return rows


def _write_section(
    fh,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'-' * len(title)}\n")
    fh.write(",".join(columns) + "\n")
    for row in rows:
        fh.write(",".join(_fmt(value) for value in row) + "\n")
    fh.write("\n")


def write_absolute_txt(
    data: dict[str, list[HourlyCondensation]],
    path: Path,
    figure_path: Path,
    *,
    layer_bottom_km: float,
    layer_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        title = "Lower-layer DDH hourly condensation partition"
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Layer: {layer_label}.\n")
        fh.write("Variable: QV budget condensation sink.\n")
        fh.write(
            "Hourly amounts are reconstructed by differencing accumulated DDH lead means.\n"
        )
        fh.write("The plotted x-axis is shifted to UTC-4.\n")
        fh.write(
            "Amount units: g kg-1 km per hour. Rate-equivalent units: g kg-1 day-1 km.\n\n"
        )

        _write_section(
            fh,
            "Daily sums from reconstructed hourly amounts",
            (
                "experiment",
                "label",
                "layer_bottom_km",
                "layer_top_km",
                "convective_amount_gkg_km",
                "resolved_amount_gkg_km",
                "total_amount_gkg_km",
                "n_days",
            ),
            _daily_rows(data),
        )

        rows = []
        for experiment, records in data.items():
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for record in records:
                rows.append(
                    (
                        experiment,
                        label,
                        record.lead,
                        record.layer_bottom_km,
                        record.layer_top_km,
                        record.hour_start,
                        record.hour_end,
                        record.hour_center,
                        _shifted_hour(record.hour_center),
                        record.convective_amount_gkg_km,
                        record.resolved_amount_gkg_km,
                        record.total_amount_gkg_km,
                        record.convective_rate_equiv_gkgday_km,
                        record.resolved_rate_equiv_gkgday_km,
                        record.total_rate_equiv_gkgday_km,
                        record.cumulative_convective_amount_gkg_km,
                        record.cumulative_resolved_amount_gkg_km,
                        record.cumulative_total_amount_gkg_km,
                        record.n_days,
                    )
                )
        _write_section(
            fh,
            "Hourly plot data",
            (
                "experiment",
                "label",
                "lead",
                "layer_bottom_km",
                "layer_top_km",
                "hour_start",
                "hour_end",
                "hour_center_utc",
                "hour_center_utc_minus4",
                "convective_amount_gkg_km",
                "resolved_amount_gkg_km",
                "total_amount_gkg_km",
                "convective_rate_equiv_gkgday_km",
                "resolved_rate_equiv_gkgday_km",
                "total_rate_equiv_gkgday_km",
                "cumulative_convective_amount_gkg_km",
                "cumulative_resolved_amount_gkg_km",
                "cumulative_total_amount_gkg_km",
                "n_days",
            ),
            rows,
        )


def write_difference_txt(
    data: dict[str, list[HourlyCondensation]],
    path: Path,
    figure_path: Path,
    *,
    reference: str,
    layer_bottom_km: float,
    layer_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_records = data[reference]
    with path.open("w", encoding="utf-8") as fh:
        title = "Lower-layer DDH hourly condensation differences"
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Reference experiment: {EXPERIMENT_LABELS.get(reference, reference)}.\n")
        fh.write(f"Layer: {layer_label}.\n")
        fh.write("Differences are experiment minus reference.\n")
        fh.write(
            "For freezing-level runs, each experiment is integrated to its own "
            "mean freezing level.\n"
        )
        fh.write("The plotted x-axis is shifted to UTC-4.\n")
        fh.write("Amount units: g kg-1 km per hour.\n\n")

        rows = []
        for experiment, records in data.items():
            if experiment == reference:
                continue
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            ref_label = EXPERIMENT_LABELS.get(reference, reference)
            conv_diff = sum(
                rec.convective_amount_gkg_km - ref.convective_amount_gkg_km
                for rec, ref in zip(records, ref_records)
            )
            resolved_diff = sum(
                rec.resolved_amount_gkg_km - ref.resolved_amount_gkg_km
                for rec, ref in zip(records, ref_records)
            )
            total_diff = sum(
                rec.total_amount_gkg_km - ref.total_amount_gkg_km
                for rec, ref in zip(records, ref_records)
            )
            rows.append(
                (
                    experiment,
                    label,
                    records[-1].layer_bottom_km if records else "",
                    records[-1].layer_top_km if records else "",
                    reference,
                    ref_label,
                    ref_records[-1].layer_bottom_km if ref_records else "",
                    ref_records[-1].layer_top_km if ref_records else "",
                    conv_diff,
                    resolved_diff,
                    total_diff,
                )
            )
        _write_section(
            fh,
            "Daily-sum differences",
            (
                "experiment",
                "label",
                "layer_bottom_km",
                "layer_top_km",
                "reference",
                "reference_label",
                "reference_layer_bottom_km",
                "reference_layer_top_km",
                "convective_amount_difference_gkg_km",
                "resolved_amount_difference_gkg_km",
                "total_amount_difference_gkg_km",
            ),
            rows,
        )

        hourly_rows = []
        for experiment, records in data.items():
            if experiment == reference:
                continue
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            ref_label = EXPERIMENT_LABELS.get(reference, reference)
            for record, ref in zip(records, ref_records):
                hourly_rows.append(
                    (
                        experiment,
                        label,
                        reference,
                        ref_label,
                        record.lead,
                        record.layer_bottom_km,
                        record.layer_top_km,
                        ref.layer_bottom_km,
                        ref.layer_top_km,
                        record.hour_start,
                        record.hour_end,
                        record.hour_center,
                        _shifted_hour(record.hour_center),
                        record.convective_amount_gkg_km - ref.convective_amount_gkg_km,
                        record.resolved_amount_gkg_km - ref.resolved_amount_gkg_km,
                        record.total_amount_gkg_km - ref.total_amount_gkg_km,
                    )
                )
        _write_section(
            fh,
            "Hourly difference plot data",
            (
                "experiment",
                "label",
                "reference",
                "reference_label",
                "lead",
                "layer_bottom_km",
                "layer_top_km",
                "reference_layer_bottom_km",
                "reference_layer_top_km",
                "hour_start",
                "hour_end",
                "hour_center_utc",
                "hour_center_utc_minus4",
                "convective_amount_difference_gkg_km",
                "resolved_amount_difference_gkg_km",
                "total_amount_difference_gkg_km",
            ),
            hourly_rows,
        )


def write_csv(path: Path, data: dict[str, list[HourlyCondensation]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "label",
        "layer_bottom_km",
        "layer_top_km",
        "lead",
        "hour_start",
        "hour_end",
        "hour_center_utc",
        "hour_center_utc_minus4",
        "convective_amount_gkg_km",
        "resolved_amount_gkg_km",
        "total_amount_gkg_km",
        "convective_rate_equiv_gkgday_km",
        "resolved_rate_equiv_gkgday_km",
        "total_rate_equiv_gkgday_km",
        "cumulative_convective_amount_gkg_km",
        "cumulative_resolved_amount_gkg_km",
        "cumulative_total_amount_gkg_km",
        "n_days",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for experiment, records in data.items():
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for record in records:
                writer.writerow(
                    {
                        "experiment": experiment,
                        "label": label,
                        "layer_bottom_km": _fmt(record.layer_bottom_km),
                        "layer_top_km": _fmt(record.layer_top_km),
                        "lead": record.lead,
                        "hour_start": record.hour_start,
                        "hour_end": record.hour_end,
                        "hour_center_utc": record.hour_center,
                        "hour_center_utc_minus4": _fmt(_shifted_hour(record.hour_center)),
                        "convective_amount_gkg_km": _fmt(record.convective_amount_gkg_km),
                        "resolved_amount_gkg_km": _fmt(record.resolved_amount_gkg_km),
                        "total_amount_gkg_km": _fmt(record.total_amount_gkg_km),
                        "convective_rate_equiv_gkgday_km": _fmt(
                            record.convective_rate_equiv_gkgday_km
                        ),
                        "resolved_rate_equiv_gkgday_km": _fmt(
                            record.resolved_rate_equiv_gkgday_km
                        ),
                        "total_rate_equiv_gkgday_km": _fmt(
                            record.total_rate_equiv_gkgday_km
                        ),
                        "cumulative_convective_amount_gkg_km": _fmt(
                            record.cumulative_convective_amount_gkg_km
                        ),
                        "cumulative_resolved_amount_gkg_km": _fmt(
                            record.cumulative_resolved_amount_gkg_km
                        ),
                        "cumulative_total_amount_gkg_km": _fmt(
                            record.cumulative_total_amount_gkg_km
                        ),
                        "n_days": record.n_days,
                    }
                )


def plot_absolute(
    data: dict[str, list[HourlyCondensation]],
    path: Path,
    *,
    layer_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for records in data.values():
        values.extend(_series(records, "convective_amount_gkg_km"))
        values.extend(_series(records, "resolved_amount_gkg_km"))
        values.extend(_series(records, "total_amount_gkg_km"))
    ymax = max(values) * 1.12 if values else 1.0

    fig, axes = plt.subplots(
        len(data),
        1,
        figsize=(11.5, 8.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(data) == 1:
        axes = [axes]

    for idx, (ax, (experiment, records)) in enumerate(zip(axes, data.items())):
        total_hours, total_values = _plot_xy(records, "total_amount_gkg_km")
        convective_hours, convective_values = _plot_xy(records, "convective_amount_gkg_km")
        resolved_hours, resolved_values = _plot_xy(records, "resolved_amount_gkg_km")
        ax.plot(
            total_hours,
            total_values,
            color=TOTAL_COLOR,
            lw=TOTAL_LINEWIDTH,
            label="Total",
        )
        ax.plot(
            convective_hours,
            convective_values,
            color=CONVECTION_COLOR,
            lw=2.0,
            ls=CONVECTION_LINESTYLE,
            label="Convection scheme",
        )
        ax.plot(
            resolved_hours,
            resolved_values,
            color=RESOLVED_COLOR,
            lw=2.0,
            ls=RESOLVED_LINESTYLE,
            label="Resolved microphysics",
        )
        ax.set_ylim(0.0, ymax)
        ax.set_xlim(0.0, 24.0)
        ax.set_ylabel("Hourly amount\n(g kg$^{-1}$ km)")
        layer_top = records[-1].layer_top_km if records else float("nan")
        ax.set_title(
            f"{EXPERIMENT_LABELS.get(experiment, experiment)}  "
            f"(top {layer_top:.2f} km)",
            loc="left",
        )
        ax.grid(alpha=PANEL_GRID_ALPHA)
        ax.xaxis.set_major_formatter(tick_formatter())
        panel_label = PANEL_LABELS[idx] if idx < len(PANEL_LABELS) else f"({idx + 1})"
        ax.text(
            0.985,
            0.9,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
        if idx == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=PANEL_LEGEND_FONTSIZE)

    axes[-1].set_xlabel("Hour of day (UTC-4)")
    fig.suptitle(
        f"{layer_label} condensation partition through the day",
        fontweight="bold",
    )
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_difference(
    data: dict[str, list[HourlyCondensation]],
    path: Path,
    *,
    reference: str,
    layer_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparisons = [(exp, records) for exp, records in data.items() if exp != reference]
    ref_records = data[reference]

    diff_values = []
    for _, records in comparisons:
        for field in (
            "convective_amount_gkg_km",
            "resolved_amount_gkg_km",
            "total_amount_gkg_km",
        ):
            diff_values.extend(_series(records, field) - _series(ref_records, field))
    max_abs = max(abs(float(value)) for value in diff_values) if diff_values else 1.0
    ylim = (-max_abs * 1.15, max_abs * 1.15)

    fig, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(11.5, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(comparisons) == 1:
        axes = [axes]

    for idx, (ax, (experiment, records)) in enumerate(zip(axes, comparisons)):
        total_hours, total_values = _plot_xy(records, "total_amount_gkg_km")
        convective_hours, convective_values = _plot_xy(records, "convective_amount_gkg_km")
        resolved_hours, resolved_values = _plot_xy(records, "resolved_amount_gkg_km")
        _, ref_total_values = _plot_xy(ref_records, "total_amount_gkg_km")
        _, ref_convective_values = _plot_xy(ref_records, "convective_amount_gkg_km")
        _, ref_resolved_values = _plot_xy(ref_records, "resolved_amount_gkg_km")
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.plot(
            total_hours,
            total_values - ref_total_values,
            color=TOTAL_COLOR,
            lw=TOTAL_LINEWIDTH,
            label="Total",
        )
        ax.plot(
            convective_hours,
            convective_values - ref_convective_values,
            color=CONVECTION_COLOR,
            lw=2.0,
            ls=CONVECTION_LINESTYLE,
            label="Convection scheme",
        )
        ax.plot(
            resolved_hours,
            resolved_values - ref_resolved_values,
            color=RESOLVED_COLOR,
            lw=2.0,
            ls=RESOLVED_LINESTYLE,
            label="Resolved microphysics",
        )
        label = EXPERIMENT_LABELS.get(experiment, experiment)
        ref_label = EXPERIMENT_LABELS.get(reference, reference)
        ax.set_title(f"{label} - {ref_label}", loc="left")
        ax.set_ylabel("Hourly difference\n(g kg$^{-1}$ km)")
        ax.set_xlim(0.0, 24.0)
        ax.set_ylim(*ylim)
        ax.grid(alpha=PANEL_GRID_ALPHA)
        ax.xaxis.set_major_formatter(tick_formatter())
        panel_label = PANEL_LABELS[idx] if idx < len(PANEL_LABELS) else f"({idx + 1})"
        ax.text(
            0.985,
            0.9,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
        if idx == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=PANEL_LEGEND_FONTSIZE)

    axes[-1].set_xlabel("Hour of day (UTC-4)")
    fig.suptitle(
        f"{layer_label} hourly condensation differences relative to C1M",
        fontweight="bold",
    )
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    agg_dir: Path = AGG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    layer_bottom_km: float = 0.0,
    layer_top_km: float | None = None,
    reference: str = "control",
) -> dict[str, Path]:
    data = build_dataset(
        agg_dir=agg_dir,
        layer_bottom_km=layer_bottom_km,
        layer_top_km=layer_top_km,
    )
    if layer_top_km is None:
        layer_slug = "0_to_freezing_level"
        layer_label = "0-freezing level"
    else:
        layer_slug = f"0_to_{float(layer_top_km):g}km".replace(".", "p")
        layer_label = f"0-{float(layer_top_km):g} km"

    absolute_figure = output_dir / f"{layer_slug}_condensation_diurnal_by_experiment.png"
    difference_figure = output_dir / f"{layer_slug}_condensation_diurnal_difference_vs_c1m.png"
    data_txt_dir = output_dir / DATA_TXT_SUBDIR
    absolute_txt = data_txt_dir / f"{layer_slug}_condensation_diurnal_by_experiment.txt"
    difference_txt = data_txt_dir / f"{layer_slug}_condensation_diurnal_difference_vs_c1m.txt"

    plot_absolute(data, absolute_figure, layer_label=layer_label)
    plot_difference(data, difference_figure, reference=reference, layer_label=layer_label)
    write_absolute_txt(
        data,
        absolute_txt,
        absolute_figure,
        layer_bottom_km=layer_bottom_km,
        layer_label=layer_label,
    )
    write_difference_txt(
        data,
        difference_txt,
        difference_figure,
        reference=reference,
        layer_bottom_km=layer_bottom_km,
        layer_label=layer_label,
    )

    outputs = {
        "absolute_figure": absolute_figure,
        "absolute_txt": absolute_txt,
        "difference_figure": difference_figure,
        "difference_txt": difference_txt,
    }
    for key, path in outputs.items():
        print(f"{key}: {path}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agg-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer-bottom-km", type=float, default=0.0)
    parser.add_argument(
        "--layer-top-km",
        default="freezing",
        help="Numeric altitude in km, or 'freezing' for each experiment's mean freezing level.",
    )
    parser.add_argument("--reference", default="control")
    args = parser.parse_args()
    run(
        agg_dir=args.agg_dir,
        output_dir=args.output_dir,
        layer_bottom_km=args.layer_bottom_km,
        layer_top_km=parse_layer_top(args.layer_top_km),
        reference=args.reference,
    )


def parse_layer_top(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"freezing", "freeze", "freezing_level", "fl"}:
        return None
    return float(text)


if __name__ == "__main__":
    main()
