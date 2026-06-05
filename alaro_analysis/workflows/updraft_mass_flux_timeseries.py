"""Time series of domain-mean high-percentile updraft mass flux."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.constants import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    G,
)
from alaro_analysis.common.timeparse import parse_utc_hour_from_name
from alaro_analysis.plotting.style import resolve_workers


from alaro_analysis.common.constants import RUNS_ROOT
DEFAULT_DATA_ROOT = RUNS_ROOT / "ALARO"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "updraft_mass_flux_timeseries"
FIGURE_NAME = "updraft_mass_flux_p90_domain_mean_timeseries.png"
TEXT_NAME = "updraft_mass_flux_p90_domain_mean_timeseries.txt"
CSV_NAME = "updraft_mass_flux_p90_domain_mean_timeseries.csv"
TOP_DAYS_FIGURE_TEMPLATE = "{slug}_top_{top_days}_updraft_days_p90_points.png"
TOP_DAYS_TEXT_TEMPLATE = "{slug}_top_{top_days}_updraft_days_p90_points.txt"
TOP_DAYS_CSV_TEMPLATE = "{slug}_top_{top_days}_updraft_days_p90_points.csv"
TOP_PERCENTILE_DAYS_FIGURE_TEMPLATE = "{slug}_top_p{day_percentile:g}_daily_updraft_days_p90_points.png"
TOP_PERCENTILE_DAYS_TEXT_TEMPLATE = "{slug}_top_p{day_percentile:g}_daily_updraft_days_p90_points.txt"
TOP_PERCENTILE_DAYS_CSV_TEMPLATE = "{slug}_top_p{day_percentile:g}_daily_updraft_days_p90_points.csv"
DAILY_TOP_DAYS_FIGURE_TEMPLATE = "{slug}_top_{top_days}_daily_updraft_days_p90_points.png"
DAILY_TOP_DAYS_TEXT_TEMPLATE = "{slug}_top_{top_days}_daily_updraft_days_p90_points.txt"
DAILY_TOP_DAYS_CSV_TEMPLATE = "{slug}_top_{top_days}_daily_updraft_days_p90_points.csv"


@dataclass(frozen=True)
class UpdraftFluxRecord:
    experiment: str
    valid_time: datetime
    day: str
    filename: str
    domain_mean_kg_m2_s: float
    active_p90_kg_m2_s: float
    top10_domain_mean_kg_m2_s: float
    top10_active_mean_kg_m2_s: float
    active_mean_kg_m2_s: float
    active_fraction: float
    finite_count: int
    active_count: int
    top10_count: int


@dataclass(frozen=True)
class DailyUpdraftFluxRecord:
    experiment: str
    date: datetime
    day: str
    n_hours: int
    daily_mean_domain_mean_kg_m2_s: float
    daily_mean_active_p90_kg_m2_s: float
    daily_mean_top10_domain_mean_kg_m2_s: float
    daily_mean_top10_active_mean_kg_m2_s: float
    daily_mean_active_mean_kg_m2_s: float
    daily_mean_active_fraction: float
    selected: bool = False


def experiment_slug(experiments: Sequence[str]) -> str:
    labels = [EXPERIMENT_LABELS.get(exp, exp).lower() for exp in experiments]
    return "_".join(label.replace("-", "_") for label in labels)


def valid_time_from_file(day_name: str, file_name: str) -> datetime | None:
    hour = parse_utc_hour_from_name(file_name)
    if hour is None:
        return None
    try:
        base = datetime.strptime(day_name[2:], "%Y%m%d")
    except ValueError:
        return None
    return base + timedelta(hours=hour)


def date_from_day_name(day_name: str) -> datetime | None:
    try:
        return datetime.strptime(day_name[2:], "%Y%m%d")
    except ValueError:
        return None


def read_nc_array(path: Path, variable: str) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        var_name = variable if variable in ds.data_vars else next(iter(ds.data_vars))
        return np.asarray(ds[var_name].isel(time=0).values, dtype=np.float64)


def updraft_mass_flux(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Parameterized updraft mass flux, positive upward, kg m-2 s-1."""
    return np.where(np.isfinite(omega) & np.isfinite(mesh) & (mesh > 0), (-omega * mesh) / G, 0.0)


def summarize_flux(
    flux: np.ndarray,
    finite_mask: np.ndarray,
    *,
    percentile: float,
) -> tuple[float, float, float, float, float, float, int, int, int]:
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0)

    finite_flux = np.where(finite_mask, flux, np.nan)
    domain_mean = float(np.nanmean(finite_flux))

    active = finite_mask & np.isfinite(flux) & (flux > 0.0)
    active_count = int(np.count_nonzero(active))
    active_fraction = active_count / finite_count
    if active_count == 0:
        return (domain_mean, np.nan, 0.0, np.nan, np.nan, active_fraction, finite_count, 0, 0)

    active_values = flux[active]
    active_p90 = float(np.nanpercentile(active_values, percentile))
    top = active & (flux >= active_p90)
    top10_count = int(np.count_nonzero(top))
    top_flux = np.where(top, flux, 0.0)
    top10_domain_mean = float(np.nanmean(np.where(finite_mask, top_flux, np.nan)))
    top10_active_mean = float(np.nanmean(flux[top])) if top10_count else np.nan
    active_mean = float(np.nanmean(active_values))
    return (
        domain_mean,
        active_p90,
        top10_domain_mean,
        top10_active_mean,
        active_mean,
        active_fraction,
        finite_count,
        active_count,
        top10_count,
    )


def list_day_dirs(data_root: Path, experiment: str) -> list[Path]:
    root = data_root / experiment / "masked-netcdf" / "UD_OMEGA"
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("pf"))


def process_day(task: tuple[str, Path, Path, float]) -> list[UpdraftFluxRecord]:
    experiment, data_root, day_dir, percentile = task
    mesh_day_dir = data_root / experiment / "masked-netcdf" / "UD_MESH_FRAC" / day_dir.name
    records: list[UpdraftFluxRecord] = []
    for omega_path in sorted(day_dir.glob("*.nc")):
        valid_time = valid_time_from_file(day_dir.name, omega_path.name)
        if valid_time is None:
            continue
        mesh_path = mesh_day_dir / omega_path.name
        if not mesh_path.exists():
            continue

        omega = read_nc_array(omega_path, "UD_OMEGA")
        mesh = read_nc_array(mesh_path, "UD_MESH_FRAC")
        n_levels = min(omega.shape[0], mesh.shape[0])
        omega = omega[:n_levels]
        mesh = mesh[:n_levels]
        finite_mask = np.isfinite(omega) & np.isfinite(mesh)
        flux = updraft_mass_flux(omega, mesh)
        (
            domain_mean,
            active_p90,
            top10_domain_mean,
            top10_active_mean,
            active_mean,
            active_fraction,
            finite_count,
            active_count,
            top10_count,
        ) = summarize_flux(flux, finite_mask, percentile=percentile)
        records.append(
            UpdraftFluxRecord(
                experiment=experiment,
                valid_time=valid_time,
                day=day_dir.name,
                filename=omega_path.name,
                domain_mean_kg_m2_s=domain_mean,
                active_p90_kg_m2_s=active_p90,
                top10_domain_mean_kg_m2_s=top10_domain_mean,
                top10_active_mean_kg_m2_s=top10_active_mean,
                active_mean_kg_m2_s=active_mean,
                active_fraction=active_fraction,
                finite_count=finite_count,
                active_count=active_count,
                top10_count=top10_count,
            )
        )
    return records


def build_timeseries(
    *,
    data_root: Path,
    experiments: Sequence[str],
    percentile: float,
    max_days: int | None,
    workers: int,
) -> list[UpdraftFluxRecord]:
    tasks: list[tuple[str, Path, Path, float]] = []
    for experiment in experiments:
        days = list_day_dirs(data_root, experiment)
        if max_days is not None:
            days = days[:max_days]
        tasks.extend((experiment, data_root, day, percentile) for day in days)

    if workers == 1:
        out: list[UpdraftFluxRecord] = []
        for idx, task in enumerate(tasks, start=1):
            out.extend(process_day(task))
            if idx % 25 == 0:
                print(f"processed {idx}/{len(tasks)} day tasks", flush=True)
        return sorted(out, key=lambda row: (row.experiment, row.valid_time))

    out = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(process_day, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                out.extend(future.result())
            except Exception as exc:
                print(f"[warn] skipped {task[0]} {task[2].name}: {exc}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"processed {done}/{len(tasks)} day tasks", flush=True)
    return sorted(out, key=lambda row: (row.experiment, row.valid_time))


def write_csv(path: Path, records: Sequence[UpdraftFluxRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment",
        "label",
        "valid_time_utc",
        "day",
        "filename",
        "domain_mean_kg_m2_s",
        "active_p90_kg_m2_s",
        "top10_domain_mean_kg_m2_s",
        "top10_active_mean_kg_m2_s",
        "active_mean_kg_m2_s",
        "active_fraction",
        "finite_count",
        "active_count",
        "top10_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "experiment": row.experiment,
                    "label": EXPERIMENT_LABELS.get(row.experiment, row.experiment),
                    "valid_time_utc": row.valid_time.isoformat(),
                    "day": row.day,
                    "filename": row.filename,
                    "domain_mean_kg_m2_s": f"{row.domain_mean_kg_m2_s:.12g}",
                    "active_p90_kg_m2_s": f"{row.active_p90_kg_m2_s:.12g}",
                    "top10_domain_mean_kg_m2_s": f"{row.top10_domain_mean_kg_m2_s:.12g}",
                    "top10_active_mean_kg_m2_s": f"{row.top10_active_mean_kg_m2_s:.12g}",
                    "active_mean_kg_m2_s": f"{row.active_mean_kg_m2_s:.12g}",
                    "active_fraction": f"{row.active_fraction:.12g}",
                    "finite_count": row.finite_count,
                    "active_count": row.active_count,
                    "top10_count": row.top10_count,
                }
            )


def write_txt(path: Path, records: Sequence[UpdraftFluxRecord], *, figure_path: Path, csv_path: Path, percentile: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "Updraft mass-flux high-percentile time series"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"CSV data: {csv_path}\n")
        fh.write("Formula: UPDRAFT_FLUX = (-UD_OMEGA * UD_MESH_FRAC) / g, positive upward.\n")
        fh.write("Inactive/invalid mesh cells are included as zero for domain-mean quantities.\n")
        fh.write(f"Percentile: active-cell spatial p{percentile:g} at each timestamp.\n")
        fh.write("Plotted metric: top10_domain_mean_kg_m2_s, i.e. domain mean of cells at/above the active-cell p90 threshold.\n")
        fh.write("Times are UTC. +0024 files are ignored to avoid duplicate valid times.\n\n")

        fh.write("Summary by experiment\n")
        fh.write("---------------------\n")
        fh.write("experiment,label,n_times,mean_domain_flux,mean_active_p90,mean_top10_domain_flux,mean_active_fraction\n")
        present_experiments = [exp for exp in EXPERIMENTS if any(row.experiment == exp for row in records)]
        for experiment in present_experiments:
            rows = [row for row in records if row.experiment == experiment]
            if not rows:
                continue
            fh.write(
                f"{experiment},{EXPERIMENT_LABELS[experiment]},{len(rows)},"
                f"{np.nanmean([r.domain_mean_kg_m2_s for r in rows]):.12g},"
                f"{np.nanmean([r.active_p90_kg_m2_s for r in rows]):.12g},"
                f"{np.nanmean([r.top10_domain_mean_kg_m2_s for r in rows]):.12g},"
                f"{np.nanmean([r.active_fraction for r in rows]):.12g}\n"
            )


def metric_value(row: UpdraftFluxRecord, metric: str) -> float:
    if not hasattr(row, metric):
        raise ValueError(f"Unknown metric: {metric}")
    return float(getattr(row, metric))


def rank_top_days(
    records: Sequence[UpdraftFluxRecord],
    *,
    metric: str,
    top_days: int,
) -> list[UpdraftFluxRecord]:
    best_by_day: dict[tuple[str, str], UpdraftFluxRecord] = {}
    for row in records:
        value = metric_value(row, metric)
        if not np.isfinite(value):
            continue
        key = (row.experiment, row.day)
        current = best_by_day.get(key)
        if current is None or value > metric_value(current, metric):
            best_by_day[key] = row
    ranked = sorted(best_by_day.values(), key=lambda row: metric_value(row, metric), reverse=True)
    return ranked[: max(1, int(top_days))]


def daily_metric_value(row: DailyUpdraftFluxRecord, metric: str) -> float:
    if not hasattr(row, metric):
        raise ValueError(f"Unknown daily metric: {metric}")
    return float(getattr(row, metric))


def aggregate_daily_records(records: Sequence[UpdraftFluxRecord]) -> list[DailyUpdraftFluxRecord]:
    grouped: dict[tuple[str, str], list[UpdraftFluxRecord]] = {}
    for row in records:
        grouped.setdefault((row.experiment, row.day), []).append(row)

    daily: list[DailyUpdraftFluxRecord] = []
    for (experiment, day), rows in grouped.items():
        date = date_from_day_name(day)
        if date is None:
            continue
        rows = sorted(rows, key=lambda row: row.valid_time)
        daily.append(
            DailyUpdraftFluxRecord(
                experiment=experiment,
                date=date,
                day=day,
                n_hours=len(rows),
                daily_mean_domain_mean_kg_m2_s=float(np.nanmean([row.domain_mean_kg_m2_s for row in rows])),
                daily_mean_active_p90_kg_m2_s=float(np.nanmean([row.active_p90_kg_m2_s for row in rows])),
                daily_mean_top10_domain_mean_kg_m2_s=float(np.nanmean([row.top10_domain_mean_kg_m2_s for row in rows])),
                daily_mean_top10_active_mean_kg_m2_s=float(np.nanmean([row.top10_active_mean_kg_m2_s for row in rows])),
                daily_mean_active_mean_kg_m2_s=float(np.nanmean([row.active_mean_kg_m2_s for row in rows])),
                daily_mean_active_fraction=float(np.nanmean([row.active_fraction for row in rows])),
            )
        )
    return sorted(daily, key=lambda row: (row.experiment, row.date))


def select_top_percentile_days(
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
    day_percentile: float,
) -> tuple[list[DailyUpdraftFluxRecord], float]:
    values = np.asarray([daily_metric_value(row, metric) for row in daily_records], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite daily values available for percentile selection.")
    threshold = float(np.nanpercentile(values, day_percentile))
    selected = [
        DailyUpdraftFluxRecord(
            experiment=row.experiment,
            date=row.date,
            day=row.day,
            n_hours=row.n_hours,
            daily_mean_domain_mean_kg_m2_s=row.daily_mean_domain_mean_kg_m2_s,
            daily_mean_active_p90_kg_m2_s=row.daily_mean_active_p90_kg_m2_s,
            daily_mean_top10_domain_mean_kg_m2_s=row.daily_mean_top10_domain_mean_kg_m2_s,
            daily_mean_top10_active_mean_kg_m2_s=row.daily_mean_top10_active_mean_kg_m2_s,
            daily_mean_active_mean_kg_m2_s=row.daily_mean_active_mean_kg_m2_s,
            daily_mean_active_fraction=row.daily_mean_active_fraction,
            selected=daily_metric_value(row, metric) >= threshold,
        )
        for row in daily_records
    ]
    return selected, threshold


def rank_daily_top_days(
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
    top_days: int,
) -> list[DailyUpdraftFluxRecord]:
    finite = [row for row in daily_records if np.isfinite(daily_metric_value(row, metric))]
    ranked = sorted(finite, key=lambda row: daily_metric_value(row, metric), reverse=True)
    return [
        DailyUpdraftFluxRecord(
            experiment=row.experiment,
            date=row.date,
            day=row.day,
            n_hours=row.n_hours,
            daily_mean_domain_mean_kg_m2_s=row.daily_mean_domain_mean_kg_m2_s,
            daily_mean_active_p90_kg_m2_s=row.daily_mean_active_p90_kg_m2_s,
            daily_mean_top10_domain_mean_kg_m2_s=row.daily_mean_top10_domain_mean_kg_m2_s,
            daily_mean_top10_active_mean_kg_m2_s=row.daily_mean_top10_active_mean_kg_m2_s,
            daily_mean_active_mean_kg_m2_s=row.daily_mean_active_mean_kg_m2_s,
            daily_mean_active_fraction=row.daily_mean_active_fraction,
            selected=True,
        )
        for row in ranked[: max(1, int(top_days))]
    ]


def write_top_days_csv(path: Path, records: Sequence[UpdraftFluxRecord], *, metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "experiment",
        "label",
        "day",
        "peak_valid_time_utc",
        "filename",
        "ranking_metric",
        "ranking_metric_value_kg_m2_s",
        "domain_mean_kg_m2_s",
        "active_p90_kg_m2_s",
        "top10_domain_mean_kg_m2_s",
        "top10_active_mean_kg_m2_s",
        "active_mean_kg_m2_s",
        "active_fraction",
        "finite_count",
        "active_count",
        "top10_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "experiment": row.experiment,
                    "label": EXPERIMENT_LABELS.get(row.experiment, row.experiment),
                    "day": row.day,
                    "peak_valid_time_utc": row.valid_time.isoformat(),
                    "filename": row.filename,
                    "ranking_metric": metric,
                    "ranking_metric_value_kg_m2_s": f"{metric_value(row, metric):.12g}",
                    "domain_mean_kg_m2_s": f"{row.domain_mean_kg_m2_s:.12g}",
                    "active_p90_kg_m2_s": f"{row.active_p90_kg_m2_s:.12g}",
                    "top10_domain_mean_kg_m2_s": f"{row.top10_domain_mean_kg_m2_s:.12g}",
                    "top10_active_mean_kg_m2_s": f"{row.top10_active_mean_kg_m2_s:.12g}",
                    "active_mean_kg_m2_s": f"{row.active_mean_kg_m2_s:.12g}",
                    "active_fraction": f"{row.active_fraction:.12g}",
                    "finite_count": row.finite_count,
                    "active_count": row.active_count,
                    "top10_count": row.top10_count,
                }
            )


def write_top_days_txt(
    path: Path,
    records: Sequence[UpdraftFluxRecord],
    *,
    figure_path: Path,
    csv_path: Path,
    percentile: float,
    metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "Top G1M updraft-mass-flux days"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"CSV data used for plotted points: {csv_path}\n")
        fh.write("Formula: UPDRAFT_FLUX = (-UD_OMEGA * UD_MESH_FRAC) / g, positive upward.\n")
        fh.write("Inactive/invalid mesh cells are included as zero for domain-mean quantities.\n")
        fh.write(f"Percentile: active-cell spatial p{percentile:g} at each timestamp.\n")
        fh.write(f"Ranking metric: daily maximum of {metric}.\n")
        fh.write("Each plotted point is the strongest hourly value within one ranked day. Times are UTC.\n\n")
        fh.write("rank,day,peak_valid_time_utc,ranking_metric_value_kg_m2_s,active_p90_kg_m2_s,active_fraction\n")
        for rank, row in enumerate(records, start=1):
            fh.write(
                f"{rank},{row.day},{row.valid_time.isoformat()},"
                f"{metric_value(row, metric):.12g},{row.active_p90_kg_m2_s:.12g},"
                f"{row.active_fraction:.12g}\n"
            )


def write_top_percentile_days_csv(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank_selected",
        "selected",
        "experiment",
        "label",
        "date",
        "day",
        "n_hours",
        "ranking_metric",
        "threshold_kg_m2_s",
        "ranking_metric_value_kg_m2_s",
        "daily_mean_domain_mean_kg_m2_s",
        "daily_mean_active_p90_kg_m2_s",
        "daily_mean_top10_domain_mean_kg_m2_s",
        "daily_mean_top10_active_mean_kg_m2_s",
        "daily_mean_active_mean_kg_m2_s",
        "daily_mean_active_fraction",
    ]
    selected_sorted = sorted(
        [row for row in daily_records if row.selected],
        key=lambda row: daily_metric_value(row, metric),
        reverse=True,
    )
    selected_rank = {(row.experiment, row.day): rank for rank, row in enumerate(selected_sorted, start=1)}
    all_sorted = sorted(daily_records, key=lambda row: (not row.selected, -daily_metric_value(row, metric)))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in all_sorted:
            writer.writerow(
                {
                    "rank_selected": selected_rank.get((row.experiment, row.day), ""),
                    "selected": int(row.selected),
                    "experiment": row.experiment,
                    "label": EXPERIMENT_LABELS.get(row.experiment, row.experiment),
                    "date": row.date.date().isoformat(),
                    "day": row.day,
                    "n_hours": row.n_hours,
                    "ranking_metric": metric,
                    "threshold_kg_m2_s": f"{threshold:.12g}",
                    "ranking_metric_value_kg_m2_s": f"{daily_metric_value(row, metric):.12g}",
                    "daily_mean_domain_mean_kg_m2_s": f"{row.daily_mean_domain_mean_kg_m2_s:.12g}",
                    "daily_mean_active_p90_kg_m2_s": f"{row.daily_mean_active_p90_kg_m2_s:.12g}",
                    "daily_mean_top10_domain_mean_kg_m2_s": f"{row.daily_mean_top10_domain_mean_kg_m2_s:.12g}",
                    "daily_mean_top10_active_mean_kg_m2_s": f"{row.daily_mean_top10_active_mean_kg_m2_s:.12g}",
                    "daily_mean_active_mean_kg_m2_s": f"{row.daily_mean_active_mean_kg_m2_s:.12g}",
                    "daily_mean_active_fraction": f"{row.daily_mean_active_fraction:.12g}",
                }
            )


def write_top_percentile_days_txt(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    figure_path: Path,
    csv_path: Path,
    percentile: float,
    day_percentile: float,
    metric: str,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(
        [row for row in daily_records if row.selected],
        key=lambda row: daily_metric_value(row, metric),
        reverse=True,
    )
    title = f"G1M daily top-p{day_percentile:g} updraft-mass-flux days"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"CSV data used for plotted points: {csv_path}\n")
        fh.write("Formula: UPDRAFT_FLUX = (-UD_OMEGA * UD_MESH_FRAC) / g, positive upward.\n")
        fh.write("Inactive/invalid mesh cells are included as zero for domain-mean quantities.\n")
        fh.write(f"Hourly spatial percentile: active-cell p{percentile:g} at each timestamp.\n")
        fh.write(f"Daily selection percentile: p{day_percentile:g} across daily means.\n")
        fh.write(f"Daily ranking metric: {metric}.\n")
        fh.write(f"Selection threshold: {threshold:.12g} kg m^-2 s^-1.\n")
        fh.write(f"Selected days: {len(selected)} of {len(daily_records)}.\n\n")
        fh.write("Selected Dates\n")
        fh.write("--------------\n")
        for row in selected:
            fh.write(row.date.date().isoformat() + "\n")
        fh.write("\nrank,date,ranking_metric_value_kg_m2_s,daily_mean_active_p90_kg_m2_s,n_hours\n")
        for rank, row in enumerate(selected, start=1):
            fh.write(
                f"{rank},{row.date.date().isoformat()},"
                f"{daily_metric_value(row, metric):.12g},"
                f"{row.daily_mean_active_p90_kg_m2_s:.12g},{row.n_hours}\n"
            )


def write_daily_top_days_csv(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "experiment",
        "label",
        "date",
        "day",
        "n_hours",
        "ranking_metric",
        "ranking_metric_value_kg_m2_s",
        "daily_mean_domain_mean_kg_m2_s",
        "daily_mean_active_p90_kg_m2_s",
        "daily_mean_top10_domain_mean_kg_m2_s",
        "daily_mean_top10_active_mean_kg_m2_s",
        "daily_mean_active_mean_kg_m2_s",
        "daily_mean_active_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(daily_records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "experiment": row.experiment,
                    "label": EXPERIMENT_LABELS.get(row.experiment, row.experiment),
                    "date": row.date.date().isoformat(),
                    "day": row.day,
                    "n_hours": row.n_hours,
                    "ranking_metric": metric,
                    "ranking_metric_value_kg_m2_s": f"{daily_metric_value(row, metric):.12g}",
                    "daily_mean_domain_mean_kg_m2_s": f"{row.daily_mean_domain_mean_kg_m2_s:.12g}",
                    "daily_mean_active_p90_kg_m2_s": f"{row.daily_mean_active_p90_kg_m2_s:.12g}",
                    "daily_mean_top10_domain_mean_kg_m2_s": f"{row.daily_mean_top10_domain_mean_kg_m2_s:.12g}",
                    "daily_mean_top10_active_mean_kg_m2_s": f"{row.daily_mean_top10_active_mean_kg_m2_s:.12g}",
                    "daily_mean_active_mean_kg_m2_s": f"{row.daily_mean_active_mean_kg_m2_s:.12g}",
                    "daily_mean_active_fraction": f"{row.daily_mean_active_fraction:.12g}",
                }
            )


def write_daily_top_days_txt(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    figure_path: Path,
    csv_path: Path,
    percentile: float,
    metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = f"G1M top {len(daily_records)} daily updraft-mass-flux days"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"CSV data used for plotted points: {csv_path}\n")
        fh.write("Formula: UPDRAFT_FLUX = (-UD_OMEGA * UD_MESH_FRAC) / g, positive upward.\n")
        fh.write("Inactive/invalid mesh cells are included as zero for domain-mean quantities.\n")
        fh.write(f"Hourly spatial percentile: active-cell p{percentile:g} at each timestamp.\n")
        fh.write(f"Daily ranking metric: {metric}.\n")
        fh.write("Selection: top daily means, not hourly peaks.\n\n")
        fh.write("Selected Dates\n")
        fh.write("--------------\n")
        for row in daily_records:
            fh.write(row.date.date().isoformat() + "\n")
        fh.write("\nrank,date,ranking_metric_value_kg_m2_s,daily_mean_active_p90_kg_m2_s,n_hours\n")
        for rank, row in enumerate(daily_records, start=1):
            fh.write(
                f"{rank},{row.date.date().isoformat()},"
                f"{daily_metric_value(row, metric):.12g},"
                f"{row.daily_mean_active_p90_kg_m2_s:.12g},{row.n_hours}\n"
            )


def plot_timeseries(path: Path, records: Sequence[UpdraftFluxRecord], *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    present_experiments = [exp for exp in EXPERIMENTS if any(row.experiment == exp for row in records)]
    for experiment in present_experiments:
        rows = [row for row in records if row.experiment == experiment]
        if not rows:
            continue
        times = [row.valid_time for row in rows]
        values = np.asarray([row.top10_domain_mean_kg_m2_s for row in rows], dtype=np.float64)
        ax.plot(
            times,
            values,
            color=EXPERIMENT_COLORS[experiment],
            lw=0.65,
            alpha=0.9,
            label=EXPERIMENT_LABELS[experiment],
        )

    ax.set_ylabel(r"Domain mean of top 10% updraft flux (kg m$^{-2}$ s$^{-1}$)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_top_days(path: Path, records: Sequence[UpdraftFluxRecord], *, metric: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    values = np.asarray([metric_value(row, metric) for row in records], dtype=np.float64)
    colors = np.asarray([row.active_p90_kg_m2_s for row in records], dtype=np.float64)
    sizes = 35.0 + 350.0 * np.asarray([row.active_fraction for row in records], dtype=np.float64)
    scatter = ax.scatter(
        [row.valid_time for row in records],
        values,
        c=colors,
        s=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
        alpha=0.92,
    )
    ax.set_ylabel(r"Domain mean of top 10% updraft flux (kg m$^{-2}$ s$^{-1}$)")
    ax.set_xlabel("Peak hour of selected day (UTC)")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label(r"Active-cell P90 flux (kg m$^{-2}$ s$^{-1}$)")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_top_percentile_days(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
    threshold: float,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in daily_records if row.selected]
    other = [row for row in daily_records if not row.selected]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    if other:
        ax.scatter(
            [row.date for row in other],
            [daily_metric_value(row, metric) for row in other],
            s=16,
            color="0.78",
            alpha=0.55,
            linewidths=0,
            label="below p95",
        )
    if selected:
        sizes = 55.0 + 350.0 * np.asarray([row.daily_mean_active_fraction for row in selected], dtype=np.float64)
        scatter = ax.scatter(
            [row.date for row in selected],
            [daily_metric_value(row, metric) for row in selected],
            c=[row.daily_mean_active_p90_kg_m2_s for row in selected],
            s=sizes,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.45,
            alpha=0.95,
            label="daily >= p95",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
        cbar.set_label(r"Daily mean active-cell P90 flux (kg m$^{-2}$ s$^{-1}$)")

    ax.axhline(threshold, color="black", lw=1.0, ls="--", label="p95 threshold")
    ax.set_ylabel(r"Daily mean top-10% updraft flux (kg m$^{-2}$ s$^{-1}$)")
    ax.set_xlabel("Date (UTC)")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_daily_top_days(
    path: Path,
    daily_records: Sequence[DailyUpdraftFluxRecord],
    *,
    metric: str,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    values = np.asarray([daily_metric_value(row, metric) for row in daily_records], dtype=np.float64)
    colors = np.asarray([row.daily_mean_active_p90_kg_m2_s for row in daily_records], dtype=np.float64)
    sizes = 70.0 + 420.0 * np.asarray([row.daily_mean_active_fraction for row in daily_records], dtype=np.float64)
    scatter = ax.scatter(
        [row.date for row in daily_records],
        values,
        c=colors,
        s=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
    )
    for rank, row in enumerate(daily_records, start=1):
        ax.annotate(
            str(rank),
            (row.date, daily_metric_value(row, metric)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel(r"Daily mean top-10% updraft flux (kg m$^{-2}$ s$^{-1}$)")
    ax.set_xlabel("Date (UTC)")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label(r"Daily mean active-cell P90 flux (kg m$^{-2}$ s$^{-1}$)")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_outputs(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiments: Sequence[str] = EXPERIMENTS,
    percentile: float = 90.0,
    plot_kind: str = "timeseries",
    top_days: int = 40,
    day_percentile: float = 95.0,
    max_days: int | None = None,
    workers: int = 32,
    dpi: int = 400,
) -> dict[str, Path]:
    workers = resolve_workers(workers)
    records = build_timeseries(
        data_root=data_root,
        experiments=experiments,
        percentile=percentile,
        max_days=max_days,
        workers=workers,
    )
    if not records:
        raise RuntimeError("No updraft mass-flux records were produced.")

    if plot_kind == "top-days":
        slug = experiment_slug(experiments)
        metric = "top10_domain_mean_kg_m2_s"
        top_records = rank_top_days(records, metric=metric, top_days=top_days)
        figure_path = output_dir / TOP_DAYS_FIGURE_TEMPLATE.format(slug=slug, top_days=top_days)
        csv_path = output_dir / "data_txt" / TOP_DAYS_CSV_TEMPLATE.format(slug=slug, top_days=top_days)
        txt_path = output_dir / "data_txt" / TOP_DAYS_TEXT_TEMPLATE.format(slug=slug, top_days=top_days)
        write_top_days_csv(csv_path, top_records, metric=metric)
        write_top_days_txt(
            txt_path,
            top_records,
            figure_path=figure_path,
            csv_path=csv_path,
            percentile=percentile,
            metric=metric,
        )
        plot_top_days(figure_path, top_records, metric=metric, dpi=dpi)
    elif plot_kind == "top-percentile-days":
        slug = experiment_slug(experiments)
        metric = "daily_mean_top10_domain_mean_kg_m2_s"
        daily_records = aggregate_daily_records(records)
        daily_records, threshold = select_top_percentile_days(
            daily_records,
            metric=metric,
            day_percentile=day_percentile,
        )
        figure_path = output_dir / TOP_PERCENTILE_DAYS_FIGURE_TEMPLATE.format(
            slug=slug,
            day_percentile=day_percentile,
        )
        csv_path = output_dir / "data_txt" / TOP_PERCENTILE_DAYS_CSV_TEMPLATE.format(
            slug=slug,
            day_percentile=day_percentile,
        )
        txt_path = output_dir / "data_txt" / TOP_PERCENTILE_DAYS_TEXT_TEMPLATE.format(
            slug=slug,
            day_percentile=day_percentile,
        )
        write_top_percentile_days_csv(csv_path, daily_records, metric=metric, threshold=threshold)
        write_top_percentile_days_txt(
            txt_path,
            daily_records,
            figure_path=figure_path,
            csv_path=csv_path,
            percentile=percentile,
            day_percentile=day_percentile,
            metric=metric,
            threshold=threshold,
        )
        plot_top_percentile_days(figure_path, daily_records, metric=metric, threshold=threshold, dpi=dpi)
    elif plot_kind == "daily-top-days":
        slug = experiment_slug(experiments)
        metric = "daily_mean_top10_domain_mean_kg_m2_s"
        daily_records = aggregate_daily_records(records)
        top_daily_records = rank_daily_top_days(daily_records, metric=metric, top_days=top_days)
        figure_path = output_dir / DAILY_TOP_DAYS_FIGURE_TEMPLATE.format(slug=slug, top_days=top_days)
        csv_path = output_dir / "data_txt" / DAILY_TOP_DAYS_CSV_TEMPLATE.format(slug=slug, top_days=top_days)
        txt_path = output_dir / "data_txt" / DAILY_TOP_DAYS_TEXT_TEMPLATE.format(slug=slug, top_days=top_days)
        write_daily_top_days_csv(csv_path, top_daily_records, metric=metric)
        write_daily_top_days_txt(
            txt_path,
            top_daily_records,
            figure_path=figure_path,
            csv_path=csv_path,
            percentile=percentile,
            metric=metric,
        )
        plot_daily_top_days(figure_path, top_daily_records, metric=metric, dpi=dpi)
    else:
        figure_path = output_dir / FIGURE_NAME
        csv_path = output_dir / "data_txt" / CSV_NAME
        txt_path = output_dir / "data_txt" / TEXT_NAME
        write_csv(csv_path, records)
        write_txt(txt_path, records, figure_path=figure_path, csv_path=csv_path, percentile=percentile)
        plot_timeseries(figure_path, records, dpi=dpi)
    return {"plot": figure_path, "txt": txt_path, "csv": csv_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a time series of high-percentile domain-mean updraft mass flux."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument(
        "--plot-kind",
        choices=("timeseries", "top-days", "top-percentile-days", "daily-top-days"),
        default="timeseries",
    )
    parser.add_argument("--top-days", type=int, default=40)
    parser.add_argument("--day-percentile", type=float, default=95.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=450)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = make_outputs(
        data_root=args.data_root,
        output_dir=args.output_dir,
        experiments=tuple(args.experiments),
        percentile=args.percentile,
        plot_kind=args.plot_kind,
        top_days=args.top_days,
        day_percentile=args.day_percentile,
        max_days=args.max_days,
        workers=args.workers,
        dpi=args.dpi,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
