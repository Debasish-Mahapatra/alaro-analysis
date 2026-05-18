#!/usr/bin/env python3
"""Raw-FA updraft diagnostics over the DDH namelist domain.

This recomputes the parameterized updraft diagnostics directly from raw FA
files, using the NAMDDH rectangle rather than the radar mask.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS, G
from alaro_analysis.ddh.io import AGG_DIR
from alaro_analysis.ddh.plot_g1m_c1m_process_fingerprint import (
    FingerprintRow,
    make_plot as make_fingerprint_plot,
)
from alaro_analysis.ddh.plot_warm_layer_pathway_summary import compute_layer_metrics
from alaro_analysis.plotting.style import resolve_workers


RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
DEFAULT_DATA_ROOT = RUNS_ROOT / "ALARO"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "plots" / "analysis-on-the-ddh-domain"
DEFAULT_LEAD = "0024"

HOUR_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)?$")
UTC_OFFSET_HOURS = -4

DDH_LON_MIN = -61.269
DDH_LON_MAX = -59.927
DDH_LAT_MIN = -3.884
DDH_LAT_MAX = -2.544

METRICS = ("updraft_extent", "updraft_flux", "updraft_intensity")
METRIC_LABELS = {
    "updraft_extent": "Updraft extent",
    "updraft_flux": "Updraft mass flux",
    "updraft_intensity": "Updraft intensity",
}
METRIC_UNITS = {
    "updraft_extent": "fraction",
    "updraft_flux": r"kg m$^{-2}$ s$^{-1}$",
    "updraft_intensity": r"Pa s$^{-1}$",
}
TEXT_UNITS = {
    "updraft_extent": "fraction",
    "updraft_flux": "kg m-2 s-1",
    "updraft_intensity": "Pa s-1",
}


@dataclass(frozen=True)
class DomainWindow:
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    mask: np.ndarray
    height_m: np.ndarray
    sample_file: str


@dataclass(frozen=True)
class DayResult:
    experiment: str
    n_files: int
    sums: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RainfallResult:
    experiment: str
    n_hours: int
    hourly_sum: np.ndarray
    hourly_count: np.ndarray
    total_sum: float
    total_count: int
    warnings: tuple[str, ...]


def parse_hour(path: Path) -> int | None:
    match = HOUR_RE.search(path.name)
    if match is None:
        return None
    hour = int(match.group(1))
    if 0 <= hour <= 24:
        return hour
    return None


def day_dirs(root: Path, max_days: int | None) -> list[Path]:
    out = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("pf"))
    if max_days is not None:
        out = out[:max_days]
    return out


def fa_file_for_hour(day_dir: Path, hour: int) -> Path:
    return day_dir / f"pfABOFABOF+{hour:04d}"


def open_fa_dataset(path: Path, variables: Sequence[str]):
    import faxarray as fx

    return fx.open_dataset(str(path), variables=list(variables), stack_levels=True)


def data_var(ds, requested: str) -> str:
    if requested in ds.data_vars:
        return requested
    compact = requested.replace(".", "_")
    if compact in ds.data_vars:
        return compact
    if "." in requested:
        token = requested.replace(".", "").upper()
        for name in ds.data_vars:
            if name.replace("_", "").replace(".", "").upper() == token:
                return name
    raise KeyError(f"{requested!r} not found; available={list(ds.data_vars)}")


def build_domain_window(sample_file: Path) -> DomainWindow:
    with open_fa_dataset(sample_file, ["GEOPOTENTIEL"]) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        full_mask = (
            (lon >= DDH_LON_MIN)
            & (lon <= DDH_LON_MAX)
            & (lat >= DDH_LAT_MIN)
            & (lat <= DDH_LAT_MAX)
        )
        ys, xs = np.where(full_mask)
        if ys.size == 0:
            raise RuntimeError("DDH lon/lat box selects no raw-FA grid cells.")
        y_start, y_stop = int(ys.min()), int(ys.max()) + 1
        x_start, x_stop = int(xs.min()), int(xs.max()) + 1
        mask = full_mask[y_start:y_stop, x_start:x_stop]
        gz_name = data_var(ds, "GEOPOTENTIEL")
        gz = np.asarray(
            ds[gz_name]
            .isel(time=0, y=slice(y_start, y_stop), x=slice(x_start, x_stop))
            .values,
            dtype=np.float64,
        )
    height_m = np.nanmean(np.where(mask[np.newaxis, :, :], gz / G, np.nan), axis=(1, 2))
    return DomainWindow(
        y_start=y_start,
        y_stop=y_stop,
        x_start=x_start,
        x_stop=x_stop,
        mask=mask,
        height_m=height_m,
        sample_file=str(sample_file),
    )


def profile_from_fa(path: Path, window: DomainWindow) -> dict[str, np.ndarray]:
    with open_fa_dataset(path, ["UD_OMEGA", "UD_MESH_FRAC"]) as ds:
        omega = np.asarray(
            ds[data_var(ds, "UD_OMEGA")]
            .isel(
                time=0,
                y=slice(window.y_start, window.y_stop),
                x=slice(window.x_start, window.x_stop),
            )
            .values,
            dtype=np.float64,
        )
        mesh = np.asarray(
            ds[data_var(ds, "UD_MESH_FRAC")]
            .isel(
                time=0,
                y=slice(window.y_start, window.y_stop),
                x=slice(window.x_start, window.x_stop),
            )
            .values,
            dtype=np.float64,
        )

    if omega.shape != mesh.shape:
        raise ValueError(f"UD_OMEGA/UD_MESH_FRAC shape mismatch for {path}: {omega.shape} vs {mesh.shape}")

    domain_mask = window.mask[np.newaxis, :, :]
    finite = np.isfinite(omega) & np.isfinite(mesh) & domain_mask
    extent = np.where(finite, mesh, np.nan)
    flux = np.where(finite, np.where(mesh > 0.0, (-omega * mesh) / G, 0.0), np.nan)
    intensity = np.where(finite, np.where(mesh > 0.0, np.abs(omega), 0.0), np.nan)
    return {
        "updraft_extent": np.nanmean(extent, axis=(1, 2)),
        "updraft_flux": np.nanmean(flux, axis=(1, 2)),
        "updraft_intensity": np.nanmean(intensity, axis=(1, 2)),
    }


def process_updraft_day(task: tuple[str, str, DomainWindow]) -> DayResult:
    experiment, day_dir_raw, window = task
    day_dir = Path(day_dir_raw)
    n_levels = window.height_m.size
    sums = {metric: np.zeros((n_levels, 24), dtype=np.float64) for metric in METRICS}
    counts = {metric: np.zeros((n_levels, 24), dtype=np.int64) for metric in METRICS}
    warnings: list[str] = []
    n_files = 0

    for path in sorted(day_dir.iterdir()):
        if not path.is_file():
            continue
        hour = parse_hour(path)
        if hour is None or hour == 24:
            continue
        local_hour = (hour + UTC_OFFSET_HOURS) % 24
        try:
            profiles = profile_from_fa(path, window)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{experiment} {path}: {exc}")
            continue
        n_files += 1
        for metric, profile in profiles.items():
            valid = np.isfinite(profile)
            sums[metric][valid, local_hour] += profile[valid]
            counts[metric][valid, local_hour] += 1

    return DayResult(
        experiment=experiment,
        n_files=n_files,
        sums=sums,
        counts=counts,
        warnings=tuple(warnings),
    )


def rainfall_domain_mean(path: Path, window: DomainWindow) -> float:
    with open_fa_dataset(path, ["SURFPREC.EAU.GEC", "SURFPREC.EAU.CON"]) as ds:
        gec = np.asarray(
            ds[data_var(ds, "SURFPREC.EAU.GEC")]
            .isel(time=0, y=slice(window.y_start, window.y_stop), x=slice(window.x_start, window.x_stop))
            .values,
            dtype=np.float64,
        )
        con = np.asarray(
            ds[data_var(ds, "SURFPREC.EAU.CON")]
            .isel(time=0, y=slice(window.y_start, window.y_stop), x=slice(window.x_start, window.x_stop))
            .values,
            dtype=np.float64,
        )
    total = gec + con
    return float(np.nanmean(np.where(window.mask, total, np.nan)))


def process_rainfall_day(task: tuple[str, str, DomainWindow]) -> RainfallResult:
    experiment, day_dir_raw, window = task
    day_dir = Path(day_dir_raw)
    cumulative: dict[int, float] = {}
    warnings: list[str] = []

    for hour in range(25):
        path = fa_file_for_hour(day_dir, hour)
        if not path.exists():
            warnings.append(f"{experiment} {path}: missing")
            continue
        try:
            cumulative[hour] = rainfall_domain_mean(path, window)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{experiment} {path}: {exc}")

    hourly_sum = np.zeros(24, dtype=np.float64)
    hourly_count = np.zeros(24, dtype=np.int64)
    total_sum = 0.0
    total_count = 0
    for hour in range(1, 25):
        if hour not in cumulative or (hour - 1) not in cumulative:
            continue
        value = cumulative[hour] - cumulative[hour - 1]
        if not np.isfinite(value):
            continue
        value = max(float(value), 0.0)
        local_hour = (hour + UTC_OFFSET_HOURS) % 24
        hourly_sum[local_hour] += value
        hourly_count[local_hour] += 1
        total_sum += value
        total_count += 1

    return RainfallResult(
        experiment=experiment,
        n_hours=total_count,
        hourly_sum=hourly_sum,
        hourly_count=hourly_count,
        total_sum=total_sum,
        total_count=total_count,
        warnings=tuple(warnings),
    )


def load_profile_cache(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int], dict[str, str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        profiles: dict[str, dict[str, np.ndarray]] = {}
        n_files: dict[str, int] = {}
        sample_files: dict[str, str] = {}
        for exp in EXPERIMENTS:
            profiles[exp] = {}
            for metric in METRICS:
                profiles[exp][metric] = np.asarray(data[f"{exp}_{metric}_mean"], dtype=np.float64)
                profiles[exp][f"{metric}_counts"] = np.asarray(data[f"{exp}_{metric}_counts"], dtype=np.int64)
            n_files[exp] = int(data[f"{exp}_n_files"][0])
            sample_files[exp] = str(data[f"{exp}_sample_file"][0])
        height_m = np.asarray(data["height_m"], dtype=np.float64)
    return profiles, n_files, sample_files, height_m


def save_profile_cache(
    path: Path,
    *,
    profiles: dict[str, dict[str, np.ndarray]],
    counts: dict[str, dict[str, np.ndarray]],
    n_files: dict[str, int],
    sample_files: dict[str, str],
    height_m: np.ndarray,
    window: DomainWindow,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "height_m": np.asarray(height_m, dtype=np.float64),
        "ddh_bounds": np.asarray([DDH_LON_MIN, DDH_LON_MAX, DDH_LAT_MIN, DDH_LAT_MAX], dtype=np.float64),
        "window_yx": np.asarray([window.y_start, window.y_stop, window.x_start, window.x_stop], dtype=np.int64),
        "window_mask": np.asarray(window.mask, dtype=bool),
    }
    for exp in EXPERIMENTS:
        payload[f"{exp}_n_files"] = np.asarray([n_files[exp]], dtype=np.int64)
        payload[f"{exp}_sample_file"] = np.asarray([sample_files[exp]])
        for metric in METRICS:
            payload[f"{exp}_{metric}_mean"] = np.asarray(profiles[exp][metric], dtype=np.float64)
            payload[f"{exp}_{metric}_counts"] = np.asarray(counts[exp][metric], dtype=np.int64)
    np.savez_compressed(path, **payload)


def compute_updraft_profiles(
    *,
    data_root: Path,
    output_dir: Path,
    max_days: int | None,
    workers: int,
    force: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, int], dict[str, str], np.ndarray, DomainWindow]:
    cache_path = output_dir / "data_txt" / "raw_fa_ddh_updraft_diurnal_profiles.npz"
    sample_file = fa_file_for_hour(day_dirs(data_root / "control" / "untar-output", 1)[0], 0)
    window = build_domain_window(sample_file)
    if cache_path.exists() and not force:
        loaded_profiles, n_files, sample_files, height_m = load_profile_cache(cache_path)
        loaded_counts = {
            exp: {metric: loaded_profiles[exp].pop(f"{metric}_counts") for metric in METRICS}
            for exp in EXPERIMENTS
        }
        return loaded_profiles, loaded_counts, n_files, sample_files, height_m, window

    workers = resolve_workers(workers)
    all_profiles: dict[str, dict[str, np.ndarray]] = {}
    all_counts: dict[str, dict[str, np.ndarray]] = {}
    n_files: dict[str, int] = {}
    sample_files: dict[str, str] = {}
    warnings: list[str] = []

    for exp in EXPERIMENTS:
        exp_root = data_root / exp / "untar-output"
        days = day_dirs(exp_root, max_days)
        if not days:
            raise RuntimeError(f"No raw FA day directories found under {exp_root}")

        n_levels = window.height_m.size
        sums = {metric: np.zeros((n_levels, 24), dtype=np.float64) for metric in METRICS}
        counts = {metric: np.zeros((n_levels, 24), dtype=np.int64) for metric in METRICS}
        done = 0
        total_files = 0
        tasks = [(exp, str(day), window) for day in days]
        print(f"[{exp}] raw-FA updraft tasks: {len(tasks)} days", flush=True)
        if workers == 1:
            iterator: Iterable[DayResult] = (process_updraft_day(task) for task in tasks)
            for result in iterator:
                done += 1
                total_files += result.n_files
                warnings.extend(result.warnings)
                for metric in METRICS:
                    sums[metric] += result.sums[metric]
                    counts[metric] += result.counts[metric]
                if done % 25 == 0 or done == len(tasks):
                    print(f"[{exp}] {done}/{len(tasks)} days, {total_files} files", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_updraft_day, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    total_files += result.n_files
                    warnings.extend(result.warnings)
                    for metric in METRICS:
                        sums[metric] += result.sums[metric]
                        counts[metric] += result.counts[metric]
                    if done % 25 == 0 or done == len(tasks):
                        print(f"[{exp}] {done}/{len(tasks)} days, {total_files} files", flush=True)

        profiles = {}
        for metric in METRICS:
            mean = np.full_like(sums[metric], np.nan)
            valid = counts[metric] > 0
            mean[valid] = sums[metric][valid] / counts[metric][valid]
            profiles[metric] = mean
        all_profiles[exp] = profiles
        all_counts[exp] = counts
        n_files[exp] = total_files
        sample_files[exp] = str(fa_file_for_hour(days[0], 0))

    if warnings:
        warn_path = output_dir / "data_txt" / "raw_fa_ddh_updraft_warnings.txt"
        warn_path.parent.mkdir(parents=True, exist_ok=True)
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[warn] wrote {warn_path}", flush=True)

    save_profile_cache(
        cache_path,
        profiles=all_profiles,
        counts=all_counts,
        n_files=n_files,
        sample_files=sample_files,
        height_m=window.height_m,
        window=window,
    )
    print(f"[saved] {cache_path}", flush=True)
    return all_profiles, all_counts, n_files, sample_files, window.height_m, window


def compute_rainfall(
    *,
    data_root: Path,
    output_dir: Path,
    window: DomainWindow,
    max_days: int | None,
    workers: int,
    force: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    cache_path = output_dir / "data_txt" / "raw_fa_ddh_rainfall_summary.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            mean = {exp: float(data[f"{exp}_mean_mm_h"][0]) for exp in EXPERIMENTS}
            hourly = {exp: np.asarray(data[f"{exp}_hourly_mean_mm_h"], dtype=np.float64) for exp in EXPERIMENTS}
            hourly_count = {exp: np.asarray(data[f"{exp}_hourly_count"], dtype=np.int64) for exp in EXPERIMENTS}
            n_hours = {exp: int(data[f"{exp}_n_hours"][0]) for exp in EXPERIMENTS}
        return mean, hourly, hourly_count, n_hours

    workers = resolve_workers(workers)
    mean: dict[str, float] = {}
    hourly: dict[str, np.ndarray] = {}
    hourly_count: dict[str, np.ndarray] = {}
    n_hours: dict[str, int] = {}
    warnings: list[str] = []

    for exp in EXPERIMENTS:
        exp_root = data_root / exp / "untar-output"
        days = day_dirs(exp_root, max_days)
        tasks = [(exp, str(day), window) for day in days]
        done = 0
        hsum = np.zeros(24, dtype=np.float64)
        hcount = np.zeros(24, dtype=np.int64)
        total_sum = 0.0
        total_count = 0
        print(f"[{exp}] raw-FA rainfall tasks: {len(tasks)} days", flush=True)
        if workers == 1:
            iterator: Iterable[RainfallResult] = (process_rainfall_day(task) for task in tasks)
            for result in iterator:
                done += 1
                hsum += result.hourly_sum
                hcount += result.hourly_count
                total_sum += result.total_sum
                total_count += result.total_count
                warnings.extend(result.warnings)
                if done % 50 == 0 or done == len(tasks):
                    print(f"[{exp}/rain] {done}/{len(tasks)} days", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_rainfall_day, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    hsum += result.hourly_sum
                    hcount += result.hourly_count
                    total_sum += result.total_sum
                    total_count += result.total_count
                    warnings.extend(result.warnings)
                    if done % 50 == 0 or done == len(tasks):
                        print(f"[{exp}/rain] {done}/{len(tasks)} days", flush=True)

        hourly_mean = np.full(24, np.nan, dtype=np.float64)
        ok = hcount > 0
        hourly_mean[ok] = hsum[ok] / hcount[ok]
        mean[exp] = total_sum / total_count if total_count else np.nan
        hourly[exp] = hourly_mean
        hourly_count[exp] = hcount
        n_hours[exp] = total_count

    if warnings:
        warn_path = output_dir / "data_txt" / "raw_fa_ddh_rainfall_warnings.txt"
        warn_path.parent.mkdir(parents=True, exist_ok=True)
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[warn] wrote {warn_path}", flush=True)

    payload = {}
    for exp in EXPERIMENTS:
        payload[f"{exp}_mean_mm_h"] = np.asarray([mean[exp]], dtype=np.float64)
        payload[f"{exp}_hourly_mean_mm_h"] = hourly[exp]
        payload[f"{exp}_hourly_count"] = hourly_count[exp]
        payload[f"{exp}_n_hours"] = np.asarray([n_hours[exp]], dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"[saved] {cache_path}", flush=True)
    return mean, hourly, hourly_count, n_hours


def center_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mid, [last]])


def symmetric_limit(values: Sequence[np.ndarray], percentile: float = 98.0) -> float:
    parts = [np.abs(np.asarray(v, dtype=np.float64)).ravel() for v in values]
    finite = np.concatenate(parts)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return limit if limit > 0.0 else 1.0


def plot_metric_panel(
    metric: str,
    *,
    profiles: dict[str, dict[str, np.ndarray]],
    counts: dict[str, dict[str, np.ndarray]],
    height_m: np.ndarray,
    n_files: dict[str, int],
    sample_files: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> Path:
    height_km = np.asarray(height_m, dtype=np.float64) / 1000.0
    hours = np.arange(24, dtype=np.float64)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = center_edges(height_km)

    c1 = profiles["control"][metric]
    g1 = profiles["graupel"][metric]
    g2 = profiles["2mom"][metric]
    d_g1 = g1 - c1
    d_g2 = g2 - g1

    finite_abs = np.concatenate([c1.ravel(), g1.ravel(), g2.ravel()])
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    vmax = float(np.nanpercentile(finite_abs, 98)) if finite_abs.size else 1.0
    if vmax <= 0.0:
        vmax = float(np.nanmax(finite_abs)) if finite_abs.size else 1.0
    vmax = vmax if vmax > 0.0 else 1.0
    diff_lim = symmetric_limit([d_g1, d_g2])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    meshes = []
    meshes.append(
        axes[0].pcolormesh(hour_edges, y_edges, c1, cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
    )
    meshes.append(
        axes[1].pcolormesh(hour_edges, y_edges, d_g1, cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim, shading="auto")
    )
    meshes.append(
        axes[2].pcolormesh(hour_edges, y_edges, d_g2, cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim, shading="auto")
    )
    titles = (
        f"{EXPERIMENT_LABELS['control']} absolute",
        f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
        f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0.0, min(20.0, float(np.nanmax(height_km))))
        ax.set_xlabel("Local hour (UTC-4)")
        ax.set_xticks(np.arange(0, 24, 3))
        ax.grid(color="white", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Height (km)")

    label = METRIC_LABELS[metric]
    unit = METRIC_UNITS[metric]
    fig.suptitle(f"Raw-FA DDH-domain {label} Diurnal Profile", fontsize=14, fontweight="bold")
    cbar0 = fig.colorbar(meshes[0], ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar0.set_label(f"{label} [{unit}]")
    cbar1 = fig.colorbar(meshes[1], ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar1.set_label(f"{label} anomaly [{unit}]")
    fig.text(
        0.01,
        0.01,
        "Raw FA inputs: UD_OMEGA and UD_MESH_FRAC only; DDH namelist box; no radar mask.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.08)

    fig_path = output_dir / f"{metric}_panel_c1m_g1m-c1m_g2m-g1m.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    txt_path = output_dir / "data_txt" / f"{metric}_panel_c1m_g1m-c1m_g2m-g1m.txt"
    write_metric_text(
        txt_path,
        figure_path=fig_path,
        metric=metric,
        profiles=profiles,
        counts=counts,
        height_m=height_m,
        n_files=n_files,
        sample_files=sample_files,
    )
    print(f"[saved] {fig_path}", flush=True)
    print(f"[saved] {txt_path}", flush=True)
    return fig_path


def write_metric_text(
    path: Path,
    *,
    figure_path: Path,
    metric: str,
    profiles: dict[str, dict[str, np.ndarray]],
    counts: dict[str, dict[str, np.ndarray]],
    height_m: np.ndarray,
    n_files: dict[str, int],
    sample_files: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height_km = np.asarray(height_m, dtype=np.float64) / 1000.0
    title = f"Raw-FA DDH-domain {METRIC_LABELS[metric]} diurnal profile plot data"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Variable: {metric.upper()}\n")
        fh.write(f"Units: {TEXT_UNITS[metric]}\n")
        fh.write("Source: raw FA files, not converted/masked NetCDF.\n")
        fh.write(
            f"DDH box: lon=[{DDH_LON_MIN}, {DDH_LON_MAX}], "
            f"lat=[{DDH_LAT_MIN}, {DDH_LAT_MAX}]\n"
        )
        fh.write("Radar mask: not applied.\n")
        fh.write("Local hour: Amazon UTC-4. +0024 state files ignored.\n")
        fh.write("Inputs: UD_OMEGA and UD_MESH_FRAC only. VITESSE_VERT is not used.\n")
        if metric == "updraft_flux":
            fh.write("UPDRAFT_FLUX = (-UD_OMEGA * UD_MESH_FRAC) / g where UD_MESH_FRAC > 0, inactive cells as zero.\n")
        elif metric == "updraft_extent":
            fh.write("UPDRAFT_EXTENT = DDH-domain mean of UD_MESH_FRAC.\n")
        elif metric == "updraft_intensity":
            fh.write("UPDRAFT_INTENSITY = abs(UD_OMEGA) where UD_MESH_FRAC > 0, inactive cells as zero.\n")
        fh.write("\nSources\n-------\n")
        for exp in EXPERIMENTS:
            fh.write(
                f"{EXPERIMENT_LABELS[exp]} sample_file: {sample_files[exp]} "
                f"(n_files={n_files[exp]})\n"
            )
        fh.write("\nData\n----\n")
        fh.write("experiment,label,level_index,height_km,local_hour,value,count\n")
        for exp in EXPERIMENTS:
            arr = profiles[exp][metric]
            cnt = counts[exp][metric]
            for level in range(arr.shape[0]):
                for hour in range(arr.shape[1]):
                    fh.write(
                        f"{exp},{EXPERIMENT_LABELS[exp]},{level},"
                        f"{height_km[level]:.12g},{hour},"
                        f"{arr[level, hour]:.12g},{int(cnt[level, hour])}\n"
                    )


def layer_summary(
    profiles: dict[str, dict[str, np.ndarray]],
    height_m: np.ndarray,
    *,
    layer_top_km: float = 3.0,
) -> list[dict[str, object]]:
    height_km = np.asarray(height_m, dtype=np.float64) / 1000.0
    level_mask = np.isfinite(height_km) & (height_km >= 0.0) & (height_km <= layer_top_km)
    windows = {
        "all_hours": np.arange(24),
        "local_12_17": np.arange(12, 18),
    }
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for exp in EXPERIMENTS:
            arr = profiles[exp][metric]
            for window_name, hours in windows.items():
                values = arr[np.ix_(level_mask, hours)]
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric].replace("Updraft ", "").title(),
                        "unit": TEXT_UNITS[metric],
                        "experiment": exp,
                        "label": EXPERIMENT_LABELS[exp],
                        "layer_top_km": layer_top_km,
                        "hour_window": window_name,
                        "mean_value": float(np.nanmean(finite)) if finite.size else np.nan,
                        "max_value": float(np.nanmax(finite)) if finite.size else np.nan,
                        "n_points": int(finite.size),
                    }
                )
    return rows


def write_updraft_summary(
    path: Path,
    *,
    figure_paths: Sequence[Path],
    rows: list[dict[str, object]],
    rainfall_mean: dict[str, float],
    rainfall_hours: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Raw-FA DDH-domain updraft diagnostics\n")
        fh.write("====================================\n")
        for fig in figure_paths:
            fh.write(f"Figure: {fig}\n")
        fh.write("Source: raw FA files, DDH namelist lon/lat box, no radar mask.\n")
        fh.write("No VITESSE_VERT is used. Updraft flux uses only UD_OMEGA and UD_MESH_FRAC: (-UD_OMEGA * UD_MESH_FRAC) / g.\n")
        fh.write(
            f"DDH box: lon=[{DDH_LON_MIN}, {DDH_LON_MAX}], "
            f"lat=[{DDH_LAT_MIN}, {DDH_LAT_MAX}]\n\n"
        )
        fh.write("0-3 km updraft diagnostics\n")
        fh.write("--------------------------\n")
        fh.write("metric,metric_label,unit,experiment,label,layer_top_km,hour_window,mean_value,max_value,n_points\n")
        for row in rows:
            fh.write(
                f"{row['metric']},{row['metric_label']},{row['unit']},"
                f"{row['experiment']},{row['label']},{row['layer_top_km']},"
                f"{row['hour_window']},{row['mean_value']:.12g},"
                f"{row['max_value']:.12g},{row['n_points']}\n"
            )
        fh.write("\nDDH-domain rainfall from raw FA surface precipitation\n")
        fh.write("----------------------------------------------------\n")
        fh.write("experiment,label,domain_mean_mm_h,n_hourly_accumulations\n")
        for exp in EXPERIMENTS:
            fh.write(
                f"{exp},{EXPERIMENT_LABELS[exp]},"
                f"{rainfall_mean[exp]:.12g},{rainfall_hours[exp]}\n"
            )


def metric_value_from_rows(rows: list[dict[str, object]], *, metric: str, label: str, hour_window: str = "all_hours") -> float:
    for row in rows:
        if row["metric"] == metric and row["label"] == label and row["hour_window"] == hour_window:
            return float(row["mean_value"])
    raise KeyError(f"Missing metric={metric} label={label} hour_window={hour_window}")


def layer_lookup(rows, experiment: str, layer: str):
    for row in rows:
        if row.experiment == experiment and row.layer == layer:
            return row
    raise KeyError(f"Missing DDH layer {experiment} {layer}")


def build_fingerprint_rows(
    *,
    updraft_rows: list[dict[str, object]],
    rainfall_mean: dict[str, float],
    agg_dir: Path,
    lead: str,
) -> list[FingerprintRow]:
    ddh_rows = compute_layer_metrics(agg_dir=agg_dir, lead=lead)
    c_03 = layer_lookup(ddh_rows, "control", "0-3 km")
    g_03 = layer_lookup(ddh_rows, "graupel", "0-3 km")
    c_fl = layer_lookup(ddh_rows, "control", "0-freezing level")
    g_fl = layer_lookup(ddh_rows, "graupel", "0-freezing level")

    return [
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft area fraction",
            "fraction",
            metric_value_from_rows(updraft_rows, metric="updraft_extent", label="C1M"),
            metric_value_from_rows(updraft_rows, metric="updraft_extent", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft mass flux",
            "kg m-2 s-1",
            metric_value_from_rows(updraft_rows, metric="updraft_flux", label="C1M"),
            metric_value_from_rows(updraft_rows, metric="updraft_flux", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft intensity",
            "Pa s-1",
            metric_value_from_rows(updraft_rows, metric="updraft_intensity", label="C1M"),
            metric_value_from_rows(updraft_rows, metric="updraft_intensity", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Convection-scheme condensation",
            "g kg-1 day-1 km",
            c_03.qv_condcv_sink,
            g_03.qv_condcv_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Resolved-microphysics condensation",
            "g kg-1 day-1 km",
            c_03.qv_condrs_sink,
            g_03.qv_condrs_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Total condensation",
            "g kg-1 day-1 km",
            c_03.qv_total_cond_sink,
            g_03.qv_total_cond_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Warm-water reservoir",
            "Cloud liquid plus rain amount",
            "g kg-1 km",
            c_03.warm_liquid_rain_amount,
            g_03.warm_liquid_rain_amount,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Convection-scheme condensation",
            "g kg-1 day-1 km",
            c_fl.qv_condcv_sink,
            g_fl.qv_condcv_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Resolved-microphysics condensation",
            "g kg-1 day-1 km",
            c_fl.qv_condrs_sink,
            g_fl.qv_condrs_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Total condensation",
            "g kg-1 day-1 km",
            c_fl.qv_total_cond_sink,
            g_fl.qv_total_cond_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Warm-water reservoir",
            "Cloud liquid plus rain amount",
            "g kg-1 km",
            c_fl.warm_liquid_rain_amount,
            g_fl.warm_liquid_rain_amount,
        ),
        FingerprintRow(
            "Rainfall",
            "Surface rainfall",
            "Domain-mean rainfall",
            "mm h-1",
            rainfall_mean["control"],
            rainfall_mean["graupel"],
        ),
    ]


def write_fingerprint_text(
    path: Path,
    *,
    figure_path: Path,
    rows: list[FingerprintRow],
    updraft_summary_path: Path,
    lead: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Raw-FA DDH-domain G1M-C1M Process Fingerprint Plot Data\n")
        fh.write("======================================================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"DDH aggregated directory: {AGG_DIR}\n")
        fh.write(f"DDH lead: {lead}\n")
        fh.write(f"Raw-FA DDH updraft/rainfall source text: {updraft_summary_path}\n")
        fh.write("Percent change formula: (G1M / C1M - 1) * 100.\n")
        fh.write("Updraft diagnostics use UD_OMEGA and UD_MESH_FRAC only; VITESSE_VERT is not used.\n")
        fh.write("Spatial domain is the NAMDDH box; radar mask is not applied.\n\n")
        fh.write("Plotted rows\n")
        fh.write("------------\n")
        fh.write("panel,group,quantity,unit,c1m_value,g1m_value,g1m_minus_c1m_percent\n")
        for row in rows:
            fh.write(
                f"{row.panel},{row.group},{row.label},{row.unit},"
                f"{row.c1m_value:.12g},{row.g1m_value:.12g},"
                f"{row.percent_change:.12g}\n"
            )


def write_layer_diurnal_plot(
    output_dir: Path,
    profiles: dict[str, dict[str, np.ndarray]],
    height_m: np.ndarray,
    *,
    dpi: int,
) -> Path:
    height_km = np.asarray(height_m, dtype=np.float64) / 1000.0
    level_mask = np.isfinite(height_km) & (height_km >= 0.0) & (height_km <= 3.0)
    hours = np.arange(24)
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.8), sharex=True)
    colors = {"control": "#d62728", "graupel": "#1f77b4", "2mom": "#2ca02c"}
    for ax, metric in zip(axes, METRICS):
        for exp in EXPERIMENTS:
            values = np.nanmean(profiles[exp][metric][level_mask, :], axis=0)
            ax.plot(hours, values, color=colors[exp], lw=2.0, label=EXPERIMENT_LABELS[exp])
        ax.set_ylabel(f"{METRIC_LABELS[metric]}\n({TEXT_UNITS[metric]})")
        ax.grid(True, color="0.88", linewidth=0.8)
    axes[0].legend(frameon=False, ncols=3, loc="upper right")
    axes[-1].set_xlabel("Local hour (UTC-4)")
    axes[-1].set_xticks(np.arange(0, 24, 3))
    fig.suptitle("Raw-FA DDH-domain 0-3 km updraft diurnal cycles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_path = output_dir / "updraft_0_3km_diurnal_cycles.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {fig_path}", flush=True)
    return fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--agg-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--lead", default=DEFAULT_LEAD)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-rainfall", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_txt").mkdir(parents=True, exist_ok=True)

    profiles, counts, n_files, sample_files, height_m, window = compute_updraft_profiles(
        data_root=args.data_root.resolve(),
        output_dir=output_dir,
        max_days=args.max_days,
        workers=args.workers,
        force=args.force,
    )

    if args.skip_rainfall:
        rainfall_mean = {"control": np.nan, "graupel": np.nan, "2mom": np.nan}
        rainfall_hourly = {exp: np.full(24, np.nan) for exp in EXPERIMENTS}
        rainfall_counts = {exp: np.zeros(24, dtype=np.int64) for exp in EXPERIMENTS}
        rainfall_hours = {exp: 0 for exp in EXPERIMENTS}
    else:
        rainfall_mean, rainfall_hourly, rainfall_counts, rainfall_hours = compute_rainfall(
            data_root=args.data_root.resolve(),
            output_dir=output_dir,
            window=window,
            max_days=args.max_days,
            workers=args.workers,
            force=args.force,
        )
        _ = rainfall_hourly, rainfall_counts

    figure_paths = []
    for metric in METRICS:
        figure_paths.append(
            plot_metric_panel(
                metric,
                profiles=profiles,
                counts=counts,
                height_m=height_m,
                n_files=n_files,
                sample_files=sample_files,
                output_dir=output_dir,
                dpi=args.dpi,
            )
        )
    figure_paths.append(write_layer_diurnal_plot(output_dir, profiles, height_m, dpi=args.dpi))

    updraft_rows = layer_summary(profiles, height_m, layer_top_km=3.0)
    updraft_summary_path = output_dir / "data_txt" / "raw_fa_ddh_updraft_summary.txt"
    write_updraft_summary(
        updraft_summary_path,
        figure_paths=figure_paths,
        rows=updraft_rows,
        rainfall_mean=rainfall_mean,
        rainfall_hours=rainfall_hours,
    )
    print(f"[saved] {updraft_summary_path}", flush=True)

    fingerprint_rows = build_fingerprint_rows(
        updraft_rows=updraft_rows,
        rainfall_mean=rainfall_mean,
        agg_dir=args.agg_dir.resolve(),
        lead=args.lead,
    )
    fingerprint_path = output_dir / "g1m_c1m_process_fingerprint.png"
    make_fingerprint_plot(fingerprint_rows, fingerprint_path)
    fingerprint_text = output_dir / "data_txt" / "g1m_c1m_process_fingerprint.txt"
    write_fingerprint_text(
        fingerprint_text,
        figure_path=fingerprint_path,
        rows=fingerprint_rows,
        updraft_summary_path=updraft_summary_path,
        lead=args.lead,
    )
    print(f"[saved] {fingerprint_path}", flush=True)
    print(f"[saved] {fingerprint_text}", flush=True)

    manifest = output_dir / "data_txt" / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "path"])
        for path in figure_paths:
            writer.writerow(["figure", path])
        writer.writerow(["figure", fingerprint_path])
        writer.writerow(["text", updraft_summary_path])
        writer.writerow(["text", fingerprint_text])
    print(f"[saved] {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
