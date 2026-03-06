#!/usr/bin/env python3
"""
Temperature-only diurnal-height panel plots for ALARO masked-netcdf runs.

This script mirrors the main hydrometeor workflow but focuses on TEMPERATURE:
1) Builds full 2-year and seasonal diurnal profiles.
2) Caches intermediate .npz files (resume-friendly by default).
3) Uses GEOPOTENTIEL (meters) for vertical axis.
4) Plots 3 panels:
   (a) C1M absolute temperature
   (b) G1M - C1M anomaly
   (c) G2M - G1M anomaly
5) Saves figures at 450 dpi.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cmaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.constants import (
    EXPERIMENTS,
    EXPERIMENT_LABELS,
    FREEZING_K,
    SEASONS,
)
from alaro_analysis.common.models import PeriodSpec, VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.timeparse import (
    has_pf_subdirs,
    parse_month_from_day_name,
    parse_utc_hour_from_name,
)
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.data.discovery import discover_variables
from alaro_analysis.plotting.style import resolve_workers

DEFAULT_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf-2"
)
DEFAULT_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf-2"
)
DEFAULT_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf-2"
)

DEFAULT_CONTROL_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/masked-netcdf/GEOPOTENTIEL"
)
DEFAULT_GRAUPEL_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/graupel/masked-netcdf/GEOPOTENTIEL"
)
DEFAULT_2MOM_GEO_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/2mom/masked-netcdf/GEOPOTENTIEL"
)

DEFAULT_OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures")
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full+seasonal temperature diurnal panels from masked-netcdf-2."
    )
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)

    parser.add_argument("--control-geopotential-dir", type=Path, default=DEFAULT_CONTROL_GEO_DIR)
    parser.add_argument("--graupel-geopotential-dir", type=Path, default=DEFAULT_GRAUPEL_GEO_DIR)
    parser.add_argument("--twomom-geopotential-dir", type=Path, default=DEFAULT_2MOM_GEO_DIR)

    parser.add_argument(
        "--temperature-variable",
        default="TEMPERATURE",
        help="Temperature variable name (default: TEMPERATURE).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Subset of seasons (wet dry ...). Use 'all' for all seasons.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help="Directory for cached/intermediate .npz outputs.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate .npz files.",
    )
    parser.add_argument(
        "--recompute",
        dest="overwrite_intermediate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=-4,
        help="Local timezone offset from UTC (Amazon is -4).",
    )
    parser.add_argument(
        "--height-variable",
        default="GEOPOTENTIEL",
        help="Variable name used inside geopotential files.",
    )
    parser.add_argument(
        "--height-aggregate",
        choices=("first", "mean-all"),
        default="first",
        help="How to aggregate geopotential profiles by period.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Multiprocessing worker count (capped to 16).",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit for height-based plots.",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=98.0,
        help="Percentile for fixed symmetric anomaly scale (default: 98).",
    )
    parser.add_argument(
        "--absolute-low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for fixed absolute colorbar range (default: 2).",
    )
    parser.add_argument(
        "--absolute-high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for fixed absolute colorbar range (default: 98).",
    )
    return parser.parse_args()


def resolve_data_var_name(ds: xr.Dataset, requested: str) -> str:
    if requested in ds.data_vars:
        return requested
    req_lower = requested.lower()
    for name in ds.data_vars:
        if name.lower() == req_lower:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.keys()))
    raise KeyError(f"Variable '{requested}' not found. Available: {list(ds.data_vars.keys())}")


def read_vertical_profile(file_path: Path, requested_variable: str) -> tuple[np.ndarray, np.ndarray | None]:
    with xr.open_dataset(file_path, decode_times=False) as ds:
        var_name = resolve_data_var_name(ds, requested_variable)
        arr = np.asarray(ds[var_name].values, dtype=np.float64)
        dims = tuple(ds[var_name].dims)
        if arr.ndim == 4:
            profile = np.nanmean(arr, axis=(0, 2, 3))
            vertical_dim = dims[1]
        elif arr.ndim == 3:
            profile = np.nanmean(arr, axis=(1, 2))
            vertical_dim = dims[0]
        elif arr.ndim == 2:
            profile = np.nanmean(arr, axis=(1,))
            vertical_dim = dims[0]
        elif arr.ndim == 1:
            profile = arr
            vertical_dim = dims[0]
        else:
            raise ValueError(f"Unsupported shape {arr.shape} in {file_path}")

        vertical_coord = None
        if vertical_dim in ds.coords:
            coord = np.asarray(ds[vertical_dim].values, dtype=np.float64)
            if coord.ndim == 1 and coord.size == profile.size:
                vertical_coord = coord
    return np.asarray(profile, dtype=np.float64), vertical_coord


def collect_file_records(
    variable_dir: Path,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> list[tuple[int, Path]]:
    if not variable_dir.exists():
        raise FileNotFoundError(f"Missing directory: {variable_dir}")
    allowed_set = set(allowed_months) if allowed_months is not None else None
    day_dirs = sorted(p for p in variable_dir.iterdir() if p.is_dir() and p.name.startswith("pf"))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    records: list[tuple[int, Path]] = []
    for day_dir in day_dirs:
        month = parse_month_from_day_name(day_dir.name)
        if allowed_set is not None:
            if month is None or month not in allowed_set:
                continue
        for file_path in sorted(day_dir.glob("*.nc")):
            utc_hour = parse_utc_hour_from_name(file_path.name)
            if utc_hour is None:
                continue
            local_hour = (utc_hour + utc_offset_hours) % 24
            records.append((local_hour, file_path))
    return records


def compute_diurnal_profile(
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    variable_dir = experiment_dir / variable
    records = collect_file_records(
        variable_dir=variable_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid +0000..+0023 files found in {variable_dir}")

    first_profile, _ = read_vertical_profile(records[0][1], variable)
    n_levels = first_profile.size
    sums = np.zeros((n_levels, 24), dtype=np.float64)
    counts = np.zeros((n_levels, 24), dtype=np.int64)

    for idx, (local_hour, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(file_path, variable)
        if profile.size != n_levels:
            raise ValueError(
                f"Inconsistent vertical levels in {file_path}: {profile.size} vs {n_levels}"
            )
        valid = np.isfinite(profile)
        sums[valid, local_hour] += profile[valid]
        counts[valid, local_hour] += 1
        if idx % 2000 == 0 or idx == len(records):
            print(f"[{experiment_dir.name}/{variable}] {idx}/{len(records)} files", flush=True)

    mean = np.full_like(sums, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    return mean, counts, len(records), records[0][1]


def compute_geopotential_height_profile(
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
) -> tuple[np.ndarray, int]:
    records = collect_file_records(
        variable_dir=geopotential_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid geopotential files found in {geopotential_dir}")

    if aggregate == "first":
        profile, _ = read_vertical_profile(records[0][1], height_variable)
        return profile, 1

    first, _ = read_vertical_profile(records[0][1], height_variable)
    sums = np.zeros_like(first, dtype=np.float64)
    counts = np.zeros_like(first, dtype=np.int64)
    for idx, (_, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(file_path, height_variable)
        valid = np.isfinite(profile)
        sums[valid] += profile[valid]
        counts[valid] += 1
        if idx % 4000 == 0 or idx == len(records):
            print(f"[{geopotential_dir.parent.name}/GEOPOTENTIEL] {idx}/{len(records)} files", flush=True)

    mean = np.full_like(sums, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    return mean, len(records)


def build_diurnal_cache_file(intermediate_dir: Path, variable: str, period_subdir: Path, experiment: str) -> Path:
    return intermediate_dir / safe_name(variable) / period_subdir / f"{experiment}_diurnal_profile.npz"


def build_height_cache_file(intermediate_dir: Path, period_subdir: Path, experiment: str, aggregate: str) -> Path:
    return intermediate_dir / "geopotential" / period_subdir / f"{experiment}_height_profile_{aggregate}.npz"


def load_or_compute_diurnal(
    cache_file: Path,
    experiment_dir: Path,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, Path]:
    if cache_file.exists() and (not overwrite_intermediate):
        payload = np.load(cache_file)
        sample_file = Path(str(payload["sample_file"].item()))
        return np.asarray(payload["mean"], dtype=np.float64), sample_file

    mean, counts, n_files, sample_file = compute_diurnal_profile(
        experiment_dir=experiment_dir,
        variable=variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if max_days is None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            mean=mean,
            counts=counts,
            n_files=np.array([n_files], dtype=np.int64),
            sample_file=np.array([str(sample_file)]),
        )
    return mean, sample_file


def load_or_compute_height(
    cache_file: Path,
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
) -> np.ndarray:
    if cache_file.exists() and (not overwrite_intermediate):
        payload = np.load(cache_file)
        return np.asarray(payload["height_m"], dtype=np.float64)

    height_m, n_files = compute_geopotential_height_profile(
        geopotential_dir=geopotential_dir,
        height_variable=height_variable,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
        aggregate=aggregate,
    )
    if max_days is None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            height_m=height_m,
            n_files=np.array([n_files], dtype=np.int64),
        )
    return height_m


def infer_freezing_threshold(temperature_profile: np.ndarray) -> float | None:
    valid = temperature_profile[np.isfinite(temperature_profile)]
    if valid.size == 0:
        return None
    return FREEZING_K if float(np.median(valid)) > 150.0 else 0.0


def compute_freezing_line_km(axis: VerticalAxis, temperature_profiles: list[np.ndarray]) -> np.ndarray | None:
    if not axis.is_height_km or not temperature_profiles:
        return None
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in temperature_profiles))
    if n_levels < 2:
        return None

    stacked = np.stack([arr[:n_levels, :] for arr in temperature_profiles], axis=0)
    mean_temp = np.nanmean(stacked, axis=0)
    threshold = infer_freezing_threshold(mean_temp)
    if threshold is None:
        return None

    y_km = np.asarray(axis.values[:n_levels], dtype=np.float64)
    order = np.argsort(y_km)
    y_sorted = y_km[order]
    t_sorted = mean_temp[order, :]

    freeze_line = np.full((24,), np.nan, dtype=np.float64)
    for hour in range(24):
        column = t_sorted[:, hour]
        finite = np.isfinite(column) & np.isfinite(y_sorted)
        if np.sum(finite) < 2:
            continue
        yy = y_sorted[finite]
        tt = column[finite]
        for i in range(yy.size - 1):
            t1 = tt[i]
            t2 = tt[i + 1]
            y1 = yy[i]
            y2 = yy[i + 1]
            if np.isclose(t1, threshold):
                freeze_line[hour] = y1
                break
            if np.isclose(t2, threshold):
                freeze_line[hour] = y2
                break
            d1 = t1 - threshold
            d2 = t2 - threshold
            if d1 * d2 < 0 and not np.isclose(t1, t2):
                frac = (threshold - t1) / (t2 - t1)
                freeze_line[hour] = y1 + frac * (y2 - y1)
                break
    return freeze_line


def align_vertical_shapes(axis: VerticalAxis, profiles: dict[str, np.ndarray]) -> tuple[VerticalAxis, dict[str, np.ndarray]]:
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in profiles.values()))
    axis_new = VerticalAxis(values=axis.values[:n_levels], label=axis.label, is_height_km=axis.is_height_km)
    prof_new = {exp: arr[:n_levels, :] for exp, arr in profiles.items()}
    return axis_new, prof_new


def compute_temperature_scales(
    periods: list[PeriodSpec],
    results: dict[tuple[str, str], np.ndarray],
    low_pct: float,
    high_pct: float,
    anom_pct: float,
) -> tuple[tuple[float, float], float]:
    abs_chunks: list[np.ndarray] = []
    anom_chunks: list[np.ndarray] = []
    for period in periods:
        c = results[(period.key, "control")]
        g = results[(period.key, "graupel")]
        t = results[(period.key, "2mom")]
        abs_valid = c[np.isfinite(c)]
        if abs_valid.size > 0:
            abs_chunks.append(abs_valid)
        for arr in (g - c, t - g):
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                anom_chunks.append(valid)

    if abs_chunks:
        merged = np.concatenate(abs_chunks)
        vmin = float(np.percentile(merged, low_pct))
        vmax = float(np.percentile(merged, high_pct))
        if vmax <= vmin:
            vmax = vmin + 1.0
        abs_limits = (vmin, vmax)
    else:
        abs_limits = (250.0, 320.0)

    if anom_chunks:
        merged = np.concatenate(anom_chunks)
        finite = np.abs(merged[np.isfinite(merged)])
        if finite.size > 0:
            scale = float(np.percentile(finite, anom_pct))
            if scale <= 0:
                scale = float(np.max(finite))
            if scale <= 0:
                scale = 1.0
        else:
            scale = 1.0
    else:
        scale = 1.0
    return abs_limits, scale


def plot_temperature_panels(
    control_mean: np.ndarray,
    graupel_mean: np.ndarray,
    twomom_mean: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    max_height_km: float,
    period_label: str,
    variable: str,
    freezing_line_km: np.ndarray | None,
    abs_limits: tuple[float, float],
    anom_scale: float,
) -> None:
    axis_label_fs = 16
    tick_label_fs = 14
    legend_fs = 12
    panel_tag_fs = 14
    cbar_label_fs = 14
    cbar_tick_fs = 12

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    ctrl = control_mean[order, :]
    diff_g1 = (graupel_mean - control_mean)[order, :]
    diff_g2 = (twomom_mean - graupel_mean)[order, :]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        ctrl = ctrl[keep, :]
        diff_g1 = diff_g1[keep, :]
        diff_g2 = diff_g2[keep, :]

    ctrl_plot = np.ma.masked_invalid(ctrl)
    d1_plot = np.ma.masked_invalid(diff_g1)
    d2_plot = np.ma.masked_invalid(diff_g2)
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    vmin_abs, vmax_abs = abs_limits
    if vmax_abs <= vmin_abs:
        vmax_abs = vmin_abs + 1.0
    abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    if anom_scale <= 0:
        anom_scale = 1.0
    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=axis_label_fs)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(axis="both", labelsize=tick_label_fs)
        ax.set_ylabel(axis.label, fontsize=axis_label_fs)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        ax.text(
            0.02, 0.98, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=panel_tag_fs, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    pcm_abs = axes[0].pcolormesh(
        hour_edges, y_edges, ctrl_plot, cmap=cmaps.WhiteBlueGreenYellowRed, norm=abs_norm, shading="auto"
    )
    axes[0].set_title(f"{EXPERIMENT_LABELS['control']} ({variable}, Absolute)", fontsize=14, fontweight="bold")
    axes[1].pcolormesh(hour_edges, y_edges, d1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[1].set_title(f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}", fontsize=14, fontweight="bold")
    pcm_diff = axes[2].pcolormesh(hour_edges, y_edges, d2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[2].set_title(f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}", fontsize=14, fontweight="bold")

    if freezing_line_km is not None and np.isfinite(freezing_line_km).any():
        hours = np.arange(24, dtype=np.float64)
        for idx, ax in enumerate(axes):
            label = "Freezing level" if idx == 0 else None
            ax.plot(hours, freezing_line_km, color="black", linewidth=1.8, linestyle="--", label=label, zorder=10)
        axes[0].legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    fig.suptitle(f"{period_label} - {variable}", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_abs.set_label(f"Mean {variable} (Absolute)", fontsize=cbar_label_fs)
    cbar_abs.ax.tick_params(labelsize=cbar_tick_fs)

    cbar_diff = fig.colorbar(pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_diff.set_label(f"{variable} anomaly", fontsize=cbar_label_fs)
    cbar_diff.ax.tick_params(labelsize=cbar_tick_fs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def run_profile_job(
    cache_file: str,
    experiment_dir: str,
    variable: str,
    max_days: int | None,
    overwrite_intermediate: bool,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, str]:
    mean, sample_file = load_or_compute_diurnal(
        cache_file=Path(cache_file),
        experiment_dir=Path(experiment_dir),
        variable=variable,
        max_days=max_days,
        overwrite_intermediate=overwrite_intermediate,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    return mean, str(sample_file)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.anomaly_percentile <= 100.0):
        raise ValueError("--anomaly-percentile must be in (0, 100].")
    if not (0.0 <= args.absolute_low_percentile < 100.0):
        raise ValueError("--absolute-low-percentile must be in [0, 100).")
    if not (0.0 < args.absolute_high_percentile <= 100.0):
        raise ValueError("--absolute-high-percentile must be in (0, 100].")
    if args.absolute_low_percentile >= args.absolute_high_percentile:
        raise ValueError("--absolute-low-percentile must be lower than --absolute-high-percentile.")

    experiment_dirs = {
        "control": args.control_dir.resolve(),
        "graupel": args.graupel_dir.resolve(),
        "2mom": args.twomom_dir.resolve(),
    }
    geopotential_dirs = {
        "control": args.control_geopotential_dir.resolve(),
        "graupel": args.graupel_geopotential_dir.resolve(),
        "2mom": args.twomom_geopotential_dir.resolve(),
    }
    for exp, d in experiment_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"{exp} data dir not found: {d}")
    for exp, d in geopotential_dirs.items():
        if not d.exists():
            print(f"[warn] {exp} geopotential dir not found: {d}")

    available = discover_variables(experiment_dirs)
    common = set.intersection(*(available[exp] for exp in EXPERIMENTS))
    if args.temperature_variable not in common:
        raise RuntimeError(
            f"Temperature variable '{args.temperature_variable}' is not common across experiments. "
            f"Common: {sorted(common)}"
        )

    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    n_workers = resolve_workers(args.n_workers)

    print("\nInput data directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}: {experiment_dirs[exp]}")
    print("\nGeopotential directories:")
    for exp in EXPERIMENTS:
        print(f"- {exp}: {geopotential_dirs[exp]}")
    print("\nOutput directory:", output_dir)
    print("Intermediate directory:", intermediate_dir)
    print("Workers:", n_workers)
    print("Periods:", [p.key for p in periods])
    print("Overwrite intermediate:", args.overwrite_intermediate)
    print("Ignoring +0024 files by design.")
    print("Variable to plot:", args.temperature_variable)

    # Build jobs
    jobs: list[tuple[str, str, str, str, str, tuple[int, ...] | None]] = []
    for period in periods:
        for exp in EXPERIMENTS:
            cache_file = build_diurnal_cache_file(
                intermediate_dir=intermediate_dir,
                variable=args.temperature_variable,
                period_subdir=period.output_subdir,
                experiment=exp,
            )
            jobs.append(
                (
                    period.key,
                    exp,
                    str(cache_file),
                    str(experiment_dirs[exp]),
                    args.temperature_variable,
                    period.allowed_months,
                )
            )

    print(f"\nComputing {len(jobs)} profile jobs with up to {n_workers} processes...")
    results: dict[tuple[str, str], tuple[np.ndarray, Path]] = {}
    errors: dict[tuple[str, str], str] = {}

    if n_workers == 1:
        for idx, (period_key, exp, cache_file, exp_dir, var_name, months) in enumerate(jobs, start=1):
            key = (period_key, exp)
            try:
                mean, sample_file = run_profile_job(
                    cache_file=cache_file,
                    experiment_dir=exp_dir,
                    variable=var_name,
                    max_days=args.max_days,
                    overwrite_intermediate=args.overwrite_intermediate,
                    allowed_months=months,
                    utc_offset_hours=args.utc_offset_hours,
                )
                results[key] = (mean, Path(sample_file))
            except Exception as exc:
                errors[key] = str(exc)
            if idx % 20 == 0 or idx == len(jobs):
                print(f"Finished {idx}/{len(jobs)} jobs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_map = {}
            for period_key, exp, cache_file, exp_dir, var_name, months in jobs:
                key = (period_key, exp)
                fut = pool.submit(
                    run_profile_job,
                    cache_file,
                    exp_dir,
                    var_name,
                    args.max_days,
                    args.overwrite_intermediate,
                    months,
                    args.utc_offset_hours,
                )
                future_map[fut] = key
            done = 0
            for fut in as_completed(future_map):
                key = future_map[fut]
                done += 1
                try:
                    mean, sample_file = fut.result()
                    results[key] = (mean, Path(sample_file))
                except Exception as exc:
                    errors[key] = str(exc)
                if done % 20 == 0 or done == len(jobs):
                    print(f"Finished {done}/{len(jobs)} jobs", flush=True)

    # Prepare for global scaling
    scale_data: dict[tuple[str, str], np.ndarray] = {}
    for period in periods:
        for exp in EXPERIMENTS:
            key = (period.key, exp)
            if key in results:
                scale_data[key] = results[key][0]
    abs_limits, anom_scale = compute_temperature_scales(
        periods=periods,
        results=scale_data,
        low_pct=args.absolute_low_percentile,
        high_pct=args.absolute_high_percentile,
        anom_pct=args.anomaly_percentile,
    )
    print(
        f"\nFixed temperature scales: abs=[{abs_limits[0]:.3f}, {abs_limits[1]:.3f}], "
        f"anom=±{anom_scale:.3f}"
    )

    total_saved = 0
    for period in periods:
        print(f"\n==================== {period.label} ({period.key}) ====================")
        profiles: dict[str, np.ndarray] = {}
        for exp in EXPERIMENTS:
            key = (period.key, exp)
            if key not in results:
                print(f"Skipping {period.key}: {exp} unavailable ({errors.get(key, 'missing')})")
                profiles = {}
                break
            profiles[exp] = results[key][0]
        if not profiles:
            continue

        height_cache = build_height_cache_file(
            intermediate_dir=intermediate_dir,
            period_subdir=period.output_subdir,
            experiment="control",
            aggregate=args.height_aggregate,
        )
        height_m = load_or_compute_height(
            cache_file=height_cache,
            geopotential_dir=geopotential_dirs["control"],
            height_variable=args.height_variable,
            max_days=args.max_days,
            overwrite_intermediate=args.overwrite_intermediate,
            allowed_months=period.allowed_months,
            utc_offset_hours=args.utc_offset_hours,
            aggregate=args.height_aggregate,
        )
        axis = VerticalAxis(values=np.asarray(height_m, dtype=np.float64) / 1000.0, label="Height (km)", is_height_km=True)
        axis, profiles = align_vertical_shapes(axis=axis, profiles=profiles)

        freezing_line_km = compute_freezing_line_km(
            axis=axis,
            temperature_profiles=[profiles[exp] for exp in EXPERIMENTS],
        )

        output_file = (
            output_dir
            / safe_name(args.temperature_variable)
            / period.output_subdir
            / f"{safe_name(args.temperature_variable)}_panel_c1m_g1m-c1m_g2m-g1m.png"
        )
        plot_temperature_panels(
            control_mean=profiles["control"],
            graupel_mean=profiles["graupel"],
            twomom_mean=profiles["2mom"],
            axis=axis,
            output_file=output_file,
            max_height_km=args.max_height_km,
            period_label=period.label,
            variable=args.temperature_variable,
            freezing_line_km=freezing_line_km,
            abs_limits=abs_limits,
            anom_scale=anom_scale,
        )
        total_saved += 1

    print(f"\nCompleted. Total figures generated: {total_saved}")
    if errors:
        print(f"Completed with {len(errors)} profile-job errors.")


if __name__ == "__main__":
    main()
