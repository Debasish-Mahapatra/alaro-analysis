"""Run tobac rainfall tracking for IMERG observations and ALARO model output."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import tobac
import xarray as xr


DEFAULT_MODEL_ROOT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/NO3M/deaccumulated-rainfall"
)
DEFAULT_IMERG_ROOT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/IMERG_AMAZON/IMERG-cropped-to-model-rainfall-boundaries"
)
DEFAULT_PROCESSED_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/Tracking/processed-data")
DEFAULT_PLOTS_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/Tracking/plots")
DEFAULT_JSON_LOG_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/Tracking/logs/json")
DEFAULT_TEXT_LOG_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/Tracking/logs")

MODEL_VARS = ("SURFPREC.EAU.GEC", "SURFPREC.EAU.CON")
IMERG_RE = re.compile(
    r"3B-HHR\.MS\.MRG\.3IMERG\.(?P<ymd>\d{8})-S(?P<hms>\d{6})-"
)


@dataclass(frozen=True)
class GridInfo:
    nx: int
    ny: int
    dxy_m: float
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class DatasetSummary:
    label: str
    hours: int
    min_mm_h: float
    mean_mm_h: float
    max_mm_h: float
    features: int
    tracks: int
    tracked_features: int
    output_features_csv: str
    output_tracks_csv: str
    output_plot: str


def parse_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Could not parse {value!r}; use YYYY-MM-DD or YYYY-MM-DDTHH:MM"
    )


def iter_hours(start: datetime, end: datetime) -> list[datetime]:
    if end <= start:
        raise ValueError("end time must be after start time")
    times: list[datetime] = []
    current = start
    while current < end:
        times.append(current)
        current += timedelta(hours=1)
    return times


def imerg_filename(time: datetime) -> str:
    minute_of_day = time.hour * 60 + time.minute
    end_time = time + timedelta(minutes=29, seconds=59)
    return (
        f"3B-HHR.MS.MRG.3IMERG.{time:%Y%m%d}-S{time:%H%M%S}-"
        f"E{end_time:%H%M%S}.{minute_of_day:04d}.V07B.HDF5.nc4"
    )


def imerg_paths_for_hour(root: Path, time: datetime) -> tuple[Path, Path]:
    return root / imerg_filename(time), root / imerg_filename(time + timedelta(minutes=30))


def model_path(root: Path, var_name: str, time: datetime) -> Path:
    return (
        root
        / var_name
        / f"pf{time:%Y%m%d}"
        / f"pfABOFABOF+{time.hour:04d}.nc"
    )


def missing_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if not path.is_file()]


def check_complete_period(
    times: list[datetime],
    model_root: Path,
    imerg_root: Path,
) -> dict[str, list[str]]:
    model_needed: list[Path] = []
    imerg_needed: list[Path] = []
    for time in times:
        for var_name in MODEL_VARS:
            model_needed.append(model_path(model_root, var_name, time))
        imerg_needed.extend(imerg_paths_for_hour(imerg_root, time))
    return {
        "model_missing": missing_paths(model_needed),
        "imerg_missing": missing_paths(imerg_needed),
    }


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def model_grid_info(sample_path: Path) -> tuple[GridInfo, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(sample_path) as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
    ny, nx = lon.shape
    y = ny // 2
    x = nx // 2
    dxy = np.mean(
        [
            haversine_m(float(lon[y, x]), float(lat[y, x]), float(lon[y, x + 1]), float(lat[y, x + 1])),
            haversine_m(float(lon[y, x]), float(lat[y, x]), float(lon[y + 1, x]), float(lat[y + 1, x])),
        ]
    )
    info = GridInfo(
        nx=nx,
        ny=ny,
        dxy_m=float(dxy),
        lon_min=float(np.nanmin(lon)),
        lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)),
        lat_max=float(np.nanmax(lat)),
    )
    return info, lon, lat


def imerg_grid_info(sample_path: Path) -> tuple[GridInfo, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(sample_path) as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
    mid_lat = float(lat[len(lat) // 2])
    dxy = np.mean(
        [
            haversine_m(float(lon[0]), mid_lat, float(lon[1]), mid_lat),
            haversine_m(float(lon[0]), float(lat[0]), float(lon[0]), float(lat[1])),
        ]
    )
    info = GridInfo(
        nx=int(lon.size),
        ny=int(lat.size),
        dxy_m=float(dxy),
        lon_min=float(lon.min()),
        lon_max=float(lon.max()),
        lat_min=float(lat.min()),
        lat_max=float(lat.max()),
    )
    return info, lon, lat


def build_imerg_hourly(
    times: list[datetime],
    root: Path,
    thresholds: list[float],
) -> tuple[xr.DataArray, GridInfo]:
    sample_path, _ = imerg_paths_for_hour(root, times[0])
    grid, lon, lat = imerg_grid_info(sample_path)
    data = np.empty((len(times), grid.ny, grid.nx), dtype=np.float32)

    for i, time in enumerate(times):
        first, second = imerg_paths_for_hour(root, time)
        hourly = np.zeros((grid.ny, grid.nx), dtype=np.float32)
        for path in (first, second):
            with netCDF4.Dataset(path) as ds:
                rate = np.ma.array(ds.variables["precipitation"][0, :, :]).filled(np.nan)
            hourly += (np.asarray(rate, dtype=np.float32).T) * 0.5
        data[i, :, :] = np.where(np.isfinite(hourly) & (hourly > 0), hourly, 0.0)

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "y": np.arange(grid.ny, dtype=np.float32) * grid.dxy_m,
            "x": np.arange(grid.nx, dtype=np.float32) * grid.dxy_m,
            "lat": ("y", lat.astype(np.float32)),
            "lon": ("x", lon.astype(np.float32)),
        },
        name="rainfall",
        attrs={
            "units": "mm h-1",
            "source": "IMERG half-hourly mm/hr converted to hourly accumulation",
            "thresholds_mm_h": ",".join(str(t) for t in thresholds),
        },
    )
    return da, grid


def coarsen_mean_2d(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    ny_trim = (values.shape[0] // factor) * factor
    nx_trim = (values.shape[1] // factor) * factor
    trimmed = values[:ny_trim, :nx_trim]
    return trimmed.reshape(ny_trim // factor, factor, nx_trim // factor, factor).mean(axis=(1, 3))


def build_model_hourly(
    times: list[datetime],
    root: Path,
    thresholds: list[float],
    coarsen_factor: int,
) -> tuple[xr.DataArray, GridInfo]:
    sample_path = model_path(root, MODEL_VARS[0], times[0])
    grid_native, lon_native, lat_native = model_grid_info(sample_path)
    if coarsen_factor > 1:
        lon = coarsen_mean_2d(lon_native, coarsen_factor).astype(np.float32)
        lat = coarsen_mean_2d(lat_native, coarsen_factor).astype(np.float32)
        grid = GridInfo(
            nx=int(lon.shape[1]),
            ny=int(lon.shape[0]),
            dxy_m=float(grid_native.dxy_m * coarsen_factor),
            lon_min=float(np.nanmin(lon)),
            lon_max=float(np.nanmax(lon)),
            lat_min=float(np.nanmin(lat)),
            lat_max=float(np.nanmax(lat)),
        )
    else:
        lon = lon_native
        lat = lat_native
        grid = grid_native
    data = np.empty((len(times), grid.ny, grid.nx), dtype=np.float32)

    for i, time in enumerate(times):
        hourly = np.zeros((grid_native.ny, grid_native.nx), dtype=np.float32)
        for var_name in MODEL_VARS:
            path = model_path(root, var_name, time)
            with netCDF4.Dataset(path) as ds:
                values = np.ma.array(ds.variables[var_name][0, :, :]).filled(np.nan)
            hourly += np.asarray(values, dtype=np.float32)
        hourly = np.where(np.isfinite(hourly) & (hourly > 0), hourly, 0.0)
        data[i, :, :] = coarsen_mean_2d(hourly, coarsen_factor).astype(np.float32)

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "y": np.arange(grid.ny, dtype=np.float32) * grid.dxy_m,
            "x": np.arange(grid.nx, dtype=np.float32) * grid.dxy_m,
            "lat": (("y", "x"), lat.astype(np.float32)),
            "lon": (("y", "x"), lon.astype(np.float32)),
        },
        name="rainfall",
        attrs={
            "units": "mm h-1",
            "source": "NO3M deaccumulated SURFPREC.EAU.GEC + SURFPREC.EAU.CON",
            "coarsen_factor": coarsen_factor,
            "thresholds_mm_h": ",".join(str(t) for t in thresholds),
        },
    )
    return da, grid


def plot_track_summary(
    label: str,
    features: pd.DataFrame,
    tracks: pd.DataFrame,
    out_path: Path,
    period_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"tobac rainfall tracking: {label} ({period_label})")

    if len(features):
        counts = features.groupby("time").size()
        axes[0].plot(pd.to_datetime(counts.index), counts.values, color="#1957a6", lw=1)
    axes[0].set_title("Detected features per hour")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Features")
    axes[0].tick_params(axis="x", rotation=35)

    if len(tracks) and "cell" in tracks.columns:
        tracked = tracks.loc[tracks["cell"] >= 0]
        durations = tracked.groupby("cell").size()
        axes[1].hist(durations.values, bins=40, color="#d17a00", edgecolor="white")
    axes[1].set_title("Track duration")
    axes[1].set_xlabel("Hourly steps")
    axes[1].set_ylabel("Tracks")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_tobac_for_dataset(
    label: str,
    field: xr.DataArray,
    grid: GridInfo,
    thresholds: list[float],
    min_threshold_pixels: int,
    v_max: float,
    memory: int,
    min_track_hours: int,
    out_dir: Path,
    plot_dir: Path,
    period_label: str,
) -> DatasetSummary:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {label}: feature detection", flush=True)
    features = tobac.feature_detection_multithreshold(
        field,
        dxy=grid.dxy_m,
        threshold=thresholds,
        target="maximum",
        position_threshold="weighted_diff",
        sigma_threshold=1,
        min_distance=0,
        n_erosion_threshold=0,
        n_min_threshold=min_threshold_pixels,
    )
    features_csv = out_dir / f"{label}_features.csv"
    features.to_csv(features_csv, index=False)
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {label}: "
        f"features={len(features)} saved={features_csv}",
        flush=True,
    )

    if len(features):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {label}: linking", flush=True)
        tracks = tobac.linking_trackpy(
            features,
            field,
            dt=3600,
            dxy=grid.dxy_m,
            v_max=v_max,
            method_linking="predict",
            adaptive_stop=0.2,
            adaptive_step=0.95,
            extrapolate=0,
            order=1,
            subnetwork_size=100,
            memory=memory,
            time_cell_min=min_track_hours * 3600,
        )
    else:
        tracks = pd.DataFrame()

    tracks_csv = out_dir / f"{label}_tracks.csv"
    tracks.to_csv(tracks_csv, index=False)
    tracked_features = int(tracks["cell"].ge(0).sum()) if len(tracks) and "cell" in tracks else 0
    track_count = int(tracks.loc[tracks["cell"] >= 0, "cell"].nunique()) if tracked_features else 0
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {label}: "
        f"tracks={track_count} tracked_features={tracked_features} saved={tracks_csv}",
        flush=True,
    )

    plot_path = plot_dir / f"tobac_rainfall_{label}_{period_label}.png"
    plot_track_summary(label, features, tracks, plot_path, period_label)

    values = np.asarray(field.values, dtype=np.float32)
    return DatasetSummary(
        label=label,
        hours=int(field.sizes["time"]),
        min_mm_h=float(np.nanmin(values)),
        mean_mm_h=float(np.nanmean(values)),
        max_mm_h=float(np.nanmax(values)),
        features=int(len(features)),
        tracks=track_count,
        tracked_features=tracked_features,
        output_features_csv=str(features_csv),
        output_tracks_csv=str(tracks_csv),
        output_plot=str(plot_path),
    )


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("threshold list cannot be empty")
    return thresholds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run tobac rainfall feature tracking for IMERG and NO3M model data."
    )
    parser.add_argument("--start", type=parse_datetime, default=parse_datetime("2015-05-01"))
    parser.add_argument("--end", type=parse_datetime, default=parse_datetime("2015-11-01"))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--imerg-root", type=Path, default=DEFAULT_IMERG_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--json-log-dir", type=Path, default=DEFAULT_JSON_LOG_DIR)
    parser.add_argument("--text-log-dir", type=Path, default=DEFAULT_TEXT_LOG_DIR)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("5,10,20,40"))
    parser.add_argument("--min-threshold-pixels", type=int, default=4)
    parser.add_argument("--v-max", type=float, default=30.0, help="Maximum linking speed in m s-1.")
    parser.add_argument("--memory", type=int, default=1, help="Allow this many missing frames during linking.")
    parser.add_argument("--min-track-hours", type=int, default=3)
    parser.add_argument(
        "--model-coarsen",
        type=int,
        default=3,
        help="Coarsen model y/x grid by this factor before tracking; 3 gives about 12 km spacing.",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "obs", "model"),
        default="both",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check that all inputs exist for the requested period.",
    )
    args = parser.parse_args(argv)

    times = iter_hours(args.start, args.end)
    period_label = f"{args.start:%Y%m%d}_{args.end:%Y%m%d}"
    out_dir = args.processed_dir / f"tobac-rainfall-{period_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.json_log_dir.mkdir(parents=True, exist_ok=True)
    args.text_log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] period={args.start} to {args.end} "
        f"hours={len(times)} thresholds={args.thresholds}",
        flush=True,
    )
    missing = check_complete_period(times, args.model_root, args.imerg_root)
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"missing_model={len(missing['model_missing'])} missing_imerg={len(missing['imerg_missing'])}",
        flush=True,
    )
    if missing["model_missing"] or missing["imerg_missing"]:
        missing_path = args.json_log_dir / f"tobac_rainfall_missing_inputs_{period_label}.json"
        missing_path.write_text(json.dumps(missing, indent=2))
        print(f"Missing inputs written to {missing_path}", file=sys.stderr, flush=True)
        return 2
    if args.check_only:
        return 0

    summaries: list[DatasetSummary] = []
    if args.dataset in ("both", "obs"):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] obs: reading hourly IMERG", flush=True)
        obs, obs_grid = build_imerg_hourly(times, args.imerg_root, args.thresholds)
        summaries.append(
            run_tobac_for_dataset(
                "obs_imerg",
                obs,
                obs_grid,
                args.thresholds,
                args.min_threshold_pixels,
                args.v_max,
                args.memory,
                args.min_track_hours,
                out_dir,
                args.plots_dir,
                period_label,
            )
        )
        del obs

    if args.dataset in ("both", "model"):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] model: reading hourly NO3M rainfall", flush=True)
        model, model_grid = build_model_hourly(
            times,
            args.model_root,
            args.thresholds,
            args.model_coarsen,
        )
        summaries.append(
            run_tobac_for_dataset(
                "model_no3m",
                model,
                model_grid,
                args.thresholds,
                args.min_threshold_pixels,
                args.v_max,
                args.memory,
                args.min_track_hours,
                out_dir,
                args.plots_dir,
                period_label,
            )
        )
        del model

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "period_start": args.start.isoformat(),
        "period_end_exclusive": args.end.isoformat(),
        "hours": len(times),
        "thresholds_mm_h": args.thresholds,
        "min_threshold_pixels": args.min_threshold_pixels,
        "v_max_m_s": args.v_max,
        "memory_frames": args.memory,
        "min_track_hours": args.min_track_hours,
        "model_coarsen": args.model_coarsen,
        "model_root": str(args.model_root),
        "imerg_root": str(args.imerg_root),
        "processed_dir": str(out_dir),
        "plots_dir": str(args.plots_dir),
        "summaries": [asdict(summary) for summary in summaries],
        "tobac_version": getattr(tobac, "__version__", "unknown"),
    }
    json_path = args.json_log_dir / f"tobac_rainfall_tracking_{period_label}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] summary={json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
