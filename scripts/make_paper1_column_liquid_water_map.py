#!/usr/bin/env python3
"""Paper1 map of vertically integrated cloud liquid water.

For each experiment and each independent hourly output (+0000..+0023), this
script clips tiny negative LIQUID_WATER values to zero, multiplies by layer
thickness derived from GEOPOTENTIEL height, integrates over height at each grid
point, then time-averages the resulting 2-D field.  The result is integral
q_liquid dz, not density-weighted liquid water path.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS


DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")
OUTPUT_DIR = PAPER_ROOT / "08_column_liquid_water_map"
CACHE_DIR = Path("/tmp/paper1_column_liquid_water_map_cache")

VARIABLE = "LIQUID_WATER"
FIGURE_NAME = "column_liquid_water_map_450dpi.png"
TEXT_NAME = "column_liquid_water_map_data.txt"
STEP_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)*\.nc$")
MANAUS_LON = -60.0217
MANAUS_LAT = -3.1190


@dataclass(frozen=True)
class ColumnMap:
    experiment: str
    lon: np.ndarray
    lat: np.ndarray
    mean_vertical_integral: np.ndarray
    counts: np.ndarray
    n_days_seen: int
    n_files_seen: int
    n_files_used: int
    missing_files: int
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build maps of time-mean vertical integral of LIQUID_WATER."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--min-lead-hour", type=int, default=0)
    parser.add_argument(
        "--max-lead-hour",
        type=int,
        default=23,
        help="Maximum forecast lead hour to include. Default 23 avoids duplicate +0024.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def list_days(exp_dir: Path) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / VARIABLE
    if not var_dir.exists():
        return []
    return sorted(p for p in var_dir.iterdir() if p.is_dir() and p.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int, max_lead_hour: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(day_dir.glob("*.nc")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        lead = int(match.group(1))
        if min_lead_hour <= lead <= max_lead_hour:
            out.append(path)
    return out


def read_field(path: Path, variable: str) -> np.ndarray:
    with Dataset(path) as ds:
        raw = ds.variables[variable][:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected (level, y, x) for {path}, got {arr.shape}")
    return arr


def layer_thickness_m(height_m: np.ndarray) -> np.ndarray:
    """Approximate layer thickness from model-level centre heights.

    Interior layer edges are halfway between adjacent GEOPOTENTIEL heights.
    Bottom and top edges are extrapolated by half the neighbouring level
    spacing. Non-monotonic or invalid layers are returned as NaN.
    """
    z = np.asarray(height_m, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"Expected (level, y, x) height array, got {z.shape}")
    if z.shape[0] < 2:
        raise ValueError("Need at least two height levels to compute dz")

    edges = np.empty((z.shape[0] + 1, *z.shape[1:]), dtype=np.float64)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    dz = np.diff(edges, axis=0)

    increasing = np.diff(z, axis=0) > 0.0
    monotonic = np.empty_like(z, dtype=bool)
    monotonic[0] = increasing[0]
    monotonic[-1] = increasing[-1]
    if z.shape[0] > 2:
        monotonic[1:-1] = increasing[:-1] & increasing[1:]
    valid = np.isfinite(z) & np.isfinite(dz) & (dz > 0.0) & monotonic
    return np.where(valid, dz, np.nan)


def read_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    return lon, lat


def cache_path(args: argparse.Namespace, experiment: str) -> Path:
    day_tag = "all-days" if args.max_days is None else f"{args.max_days}days"
    return (
        args.cache_dir
        / f"{experiment}_column_{VARIABLE.lower()}_dz_integral_lead{args.min_lead_hour}"
        f"_to{args.max_lead_hour}_{day_tag}.npz"
    )


def process_day(task: tuple[Path, Path, int, int]) -> dict[str, Any]:
    exp_dir, day_dir, min_lead_hour, max_lead_hour = task
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    n_files_seen = 0
    n_files_used = 0
    missing_files = 0

    for path in list_steps(day_dir, min_lead_hour, max_lead_hour):
        n_files_seen += 1
        height_path = exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_dir.name / path.name
        try:
            field = read_field(path, VARIABLE)
            height = read_field(height_path, "GEOPOTENTIEL")
        except Exception:
            missing_files += 1
            continue

        n_levels = min(field.shape[0], height.shape[0])
        field = field[:n_levels]
        height = height[:n_levels]

        try:
            dz = layer_thickness_m(height)
        except Exception:
            missing_files += 1
            continue

        finite = np.isfinite(field) & np.isfinite(dz)
        valid_cell = np.any(finite, axis=0)
        clipped = np.where(finite, np.maximum(field, 0.0), 0.0)
        vertical_integral = np.sum(clipped * np.where(np.isfinite(dz), dz, 0.0), axis=0)
        if sums is None:
            sums = np.zeros(vertical_integral.shape, dtype=np.float64)
            counts = np.zeros(vertical_integral.shape, dtype=np.int64)
        sums[valid_cell] += vertical_integral[valid_cell]
        counts[valid_cell] += 1
        n_files_used += 1

    return {
        "sums": sums,
        "counts": counts,
        "n_files_seen": n_files_seen,
        "n_files_used": n_files_used,
        "missing_files": missing_files,
    }


def combine_results(parts: list[dict[str, Any]]) -> dict[str, Any]:
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    n_files_seen = 0
    n_files_used = 0
    missing_files = 0

    for part in parts:
        if part["sums"] is not None:
            if sums is None:
                sums = np.zeros_like(part["sums"], dtype=np.float64)
                counts = np.zeros_like(part["counts"], dtype=np.int64)
            sums += part["sums"]
            counts += part["counts"]
        n_files_seen += int(part["n_files_seen"])
        n_files_used += int(part["n_files_used"])
        missing_files += int(part["missing_files"])

    if sums is None or counts is None:
        raise RuntimeError("No valid files were processed.")
    return {
        "sums": sums,
        "counts": counts,
        "n_files_seen": n_files_seen,
        "n_files_used": n_files_used,
        "missing_files": missing_files,
    }


def build_column_map(args: argparse.Namespace, experiment: str) -> ColumnMap:
    exp_dir = args.data_root / experiment
    days = list_days(exp_dir)
    if args.max_days is not None:
        days = days[: args.max_days]
    if not days:
        raise RuntimeError(f"No day directories found for {experiment}: {exp_dir}")

    sample_steps = list_steps(days[0], args.min_lead_hour, args.max_lead_hour)
    if not sample_steps:
        raise RuntimeError(f"No sample files found in {days[0]}")
    lon, lat = read_lon_lat(sample_steps[0])

    print(
        f"[{experiment}] processing {len(days)} days with {args.workers} workers; "
        f"lead {args.min_lead_hour}..{args.max_lead_hour}",
        flush=True,
    )
    tasks = [(exp_dir, day, args.min_lead_hour, args.max_lead_hour) for day in days]

    parts: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            parts.append(process_day(task))
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"[{experiment}] {idx}/{len(tasks)} days", flush=True)
    else:
        with get_context("fork").Pool(processes=args.workers) as pool:
            for idx, part in enumerate(pool.imap_unordered(process_day, tasks), start=1):
                parts.append(part)
                if idx % args.progress_every == 0 or idx == len(tasks):
                    used = sum(int(p["n_files_used"]) for p in parts)
                    print(
                        f"[{experiment}] {idx}/{len(tasks)} days; {used} files read",
                        flush=True,
                    )

    combined = combine_results(parts)
    mean = np.full(combined["sums"].shape, np.nan, dtype=np.float64)
    ok = combined["counts"] > 0
    mean[ok] = combined["sums"][ok] / combined["counts"][ok]
    return ColumnMap(
        experiment=experiment,
        lon=lon,
        lat=lat,
        mean_vertical_integral=mean,
        counts=combined["counts"],
        n_days_seen=len(days),
        n_files_seen=combined["n_files_seen"],
        n_files_used=combined["n_files_used"],
        missing_files=combined["missing_files"],
        source=str(exp_dir / "masked-netcdf" / VARIABLE),
    )


def save_cache(path: Path, data: ColumnMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment=np.asarray(data.experiment),
        lon=data.lon,
        lat=data.lat,
        mean_vertical_integral=data.mean_vertical_integral,
        counts=data.counts,
        n_days_seen=np.asarray(data.n_days_seen, dtype=np.int64),
        n_files_seen=np.asarray(data.n_files_seen, dtype=np.int64),
        n_files_used=np.asarray(data.n_files_used, dtype=np.int64),
        missing_files=np.asarray(data.missing_files, dtype=np.int64),
        source=np.asarray(data.source),
    )


def load_cache(path: Path) -> ColumnMap:
    with np.load(path, allow_pickle=False) as data:
        return ColumnMap(
            experiment=str(data["experiment"]),
            lon=np.asarray(data["lon"], dtype=np.float64),
            lat=np.asarray(data["lat"], dtype=np.float64),
            mean_vertical_integral=np.asarray(data["mean_vertical_integral"], dtype=np.float64),
            counts=np.asarray(data["counts"], dtype=np.int64),
            n_days_seen=int(data["n_days_seen"]),
            n_files_seen=int(data["n_files_seen"]),
            n_files_used=int(data["n_files_used"]),
            missing_files=int(data["missing_files"]),
            source=str(data["source"]),
        )


def get_column_map(args: argparse.Namespace, experiment: str) -> ColumnMap:
    path = cache_path(args, experiment)
    if path.exists() and not args.recompute:
        print(f"[{experiment}] using cache {path}", flush=True)
        return load_cache(path)
    data = build_column_map(args, experiment)
    save_cache(path, data)
    print(f"[{experiment}] saved cache {path}", flush=True)
    return data


def colour_limit(maps: dict[str, ColumnMap], args: argparse.Namespace) -> float:
    if args.vmax is not None:
        return float(args.vmax)
    values = np.concatenate(
        [
            item.mean_vertical_integral[np.isfinite(item.mean_vertical_integral)]
            for item in maps.values()
        ]
    )
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(values, args.vmax_percentile))
    return max(vmax, float(np.nanmax(values)) * 0.1, 1.0e-12)


def add_map_context(ax: Any, ccrs: Any, cfeature: Any, lon: np.ndarray, lat: np.ndarray) -> None:
    ax.set_extent(
        [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="10m", linewidth=0.45)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.35, alpha=0.45)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.plot(
        MANAUS_LON,
        MANAUS_LAT,
        marker="*",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


def plot_maps(
    maps: dict[str, ColumnMap],
    experiments: list[str],
    output_path: Path,
    *,
    vmax: float,
    dpi: int,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    output_path.parent.mkdir(parents=True, exist_ok=True)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.2, 4.8))
    grid = fig.add_gridspec(2, len(experiments), height_ratios=[1.0, 0.045], hspace=0.18)
    axes = [fig.add_subplot(grid[0, i], projection=projection) for i in range(len(experiments))]
    cbar_ax = fig.add_subplot(grid[1, :])

    image = None
    for index, experiment in enumerate(experiments):
        data = maps[experiment]
        ax = axes[index]
        image = ax.pcolormesh(
            data.lon,
            data.lat,
            data.mean_vertical_integral,
            transform=projection,
            shading="auto",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(EXPERIMENT_LABELS[experiment], fontsize=10, fontweight="normal")
        add_map_context(ax, ccrs, cfeature, data.lon, data.lat)
        ax.text(
            0.02,
            0.98,
            f"mean={np.nanmean(data.mean_vertical_integral):.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
        )

    if image is None:
        raise RuntimeError("No maps to plot.")
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Time-mean vertical integral of LIQUID_WATER (kg kg$^{-1}$ m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle("Vertically integrated cloud liquid water", fontsize=12, fontweight="normal", y=0.99)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text_data(
    maps: dict[str, ColumnMap],
    experiments: list[str],
    output_path: Path,
    *,
    figure_path: Path,
    vmax: float,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Paper1 vertically integrated cloud liquid water map\n")
        fh.write("#\n")
        fh.write("# Method:\n")
        fh.write("#   For each hourly file, LIQUID_WATER is clipped at zero and multiplied by dz_m.\n")
        fh.write("#   dz_m is derived from GEOPOTENTIEL model-level centre heights using midpoint layer edges.\n")
        fh.write("#   The map shows the time mean of sum(LIQUID_WATER * dz_m) at each grid point.\n")
        fh.write("#   This is a height integral of kg/kg values, not density-weighted liquid water path.\n")
        fh.write(f"#   Forecast lead hours used: {args.min_lead_hour}..{args.max_lead_hour}; +0024 excluded.\n")
        fh.write(f"#   Colour scale: 0 to {vmax:.10e} kg kg-1 m.\n")
        fh.write("#\n")
        fh.write(f"# Figure: {figure_path}\n")
        fh.write("#\n")
        fh.write("# Summary by experiment:\n")
        for experiment in experiments:
            item = maps[experiment]
            vals = item.mean_vertical_integral[np.isfinite(item.mean_vertical_integral)]
            fh.write(
                "#   "
                f"{experiment} ({EXPERIMENT_LABELS[experiment]}): "
                f"days={item.n_days_seen}, files_seen={item.n_files_seen}, "
                f"files_used={item.n_files_used}, missing_or_failed_files={item.missing_files}, "
                f"mean={np.nanmean(vals):.10e}, median={np.nanmedian(vals):.10e}, "
                f"max={np.nanmax(vals):.10e}, source={item.source}\n"
            )
        fh.write("#\n")
        fh.write(
            "experiment\texperiment_label\ty_index\tx_index\tlat\tlon\t"
            "mean_vertical_integral_LIQUID_WATER_kg_kg-1_m\tcount_timesteps\n"
        )
        for experiment in experiments:
            item = maps[experiment]
            ny, nx = item.mean_vertical_integral.shape
            for y in range(ny):
                for x in range(nx):
                    value = item.mean_vertical_integral[y, x]
                    if not np.isfinite(value):
                        continue
                    fh.write(
                        f"{experiment}\t{EXPERIMENT_LABELS[experiment]}\t{y}\t{x}\t"
                        f"{item.lat[y, x]:.6f}\t{item.lon[y, x]:.6f}\t"
                        f"{value:.10e}\t{int(item.counts[y, x])}\n"
                    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    maps = {experiment: get_column_map(args, experiment) for experiment in args.experiments}
    vmax = colour_limit(maps, args)
    fig_path = args.output_dir / FIGURE_NAME
    txt_path = args.output_dir / TEXT_NAME
    plot_maps(maps, args.experiments, fig_path, vmax=vmax, dpi=args.dpi)
    write_text_data(maps, args.experiments, txt_path, figure_path=fig_path, vmax=vmax, args=args)
    print(f"[done] figure: {fig_path}", flush=True)
    print(f"[done] data:   {txt_path}", flush=True)


if __name__ == "__main__":
    main()
