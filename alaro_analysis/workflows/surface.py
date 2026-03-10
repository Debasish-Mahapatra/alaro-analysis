#!/usr/bin/env python3
"""
Surface-only diurnal diagnostics for masked NetCDF outputs.

Typical use case:
- Convert SURFEX hourly files such as SFX.RN to masked NetCDF first.
- Then aggregate the masked 2D field over the kept lat/lon points.
- Plot the 24-hour diurnal cycle for control / graupel / 2mom.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS, SEASONS
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.data.cache import build_cache_file, load_cache, save_cache
from alaro_analysis.data.dataset_io import read_time_level_yx
from alaro_analysis.data.discovery import collect_file_records

DEFAULT_CONTROL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/control/masked-netcdf"
)
DEFAULT_GRAUPEL_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/graupel/masked-netcdf"
)
DEFAULT_2MOM_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/SURFEX/2mom/masked-netcdf"
)

DEFAULT_OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/surface")
DEFAULT_INTERMEDIATE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/surface"
)

VAR_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diurnal cycles for masked 2D surface fields such as SURFEX SFX.RN."
    )
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--graupel-dir", type=Path, default=DEFAULT_GRAUPEL_DIR)
    parser.add_argument("--twomom-dir", type=Path, default=DEFAULT_2MOM_DIR)
    parser.add_argument(
        "--variable",
        default="SFX.RN",
        help="Masked NetCDF variable directory/name to analyze (default: SFX.RN).",
    )
    parser.add_argument(
        "--variable-label",
        default="Net radiation",
        help="Plot label for the selected variable.",
    )
    parser.add_argument(
        "--variable-unit",
        default="W m-2",
        help="Unit label for the selected variable.",
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
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate cache files.",
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
        "--y-slice",
        default=None,
        help="Optional Python-style Y slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--x-slice",
        default=None,
        help="Optional Python-style X slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List discovered variables per experiment and exit.",
    )
    parser.add_argument(
        "--zoom-inset",
        action="store_true",
        help="Add a zoomed inset around the daytime peak to show small differences.",
    )
    return parser.parse_args()


def normalize_var_token(name: str) -> str:
    return VAR_TOKEN_RE.sub("", name).upper()


def discover_variable_maps(experiment_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for exp, exp_dir in experiment_dirs.items():
        token_map: dict[str, str] = {}
        for p in sorted(exp_dir.iterdir()):
            if not p.is_dir() or p.name.startswith(".") or not has_pf_subdirs(p):
                continue
            token = normalize_var_token(p.name)
            if token and token not in token_map:
                token_map[token] = p.name
        maps[exp] = token_map
    return maps


def resolve_var_name(
    variable_maps: dict[str, dict[str, str]],
    experiment: str,
    candidates: Sequence[str],
) -> str | None:
    token_map = variable_maps[experiment]
    for cand in candidates:
        token = normalize_var_token(cand)
        if token in token_map:
            return token_map[token]
    return None


def safe_scalar_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def compute_surface_line(
    records: list[tuple[int, Path]],
    variable_name: str,
    spatial_window,
) -> tuple[np.ndarray, np.ndarray, int]:
    sums = np.zeros((24,), dtype=np.float64)
    counts = np.zeros((24,), dtype=np.int64)
    used = 0

    for idx, (hour, file_path) in enumerate(records, start=1):
        arr = read_time_level_yx(
            file_path,
            variable_name,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
        )
        value = safe_scalar_mean(arr)
        used += 1
        if np.isfinite(value):
            sums[hour] += value
            counts[hour] += 1

        if idx % 2000 == 0 or idx == len(records):
            print(f"[{variable_name}] {idx}/{len(records)} files", flush=True)

    mean = np.full((24,), np.nan, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean, counts, used


def plot_surface_diurnal(
    *,
    variable_label: str,
    variable_unit: str,
    period_label: str,
    line_data: dict[str, np.ndarray],
    output_file: Path,
    utc_offset_hours: int,
    zoom_inset: bool,
) -> None:
    colors = {
        "control": "#d62728",
        "graupel": "#1f77b4",
        "2mom": "#2ca02c",
    }
    hours = np.arange(24, dtype=np.float64)
    ylabel = variable_label
    if variable_unit:
        ylabel = f"{ylabel} [{variable_unit}]"

    def draw_lines(ax: plt.Axes, *, configure_main_axis: bool) -> None:
        for exp in EXPERIMENTS:
            arr = line_data.get(exp)
            if arr is None:
                continue
            ax.plot(
                hours,
                arr,
                linewidth=2.4,
                marker="o",
                markersize=3.5,
                color=colors[exp],
                label=EXPERIMENT_LABELS[exp],
            )
        ax.grid(alpha=0.25, linestyle="--")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if configure_main_axis:
            ax.set_xticks(np.arange(0, 24, 3))
            ax.set_xlim(0.0, 23.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    draw_lines(ax, configure_main_axis=True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_title(
        f"{period_label} - {variable_label} diurnal cycle",
        fontsize=14,
        fontweight="bold",
    )

    if zoom_inset:
        arrays = [
            np.asarray(line_data[exp], dtype=np.float64)
            for exp in EXPERIMENTS
            if line_data.get(exp) is not None and np.isfinite(line_data[exp]).any()
        ]
        if arrays:
            stacked = np.vstack(arrays)
            mean_profile = np.nanmean(stacked, axis=0)
            if np.isfinite(mean_profile).any():
                peak = float(np.nanmax(mean_profile))
                focus = np.isfinite(mean_profile) & (mean_profile >= 0.82 * peak)
                if np.any(focus):
                    focus_idx = np.where(focus)[0]
                    x0 = max(0, int(focus_idx[0]) - 1)
                    x1 = min(23, int(focus_idx[-1]) + 1)
                    zoom_vals = stacked[:, x0 : x1 + 1]
                    zoom_finite = zoom_vals[np.isfinite(zoom_vals)]
                    if zoom_finite.size > 0:
                        y0 = float(np.min(zoom_finite))
                        y1 = float(np.max(zoom_finite))
                        ypad = max(5.0, 0.14 * max(y1 - y0, 1.0))
                        axins = ax.inset_axes([0.08, 0.54, 0.36, 0.34])
                        draw_lines(axins, configure_main_axis=False)
                        axins.set_xlim(float(x0), float(x1))
                        axins.set_ylim(y0 - ypad, y1 + ypad)
                        axins.set_xticks(np.arange(x0, x1 + 1, 1))
                        axins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
                        axins.tick_params(labelsize=8)
                        ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=0.9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)

    experiment_dirs = {
        "control": args.control_dir.resolve(),
        "graupel": args.graupel_dir.resolve(),
        "2mom": args.twomom_dir.resolve(),
    }
    for exp, d in experiment_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"{exp} data dir not found: {d}")

    variable_maps = discover_variable_maps(experiment_dirs)
    if args.list_variables:
        for exp in EXPERIMENTS:
            names = sorted(variable_maps[exp].values())
            print(f"\n{exp} ({len(names)} vars):")
            print(", ".join(names) if names else "(none)")
        return

    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")

    resolved = {
        exp: resolve_var_name(variable_maps, exp, (args.variable,))
        for exp in EXPERIMENTS
    }

    print("\nInput data directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {experiment_dirs[exp]}", flush=True)
    print("\nResolved variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved[exp]}", flush=True)
    print("\nOutput directory:", args.output_dir.resolve(), flush=True)
    print("Intermediate directory:", args.intermediate_dir.resolve(), flush=True)
    print("Periods:", [p.key for p in periods], flush=True)
    print("Spatial averaging tag:", spatial_tag, flush=True)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
        flush=True,
    )
    print("Ignoring +0024 files by design.", flush=True)

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    for period in periods:
        print(f"\n===== Computing {args.variable} for {period.label} ({period.key}) =====", flush=True)
        lines_by_exp: dict[str, np.ndarray] = {}

        for exp in EXPERIMENTS:
            variable_name = resolved[exp]
            if variable_name is None:
                print(
                    f"[warn] {period.key}/{exp}: variable '{args.variable}' not found.",
                    flush=True,
                )
                continue

            variable_dir = experiment_dirs[exp] / variable_name
            records = collect_file_records(
                variable_dir=variable_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not records:
                print(f"[warn] {period.key}/{exp}: no records found in {variable_dir}", flush=True)
                continue

            cache_file = build_cache_file(
                intermediate_dir=intermediate_dir,
                analysis_name=f"{args.variable}_surface_diurnal",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            if cache_file.exists() and not args.overwrite_intermediate:
                payload = load_cache(cache_file)
                mean = np.asarray(payload["mean"], dtype=np.float64)
            else:
                mean, counts, used = compute_surface_line(
                    records=records,
                    variable_name=variable_name,
                    spatial_window=spatial_window,
                )
                if args.max_days is None:
                    save_cache(
                        cache_file,
                        {
                            "mean": mean,
                            "counts": counts,
                            "n_files": np.array([used], dtype=np.int64),
                        },
                    )
            lines_by_exp[exp] = mean

        if not lines_by_exp:
            print(f"[warn] {period.key}: no experiments produced a line.", flush=True)
            continue

        output_file = (
            output_dir
            / safe_name(args.variable)
            / period.output_subdir
            / (
                f"{safe_name(args.variable)}_diurnal_cycle"
                + (f"_{spatial_tag}" if spatial_tag != "full-domain" else "")
                + ".png"
            )
        )
        plot_surface_diurnal(
            variable_label=args.variable_label,
            variable_unit=args.variable_unit,
            period_label=period.label,
            line_data=lines_by_exp,
            output_file=output_file,
            utc_offset_hours=args.utc_offset_hours,
            zoom_inset=bool(args.zoom_inset),
        )

    print("\nCompleted surface diurnal diagnostics.", flush=True)


if __name__ == "__main__":
    main()
