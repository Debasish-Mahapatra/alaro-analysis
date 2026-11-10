"""Precipitation diurnal cycle plot from common-valid rainfall data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import RUNS_ROOT
from alaro_analysis.common.experiments import run_color, run_display, shown_runs
from alaro_analysis.common.figio import DatasetConfig, add_io_args, legend_below, save_figure

DEFAULT_DATA_DIR = (
    RUNS_ROOT / "rainfall-conservative-rebuild" / "common-valid-time-production-bilinearmask"
)
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "rainfall_diurnal_cycle"
DEFAULT_OUTPUT_NAME = "diurnal_cycle_common_valid.png"


@dataclass(frozen=True)
class HourlyStats:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


_ALL_DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall", "black", "-", 3.2),
    DatasetConfig("imerg", "IMERG", "IMERG_common_valid.nc", "rainfall", "dimgray", ":", 3.6),
    DatasetConfig("nodiag", run_display("nodiag"), "C1M_nodiaggrp_common_valid.nc", "rainfall", run_color("nodiag"), "-", 3.0),
    DatasetConfig("control", "C1M_diaggrp", "C1M_common_valid.nc", "rainfall", "#d62728", "-", 3.0),
    DatasetConfig("graupel", "G1M", "G1M_common_valid.nc", "rainfall", "#1f77b4", "-", 3.0),
    DatasetConfig("2mom", "G2M", "G2M_common_valid.nc", "rainfall", "#2ca02c", "-", 3.0),
    DatasetConfig("no3m", "G2M-XCU", "G2M-XCU_common_valid.nc", "rainfall", "#9467bd", "-", 3.0),
)
# Without the run with the diagnostic graupel when the four-run set is drawn.
DATASETS = tuple(cfg for cfg in _ALL_DATASETS if shown_runs((cfg.key,)))

OBS_SHADE_KEYS = ("radar", "imerg")
SHADE_SETTINGS = {
    "radar": {"fill_color": "deepskyblue", "alpha": 0.30},
    "imerg": {"fill_color": "lightgrey", "alpha": 0.40},
}


def local_hours_from_utc(times: np.ndarray, utc_offset_hours: int) -> np.ndarray:
    """Return local hour labels from UTC datetime64 values."""
    utc = np.asarray(times, dtype="datetime64[ns]")
    local = utc + np.timedelta64(int(utc_offset_hours), "h")
    return (local.astype("datetime64[h]").astype(np.int64) % 24).astype(np.int16)


def compute_hourly_stats(values: np.ndarray, local_hours: np.ndarray) -> HourlyStats:
    """Compute mean, sample std, and count for each local hour."""
    vals = np.asarray(values, dtype=np.float64)
    hrs = np.asarray(local_hours, dtype=np.int16)
    if vals.shape[0] != hrs.shape[0]:
        raise ValueError(f"value/time length mismatch: {vals.shape[0]} vs {hrs.shape[0]}")

    means = np.full(24, np.nan, dtype=np.float64)
    stds = np.full(24, np.nan, dtype=np.float64)
    counts = np.zeros(24, dtype=np.int64)
    for hour in range(24):
        selected = vals[hrs == hour]
        selected = selected[np.isfinite(selected)]
        counts[hour] = selected.size
        if selected.size:
            means[hour] = float(np.mean(selected))
        if selected.size > 1:
            stds[hour] = float(np.std(selected, ddof=1))
    return HourlyStats(mean=means, std=stds, count=counts)


def shading_bounds(
    stats: HourlyStats,
    *,
    mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "std":
        spread = std_multiplier * stats.std
        lower = stats.mean - spread
        upper = stats.mean + spread
    elif mode == "percent":
        lower = stats.mean * (1.0 - percent_uncertainty)
        upper = stats.mean * (1.0 + percent_uncertainty)
    elif mode == "sampling":
        # 95 % range of each hourly mean from the day-to-day spread: how much it could
        # change with another sample of days of the same kind (one value per day and hour)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = 1.96 * stats.std / np.sqrt(stats.count)
        lower = stats.mean - spread
        upper = stats.mean + spread
    else:
        raise ValueError(f"Unsupported shading mode: {mode}")
    return np.clip(lower, 0.0, None), np.clip(upper, 0.0, None)


def read_domain_mean_stats(
    data_dir: Path,
    cfg: DatasetConfig,
    *,
    utc_offset_hours: int,
) -> HourlyStats:
    path = data_dir / cfg.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {cfg.label} file: {path}")
    with xr.open_dataset(path) as ds:
        if cfg.variable not in ds:
            raise KeyError(f"{cfg.variable!r} not found in {path}")
        da = ds[cfg.variable]
        if "time" not in da.dims:
            raise ValueError(f"{cfg.variable!r} in {path} has no time dimension")
        space_dims = [dim for dim in da.dims if dim != "time"]
        if not space_dims:
            raise ValueError(f"{cfg.variable!r} in {path} has no spatial dimensions")
        domain_mean = da.mean(dim=space_dims, skipna=True)
        values = np.asarray(domain_mean.values, dtype=np.float64)
        hours = local_hours_from_utc(ds["time"].values, utc_offset_hours)
    return compute_hourly_stats(values, hours)


def plot_diurnal_cycle(
    stats_by_key: dict[str, HourlyStats],
    *,
    output_path: Path,
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
    dpi: int,
    period_label: str | None = None,
) -> None:
    """Mean diurnal cycles; ``period_label`` (e.g. "IOP1") goes just above the line legend."""
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 7))
    shade_handles, tops = _draw_cycles(ax, stats_by_key, shade_mode=shade_mode,
                                       std_multiplier=std_multiplier, percent_uncertainty=percent_uncertainty)

    ax.set_ylabel(r"Rainfall rate (mm h$^{-1}$)", fontsize=18)
    ax.set_xlabel("Local time (UTC-4)", fontsize=18)
    ax.set_xticks(hours)
    ax.set_xlim(0, 23)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.3)

    if period_label:
        # One legend, lines and shading, in the empty upper left (the afternoon peaks fill
        # the upper right), lowered a little with the period written just above it.
        line_handles, _ = ax.get_legend_handles_labels()
        legend = ax.legend(handles=line_handles + shade_handles, loc="upper left",
                           bbox_to_anchor=(0.0, 0.93), fontsize=14, framealpha=0.9, ncol=2)
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(ax.transAxes.inverted())
        # raise the top of the y-axis until nothing under the legend reaches it
        x0, x1 = ax.get_xlim()
        under = (hours >= x0 + box.x0 * (x1 - x0) - 0.5) & (hours <= x0 + box.x1 * (x1 - x0) + 0.5)
        highest = float(np.nanmax(np.stack(tops)[:, under]))
        y0, y1 = ax.get_ylim()
        room = box.y0 - 0.03
        if (highest - y0) / (y1 - y0) > room:
            ax.set_ylim(y0, y0 + (highest - y0) / room)
        ax.text(0.5 * (box.x0 + box.x1), box.y1 + 0.008, period_label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    else:
        line_legend = ax.legend(loc="upper left", fontsize=16, framealpha=0.9)
        ax.add_artist(line_legend)
        ax.legend(
            handles=shade_handles,
            loc="upper right",
            fontsize=14,
            framealpha=0.9,
            title="Shaded uncertainty",
            title_fontsize=14,
        )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_diurnal_cycle_periods(
    stats_by_period: dict[str, dict[str, HourlyStats]],
    *,
    output_path: Path,
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
    dpi: int,
) -> None:
    """Mean diurnal cycles of three periods (period label -> stats): the first across the top
    row, the other two side by side below, one legend under them; each panel keeps its own
    rain-rate axis."""
    hours = np.arange(24)
    fig = plt.figure(figsize=(15.0, 11.0))
    grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.16)
    axes = np.array([fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])])
    for index, (ax, (label, stats_by_key)) in enumerate(zip(axes, stats_by_period.items())):
        shade_handles, _ = _draw_cycles(ax, stats_by_key, shade_mode=shade_mode,
                                        std_multiplier=std_multiplier, percent_uncertainty=percent_uncertainty)
        ax.set_xticks(hours[::3])
        ax.set_xlim(0, 23)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Local time (UTC-4)", fontsize=17)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.text(0.03, 0.97, f"({'abcdefgh'[index]}) {label}", transform=ax.transAxes, ha="left", va="top",
                fontsize=17, fontweight="bold")
    for ax in axes[:2]:
        ax.set_ylabel(r"Rainfall rate (mm h$^{-1}$)", fontsize=17)
    line_handles, _ = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.10)
    legend_below(fig, list(axes[1:]), line_handles + shade_handles, ncol=4, fontsize=16)
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _draw_cycles(
    ax,
    stats_by_key: dict[str, HourlyStats],
    *,
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
) -> tuple[list[Patch], list[np.ndarray]]:
    """Every dataset's mean cycle on ``ax``, the observations shaded; returns the shading
    legend handles and the highest drawn value per hour of each line and band."""
    hours = np.arange(24)
    shade_handles: list[Patch] = []
    tops: list[np.ndarray] = []  # highest drawn value per hour, to keep the legend clear of data
    for cfg in DATASETS:
        stats = stats_by_key[cfg.key]
        if cfg.key in OBS_SHADE_KEYS:
            lower, upper = shading_bounds(
                stats,
                mode=shade_mode,
                std_multiplier=std_multiplier,
                percent_uncertainty=percent_uncertainty,
            )
            setting = SHADE_SETTINGS[cfg.key]
            tops.append(np.asarray(upper, dtype=float))
            ax.fill_between(
                hours,
                lower,
                upper,
                color=setting["fill_color"],
                alpha=setting["alpha"],
                linewidth=0,
                zorder=1,
            )
            if shade_mode == "std":
                shade_label = f"{cfg.label} ±{std_multiplier:g} sd"
            elif shade_mode == "sampling":
                shade_label = f"{cfg.label} 95 % sampling range"
            else:
                shade_label = f"{cfg.label} ±{100.0 * percent_uncertainty:g} %"
            shade_handles.append(
                Patch(
                    facecolor=setting["fill_color"],
                    edgecolor="none",
                    alpha=setting["alpha"],
                    label=shade_label,
                )
            )

        tops.append(np.asarray(stats.mean, dtype=float))
        ax.plot(
            hours,
            stats.mean,
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=cfg.linewidth,
            zorder=3,
        )
    return shade_handles, tops


def write_diurnal_txt(
    txt_path: Path,
    *,
    data_dir: Path,
    stats_by_key: dict[str, HourlyStats],
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
    utc_offset_hours: int,
    output_path: Path,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation Diurnal Cycle Plot Data\n")
        fh.write("=====================================\n")
        fh.write(f"Source plot: {output_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write(
            "Method: spatial mean over the common-valid radar mask, then mean by local hour.\n"
        )
        fh.write(f"Local time offset: UTC{utc_offset_hours:+d}\n")
        fh.write(f"Shading mode: {shade_mode}\n")
        if shade_mode == "sampling":
            fh.write("Sampling range: mean +/- 1.96 std / sqrt(n), std of the hourly domain means over "
                     "the n days at each local hour.\n")
        else:
            fh.write(f"Standard deviation multiplier: {std_multiplier:g}\n")
            fh.write(f"Percent uncertainty: {percent_uncertainty:g}\n")
        fh.write("Shaded datasets: Radar, IMERG(GPM)\n\n")

        fh.write("Dataset summary\n")
        fh.write(
            "dataset,total_hourly_samples,min_count_per_hour,max_count_per_hour,daily_sum_mm_day\n"
        )
        for cfg in DATASETS:
            stats = stats_by_key[cfg.key]
            fh.write(
                f"{cfg.label},{int(stats.count.sum())},{int(stats.count.min())},"
                f"{int(stats.count.max())},{float(np.nansum(stats.mean)):.10g}\n"
            )
        fh.write("\n")

        columns = ["local_hour"]
        for cfg in DATASETS:
            columns.extend(
                [
                    f"{cfg.label}_mean_mm_h",
                    f"{cfg.label}_std_mm_h",
                    f"{cfg.label}_count",
                    f"{cfg.label}_shade_lower",
                    f"{cfg.label}_shade_upper",
                ]
            )
        fh.write("Hourly plotted data\n")
        fh.write("-------------------\n")
        fh.write(",".join(columns) + "\n")
        for hour in range(24):
            row: list[str] = [str(hour)]
            for cfg in DATASETS:
                stats = stats_by_key[cfg.key]
                if cfg.key in OBS_SHADE_KEYS:
                    lower, upper = shading_bounds(
                        stats,
                        mode=shade_mode,
                        std_multiplier=std_multiplier,
                        percent_uncertainty=percent_uncertainty,
                    )
                else:
                    lower = np.full(24, np.nan)
                    upper = np.full(24, np.nan)
                row.extend(
                    [
                        f"{stats.mean[hour]:.12g}",
                        f"{stats.std[hour]:.12g}",
                        str(int(stats.count[hour])),
                        f"{lower[hour]:.12g}",
                        f"{upper[hour]:.12g}",
                    ]
                )
            fh.write(",".join(row) + "\n")


def make_diurnal_cycle(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = DEFAULT_OUTPUT_NAME,
    shade_mode: str = "percent",
    std_multiplier: float = 1.0,
    percent_uncertainty: float = 0.10,
    utc_offset_hours: int = -4,
    dpi: int = 400,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    stats_by_key = {
        cfg.key: read_domain_mean_stats(data_dir, cfg, utc_offset_hours=utc_offset_hours)
        for cfg in DATASETS
    }
    plot_diurnal_cycle(
        stats_by_key,
        output_path=output_path,
        shade_mode=shade_mode,
        std_multiplier=std_multiplier,
        percent_uncertainty=percent_uncertainty,
        dpi=dpi,
    )
    txt_path = output_dir / "data_txt" / f"{output_path.stem}.txt"
    write_diurnal_txt(
        txt_path,
        data_dir=data_dir,
        stats_by_key=stats_by_key,
        shade_mode=shade_mode,
        std_multiplier=std_multiplier,
        percent_uncertainty=percent_uncertainty,
        utc_offset_hours=utc_offset_hours,
        output_path=output_path,
    )
    return {"plot": output_path, "txt": txt_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot precipitation diurnal cycle from common-valid rainfall files."
    )
    add_io_args(parser, default_data_dir=DEFAULT_DATA_DIR, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--shade-mode", choices=["percent", "std"], default="percent")
    parser.add_argument("--std-multiplier", type=float, default=1.0)
    parser.add_argument("--percent-uncertainty", type=float, default=0.10)
    parser.add_argument("--utc-offset-hours", type=int, default=-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_configured_args(build_parser(), "precip_diurnal_cycle", argv=argv)
    outputs = make_diurnal_cycle(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        shade_mode=args.shade_mode,
        std_multiplier=args.std_multiplier,
        percent_uncertainty=args.percent_uncertainty,
        utc_offset_hours=args.utc_offset_hours,
        dpi=args.dpi,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    raise SystemExit(main())
