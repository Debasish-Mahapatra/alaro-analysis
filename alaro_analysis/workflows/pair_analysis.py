#!/usr/bin/env python3
"""
Cache-only plotting workflow for normal vs RAD liquid/solid water pairs.

Outputs in a separate tree:
1) Individual variable panels (C1M absolute, G1M-C1M, G2M-G1M)
2) Pair comparison panels per experiment
   - normal absolute
   - RAD absolute
   - RAD - normal
   - normal - RAD

This script does not process raw netCDF files. It reads existing intermediate
.npz files produced by the main hydrometeor script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import (
    EXPERIMENTS,
    EXPERIMENT_LABELS,
    FREEZING_K,
    SEASONS,
)
from alaro_analysis.common.models import AxisSpec, PeriodSpec
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.vertical import centers_to_edges

try:
    import cmaps  # type: ignore
except Exception:
    cmaps = None

PAIR_PRESETS = {
    "liquid": ("LIQUID_WATER", "RAD_LIQUID_W"),
    "solid": ("SOLID_WATER", "RAD_SOLID_WA"),
}

DEFAULT_INTERMEDIATE_DIRS = (
    Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/hydrometeors"),
    Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data"),
)
DEFAULT_OUTPUT_DIR = Path(
    "/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/hydrometeor_pair_analysis"
)

ABS_CMAP = cmaps.WhiteBlueGreenYellowRed if cmaps is not None else "turbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair workflow plots for LIQUID/SOLID water vs RAD variables."
    )
    parser.add_argument(
        "--intermediate-dirs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INTERMEDIATE_DIRS),
        help="One or more intermediate-cache roots to search (first match wins).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for this dedicated workflow.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=("liquid", "solid", "all"),
        default=("all",),
        help="Which pair presets to include.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=("individual", "pair", "sum_diff"),
        default=("individual", "pair", "sum_diff"),
        help="Which outputs to generate: individual panels, pair panels, sum-diff grids, or all.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Season subset for seasonal mode; use 'all' for all seasons.",
    )
    parser.add_argument(
        "--temperature-variable",
        default="TEMPERATURE",
        help="Temperature cache variable for freezing-level overlay.",
    )
    parser.add_argument(
        "--skip-freezing-level",
        action="store_true",
        help="Disable freezing-level overlay.",
    )
    parser.add_argument(
        "--height-aggregate",
        choices=("first", "mean-all"),
        default="first",
        help="Geopotential height cache style to use.",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit when using height axis.",
    )
    parser.add_argument(
        "--absolute-low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for pair-shared absolute log scale.",
    )
    parser.add_argument(
        "--absolute-high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for pair-shared absolute log scale.",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=98.0,
        help="Percentile for symmetric anomaly scales.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Figure dpi (default 450).",
    )
    parser.add_argument(
        "--scale-reference",
        choices=("full_2yr", "all_periods"),
        default="full_2yr",
        help=(
            "Reference set for colorbar limits. "
            "'full_2yr' uses only full-period caches; 'all_periods' uses all selected periods."
        ),
    )
    parser.add_argument(
        "--cross-anomaly-mode",
        choices=("both", "normal-minus-rad", "rad-minus-normal"),
        default="both",
        help=(
            "Direction(s) for pair anomalies. "
            "'normal-minus-rad' means NORMAL - RAD only."
        ),
    )
    parser.add_argument(
        "--pair-layout",
        choices=("panel", "single", "experiments"),
        default="panel",
        help=(
            "Layout for pair output. 'panel' keeps multi-subplot figure, "
            "'single' saves one selected cross-anomaly map per experiment, "
            "'experiments' saves one figure with C1M/G1M/G2M side-by-side."
        ),
    )
    return parser.parse_args()


def variable_label(name: str) -> str:
    return name


def resolve_selected_pairs(pair_args: tuple[str, ...] | list[str]) -> list[tuple[str, str]]:
    tokens = [p.strip().lower() for p in pair_args if p.strip()]
    if not tokens or "all" in tokens:
        return [PAIR_PRESETS["liquid"], PAIR_PRESETS["solid"]]
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        pair = PAIR_PRESETS[token]
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)
    return selected


def cache_relpath(variable: str, period: PeriodSpec, experiment: str) -> Path:
    return (
        Path(safe_name(variable))
        / period.output_subdir
        / f"{experiment}_diurnal_profile.npz"
    )


def height_relpaths(period: PeriodSpec, aggregate: str) -> list[Path]:
    return [
        Path("geopotential") / period.output_subdir / f"control_height_profile_{aggregate}.npz",
        Path("geopotentiel") / period.output_subdir / f"control_height_profile_{aggregate}.npz",
    ]


def find_existing_cache(intermediate_roots: list[Path], relpaths: list[Path]) -> Path | None:
    for root in intermediate_roots:
        for rel in relpaths:
            candidate = root / rel
            if candidate.exists():
                return candidate
    return None


def load_diurnal_mean(
    intermediate_roots: list[Path], variable: str, period: PeriodSpec, experiment: str
) -> np.ndarray:
    rel = cache_relpath(variable, period, experiment)
    found = find_existing_cache(intermediate_roots, [rel])
    if found is None:
        raise FileNotFoundError(f"Missing cache: {rel} in any of {intermediate_roots}")
    payload = np.load(found)
    return np.asarray(payload["mean"], dtype=np.float64)


def load_height_axis(
    intermediate_roots: list[Path],
    period: PeriodSpec,
    aggregate: str,
    fallback_levels: int,
) -> AxisSpec:
    found = find_existing_cache(intermediate_roots, height_relpaths(period, aggregate))
    if found is None:
        return AxisSpec(
            values=np.arange(fallback_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )
    payload = np.load(found)
    height_m = np.asarray(payload["height_m"], dtype=np.float64)
    return AxisSpec(values=height_m / 1000.0, label="Height (km)", is_height_km=True)


def nanmean_with_count(data: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data)
    count = np.sum(valid, axis=axis)
    total = np.nansum(data, axis=axis)
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    ok = count > 0
    mean[ok] = total[ok] / count[ok]
    return mean, count.astype(np.int64)


def infer_freezing_threshold(temperature_profile: np.ndarray) -> float | None:
    valid = temperature_profile[np.isfinite(temperature_profile)]
    if valid.size == 0:
        return None
    return FREEZING_K if float(np.median(valid)) > 150.0 else 0.0


def compute_freezing_line_km(axis: AxisSpec, temperature_profiles: list[np.ndarray]) -> np.ndarray | None:
    if (not axis.is_height_km) or (not temperature_profiles):
        return None
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in temperature_profiles))
    if n_levels < 2:
        return None
    stacked = np.stack([arr[:n_levels, :] for arr in temperature_profiles], axis=0)
    mean_temp, _ = nanmean_with_count(stacked, axis=(0,))
    threshold = infer_freezing_threshold(mean_temp)
    if threshold is None:
        return None

    y = np.asarray(axis.values[:n_levels], dtype=np.float64)
    order = np.argsort(y)
    y_sorted = y[order]
    t_sorted = mean_temp[order, :]
    line = np.full((24,), np.nan, dtype=np.float64)
    for hour in range(24):
        col = t_sorted[:, hour]
        finite = np.isfinite(col) & np.isfinite(y_sorted)
        if np.sum(finite) < 2:
            continue
        yy = y_sorted[finite]
        tt = col[finite]
        for idx in range(yy.size - 1):
            t1, t2 = tt[idx], tt[idx + 1]
            y1, y2 = yy[idx], yy[idx + 1]
            d1 = t1 - threshold
            d2 = t2 - threshold
            if np.isclose(t1, threshold):
                line[hour] = y1
                break
            if np.isclose(t2, threshold):
                line[hour] = y2
                break
            if d1 * d2 < 0 and not np.isclose(t1, t2):
                frac = (threshold - t1) / (t2 - t1)
                line[hour] = y1 + frac * (y2 - y1)
                break
    return line


def crop_to_axis(
    axis: AxisSpec,
    arrays: list[np.ndarray],
    max_height_km: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    y = np.asarray(axis.values, dtype=np.float64)
    n_levels = min(y.size, *(arr.shape[0] for arr in arrays))
    y = y[:n_levels]
    trimmed = [arr[:n_levels, :] for arr in arrays]
    order = np.argsort(y)
    y = y[order]
    trimmed = [arr[order, :] for arr in trimmed]
    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        trimmed = [arr[keep, :] for arr in trimmed]
    return y, trimmed


def compute_variable_scales(
    variable: str,
    scale_periods: list[PeriodSpec],
    profiles: dict[tuple[str, str, str], np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
    anomaly_percentile: float,
) -> tuple[tuple[float, float], float]:
    abs_chunks: list[np.ndarray] = []
    model_anom_chunks: list[np.ndarray] = []

    for period in scale_periods:
        for exp in EXPERIMENTS:
            data = profiles[(period.key, variable, exp)]
            valid = data[np.isfinite(data) & (data > 0)]
            if valid.size > 0:
                abs_chunks.append(valid)

        c = profiles[(period.key, variable, "control")]
        g = profiles[(period.key, variable, "graupel")]
        t = profiles[(period.key, variable, "2mom")]
        for arr in (g - c, t - g):
            diff_valid = arr[np.isfinite(arr)]
            if diff_valid.size > 0:
                model_anom_chunks.append(diff_valid)

    abs_limits = compute_log_abs_limits(
        abs_chunks=abs_chunks,
        absolute_low_percentile=absolute_low_percentile,
        absolute_high_percentile=absolute_high_percentile,
    )

    def anom_scale(chunks: list[np.ndarray]) -> float:
        if not chunks:
            return 1.0
        merged = np.concatenate(chunks)
        finite = np.abs(merged[np.isfinite(merged)])
        if finite.size == 0:
            return 1.0
        scale = float(np.percentile(finite, anomaly_percentile))
        if scale <= 0:
            scale = float(np.max(finite))
        return scale if scale > 0 else 1.0

    return abs_limits, anom_scale(model_anom_chunks)


def compute_pair_comparison_scales(
    pair: tuple[str, str],
    scale_periods: list[PeriodSpec],
    profiles: dict[tuple[str, str, str], np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
    anomaly_percentile: float,
) -> tuple[tuple[float, float], float]:
    normal_var, rad_var = pair
    abs_chunks: list[np.ndarray] = []
    cross_anom_chunks: list[np.ndarray] = []

    for period in scale_periods:
        for exp in EXPERIMENTS:
            normal = profiles[(period.key, normal_var, exp)]
            rad = profiles[(period.key, rad_var, exp)]

            n_valid = normal[np.isfinite(normal) & (normal > 0)]
            r_valid = rad[np.isfinite(rad) & (rad > 0)]
            if n_valid.size > 0:
                abs_chunks.append(n_valid)
            if r_valid.size > 0:
                abs_chunks.append(r_valid)

            cross = rad - normal
            c_valid = cross[np.isfinite(cross)]
            if c_valid.size > 0:
                cross_anom_chunks.append(c_valid)

    abs_limits = compute_log_abs_limits(
        abs_chunks=abs_chunks,
        absolute_low_percentile=absolute_low_percentile,
        absolute_high_percentile=absolute_high_percentile,
    )

    if cross_anom_chunks:
        merged = np.concatenate(cross_anom_chunks)
        finite = np.abs(merged[np.isfinite(merged)])
        if finite.size > 0:
            cross_scale = float(np.percentile(finite, anomaly_percentile))
            if cross_scale <= 0:
                cross_scale = float(np.max(finite))
            if cross_scale <= 0:
                cross_scale = 1.0
        else:
            cross_scale = 1.0
    else:
        cross_scale = 1.0

    return abs_limits, cross_scale


def compute_log_abs_limits(
    abs_chunks: list[np.ndarray],
    absolute_low_percentile: float,
    absolute_high_percentile: float,
) -> tuple[float, float]:
    if not abs_chunks:
        return (1e-12, 1.0)

    abs_all = np.concatenate(abs_chunks)
    pos = abs_all[np.isfinite(abs_all) & (abs_all > 0)]
    if pos.size == 0:
        return (1e-12, 1.0)

    log_pos = np.log10(pos)
    p_low = float(np.percentile(log_pos, absolute_low_percentile))
    p_high = float(np.percentile(log_pos, absolute_high_percentile))
    if p_high <= p_low:
        p_high = p_low + 1.0

    vmin = 10.0 ** p_low
    vmax = 10.0 ** p_high
    if vmax <= vmin:
        vmax = vmin * 10.0
    return (float(vmin), float(vmax))


def style_axes(ax: plt.Axes, ylabel: str, axis_label_fs: int, tick_label_fs: int) -> None:
    ax.set_facecolor("#d3d3d3")
    ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=axis_label_fs)
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_xlim(-0.5, 23.5)
    ax.tick_params(axis="both", labelsize=tick_label_fs)
    ax.set_ylabel(ylabel, fontsize=axis_label_fs)


def add_panel_tag(ax: plt.Axes, idx: int, fs: int) -> None:
    ax.text(
        0.02,
        0.98,
        f"({chr(ord('a') + idx)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fs,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
    )


def add_freezing_line(axes: list[plt.Axes], line: np.ndarray | None) -> None:
    if line is None or (not np.isfinite(line).any()):
        return
    hours = np.arange(24, dtype=np.float64)
    for idx, ax in enumerate(axes):
        label = "Freezing level" if idx == 0 else None
        ax.plot(
            hours,
            line,
            color="black",
            linewidth=1.8,
            linestyle="--",
            label=label,
            zorder=10,
        )
    axes[0].legend(loc="upper right", fontsize=12, framealpha=0.9)


def plot_individual_panels(
    variable: str,
    control: np.ndarray,
    graupel: np.ndarray,
    twomom: np.ndarray,
    axis: AxisSpec,
    max_height_km: float,
    period_label: str,
    freezing_line_km: np.ndarray | None,
    abs_limits: tuple[float, float],
    anom_scale: float,
    output_file: Path,
    dpi: int,
) -> None:
    y, arrays = crop_to_axis(axis, [control, graupel, twomom], max_height_km)
    ctrl, g1, g2 = arrays
    diff_g1 = g1 - ctrl
    diff_g2 = g2 - g1

    ctrl_plot = np.ma.masked_invalid(ctrl)
    ctrl_plot = np.ma.masked_where(ctrl_plot <= 0, ctrl_plot)
    diff_g1_plot = np.ma.masked_invalid(diff_g1)
    diff_g2_plot = np.ma.masked_invalid(diff_g2)

    vmin_abs, vmax_abs = abs_limits
    if vmax_abs <= vmin_abs:
        vmax_abs = vmin_abs * 10.0
    abs_norm = mcolors.Normalize(vmin=0, vmax=vmax_abs)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        style_axes(ax, axis.label, 16, 14)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        add_panel_tag(ax, idx, 14)

    pcm_abs = axes[0].pcolormesh(
        hour_edges, y_edges, ctrl_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto"
    )
    axes[0].set_title(f"{EXPERIMENT_LABELS['control']} ({variable_label(variable)}, Absolute)", fontsize=14, fontweight="bold")

    axes[1].pcolormesh(hour_edges, y_edges, diff_g1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[1].set_title(f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}", fontsize=14, fontweight="bold")

    pcm_diff = axes[2].pcolormesh(hour_edges, y_edges, diff_g2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[2].set_title(f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}", fontsize=14, fontweight="bold")

    add_freezing_line(list(axes), freezing_line_km)
    fig.suptitle(f"{period_label} - {variable_label(variable)}", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_abs.set_label(f"Mean {variable_label(variable)} (Absolute)", fontsize=14)
    cbar_abs.ax.tick_params(labelsize=12)

    cbar_diff = fig.colorbar(pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_diff.set_label(f"{variable_label(variable)} anomaly", fontsize=14)
    cbar_diff.ax.tick_params(labelsize=12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_pair_panel(
    pair: tuple[str, str],
    experiment: str,
    normal: np.ndarray,
    rad: np.ndarray,
    axis: AxisSpec,
    max_height_km: float,
    period_label: str,
    freezing_line_km: np.ndarray | None,
    abs_limits: tuple[float, float],
    cross_anom_scale: float,
    cross_anomaly_mode: str,
    pair_layout: str,
    output_file: Path,
    dpi: int,
) -> None:
    y, arrays = crop_to_axis(axis, [normal, rad], max_height_km)
    normal_arr, rad_arr = arrays
    rad_minus_normal = rad_arr - normal_arr
    normal_minus_rad = -rad_minus_normal

    normal_plot = np.ma.masked_invalid(normal_arr)
    normal_plot = np.ma.masked_where(normal_plot <= 0, normal_plot)
    rad_plot = np.ma.masked_invalid(rad_arr)
    rad_plot = np.ma.masked_where(rad_plot <= 0, rad_plot)
    rmn_plot = np.ma.masked_invalid(rad_minus_normal)
    nmr_plot = np.ma.masked_invalid(normal_minus_rad)

    vmin_abs, vmax_abs = abs_limits
    if vmax_abs <= vmin_abs:
        vmax_abs = vmin_abs * 10.0
    abs_norm = mcolors.LogNorm(vmin=vmin_abs, vmax=vmax_abs)
    anom_norm = mcolors.TwoSlopeNorm(vmin=-cross_anom_scale, vcenter=0.0, vmax=cross_anom_scale)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    n_label, r_label = pair
    if pair_layout == "single":
        if cross_anomaly_mode == "both":
            raise ValueError("pair_layout='single' requires one-way cross anomaly mode.")
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 7), constrained_layout=True)
        style_axes(ax, axis.label, 16, 14)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        add_panel_tag(ax, 0, 14)

        if cross_anomaly_mode == "normal-minus-rad":
            diff_arr = nmr_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({n_label} - {r_label})"
            diff_label = f"{n_label} - {r_label}"
        else:
            diff_arr = rmn_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({r_label} - {n_label})"
            diff_label = f"{r_label} - {n_label}"

        pcm_diff = ax.pcolormesh(hour_edges, y_edges, diff_arr, cmap="RdBu_r", norm=anom_norm, shading="auto")
        ax.set_title(diff_title, fontsize=14, fontweight="bold")
        add_freezing_line([ax], freezing_line_km)
        fig.suptitle(f"{period_label} - {diff_label} ({EXPERIMENT_LABELS[experiment]})", fontsize=16, fontweight="bold")

        cbar_diff = fig.colorbar(pcm_diff, ax=ax, orientation="horizontal", fraction=0.08, pad=0.16)
        cbar_diff.set_label(f"Pair anomaly ({diff_label})", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)
    elif cross_anomaly_mode == "both":
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
        flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        for idx, ax in enumerate(flat_axes):
            style_axes(ax, axis.label, 16, 14)
            if axis.is_height_km:
                ax.set_ylim(0.0, max_height_km)
            add_panel_tag(ax, idx, 14)

        pcm_abs_1 = axes[0, 0].pcolormesh(hour_edges, y_edges, normal_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto")
        axes[0, 0].set_title(f"{EXPERIMENT_LABELS[experiment]} {n_label} (Absolute)", fontsize=14, fontweight="bold")
        axes[0, 1].pcolormesh(hour_edges, y_edges, rad_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto")
        axes[0, 1].set_title(f"{EXPERIMENT_LABELS[experiment]} {r_label} (Absolute)", fontsize=14, fontweight="bold")

        pcm_diff = axes[1, 0].pcolormesh(hour_edges, y_edges, rmn_plot, cmap="RdBu_r", norm=anom_norm, shading="auto")
        axes[1, 0].set_title(f"{EXPERIMENT_LABELS[experiment]} ({r_label} - {n_label})", fontsize=14, fontweight="bold")
        axes[1, 1].pcolormesh(hour_edges, y_edges, nmr_plot, cmap="RdBu_r", norm=anom_norm, shading="auto")
        axes[1, 1].set_title(f"{EXPERIMENT_LABELS[experiment]} ({n_label} - {r_label})", fontsize=14, fontweight="bold")

        add_freezing_line(flat_axes, freezing_line_km)
        fig.suptitle(f"{period_label} - {n_label} vs {r_label} ({EXPERIMENT_LABELS[experiment]})", fontsize=16, fontweight="bold")

        cbar_abs = fig.colorbar(pcm_abs_1, ax=[axes[0, 0], axes[0, 1]], orientation="horizontal", fraction=0.08, pad=0.10)
        cbar_abs.set_label(f"Mean absolute ({n_label}, {r_label})", fontsize=14)
        cbar_abs.ax.tick_params(labelsize=12)

        cbar_diff = fig.colorbar(pcm_diff, ax=[axes[1, 0], axes[1, 1]], orientation="horizontal", fraction=0.08, pad=0.12)
        cbar_diff.set_label(f"Pair anomaly ({r_label}-{n_label} and reverse)", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
        for idx, ax in enumerate(axes):
            style_axes(ax, axis.label, 16, 14)
            if axis.is_height_km:
                ax.set_ylim(0.0, max_height_km)
            add_panel_tag(ax, idx, 14)

        pcm_abs_1 = axes[0].pcolormesh(hour_edges, y_edges, normal_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto")
        axes[0].set_title(f"{EXPERIMENT_LABELS[experiment]} {n_label} (Absolute)", fontsize=14, fontweight="bold")
        axes[1].pcolormesh(hour_edges, y_edges, rad_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto")
        axes[1].set_title(f"{EXPERIMENT_LABELS[experiment]} {r_label} (Absolute)", fontsize=14, fontweight="bold")

        if cross_anomaly_mode == "normal-minus-rad":
            diff_arr = nmr_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({n_label} - {r_label})"
            diff_label = f"{n_label} - {r_label}"
        else:
            diff_arr = rmn_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({r_label} - {n_label})"
            diff_label = f"{r_label} - {n_label}"
        pcm_diff = axes[2].pcolormesh(hour_edges, y_edges, diff_arr, cmap="RdBu_r", norm=anom_norm, shading="auto")
        axes[2].set_title(diff_title, fontsize=14, fontweight="bold")

        add_freezing_line(list(axes), freezing_line_km)
        fig.suptitle(f"{period_label} - {n_label} vs {r_label} ({EXPERIMENT_LABELS[experiment]})", fontsize=16, fontweight="bold")

        cbar_abs = fig.colorbar(pcm_abs_1, ax=axes[:2], orientation="horizontal", fraction=0.08, pad=0.16)
        cbar_abs.set_label(f"Mean absolute ({n_label}, {r_label})", fontsize=14)
        cbar_abs.ax.tick_params(labelsize=12)

        cbar_diff = fig.colorbar(pcm_diff, ax=axes[2], orientation="horizontal", fraction=0.08, pad=0.16)
        cbar_diff.set_label(f"Pair anomaly ({diff_label})", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_pair_across_experiments(
    pair: tuple[str, str],
    period_label: str,
    axis: AxisSpec,
    max_height_km: float,
    freezing_line_km: np.ndarray | None,
    cross_anom_scale: float,
    cross_anomaly_mode: str,
    per_experiment_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    output_file: Path,
    dpi: int,
) -> None:
    if cross_anomaly_mode == "both":
        raise ValueError("pair_layout='experiments' requires one-way cross anomaly mode.")

    n_label, r_label = pair
    if cross_anomaly_mode == "normal-minus-rad":
        diff_label = f"{n_label} - {r_label}"
    else:
        diff_label = f"{r_label} - {n_label}"

    y_ref, _ = crop_to_axis(
        axis,
        [per_experiment_arrays["control"][0], per_experiment_arrays["control"][1]],
        max_height_km,
    )
    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y_ref)

    diff_norm = mcolors.TwoSlopeNorm(
        vmin=-cross_anom_scale, vcenter=0.0, vmax=cross_anom_scale
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    pcm = None
    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        style_axes(ax, axis.label, 16, 14)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        add_panel_tag(ax, idx, 14)

        normal, rad = per_experiment_arrays[exp]
        y_cur, arrays = crop_to_axis(axis, [normal, rad], max_height_km)
        normal_cur, rad_cur = arrays
        if y_cur.size != y_ref.size or not np.allclose(y_cur, y_ref, equal_nan=True):
            min_levels = min(y_ref.size, y_cur.size)
            y_plot = y_cur[:min_levels]
            y_edges_plot = centers_to_edges(y_plot)
            if cross_anomaly_mode == "normal-minus-rad":
                diff_arr = normal_cur[:min_levels, :] - rad_cur[:min_levels, :]
            else:
                diff_arr = rad_cur[:min_levels, :] - normal_cur[:min_levels, :]
            diff_plot = np.ma.masked_invalid(diff_arr)
            pcm = ax.pcolormesh(
                hour_edges,
                y_edges_plot,
                diff_plot,
                cmap="RdBu_r",
                norm=diff_norm,
                shading="auto",
            )
        else:
            if cross_anomaly_mode == "normal-minus-rad":
                diff_arr = normal_cur - rad_cur
            else:
                diff_arr = rad_cur - normal_cur
            diff_plot = np.ma.masked_invalid(diff_arr)
            pcm = ax.pcolormesh(
                hour_edges,
                y_edges,
                diff_plot,
                cmap="RdBu_r",
                norm=diff_norm,
                shading="auto",
            )
        ax.set_title(f"{EXPERIMENT_LABELS[exp]} ({diff_label})", fontsize=14, fontweight="bold")

    add_freezing_line(list(axes), freezing_line_km)
    fig.suptitle(f"{period_label} - {diff_label}", fontsize=16, fontweight="bold")

    cbar = fig.colorbar(
        pcm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.16
    )
    cbar.set_label(f"Pair anomaly ({diff_label})", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_sum_diff_grid(
    pair: tuple[str, str],
    profiles: dict[tuple[str, str, str], np.ndarray],
    period: PeriodSpec,
    axis: AxisSpec,
    max_height_km: float,
    freezing_line_km: np.ndarray | None,
    output_file: Path,
    dpi: int,
) -> None:
    """3-row × 2-column grid.

    Rows  : control (C1M), graupel (G1M), 2mom (G2M)
    Col 1 : normal + RAD  (combined absolute, linear scale)
    Col 2 : (normal + RAD) − RAD  (= normal, linear scale)
    """
    normal_var, rad_var = pair
    n_label, r_label = normal_var, rad_var

    # --- collect per-experiment arrays --------------------------------
    sum_arrays: list[np.ndarray] = []
    diff_arrays: list[np.ndarray] = []
    for exp in EXPERIMENTS:
        normal = profiles[(period.key, normal_var, exp)]
        rad = profiles[(period.key, rad_var, exp)]
        combined = normal + rad
        difference = combined - rad  # = normal
        sum_arrays.append(combined)
        diff_arrays.append(difference)

    # --- crop to max height -------------------------------------------
    all_arrays = sum_arrays + diff_arrays
    y, cropped = crop_to_axis(axis, all_arrays, max_height_km)
    sum_cropped = cropped[:3]
    diff_cropped = cropped[3:]

    # --- shared linear colour limits ----------------------------------
    all_finite = np.concatenate(
        [arr[np.isfinite(arr)] for arr in sum_cropped + diff_cropped if np.isfinite(arr).any()]
    )
    if all_finite.size > 0:
        vmax = float(np.percentile(all_finite, 98))
    else:
        vmax = 1.0
    if vmax <= 0:
        vmax = 1.0
    abs_norm = mcolors.Normalize(vmin=0, vmax=vmax)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    cmap = ABS_CMAP
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), constrained_layout=True)

    col_titles = [
        f"{n_label} + {r_label}",
        f"({n_label} + {r_label}) − {r_label}",
    ]

    panel_idx = 0
    for row_idx, exp in enumerate(EXPERIMENTS):
        for col_idx, arr in enumerate((sum_cropped[row_idx], diff_cropped[row_idx])):
            ax = axes[row_idx, col_idx]
            plot_data = np.ma.masked_invalid(arr)
            pcm = ax.pcolormesh(
                hour_edges, y_edges, plot_data,
                cmap=cmap, norm=abs_norm, shading="auto",
            )
            style_axes(ax, axis.label, 14, 12)
            if axis.is_height_km:
                ax.set_ylim(0.0, max_height_km)
            add_panel_tag(ax, panel_idx, 13)
            ax.set_title(
                f"{EXPERIMENT_LABELS[exp]} — {col_titles[col_idx]}",
                fontsize=12, fontweight="bold",
            )

            # freezing line
            if freezing_line_km is not None and np.isfinite(freezing_line_km).any():
                hours = np.arange(24, dtype=np.float64)
                lbl = "Freezing level" if panel_idx == 0 else None
                ax.plot(
                    hours, freezing_line_km,
                    color="black", linewidth=1.5, linestyle="--",
                    label=lbl, zorder=10,
                )
                if lbl:
                    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
            panel_idx += 1

    fig.suptitle(
        f"{period.label} — {n_label} / {r_label} sum & difference",
        fontsize=15, fontweight="bold",
    )

    cbar = fig.colorbar(
        pcm, ax=axes, orientation="horizontal",
        fraction=0.04, pad=0.06,
    )
    cbar.set_label("Absolute (linear)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.absolute_low_percentile < 100.0):
        raise ValueError("--absolute-low-percentile must be in [0, 100).")
    if not (0.0 < args.absolute_high_percentile <= 100.0):
        raise ValueError("--absolute-high-percentile must be in (0, 100].")
    if args.absolute_low_percentile >= args.absolute_high_percentile:
        raise ValueError("--absolute-low-percentile must be lower than --absolute-high-percentile.")
    if not (0.0 < args.anomaly_percentile <= 100.0):
        raise ValueError("--anomaly-percentile must be in (0, 100].")
    if args.pair_layout in ("single", "experiments") and args.cross_anomaly_mode == "both":
        raise ValueError(
            "--pair-layout single|experiments requires "
            "--cross-anomaly-mode normal-minus-rad or rad-minus-normal."
        )

    intermediate_roots: list[Path] = []
    for root in args.intermediate_dirs:
        resolved = root.resolve()
        if resolved not in intermediate_roots:
            intermediate_roots.append(resolved)
    for root in intermediate_roots:
        if not root.exists():
            print(f"[warn] Intermediate root not found: {root}")
    if not any(root.exists() for root in intermediate_roots):
        raise FileNotFoundError("None of the intermediate roots exist.")

    mode_set = set(args.analysis_modes)
    periods = build_period_specs(mode_set, resolve_seasons(args.seasons))
    if not periods:
        raise RuntimeError("No periods selected.")

    selected_pairs = resolve_selected_pairs(args.pairs)
    variables = sorted({v for pair in selected_pairs for v in pair})
    period_map = {p.key: p for p in periods}
    if args.scale_reference == "full_2yr" and "full_2yr" in period_map:
        scale_periods = [period_map["full_2yr"]]
    else:
        scale_periods = periods

    print("Intermediate roots (search order):")
    for root in intermediate_roots:
        print(f"- {root}")
    print("Selected pairs:", selected_pairs)
    print("Outputs:", list(args.outputs))
    print("Pair layout:", args.pair_layout)
    print("Periods:", [p.key for p in periods])
    print("Scale reference periods:", [p.key for p in scale_periods])
    print(
        f"Scales: abs p{args.absolute_low_percentile:g}-p{args.absolute_high_percentile:g}, "
        f"anomaly p{args.anomaly_percentile:g}"
    )
    print("Freezing-level overlay:", not args.skip_freezing_level)

    profiles: dict[tuple[str, str, str], np.ndarray] = {}
    for period in periods:
        for variable in variables:
            for exp in EXPERIMENTS:
                profiles[(period.key, variable, exp)] = load_diurnal_mean(
                    intermediate_roots, variable, period, exp
                )

    axes_by_period: dict[str, AxisSpec] = {}
    for period in periods:
        fallback_levels = profiles[(period.key, variables[0], "control")].shape[0]
        axes_by_period[period.key] = load_height_axis(
            intermediate_roots=intermediate_roots,
            period=period,
            aggregate=args.height_aggregate,
            fallback_levels=fallback_levels,
        )

    freezing_by_period: dict[str, np.ndarray | None] = {p.key: None for p in periods}
    if not args.skip_freezing_level:
        missing_temp = False
        temp_profiles: dict[tuple[str, str], np.ndarray] = {}
        for period in periods:
            for exp in EXPERIMENTS:
                try:
                    temp_profiles[(period.key, exp)] = load_diurnal_mean(
                        intermediate_roots, args.temperature_variable, period, exp
                    )
                except FileNotFoundError:
                    missing_temp = True
                    break
            if missing_temp:
                break
        if missing_temp:
            print(
                f"[warn] Temperature cache '{args.temperature_variable}' missing in at least one period/experiment. "
                "Freezing-level overlay disabled."
            )
        else:
            for period in periods:
                freezing_by_period[period.key] = compute_freezing_line_km(
                    axes_by_period[period.key],
                    [temp_profiles[(period.key, exp)] for exp in EXPERIMENTS],
                )

    variable_scales: dict[str, tuple[tuple[float, float], float]] = {}
    for variable in variables:
        variable_scales[variable] = compute_variable_scales(
            variable=variable,
            scale_periods=scale_periods,
            profiles=profiles,
            absolute_low_percentile=args.absolute_low_percentile,
            absolute_high_percentile=args.absolute_high_percentile,
            anomaly_percentile=args.anomaly_percentile,
        )
        abs_limits_var, model_anom_scale_var = variable_scales[variable]
        print(
            f"{safe_name(variable)}: abs=[{abs_limits_var[0]:.3e}, {abs_limits_var[1]:.3e}], "
            f"model_anom=±{model_anom_scale_var:.3e}"
        )

    pair_scales: dict[str, tuple[tuple[float, float], float]] = {}
    for pair in selected_pairs:
        pair_id = f"{safe_name(pair[0])}_vs_{safe_name(pair[1])}"
        pair_scales[pair_id] = compute_pair_comparison_scales(
            pair=pair,
            scale_periods=scale_periods,
            profiles=profiles,
            absolute_low_percentile=args.absolute_low_percentile,
            absolute_high_percentile=args.absolute_high_percentile,
            anomaly_percentile=args.anomaly_percentile,
        )
        abs_limits, cross_anom_scale = pair_scales[pair_id]
        print(
            f"{pair_id}: shared_abs=[{abs_limits[0]:.3e}, {abs_limits[1]:.3e}], "
            f"cross_anom=±{cross_anom_scale:.3e}"
        )

    output_root = args.output_dir.resolve()
    saved = 0
    output_set = set(args.outputs)

    for pair in selected_pairs:
        normal_var, rad_var = pair
        pair_id = f"{safe_name(normal_var)}_vs_{safe_name(rad_var)}"
        pair_abs_limits, cross_anom_scale = pair_scales[pair_id]

        if "individual" in output_set:
            for variable in (normal_var, rad_var):
                var_abs_limits, var_anom_scale = variable_scales[variable]
                for period in periods:
                    output_file = (
                        output_root
                        / "individual"
                        / safe_name(variable)
                        / period.output_subdir
                        / f"{safe_name(variable)}_panel_c1m_g1m-c1m_g2m-g1m.png"
                    )
                    plot_individual_panels(
                        variable=variable,
                        control=profiles[(period.key, variable, "control")],
                        graupel=profiles[(period.key, variable, "graupel")],
                        twomom=profiles[(period.key, variable, "2mom")],
                        axis=axes_by_period[period.key],
                        max_height_km=args.max_height_km,
                        period_label=period.label,
                        freezing_line_km=freezing_by_period[period.key],
                        abs_limits=var_abs_limits,
                        anom_scale=var_anom_scale,
                        output_file=output_file,
                        dpi=args.dpi,
                    )
                    saved += 1

        if "pair" in output_set:
            if args.cross_anomaly_mode == "both":
                suffix = "panel_abs_rad-minus-normal_normal-minus-rad"
            elif args.cross_anomaly_mode == "normal-minus-rad":
                suffix = "normal-minus-rad"
            else:
                suffix = "rad-minus-normal"
            if args.pair_layout == "panel":
                suffix = f"panel_abs_{suffix}" if "panel_abs" not in suffix else suffix
            elif args.pair_layout == "single":
                suffix = f"single_{suffix}"
                for period in periods:
                    for exp in EXPERIMENTS:
                        output_file = (
                            output_root
                            / "pair_comparison"
                            / pair_id
                            / exp
                            / period.output_subdir
                            / f"{pair_id}_{exp}_{suffix}.png"
                        )
                        plot_pair_panel(
                            pair=pair,
                            experiment=exp,
                            normal=profiles[(period.key, normal_var, exp)],
                            rad=profiles[(period.key, rad_var, exp)],
                            axis=axes_by_period[period.key],
                            max_height_km=args.max_height_km,
                            period_label=period.label,
                            freezing_line_km=freezing_by_period[period.key],
                            abs_limits=pair_abs_limits,
                            cross_anom_scale=cross_anom_scale,
                            cross_anomaly_mode=args.cross_anomaly_mode,
                            pair_layout=args.pair_layout,
                            output_file=output_file,
                            dpi=args.dpi,
                        )
                        saved += 1
            else:
                suffix = f"experiments_{suffix}"
                for period in periods:
                    output_file = (
                        output_root
                        / "pair_comparison"
                        / pair_id
                        / period.output_subdir
                        / f"{pair_id}_{suffix}.png"
                    )
                    per_exp = {
                        exp: (
                            profiles[(period.key, normal_var, exp)],
                            profiles[(period.key, rad_var, exp)],
                        )
                        for exp in EXPERIMENTS
                    }
                    plot_pair_across_experiments(
                        pair=pair,
                        period_label=period.label,
                        axis=axes_by_period[period.key],
                        max_height_km=args.max_height_km,
                        freezing_line_km=freezing_by_period[period.key],
                        cross_anom_scale=cross_anom_scale,
                        cross_anomaly_mode=args.cross_anomaly_mode,
                        per_experiment_arrays=per_exp,
                        output_file=output_file,
                        dpi=args.dpi,
                    )
                    saved += 1

        if "sum_diff" in output_set:
            for period in periods:
                output_file = (
                    output_root
                    / "sum_diff"
                    / pair_id
                    / period.output_subdir
                    / f"{pair_id}_sum_diff_grid.png"
                )
                plot_sum_diff_grid(
                    pair=pair,
                    profiles=profiles,
                    period=period,
                    axis=axes_by_period[period.key],
                    max_height_km=args.max_height_km,
                    freezing_line_km=freezing_by_period[period.key],
                    output_file=output_file,
                    dpi=args.dpi,
                )
                saved += 1

    print(f"\nCompleted. Total figures generated: {saved}")


if __name__ == "__main__":
    main()
