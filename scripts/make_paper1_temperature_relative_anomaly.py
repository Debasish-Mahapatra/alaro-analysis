#!/usr/bin/env python3
"""Paper1 full-period temperature panel with relative anomaly panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from alaro_analysis.common.figio import strip_cbar_zeros
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges, compute_freezing_line_km
from alaro_analysis.data.cache import (
    cache_relpath,
    find_cache_file,
    find_existing_cache,
    height_relpaths,
    load_diurnal_profile_cache,
    load_height_profile_cache,
)

try:
    import cmaps  # type: ignore
except Exception:  # pragma: no cover - optional plotting dependency
    cmaps = None


PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")
OUTPUT_DIR = PAPER_ROOT / "09_full_two_year_temperature_relative_anomaly"
PROCESSED_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
PERIOD_SUBDIR = Path("2years")
VARIABLE = "TEMPERATURE"
SPATIAL_TAG = "full-domain"
FIGURE_NAME = "full_two_year_temperature_relative_anomaly_450dpi.png"
TEXT_NAME = "full_two_year_temperature_relative_anomaly_data.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper1 Figure 9: full two-year temperature relative anomalies."
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-height-km", type=float, default=20.0)
    parser.add_argument("--absolute-low-percentile", type=float, default=2.0)
    parser.add_argument("--absolute-high-percentile", type=float, default=98.0)
    parser.add_argument("--relative-anomaly-percentile", type=float, default=98.0)
    parser.add_argument("--relative-scale", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def cache_roots(processed_root: Path) -> list[Path]:
    return [processed_root, processed_root / "data"]


def load_temperature_mean(
    roots: list[Path],
    experiment: str,
) -> tuple[np.ndarray, np.ndarray | None, int | None, Path | None, Path]:
    relpaths = [
        cache_relpath(VARIABLE, PERIOD_SUBDIR, experiment, SPATIAL_TAG),
        cache_relpath(VARIABLE, PERIOD_SUBDIR, experiment, None),
    ]
    for relpath in relpaths:
        path = find_cache_file(roots, relpath)
        if path is not None:
            mean, counts, n_files, sample_file = load_diurnal_profile_cache(path)
            return mean, counts, n_files, sample_file, path
    raise FileNotFoundError(
        f"Temperature cache not found for {experiment}. Searched roots: {roots}"
    )


def load_height_axis(roots: list[Path]) -> tuple[VerticalAxis, Path]:
    relpaths = [
        *height_relpaths(PERIOD_SUBDIR, "control", "first", SPATIAL_TAG),
        *height_relpaths(PERIOD_SUBDIR, "control", "first", None),
    ]
    path = find_existing_cache(roots, relpaths)
    if path is None:
        raise FileNotFoundError(f"Height cache not found. Searched roots: {roots}")
    height_m = load_height_profile_cache(path)
    axis = VerticalAxis(
        values=np.asarray(height_m, dtype=np.float64) / 1000.0,
        label="Height (km)",
        is_height_km=True,
    )
    return axis, path


def align_vertical_shapes(
    axis: VerticalAxis,
    profiles: dict[str, np.ndarray],
    counts: dict[str, np.ndarray | None],
) -> tuple[VerticalAxis, dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    n_levels = min(axis.values.size, *(arr.shape[0] for arr in profiles.values()))
    aligned_axis = VerticalAxis(
        values=axis.values[:n_levels],
        label=axis.label,
        is_height_km=axis.is_height_km,
    )
    aligned_profiles = {exp: arr[:n_levels, :] for exp, arr in profiles.items()}
    aligned_counts = {
        exp: None if arr is None else arr[:n_levels, :] for exp, arr in counts.items()
    }
    return aligned_axis, aligned_profiles, aligned_counts


def relative_anomaly_pct(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    out = np.full(candidate.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(candidate) & np.isfinite(baseline) & (baseline != 0.0)
    out[valid] = ((candidate[valid] - baseline[valid]) / baseline[valid]) * 100.0
    return out


def robust_symmetric_scale(*arrays: np.ndarray, percentile: float) -> float:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        finite = np.abs(np.asarray(arr, dtype=np.float64))
        finite = finite[np.isfinite(finite)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return 1.0
    merged = np.concatenate(chunks)
    scale = float(np.percentile(merged, percentile))
    if scale <= 0.0:
        scale = float(np.max(merged))
    if scale <= 0.0:
        scale = 1.0
    return scale


def prepare_plot_arrays(
    axis: VerticalAxis,
    profiles: dict[str, np.ndarray],
    max_height_km: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    arrays = {exp: profiles[exp][order, :] for exp in EXPERIMENTS}

    keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
    y = y[keep]
    arrays = {exp: arr[keep, :] for exp, arr in arrays.items()}
    return y, arrays


def plot_temperature_relative_anomaly(
    axis: VerticalAxis,
    profiles: dict[str, np.ndarray],
    output_file: Path,
    *,
    max_height_km: float,
    abs_limits: tuple[float, float],
    relative_scale: float,
    freezing_line_km: np.ndarray | None,
    dpi: int,
) -> None:
    y, arrays = prepare_plot_arrays(axis, profiles, max_height_km)
    control = arrays["control"]
    graupel = arrays["graupel"]
    twomom = arrays["2mom"]

    rel_g1_c1 = relative_anomaly_pct(graupel, control)
    rel_g2_g1 = relative_anomaly_pct(twomom, graupel)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    ctrl_plot = np.ma.masked_invalid(control)
    g1_plot = np.ma.masked_invalid(rel_g1_c1)
    g2_plot = np.ma.masked_invalid(rel_g2_g1)

    vmin_abs, vmax_abs = abs_limits
    if vmax_abs <= vmin_abs:
        vmax_abs = vmin_abs + 1.0
    abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    if relative_scale <= 0.0:
        relative_scale = 1.0
    diff_norm = mcolors.TwoSlopeNorm(
        vmin=-relative_scale,
        vcenter=0.0,
        vmax=relative_scale,
    )
    abs_cmap = cmaps.WhiteBlueGreenYellowRed if cmaps is not None else "turbo"

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=16)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylabel(axis.label, fontsize=16)
        ax.set_ylim(0.0, max_height_km)
        ax.tick_params(axis="both", labelsize=14)
        ax.text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
                "pad": 2.0,
            },
        )

    pcm_abs = axes[0].pcolormesh(
        hour_edges,
        y_edges,
        ctrl_plot,
        cmap=abs_cmap,
        norm=abs_norm,
        shading="auto",
    )
    axes[0].set_title(f"{EXPERIMENT_LABELS['control']} mean temperature", fontsize=14, fontweight="bold")

    axes[1].pcolormesh(
        hour_edges,
        y_edges,
        g1_plot,
        cmap="RdBu_r",
        norm=diff_norm,
        shading="auto",
    )
    axes[1].set_title("(G1M-C1M) / C1M", fontsize=14, fontweight="bold")

    pcm_diff = axes[2].pcolormesh(
        hour_edges,
        y_edges,
        g2_plot,
        cmap="RdBu_r",
        norm=diff_norm,
        shading="auto",
    )
    axes[2].set_title("(G2M-G1M) / G1M", fontsize=14, fontweight="bold")

    if freezing_line_km is not None and np.isfinite(freezing_line_km).any():
        hours = np.arange(24, dtype=np.float64)
        for idx, ax in enumerate(axes):
            label = "Freezing level" if idx == 0 else None
            ax.plot(
                hours,
                freezing_line_km,
                color="black",
                linewidth=1.8,
                linestyle="--",
                label=label,
                zorder=10,
            )
        axes[0].legend(loc="upper right", fontsize=12, framealpha=0.9)

    fig.suptitle("Full 2-year period - Temperature", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(
        pcm_abs,
        ax=axes[0],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    cbar_abs.set_label("Mean TEMPERATURE [K]", fontsize=14)
    cbar_abs.ax.tick_params(labelsize=12)
    strip_cbar_zeros(cbar_abs, axis="x")

    cbar_diff = fig.colorbar(
        pcm_diff,
        ax=axes[1:],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    cbar_diff.set_label("Relative temperature anomaly [%]", fontsize=14)
    cbar_diff.ax.tick_params(labelsize=12)
    strip_cbar_zeros(cbar_diff, axis="x")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_data_text(
    path: Path,
    axis: VerticalAxis,
    profiles: dict[str, np.ndarray],
    counts: dict[str, np.ndarray | None],
    sources: dict[str, Path],
    height_source: Path,
    freezing_line_km: np.ndarray | None,
    *,
    max_height_km: float,
    abs_limits: tuple[float, float],
    relative_scale: float,
) -> None:
    y, arrays = prepare_plot_arrays(axis, profiles, max_height_km)
    rel_g1_c1 = relative_anomaly_pct(arrays["graupel"], arrays["control"])
    rel_g2_g1 = relative_anomaly_pct(arrays["2mom"], arrays["graupel"])

    count_arrays: dict[str, np.ndarray | None] = {}
    for exp, arr in counts.items():
        if arr is None:
            count_arrays[exp] = None
            continue
        _, prepared = prepare_plot_arrays(axis, {name: arr if name == exp else profiles[name] for name in EXPERIMENTS}, max_height_km)
        count_arrays[exp] = prepared[exp]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Paper1 Figure 9 data: full two-year temperature relative anomaly panel\n")
        f.write("Panel a: C1M mean TEMPERATURE in K.\n")
        f.write("Panel b formula: ((G1M - C1M) / C1M) * 100, in percent.\n")
        f.write("Panel c formula: ((G2M - G1M) / G1M) * 100, in percent.\n")
        f.write(f"Height source: {height_source}\n")
        for exp in EXPERIMENTS:
            f.write(f"{EXPERIMENT_LABELS[exp]} source: {sources[exp]}\n")
        f.write(f"Absolute color limits K: {abs_limits[0]:.12g}, {abs_limits[1]:.12g}\n")
        f.write(f"Relative anomaly symmetric color limit percent: {relative_scale:.12g}\n")
        f.write("\nFreezing level by local hour\n")
        f.write("hour,freezing_level_km\n")
        for hour in range(24):
            val = np.nan if freezing_line_km is None else freezing_line_km[hour]
            f.write(f"{hour},{val:.12g}\n")
        f.write("\nPanel grid data\n")
        f.write(
            "height_km,hour,c1m_temperature_k,g1m_temperature_k,g2m_temperature_k,"
            "g1m_minus_c1m_relative_pct,g2m_minus_g1m_relative_pct,"
            "c1m_count,g1m_count,g2m_count\n"
        )
        for iz, height in enumerate(y):
            for hour in range(24):
                c1 = arrays["control"][iz, hour]
                g1 = arrays["graupel"][iz, hour]
                g2 = arrays["2mom"][iz, hour]
                d1 = rel_g1_c1[iz, hour]
                d2 = rel_g2_g1[iz, hour]
                c1_count = count_arrays["control"]
                g1_count = count_arrays["graupel"]
                g2_count = count_arrays["2mom"]
                f.write(
                    f"{height:.12g},{hour},"
                    f"{c1:.12g},{g1:.12g},{g2:.12g},"
                    f"{d1:.12g},{d2:.12g},"
                    f"{'' if c1_count is None else int(c1_count[iz, hour])},"
                    f"{'' if g1_count is None else int(g1_count[iz, hour])},"
                    f"{'' if g2_count is None else int(g2_count[iz, hour])}\n"
                )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    roots = cache_roots(args.processed_root.resolve())
    profiles: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray | None] = {}
    sources: dict[str, Path] = {}
    for exp in EXPERIMENTS:
        mean, exp_counts, _, _, source = load_temperature_mean(roots, exp)
        profiles[exp] = mean
        counts[exp] = exp_counts
        sources[exp] = source

    axis, height_source = load_height_axis(roots)
    axis, profiles, counts = align_vertical_shapes(axis, profiles, counts)

    freezing_line_km = compute_freezing_line_km(
        axis=axis,
        temperature_profiles=[profiles[exp] for exp in EXPERIMENTS],
    )

    _, plot_profiles = prepare_plot_arrays(axis, profiles, args.max_height_km)
    control = plot_profiles["control"]
    rel_g1_c1 = relative_anomaly_pct(plot_profiles["graupel"], control)
    rel_g2_g1 = relative_anomaly_pct(plot_profiles["2mom"], plot_profiles["graupel"])

    abs_valid = control[np.isfinite(control)]
    if abs_valid.size:
        abs_limits = (
            float(np.percentile(abs_valid, args.absolute_low_percentile)),
            float(np.percentile(abs_valid, args.absolute_high_percentile)),
        )
    else:
        abs_limits = (250.0, 320.0)

    relative_scale = args.relative_scale
    if relative_scale is None:
        relative_scale = robust_symmetric_scale(
            rel_g1_c1,
            rel_g2_g1,
            percentile=args.relative_anomaly_percentile,
        )

    output_dir = args.output_dir.resolve()
    figure_path = output_dir / FIGURE_NAME
    text_path = output_dir / TEXT_NAME
    plot_temperature_relative_anomaly(
        axis,
        profiles,
        figure_path,
        max_height_km=args.max_height_km,
        abs_limits=abs_limits,
        relative_scale=relative_scale,
        freezing_line_km=freezing_line_km,
        dpi=args.dpi,
    )
    write_data_text(
        text_path,
        axis,
        profiles,
        counts,
        sources,
        height_source,
        freezing_line_km,
        max_height_km=args.max_height_km,
        abs_limits=abs_limits,
        relative_scale=relative_scale,
    )
    print(f"[saved] {figure_path}")
    print(f"[saved] {text_path}")
    print(
        f"[scale] absolute K = {abs_limits[0]:.3f}..{abs_limits[1]:.3f}; "
        f"relative anomaly = +/-{relative_scale:.4g}%"
    )


if __name__ == "__main__":
    main()
