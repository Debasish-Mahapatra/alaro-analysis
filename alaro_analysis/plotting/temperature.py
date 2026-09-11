"""Temperature. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.figio import strip_cbar_zeros
from alaro_analysis.common.models import PeriodSpec, VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges


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
    import cmaps

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
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=panel_tag_fs,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    pcm_abs = axes[0].pcolormesh(
        hour_edges,
        y_edges,
        ctrl_plot,
        cmap=cmaps.WhiteBlueGreenYellowRed,
        norm=abs_norm,
        shading="auto",
    )
    axes[0].set_title(
        f"{EXPERIMENT_LABELS['control']} ({variable}, Absolute)", fontsize=14, fontweight="bold"
    )
    axes[1].pcolormesh(hour_edges, y_edges, d1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[1].set_title(
        f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
        fontsize=14,
        fontweight="bold",
    )
    pcm_diff = axes[2].pcolormesh(
        hour_edges, y_edges, d2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto"
    )
    axes[2].set_title(
        f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
        fontsize=14,
        fontweight="bold",
    )

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
        axes[0].legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    fig.suptitle(f"{period_label} - {variable}", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    strip_cbar_zeros(cbar_abs, axis="x")
    cbar_abs.set_label(f"Mean {variable} (Absolute)", fontsize=cbar_label_fs)
    cbar_abs.ax.tick_params(labelsize=cbar_tick_fs)

    cbar_diff = fig.colorbar(
        pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16
    )
    strip_cbar_zeros(cbar_diff, axis="x")
    cbar_diff.set_label(f"{variable} anomaly", fontsize=cbar_label_fs)
    cbar_diff.ax.tick_params(labelsize=cbar_tick_fs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
