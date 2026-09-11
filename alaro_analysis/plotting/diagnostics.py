"""Diagnostics. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS
from alaro_analysis.common.figio import strip_cbar_zeros
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.plotting.scales import infer_abs_limits, infer_anom_scale


def plot_three_panel_anomaly(
    title_label: str,
    unit: str,
    control: np.ndarray,
    graupel: np.ndarray,
    twomom: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    period_label: str,
    max_height_km: float,
    abs_linear: bool = True,
    use_abs_control: bool = True,
    freezing_lines: dict[str, np.ndarray] | None = None,
) -> None:
    import cmaps

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    c = control[order, :]
    g = graupel[order, :]
    t = twomom[order, :]

    d1 = g - c
    d2 = t - g

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        c = c[keep, :]
        g = g[keep, :]
        t = t[keep, :]
        d1 = d1[keep, :]
        d2 = d2[keep, :]

    abs_field = c if use_abs_control else g
    vmin_abs, vmax_abs = infer_abs_limits(abs_field, linear=abs_linear)
    anom_scale = infer_anom_scale(d1, d2)

    if abs_linear:
        abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    else:
        abs_norm = mcolors.LogNorm(vmin=max(vmin_abs, 1e-12), vmax=max(vmax_abs, vmin_abs * 10.0))
    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)
    c_plot = np.ma.masked_invalid(abs_field)
    d1_plot = np.ma.masked_invalid(d1)
    d2_plot = np.ma.masked_invalid(d2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=14)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylabel(axis.label, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        ax.text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )
        if freezing_lines is not None:
            exp_key = EXPERIMENTS[idx] if idx < len(EXPERIMENTS) else None
            if exp_key and exp_key in freezing_lines:
                fl = freezing_lines[exp_key]
                if fl is not None and np.isfinite(fl).any():
                    hours = np.arange(24, dtype=np.float64)
                    axes[idx].plot(
                        hours,
                        fl,
                        color="black",
                        linewidth=1.8,
                        linestyle="--",
                        label=f"Freezing level ({EXPERIMENT_LABELS[exp_key]})",
                        zorder=10,
                    )
                    if idx == 0:
                        axes[idx].legend(loc="upper right", fontsize=12, framealpha=0.9)

    pcm_abs = axes[0].pcolormesh(
        hour_edges,
        y_edges,
        c_plot,
        cmap=cmaps.WhiteBlueGreenYellowRed,
        norm=abs_norm,
        shading="auto",
    )
    axes[0].set_title(
        f"{EXPERIMENT_LABELS['control']} ({title_label}, Absolute)", fontsize=12, fontweight="bold"
    )

    axes[1].pcolormesh(hour_edges, y_edges, d1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto")
    axes[1].set_title(
        f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
        fontsize=12,
        fontweight="bold",
    )

    pcm_diff = axes[2].pcolormesh(
        hour_edges, y_edges, d2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto"
    )
    axes[2].set_title(
        f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
        fontsize=12,
        fontweight="bold",
    )

    fig.suptitle(f"{period_label} - {title_label}", fontsize=15, fontweight="bold")
    unit_tag = f" [{unit}]" if unit else ""

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    strip_cbar_zeros(cbar_abs, axis="x")
    cbar_abs.set_label(f"Mean {title_label}{unit_tag}", fontsize=12)
    cbar_abs.ax.tick_params(labelsize=11)

    cbar_diff = fig.colorbar(
        pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16
    )
    strip_cbar_zeros(cbar_diff, axis="x")
    cbar_diff.set_label(f"{title_label} anomaly{unit_tag}", fontsize=12)
    cbar_diff.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_three_experiment_absolute(
    title_label: str,
    unit: str,
    profiles: dict[str, np.ndarray],
    axis: VerticalAxis,
    output_file: Path,
    period_label: str,
    max_height_km: float,
    diverging: bool = False,
    freezing_lines: dict[str, np.ndarray] | None = None,
) -> None:
    import cmaps

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    p = {exp: profiles[exp][order, :] for exp in EXPERIMENTS}
    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        p = {exp: arr[keep, :] for exp, arr in p.items()}

    chunks = [arr[np.isfinite(arr)] for arr in p.values() if np.isfinite(arr).any()]
    if not chunks:
        vmin, vmax = -1.0, 1.0
    else:
        all_vals = np.concatenate(chunks)
        if diverging:
            scale = float(np.percentile(np.abs(all_vals), 98))
            if scale <= 0:
                scale = 1.0
            vmin, vmax = -scale, scale
        else:
            vmin = float(np.percentile(all_vals, 2))
            vmax = float(np.percentile(all_vals, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6

    if diverging:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmaps.WhiteBlueGreenYellowRed

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    pcm = None
    for idx, exp in enumerate(EXPERIMENTS):
        arr = np.ma.masked_invalid(p[exp])
        pcm = axes[idx].pcolormesh(hour_edges, y_edges, arr, cmap=cmap, norm=norm, shading="auto")
        axes[idx].set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        axes[idx].set_facecolor("#d3d3d3")
        axes[idx].set_xlabel("Hour (Amazon UTC-4)", fontsize=14)
        axes[idx].set_xticks(np.arange(0, 24, 6))
        axes[idx].set_xlim(-0.5, 23.5)
        axes[idx].set_ylabel(axis.label, fontsize=14)
        axes[idx].tick_params(axis="both", labelsize=12)
        if axis.is_height_km:
            axes[idx].set_ylim(0.0, max_height_km)

        if freezing_lines is not None and exp in freezing_lines:
            fl = freezing_lines[exp]
            if fl is not None and np.isfinite(fl).any():
                hours = np.arange(24, dtype=np.float64)
                axes[idx].plot(
                    hours,
                    fl,
                    color="black",
                    linewidth=1.8,
                    linestyle="--",
                    label=f"Freezing level ({EXPERIMENT_LABELS[exp]})",
                    zorder=10,
                )
                if idx == 0:
                    axes[idx].legend(loc="upper right", fontsize=12, framealpha=0.9)

        axes[idx].text(
            0.02,
            0.98,
            f"({chr(ord('a') + idx)})",
            transform=axes[idx].transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    fig.suptitle(f"{period_label} - {title_label}", fontsize=15, fontweight="bold")
    unit_tag = f" [{unit}]" if unit else ""
    cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.16)
    strip_cbar_zeros(cbar, axis="x")
    cbar.set_label(f"{title_label}{unit_tag}", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_lines_stacked(
    title: str,
    period_label: str,
    diag_order: Sequence[str],
    diag_labels: dict[str, str],
    unit_map: dict[str, str],
    line_data: dict[str, dict[str, np.ndarray]],
    output_file: Path,
) -> None:
    n = len(diag_order)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    hours = np.arange(24, dtype=np.float64)

    for idx, diag in enumerate(diag_order):
        ax = axes[idx]
        for exp in EXPERIMENTS:
            arr = line_data.get(diag, {}).get(exp)
            if arr is None:
                continue
            ax.plot(hours, arr, linewidth=2.0, label=EXPERIMENT_LABELS[exp])
        ylabel = diag_labels.get(diag, diag)
        unit = unit_map.get(diag, "")
        if unit:
            ylabel = f"{ylabel} [{unit}]"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(axis="both", labelsize=10)
        ax.text(
            0.01,
            0.92,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        )

    axes[-1].set_xlabel("Hour (Amazon UTC-4)", fontsize=12)
    axes[-1].set_xticks(np.arange(0, 24, 3))
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=3)
    fig.suptitle(f"{period_label} - {title}", fontsize=14, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
