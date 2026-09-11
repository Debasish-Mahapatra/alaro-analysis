"""Pair analysis."""

from __future__ import annotations

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS
from alaro_analysis.common.models import PeriodSpec, VerticalAxis

try:
    import cmaps
except ImportError:
    cmaps = None

ABS_CMAP = cmaps.WhiteBlueGreenYellowRed if cmaps is not None else "turbo"
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.vertical import centers_to_edges


def variable_label(name: str) -> str:
    return name


def crop_to_axis(
    axis: VerticalAxis,
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
    axis: VerticalAxis,
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
    axes[0].set_title(
        f"{EXPERIMENT_LABELS['control']} ({variable_label(variable)}, Absolute)",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].pcolormesh(
        hour_edges, y_edges, diff_g1_plot, cmap="RdBu_r", norm=diff_norm, shading="auto"
    )
    axes[1].set_title(
        f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
        fontsize=14,
        fontweight="bold",
    )

    pcm_diff = axes[2].pcolormesh(
        hour_edges, y_edges, diff_g2_plot, cmap="RdBu_r", norm=diff_norm, shading="auto"
    )
    axes[2].set_title(
        f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
        fontsize=14,
        fontweight="bold",
    )

    add_freezing_line(list(axes), freezing_line_km)
    fig.suptitle(f"{period_label} - {variable_label(variable)}", fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16)
    cbar_abs.set_label(f"Mean {variable_label(variable)} (Absolute)", fontsize=14)
    cbar_abs.ax.tick_params(labelsize=12)

    cbar_diff = fig.colorbar(
        pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16
    )
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
    axis: VerticalAxis,
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

        pcm_diff = ax.pcolormesh(
            hour_edges, y_edges, diff_arr, cmap="RdBu_r", norm=anom_norm, shading="auto"
        )
        ax.set_title(diff_title, fontsize=14, fontweight="bold")
        add_freezing_line([ax], freezing_line_km)
        fig.suptitle(
            f"{period_label} - {diff_label} ({EXPERIMENT_LABELS[experiment]})",
            fontsize=16,
            fontweight="bold",
        )

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

        pcm_abs_1 = axes[0, 0].pcolormesh(
            hour_edges, y_edges, normal_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto"
        )
        axes[0, 0].set_title(
            f"{EXPERIMENT_LABELS[experiment]} {n_label} (Absolute)", fontsize=14, fontweight="bold"
        )
        axes[0, 1].pcolormesh(
            hour_edges, y_edges, rad_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto"
        )
        axes[0, 1].set_title(
            f"{EXPERIMENT_LABELS[experiment]} {r_label} (Absolute)", fontsize=14, fontweight="bold"
        )

        pcm_diff = axes[1, 0].pcolormesh(
            hour_edges, y_edges, rmn_plot, cmap="RdBu_r", norm=anom_norm, shading="auto"
        )
        axes[1, 0].set_title(
            f"{EXPERIMENT_LABELS[experiment]} ({r_label} - {n_label})",
            fontsize=14,
            fontweight="bold",
        )
        axes[1, 1].pcolormesh(
            hour_edges, y_edges, nmr_plot, cmap="RdBu_r", norm=anom_norm, shading="auto"
        )
        axes[1, 1].set_title(
            f"{EXPERIMENT_LABELS[experiment]} ({n_label} - {r_label})",
            fontsize=14,
            fontweight="bold",
        )

        add_freezing_line(flat_axes, freezing_line_km)
        fig.suptitle(
            f"{period_label} - {n_label} vs {r_label} ({EXPERIMENT_LABELS[experiment]})",
            fontsize=16,
            fontweight="bold",
        )

        cbar_abs = fig.colorbar(
            pcm_abs_1,
            ax=[axes[0, 0], axes[0, 1]],
            orientation="horizontal",
            fraction=0.08,
            pad=0.10,
        )
        cbar_abs.set_label(f"Mean absolute ({n_label}, {r_label})", fontsize=14)
        cbar_abs.ax.tick_params(labelsize=12)

        cbar_diff = fig.colorbar(
            pcm_diff, ax=[axes[1, 0], axes[1, 1]], orientation="horizontal", fraction=0.08, pad=0.12
        )
        cbar_diff.set_label(f"Pair anomaly ({r_label}-{n_label} and reverse)", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
        for idx, ax in enumerate(axes):
            style_axes(ax, axis.label, 16, 14)
            if axis.is_height_km:
                ax.set_ylim(0.0, max_height_km)
            add_panel_tag(ax, idx, 14)

        pcm_abs_1 = axes[0].pcolormesh(
            hour_edges, y_edges, normal_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto"
        )
        axes[0].set_title(
            f"{EXPERIMENT_LABELS[experiment]} {n_label} (Absolute)", fontsize=14, fontweight="bold"
        )
        axes[1].pcolormesh(
            hour_edges, y_edges, rad_plot, cmap=ABS_CMAP, norm=abs_norm, shading="auto"
        )
        axes[1].set_title(
            f"{EXPERIMENT_LABELS[experiment]} {r_label} (Absolute)", fontsize=14, fontweight="bold"
        )

        if cross_anomaly_mode == "normal-minus-rad":
            diff_arr = nmr_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({n_label} - {r_label})"
            diff_label = f"{n_label} - {r_label}"
        else:
            diff_arr = rmn_plot
            diff_title = f"{EXPERIMENT_LABELS[experiment]} ({r_label} - {n_label})"
            diff_label = f"{r_label} - {n_label}"
        pcm_diff = axes[2].pcolormesh(
            hour_edges, y_edges, diff_arr, cmap="RdBu_r", norm=anom_norm, shading="auto"
        )
        axes[2].set_title(diff_title, fontsize=14, fontweight="bold")

        add_freezing_line(list(axes), freezing_line_km)
        fig.suptitle(
            f"{period_label} - {n_label} vs {r_label} ({EXPERIMENT_LABELS[experiment]})",
            fontsize=16,
            fontweight="bold",
        )

        cbar_abs = fig.colorbar(
            pcm_abs_1, ax=axes[:2], orientation="horizontal", fraction=0.08, pad=0.16
        )
        cbar_abs.set_label(f"Mean absolute ({n_label}, {r_label})", fontsize=14)
        cbar_abs.ax.tick_params(labelsize=12)

        cbar_diff = fig.colorbar(
            pcm_diff, ax=axes[2], orientation="horizontal", fraction=0.08, pad=0.16
        )
        cbar_diff.set_label(f"Pair anomaly ({diff_label})", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_pair_across_experiments(
    pair: tuple[str, str],
    period_label: str,
    axis: VerticalAxis,
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

    diff_norm = mcolors.TwoSlopeNorm(vmin=-cross_anom_scale, vcenter=0.0, vmax=cross_anom_scale)

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

    cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.16)
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
    axis: VerticalAxis,
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
                hour_edges,
                y_edges,
                plot_data,
                cmap=cmap,
                norm=abs_norm,
                shading="auto",
            )
            style_axes(ax, axis.label, 14, 12)
            if axis.is_height_km:
                ax.set_ylim(0.0, max_height_km)
            add_panel_tag(ax, panel_idx, 13)
            ax.set_title(
                f"{EXPERIMENT_LABELS[exp]} - {col_titles[col_idx]}",
                fontsize=12,
                fontweight="bold",
            )

            # freezing line
            if freezing_line_km is not None and np.isfinite(freezing_line_km).any():
                hours = np.arange(24, dtype=np.float64)
                lbl = "Freezing level" if panel_idx == 0 else None
                ax.plot(
                    hours,
                    freezing_line_km,
                    color="black",
                    linewidth=1.5,
                    linestyle="--",
                    label=lbl,
                    zorder=10,
                )
                if lbl:
                    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
            panel_idx += 1

    fig.suptitle(
        f"{period.label} - {n_label} / {r_label} sum & difference",
        fontsize=15,
        fontweight="bold",
    )

    cbar = fig.colorbar(
        pcm,
        ax=axes,
        orientation="horizontal",
        fraction=0.04,
        pad=0.06,
    )
    cbar.set_label("Absolute (linear)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
