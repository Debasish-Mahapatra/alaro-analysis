"""
Reusable plot builders for common ALARO visualisations.

- ``plot_surface_diurnal_cycle`` -- three-line diurnal cycle with optional zoom inset
- ``plot_three_panel_diurnal``  -- height-time pcolormesh (absolute + 2 anomaly panels)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.plotting.scales import robust_anomaly_scale, robust_log_limits

# Default experiment colours (consistent across all plots)
EXPERIMENT_COLORS: dict[str, str] = {
    "control": "#d62728",
    "graupel": "#1f77b4",
    "2mom": "#2ca02c",
    "2mom-xcu": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Surface (1-D line) diurnal cycle
# ---------------------------------------------------------------------------


def plot_surface_diurnal_cycle(
    line_data: dict[str, np.ndarray],
    output_file: Path,
    *,
    variable_label: str = "",
    variable_unit: str = "",
    period_label: str = "",
    utc_offset_hours: int = -4,
    zoom_inset: bool = False,
    colors: dict[str, str] | None = None,
    dpi: int = 450,
) -> None:
    """Plot a 24-hour diurnal cycle with one line per experiment.

    Parameters
    ----------
    line_data : dict[str, ndarray]
        Mapping from experiment name (``"control"``, ``"graupel"``, ``"2mom"``)
        to a 24-element array of mean values.
    output_file : Path
        Where to save the figure.
    variable_label, variable_unit : str
        For the y-axis label (``"label [unit]"``).
    period_label : str
        Title prefix (e.g. ``"Full 2-year period"``).
    utc_offset_hours : int
        Used in the x-axis label.
    zoom_inset : bool
        If *True*, add a zoomed inset around the daytime peak.
    colors : dict or None
        Override default experiment colours.
    dpi : int
        Figure resolution.
    """
    if colors is None:
        colors = EXPERIMENT_COLORS

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
                color=colors.get(exp, "gray"),
                label=EXPERIMENT_LABELS[exp],
            )
        ax.grid(alpha=0.25, linestyle="--")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if configure_main_axis:
            ax.set_xticks(np.arange(0, 24, 3))
            ax.set_xlim(0.0, 23.0)

    fig_width = 12.4 if zoom_inset else 10.5
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
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
                    peak_center = 0.5 * (x0 + x1)
                    zoom_vals = stacked[:, x0 : x1 + 1]
                    zoom_finite = zoom_vals[np.isfinite(zoom_vals)]
                    if zoom_finite.size > 0:
                        y0 = float(np.min(zoom_finite))
                        y1 = float(np.max(zoom_finite))
                        ypad = max(5.0, 0.14 * max(y1 - y0, 1.0))
                        inset_bounds = [1.02, 0.54, 0.34, 0.34]
                        if peak_center > 11.5:
                            inset_bounds = [-0.38, 0.54, 0.34, 0.34]
                        axins = ax.inset_axes(inset_bounds)
                        draw_lines(axins, configure_main_axis=False)
                        axins.set_xlim(float(x0), float(x1))
                        axins.set_ylim(y0 - ypad, y1 + ypad)
                        axins.set_xticks(np.arange(x0, x1 + 1, 1))
                        axins.yaxis.set_major_locator(
                            mticker.MaxNLocator(nbins=6)
                        )
                        axins.tick_params(labelsize=8)
                        ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=0.9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


# ---------------------------------------------------------------------------
# Three-panel height-time pcolormesh
# ---------------------------------------------------------------------------


def plot_three_panel_diurnal(
    control: np.ndarray,
    graupel: np.ndarray,
    twomom: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    *,
    variable_label: str = "",
    variable_unit: str = "",
    period_label: str = "",
    max_height_km: float = 20.0,
    abs_limits: tuple[float, float] | None = None,
    anom_scale: float | None = None,
    use_linear_abs: bool = True,
    abs_cmap: str = "turbo",
    anom_cmap: str = "RdBu_r",
    freezing_lines_km: dict[str, np.ndarray | None] | None = None,
    dpi: int = 450,
) -> None:
    """Three-panel diurnal-height plot: absolute + two anomaly panels.

    Panels:
    (a) C1M absolute
    (b) G1M - C1M anomaly
    (c) G2M - G1M anomaly
    """
    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    ctrl = control[order, :]
    diff_g1 = (graupel - control)[order, :]
    diff_g2 = (twomom - graupel)[order, :]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        ctrl = ctrl[keep, :]
        diff_g1 = diff_g1[keep, :]
        diff_g2 = diff_g2[keep, :]

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    ctrl_plot = np.ma.masked_invalid(ctrl)
    if not use_linear_abs:
        ctrl_plot = np.ma.masked_where(ctrl_plot <= 0, ctrl_plot)

    diff_g1_plot = np.ma.masked_invalid(diff_g1)
    diff_g2_plot = np.ma.masked_invalid(diff_g2)

    # Absolute colorscale
    if abs_limits is None:
        if use_linear_abs:
            valid = ctrl_plot.compressed()
            vmin = float(np.percentile(valid, 2)) if valid.size else 0.0
            vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
        else:
            vmin, vmax = robust_log_limits(ctrl_plot.filled(np.nan))
    else:
        vmin, vmax = abs_limits

    if use_linear_abs:
        abs_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        if vmax <= vmin:
            vmax = vmin * 10.0
        abs_norm = mcolors.LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)

    # Anomaly colorscale
    if anom_scale is None:
        anom_scale_val = robust_anomaly_scale(
            diff_g1_plot.filled(np.nan),
            diff_g2_plot.filled(np.nan),
        )
    else:
        anom_scale_val = anom_scale
    diff_norm = mcolors.TwoSlopeNorm(
        vmin=-anom_scale_val, vcenter=0.0, vmax=anom_scale_val
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.set_facecolor("#d3d3d3")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=16)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_ylabel(axis.label, fontsize=16)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)
        ax.text(
            0.02, 0.98, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, ha="left", va="top", fontsize=14,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

    # Panel (a): absolute
    pcm_abs = axes[0].pcolormesh(
        hour_edges, y_edges, ctrl_plot,
        cmap=abs_cmap, norm=abs_norm, shading="auto",
    )
    axes[0].set_title(
        f"{EXPERIMENT_LABELS['control']} ({variable_label}, Absolute)",
        fontsize=14, fontweight="bold",
    )

    # Panel (b): G1M - C1M
    axes[1].pcolormesh(
        hour_edges, y_edges, diff_g1_plot,
        cmap=anom_cmap, norm=diff_norm, shading="auto",
    )
    axes[1].set_title(
        f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
        fontsize=14, fontweight="bold",
    )

    # Panel (c): G2M - G1M
    pcm_diff = axes[2].pcolormesh(
        hour_edges, y_edges, diff_g2_plot,
        cmap=anom_cmap, norm=diff_norm, shading="auto",
    )
    axes[2].set_title(
        f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
        fontsize=14, fontweight="bold",
    )

    # Freezing level overlay
    if freezing_lines_km:
        hours_arr = np.arange(24, dtype=np.float64)
        panel_experiments = ("control", "graupel", "2mom")
        for idx, ax in enumerate(axes):
            exp = panel_experiments[idx]
            line = freezing_lines_km.get(exp)
            if line is None or not np.isfinite(line).any():
                continue
            ax.plot(
                hours_arr, line,
                color="black", linewidth=1.8, linestyle="--",
                label=f"Freezing level ({EXPERIMENT_LABELS[exp]})", zorder=10,
            )
            ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    fig.suptitle(
        f"{period_label} - {variable_label}", fontsize=16, fontweight="bold"
    )

    # Colorbars
    abs_label = f"Mean {variable_label}"
    if variable_unit:
        abs_label += f" [{variable_unit}]"
    cbar_abs = fig.colorbar(
        pcm_abs, ax=axes[0], orientation="horizontal", fraction=0.08, pad=0.16,
    )
    cbar_abs.set_label(abs_label, fontsize=14)
    cbar_abs.ax.tick_params(labelsize=12)

    diff_label = f"{variable_label} anomaly"
    if variable_unit:
        diff_label += f" [{variable_unit}]"
    cbar_diff = fig.colorbar(
        pcm_diff, ax=axes[1:], orientation="horizontal", fraction=0.08, pad=0.16,
    )
    cbar_diff.set_label(diff_label, fontsize=14)
    cbar_diff.ax.tick_params(labelsize=12)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
