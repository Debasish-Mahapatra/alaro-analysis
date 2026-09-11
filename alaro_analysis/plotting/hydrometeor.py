"""Hydrometeor. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.models import PeriodSpec, VerticalAxis
from alaro_analysis.common.variables import (
    is_unit_interval_variable,
    uses_linear_absolute_scale,
    variable_key,
    variable_label,
    variable_unit,
)
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.data.profile_sources import is_updraft_derived_variable
from alaro_analysis.plotting.scales import robust_anomaly_scale, robust_log_limits

DEFAULT_SHARED_SCALE_GROUPS = (
    ("LIQUID_WATER", "RAD_LIQUID_W"),
    ("SOLID_WATER", "RAD_SOLID_WA"),
)


def build_scale_group_map(
    plot_variables: list[str], disable_default_groups: bool
) -> dict[str, str]:
    group_map = {var: f"var:{variable_key(var)}" for var in plot_variables}
    if disable_default_groups:
        return group_map

    key_to_var = {variable_key(var): var for var in plot_variables}
    for idx, members in enumerate(DEFAULT_SHARED_SCALE_GROUPS):
        present = [key_to_var[m] for m in members if m in key_to_var]
        if len(present) < 2:
            continue
        group_id = f"default-group:{idx}"
        for var in present:
            group_map[var] = group_id
    return group_map


def compute_global_plot_scales(
    plot_variables: list[str],
    periods: list[PeriodSpec],
    results: dict[tuple[str, str, str], tuple[np.ndarray, Path]],
    group_map: dict[str, str],
    anomaly_percentile: float,
    absolute_low_percentile: float,
    absolute_high_percentile: float,
) -> dict[str, dict[str, object]]:
    scales: dict[str, dict[str, object]] = {}
    group_ids = sorted(set(group_map.values()))

    for group_id in group_ids:
        abs_chunks: list[np.ndarray] = []
        anom_chunks: list[np.ndarray] = []

        group_vars = [v for v in plot_variables if group_map[v] == group_id]
        abs_linear = all(uses_linear_absolute_scale(v) for v in group_vars)
        group_is_unit_interval = all(is_unit_interval_variable(v) for v in group_vars)
        for variable in group_vars:
            var_key = variable_key(variable)
            for period in periods:
                k_ctrl = (period.key, variable, "control")
                k_g1 = (period.key, variable, "graupel")
                k_g2 = (period.key, variable, "2mom")
                if k_ctrl not in results or k_g1 not in results or k_g2 not in results:
                    continue

                ctrl = results[k_ctrl][0]
                g1 = results[k_g1][0]
                g2 = results[k_g2][0]

                abs_field = g1 if var_key == "GRAUPEL" else ctrl
                if abs_linear:
                    abs_valid = abs_field[np.isfinite(abs_field)]
                else:
                    abs_valid = abs_field[np.isfinite(abs_field) & (abs_field > 0)]
                if abs_valid.size > 0:
                    abs_chunks.append(abs_valid)

                diff_g1 = g1 - ctrl
                diff_g2 = g2 - g1
                if var_key == "GRAUPEL":
                    diff_valid = diff_g2[np.isfinite(diff_g2)]
                    if diff_valid.size > 0:
                        anom_chunks.append(diff_valid)
                else:
                    d1 = diff_g1[np.isfinite(diff_g1)]
                    d2 = diff_g2[np.isfinite(diff_g2)]
                    if d1.size > 0:
                        anom_chunks.append(d1)
                    if d2.size > 0:
                        anom_chunks.append(d2)

        if abs_chunks:
            abs_all = np.concatenate(abs_chunks)
            vmin = float(np.percentile(abs_all, absolute_low_percentile))
            vmax = float(np.percentile(abs_all, absolute_high_percentile))
            if abs_linear:
                if group_is_unit_interval:
                    vmin = max(0.0, vmin)
                    vmax = min(1.0, vmax)
                if vmax <= vmin:
                    vmax = min(1.0, vmin + 1e-6)
            else:
                vmin = max(vmin, float(np.min(abs_all)))
                if vmin <= 0:
                    pos = abs_all[abs_all > 0]
                    vmin = float(np.min(pos)) if pos.size > 0 else 1e-12
                if vmax <= vmin:
                    vmax = vmin * 10.0
            abs_limits = (vmin, vmax)
        else:
            abs_limits = (1e-12, 1.0)

        if anom_chunks:
            anom_all = np.concatenate(anom_chunks)
            anom_abs = np.abs(anom_all[np.isfinite(anom_all)])
            if anom_abs.size > 0:
                scale = float(np.percentile(anom_abs, anomaly_percentile))
                if scale <= 0:
                    scale = float(np.max(anom_abs))
                if scale <= 0:
                    scale = 1.0
            else:
                scale = 1.0
        else:
            scale = 1.0

        scales[group_id] = {
            "abs_limits": abs_limits,
            "anom_scale": scale,
            "abs_linear": abs_linear,
        }

    return scales


def plot_three_panels(
    variable: str,
    control_mean: np.ndarray,
    graupel_mean: np.ndarray,
    twomom_mean: np.ndarray,
    axis: VerticalAxis,
    output_file: Path,
    max_height_km: float,
    period_label: str,
    freezing_lines_km: dict[str, np.ndarray | None] | None,
    fixed_abs_limits: tuple[float, float] | None = None,
    fixed_anom_scale: float | None = None,
    fixed_abs_linear: bool | None = None,
) -> None:
    import cmaps

    axis_label_fs = 16
    tick_label_fs = 14
    legend_fs = 12
    panel_tag_fs = 14
    cbar_label_fs = 14
    cbar_tick_fs = 12
    var_label = variable_label(variable)
    unit = variable_unit(variable)
    is_updraft = is_updraft_derived_variable(variable)
    abs_linear = (
        uses_linear_absolute_scale(variable) if fixed_abs_linear is None else bool(fixed_abs_linear)
    )

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    ctrl = control_mean[order, :]
    g1_abs = graupel_mean[order, :]
    diff_g1 = (graupel_mean - control_mean)[order, :]
    diff_g2_chain = (twomom_mean - graupel_mean)[order, :]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
        ctrl = ctrl[keep, :]
        g1_abs = g1_abs[keep, :]
        diff_g1 = diff_g1[keep, :]
        diff_g2_chain = diff_g2_chain[keep, :]

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    is_special_graupel = variable.strip().upper() == "GRAUPEL"
    ctrl_plot = np.ma.masked_invalid(ctrl)
    g1_abs_plot = np.ma.masked_invalid(g1_abs)
    if not abs_linear:
        ctrl_plot = np.ma.masked_where(ctrl_plot <= 0, ctrl_plot)
        g1_abs_plot = np.ma.masked_where(g1_abs_plot <= 0, g1_abs_plot)
    diff_g1_plot = np.ma.masked_invalid(diff_g1)
    diff_g2_chain_plot = np.ma.masked_invalid(diff_g2_chain)

    if fixed_abs_limits is None:
        abs_ref = g1_abs_plot.filled(np.nan) if is_special_graupel else ctrl_plot.filled(np.nan)
        if abs_linear:
            valid = abs_ref[np.isfinite(abs_ref)]
            if valid.size > 0:
                vmin_abs = float(np.percentile(valid, 2.0))
                vmax_abs = float(np.percentile(valid, 98.0))
            else:
                vmin_abs, vmax_abs = 0.0, 1.0
            if is_unit_interval_variable(variable):
                vmin_abs = max(0.0, vmin_abs)
                vmax_abs = min(1.0, vmax_abs)
            if vmax_abs <= vmin_abs:
                vmax_abs = min(1.0, vmin_abs + 1e-6)
        else:
            vmin_abs, vmax_abs = robust_log_limits(abs_ref)
    else:
        vmin_abs, vmax_abs = fixed_abs_limits
        if abs_linear and is_unit_interval_variable(variable):
            vmin_abs = max(0.0, vmin_abs)
            vmax_abs = min(1.0, vmax_abs)
        if vmax_abs <= vmin_abs:
            if abs_linear:
                vmax_abs = vmin_abs + 1e-6
            else:
                vmax_abs = vmin_abs * 10.0

    if abs_linear:
        abs_norm = mcolors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    else:
        if vmax_abs <= vmin_abs:
            vmax_abs = vmin_abs * 10.0
        abs_norm = mcolors.LogNorm(vmin=vmin_abs, vmax=vmax_abs)

    if fixed_anom_scale is None:
        if is_special_graupel:
            anom_scale = robust_anomaly_scale(diff_g2_chain_plot.filled(np.nan))
        else:
            anom_scale = robust_anomaly_scale(
                diff_g1_plot.filled(np.nan),
                diff_g2_chain_plot.filled(np.nan),
            )
    else:
        anom_scale = float(fixed_anom_scale)
        if anom_scale <= 0:
            anom_scale = 1.0

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

    diff_norm = mcolors.TwoSlopeNorm(vmin=-anom_scale, vcenter=0.0, vmax=anom_scale)
    if is_special_graupel:
        axes[0].set_title(
            f"{EXPERIMENT_LABELS['control']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].text(
            0.5,
            0.5,
            "No Data",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.95, "pad": 4.0},
        )

        pcm_abs = axes[1].pcolormesh(
            hour_edges,
            y_edges,
            g1_abs_plot,
            cmap=cmaps.WhiteBlueGreenYellowRed,
            norm=abs_norm,
            shading="auto",
        )
        axes[1].set_title(
            f"{EXPERIMENT_LABELS['graupel']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )

        pcm_diff = axes[2].pcolormesh(
            hour_edges,
            y_edges,
            diff_g2_chain_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[2].set_title(
            f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
            fontsize=14,
            fontweight="bold",
        )
    else:
        pcm_abs = axes[0].pcolormesh(
            hour_edges,
            y_edges,
            ctrl_plot,
            cmap=cmaps.WhiteBlueGreenYellowRed,
            norm=abs_norm,
            shading="auto",
        )
        axes[0].set_title(
            f"{EXPERIMENT_LABELS['control']} ({var_label}, Absolute)",
            fontsize=14,
            fontweight="bold",
        )

        axes[1].pcolormesh(
            hour_edges,
            y_edges,
            diff_g1_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[1].set_title(
            f"{EXPERIMENT_LABELS['graupel']} - {EXPERIMENT_LABELS['control']}",
            fontsize=14,
            fontweight="bold",
        )

        rhs_plot = diff_g2_chain_plot
        pcm_diff = axes[2].pcolormesh(
            hour_edges,
            y_edges,
            rhs_plot,
            cmap="RdBu_r",
            norm=diff_norm,
            shading="auto",
        )
        axes[2].set_title(
            f"{EXPERIMENT_LABELS['2mom']} - {EXPERIMENT_LABELS['graupel']}",
            fontsize=14,
            fontweight="bold",
        )

    if freezing_lines_km:
        hours = np.arange(24, dtype=np.float64)
        panel_experiment = ("control", "graupel", "2mom")
        for idx, ax in enumerate(axes):
            exp = panel_experiment[idx]
            line = freezing_lines_km.get(exp)
            if line is None or not np.isfinite(line).any():
                continue
            ax.plot(
                hours,
                line,
                color="black",
                linewidth=1.8,
                linestyle="--",
                label=f"Freezing level ({EXPERIMENT_LABELS[exp]})",
                zorder=10,
            )
            ax.legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    title = f"{period_label} - {var_label}" if period_label else var_label
    fig.suptitle(title, fontsize=16, fontweight="bold")

    cbar_abs = fig.colorbar(
        pcm_abs,
        ax=axes[1] if is_special_graupel else axes[0],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    if is_updraft and variable.strip().upper() == "UPDRAFT_FLUX":
        abs_label = f"Mean Updraft Mass Flux [{unit}]"
    elif unit:
        abs_label = f"Mean {var_label} [{unit}]"
    else:
        abs_label = f"Mean {var_label} (Absolute)"
    cbar_abs.set_label(abs_label, fontsize=cbar_label_fs)
    cbar_abs.ax.tick_params(labelsize=cbar_tick_fs)
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    cbar_diff = fig.colorbar(
        pcm_diff,
        ax=axes[2] if is_special_graupel else axes[1:],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
    )
    if is_special_graupel:
        diff_label = (
            f"{var_label} anomaly ({EXPERIMENT_LABELS['2mom']}-{EXPERIMENT_LABELS['graupel']})"
        )
    elif is_updraft:
        diff_label = f"{var_label} anomaly [{unit}]" if unit else f"{var_label} anomaly"
    else:
        diff_label = f"{var_label} anomaly"
    cbar_diff.set_label(diff_label, fontsize=cbar_label_fs)
    cbar_diff.ax.tick_params(labelsize=cbar_tick_fs)
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
