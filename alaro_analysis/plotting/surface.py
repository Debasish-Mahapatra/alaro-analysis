"""Surface."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS


def plot_surface_diurnal(
    *,
    variable_label: str,
    variable_unit: str,
    period_label: str,
    line_data: dict[str, np.ndarray],
    lcl_data: dict[str, np.ndarray] | None,
    output_file: Path,
    utc_offset_hours: int,
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
    has_lcl = lcl_data is not None and any(
        lcl_data.get(exp) is not None and np.isfinite(lcl_data[exp]).any() for exp in EXPERIMENTS
    )
    if has_lcl and variable_unit:
        ylabel = f"{variable_label} and LCL [{variable_unit}]"

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
                label=f"{EXPERIMENT_LABELS[exp]} PBLH" if has_lcl else EXPERIMENT_LABELS[exp],
            )
            if has_lcl and lcl_data is not None:
                lcl_arr = lcl_data.get(exp)
                if lcl_arr is not None:
                    ax.plot(
                        hours,
                        lcl_arr,
                        linewidth=2.0,
                        linestyle="--",
                        marker="s",
                        markersize=3.0,
                        color=colors[exp],
                        alpha=0.85,
                        label=f"{EXPERIMENT_LABELS[exp]} LCL",
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
    title_label = f"{variable_label} and LCL" if has_lcl else variable_label
    ax.set_title(f"{period_label} - {title_label} diurnal cycle", fontsize=14, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
