"""Dsd profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS
from alaro_analysis.common.dsd_models import DdhQvProfile, DsdVerticalProfile
from alaro_analysis.dsd.vertical_profiles import freezing_levels_by_experiment

PANEL_SPECS = (
    ("d0_mm", "D$_0$", "D$_0$ (mm)"),
    ("log_nw", "log$_{10}$ N$_w$", "log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)"),
    ("qv", "DDH QV", "DDH QV tendency (g kg$^{-1}$ day$^{-1}$)"),
)


PANEL_LABELS = ("a", "b", "c")


def sorted_profile_xy(height: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(height) & np.isfinite(values)
    if not ok.any():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    z = height[ok]
    x = values[ok]
    order = np.argsort(z)
    return x[order], z[order]


def plot_profiles(
    output_path: Path,
    dsd_profiles: dict[str, DsdVerticalProfile],
    qv_profiles: dict[str, DdhQvProfile],
    experiments: list[str],
    *,
    qv_label: str,
    max_height_km: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(PANEL_SPECS),
        figsize=(5.0 * len(PANEL_SPECS), 8.0),
        sharey=True,
    )
    freezing_levels = freezing_levels_by_experiment(dsd_profiles, experiments)

    for panel_idx, (field, title, xlabel) in enumerate(PANEL_SPECS):
        ax = axes[panel_idx]
        for experiment in experiments:
            color = EXPERIMENT_COLORS.get(experiment, "black")
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            if field == "qv":
                values = qv_profiles[experiment].values
                height = qv_profiles[experiment].height_km
            else:
                values = dsd_profiles[experiment].values[field]
                height = dsd_profiles[experiment].height_km
            x, z = sorted_profile_xy(height, values)
            ax.plot(x, z, color=color, linewidth=2.4, label=label)
            freeze_z = freezing_levels[experiment]
            if np.isfinite(freeze_z):
                ax.axhline(
                    freeze_z,
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.75,
                    zorder=1,
                )

        ax.set_title(f"({PANEL_LABELS[panel_idx]}) {title}", fontsize=13)
        ax.set_xlabel(qv_label if field == "qv" else xlabel)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if panel_idx == 0:
            ax.set_ylabel("Height (km)")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color="0.25", linestyle="--", linewidth=1.3))
            labels.append("0 $^{\\circ}$C level")
            ax.legend(handles=handles, labels=labels, loc="best", frameon=False)

    axes[0].set_ylim(0.0, max_height_km)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
