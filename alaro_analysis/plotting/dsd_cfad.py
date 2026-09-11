"""Dsd cfad. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.cfad import frequency_percent
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.dsd_models import ExperimentCfad
from alaro_analysis.common.figio import strip_cbar_zeros
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.dsd.profile_cfad import PANEL_FIELDS

FIELD_LABELS = {
    "d0_mm": "D$_0$ (mm)",
    "log_nw": "log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)",
}


ROW_LABELS = {
    "d0_mm": "D$_0$",
    "log_nw": "log$_{10}$ N$_w$",
}


PANEL_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")


TITLE_FONTSIZE = 16


AXIS_LABEL_FONTSIZE = 16


TICK_LABEL_FONTSIZE = 14


PANEL_LABEL_FONTSIZE = 16


COLORBAR_LABEL_FONTSIZE = 16


COLORBAR_TICK_FONTSIZE = 14


def row_vmax(cfads: dict[str, ExperimentCfad], experiments: list[str], field: str) -> float:
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    return max(1.0, float(np.nanpercentile(merged, 99.0)))


def cfad_colormap():
    try:
        import cmaps
    except ImportError as exc:  # pragma: no cover - depends on optional runtime env
        raise RuntimeError(
            "The cmaps package is required for this plot; install the full extra "
            "or run in an environment that provides cmaps."
        ) from exc
    return cmaps.WhViBlGrYeOrRe


def plot_cfads(
    output_path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: list[str],
    *,
    max_height_km: float,
    dpi: int,
    row_vmax_values: Mapping[str, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = cfad_colormap()
    fig, axes = plt.subplots(
        len(PANEL_FIELDS),
        len(experiments),
        figsize=(5.4 * len(experiments), 4.0 * len(PANEL_FIELDS)),
        sharey=True,
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for row, field in enumerate(PANEL_FIELDS):
        vmax = (
            float(row_vmax_values[field])
            if row_vmax_values is not None and field in row_vmax_values
            else row_vmax(cfads, experiments, field)
        )
        last_im = None
        for col, experiment in enumerate(experiments):
            ax = axes[row, col]
            panel_idx = row * len(experiments) + col
            grid = cfads[experiment].grids[field]
            freq = frequency_percent(grid.hist)
            y_edges = centers_to_edges(grid.height_km)
            last_im = ax.pcolormesh(
                grid.x_edges,
                y_edges,
                freq,
                cmap=cmap,
                shading="auto",
                vmin=0.0,
                vmax=vmax,
            )
            freeze = cfads[experiment].freezing_level_km
            if np.isfinite(freeze):
                ax.axhline(freeze, color="white", linestyle="--", linewidth=1.4, alpha=0.95)
                ax.axhline(freeze, color="black", linestyle="--", linewidth=0.7, alpha=0.75)

            if row == 0:
                ax.set_title(
                    EXPERIMENT_LABELS.get(experiment, experiment),
                    fontsize=TITLE_FONTSIZE,
                    pad=10,
                )
            ax.text(
                0.03,
                0.96,
                f"({PANEL_LABELS[panel_idx]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
            if col == 0:
                ax.set_ylabel(
                    f"{ROW_LABELS[field]}\nHeight (km)",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            ax.set_xlabel(FIELD_LABELS[field], fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylim(0.0, max_height_km)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(False)

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes[row, :], shrink=0.84, pad=0.012)
            strip_cbar_zeros(cbar)
            cbar.set_label("Frequency per level (%)", fontsize=COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
