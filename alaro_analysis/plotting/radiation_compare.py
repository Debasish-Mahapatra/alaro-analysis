"""Radiation compare."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.radiation_compare import compute_mae
from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS


def plot_net_comparison(
    *,
    period_label: str,
    output_file: Path,
    alaro_lines: dict[str, dict[str, np.ndarray]],
    surfex_lines: dict[str, dict[str, np.ndarray]],
    utc_offset_hours: int,
) -> None:
    hours = np.arange(24, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True, constrained_layout=True)

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        alaro_rn = alaro_lines[exp]["ALARO_RN"]
        surfex_rn = surfex_lines[exp]["SURFEX_RN"]

        ax.plot(
            hours, alaro_rn, color="black", linewidth=2.6, label="ALARO RN (de-acc.) = SOLA + THER"
        )
        ax.plot(
            hours, surfex_rn, color="#0b7285", linewidth=2.4, linestyle="--", label="SURFEX SFX.RN"
        )
        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=11)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlim(0.0, 23.0)
        ax.grid(alpha=0.25, linestyle="--")

        mae = compute_mae(alaro_rn, surfex_rn)
        if mae is not None:
            ax.text(
                0.02,
                0.95,
                f"MAE = {mae:.1f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )

    axes[0].set_ylabel("Net radiation [W m-2]", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.suptitle(
        f"{period_label} - ALARO net radiation vs SURFEX SFX.RN", fontsize=15, fontweight="bold"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_alaro_components(
    *,
    period_label: str,
    output_file: Path,
    alaro_lines: dict[str, dict[str, np.ndarray]],
    utc_offset_hours: int,
) -> None:
    hours = np.arange(24, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True, constrained_layout=True)

    component_styles = {
        "SW_DOWN": {"color": "#f59f00", "linestyle": "-", "label": "SWD = SURFRF.SHORT.DO"},
        "LW_DOWN": {"color": "#5f3dc4", "linestyle": "-", "label": "LWD = SURFRF.LONG.DO"},
        "SW_NET": {"color": "#ff922b", "linestyle": "--", "label": "SWnet = SURFFLU.RAY.SOLA"},
        "LW_NET": {"color": "#1c7ed6", "linestyle": "--", "label": "LWnet = SURFFLU.RAY.THER"},
        "SW_UP": {"color": "#e8590c", "linestyle": ":", "label": "SWU = SWD - SWnet"},
        "LW_UP": {"color": "#6741d9", "linestyle": ":", "label": "LWU = LWD - LWnet"},
        "ALARO_RN": {"color": "black", "linestyle": "-", "label": "RN = SWnet + LWnet"},
    }

    for idx, exp in enumerate(EXPERIMENTS):
        ax = axes[idx]
        for key, style in component_styles.items():
            ax.plot(
                hours,
                alaro_lines[exp][key],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2 if key == "ALARO_RN" else 2.0,
                label=style["label"],
            )
        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Hour (local UTC{utc_offset_hours:+d})", fontsize=11)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlim(0.0, 23.0)
        ax.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Radiative flux [W m-2]", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"{period_label} - ALARO radiation components", fontsize=15, fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)
