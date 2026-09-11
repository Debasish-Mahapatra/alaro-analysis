"""Strong updraft profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.strong_profile_settings import ISOTHERM_STYLE, ISOTHERMS_C
from alaro_analysis.common.study_specs import RAIN_PROFILE, ProfileSpec
from alaro_analysis.data.strong_updraft_profiles import RainProfile


def plot_profiles(
    profiles: dict[str, RainProfile],
    experiments: list[str],
    output_path: Path,
    *,
    max_height_km: float,
    rain_xmin_kg_kg: float,
    rain_xmax_kg_kg: float,
    dpi: int,
    spec: ProfileSpec = RAIN_PROFILE,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, len(experiments), figsize=(9.8, 4.4), sharey=True)
    if len(experiments) == 1:
        axes = np.asarray([axes])
    legend_handles = []
    legend_labels = []
    for ax, experiment in zip(axes, experiments):
        result = profiles[experiment]
        centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
        height_ok = centers <= max_height_km
        rain_kg_kg = result.rain_kg_kg
        ok = height_ok & np.isfinite(rain_kg_kg) & (rain_kg_kg > 0.0) & (result.rain_count > 0)
        rain_line = ax.plot(
            rain_kg_kg[ok],
            centers[ok],
            color=spec.color,
            linewidth=2.0,
            label=f"{EXPERIMENT_LABELS[experiment]} {spec.slug}",
        )[0]
        if experiment == experiments[0]:
            legend_handles.append(rain_line)
            legend_labels.append(spec.label)
        for target_c in ISOTHERMS_C:
            y = result.isotherm_heights_km.get(target_c, float("nan"))
            if not np.isfinite(y) or y < 0 or y > max_height_km:
                continue
            line = ax.axhline(y, **ISOTHERM_STYLE[target_c])
            label = f"{target_c:g} C"
            ax.text(
                rain_xmax_kg_kg * 0.93,
                y + 0.05,
                label,
                ha="right",
                va="bottom",
                fontsize=7,
                color=ISOTHERM_STYLE[target_c]["color"],
            )
            if experiment == experiments[0]:
                legend_handles.append(line)
                legend_labels.append(label)
        ax.set_title(EXPERIMENT_LABELS[experiment], fontweight="normal", pad=6)
        ax.set_xlabel(f"{spec.label} mixing ratio (kg kg$^{{-1}}$)")
        ax.set_xscale("log")
        ax.set_xlim(rain_xmin_kg_kg, rain_xmax_kg_kg)
        ax.set_ylim(0.0, max_height_km)
        ax.grid(True, which="major", color="#d0d0d0", linewidth=0.55, alpha=0.7)
        ax.grid(True, which="minor", color="#e6e6e6", linewidth=0.35, alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(
            0.04,
            0.96,
            f"{result.n_files_with_strong_updraft:,}/{result.n_files_used:,} timesteps",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#333333",
        )
    axes[0].set_ylabel("Height (km)")
    fig.suptitle(f"Strong-updraft {spec.slug} profiles", fontsize=11, fontweight="normal", y=0.99)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text_data(
    profiles: dict[str, RainProfile],
    experiments: list[str],
    output_path: Path,
    *,
    args: argparse.Namespace,
    spec: ProfileSpec = RAIN_PROFILE,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Paper1 strong-updraft {spec.slug} vertical profiles\n")
        f.write("#\n")
        f.write("# Strong updraft definition:\n")
        f.write(f"#   UD_MESH_FRAC > {args.min_updraft_mesh_frac:g}\n")
        f.write(f"#   -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s\n")
        f.write(f"#   forecast lead hours {args.min_lead_hour}..{args.max_lead_hour}\n")
        f.write("#   +0024 is excluded by default to avoid duplicate valid times.\n")
        f.write("#\n")
        f.write("# Units:\n")
        f.write(f"#   Raw FA {spec.variable} fields have no plain text unit string, but their\n")
        f.write(f"#   generic identifier maps to {spec.generic_id}.\n")
        f.write("#   Converted NetCDF metadata stores this as units=1, i.e. dimensionless\n")
        f.write("#   kg/kg mixing ratio. The plot x-axis uses kg/kg directly.\n")
        f.write(
            f"#   Plot x-axis is logarithmic from {args.xmin_kg_kg:g} to {args.xmax_kg_kg:g} kg/kg.\n"
        )
        f.write("#\n")
        f.write("# Isotherms:\n")
        f.write("#   Isotherm heights are interpolated from the strong-updraft-mean\n")
        f.write("#   TEMPERATURE profile in each experiment.\n")
        for experiment in experiments:
            result = profiles[experiment]
            parts = [
                f"{target:g}C={result.isotherm_heights_km.get(target, float('nan')):.3f} km"
                for target in ISOTHERMS_C
            ]
            f.write(
                f"#   {experiment} ({EXPERIMENT_LABELS[experiment]}): " + ", ".join(parts) + "\n"
            )
        f.write("#\n")
        f.write("# Summary by experiment:\n")
        for experiment in experiments:
            result = profiles[experiment]
            f.write(
                f"#   {experiment} ({EXPERIMENT_LABELS[experiment]}): days={result.n_days_seen}, files_seen={result.n_files_seen}, files_used={result.n_files_used}, files_with_strong_updraft={result.n_files_with_strong_updraft}, missing_or_failed_file_groups={result.missing_files}, strong_gridpoint_samples={int(np.sum(result.strong_sample_count))}\n"
            )
        f.write("#\n")
        header = [
            "experiment",
            "experiment_label",
            "height_bin_left_km",
            "height_bin_right_km",
            "height_center_km",
            "strong_gridpoint_samples",
            f"mean_{spec.variable}_kg_kg-1",
            f"mean_{spec.variable}_g_kg-1",
            f"count_{spec.variable}",
            "mean_TEMPERATURE_K",
            "mean_TEMPERATURE_C",
            "count_TEMPERATURE",
        ]
        f.write("\t".join(header) + "\n")
        for experiment in experiments:
            result = profiles[experiment]
            centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
            for idx, center in enumerate(centers):
                rain_kg_kg = result.rain_kg_kg[idx]
                rain_g_kg = rain_kg_kg * 1000.0 if np.isfinite(rain_kg_kg) else np.nan
                temp_k = result.temperature_k[idx]
                temp_c = temp_k - 273.15 if np.isfinite(temp_k) else np.nan
                row = [
                    experiment,
                    EXPERIMENT_LABELS[experiment],
                    f"{result.h_bins_km[idx]:.3f}",
                    f"{result.h_bins_km[idx + 1]:.3f}",
                    f"{center:.3f}",
                    str(int(result.strong_sample_count[idx])),
                    f"{rain_kg_kg:.10e}" if np.isfinite(rain_kg_kg) else "nan",
                    f"{rain_g_kg:.10e}" if np.isfinite(rain_g_kg) else "nan",
                    str(int(result.rain_count[idx])),
                    f"{temp_k:.6f}" if np.isfinite(temp_k) else "nan",
                    f"{temp_c:.6f}" if np.isfinite(temp_c) else "nan",
                    str(int(result.temperature_count[idx])),
                ]
                f.write("\t".join(row) + "\n")
