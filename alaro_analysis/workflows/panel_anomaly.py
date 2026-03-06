#!/usr/bin/env python3
"""
Multi-variable anomaly panel from cached diurnal profiles.

Produces a single figure with 3 pcolormesh subplots stacked vertically:
  (a)  Cloud Fraction anomaly    (G1M − C1M)
  (b)  Relative Humidity anomaly (G1M − C1M)
  (c)  Rain + Liquid Water Content anomaly (G2M − G1M)

Reads pre-computed .npz caches produced by
plot_hydrometeor_diurnal_panels_masked_netcdf2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cmaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS, FREEZING_K
from alaro_analysis.common.models import AxisSpec
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.vertical import centers_to_edges

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INTERMEDIATE_DIRS = [
    Path("/Users/dev/Desktop/paper-1/data.npz/data"),
    Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data"),
]
DEFAULT_OUTPUT_DIR = Path("/Users/dev/Desktop/paper-1/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_cache_file(
    intermediate_roots: list[Path], relpath: Path
) -> Path | None:
    for root in intermediate_roots:
        candidate = root / relpath
        if candidate.exists():
            return candidate
    return None


def load_diurnal_mean(
    intermediate_roots: list[Path],
    variable: str,
    period_subdir: str,
    experiment: str,
    spatial_tag: str,
) -> np.ndarray:
    relpath = (
        Path(safe_name(variable))
        / period_subdir
        / f"{experiment}_{spatial_tag}_diurnal_profile.npz"
    )
    path = find_cache_file(intermediate_roots, relpath)
    if path is None:
        raise FileNotFoundError(
            f"Cache not found for {variable} / {experiment}: {relpath}\n"
            f"Searched roots: {intermediate_roots}"
        )
    payload = np.load(path)
    return np.asarray(payload["mean"], dtype=np.float64)


def load_height_axis(
    intermediate_roots: list[Path],
    period_subdir: str,
    experiment: str,
    spatial_tag: str,
    aggregate: str = "first",
) -> AxisSpec:
    relpath = (
        Path("geopotential")
        / period_subdir
        / f"{experiment}_{spatial_tag}_height_profile_{aggregate}.npz"
    )
    path = find_cache_file(intermediate_roots, relpath)
    if path is None:
        raise FileNotFoundError(
            f"Height cache not found for {experiment}: {relpath}\n"
            f"Searched roots: {intermediate_roots}"
        )
    payload = np.load(path)
    height_m = np.asarray(payload["height_m"], dtype=np.float64)
    return AxisSpec(values=height_m / 1000.0, label="Height (km)", is_height_km=True)


def load_temperature_profile(
    intermediate_roots: list[Path],
    period_subdir: str,
    experiment: str,
    spatial_tag: str,
) -> np.ndarray | None:
    relpath = (
        Path(safe_name("TEMPERATURE"))
        / period_subdir
        / f"{experiment}_{spatial_tag}_diurnal_profile.npz"
    )
    path = find_cache_file(intermediate_roots, relpath)
    if path is None:
        return None
    try:
        payload = np.load(path, allow_pickle=True)
        return np.asarray(payload["mean"], dtype=np.float64)
    except Exception:
        return None


def interpolate_profile_to_target_height(
    source_height_km: np.ndarray,
    source_profile: np.ndarray,
    target_height_km: np.ndarray,
) -> np.ndarray:
    out = np.full(
        (target_height_km.size, source_profile.shape[1]), np.nan, dtype=np.float64
    )
    for hour in range(source_profile.shape[1]):
        col = source_profile[:, hour]
        finite = np.isfinite(source_height_km) & np.isfinite(col)
        if np.sum(finite) < 2:
            continue
        z = source_height_km[finite]
        v = col[finite]
        order = np.argsort(z)
        z, v = z[order], v[order]
        unique = np.concatenate(([True], np.diff(z) > 0.0))
        z, v = z[unique], v[unique]
        if z.size < 2:
            continue
        out[:, hour] = np.interp(target_height_km, z, v, left=np.nan, right=np.nan)
    return out


def infer_freezing_threshold(temp: np.ndarray) -> float | None:
    valid = temp[np.isfinite(temp)]
    if valid.size == 0:
        return None
    return FREEZING_K if float(np.median(valid)) > 150.0 else 0.0


def compute_freezing_line_km(
    axis: AxisSpec, temperature_profile: np.ndarray
) -> np.ndarray | None:
    if not axis.is_height_km:
        return None
    n_levels = min(axis.values.size, temperature_profile.shape[0])
    if n_levels < 2:
        return None
    temp = temperature_profile[:n_levels, :]
    threshold = infer_freezing_threshold(temp)
    if threshold is None:
        return None
    y_km = np.asarray(axis.values[:n_levels], dtype=np.float64)
    order = np.argsort(y_km)
    y_sorted = y_km[order]
    t_sorted = temp[order, :]
    freeze_line = np.full((24,), np.nan, dtype=np.float64)
    for hour in range(24):
        column = t_sorted[:, hour]
        finite = np.isfinite(column) & np.isfinite(y_sorted)
        if np.sum(finite) < 2:
            continue
        yy, tt = y_sorted[finite], column[finite]
        for i in range(yy.size - 1):
            t1, t2 = tt[i], tt[i + 1]
            y1, y2 = yy[i], yy[i + 1]
            if np.isclose(t1, threshold):
                freeze_line[hour] = y1
                break
            if np.isclose(t2, threshold):
                freeze_line[hour] = y2
                break
            d1, d2 = t1 - threshold, t2 - threshold
            if d1 * d2 < 0 and not np.isclose(t1, t2):
                frac = (threshold - t1) / (t2 - t1)
                freeze_line[hour] = y1 + frac * (y2 - y1)
                break
    return freeze_line


def robust_anomaly_scale(*arrays: np.ndarray) -> float:
    chunks = [arr[np.isfinite(arr)] for arr in arrays]
    chunks = [c for c in chunks if c.size > 0]
    if not chunks:
        return 1.0
    merged = np.concatenate(chunks)
    scale = float(np.percentile(np.abs(merged), 98))
    return scale if scale > 0 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multi_variable_anomaly_panel(
    anomalies: list[np.ndarray],
    panel_titles: list[str],
    panel_labels: list[str],
    axis: AxisSpec,
    max_height_km: float,
    freezing_lines_km: list[np.ndarray | None],
    output_file: Path,
    period_label: str,
    fixed_anom_scales: list[float | None] | None = None,
    dpi: int = 450,
) -> None:
    n_panels = len(anomalies)
    axis_label_fs = 22
    tick_label_fs = 20
    panel_tag_fs = 20
    cbar_label_fs = 20
    cbar_tick_fs = 18
    legend_fs = 18
    title_fs = 20

    if fixed_anom_scales is None:
        fixed_anom_scales = [None] * n_panels

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
    else:
        keep = slice(None)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(20, 7), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    for idx, (anom, title, label) in enumerate(
        zip(anomalies, panel_titles, panel_labels)
    ):
        ax = axes[idx]
        anom_sorted = anom[order, :]
        if axis.is_height_km:
            anom_sorted = anom_sorted[keep, :]

        anom_plot = np.ma.masked_invalid(anom_sorted)

        if fixed_anom_scales[idx] is not None:
            anom_scale = fixed_anom_scales[idx]
        else:
            anom_scale = robust_anomaly_scale(anom_plot.filled(np.nan))
        diff_norm = mcolors.TwoSlopeNorm(
            vmin=-anom_scale, vcenter=0.0, vmax=anom_scale
        )

        ax.set_facecolor("#d3d3d3")
        pcm = ax.pcolormesh(
            hour_edges, y_edges, anom_plot,
            cmap="RdBu_r", norm=diff_norm, shading="auto",
        )
        ax.set_title(title, fontsize=title_fs, fontweight="bold")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=axis_label_fs)
        if idx == 0:
            ax.set_ylabel(axis.label, fontsize=axis_label_fs)
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(axis="both", labelsize=tick_label_fs)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)

        ax.text(
            0.02, 0.98, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=panel_tag_fs, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.16)
        cbar.set_label(label, fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

        fl = freezing_lines_km[idx] if idx < len(freezing_lines_km) else None
        if fl is not None and np.isfinite(fl).any():
            hours = np.arange(24, dtype=np.float64)
            ax.plot(
                hours, fl,
                color="black", linewidth=1.8, linestyle="--",
                label="Freezing level", zorder=10,
            )
            ax.legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


def plot_abs_anomaly_panel(
    absolute: np.ndarray,
    anomaly1: np.ndarray,
    anomaly2: np.ndarray,
    panel_titles: list[str],
    abs_cbar_label: str,
    anom_cbar_label: str,
    axis: AxisSpec,
    max_height_km: float,
    freezing_lines_km: list[np.ndarray | None],
    output_file: Path,
    fixed_anom_scale: float | None = None,
    dpi: int = 450,
) -> None:
    axis_label_fs = 22
    tick_label_fs = 20
    panel_tag_fs = 20
    cbar_label_fs = 20
    cbar_tick_fs = 18
    legend_fs = 18
    title_fs = 20

    y = np.asarray(axis.values, dtype=np.float64)
    order = np.argsort(y)
    y = y[order]

    if axis.is_height_km:
        keep = np.isfinite(y) & (y >= 0.0) & (y <= max_height_km)
        y = y[keep]
    else:
        keep = slice(None)

    hour_edges = np.arange(25, dtype=np.float64) - 0.5
    y_edges = centers_to_edges(y)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    panels = [absolute, anomaly1, anomaly2]
    labels = [abs_cbar_label, anom_cbar_label, anom_cbar_label]

    for idx, (data, title, label) in enumerate(
        zip(panels, panel_titles, labels)
    ):
        ax = axes[idx]
        data_sorted = data[order, :]
        if axis.is_height_km:
            data_sorted = data_sorted[keep, :]

        data_plot = np.ma.masked_invalid(data_sorted)

        ax.set_facecolor("#d3d3d3")

        if idx == 0:
            # Absolute panel — log scale
            data_plot = np.ma.masked_where(data_plot <= 0, data_plot)
            valid = data_plot.compressed()
            if valid.size > 0:
                vmin = float(np.percentile(valid, 5))
                vmax = float(np.percentile(valid, 99))
                vmin = max(vmin, float(np.min(valid)))
                if vmax <= vmin:
                    vmax = vmin * 10.0
            else:
                vmin, vmax = 1e-12, 1.0
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(
                hour_edges, y_edges, data_plot,
                cmap=cmaps.WhiteBlueGreenYellowRed, norm=norm, shading="auto",
            )
        else:
            # Anomaly panel — diverging
            if fixed_anom_scale is not None:
                anom_scale = fixed_anom_scale
            else:
                anom_scale = robust_anomaly_scale(data_plot.filled(np.nan))
            norm = mcolors.TwoSlopeNorm(
                vmin=-anom_scale, vcenter=0.0, vmax=anom_scale
            )
            pcm = ax.pcolormesh(
                hour_edges, y_edges, data_plot,
                cmap="RdBu_r", norm=norm, shading="auto",
            )

        ax.set_title(title, fontsize=title_fs, fontweight="bold")
        ax.set_xlabel("Hour (Amazon UTC-4)", fontsize=axis_label_fs)
        if idx == 0:
            ax.set_ylabel(axis.label, fontsize=axis_label_fs)
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(0, 24, 6))
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(axis="both", labelsize=tick_label_fs)
        if axis.is_height_km:
            ax.set_ylim(0.0, max_height_km)

        ax.text(
            0.02, 0.98, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=panel_tag_fs, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
        )

        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.16)
        cbar.set_label(label, fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

        fl = freezing_lines_km[idx] if idx < len(freezing_lines_km) else None
        if fl is not None and np.isfinite(fl).any():
            hours = np.arange(24, dtype=np.float64)
            ax.plot(
                hours, fl,
                color="black", linewidth=1.8, linestyle="--",
                label="Freezing level", zorder=10,
            )
            ax.legend(loc="upper right", fontsize=legend_fs, framealpha=0.9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {output_file}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-variable anomaly panel from cached diurnal profiles. "
            "Produces a single figure with 3 pcolormesh subplots."
        ),
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        nargs="+",
        default=DEFAULT_INTERMEDIATE_DIRS,
        help="Root(s) for cached .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--period-subdir",
        default="2years",
        help="Period subdirectory inside intermediate dirs (default: 2years).",
    )
    parser.add_argument(
        "--period-label",
        default="Full 2-year (all months)",
        help="Label for the figure suptitle.",
    )
    parser.add_argument(
        "--spatial-tag",
        default="full-domain",
        help="Spatial averaging tag used during caching.",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit for height-based plots.",
    )
    parser.add_argument(
        "--height-aggregate",
        default="first",
        help="Geopotential aggregation method.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--output-filename",
        default="multi_variable_anomaly_panel.png",
        help="Output filename.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    roots = [p.resolve() for p in args.intermediate_dir]
    period_sub = args.period_subdir
    stag = args.spatial_tag

    # ------- load height axis (use control as reference) -------
    axis_ctrl = load_height_axis(roots, period_sub, "control", stag, args.height_aggregate)

    # ------- load diurnal profiles -------
    print("Loading cached diurnal profiles ...", flush=True)

    cf_ctrl = load_diurnal_mean(roots, "CLOUD_FRACTI", period_sub, "control", stag)
    cf_g1   = load_diurnal_mean(roots, "CLOUD_FRACTI", period_sub, "graupel", stag)

    rh_ctrl = load_diurnal_mean(roots, "HUMI.RELATIVE", period_sub, "control", stag)
    rh_g1   = load_diurnal_mean(roots, "HUMI.RELATIVE", period_sub, "graupel", stag)

    rain_ctrl = load_diurnal_mean(roots, "RAIN", period_sub, "control", stag)
    rain_g1   = load_diurnal_mean(roots, "RAIN", period_sub, "graupel", stag)
    rain_g2   = load_diurnal_mean(roots, "RAIN", period_sub, "2mom", stag)

    lwc_ctrl = load_diurnal_mean(roots, "LIQUID_WATER", period_sub, "control", stag)
    lwc_g1   = load_diurnal_mean(roots, "LIQUID_WATER", period_sub, "graupel", stag)
    lwc_g2   = load_diurnal_mean(roots, "LIQUID_WATER", period_sub, "2mom", stag)

    # ------- load height axes for interpolation -------
    axis_g1  = load_height_axis(roots, period_sub, "graupel", stag, args.height_aggregate)
    axis_g2  = load_height_axis(roots, period_sub, "2mom",    stag, args.height_aggregate)
    target_h = np.asarray(axis_ctrl.values, dtype=np.float64)

    def interp_to_ctrl(src_axis: AxisSpec, profile: np.ndarray) -> np.ndarray:
        n = min(src_axis.values.size, profile.shape[0])
        return interpolate_profile_to_target_height(
            source_height_km=src_axis.values[:n],
            source_profile=profile[:n, :],
            target_height_km=target_h,
        )

    # Interpolate all non-control profiles to control height grid
    cf_g1_i   = interp_to_ctrl(axis_g1, cf_g1)
    rh_g1_i   = interp_to_ctrl(axis_g1, rh_g1)
    rain_g1_i = interp_to_ctrl(axis_g1, rain_g1)
    rain_g2_i = interp_to_ctrl(axis_g2, rain_g2)
    lwc_g1_i  = interp_to_ctrl(axis_g1, lwc_g1)
    lwc_g2_i  = interp_to_ctrl(axis_g2, lwc_g2)

    # Truncate control profiles to same vertical extent
    n = target_h.size
    cf_ctrl_t = cf_ctrl[:n, :]
    rh_ctrl_t = rh_ctrl[:n, :]
    rain_ctrl_t = rain_ctrl[:n, :]
    lwc_ctrl_t  = lwc_ctrl[:n, :]

    # ------- compute anomalies -------
    anom_cf = cf_g1_i - cf_ctrl_t            # G1M − C1M
    anom_rh = rh_g1_i - rh_ctrl_t            # G1M − C1M
    anom_rain_lwc = (rain_g2_i + lwc_g2_i) - (rain_g1_i + lwc_g1_i)  # G2M − G1M

    print("Anomalies computed.", flush=True)

    # ------- freezing levels (per experiment) -------
    freezing_lines = {}
    for exp, ax_exp in [("control", axis_ctrl), ("graupel", axis_g1), ("2mom", axis_g2)]:
        temp = load_temperature_profile(roots, period_sub, exp, stag)
        if temp is not None:
            # Interpolate temperature to control height grid if needed
            if exp != "control":
                n_t = min(ax_exp.values.size, temp.shape[0])
                temp = interpolate_profile_to_target_height(
                    source_height_km=ax_exp.values[:n_t],
                    source_profile=temp[:n_t, :],
                    target_height_km=target_h,
                )
            fl = compute_freezing_line_km(axis_ctrl, temp)
            if fl is not None:
                freezing_lines[exp] = fl
                print(f"Freezing level computed from {exp} temperature.", flush=True)
            else:
                freezing_lines[exp] = None
                print(f"[warn] Could not compute freezing level for {exp}.", flush=True)
        else:
            freezing_lines[exp] = None
            print(f"[warn] Temperature cache not found for {exp}; no freezing level.", flush=True)

    # ------- plot multi-variable anomaly panel -------
    # Panels: (a) CF G1M-C1M → graupel, (b) RH G1M-C1M → graupel, (c) Rain+LWC G2M-G1M → 2mom
    output_file = args.output_dir.resolve() / args.output_filename
    plot_multi_variable_anomaly_panel(
        anomalies=[anom_cf, anom_rh, anom_rain_lwc],
        panel_titles=[
            f"Cloud Fraction ({EXPERIMENT_LABELS['graupel']} \u2212 {EXPERIMENT_LABELS['control']})",
            f"Relative Humidity ({EXPERIMENT_LABELS['graupel']} \u2212 {EXPERIMENT_LABELS['control']})",
            f"Rain + LWC ({EXPERIMENT_LABELS['2mom']} \u2212 {EXPERIMENT_LABELS['graupel']})",
        ],
        panel_labels=[
            "Cloud fraction anomaly",
            "Relative humidity anomaly",
            "Rain + LWC anomaly",
        ],
        axis=axis_ctrl,
        max_height_km=args.max_height_km,
        freezing_lines_km=[
            freezing_lines.get("graupel"),
            freezing_lines.get("graupel"),
            freezing_lines.get("2mom"),
        ],
        output_file=output_file,
        period_label=args.period_label,
        fixed_anom_scales=[None, None, 2e-5],
        dpi=args.dpi,
    )

    # ------- Rain + LWC  absolute + anomaly panel -------
    rlwc_ctrl = rain_ctrl_t + lwc_ctrl_t
    rlwc_g1   = rain_g1_i + lwc_g1_i
    rlwc_g2   = rain_g2_i + lwc_g2_i
    anom_rlwc_g1_c1 = rlwc_g1 - rlwc_ctrl   # G1M − C1M
    anom_rlwc_g2_g1 = rlwc_g2 - rlwc_g1     # G2M − G1M

    # Panels: (a) C1M abs → control, (b) G1M-C1M → graupel, (c) G2M-G1M → 2mom
    output_rlwc = args.output_dir.resolve() / "rain_lwc_abs_anomaly_panel.png"
    plot_abs_anomaly_panel(
        absolute=rlwc_ctrl,
        anomaly1=anom_rlwc_g1_c1,
        anomaly2=anom_rlwc_g2_g1,
        panel_titles=[
            f"{EXPERIMENT_LABELS['control']} (Rain + LWC)",
            f"{EXPERIMENT_LABELS['graupel']} \u2212 {EXPERIMENT_LABELS['control']}",
            f"{EXPERIMENT_LABELS['2mom']} \u2212 {EXPERIMENT_LABELS['graupel']}",
        ],
        abs_cbar_label="Mean Rain + LWC [kg/kg]",
        anom_cbar_label="Rain + LWC anomaly",
        axis=axis_ctrl,
        max_height_km=args.max_height_km,
        freezing_lines_km=[
            freezing_lines.get("control"),
            freezing_lines.get("graupel"),
            freezing_lines.get("2mom"),
        ],
        output_file=output_rlwc,
        fixed_anom_scale=2e-5,
        dpi=args.dpi,
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
