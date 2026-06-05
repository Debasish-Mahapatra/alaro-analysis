"""Rainfall intensity-hour heatmaps from common-valid rainfall data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.figio import DatasetConfig, add_io_args, save_figure


from alaro_analysis.common.constants import RUNS_ROOT
DEFAULT_DATA_DIR = (
    RUNS_ROOT
    / "rainfall-regridded-to-imerge"
    / "masked-production-final"
    / "common-valid-time-production"
)
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "rainfall_intensity_heatmaps"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")


@dataclass(frozen=True)
class ModelRainDataset:
    key: str
    label: str
    filename: str
    convective_variable: str = "convective_rain"
    stratiform_variable: str = "stratiform_rain"
    total_variable: str = "total_rain"


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall_rate"),
    DatasetConfig("imerg", "IMERG", "IMERG_common_valid.nc", "precipitation"),
    DatasetConfig("control", "C1M", "Control_common_valid.nc", "total_rain"),
    DatasetConfig("graupel", "G1M", "Graupel_common_valid.nc", "total_rain"),
    DatasetConfig("2mom", "G2M", "2-Moment_common_valid.nc", "total_rain"),
)

MODEL_DATASETS: tuple[ModelRainDataset, ...] = (
    ModelRainDataset("control", "C1M", "Control_common_valid.nc"),
    ModelRainDataset("graupel", "G1M", "Graupel_common_valid.nc"),
    ModelRainDataset("2mom", "G2M", "2-Moment_common_valid.nc"),
)


def local_hours_from_utc(times: np.ndarray, utc_offset_hours: int) -> np.ndarray:
    utc = np.asarray(times, dtype="datetime64[ns]")
    local = utc + np.timedelta64(int(utc_offset_hours), "h")
    return (local.astype("datetime64[h]").astype(np.int64) % 24).astype(np.int16)


def compute_intensity_hour_histogram(
    values: np.ndarray,
    times: np.ndarray,
    intensity_bins: np.ndarray,
    *,
    wet_threshold: float,
    utc_offset_hours: int,
) -> np.ndarray:
    """Return wet-pixel count histogram with shape (intensity_bin, local_hour)."""
    rain = np.asarray(values, dtype=np.float64)
    if rain.shape[0] != len(times):
        raise ValueError(f"value/time length mismatch: {rain.shape[0]} vs {len(times)}")
    hours = local_hours_from_utc(times, utc_offset_hours)

    histogram = np.zeros((len(intensity_bins) - 1, 24), dtype=np.float64)
    for hour in range(24):
        selected = rain[hours == hour].ravel()
        selected = selected[np.isfinite(selected) & (selected >= wet_threshold)]
        if selected.size:
            counts, _ = np.histogram(selected, bins=intensity_bins)
            histogram[:, hour] = counts
    return histogram


def _read_variable(data_dir: Path, filename: str, variable: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {path}")
        values = np.asarray(ds[variable].values, dtype=np.float64)
        times = np.asarray(ds["time"].values)
    return values, times, path


def read_intensity_heatmaps(
    data_dir: Path,
    intensity_bins: np.ndarray,
    *,
    wet_threshold: float,
    utc_offset_hours: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    heatmaps: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for cfg in DATASETS:
        values, times, path = _read_variable(data_dir, cfg.filename, cfg.variable)
        heatmaps[cfg.key] = compute_intensity_hour_histogram(
            values,
            times,
            intensity_bins,
            wet_threshold=wet_threshold,
            utc_offset_hours=utc_offset_hours,
        )
        paths[cfg.key] = path
    return heatmaps, paths


def read_convective_stratiform_heatmaps(
    data_dir: Path,
    intensity_bins: np.ndarray,
    *,
    wet_threshold: float,
    utc_offset_hours: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    heatmaps: dict[str, dict[str, np.ndarray]] = {}
    paths: dict[str, Path] = {}
    for cfg in MODEL_DATASETS:
        path = data_dir / cfg.filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with xr.open_dataset(path) as ds:
            for variable in (cfg.total_variable, cfg.convective_variable, cfg.stratiform_variable):
                if variable not in ds:
                    raise KeyError(f"{variable!r} not found in {path}")
            total = np.asarray(ds[cfg.total_variable].values, dtype=np.float64)
            convective = np.asarray(ds[cfg.convective_variable].values, dtype=np.float64)
            stratiform = np.asarray(ds[cfg.stratiform_variable].values, dtype=np.float64)
            times = np.asarray(ds["time"].values)

        mask = ~np.isfinite(total)
        convective = np.where(mask, np.nan, convective)
        stratiform = np.where(mask, np.nan, stratiform)
        heatmaps[cfg.key] = {
            "convective": compute_intensity_hour_histogram(
                convective,
                times,
                intensity_bins,
                wet_threshold=wet_threshold,
                utc_offset_hours=utc_offset_hours,
            ),
            "stratiform": compute_intensity_hour_histogram(
                stratiform,
                times,
                intensity_bins,
                wet_threshold=wet_threshold,
                utc_offset_hours=utc_offset_hours,
            ),
        }
        paths[cfg.key] = path
    return heatmaps, paths


def _panel_label(ax, index: int) -> None:
    ax.text(
        0.03,
        0.97,
        PANEL_LABELS[index],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.6,
            "boxstyle": "round,pad=0.18",
        },
        zorder=20,
    )


def _heatmap_axes_format(ax, intensity_bins: np.ndarray, *, x_label: bool, y_label: bool) -> None:
    ax.set_yscale("log")
    ax.set_ylim(float(intensity_bins[0]), float(intensity_bins[-1]))
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.0)
    if x_label:
        ax.set_xlabel(r"Local Time (UTC$-$4)", fontsize=13)
    if y_label:
        ax.set_ylabel(r"Intensity (mm h$^{-1}$)", fontsize=13)


def _rain_cmap():
    try:
        import cmaps

        return cmaps.WhiteBlueGreenYellowRed
    except ImportError:  # pragma: no cover - fallback for minimal environments.
        return "turbo"


def plot_intensity_evolution(
    heatmaps: dict[str, np.ndarray],
    intensity_bins: np.ndarray,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_edges = np.arange(25)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = np.sqrt(intensity_bins[:-1] * intensity_bins[1:])
    radar = heatmaps["radar"]
    global_max = max(1.0, max(float(np.nanmax(h)) for h in heatmaps.values()))
    max_diff = max(float(np.nanmax(np.abs(h - radar))) for key, h in heatmaps.items() if key != "radar")
    max_diff = max(max_diff, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi)
    axes_flat = axes.ravel()
    radar_mesh = None
    diff_mesh = None
    contour_levels_pos = [200, 500, 1000, 2000, 5000]
    contour_levels_neg = [-5000, -2000, -1000, -500, -200]

    for index, cfg in enumerate(DATASETS):
        ax = axes_flat[index]
        heatmap = heatmaps[cfg.key]
        row = index // 3
        col = index % 3
        if cfg.key == "radar":
            radar_mesh = ax.pcolormesh(
                x_edges,
                intensity_bins,
                heatmap,
                norm=mcolors.LogNorm(vmin=1, vmax=global_max),
                cmap=_rain_cmap(),
                shading="flat",
            )
            ax.set_title(cfg.label, fontsize=16, fontweight="bold", color="black")
        else:
            diff = heatmap - radar
            diff_mesh = ax.pcolormesh(
                x_edges,
                intensity_bins,
                diff,
                norm=mcolors.SymLogNorm(linthresh=10, linscale=1, vmin=-max_diff, vmax=max_diff),
                cmap="RdBu_r",
                shading="flat",
            )
            ax.set_title(f"{cfg.label} - Radar", fontsize=16, fontweight="bold", color="black")

            positive = np.maximum(diff, 0)
            levels_pos = [level for level in contour_levels_pos if level < float(np.nanmax(positive))]
            if levels_pos:
                contours = ax.contour(
                    x_centers,
                    y_centers,
                    positive,
                    levels=levels_pos,
                    colors="black",
                    linewidths=1.0,
                    linestyles="solid",
                    alpha=0.9,
                )
                ax.clabel(contours, inline=True, fmt="%d", fontsize=8, colors="black")

            levels_neg = [level for level in contour_levels_neg if level > float(np.nanmin(diff))]
            if levels_neg:
                contours = ax.contour(
                    x_centers,
                    y_centers,
                    diff,
                    levels=levels_neg,
                    colors="black",
                    linewidths=1.0,
                    linestyles="dashed",
                    alpha=0.9,
                )
                ax.clabel(contours, inline=True, fmt="%d", fontsize=8, colors="black")

        _panel_label(ax, index)
        _heatmap_axes_format(
            ax,
            intensity_bins,
            x_label=True,
            y_label=(col == 0),
        )

    fig.delaxes(axes_flat[5])
    if radar_mesh is None or diff_mesh is None:
        raise ValueError("Both Radar and comparison heatmaps are required")

    cax_radar = fig.add_axes([0.93, 0.55, 0.015, 0.35])
    cb_radar = fig.colorbar(radar_mesh, cax=cax_radar)
    cb_radar.set_label("Frequency", fontsize=14)
    cb_radar.ax.tick_params(labelsize=12)

    cax_diff = fig.add_axes([0.93, 0.15, 0.015, 0.35])
    cb_diff = fig.colorbar(diff_mesh, cax=cax_diff, extend="both")
    cb_diff.set_label(r"$\Delta$ Frequency", fontsize=14)
    cb_diff.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.10, wspace=0.30, hspace=0.35)
    save_figure(fig, output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_convective_stratiform(
    heatmaps: dict[str, dict[str, np.ndarray]],
    intensity_bins: np.ndarray,
    *,
    output_path: Path,
    dpi: int,
    vmax: float | None,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_edges = np.arange(25)
    if vmax is None:
        vmax = max(
            float(np.nanmax(heatmaps[cfg.key][kind]))
            for cfg in MODEL_DATASETS
            for kind in ("convective", "stratiform")
        )
    vmax = max(float(vmax), 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    image = None
    for col, cfg in enumerate(MODEL_DATASETS):
        for row, kind in enumerate(("convective", "stratiform")):
            ax = axes[row, col]
            image = ax.pcolormesh(
                x_edges,
                intensity_bins,
                heatmaps[cfg.key][kind],
                cmap="turbo",
                vmin=0.0,
                vmax=vmax,
                shading="flat",
            )
            kind_label = "Convective" if kind == "convective" else "Stratiform"
            ax.set_title(
                f"{cfg.label} - {kind_label} (Wet Only)",
                fontsize=13,
                fontweight="bold",
                color="black",
            )
            _panel_label(ax, row * len(MODEL_DATASETS) + col)
            _heatmap_axes_format(
                ax,
                intensity_bins,
                x_label=(row == 1),
                y_label=(col == 0),
            )

    if image is None:
        raise ValueError("No convective/stratiform heatmaps were supplied")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("Count", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    save_figure(fig, output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return vmax


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.10g}"


def write_intensity_evolution_txt(
    txt_path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    paths: dict[str, Path],
    heatmaps: dict[str, np.ndarray],
    intensity_bins: np.ndarray,
    wet_threshold: float,
    utc_offset_hours: int,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Intensity Evolution Pixel-Level Heatmap Data: {figure_path.stem}"
    radar = heatmaps["radar"]
    bin_centers = np.sqrt(intensity_bins[:-1] * intensity_bins[1:])
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: wet-pixel counts by local hour and intensity bin.\n")
        fh.write(f"Wet threshold: {wet_threshold:g} mm h^-1.\n")
        fh.write(f"Local time offset: UTC{utc_offset_hours:+d}.\n\n")

        fh.write("Source files\n")
        fh.write("------------\n")
        fh.write("dataset,variable,path\n")
        for cfg in DATASETS:
            fh.write(f"{cfg.label},{cfg.variable},{paths[cfg.key]}\n")

        fh.write("\nDataset summary\n")
        fh.write("---------------\n")
        fh.write("dataset,total_wet_pixel_count,max_bin_count,total_difference_vs_radar\n")
        for cfg in DATASETS:
            heatmap = heatmaps[cfg.key]
            diff_total = float(np.nansum(heatmap - radar)) if cfg.key != "radar" else 0.0
            fh.write(
                f"{cfg.label},{_fmt(float(np.nansum(heatmap)))},"
                f"{_fmt(float(np.nanmax(heatmap)))},{_fmt(diff_total)}\n"
            )

        fh.write("\nHeatmap data\n")
        fh.write("------------\n")
        fh.write(
            "dataset,hour_local,intensity_left_mm_h,intensity_right_mm_h,"
            "intensity_center_mm_h,count,difference_vs_radar\n"
        )
        for cfg in DATASETS:
            heatmap = heatmaps[cfg.key]
            diff = heatmap - radar if cfg.key != "radar" else np.zeros_like(heatmap)
            for bin_index in range(heatmap.shape[0]):
                for hour in range(24):
                    fh.write(
                        ",".join(
                            (
                                cfg.label,
                                str(hour),
                                _fmt(float(intensity_bins[bin_index])),
                                _fmt(float(intensity_bins[bin_index + 1])),
                                _fmt(float(bin_centers[bin_index])),
                                _fmt(float(heatmap[bin_index, hour])),
                                _fmt(float(diff[bin_index, hour])),
                            )
                        )
                        + "\n"
                    )


def write_convective_stratiform_txt(
    txt_path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    paths: dict[str, Path],
    heatmaps: dict[str, dict[str, np.ndarray]],
    intensity_bins: np.ndarray,
    wet_threshold: float,
    utc_offset_hours: int,
    vmax: float,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Convective/Stratiform Intensity-Hour Heatmap Data: {figure_path.stem}"
    bin_centers = np.sqrt(intensity_bins[:-1] * intensity_bins[1:])
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: wet-pixel counts by local hour and intensity bin for model rain components.\n")
        fh.write(f"Wet threshold: {wet_threshold:g} mm h^-1.\n")
        fh.write(f"Local time offset: UTC{utc_offset_hours:+d}.\n")
        fh.write(f"Plot colour range: 0 to {vmax:g} counts.\n\n")

        fh.write("Source files\n")
        fh.write("------------\n")
        fh.write("dataset,convective_variable,stratiform_variable,total_variable,path\n")
        for cfg in MODEL_DATASETS:
            fh.write(
                f"{cfg.label},{cfg.convective_variable},{cfg.stratiform_variable},"
                f"{cfg.total_variable},{paths[cfg.key]}\n"
            )

        fh.write("\nDataset summary\n")
        fh.write("---------------\n")
        fh.write("dataset,component,total_wet_pixel_count,max_bin_count\n")
        for cfg in MODEL_DATASETS:
            for component in ("convective", "stratiform"):
                heatmap = heatmaps[cfg.key][component]
                fh.write(
                    f"{cfg.label},{component},{_fmt(float(np.nansum(heatmap)))},"
                    f"{_fmt(float(np.nanmax(heatmap)))}\n"
                )

        fh.write("\nHeatmap data\n")
        fh.write("------------\n")
        fh.write(
            "dataset,component,hour_local,intensity_left_mm_h,intensity_right_mm_h,"
            "intensity_center_mm_h,count\n"
        )
        for cfg in MODEL_DATASETS:
            for component in ("convective", "stratiform"):
                heatmap = heatmaps[cfg.key][component]
                for bin_index in range(heatmap.shape[0]):
                    for hour in range(24):
                        fh.write(
                            ",".join(
                                (
                                    cfg.label,
                                    component,
                                    str(hour),
                                    _fmt(float(intensity_bins[bin_index])),
                                    _fmt(float(intensity_bins[bin_index + 1])),
                                    _fmt(float(bin_centers[bin_index])),
                                    _fmt(float(heatmap[bin_index, hour])),
                                )
                            )
                            + "\n"
                        )


def run(
    *,
    data_dir: Path,
    output_dir: Path,
    utc_offset_hours: int,
    wet_threshold: float,
    intensity_min: float,
    intensity_max: float,
    n_intensity_bins: int,
    convective_vmax: float | None,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    intensity_bins = np.logspace(
        np.log10(intensity_min),
        np.log10(intensity_max),
        n_intensity_bins + 1,
    )

    intensity_heatmaps, intensity_paths = read_intensity_heatmaps(
        data_dir,
        intensity_bins,
        wet_threshold=wet_threshold,
        utc_offset_hours=utc_offset_hours,
    )
    intensity_figure = output_dir / "intensity_evolution_pixel_level_both_contours.png"
    plot_intensity_evolution(
        intensity_heatmaps,
        intensity_bins,
        output_path=intensity_figure,
        dpi=dpi,
    )
    intensity_txt = txt_dir / f"{intensity_figure.stem}.txt"
    write_intensity_evolution_txt(
        intensity_txt,
        figure_path=intensity_figure,
        data_dir=data_dir,
        paths=intensity_paths,
        heatmaps=intensity_heatmaps,
        intensity_bins=intensity_bins,
        wet_threshold=wet_threshold,
        utc_offset_hours=utc_offset_hours,
    )
    print(f"fig: {intensity_figure}")
    print(f"txt: {intensity_txt}")

    conv_heatmaps, conv_paths = read_convective_stratiform_heatmaps(
        data_dir,
        intensity_bins,
        wet_threshold=wet_threshold,
        utc_offset_hours=utc_offset_hours,
    )
    conv_figure = output_dir / "convective_stratiform_intensity_hour_heatmap.png"
    used_vmax = plot_convective_stratiform(
        conv_heatmaps,
        intensity_bins,
        output_path=conv_figure,
        dpi=dpi,
        vmax=convective_vmax,
    )
    conv_txt = txt_dir / f"{conv_figure.stem}.txt"
    write_convective_stratiform_txt(
        conv_txt,
        figure_path=conv_figure,
        data_dir=data_dir,
        paths=conv_paths,
        heatmaps=conv_heatmaps,
        intensity_bins=intensity_bins,
        wet_threshold=wet_threshold,
        utc_offset_hours=utc_offset_hours,
        vmax=used_vmax,
    )
    print(f"fig: {conv_figure}")
    print(f"txt: {conv_txt}")
    return [intensity_figure, intensity_txt, conv_figure, conv_txt]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot rainfall intensity-hour heatmaps from common-valid rainfall data."
    )
    add_io_args(parser, default_data_dir=DEFAULT_DATA_DIR, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--utc-offset-hours", type=int, default=-4)
    parser.add_argument("--wet-threshold", type=float, default=0.1)
    parser.add_argument("--intensity-min", type=float, default=0.1)
    parser.add_argument("--intensity-max", type=float, default=150.0)
    parser.add_argument("--n-intensity-bins", type=int, default=29)
    parser.add_argument(
        "--convective-vmax",
        type=float,
        default=2000.0,
        help="Shared count colourbar maximum for the convective/stratiform heatmap; use 0 for auto.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    convective_vmax = None if args.convective_vmax == 0 else args.convective_vmax
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        utc_offset_hours=args.utc_offset_hours,
        wet_threshold=args.wet_threshold,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        n_intensity_bins=args.n_intensity_bins,
        convective_vmax=convective_vmax,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
