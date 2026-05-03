"""Spatial precipitation relative-bias maps for common-valid rainfall data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
DEFAULT_DATA_DIR = (
    RUNS_ROOT
    / "rainfall-regridded-to-imerge"
    / "masked-production-final"
    / "common-valid-time-production"
)
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "rainfall_spatial_bias_maps"
DEFAULT_MIN_REFERENCE = 0.01
MANAUS_LON = -60.0217
MANAUS_LAT = -3.1190
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    filename: str
    variable: str


@dataclass(frozen=True)
class SpatialField:
    config: DatasetConfig
    source_path: Path
    lat: np.ndarray
    lon: np.ndarray
    values: np.ndarray
    n_time: int


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall_rate"),
    DatasetConfig("imerg", "IMERG(GPM)", "IMERG_common_valid.nc", "precipitation"),
    DatasetConfig("control", "C1M", "Control_common_valid.nc", "total_rain"),
    DatasetConfig("graupel", "G1M", "Graupel_common_valid.nc", "total_rain"),
    DatasetConfig("2mom", "G2M", "2-Moment_common_valid.nc", "total_rain"),
)
DATASET_BY_KEY = {cfg.key: cfg for cfg in DATASETS}


def calc_relative_bias_map(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    min_reference: float = DEFAULT_MIN_REFERENCE,
) -> np.ndarray:
    """Return 100 * (candidate - reference) / reference.

    Grid cells where the reference climatological mean is too small are masked
    so that tiny denominators do not dominate the spatial map.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate shape mismatch: {ref.shape} vs {cand.shape}")

    with np.errstate(divide="ignore", invalid="ignore"):
        bias = 100.0 * (cand - ref) / ref
    valid = np.isfinite(ref) & np.isfinite(cand) & (ref > min_reference)
    return np.where(valid, bias, np.nan)


def _coord_values(ds: xr.Dataset, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in ds.coords:
            return np.asarray(ds[name].values, dtype=np.float64)
        if name in ds:
            return np.asarray(ds[name].values, dtype=np.float64)
    raise KeyError(f"None of these coordinates were found: {tuple(names)}")


def read_spatial_mean(data_dir: Path, cfg: DatasetConfig) -> SpatialField:
    path = data_dir / cfg.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {cfg.label} file: {path}")

    with xr.open_dataset(path) as ds:
        if cfg.variable not in ds:
            raise KeyError(f"{cfg.variable!r} not found in {path}")
        da = ds[cfg.variable]
        if "time" not in da.dims:
            raise ValueError(f"{cfg.variable!r} in {path} has no time dimension")
        lat = _coord_values(ds, ("lat", "latitude"))
        lon = _coord_values(ds, ("lon", "longitude"))
        mean = da.mean(dim="time", skipna=True)
        n_time = int(ds.sizes["time"])
        values = np.asarray(mean.values, dtype=np.float64)

    return SpatialField(
        config=cfg,
        source_path=path,
        lat=lat,
        lon=lon,
        values=values,
        n_time=n_time,
    )


def read_all_spatial_means(data_dir: Path) -> dict[str, SpatialField]:
    fields = {cfg.key: read_spatial_mean(data_dir, cfg) for cfg in DATASETS}
    ref_shape = fields["radar"].values.shape
    for key, field in fields.items():
        if field.values.shape != ref_shape:
            raise ValueError(f"{field.config.label} shape differs from Radar: {field.values.shape}")
        if field.lat.shape != fields["radar"].lat.shape:
            raise ValueError(f"{field.config.label} latitude grid differs from Radar")
        if field.lon.shape != fields["radar"].lon.shape:
            raise ValueError(f"{field.config.label} longitude grid differs from Radar")
    return fields


def build_bias_maps(
    fields: dict[str, SpatialField],
    *,
    reference_key: str,
    comparison_keys: Sequence[str],
    min_reference: float,
) -> dict[str, np.ndarray]:
    reference = fields[reference_key].values
    return {
        key: calc_relative_bias_map(
            reference,
            fields[key].values,
            min_reference=min_reference,
        )
        for key in comparison_keys
    }


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.10g}"


def _summary_rows(
    fields: dict[str, SpatialField],
    bias_maps: dict[str, np.ndarray],
    *,
    reference_key: str,
) -> list[tuple[str, ...]]:
    ref = fields[reference_key]
    rows: list[tuple[str, ...]] = []
    for index, key in enumerate(bias_maps):
        candidate = fields[key]
        bias = bias_maps[key]
        finite = bias[np.isfinite(bias)]
        rows.append(
            (
                PANEL_LABELS[index],
                candidate.config.label,
                ref.config.label,
                str(int(np.size(bias))),
                str(int(finite.size)),
                _fmt(float(np.nanmean(ref.values))),
                _fmt(float(np.nanmean(candidate.values))),
                _fmt(float(np.nanmean(bias))),
                _fmt(float(np.nanmedian(bias))),
                _fmt(float(np.nanmin(bias))),
                _fmt(float(np.nanmax(bias))),
            )
        )
    return rows


def write_bias_txt(
    txt_path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    fields: dict[str, SpatialField],
    bias_maps: dict[str, np.ndarray],
    reference_key: str,
    min_reference: float,
    vmin: float,
    vmax: float,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    title = f"Spatial Relative Bias Map Data: {figure_path.stem}"
    lon2d, lat2d = np.meshgrid(reference.lon, reference.lat)

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write(f"Reference dataset: {reference.config.label}\n")
        fh.write("Method: time mean at each radar-masked common-valid grid cell, then relative bias.\n")
        fh.write("Formula: 100 * (candidate_mean - reference_mean) / reference_mean.\n")
        fh.write(f"Reference threshold: cells with reference_mean <= {min_reference:g} mm h^-1 are masked.\n")
        fh.write(f"Plot colour limits: {vmin:g}, {vmax:g} percent.\n\n")

        fh.write("Source files\n")
        fh.write("------------\n")
        fh.write("dataset,variable,n_time,path\n")
        for field in fields.values():
            fh.write(
                f"{field.config.label},{field.config.variable},{field.n_time},{field.source_path}\n"
            )

        fh.write("\nPanel summary\n")
        fh.write("-------------\n")
        fh.write(
            "panel,candidate,reference,n_grid_cells,n_valid_bias_cells,"
            "domain_mean_reference_mm_h,domain_mean_candidate_mm_h,"
            "mean_relative_bias_percent,median_relative_bias_percent,"
            "min_relative_bias_percent,max_relative_bias_percent\n"
        )
        for row in _summary_rows(fields, bias_maps, reference_key=reference_key):
            fh.write(",".join(row) + "\n")

        fh.write("\nGridpoint data\n")
        fh.write("--------------\n")
        fh.write(
            "panel,candidate,reference,lat,lon,"
            "reference_mean_mm_h,candidate_mean_mm_h,relative_bias_percent\n"
        )
        for index, key in enumerate(bias_maps):
            panel = PANEL_LABELS[index]
            candidate = fields[key]
            bias = bias_maps[key]
            for lat, lon, ref_value, cand_value, bias_value in zip(
                lat2d.ravel(),
                lon2d.ravel(),
                reference.values.ravel(),
                candidate.values.ravel(),
                bias.ravel(),
                strict=True,
            ):
                fh.write(
                    ",".join(
                        (
                            panel,
                            candidate.config.label,
                            reference.config.label,
                            _fmt(float(lat)),
                            _fmt(float(lon)),
                            _fmt(float(ref_value)),
                            _fmt(float(cand_value)),
                            _fmt(float(bias_value)),
                        )
                    )
                    + "\n"
                )


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


def _mean_label(ax, data: np.ndarray) -> None:
    ax.text(
        0.03,
        0.02,
        f"Mean: {float(np.nanmean(data)):+.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "boxstyle": "round,pad=0.25",
        },
        zorder=20,
    )


def _mean_rain_label(ax, data: np.ndarray) -> None:
    ax.text(
        0.03,
        0.02,
        f"Mean: {float(np.nanmean(data)):.2f} mm/hr",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "boxstyle": "round,pad=0.25",
        },
        zorder=20,
    )


def _add_map_context(ax, ccrs, cfeature) -> None:
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="blue")
    grid = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False


def _add_manaus(ax, ccrs) -> None:
    transform = ccrs.PlateCarree()
    ax.plot(MANAUS_LON, MANAUS_LAT, marker="*", markersize=10, color="black", transform=transform, zorder=10)
    ax.text(
        MANAUS_LON + 0.1,
        MANAUS_LAT + 0.1,
        "Manaus",
        transform=transform,
        fontsize=12,
        fontweight="bold",
        color="black",
        zorder=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.3, "edgecolor": "none"},
    )


def plot_bias_maps(
    fields: dict[str, SpatialField],
    bias_maps: dict[str, np.ndarray],
    *,
    reference_key: str,
    output_path: Path,
    title_suffix: str,
    figsize: tuple[float, float],
    layout: tuple[int, int],
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError("cartopy is required for spatial bias map plotting") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        layout[0],
        layout[1] + 1,
        width_ratios=[1.0] * layout[1] + [0.05],
        wspace=0.15,
        hspace=0.20 if layout[0] > 1 else 0.08,
    )
    axes_flat = [
        fig.add_subplot(grid[row, col], projection=projection)
        for row in range(layout[0])
        for col in range(layout[1])
    ]
    cbar_ax = fig.add_subplot(grid[:, -1])
    image = None

    for index, (key, data) in enumerate(bias_maps.items()):
        ax = axes_flat[index]
        image = ax.pcolormesh(
            reference.lon,
            reference.lat,
            data,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            transform=projection,
        )
        ax.set_title(
            f"{fields[key].config.label} - {title_suffix}",
            fontsize=16,
            fontweight="bold",
            color="black",
        )
        _add_map_context(ax, ccrs, cfeature)
        _add_manaus(ax, ccrs)
        _panel_label(ax, index)
        _mean_label(ax, data)

    for ax in axes_flat[len(bias_maps):]:
        fig.delaxes(ax)

    if image is None:
        raise ValueError("No bias maps were supplied for plotting")
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Relative Bias (%)", fontsize=14)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_spatial_mean_txt(
    txt_path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    fields: dict[str, SpatialField],
    vmin: float,
    vmax: float,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields["radar"]
    lon2d, lat2d = np.meshgrid(reference.lon, reference.lat)
    title = f"Spatial Mean Rainfall Map Data: {figure_path.stem}"

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: time mean at each radar-masked common-valid grid cell.\n")
        fh.write(f"Plot colour limits: {vmin:g}, {vmax:g} mm h^-1.\n\n")

        fh.write("Source files\n")
        fh.write("------------\n")
        fh.write("dataset,variable,n_time,path\n")
        for field in fields.values():
            fh.write(
                f"{field.config.label},{field.config.variable},{field.n_time},{field.source_path}\n"
            )

        fh.write("\nPanel summary\n")
        fh.write("-------------\n")
        fh.write("panel,dataset,n_grid_cells,n_valid_cells,domain_mean_mm_h,median_mm_h,min_mm_h,max_mm_h\n")
        for index, field in enumerate(fields.values()):
            values = field.values
            finite = values[np.isfinite(values)]
            fh.write(
                ",".join(
                    (
                        PANEL_LABELS[index],
                        field.config.label,
                        str(int(values.size)),
                        str(int(finite.size)),
                        _fmt(float(np.nanmean(values))),
                        _fmt(float(np.nanmedian(values))),
                        _fmt(float(np.nanmin(values))),
                        _fmt(float(np.nanmax(values))),
                    )
                )
                + "\n"
            )

        fh.write("\nGridpoint data\n")
        fh.write("--------------\n")
        fh.write("panel,dataset,lat,lon,mean_rainfall_mm_h\n")
        for index, field in enumerate(fields.values()):
            panel = PANEL_LABELS[index]
            for lat, lon, value in zip(
                lat2d.ravel(),
                lon2d.ravel(),
                field.values.ravel(),
                strict=True,
            ):
                fh.write(
                    ",".join(
                        (
                            panel,
                            field.config.label,
                            _fmt(float(lat)),
                            _fmt(float(lon)),
                            _fmt(float(value)),
                        )
                    )
                    + "\n"
                )


def plot_spatial_mean_maps(
    fields: dict[str, SpatialField],
    *,
    output_path: Path,
    dpi: int,
) -> tuple[float, float]:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError("cartopy is required for spatial rainfall map plotting") from exc

    try:
        import cmaps

        cmap = cmaps.WhiteBlueGreenYellowRed
    except ImportError:  # pragma: no cover - fallback for minimal environments.
        cmap = "turbo"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets_for_map = list(fields.values())
    stack = np.stack([field.values for field in datasets_for_map], axis=0)
    vmin = 0.0
    vmax = float(np.nanmax(stack))

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15.0, 8.0))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.05],
        wspace=0.15,
        hspace=0.20,
    )
    axes = [
        fig.add_subplot(grid[row, col], projection=projection)
        for row in range(2)
        for col in range(3)
    ]
    cbar_ax = fig.add_subplot(grid[:, -1])
    image = None

    for index, field in enumerate(datasets_for_map):
        ax = axes[index]
        image = ax.pcolormesh(
            field.lon,
            field.lat,
            field.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=projection,
        )
        ax.set_title(field.config.label, fontsize=16, fontweight="bold", color="black")
        _add_map_context(ax, ccrs, cfeature)
        _add_manaus(ax, ccrs)
        _panel_label(ax, index)
        _mean_rain_label(ax, field.values)

    for ax in axes[len(datasets_for_map):]:
        fig.delaxes(ax)

    if image is None:
        raise ValueError("No spatial rainfall fields were supplied for plotting")
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label(r"Mean Rainfall (mm h$^{-1}$)", fontsize=14)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return vmin, vmax


def run(
    *,
    data_dir: Path,
    output_dir: Path,
    min_reference: float,
    vmin: float,
    vmax: float,
    dpi: int,
) -> list[Path]:
    fields = read_all_spatial_means(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    written: list[Path] = []

    mean_figure_path = output_dir / "spatial_mean_rainfall_maps.png"
    rain_vmin, rain_vmax = plot_spatial_mean_maps(
        fields,
        output_path=mean_figure_path,
        dpi=dpi,
    )
    mean_txt_path = txt_dir / f"{mean_figure_path.stem}.txt"
    write_spatial_mean_txt(
        mean_txt_path,
        figure_path=mean_figure_path,
        data_dir=data_dir,
        fields=fields,
        vmin=rain_vmin,
        vmax=rain_vmax,
    )
    written.extend([mean_figure_path, mean_txt_path])
    print(f"fig: {mean_figure_path}")
    print(f"txt: {mean_txt_path}")

    plot_specs = (
        {
            "reference_key": "radar",
            "comparison_keys": ("imerg", "control", "graupel", "2mom"),
            "filename": "spatial_relative_bias_vs_radar.png",
            "title_suffix": "Radar",
            "figsize": (12.0, 10.0),
            "layout": (2, 2),
        },
        {
            "reference_key": "imerg",
            "comparison_keys": ("control", "graupel", "2mom"),
            "filename": "spatial_relative_bias_vs_imerg.png",
            "title_suffix": "IMERG(GPM)",
            "figsize": (15.0, 5.0),
            "layout": (1, 3),
        },
    )

    for spec in plot_specs:
        bias_maps = build_bias_maps(
            fields,
            reference_key=str(spec["reference_key"]),
            comparison_keys=tuple(spec["comparison_keys"]),
            min_reference=min_reference,
        )
        figure_path = output_dir / str(spec["filename"])
        plot_bias_maps(
            fields,
            bias_maps,
            reference_key=str(spec["reference_key"]),
            output_path=figure_path,
            title_suffix=str(spec["title_suffix"]),
            figsize=spec["figsize"],
            layout=spec["layout"],
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )
        txt_path = txt_dir / f"{figure_path.stem}.txt"
        write_bias_txt(
            txt_path,
            figure_path=figure_path,
            data_dir=data_dir,
            fields=fields,
            bias_maps=bias_maps,
            reference_key=str(spec["reference_key"]),
            min_reference=min_reference,
            vmin=vmin,
            vmax=vmax,
        )
        written.extend([figure_path, txt_path])
        print(f"fig: {figure_path}")
        print(f"txt: {txt_path}")
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spatial relative-bias maps from common-valid rainfall data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-reference", type=float, default=DEFAULT_MIN_REFERENCE)
    parser.add_argument("--vmin", type=float, default=-30.0)
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_reference=args.min_reference,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
