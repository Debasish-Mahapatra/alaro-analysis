"""Spatial precipitation relative-bias maps for common-valid rainfall data."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import RUNS_ROOT
from alaro_analysis.common.experiments import run_display, shown_runs
from alaro_analysis.common.figio import DatasetConfig, add_io_args, strip_cbar_zeros

DEFAULT_DATA_DIR = RUNS_ROOT / "rainfall-conservative-rebuild" / "common-valid-time-production"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "rainfall_spatial_bias_maps"
DEFAULT_MIN_REFERENCE = 0.01
MANAUS_LON = -60.0217
MANAUS_LAT = -3.1190
PANEL_LABELS = tuple(f"({letter})" for letter in "abcdefghijkl")


@dataclass(frozen=True)
class SpatialField:
    config: DatasetConfig
    source_path: Path
    lat: np.ndarray
    lon: np.ndarray
    values: np.ndarray
    n_time: int


_ALL_DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall"),
    DatasetConfig("imerg", "IMERG(GPM)", "IMERG_common_valid.nc", "rainfall"),
    DatasetConfig("nodiag", run_display("nodiag"), "C1M_nodiaggrp_common_valid.nc", "rainfall"),
    DatasetConfig("control", "C1M_diaggrp", "C1M_common_valid.nc", "rainfall"),
    DatasetConfig("graupel", "G1M", "G1M_common_valid.nc", "rainfall"),
    DatasetConfig("2mom", "G2M", "G2M_common_valid.nc", "rainfall"),
    DatasetConfig("no3m", "G2M-XCU", "G2M-XCU_common_valid.nc", "rainfall"),
)
# Without the run with the diagnostic graupel when the four-run set is drawn.
DATASETS = tuple(cfg for cfg in _ALL_DATASETS if shown_runs((cfg.key,)))
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


def build_rmse_maps(
    fields: dict[str, SpatialField],
    *,
    reference_key: str,
    comparison_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Per-cell RMSE (mm h-1) of the hourly rain against the reference, over the hours
    both have, read from the hourly files behind ``fields``."""

    def hourly(key: str) -> np.ndarray:
        field = fields[key]
        with xr.open_dataset(field.source_path) as ds:
            return np.asarray(ds[field.config.variable].values, dtype=np.float64)

    reference = hourly(reference_key)
    out = {}
    for key in comparison_keys:
        diff = hourly(key) - reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[key] = np.sqrt(np.nanmean(diff**2, axis=0))
    return out


def pooled_rmse(rmse_map: np.ndarray) -> float:
    """RMSE over all cells and hours (cells share the common-valid hours)."""
    return float(np.sqrt(np.nanmean(rmse_map**2)))


def write_rmse_txt(
    output_path: Path,
    *,
    figure_path: Path,
    fields: dict[str, SpatialField],
    rmse_maps: dict[str, np.ndarray],
    reference_key: str,
    vmax: float,
) -> None:
    """Data behind an RMSE map figure: per-panel summary, then every cell."""
    reference = fields[reference_key]
    lines = [
        f"RMSE Map Data: {figure_path.stem}",
        "=" * (15 + len(figure_path.stem)),
        f"Figure: {figure_path}",
        f"Reference dataset: {reference.config.label}",
        "Method: at each cell, sqrt(mean over the common-valid hours of (candidate - reference)^2),",
        "hourly rain in mm h^-1 on the 0.1 deg grid. Panel value: RMSE over all cells and hours.",
        f"Plot colour limits: 0, {vmax:g} mm h^-1.",
        "",
        "Panel summary",
        "-------------",
        "panel,candidate,reference,n_cells,reference_mean_mm_h,pooled_rmse_mm_h,"
        "mean_cell_rmse_mm_h,median_cell_rmse_mm_h,min_cell_rmse_mm_h,max_cell_rmse_mm_h",
    ]
    for index, (key, data) in enumerate(rmse_maps.items()):
        ok = np.isfinite(data)
        lines.append(",".join([
            PANEL_LABELS[index], fields[key].config.label, reference.config.label, str(int(ok.sum())),
            _fmt(np.nanmean(reference.values[ok])), _fmt(pooled_rmse(data)), _fmt(np.nanmean(data)),
            _fmt(np.nanmedian(data)), _fmt(np.nanmin(data)), _fmt(np.nanmax(data)),
        ]))
    lines += ["", "Cells", "-----", "lat,lon," + ",".join(fields[k].config.label for k in rmse_maps)]
    lon2d, lat2d = np.meshgrid(reference.lon, reference.lat)
    ok = np.any([np.isfinite(m) for m in rmse_maps.values()], axis=0)
    for jy, jx in zip(*np.nonzero(ok)):
        lines.append(f"{lat2d[jy, jx]:.4f},{lon2d[jy, jx]:.4f},"
                     + ",".join(_fmt(m[jy, jx]) for m in rmse_maps.values()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        fh.write(
            "Method: time mean at each radar-masked common-valid grid cell, then relative bias.\n"
        )
        fh.write("Formula: 100 * (candidate_mean - reference_mean) / reference_mean.\n")
        fh.write(
            f"Reference threshold: cells with reference_mean <= {min_reference:g} mm h^-1 are masked.\n"
        )
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


def _mean_label(ax, data: np.ndarray, text: str | None = None) -> None:
    ax.text(
        0.03,
        0.02,
        text if text is not None else f"Mean: {float(np.nanmean(data)):+.1f}%",
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
    ax.plot(
        MANAUS_LON,
        MANAUS_LAT,
        marker="*",
        markersize=10,
        color="black",
        transform=transform,
        zorder=10,
    )
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
    period_label: str | None = None,
    cmap: str = "RdBu_r",
    colorbar_label: str = "Relative Bias (%)",
    title_join: str = " - ",
    panel_note=None,
) -> None:
    """Maps of one statistic per dataset (relative bias by default); ``period_label``
    (e.g. "IOP1") is written under the colorbar, ``panel_note(data)`` gives each
    panel's corner text (default: the mean relative bias)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError("cartopy is required for spatial bias map plotting") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    rows, cols = layout
    n_panels = len(bias_maps)
    if n_panels < rows * cols:
        # An empty panel slot exists: drop the dedicated colorbar column and put a
        # narrow vertical colorbar into that slot, so both side margins trim away.
        grid = fig.add_gridspec(rows, cols, wspace=0.12, hspace=0.20 if rows > 1 else 0.08)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        axes_flat = [
            fig.add_subplot(grid[r, c], projection=projection) for (r, c) in slots[:n_panels]
        ]
        er, ec = slots[n_panels]
        holder = fig.add_subplot(grid[er, ec])
        holder.axis("off")
        if period_label:  # shorten the bar to leave room for the label under it
            cbar_ax = holder.inset_axes([0.46, 0.20, 0.07, 0.72])
            holder.text(0.5, 0.07, period_label, transform=holder.transAxes, ha="center",
                        va="center", fontsize=15, fontweight="bold")
        else:
            cbar_ax = holder.inset_axes([0.46, 0.08, 0.07, 0.84])
    else:
        grid = fig.add_gridspec(
            rows,
            cols + 1,
            width_ratios=[1.0] * cols + [0.05],
            wspace=0.15,
            hspace=0.20 if rows > 1 else 0.08,
        )
        axes_flat = [
            fig.add_subplot(grid[row, col], projection=projection)
            for row in range(rows)
            for col in range(cols)
        ]
        if period_label:  # shorten the bar to leave room for the label under it
            holder = fig.add_subplot(grid[:, -1])
            holder.axis("off")
            cbar_ax = holder.inset_axes([0.0, 0.10, 1.0, 0.86])
            holder.text(0.5, 0.025, period_label, transform=holder.transAxes, ha="center",
                        va="center", fontsize=15, fontweight="bold")
        else:
            cbar_ax = fig.add_subplot(grid[:, -1])
    image = None

    for index, (key, data) in enumerate(bias_maps.items()):
        ax = axes_flat[index]
        image = ax.pcolormesh(
            reference.lon,
            reference.lat,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=projection,
        )
        ax.set_title(
            f"{fields[key].config.label}{title_join}{title_suffix}",
            fontsize=16,
            fontweight="bold",
            color="black",
        )
        _add_map_context(ax, ccrs, cfeature)
        _add_manaus(ax, ccrs)
        _panel_label(ax, index)
        _mean_label(ax, data, panel_note(data) if panel_note else None)

    for ax in axes_flat[len(bias_maps) :]:
        fig.delaxes(ax)

    if image is None:
        raise ValueError("No bias maps were supplied for plotting")
    # The limit is a high percentile, so a few cells pass it: arrows say they are clipped.
    values = np.concatenate([np.ravel(data) for data in bias_maps.values()])
    values = values[np.isfinite(values)]
    over, under = bool((values > vmax).any()), bool((values < vmin).any())
    extend = {(True, True): "both", (True, False): "max", (False, True): "min"}.get((over, under), "neither")
    cbar = fig.colorbar(image, cax=cbar_ax, extend=extend)
    strip_cbar_zeros(cbar)
    cbar.set_label(colorbar_label, fontsize=14)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _extend(maps: Iterable[np.ndarray], vmin: float, vmax: float) -> str:
    """Colour-bar arrows for the values the limits clip."""
    values = np.concatenate([np.ravel(data) for data in maps])
    values = values[np.isfinite(values)]
    over, under = bool((values > vmax).any()), bool((values < vmin).any())
    return {(True, True): "both", (True, False): "max", (False, True): "min"}.get((over, under), "neither")


def plot_map_rows(
    fields: dict[str, SpatialField],
    rows: Sequence[dict],
    *,
    reference_key: str,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    period_label: str | None = None,
) -> None:
    """One row of maps per statistic, the datasets in columns, a colour bar per row.

    Each entry of ``rows`` holds ``maps`` (dataset key -> map), ``vmin``, ``vmax``, ``cmap``,
    ``colorbar_label`` and ``panel_note`` (map -> corner text). The column titles are the
    dataset names; panel letters run along the rows; ``period_label`` goes under the figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    projection = ccrs.PlateCarree()
    keys = list(rows[0]["maps"])
    nrow, ncol = len(rows), len(keys)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1.0] * ncol + [0.05], wspace=0.08, hspace=0.10)
    index = 0
    for r, row in enumerate(rows):
        image = None
        for c, key in enumerate(keys):
            ax = fig.add_subplot(grid[r, c], projection=projection)
            data = row["maps"][key]
            image = ax.pcolormesh(reference.lon, reference.lat, data, cmap=row["cmap"],
                                  vmin=row["vmin"], vmax=row["vmax"], transform=projection)
            ax.coastlines(resolution="10m", linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="blue")
            lines = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            lines.xlocator = matplotlib.ticker.FixedLocator([-61.0, -60.0, -59.0])  # narrow panels
            lines.top_labels = lines.right_labels = False
            lines.left_labels = c == 0
            lines.bottom_labels = r == nrow - 1
            if r == 0:
                ax.set_title(fields[key].config.label, fontsize=16, fontweight="bold", color="black")
            _add_manaus(ax, ccrs)
            _panel_label(ax, index)
            _mean_label(ax, data, row["panel_note"](data))
            index += 1
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[r, -1]),
                            extend=_extend(row["maps"].values(), row["vmin"], row["vmax"]))
        strip_cbar_zeros(cbar)
        cbar.set_label(row["colorbar_label"], fontsize=14)
    if period_label:
        fig.text(0.5, 0.06, period_label, ha="center", va="top", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
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
        fh.write(
            "panel,dataset,n_grid_cells,n_valid_cells,domain_mean_mm_h,median_mm_h,min_mm_h,max_mm_h\n"
        )
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

    for ax in axes[len(datasets_for_map) :]:
        fig.delaxes(ax)

    if image is None:
        raise ValueError("No spatial rainfall fields were supplied for plotting")
    cbar = fig.colorbar(image, cax=cbar_ax)
    strip_cbar_zeros(cbar)
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
            "comparison_keys": ("imerg", "control", "graupel", "2mom", "no3m"),
            "filename": "spatial_relative_bias_vs_radar.png",
            "title_suffix": "Radar",
            "figsize": (16.5, 10.5),
            "layout": (2, 3),
        },
        {
            "reference_key": "imerg",
            "comparison_keys": ("control", "graupel", "2mom", "no3m"),
            "filename": "spatial_relative_bias_vs_imerg.png",
            "title_suffix": "IMERG",
            "figsize": (12.0, 10.0),
            "layout": (2, 2),
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
    add_io_args(parser, default_data_dir=DEFAULT_DATA_DIR, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-reference", type=float, default=DEFAULT_MIN_REFERENCE)
    parser.add_argument("--vmin", type=float, default=-30.0)
    parser.add_argument("--vmax", type=float, default=30.0)
    return parse_configured_args(parser, "precip_spatial_bias_maps", argv=argv)


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


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
