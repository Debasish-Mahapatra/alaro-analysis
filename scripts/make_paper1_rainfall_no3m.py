"""Make paper1 rainfall figures with the NO3M/G2M-XCU experiment included.

This script is intentionally self-contained because the existing reusable
workflows describe the standard three model experiments.  It creates the
missing NO3M common-valid rainfall file on the same IMERG/radar grid, then
uses the same common-valid mask/time logic for the paper figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import shutil
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import netCDF4
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import xarray as xr


from alaro_analysis.common.constants import RUNS_ROOT
RAINFALL_ROOT = RUNS_ROOT / "rainfall-regridded-to-imerge"
BASE_COMMON_DIR = (
    RAINFALL_ROOT
    / "masked-production-final-hourly-imerg"
    / "common-valid-time-production"
)
WORK_ROOT = RAINFALL_ROOT / "masked-production-final-hourly-imerg-no3m"
COMBINED_COMMON_DIR = WORK_ROOT / "common-valid-time-production"
NO3M_ACCUM_ROOT = RUNS_ROOT / "ALARO" / "NO3M" / "masked-netcdf"
OUTPUT_DIR = RUNS_ROOT / "Analysis" / "figures" / "paper1_rainfall_no3m"
PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")

MANAUS_LON = -60.0217
MANAUS_LAT = -3.1190
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    filename: str
    variable: str
    color: str = "black"
    linestyle: str = "-"
    linewidth: float = 3.0


@dataclass(frozen=True)
class SpatialField:
    config: DatasetConfig
    source_path: Path
    lat: np.ndarray
    lon: np.ndarray
    values: np.ndarray
    n_time: int


@dataclass(frozen=True)
class HourlyStats:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class SampleSet:
    config: DatasetConfig
    source_path: Path
    values: np.ndarray
    total_values: int

    @property
    def n_valid(self) -> int:
        return int(self.values.size)

    @property
    def n_positive(self) -> int:
        return int(np.count_nonzero(self.values > 0.0))

    def n_ge(self, threshold: float) -> int:
        return int(np.count_nonzero(self.values >= threshold))

    def n_gt(self, threshold: float) -> int:
        return int(np.count_nonzero(self.values > threshold))


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall_rate", "black", "-", 3.2),
    DatasetConfig("imerg", "IMERG", "IMERG_common_valid.nc", "precipitation", "dimgray", ":", 3.6),
    DatasetConfig("control", "C1M", "Control_common_valid.nc", "total_rain", "#d62728", "-", 3.0),
    DatasetConfig("graupel", "G1M", "Graupel_common_valid.nc", "total_rain", "#1f77b4", "-", 3.0),
    DatasetConfig("2mom", "G2M", "2-Moment_common_valid.nc", "total_rain", "#2ca02c", "-", 3.0),
    DatasetConfig("no3m", "G2M-XCU", "NO3M_common_valid.nc", "total_rain", "#9467bd", "-", 3.0),
)
DATASET_BY_KEY = {cfg.key: cfg for cfg in DATASETS}
OBS_SHADE_KEYS = ("radar", "imerg")
SHADE_SETTINGS = {
    "radar": {"fill_color": "deepskyblue", "alpha": 0.30},
    "imerg": {"fill_color": "lightgrey", "alpha": 0.40},
}


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.10g}"


def netcdf_encoding(ds: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, da in ds.data_vars.items():
        item: dict[str, object] = {"zlib": True, "complevel": 4}
        if np.issubdtype(da.dtype, np.floating):
            item["_FillValue"] = np.nan
        encoding[name] = item
    return encoding


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def time_to_day_hour(value: np.datetime64) -> tuple[str, int]:
    text = str(np.datetime64(value, "h"))
    return text[:10].replace("-", ""), int(text[11:13])


def no3m_accumulated_path(root: Path, variable: str, day: str, lead_hour: int) -> Path:
    return root / variable / f"pf{day}" / f"pfABOFABOF+{lead_hour:04d}.nc"


def no3m_increment_paths(root: Path, variable: str, value: np.datetime64) -> tuple[Path, Path]:
    day, hour = time_to_day_hour(value)
    return (
        no3m_accumulated_path(root, variable, day, hour + 1),
        no3m_accumulated_path(root, variable, day, hour),
    )


def build_regrid_weights(
    sample_path: Path,
    *,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    margin_deg: float,
) -> dict[str, object]:
    with xr.open_dataset(sample_path) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)

    target_lon2d, target_lat2d = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])
    crop = (
        (lon >= float(np.nanmin(target_lons)) - margin_deg)
        & (lon <= float(np.nanmax(target_lons)) + margin_deg)
        & (lat >= float(np.nanmin(target_lats)) - margin_deg)
        & (lat <= float(np.nanmax(target_lats)) + margin_deg)
    )
    ys, xs = np.where(crop)
    if ys.size == 0:
        raise ValueError("NO3M source crop does not overlap the target grid.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_slice_mask = crop[y0:y1, x0:x1]
    source_points = np.column_stack(
        [
            lon[y0:y1, x0:x1][crop_slice_mask],
            lat[y0:y1, x0:x1][crop_slice_mask],
        ]
    )
    triangulation = Delaunay(source_points)
    simplex = triangulation.find_simplex(target_points)
    tree = cKDTree(source_points)
    _, nearest = tree.query(target_points)

    vertices = np.full((target_points.shape[0], 3), -1, dtype=np.int64)
    weights = np.full((target_points.shape[0], 3), np.nan, dtype=np.float64)
    linear_mask = simplex >= 0
    if np.any(linear_mask):
        transform = triangulation.transform[simplex[linear_mask]]
        delta = target_points[linear_mask] - transform[:, 2]
        bary = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
        weights[linear_mask, :2] = bary
        weights[linear_mask, 2] = 1.0 - bary.sum(axis=1)
        vertices[linear_mask] = triangulation.simplices[simplex[linear_mask]]

    return {
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
        "crop_slice_mask": crop_slice_mask,
        "vertices": vertices,
        "weights": weights,
        "nearest": nearest.astype(np.int64),
        "target_shape": (len(target_lats), len(target_lons)),
        "source_point_count": int(source_points.shape[0]),
        "linear_target_count": int(np.count_nonzero(linear_mask)),
    }


def read_accumulated_source_values(path: Path, variable: str, weights: dict[str, object]) -> np.ndarray:
    y0 = int(weights["y0"])
    y1 = int(weights["y1"])
    x0 = int(weights["x0"])
    x1 = int(weights["x1"])
    crop_slice_mask = weights["crop_slice_mask"]
    with netCDF4.Dataset(path) as ds:
        raw = np.ma.array(ds.variables[variable][0, y0:y1, x0:x1]).filled(np.nan)
    arr = np.asarray(raw, dtype=np.float32)
    return arr[crop_slice_mask]


def read_increment_source_values(
    root: Path,
    variable: str,
    timestamp: np.datetime64,
    weights: dict[str, object],
) -> np.ndarray:
    current_path, previous_path = no3m_increment_paths(root, variable, timestamp)
    if not current_path.exists():
        raise FileNotFoundError(current_path)
    if not previous_path.exists():
        raise FileNotFoundError(previous_path)
    current = read_accumulated_source_values(current_path, variable, weights)
    previous = read_accumulated_source_values(previous_path, variable, weights)
    increment = current - previous
    return np.where(np.isfinite(increment) & (increment > 0.0), increment, 0.0).astype(np.float32)


def interpolate_to_target(source_values: np.ndarray, weights: dict[str, object]) -> np.ndarray:
    vertices = weights["vertices"]
    bary_weights = weights["weights"]
    nearest = weights["nearest"]
    target_shape = weights["target_shape"]
    out = source_values[nearest].astype(np.float64)
    linear = vertices[:, 0] >= 0
    if np.any(linear):
        vertex_values = source_values[vertices[linear]]
        linear_values = np.einsum("ij,ij->i", vertex_values, bary_weights[linear])
        finite = np.isfinite(vertex_values).all(axis=1) & np.isfinite(linear_values)
        linear_indices = np.where(linear)[0]
        out[linear_indices[finite]] = linear_values[finite]
    return out.reshape(target_shape).astype(np.float32)


def build_no3m_common_valid(
    *,
    base_common_dir: Path,
    no3m_root: Path,
    output_path: Path,
    force: bool,
    margin_deg: float,
    progress_interval: int,
) -> Path:
    if output_path.exists() and not force:
        print(f"NO3M common-valid already exists: {output_path}", flush=True)
        return output_path

    template_path = base_common_dir / "Radar_common_valid.nc"
    with xr.open_dataset(template_path) as template:
        template = template.load()
    times = np.asarray(template["time"].values, dtype="datetime64[ns]")
    target_lats = np.asarray(template["lat"].values, dtype=np.float64)
    target_lons = np.asarray(template["lon"].values, dtype=np.float64)
    radar_mask = np.asarray(template["radar_mask"].values, dtype=bool)

    sample_path, _ = no3m_increment_paths(no3m_root, "SURFPREC.EAU.CON", times[0])
    weights = build_regrid_weights(
        sample_path,
        target_lats=target_lats,
        target_lons=target_lons,
        margin_deg=margin_deg,
    )
    print(
        "NO3M interpolation crop: "
        f"{weights['source_point_count']} source points, "
        f"{weights['linear_target_count']} linear target points",
        flush=True,
    )

    shape = (len(times), len(target_lats), len(target_lons))
    convective = np.full(shape, np.nan, dtype=np.float32)
    stratiform = np.full(shape, np.nan, dtype=np.float32)
    missing: list[str] = []
    for index, timestamp in enumerate(times):
        try:
            con_values = read_increment_source_values(no3m_root, "SURFPREC.EAU.CON", timestamp, weights)
            gec_values = read_increment_source_values(no3m_root, "SURFPREC.EAU.GEC", timestamp, weights)
        except FileNotFoundError:
            missing.append(str(np.datetime64(timestamp, "h")))
            continue
        con_target = interpolate_to_target(con_values, weights)
        gec_target = interpolate_to_target(gec_values, weights)
        con_target = np.where(radar_mask, con_target, np.nan)
        gec_target = np.where(radar_mask, gec_target, np.nan)
        convective[index] = con_target
        stratiform[index] = gec_target
        if progress_interval > 0 and (
            index == 0 or (index + 1) % progress_interval == 0 or index + 1 == len(times)
        ):
            print(f"NO3M regrid: {index + 1}/{len(times)} hours", flush=True)

    total = convective + stratiform
    ds = xr.Dataset(
        data_vars={
            "stratiform_rain": (
                ("time", "lat", "lon"),
                stratiform,
                {"units": "mm/h", "long_name": "NO3M stratiform rainfall regridded to IMERG grid"},
            ),
            "convective_rain": (
                ("time", "lat", "lon"),
                convective,
                {"units": "mm/h", "long_name": "NO3M convective rainfall regridded to IMERG grid"},
            ),
            "total_rain": (
                ("time", "lat", "lon"),
                total.astype(np.float32),
                {"units": "mm/h", "long_name": "NO3M total rainfall regridded to IMERG grid"},
            ),
            "radar_mask": (("lat", "lon"), radar_mask.astype(bool)),
        },
        coords={
            "time": times,
            "lat": template["lat"],
            "lon": template["lon"],
        },
        attrs={
            "title": "NO3M/G2M-XCU rainfall on common-valid IMERG grid",
            "source": str(no3m_root),
            "method": "Hourly increments from accumulated SURFPREC.EAU.CON and SURFPREC.EAU.GEC using lead h+1 minus lead h for each UTC hour h, assigned to the interval start time, linearly interpolated to the existing IMERG grid, then masked with the radar common-valid spatial mask.",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "missing_source_hours": str(len(missing)),
        },
    )
    for name in ("lat_bnds", "lon_bnds"):
        if name in template:
            ds[name] = template[name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, encoding=netcdf_encoding(ds))
    print(f"wrote NO3M common-valid: {output_path}", flush=True)
    if missing:
        missing_path = output_path.with_suffix(".missing_hours.txt")
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"NO3M missing hours listed in: {missing_path}", flush=True)
    return output_path


def prepare_combined_common_dir(base_common_dir: Path, no3m_path_out: Path, combined_dir: Path) -> Path:
    combined_dir.mkdir(parents=True, exist_ok=True)
    for cfg in DATASETS:
        if cfg.key == "no3m":
            src = no3m_path_out
        else:
            src = base_common_dir / cfg.filename
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, combined_dir / cfg.filename)
    return combined_dir


def read_spatial_mean(data_dir: Path, cfg: DatasetConfig) -> SpatialField:
    path = data_dir / cfg.filename
    with xr.open_dataset(path) as ds:
        da = ds[cfg.variable]
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        mean = da.mean(dim="time", skipna=True)
        values = np.asarray(mean.values, dtype=np.float64)
        n_time = int(ds.sizes["time"])
    return SpatialField(cfg, path, lat, lon, values, n_time)


def read_all_spatial_means(data_dir: Path) -> dict[str, SpatialField]:
    fields = {cfg.key: read_spatial_mean(data_dir, cfg) for cfg in DATASETS}
    ref = fields["radar"]
    for field in fields.values():
        if field.values.shape != ref.values.shape:
            raise ValueError(f"{field.config.label} shape mismatch: {field.values.shape}")
    return fields


def calc_relative_bias_map(reference: np.ndarray, candidate: np.ndarray, min_reference: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = 100.0 * (candidate - reference) / reference
    valid = np.isfinite(reference) & np.isfinite(candidate) & (reference > min_reference)
    return np.where(valid, bias, np.nan)


def build_bias_maps(
    fields: dict[str, SpatialField],
    *,
    reference_key: str,
    comparison_keys: Sequence[str],
    min_reference: float,
) -> dict[str, np.ndarray]:
    reference = fields[reference_key].values
    return {
        key: calc_relative_bias_map(reference, fields[key].values, min_reference)
        for key in comparison_keys
    }


def _panel_label(ax, index: int) -> None:
    ax.text(
        0.03,
        0.97,
        PANEL_LABELS[index],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "boxstyle": "round,pad=0.18"},
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
        fontsize=8,
        fontweight="normal",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "boxstyle": "round,pad=0.25"},
        zorder=20,
    )


def add_map_context(ax, ccrs, cfeature) -> None:
    try:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="blue")
    except Exception as exc:
        print(f"map feature warning: {exc}", flush=True)
    grid = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    grid.xlabel_style = {"size": 8}
    grid.ylabel_style = {"size": 8}


def add_manaus(ax, ccrs) -> None:
    transform = ccrs.PlateCarree()
    ax.plot(MANAUS_LON, MANAUS_LAT, marker="*", markersize=10, color="black", transform=transform, zorder=10)
    ax.text(
        MANAUS_LON + 0.1,
        MANAUS_LAT + 0.1,
        "Manaus",
        transform=transform,
        fontsize=9,
        fontweight="normal",
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
    vmin: float,
    vmax: float,
    dpi: int,
    cbar_mode: str,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    projection = ccrs.PlateCarree()

    if cbar_mode == "sixth-slot":
        fig = plt.figure(figsize=(15.0, 8.6))
        grid = fig.add_gridspec(2, 3, wspace=0.13, hspace=0.16)
        axes = [
            fig.add_subplot(grid[row, col], projection=projection)
            for row in range(2)
            for col in range(3)
        ]
        cbar_ax = axes[-1]
        fig.delaxes(cbar_ax)
        cbar_ax = fig.add_subplot(grid[1, 2])
    elif cbar_mode == "right":
        fig = plt.figure(figsize=(12.0, 8.4))
        grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.025], wspace=0.13, hspace=0.16)
        axes = [
            fig.add_subplot(grid[row, col], projection=projection)
            for row in range(2)
            for col in range(2)
        ]
        cbar_ax = fig.add_subplot(grid[:, 2])
    else:
        raise ValueError(f"Unsupported cbar mode: {cbar_mode}")

    image = None
    for index, (key, data) in enumerate(bias_maps.items()):
        ax = axes[index]
        image = ax.pcolormesh(
            reference.lon,
            reference.lat,
            data,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            transform=projection,
        )
        ax.set_title(f"{fields[key].config.label} - {title_suffix}", fontsize=10, fontweight="normal")
        add_map_context(ax, ccrs, cfeature)
        add_manaus(ax, ccrs)
        _panel_label(ax, index)
        _mean_label(ax, data)

    if cbar_mode != "sixth-slot":
        for ax in axes[len(bias_maps):]:
            fig.delaxes(ax)
    if image is None:
        raise ValueError("No bias maps supplied.")
    if cbar_mode == "sixth-slot":
        cbar_ax.axis("off")
        bar_ax = cbar_ax.inset_axes([0.14, 0.47, 0.72, 0.07])
        cbar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
    else:
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Relative Bias (%)", fontsize=9)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


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
    layout_note: str,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    reference = fields[reference_key]
    lon2d, lat2d = np.meshgrid(reference.lon, reference.lat)
    title = f"Spatial Relative Bias Map Data: {figure_path.stem}"
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write(f"Reference dataset: {reference.config.label}\n")
        fh.write("Method: time mean at each radar-masked common-valid grid cell, then relative bias.\n")
        fh.write("Formula: 100 * (candidate_mean - reference_mean) / reference_mean.\n")
        fh.write(f"Reference threshold: cells with reference_mean <= {min_reference:g} mm h^-1 are masked.\n")
        fh.write(f"Plot colour limits: {vmin:g}, {vmax:g} percent.\n")
        fh.write(f"Layout: {layout_note}\n\n")
        fh.write("Source files\n")
        fh.write("------------\n")
        fh.write("dataset,variable,n_time,path\n")
        for field in fields.values():
            fh.write(f"{field.config.label},{field.config.variable},{field.n_time},{field.source_path}\n")
        fh.write("\nPanel summary\n")
        fh.write("-------------\n")
        fh.write(
            "panel,candidate,reference,n_grid_cells,n_valid_bias_cells,"
            "domain_mean_reference_mm_h,domain_mean_candidate_mm_h,"
            "mean_relative_bias_percent,median_relative_bias_percent,"
            "min_relative_bias_percent,max_relative_bias_percent\n"
        )
        for index, key in enumerate(bias_maps):
            candidate = fields[key]
            bias = bias_maps[key]
            finite = bias[np.isfinite(bias)]
            row = (
                PANEL_LABELS[index],
                candidate.config.label,
                reference.config.label,
                str(int(np.size(bias))),
                str(int(finite.size)),
                _fmt(float(np.nanmean(reference.values))),
                _fmt(float(np.nanmean(candidate.values))),
                _fmt(float(np.nanmean(bias))),
                _fmt(float(np.nanmedian(bias))),
                _fmt(float(np.nanmin(bias))),
                _fmt(float(np.nanmax(bias))),
            )
            fh.write(",".join(row) + "\n")
        fh.write("\nGridpoint data\n")
        fh.write("--------------\n")
        fh.write("panel,candidate,reference,lat,lon,reference_mean_mm_h,candidate_mean_mm_h,relative_bias_percent\n")
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


def local_hours_from_utc(times: np.ndarray, utc_offset_hours: int) -> np.ndarray:
    utc = np.asarray(times, dtype="datetime64[ns]")
    local = utc + np.timedelta64(int(utc_offset_hours), "h")
    return (local.astype("datetime64[h]").astype(np.int64) % 24).astype(np.int16)


def compute_hourly_stats(values: np.ndarray, local_hours: np.ndarray) -> HourlyStats:
    vals = np.asarray(values, dtype=np.float64)
    hrs = np.asarray(local_hours, dtype=np.int16)
    means = np.full(24, np.nan, dtype=np.float64)
    stds = np.full(24, np.nan, dtype=np.float64)
    counts = np.zeros(24, dtype=np.int64)
    for hour in range(24):
        selected = vals[hrs == hour]
        selected = selected[np.isfinite(selected)]
        counts[hour] = selected.size
        if selected.size:
            means[hour] = float(np.mean(selected))
        if selected.size > 1:
            stds[hour] = float(np.std(selected, ddof=1))
    return HourlyStats(mean=means, std=stds, count=counts)


def read_domain_mean_stats(data_dir: Path, cfg: DatasetConfig, utc_offset_hours: int) -> HourlyStats:
    with xr.open_dataset(data_dir / cfg.filename) as ds:
        da = ds[cfg.variable]
        space_dims = [dim for dim in da.dims if dim != "time"]
        domain_mean = da.mean(dim=space_dims, skipna=True)
        values = np.asarray(domain_mean.values, dtype=np.float64)
        hours = local_hours_from_utc(ds["time"].values, utc_offset_hours)
    return compute_hourly_stats(values, hours)


def shading_bounds(stats: HourlyStats, *, mode: str, std_multiplier: float, percent_uncertainty: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "std":
        spread = std_multiplier * stats.std
        lower = stats.mean - spread
        upper = stats.mean + spread
    elif mode == "percent":
        lower = stats.mean * (1.0 - percent_uncertainty)
        upper = stats.mean * (1.0 + percent_uncertainty)
    else:
        raise ValueError(mode)
    return np.clip(lower, 0.0, None), np.clip(upper, 0.0, None)


def plot_diurnal_cycle(
    stats_by_key: dict[str, HourlyStats],
    *,
    output_path: Path,
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
    dpi: int,
) -> None:
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    shade_handles: list[Patch] = []
    for cfg in DATASETS:
        stats = stats_by_key[cfg.key]
        if cfg.key in OBS_SHADE_KEYS:
            lower, upper = shading_bounds(
                stats,
                mode=shade_mode,
                std_multiplier=std_multiplier,
                percent_uncertainty=percent_uncertainty,
            )
            setting = SHADE_SETTINGS[cfg.key]
            ax.fill_between(hours, lower, upper, color=setting["fill_color"], alpha=setting["alpha"], linewidth=0, zorder=1)
            shade_label = (
                f"{cfg.label} +/- {std_multiplier:g} sigma"
                if shade_mode == "std"
                else f"{cfg.label} +/- {100.0 * percent_uncertainty:g}%"
            )
            shade_handles.append(Patch(facecolor=setting["fill_color"], edgecolor="none", alpha=setting["alpha"], label=shade_label))
        ax.plot(
            hours,
            stats.mean,
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=cfg.linewidth,
            zorder=3,
        )
    ax.set_ylabel(r"Rainfall rate (mm h$^{-1}$)", fontsize=18)
    ax.set_xlabel("Local time (UTC-4)", fontsize=18)
    ax.set_xticks(hours)
    ax.set_xlim(0, 23)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.3)
    line_legend = ax.legend(loc="upper left", fontsize=15, framealpha=0.9)
    ax.add_artist(line_legend)
    ax.legend(handles=shade_handles, loc="upper right", fontsize=13, framealpha=0.9, title="Shaded uncertainty", title_fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_diurnal_txt(
    txt_path: Path,
    *,
    data_dir: Path,
    stats_by_key: dict[str, HourlyStats],
    shade_mode: str,
    std_multiplier: float,
    percent_uncertainty: float,
    utc_offset_hours: int,
    output_path: Path,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation Diurnal Cycle Plot Data\n")
        fh.write("=====================================\n")
        fh.write(f"Source plot: {output_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: spatial mean over the common-valid radar mask, then mean by local hour.\n")
        fh.write(f"Local time offset: UTC{utc_offset_hours:+d}\n")
        fh.write(f"Shading mode: {shade_mode}\n")
        fh.write(f"Standard deviation multiplier: {std_multiplier:g}\n")
        fh.write(f"Percent uncertainty: {percent_uncertainty:g}\n")
        fh.write("Shaded datasets: Radar, IMERG\n\n")
        fh.write("Source files\n")
        fh.write("dataset,variable,path\n")
        for cfg in DATASETS:
            fh.write(f"{cfg.label},{cfg.variable},{data_dir / cfg.filename}\n")
        fh.write("\nDataset summary\n")
        fh.write("dataset,total_hourly_samples,min_count_per_hour,max_count_per_hour,daily_sum_mm_day\n")
        for cfg in DATASETS:
            stats = stats_by_key[cfg.key]
            fh.write(
                f"{cfg.label},{int(stats.count.sum())},{int(stats.count.min())},"
                f"{int(stats.count.max())},{float(np.nansum(stats.mean)):.10g}\n"
            )
        fh.write("\nHourly plotted data\n")
        columns = ["local_hour"]
        for cfg in DATASETS:
            columns.extend(
                [
                    f"{cfg.label}_mean_mm_h",
                    f"{cfg.label}_std_mm_h",
                    f"{cfg.label}_count",
                    f"{cfg.label}_shade_lower",
                    f"{cfg.label}_shade_upper",
                ]
            )
        fh.write(",".join(columns) + "\n")
        for hour in range(24):
            row = [str(hour)]
            for cfg in DATASETS:
                stats = stats_by_key[cfg.key]
                if cfg.key in OBS_SHADE_KEYS:
                    lower, upper = shading_bounds(
                        stats,
                        mode=shade_mode,
                        std_multiplier=std_multiplier,
                        percent_uncertainty=percent_uncertainty,
                    )
                else:
                    lower = np.full(24, np.nan)
                    upper = np.full(24, np.nan)
                row.extend(
                    [
                        f"{stats.mean[hour]:.12g}",
                        f"{stats.std[hour]:.12g}",
                        str(int(stats.count[hour])),
                        f"{lower[hour]:.12g}",
                        f"{upper[hour]:.12g}",
                    ]
                )
            fh.write(",".join(row) + "\n")


def read_samples(data_dir: Path) -> list[SampleSet]:
    samples: list[SampleSet] = []
    for cfg in DATASETS:
        path = data_dir / cfg.filename
        with xr.open_dataset(path) as ds:
            values_all = np.asarray(ds[cfg.variable].values, dtype=np.float64).ravel()
        values = values_all[np.isfinite(values_all)]
        if values.size == 0:
            raise ValueError(f"No finite values for {cfg.label}: {path}")
        samples.append(SampleSet(cfg, path, values, int(values_all.size)))
    return samples


def common_log_bins(samples: list[SampleSet], *, lower: float, n_bins: int, upper: float | None = None) -> np.ndarray:
    if upper is None:
        upper = max(float(np.nanmax(s.values)) for s in samples if s.values.size)
    return np.logspace(np.log10(lower), np.log10(upper), n_bins + 1)


def compute_unconditional_pdf(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts.astype(np.float64) / (values.size * widths)
    return counts.astype(np.int64), density


def plot_pdf(
    samples: list[SampleSet],
    *,
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    output_path: Path,
    xscale_log: bool,
    yscale_log: bool,
    dpi: int,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    positive_y: list[float] = []
    for sample in samples:
        cfg = sample.config
        density = densities[cfg.key]
        mask = density > 0.0
        positive_y.extend(density[mask].tolist())
        ax.plot(
            centers[mask],
            density[mask],
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=2.0,
        )
    if xscale_log:
        ax.set_xscale("log")
    if yscale_log:
        ax.set_yscale("log")
        if positive_y:
            ax.set_ylim(bottom=max(min(positive_y) * 0.6, 1.0e-10))
    ax.set_xlabel(r"Precipitation intensity (mm h$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"PDF density (mm$^{-1}$ h)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.25)
    ax.legend(loc="upper right", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_pdf_txt(
    path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    samples: list[SampleSet],
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    min_threshold: float,
    x_axis: str,
    y_axis: str,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation PDF data\n")
        fh.write("======================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write(f"X axis scale: {x_axis}\n")
        fh.write(f"Y axis scale: {y_axis}\n")
        fh.write("Method: common log-spaced bins for every dataset; no Radar clipping.\n")
        fh.write(
            "Density: bin_count / (all_finite_sample_count * bin_width), "
            "so the plotted PDF is not renormalized after dry/light values are removed.\n"
        )
        fh.write(f"First plotted bin edge: {min_threshold:g} mm h^-1\n")
        fh.write("\nSource files\n")
        fh.write("dataset,variable,path\n")
        for sample in samples:
            fh.write(f"{sample.config.label},{sample.config.variable},{sample.source_path}\n")
        fh.write("\nDataset summary\n")
        fh.write("dataset,total_grid_time_values,n_valid,n_positive,n_ge_first_edge,n_gt_100,max_mm_h\n")
        for sample in samples:
            vals = sample.values
            fh.write(
                f"{sample.config.label},{sample.total_values},{sample.n_valid},{sample.n_positive},"
                f"{sample.n_ge(min_threshold)},{sample.n_gt(100.0)},"
                f"{float(np.nanmax(vals)):.10g}\n"
            )
        fh.write("\nPlotted data\n")
        header = ["bin_left_mm_h", "bin_right_mm_h", "bin_center_mm_h"]
        for sample in samples:
            header.append(f"{sample.config.label}_density")
            header.append(f"{sample.config.label}_count")
        fh.write(",".join(header) + "\n")
        for i, center in enumerate(centers):
            row = [f"{edges[i]:.10g}", f"{edges[i + 1]:.10g}", f"{center:.10g}"]
            for sample in samples:
                row.append(f"{densities[sample.config.key][i]:.12g}")
                row.append(str(int(counts[sample.config.key][i])))
            fh.write(",".join(row) + "\n")


def make_spatial_bias_outputs(data_dir: Path, output_dir: Path, *, min_reference: float, vmin: float, vmax: float, dpi: int) -> dict[str, Path]:
    fields = read_all_spatial_means(data_dir)
    txt_dir = output_dir / "data_txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    radar_bias = build_bias_maps(
        fields,
        reference_key="radar",
        comparison_keys=("imerg", "control", "graupel", "2mom", "no3m"),
        min_reference=min_reference,
    )
    radar_fig = output_dir / "spatial_relative_bias_vs_radar.png"
    radar_txt = txt_dir / "spatial_relative_bias_vs_radar.txt"
    plot_bias_maps(
        fields,
        radar_bias,
        reference_key="radar",
        output_path=radar_fig,
        title_suffix="Radar",
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
        cbar_mode="sixth-slot",
    )
    write_bias_txt(
        radar_txt,
        figure_path=radar_fig,
        data_dir=data_dir,
        fields=fields,
        bias_maps=radar_bias,
        reference_key="radar",
        min_reference=min_reference,
        vmin=vmin,
        vmax=vmax,
        layout_note="2 x 3 grid: five bias panels plus colour bar in the sixth slot.",
    )
    outputs["bias_vs_radar_plot"] = radar_fig
    outputs["bias_vs_radar_txt"] = radar_txt

    imerg_bias = build_bias_maps(
        fields,
        reference_key="imerg",
        comparison_keys=("control", "graupel", "2mom", "no3m"),
        min_reference=min_reference,
    )
    imerg_fig = output_dir / "spatial_relative_bias_vs_imerg.png"
    imerg_txt = txt_dir / "spatial_relative_bias_vs_imerg.txt"
    plot_bias_maps(
        fields,
        imerg_bias,
        reference_key="imerg",
        output_path=imerg_fig,
        title_suffix="IMERG",
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
        cbar_mode="right",
    )
    write_bias_txt(
        imerg_txt,
        figure_path=imerg_fig,
        data_dir=data_dir,
        fields=fields,
        bias_maps=imerg_bias,
        reference_key="imerg",
        min_reference=min_reference,
        vmin=vmin,
        vmax=vmax,
        layout_note="2 x 2 grid with the colour bar on the right.",
    )
    outputs["bias_vs_imerg_plot"] = imerg_fig
    outputs["bias_vs_imerg_txt"] = imerg_txt
    return outputs


def make_diurnal_outputs(data_dir: Path, output_dir: Path, *, dpi: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diurnal_cycle_common_valid.png"
    txt_path = txt_dir / "diurnal_cycle_common_valid.txt"
    stats_by_key = {cfg.key: read_domain_mean_stats(data_dir, cfg, utc_offset_hours=-4) for cfg in DATASETS}
    plot_diurnal_cycle(
        stats_by_key,
        output_path=output_path,
        shade_mode="percent",
        std_multiplier=1.0,
        percent_uncertainty=0.10,
        dpi=dpi,
    )
    write_diurnal_txt(
        txt_path,
        data_dir=data_dir,
        stats_by_key=stats_by_key,
        shade_mode="percent",
        std_multiplier=1.0,
        percent_uncertainty=0.10,
        utc_offset_hours=-4,
        output_path=output_path,
    )
    return {"diurnal_plot": output_path, "diurnal_txt": txt_path}


def make_pdf_outputs(data_dir: Path, output_dir: Path, *, dpi: int, pdf_min_threshold: float, pdf_bins: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(data_dir)
    edges = common_log_bins(samples, lower=pdf_min_threshold, n_bins=pdf_bins)
    counts: dict[str, np.ndarray] = {}
    densities: dict[str, np.ndarray] = {}
    for sample in samples:
        sample_counts, sample_density = compute_unconditional_pdf(sample.values, edges)
        counts[sample.config.key] = sample_counts
        densities[sample.config.key] = sample_density
    ylog_fig = output_dir / "precip_pdf.png"
    loglog_fig = output_dir / "precip_pdf_loglog_allgrid.png"
    ylog_txt = txt_dir / "precip_pdf.txt"
    loglog_txt = txt_dir / "precip_pdf_loglog_allgrid.txt"
    plot_pdf(samples, edges=edges, densities=densities, output_path=ylog_fig, xscale_log=False, yscale_log=True, dpi=dpi)
    plot_pdf(samples, edges=edges, densities=densities, output_path=loglog_fig, xscale_log=True, yscale_log=True, dpi=dpi)
    write_pdf_txt(
        ylog_txt,
        figure_path=ylog_fig,
        data_dir=data_dir,
        samples=samples,
        edges=edges,
        densities=densities,
        counts=counts,
        min_threshold=pdf_min_threshold,
        x_axis="linear",
        y_axis="log",
    )
    write_pdf_txt(
        loglog_txt,
        figure_path=loglog_fig,
        data_dir=data_dir,
        samples=samples,
        edges=edges,
        densities=densities,
        counts=counts,
        min_threshold=pdf_min_threshold,
        x_axis="log",
        y_axis="log",
    )
    return {
        "pdf_ylog_plot": ylog_fig,
        "pdf_ylog_txt": ylog_txt,
        "pdf_loglog_plot": loglog_fig,
        "pdf_loglog_txt": loglog_txt,
    }


def copy_pair(src_plot: Path, src_txt: Path, folder: Path, plot_name: str, txt_name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dst_plot = folder / plot_name
    dst_txt = folder / txt_name
    shutil.copy2(src_plot, dst_plot)
    text = src_txt.read_text(encoding="utf-8")
    text = text.replace(str(src_plot), str(dst_plot))
    dst_txt.write_text(text, encoding="utf-8")


def package_paper1(outputs: dict[str, Path], paper_root: Path) -> None:
    copy_pair(
        outputs["bias_vs_radar_plot"],
        outputs["bias_vs_radar_txt"],
        paper_root / "01_spatial_bias_vs_radar",
        "spatial_bias_vs_radar_450dpi.png",
        "spatial_bias_vs_radar_data.txt",
    )
    copy_pair(
        outputs["bias_vs_imerg_plot"],
        outputs["bias_vs_imerg_txt"],
        paper_root / "02_spatial_bias_vs_gpm",
        "spatial_bias_vs_gpm_450dpi.png",
        "spatial_bias_vs_gpm_data.txt",
    )
    copy_pair(
        outputs["diurnal_plot"],
        outputs["diurnal_txt"],
        paper_root / "03_mean_diurnal_cycle",
        "mean_diurnal_cycle_450dpi.png",
        "mean_diurnal_cycle_data.txt",
    )
    copy_pair(
        outputs["pdf_ylog_plot"],
        outputs["pdf_ylog_txt"],
        paper_root / "04_rain_pdf_log_linear",
        "rain_pdf_log_linear_450dpi.png",
        "rain_pdf_log_linear_data.txt",
    )
    copy_pair(
        outputs["pdf_loglog_plot"],
        outputs["pdf_loglog_txt"],
        paper_root / "05_rain_pdf_log_log",
        "rain_pdf_log_log_450dpi.png",
        "rain_pdf_log_log_data.txt",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-common-dir", type=Path, default=BASE_COMMON_DIR)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--no3m-root", type=Path, default=NO3M_ACCUM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--min-reference", type=float, default=0.01)
    parser.add_argument("--vmin", type=float, default=-30.0)
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--pdf-min-threshold", type=float, default=0.1)
    parser.add_argument("--pdf-bins", type=int, default=99)
    parser.add_argument("--margin-deg", type=float, default=0.25)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--force-no3m", action="store_true")
    parser.add_argument("--skip-no3m-build", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    combined_dir = args.work_root / "common-valid-time-production"
    no3m_common = combined_dir / "NO3M_common_valid.nc"
    if not args.skip_no3m_build:
        build_no3m_common_valid(
            base_common_dir=args.base_common_dir,
            no3m_root=args.no3m_root,
            output_path=no3m_common,
            force=args.force_no3m,
            margin_deg=args.margin_deg,
            progress_interval=args.progress_interval,
        )
    prepare_combined_common_dir(args.base_common_dir, no3m_common, combined_dir)

    outputs: dict[str, Path] = {}
    outputs.update(
        make_spatial_bias_outputs(
            combined_dir,
            args.output_dir / "rainfall_spatial_bias_maps",
            min_reference=args.min_reference,
            vmin=args.vmin,
            vmax=args.vmax,
            dpi=args.dpi,
        )
    )
    outputs.update(make_diurnal_outputs(combined_dir, args.output_dir / "rainfall_diurnal_cycle", dpi=args.dpi))
    outputs.update(
        make_pdf_outputs(
            combined_dir,
            args.output_dir / "precip_distribution_corrected",
            dpi=args.dpi,
            pdf_min_threshold=args.pdf_min_threshold,
            pdf_bins=args.pdf_bins,
        )
    )
    package_paper1(outputs, args.paper_root)
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)
    print(f"paper1 packaged under: {args.paper_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
