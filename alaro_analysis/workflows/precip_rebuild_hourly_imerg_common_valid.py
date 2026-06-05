"""Rebuild common-valid rainfall files with complete hourly-mean IMERG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import xarray as xr


from alaro_analysis.common.constants import RUNS_ROOT
RAINFALL_ROOT = RUNS_ROOT / "rainfall-regridded-to-imerge"
DEFAULT_REGRIDDED_DIR = RAINFALL_ROOT / "cropped-regrided-imerg"
DEFAULT_RAW_IMERG_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/IMERG_AMAZON")
DEFAULT_OUTPUT_ROOT = RAINFALL_ROOT / "masked-production-final-hourly-imerg"
DEFAULT_START = "2014-01-01T00:00:00"
DEFAULT_END = "2015-12-31T23:00:00"

IMERG_NAME_RE = re.compile(r"(?P<date>\d{8})-S(?P<start>\d{6})")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    source_filename: str
    masked_filename: str
    time_masked_filename: str
    common_valid_filename: str
    variable: str


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        "radar",
        "Radar",
        "Radar_to_imerg.nc",
        "Radar_masked.nc",
        "Radar_time_masked.nc",
        "Radar_common_valid.nc",
        "rainfall_rate",
    ),
    DatasetConfig(
        "imerg",
        "IMERG(GPM)",
        "IMERG_hourly_from_complete_halfhour.nc",
        "IMERG_masked.nc",
        "IMERG_time_masked.nc",
        "IMERG_common_valid.nc",
        "precipitation",
    ),
    DatasetConfig(
        "control",
        "C1M",
        "Control_to_imerg.nc",
        "Control_masked.nc",
        "Control_time_masked.nc",
        "Control_common_valid.nc",
        "total_rain",
    ),
    DatasetConfig(
        "graupel",
        "G1M",
        "Graupel_to_imerg.nc",
        "Graupel_masked.nc",
        "Graupel_time_masked.nc",
        "Graupel_common_valid.nc",
        "total_rain",
    ),
    DatasetConfig(
        "2mom",
        "G2M",
        "2mom_to_imerg.nc",
        "2mom_masked.nc",
        "2-Moment_time_masked.nc",
        "2-Moment_common_valid.nc",
        "total_rain",
    ),
)


def parse_imerg_start_time(path: Path) -> np.datetime64:
    """Parse the half-hour start time encoded in an IMERG filename."""
    match = IMERG_NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse IMERG timestamp from {path.name!r}")
    stamp = f"{match.group('date')}{match.group('start')}"
    return np.datetime64(datetime.strptime(stamp, "%Y%m%d%H%M%S"), "ns")


def hourly_times(start: str, end: str) -> np.ndarray:
    start_dt = np.datetime64(start, "ns")
    end_dt = np.datetime64(end, "ns")
    if end_dt < start_dt:
        raise ValueError(f"End time {end!r} is before start time {start!r}")
    return np.arange(start_dt, end_dt + np.timedelta64(1, "h"), np.timedelta64(1, "h"))


def needed_halfhour_times(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype="datetime64[ns]")
    paired = np.column_stack([times, times + np.timedelta64(30, "m")])
    return paired.ravel()


def build_imerg_file_index(
    raw_imerg_dir: Path,
    expected_times: np.ndarray,
) -> tuple[dict[np.datetime64, Path], list[np.datetime64]]:
    expected = {np.datetime64(t, "ns") for t in np.asarray(expected_times, dtype="datetime64[ns]")}
    index: dict[np.datetime64, Path] = {}
    for path in sorted(raw_imerg_dir.glob("*.nc4")):
        try:
            timestamp = parse_imerg_start_time(path)
        except ValueError:
            continue
        if timestamp in expected:
            index[timestamp] = path
    missing = sorted(expected.difference(index))
    return index, missing


def clean_precipitation(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr) & (arr > -9990.0), arr, np.nan)
    return arr.astype(np.float32, copy=False)


def read_imerg_halfhour(
    path: Path,
    *,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    max_grid_distance: float,
) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if "precipitation" not in ds:
            raise KeyError(f"'precipitation' not found in {path}")
        da = ds["precipitation"].isel(time=0).sel(
            lat=target_lats,
            lon=target_lons,
            method="nearest",
        )
        lat_diff = float(np.max(np.abs(np.asarray(da["lat"].values) - target_lats)))
        lon_diff = float(np.max(np.abs(np.asarray(da["lon"].values) - target_lons)))
        if lat_diff > max_grid_distance or lon_diff > max_grid_distance:
            raise ValueError(
                f"Nearest IMERG grid point too far for {path}: "
                f"max lat diff={lat_diff:g}, max lon diff={lon_diff:g}"
            )
        return clean_precipitation(da.transpose("lat", "lon").values)


def hourly_mean_from_halfhours(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    stacked = np.stack([clean_precipitation(first), clean_precipitation(second)], axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0).astype(np.float32)


def build_hourly_imerg_array(
    *,
    file_index: dict[np.datetime64, Path],
    times: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    max_grid_distance: float,
    progress_interval: int,
) -> tuple[np.ndarray, list[np.datetime64]]:
    out = np.full(
        (len(times), len(target_lats), len(target_lons)),
        np.nan,
        dtype=np.float32,
    )
    missing_used: list[np.datetime64] = []
    for index, timestamp in enumerate(np.asarray(times, dtype="datetime64[ns]")):
        half_times = (timestamp, timestamp + np.timedelta64(30, "m"))
        pieces: list[np.ndarray] = []
        for half_time in half_times:
            path = file_index.get(half_time)
            if path is None:
                missing_used.append(half_time)
                continue
            pieces.append(
                read_imerg_halfhour(
                    path,
                    target_lats=target_lats,
                    target_lons=target_lons,
                    max_grid_distance=max_grid_distance,
                )
            )
        if pieces:
            with np.errstate(invalid="ignore"):
                out[index] = np.nanmean(np.stack(pieces, axis=0), axis=0).astype(np.float32)
        if progress_interval > 0 and (
            index == 0 or (index + 1) % progress_interval == 0 or index + 1 == len(times)
        ):
            print(f"IMERG hourly averaging: {index + 1}/{len(times)} hours", flush=True)
    return out, missing_used


def make_hourly_imerg_dataset(
    *,
    precipitation: np.ndarray,
    times: np.ndarray,
    template: xr.Dataset,
) -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={
            "precipitation": (
                ("time", "lat", "lon"),
                precipitation,
                {
                    "units": "mm/hr",
                    "long_name": "hourly mean precipitation from complete IMERG half-hour rates",
                    "source_halfhours": "mean of S000000-S002959 and S003000-S005959 files",
                },
            )
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "lat": template["lat"],
            "lon": template["lon"],
        },
        attrs={
            "title": "IMERG hourly mean cropped to ALARO/Radar IMERG grid",
            "method": "Each hourly value is the arithmetic mean of the :00 and :30 IMERG mm/hr rates.",
        },
    )
    for name in ("lat_bnds", "lon_bnds"):
        if name in template:
            ds[name] = template[name]
    return ds


def build_radar_mask(radar_ds: xr.Dataset) -> xr.DataArray:
    if "rainfall_rate" not in radar_ds:
        raise KeyError("'rainfall_rate' not found in radar dataset")
    mask = radar_ds["rainfall_rate"].notnull().any(dim="time")
    mask = mask.astype(bool)
    mask.name = "radar_mask"
    mask.attrs = {
        "long_name": "radar valid spatial mask",
        "method": "Grid cells with at least one finite radar rainfall_rate value.",
    }
    return mask


def apply_spatial_mask(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    out = ds.copy(deep=False)
    for name, da in list(out.data_vars.items()):
        if {"lat", "lon"}.issubset(da.dims):
            out[name] = da.where(mask)
    out["radar_mask"] = mask.astype(bool)
    return out


def netcdf_encoding(ds: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, da in ds.data_vars.items():
        item: dict[str, object] = {"zlib": True, "complevel": 4}
        if np.issubdtype(da.dtype, np.floating):
            item["_FillValue"] = np.nan
        encoding[name] = item
    return encoding


def write_dataset(ds: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=netcdf_encoding(ds))
    print(f"wrote: {path}", flush=True)
    return path


def write_radar_mask_files(mask: xr.DataArray, output_dir: Path) -> dict[str, Path]:
    latlon = xr.Dataset({"radar_mask": mask})
    lonlat_mask = mask.transpose("lon", "lat")
    both = xr.Dataset(
        {
            "radar_mask_latlon": mask,
            "radar_mask_lonlat": lonlat_mask,
        }
    )
    return {
        "radar_mask_latlon": write_dataset(latlon, output_dir / "Radar_mask_latlon.nc"),
        "radar_mask_from_radar": write_dataset(both, output_dir / "Radar_mask_from_radar.nc"),
    }


def as_datetime64_ns(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]")


def intersect_time_values(time_values: Sequence[np.ndarray]) -> np.ndarray:
    if not time_values:
        raise ValueError("No time arrays supplied")
    common = as_datetime64_ns(time_values[0])
    for values in time_values[1:]:
        common = np.intersect1d(common, as_datetime64_ns(values), assume_unique=False)
    return np.sort(common)


def valid_domain_mean_times(ds: xr.Dataset, variable: str, *, use_radar_flag: bool) -> np.ndarray:
    da = ds[variable]
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    if not spatial_dims:
        raise ValueError(f"{variable!r} has no spatial dimensions")
    mean_values = da.mean(dim=spatial_dims, skipna=True).values
    valid = np.isfinite(mean_values)
    if use_radar_flag and "valid_time_mask" in ds:
        radar_flag = np.asarray(ds["valid_time_mask"].values)
        valid &= np.isfinite(radar_flag) & (radar_flag > 0)
    return as_datetime64_ns(ds["time"].values)[valid]


def select_times(ds: xr.Dataset, times: np.ndarray) -> xr.Dataset:
    return ds.sel(time=as_datetime64_ns(times))


def write_summary(
    path: Path,
    *,
    raw_imerg_dir: Path,
    regridded_dir: Path,
    output_root: Path,
    hourly_times_count: int,
    halfhour_expected_count: int,
    halfhour_missing_count: int,
    halfhour_missing_used_count: int,
    radar_mask_count: int,
    common_time_count: int,
    common_valid_count: int,
    time_counts: dict[str, int],
    valid_counts: dict[str, int],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Hourly IMERG Common-Valid Rainfall Rebuild\n")
        fh.write("==========================================\n")
        fh.write(f"Raw IMERG directory: {raw_imerg_dir}\n")
        fh.write(f"Regridded radar/model source directory: {regridded_dir}\n")
        fh.write(f"Output root: {output_root}\n\n")
        fh.write("Method\n")
        fh.write("------\n")
        fh.write("1. Crop complete half-hour IMERG files to the existing 10 km IMERG grid.\n")
        fh.write("2. Average each :00 and :30 IMERG pair to one hourly mm/hr field.\n")
        fh.write("3. Regenerate the radar spatial mask from finite radar coverage.\n")
        fh.write("4. Select the common hourly timestamps across radar, hourly IMERG, and all models.\n")
        fh.write("5. Select common-valid timestamps where each domain mean is finite and radar valid_time_mask is true.\n\n")
        fh.write("Counts\n")
        fh.write("------\n")
        fh.write(f"Hourly timestamps requested: {hourly_times_count}\n")
        fh.write(f"Half-hour IMERG files expected: {halfhour_expected_count}\n")
        fh.write(f"Half-hour IMERG files missing from archive: {halfhour_missing_count}\n")
        fh.write(f"Half-hour IMERG slots missing while averaging: {halfhour_missing_used_count}\n")
        fh.write(f"Radar spatial mask cells: {radar_mask_count}\n")
        fh.write(f"Common hourly timestamps: {common_time_count}\n")
        fh.write(f"Common-valid timestamps: {common_valid_count}\n\n")
        fh.write("Dataset time counts\n")
        fh.write("-------------------\n")
        fh.write("dataset,time_masked_count,valid_domain_mean_count\n")
        for cfg in DATASETS:
            fh.write(f"{cfg.label},{time_counts[cfg.key]},{valid_counts[cfg.key]}\n")
    print(f"wrote: {path}", flush=True)
    return path


def rebuild_common_valid(
    *,
    raw_imerg_dir: Path = DEFAULT_RAW_IMERG_DIR,
    regridded_dir: Path = DEFAULT_REGRIDDED_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_grid_distance: float = 0.051,
    progress_interval: int = 500,
) -> dict[str, Path]:
    spatial_dir = output_root / "spatial_mask"
    time_dir = output_root / "time_masked"
    common_dir = output_root / "common-valid-time-production"
    diagnostics_dir = output_root / "diagnostics"

    times = hourly_times(start, end)
    halfhours = needed_halfhour_times(times)
    print(f"requested hourly IMERG timestamps: {len(times)}", flush=True)
    print(f"expected half-hour IMERG files: {len(halfhours)}", flush=True)

    radar_source_path = regridded_dir / "Radar_to_imerg.nc"
    with xr.open_dataset(radar_source_path) as template:
        template = template.load()
    target_lats = np.asarray(template["lat"].values)
    target_lons = np.asarray(template["lon"].values)

    file_index, missing = build_imerg_file_index(raw_imerg_dir, halfhours)
    if missing:
        print(f"missing half-hour IMERG files in requested period: {len(missing)}", flush=True)
    else:
        print("all expected half-hour IMERG files are present", flush=True)

    imerg_array, missing_used = build_hourly_imerg_array(
        file_index=file_index,
        times=times,
        target_lats=target_lats,
        target_lons=target_lons,
        max_grid_distance=max_grid_distance,
        progress_interval=progress_interval,
    )
    imerg_hourly = make_hourly_imerg_dataset(
        precipitation=imerg_array,
        times=times,
        template=template,
    )
    hourly_imerg_path = write_dataset(
        imerg_hourly,
        spatial_dir / "IMERG_hourly_from_complete_halfhour.nc",
    )

    radar_mask = build_radar_mask(template)
    write_radar_mask_files(radar_mask, spatial_dir)

    source_paths = {
        "radar": radar_source_path,
        "imerg": hourly_imerg_path,
        "control": regridded_dir / "Control_to_imerg.nc",
        "graupel": regridded_dir / "Graupel_to_imerg.nc",
        "2mom": regridded_dir / "2mom_to_imerg.nc",
    }

    masked_paths: dict[str, Path] = {}
    time_values: list[np.ndarray] = []
    for cfg in DATASETS:
        print(f"applying radar spatial mask: {cfg.label}", flush=True)
        with xr.open_dataset(source_paths[cfg.key]) as ds_in:
            masked = apply_spatial_mask(ds_in, radar_mask)
            masked.load()
        masked_paths[cfg.key] = write_dataset(masked, spatial_dir / cfg.masked_filename)
        time_values.append(as_datetime64_ns(masked["time"].values))

    common_times = intersect_time_values(time_values)
    print(f"common hourly timestamps after full archive rebuild: {len(common_times)}", flush=True)

    time_masked_paths: dict[str, Path] = {}
    valid_time_values: list[np.ndarray] = []
    time_counts: dict[str, int] = {}
    valid_counts: dict[str, int] = {}
    for cfg in DATASETS:
        print(f"selecting common hourly times: {cfg.label}", flush=True)
        with xr.open_dataset(masked_paths[cfg.key]) as ds_masked:
            time_masked = select_times(ds_masked, common_times)
            time_masked.load()
        time_masked_paths[cfg.key] = write_dataset(time_masked, time_dir / cfg.time_masked_filename)
        valid_times = valid_domain_mean_times(
            time_masked,
            cfg.variable,
            use_radar_flag=cfg.key == "radar",
        )
        valid_time_values.append(valid_times)
        time_counts[cfg.key] = int(time_masked.sizes["time"])
        valid_counts[cfg.key] = int(len(valid_times))

    common_valid_times = intersect_time_values(valid_time_values)
    print(f"common-valid timestamps after full archive rebuild: {len(common_valid_times)}", flush=True)

    common_paths: dict[str, Path] = {}
    for cfg in DATASETS:
        print(f"writing common-valid data: {cfg.label}", flush=True)
        with xr.open_dataset(time_masked_paths[cfg.key]) as ds_time:
            # Load the full time-masked file sequentially before selecting the
            # non-contiguous radar-valid times. This avoids very slow random
            # reads from compressed NetCDF chunks.
            loaded = ds_time.load()
            common_valid = select_times(loaded, common_valid_times)
            common_valid.load()
        common_paths[cfg.key] = write_dataset(common_valid, common_dir / cfg.common_valid_filename)

    summary_path = write_summary(
        diagnostics_dir / "rebuild_summary.txt",
        raw_imerg_dir=raw_imerg_dir,
        regridded_dir=regridded_dir,
        output_root=output_root,
        hourly_times_count=int(len(times)),
        halfhour_expected_count=int(len(halfhours)),
        halfhour_missing_count=int(len(missing)),
        halfhour_missing_used_count=int(len(missing_used)),
        radar_mask_count=int(radar_mask.sum().item()),
        common_time_count=int(len(common_times)),
        common_valid_count=int(len(common_valid_times)),
        time_counts=time_counts,
        valid_counts=valid_counts,
    )

    return {
        "hourly_imerg": hourly_imerg_path,
        "spatial_dir": spatial_dir,
        "time_dir": time_dir,
        "common_dir": common_dir,
        "summary": summary_path,
        **{f"{key}_common_valid": path for key, path in common_paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild common-valid rainfall files using complete hourly-mean IMERG."
    )
    parser.add_argument("--raw-imerg-dir", type=Path, default=DEFAULT_RAW_IMERG_DIR)
    parser.add_argument("--regridded-dir", type=Path, default=DEFAULT_REGRIDDED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--max-grid-distance", type=float, default=0.051)
    parser.add_argument("--progress-interval", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = rebuild_common_valid(
        raw_imerg_dir=args.raw_imerg_dir,
        regridded_dir=args.regridded_dir,
        output_root=args.output_root,
        start=args.start,
        end=args.end,
        max_grid_distance=args.max_grid_distance,
        progress_interval=args.progress_interval,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
