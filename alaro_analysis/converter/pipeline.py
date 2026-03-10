#!/usr/bin/env python3
"""
Standalone ALARO FA -> NetCDF converter using faxarray.

Input layout:
  <input_root>/(pf|sfx)YYYYMMDD/<hourly-file>

Output layout:
  <output_root>/<VAR>/(pf|sfx)YYYYMMDD/<hourly-file>.nc
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import xarray as xr

import faxarray as fx

from .aliases import normalize_var_token, resolve_requested_vars, var_to_ds_name
from .config import (
    AUX_VAR,
    DERIVED_MODEL_RH_VAR,
    GRAVITY,
    MODEL_LEVEL_VARS,
    MODEL_RH_PRESSURE_CANDIDATES,
    MODEL_RH_Q_VAR,
    MODEL_RH_T_VAR,
    PLEVELS_PA,
    PRESSURE_LEVEL_VARS,
    REQUESTED_VARS,
    SURFACE_VARS,
    resolve_requested_vars_from_cli,
)
from .models import CropWindow, FileTask, RunConfig, VariablePlan

DAY_DIR_RE = re.compile(r"^(?:pf|sfx)(\d{8})$")
HOUR_FILE_RE = re.compile(r"^.+\+(\d{4})(?:\.[^.]+)?$")


def _subset_pressure_levels(out_ds: xr.Dataset, var_name: str) -> xr.Dataset:
    if not PLEVELS_PA:
        return out_ds
    if var_name not in out_ds.data_vars:
        return out_ds

    da = out_ds[var_name]
    if "pressure" not in da.dims:
        return out_ds

    level_values = da.attrs.get("level_values")
    if level_values is None:
        return out_ds

    try:
        levels = np.asarray(level_values, dtype=np.int64)
    except Exception:  # noqa: BLE001
        return out_ds

    wanted = {int(v) for v in PLEVELS_PA}
    keep_idx = [i for i, v in enumerate(levels.tolist()) if int(v) in wanted]
    if not keep_idx:
        return out_ds

    out_ds = out_ds.isel(pressure=keep_idx)
    attrs = dict(out_ds[var_name].attrs)
    attrs["level_values"] = [int(levels[i]) for i in keep_idx]
    out_ds[var_name].attrs = attrs
    return out_ds


def _compute_relative_humidity(
    specific_humidity: xr.DataArray,
    temperature_k: xr.DataArray,
    pressure_pa: xr.DataArray,
) -> xr.DataArray:
    # q [kg/kg], T [K], p [Pa] -> RH [0..1]
    eps = 0.622
    q = specific_humidity.astype(np.float64)
    t = temperature_k.astype(np.float64)
    p = pressure_pa.astype(np.float64)

    q = xr.where(q < 0.0, np.nan, q)
    e = q * p / (eps + (1.0 - eps) * q)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    rh = (e / es).clip(min=0.0, max=1.0)
    rh.attrs = {
        "long_name": "Relative humidity",
        "units": "1",
        "comment": "Derived from specific humidity, temperature, and pressure.",
    }
    return rh


def _coerce_pressure_to_pa(pressure: xr.DataArray, source_name: str) -> xr.DataArray:
    p = pressure.astype(np.float64)
    units = str(pressure.attrs.get("units", "")).strip().lower()

    if units in {"pa", "pascal", "pascals"}:
        return p
    if units in {"hpa", "mbar", "mb"}:
        return p * 100.0

    values = np.asarray(p.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot derive RH: pressure '{source_name}' has no finite values.")

    p01 = float(np.nanpercentile(finite, 1))
    p99 = float(np.nanpercentile(finite, 99))

    # Heuristic for missing/unknown units.
    if 100.0 <= p01 <= 1200.0 and 100.0 <= p99 <= 2000.0:
        return p * 100.0  # likely hPa
    if 1000.0 <= p01 <= 120000.0 and 1000.0 <= p99 <= 150000.0:
        return p  # likely Pa

    raise ValueError(
        f"Cannot derive RH: unsupported pressure range for '{source_name}' "
        f"(units='{units}', p01={p01:.3g}, p99={p99:.3g})."
    )
def parse_yyyymmdd(value: Optional[str], name: str):
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid {name}='{value}', expected YYYYMMDD") from exc


def expected_hours(include_init: bool) -> Sequence[int]:
    return tuple(range(0, 25)) if include_init else tuple(range(1, 25))


def _hourly_file_sort_key(path: Path) -> tuple[int, int, str]:
    suffixes = "".join(path.suffixes)
    if suffixes == ".sfx":
        suffix_rank = 0
    elif suffixes == "":
        suffix_rank = 1
    else:
        suffix_rank = 2
    return (suffix_rank, len(path.name), path.name)


def _choose_preferred_hourly_file(left: Path, right: Path) -> Path:
    return min((left, right), key=_hourly_file_sort_key)


def discover_days(
    input_root: Path, start_date=None, end_date=None
) -> List[Tuple[datetime.date, Path]]:
    found: List[Tuple[datetime.date, Path]] = []
    for path in input_root.iterdir():
        if not path.is_dir():
            continue
        m = DAY_DIR_RE.fullmatch(path.name)
        if not m:
            continue
        day = datetime.strptime(m.group(1), "%Y%m%d").date()
        if start_date is not None and day < start_date:
            continue
        if end_date is not None and day > end_date:
            continue
        found.append((day, path))
    found.sort(key=lambda x: x[0])
    return found


def validate_day(day_dir: Path, hours: Sequence[int]) -> Tuple[bool, Dict[int, Path], str]:
    expected = set(hours)
    files_by_hour: Dict[int, Path] = {}

    for fp in day_dir.iterdir():
        if not fp.is_file():
            continue
        m = HOUR_FILE_RE.fullmatch(fp.name)
        if not m:
            continue
        hour = int(m.group(1))
        if hour not in expected:
            continue
        if hour in files_by_hour:
            files_by_hour[hour] = _choose_preferred_hourly_file(files_by_hour[hour], fp)
            continue
        files_by_hour[hour] = fp

    missing = sorted(expected - set(files_by_hour))
    if missing:
        parts = []
        parts.append("missing: " + ", ".join(f"+{h:04d}" for h in missing))
        return False, {}, "; ".join(parts)
    return True, files_by_hour, ""


def compute_crop_window(sample_file: Path, cfg: RunConfig, reference_var: str) -> CropWindow:
    ds = fx.open_dataset(str(sample_file), variables=[reference_var], stack_levels=True)
    try:
        lon = ds["lon"]
        lat = ds["lat"]
        mask = (
            (lon >= cfg.bbox_west)
            & (lon <= cfg.bbox_east)
            & (lat >= cfg.bbox_south)
            & (lat <= cfg.bbox_north)
        )
        mask_values = np.asarray(mask.values)
        if not mask_values.any():
            raise ValueError("ROI selects no points in sample file.")

        ys, xs = np.where(mask_values)
        return CropWindow(
            y_start=int(ys.min()),
            y_stop=int(ys.max()) + 1,
            x_start=int(xs.min()),
            x_stop=int(xs.max()) + 1,
            source_y=int(ds.sizes["y"]),
            source_x=int(ds.sizes["x"]),
        )
    finally:
        ds.close()


def _guess_coord_name(
    da: xr.DataArray,
    explicit_name: Optional[str],
    kind: str,
) -> str:
    if explicit_name is not None:
        if explicit_name in da.coords or explicit_name in da.dims:
            return explicit_name
        raise ValueError(f"{kind} coordinate '{explicit_name}' not found in mask variable")

    candidates = {
        "lat": ["lat", "latitude", "nav_lat", "y_lat"],
        "lon": ["lon", "longitude", "nav_lon", "x_lon"],
    }[kind]
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    for name in list(da.coords) + list(da.dims):
        low = name.lower()
        if kind == "lat" and "lat" in low:
            return name
        if kind == "lon" and ("lon" in low or "long" in low):
            return name
    raise ValueError(f"Could not auto-detect {kind} coordinate in mask variable")


def _select_mask_variable(mask_ds: xr.Dataset, requested_var: Optional[str]) -> str:
    if requested_var:
        if requested_var not in mask_ds.data_vars:
            raise ValueError(
                f"Mask variable '{requested_var}' not found. "
                f"Available: {list(mask_ds.data_vars)}"
            )
        return requested_var

    if not mask_ds.data_vars:
        raise ValueError("Mask NetCDF has no data variables")

    preferred = [
        "mask",
        "MASK",
        "spatial_mask",
        "radar_mask",
        "Radar_mask",
        "Radar_mask_latlon",
    ]
    for name in preferred:
        if name in mask_ds.data_vars:
            return name
    return list(mask_ds.data_vars)[0]


def _regrid_mask_to_model(
    mask_file: Path,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    mask_var: Optional[str],
    mask_lat_name: Optional[str],
    mask_lon_name: Optional[str],
    mask_threshold: float,
) -> Tuple[np.ndarray, str]:
    mask_ds = xr.open_dataset(mask_file)
    try:
        selected_var = _select_mask_variable(mask_ds, mask_var)
        mask_da = mask_ds[selected_var].squeeze(drop=True)
        # xarray.interp() requires numeric dtype; some masks are stored as bool.
        if mask_da.dtype == np.bool_:
            mask_da = mask_da.astype(np.float32)

        lat_name = _guess_coord_name(mask_da, mask_lat_name, "lat")
        lon_name = _guess_coord_name(mask_da, mask_lon_name, "lon")

        # Drop non-spatial dimensions by taking first index (common for time/band singleton dims).
        for dim in list(mask_da.dims):
            if dim not in {lat_name, lon_name}:
                if mask_da.sizes[dim] < 1:
                    raise ValueError(f"Mask dimension '{dim}' is empty")
                mask_da = mask_da.isel({dim: 0})

        if lat_name in mask_da.coords and lon_name in mask_da.coords:
            lat_coord = mask_da.coords[lat_name]
            lon_coord = mask_da.coords[lon_name]
        else:
            raise ValueError(
                f"Mask variable '{selected_var}' must expose '{lat_name}' and '{lon_name}' coordinates"
            )

        if lat_coord.ndim == 1 and lon_coord.ndim == 1:
            target_lat_da = xr.DataArray(target_lat.values, dims=("y", "x"))
            target_lon_da = xr.DataArray(target_lon.values, dims=("y", "x"))
            remapped = mask_da.interp(
                {lat_name: target_lat_da, lon_name: target_lon_da},
                method="nearest",
            )
        elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
            try:
                from scipy.spatial import cKDTree
            except ImportError as exc:
                raise ImportError(
                    "Mask has 2D lat/lon coordinates and requires scipy for nearest-neighbor remap."
                ) from exc

            mask_values = np.asarray(mask_da.values)
            if mask_values.ndim != 2:
                raise ValueError(
                    "Mask with 2D coordinates must be a 2D variable after squeezing extra dims."
                )

            source_points = np.column_stack(
                [np.asarray(lon_coord.values).ravel(), np.asarray(lat_coord.values).ravel()]
            )
            target_points = np.column_stack(
                [np.asarray(target_lon.values).ravel(), np.asarray(target_lat.values).ravel()]
            )
            tree = cKDTree(source_points)
            _, nearest_idx = tree.query(target_points, k=1)
            remapped_values = mask_values.ravel()[nearest_idx].reshape(target_lon.shape)
            remapped = xr.DataArray(remapped_values, dims=("y", "x"))
        else:
            raise ValueError(
                "Unsupported mask coordinate layout. Expected lat/lon as 1D or 2D coordinates."
            )

        remapped_values = np.asarray(remapped.values, dtype=np.float64)
        mask_binary = remapped_values > float(mask_threshold)
        if not np.any(mask_binary):
            raise ValueError(
                f"Mask threshold ({mask_threshold}) removed all points on model grid."
            )
        return mask_binary, selected_var
    finally:
        mask_ds.close()


def compute_crop_and_spatial_mask(
    sample_file: Path,
    cfg: RunConfig,
    reference_var: str,
) -> Tuple[CropWindow, np.ndarray, Dict[str, object]]:
    ds = fx.open_dataset(str(sample_file), variables=[reference_var], stack_levels=True)
    try:
        lon = ds["lon"]
        lat = ds["lat"]

        roi_mask = (
            (lon >= cfg.bbox_west)
            & (lon <= cfg.bbox_east)
            & (lat >= cfg.bbox_south)
            & (lat <= cfg.bbox_north)
        )
        roi_values = np.asarray(roi_mask.values, dtype=bool)
        if not np.any(roi_values):
            raise ValueError("ROI selects no points in sample file.")

        selected_mask_var = None
        if cfg.mask_file:
            mask_file = Path(cfg.mask_file)
            if not mask_file.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
            remapped_mask, selected_mask_var = _regrid_mask_to_model(
                mask_file=mask_file,
                target_lat=lat,
                target_lon=lon,
                mask_var=cfg.mask_var,
                mask_lat_name=cfg.mask_lat_name,
                mask_lon_name=cfg.mask_lon_name,
                mask_threshold=cfg.mask_threshold,
            )
            spatial_mask = roi_values & remapped_mask
        else:
            spatial_mask = roi_values

        if not np.any(spatial_mask):
            raise ValueError("Combined ROI/mask selects no model grid points.")

        ys, xs = np.where(spatial_mask)
        crop = CropWindow(
            y_start=int(ys.min()),
            y_stop=int(ys.max()) + 1,
            x_start=int(xs.min()),
            x_stop=int(xs.max()) + 1,
            source_y=int(ds.sizes["y"]),
            source_x=int(ds.sizes["x"]),
        )

        cropped_mask = spatial_mask[crop.y_start : crop.y_stop, crop.x_start : crop.x_stop]
        stats = {
            "roi_points": int(np.count_nonzero(roi_values)),
            "kept_points": int(np.count_nonzero(spatial_mask)),
            "keep_fraction_vs_roi": float(
                np.count_nonzero(spatial_mask) / np.count_nonzero(roi_values)
            ),
            "mask_file": cfg.mask_file,
            "mask_var": selected_mask_var,
            "mask_threshold": cfg.mask_threshold if cfg.mask_file else None,
        }
        return crop, cropped_mask, stats
    finally:
        ds.close()


def output_path(output_root: Path, var_name: str, task: FileTask) -> Path:
    src_name = Path(task.source_file).name
    return output_root / var_name / task.day_name / f"{src_name}.nc"


def process_task(
    task: FileTask,
    cfg: RunConfig,
    crop: CropWindow,
    spatial_mask_cropped: np.ndarray,
    var_plan: VariablePlan,
) -> Dict[str, object]:
    ds = None
    try:
        ds = fx.open_dataset(str(task.source_file), variables=list(var_plan.read_vars), stack_levels=True)
        read_var_to_ds_name = {name: var_to_ds_name(name) for name in var_plan.read_vars}
        output_var_to_ds_name = {name: var_to_ds_name(name) for name in var_plan.output_vars}

        if int(ds.sizes.get("y", -1)) != crop.source_y or int(ds.sizes.get("x", -1)) != crop.source_x:
            raise ValueError(
                f"Grid mismatch for {task.source_file}: "
                f"expected ({crop.source_y}, {crop.source_x}), "
                f"got ({int(ds.sizes.get('y', -1))}, {int(ds.sizes.get('x', -1))})"
            )

        ds = ds.isel(y=slice(crop.y_start, crop.y_stop), x=slice(crop.x_start, crop.x_stop))

        if spatial_mask_cropped.shape != (int(ds.sizes["y"]), int(ds.sizes["x"])):
            raise ValueError(
                "Spatial mask shape mismatch after crop: "
                f"mask={spatial_mask_cropped.shape}, data={(int(ds.sizes['y']), int(ds.sizes['x']))}"
            )
        mask_da = xr.DataArray(spatial_mask_cropped, dims=("y", "x"))
        ds = ds.where(mask_da, drop=False)

        missing_raw = [
            raw_name
            for raw_name, ds_name in read_var_to_ds_name.items()
            if ds_name not in ds.data_vars
        ]
        if missing_raw:
            raise KeyError(
                f"Missing variables in {task.source_file}: {', '.join(missing_raw)}"
            )

        if var_plan.derive_model_rh:
            if not (var_plan.rh_q_var and var_plan.rh_t_var and var_plan.rh_p_var):
                raise ValueError(
                    f"{DERIVED_MODEL_RH_VAR} requested but dependencies are unresolved."
                )
            q_da = ds[var_to_ds_name(var_plan.rh_q_var)]
            t_da = ds[var_to_ds_name(var_plan.rh_t_var)]
            p_src = ds[var_to_ds_name(var_plan.rh_p_var)]
            p_pa = _coerce_pressure_to_pa(p_src, var_plan.rh_p_var)
            rh_da = _compute_relative_humidity(q_da, t_da, p_pa)
            rh_da.attrs["source_specific_humidity"] = var_plan.rh_q_var
            rh_da.attrs["source_temperature"] = var_plan.rh_t_var
            rh_da.attrs["source_pressure"] = var_plan.rh_p_var
            ds[var_to_ds_name(DERIVED_MODEL_RH_VAR)] = rh_da

        if AUX_VAR in output_var_to_ds_name:
            aux_ds_name = output_var_to_ds_name[AUX_VAR]
            geo = ds[aux_ds_name] / GRAVITY
            geo.attrs = dict(ds[aux_ds_name].attrs)
            geo.attrs["units"] = "m"
            geo.attrs["long_name"] = "Geopotential height"
            geo.attrs["comment"] = f"Converted from {AUX_VAR} by division with {GRAVITY}."
            ds[aux_ds_name] = geo

        written = 0
        out_root = Path(cfg.output_root)
        for raw_name, ds_name in output_var_to_ds_name.items():
            out_file = output_path(out_root, raw_name, task)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if out_file.exists() and not cfg.overwrite:
                continue

            out_ds = ds[[ds_name]].copy()
            if ds_name != raw_name:
                out_ds = out_ds.rename({ds_name: raw_name})
            out_ds = _subset_pressure_levels(out_ds, raw_name)
            out_ds.attrs = dict(ds.attrs)
            out_ds.attrs["source_fa_file"] = str(task.source_file)
            out_ds.attrs["roi_bbox"] = (
                f"lon=[{cfg.bbox_west}, {cfg.bbox_east}], "
                f"lat=[{cfg.bbox_south}, {cfg.bbox_north}]"
            )
            out_ds.attrs["roi_mask_mode"] = "cropped_bbox_exact_mask"

            encoding = {}
            if cfg.compress == "zlib":
                encoding[raw_name] = {"zlib": True, "complevel": cfg.compress_level}

            out_ds.to_netcdf(out_file, mode="w", encoding=encoding)
            out_ds.close()
            written += 1

        return {"status": "ok", "source_file": task.source_file, "written": written}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "source_file": task.source_file, "error": str(exc)}
    finally:
        if ds is not None:
            ds.close()


def write_lines(path: Path, lines: Sequence[str]) -> None:
    if lines:
        path.write_text("\n".join(lines) + "\n")
    else:
        path.write_text("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ALARO or SURFEX hourly files to per-variable masked NetCDF outputs."
    )
    parser.add_argument(
        "input_root",
        help="Directory containing pfYYYYMMDD or sfxYYYYMMDD folders",
    )
    parser.add_argument("output_root", help="Directory where NetCDF outputs will be written")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker count (default: 16)")
    parser.add_argument("--bbox-west", type=float, default=-67.0, help="ROI west bound (default: -67)")
    parser.add_argument("--bbox-east", type=float, default=-53.0, help="ROI east bound (default: -53)")
    parser.add_argument("--bbox-south", type=float, default=-10.0, help="ROI south bound (default: -10)")
    parser.add_argument("--bbox-north", type=float, default=4.0, help="ROI north bound (default: 4)")
    parser.add_argument("--include-init", dest="include_init", action="store_true", help="Include +0000 (default)")
    parser.add_argument("--exclude-init", dest="include_init", action="store_false", help="Skip +0000")
    parser.set_defaults(include_init=True)
    parser.add_argument("--compress", choices=["zlib", "none"], default="zlib", help="Compression mode")
    parser.add_argument("--level", type=int, default=1, help="Compression level when using zlib")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite outputs (default)")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite outputs")
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--skip-incomplete-days",
        dest="skip_incomplete_days",
        action="store_true",
        help="Skip days missing required hourly files (default)",
    )
    parser.add_argument(
        "--strict-days",
        dest="skip_incomplete_days",
        action="store_false",
        help="Fail if any day is incomplete",
    )
    parser.set_defaults(skip_incomplete_days=True)
    parser.add_argument("--start-date", metavar="YYYYMMDD", help="Process days on/after this date")
    parser.add_argument("--end-date", metavar="YYYYMMDD", help="Process days on/before this date")
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help=(
            "Explicit variable list (space- or comma-separated). "
            "If provided, overrides built-in variable blocks."
        ),
    )
    parser.add_argument(
        "--vars-file",
        type=Path,
        default=None,
        help="Text file with variable names (# comments supported; one name per line).",
    )
    parser.add_argument(
        "--append-vars",
        nargs="+",
        default=None,
        help="Extra variables to append to the selected/default list.",
    )
    parser.add_argument(
        "--drop-vars",
        nargs="+",
        default=None,
        help="Variables to remove from the selected/default list.",
    )
    parser.add_argument(
        "--list-default-vars",
        action="store_true",
        help="Print built-in default variables and exit.",
    )
    parser.add_argument("--mask-file", default=None, help="Optional NetCDF mask file (e.g., 1 km radar mask)")
    parser.add_argument("--mask-var", default=None, help="Mask variable name in mask NetCDF")
    parser.add_argument("--mask-lat-name", default=None, help="Latitude coordinate name in mask file")
    parser.add_argument("--mask-lon-name", default=None, help="Longitude coordinate name in mask file")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Keep points where remapped mask > threshold (default: 0.5)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce logging")
    args = parser.parse_args()

    if args.list_default_vars:
        for name in REQUESTED_VARS:
            print(name)
        return 0

    requested_vars, requested_vars_source = resolve_requested_vars_from_cli(args)
    if not requested_vars:
        raise ValueError(
            "No variables selected. Provide --vars/--vars-file, or remove --drop-vars."
        )

    cfg = RunConfig(
        input_root=str(Path(args.input_root)),
        output_root=str(Path(args.output_root)),
        workers=max(1, int(args.workers)),
        bbox_west=float(args.bbox_west),
        bbox_east=float(args.bbox_east),
        bbox_south=float(args.bbox_south),
        bbox_north=float(args.bbox_north),
        include_init=bool(args.include_init),
        compress=str(args.compress).lower(),
        compress_level=int(args.level),
        overwrite=bool(args.overwrite),
        skip_incomplete_days=bool(args.skip_incomplete_days),
        start_date=args.start_date,
        end_date=args.end_date,
        mask_file=args.mask_file,
        mask_var=args.mask_var,
        mask_lat_name=args.mask_lat_name,
        mask_lon_name=args.mask_lon_name,
        mask_threshold=float(args.mask_threshold),
        quiet=bool(args.quiet),
    )

    if cfg.compress not in {"zlib", "none"}:
        raise ValueError("compress must be one of: zlib, none")
    if not (0 <= cfg.compress_level <= 9):
        raise ValueError("level must be in range [0, 9]")

    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    start_date = parse_yyyymmdd(cfg.start_date, "start_date")
    end_date = parse_yyyymmdd(cfg.end_date, "end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    hours = list(expected_hours(cfg.include_init))
    days = discover_days(input_root, start_date=start_date, end_date=end_date)

    skipped_days: List[str] = []
    tasks: List[FileTask] = []
    valid_day_names: List[str] = []

    for _, day_dir in days:
        ok, files_by_hour, reason = validate_day(day_dir, hours)
        if not ok:
            msg = f"{day_dir.name}: {reason}"
            if cfg.skip_incomplete_days:
                skipped_days.append(msg)
                continue
            raise ValueError(msg)
        valid_day_names.append(day_dir.name)
        for hour in hours:
            tasks.append(
                FileTask(day_name=day_dir.name, hour=hour, source_file=str(files_by_hour[hour]))
            )

    if not cfg.quiet:
        print(f"Discovered days: {len(days)}", flush=True)
        print(f"Valid days: {len(valid_day_names)}", flush=True)
        print(f"Skipped days: {len(skipped_days)}", flush=True)
        print(f"Scheduled FA files: {len(tasks)}", flush=True)
        print(
            f"Requested variables ({len(requested_vars)}): source={requested_vars_source}",
            flush=True,
        )

    processed_files = 0
    failed_files = 0
    written_netcdf_files = 0
    failures: List[str] = []
    mask_stats: Dict[str, object] = {}
    var_plan = VariablePlan(
        output_vars=tuple(),
        read_vars=tuple(),
        missing_requested=tuple(),
        derive_model_rh=False,
        rh_q_var=None,
        rh_t_var=None,
        rh_p_var=None,
    )

    if tasks:
        var_plan = resolve_requested_vars(
            Path(tasks[0].source_file),
            requested_vars,
            derived_model_rh_var=DERIVED_MODEL_RH_VAR,
            model_rh_q_var=MODEL_RH_Q_VAR,
            model_rh_t_var=MODEL_RH_T_VAR,
            model_rh_pressure_candidates=MODEL_RH_PRESSURE_CANDIDATES,
        )
        if not var_plan.output_vars:
            raise ValueError(
                "None of the requested variables were found in the sample FA file. "
                "Check --vars/--vars-file selection."
            )

        if any(msg.startswith(f"{DERIVED_MODEL_RH_VAR} requires") for msg in var_plan.missing_requested):
            missing_msg = ", ".join(
                msg for msg in var_plan.missing_requested if msg.startswith(f"{DERIVED_MODEL_RH_VAR} requires")
            )
            raise ValueError(f"Cannot derive {DERIVED_MODEL_RH_VAR}: {missing_msg}")

        reference_var = AUX_VAR if AUX_VAR in var_plan.read_vars else var_plan.read_vars[0]
        crop, spatial_mask_cropped, mask_stats = compute_crop_and_spatial_mask(
            Path(tasks[0].source_file), cfg, reference_var
        )
        if not cfg.quiet:
            print(
                f"Crop window y=[{crop.y_start}:{crop.y_stop}), "
                f"x=[{crop.x_start}:{crop.x_stop})"
            , flush=True)
            print(
                "Mask coverage vs ROI: "
                f"{mask_stats['kept_points']}/{mask_stats['roi_points']} "
                f"({100.0 * mask_stats['keep_fraction_vs_roi']:.2f}%)"
            , flush=True)
            if mask_stats.get("mask_file"):
                print(
                    f"Using external mask: {mask_stats['mask_file']} "
                    f"(var={mask_stats.get('mask_var')}, threshold>{mask_stats.get('mask_threshold')})"
                , flush=True)
            if var_plan.missing_requested:
                print(
                    "Missing variables (skipped): " + ", ".join(var_plan.missing_requested),
                    flush=True,
                )
            if var_plan.derive_model_rh:
                print(
                    f"Deriving {DERIVED_MODEL_RH_VAR} from "
                    f"{var_plan.rh_q_var}, {var_plan.rh_t_var}, {var_plan.rh_p_var}"
                , flush=True)

        if cfg.workers > 1:
            with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
                future_map = {
                    pool.submit(
                        process_task,
                        task,
                        cfg,
                        crop,
                        spatial_mask_cropped,
                        var_plan,
                    ): task
                    for task in tasks
                }
                total = len(future_map)
                for i, future in enumerate(as_completed(future_map), start=1):
                    task = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "status": "failed",
                            "source_file": task.source_file,
                            "error": f"Worker crashed: {exc}",
                        }

                    if result.get("status") == "ok":
                        processed_files += 1
                        written_netcdf_files += int(result.get("written", 0))
                    else:
                        failed_files += 1
                        failures.append(
                            f"{result.get('source_file', task.source_file)}: "
                            f"{result.get('error', 'unknown error')}"
                        )

                    if not cfg.quiet and (i % 25 == 0 or i == total):
                        print(
                            f"Progress {i}/{total}: processed={processed_files}, "
                            f"failed={failed_files}, written={written_netcdf_files}",
                            flush=True,
                        )
        else:
            total = len(tasks)
            for i, task in enumerate(tasks, start=1):
                result = process_task(task, cfg, crop, spatial_mask_cropped, var_plan)
                if result.get("status") == "ok":
                    processed_files += 1
                    written_netcdf_files += int(result.get("written", 0))
                else:
                    failed_files += 1
                    failures.append(
                        f"{result.get('source_file', task.source_file)}: "
                        f"{result.get('error', 'unknown error')}"
                    )
                if not cfg.quiet and (i % 25 == 0 or i == total):
                    print(
                        f"Progress {i}/{total}: processed={processed_files}, "
                        f"failed={failed_files}, written={written_netcdf_files}",
                        flush=True,
                    )

    skipped_log = output_root / "skipped_days.log"
    failures_log = output_root / "failures.log"
    summary_file = output_root / "summary.json"

    write_lines(skipped_log, skipped_days)
    write_lines(failures_log, failures)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "requested_variables_input": requested_vars,
        "requested_variables_source": requested_vars_source,
        "variables": list(var_plan.output_vars),
        "variables_read_for_processing": list(var_plan.read_vars),
        "missing_requested_variables": list(var_plan.missing_requested),
        "derived_relative_humidity": {
            "enabled": var_plan.derive_model_rh,
            "output_var": DERIVED_MODEL_RH_VAR if var_plan.derive_model_rh else None,
            "source_specific_humidity": var_plan.rh_q_var,
            "source_temperature": var_plan.rh_t_var,
            "source_pressure": var_plan.rh_p_var,
        },
        "configured_model_level_vars": MODEL_LEVEL_VARS,
        "configured_pressure_level_vars": PRESSURE_LEVEL_VARS,
        "configured_pressure_levels_pa": PLEVELS_PA,
        "configured_surface_vars": SURFACE_VARS,
        "discovered_days": len(days),
        "valid_days": len(valid_day_names),
        "skipped_days": len(skipped_days),
        "scheduled_files": len(tasks),
        "processed_files": processed_files,
        "failed_files": failed_files,
        "written_netcdf_files": written_netcdf_files,
        "include_init": cfg.include_init,
        "hours": [f"+{h:04d}" for h in hours],
        "workers": cfg.workers,
        "overwrite": cfg.overwrite,
        "compression": cfg.compress,
        "compression_level": cfg.compress_level,
        "skip_incomplete_days": cfg.skip_incomplete_days,
        "roi": {
            "west": cfg.bbox_west,
            "east": cfg.bbox_east,
            "south": cfg.bbox_south,
            "north": cfg.bbox_north,
        },
        "mask": mask_stats,
        "valid_day_range": {
            "start": valid_day_names[0] if valid_day_names else None,
            "end": valid_day_names[-1] if valid_day_names else None,
        },
        "logs": {
            "skipped_days": str(skipped_log),
            "failures": str(failures_log),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not cfg.quiet:
        print(f"Done. Summary written to {summary_file}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
