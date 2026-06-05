#!/usr/bin/env python3
"""From-scratch, conservative rebuild of common-valid hourly rainfall.

Rebuilds Radar, IMERG and the four model runs (C1M/G1M/G2M/G2M-XCU) on ONE
consistent grid and time convention, so the spatial-bias-vs-radar comparison is
internally consistent.  Everything is conservatively regridded with the SAME CDO
binary (no cross-version inconsistency), and every dataset is put on the same
hourly bin and label.

Key conventions (verified against the source data/scripts)
----------------------------------------------------------
* Target grid: IMERG-native 0.1 deg, cropped to the radar box, taken from
  ``cdo griddes`` of a cropped raw IMERG file (== the historical target grid).
* Hourly bin ``[H:00, H+1:00)`` labelled at the START ``H:00`` for ALL datasets:
    - Radar : the trusted gap-filled hourly product (start-labelled, has
              ``valid_time_mask``); conservatively remapped to the target grid.
    - IMERG : mean of the ``S=H:00`` and ``S=H:30`` half-hour rate files.
    - Model : ``accum(+0(H+1)) - accum(+0H)`` of SURFPREC.EAU.CON+GEC (leads
              0..24), deaccumulated per day, conservatively remapped.
* Regridding: ``cdo remapcon`` (1st-order conservative) for every rainfall
  product (model source grids are curvilinear; we write computed corner bounds).

Run under the ``epygram`` conda env (faxarray for FA reads).  CDO is invoked by
absolute path (no module load needed).  See
examples/run_precip_conservative_common_valid.sh.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CDO = "/home/eb/.local/easybuild/software/CDO/1.9.9-gompi-2019b/bin/cdo"

from alaro_analysis.common.constants import RUNS_ROOT
RAINFALL_ROOT = RUNS_ROOT / "rainfall-regridded-to-imerge"
DEFAULT_WORK_ROOT = RUNS_ROOT / "rainfall-conservative-rebuild"
DEFAULT_RAW_IMERG = Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/IMERG_AMAZON")
DEFAULT_FA_ROOT = RUNS_ROOT / "ALARO"
# Trusted gap-filled hourly radar (start-labelled, native 500x500, valid_time_mask).
DEFAULT_RADAR_HOURLY = (
    RAINFALL_ROOT / "sipam" / "Manaus_Radar_Rainfall_2014-15-hourly-masked.nc"
)

# Target box (== historical cdo-grid.sh) and a wider source crop (margin for the
# conservative stencil) for the model native grid.
TARGET_BBOX = (-61.5, -58.5, -5.0, -1.5)        # lon_min, lon_max, lat_min, lat_max
SOURCE_CROP_BBOX = (-62.0, -58.0, -5.5, -1.0)

DEFAULT_START = "2014-01-01T00:00:00"
DEFAULT_END = "2015-12-31T23:00:00"

# key, plot-label, FA experiment sub-directory
MODEL_EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    ("control", "C1M", "control"),
    ("graupel", "G1M", "graupel"),
    ("2mom", "G2M", "2mom"),
    ("no3m", "G2M-XCU", "NO3M"),
)
SURFPREC_VARS = ("SURFPREC.EAU.CON", "SURFPREC.EAU.GEC")
FA_FILE = "pfABOFABOF+{lead:04d}"
DAY_RE = re.compile(r"^pf(\d{8})$")
IMERG_RE = re.compile(r"3IMERG\.(?P<date>\d{8})-S(?P<s>\d{6})-E\d{6}")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def run_cdo(*args: str) -> None:
    proc = subprocess.run([CDO, "-s", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cdo {' '.join(args)}\n{proc.stderr[-2000:]}")


def hourly_axis(start: str, end: str) -> np.ndarray:
    s = np.datetime64(start, "ns")
    e = np.datetime64(end, "ns")
    return np.arange(s, e + np.timedelta64(1, "h"), np.timedelta64(1, "h"))


def corners_from_centers(c: np.ndarray) -> np.ndarray:
    """Cell-corner coordinates (ny+1, nx+1) from 2-D centers via midpoints."""
    cp = np.empty((c.shape[0] + 2, c.shape[1] + 2), dtype=np.float64)
    cp[1:-1, 1:-1] = c
    cp[1:-1, 0] = 2 * c[:, 0] - c[:, 1]
    cp[1:-1, -1] = 2 * c[:, -1] - c[:, -2]
    cp[0, :] = 2 * cp[1, :] - cp[2, :]
    cp[-1, :] = 2 * cp[-2, :] - cp[-3, :]
    return 0.25 * (cp[:-1, :-1] + cp[:-1, 1:] + cp[1:, :-1] + cp[1:, 1:])


def write_curvilinear_nc(
    path: Path,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    varname: str,
    units: str,
) -> None:
    """Write a CF curvilinear (time, y, x) field with computed corner bounds."""
    ny, nx = lon.shape
    plon = corners_from_centers(lon)
    plat = corners_from_centers(lat)
    lon_b = np.stack([plon[:-1, :-1], plon[:-1, 1:], plon[1:, 1:], plon[1:, :-1]], -1)
    lat_b = np.stack([plat[:-1, :-1], plat[:-1, 1:], plat[1:, 1:], plat[1:, :-1]], -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.createDimension("nv4", 4)
        tv = nc.createVariable("time", "f8", ("time",))
        tv.units = "hours since 2014-01-01 00:00:00"
        tv.calendar = "standard"
        pyt = [datetime.utcfromtimestamp((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")) for t in times]
        tv[:] = date2num(pyt, units=tv.units, calendar=tv.calendar)
        vlon = nc.createVariable("lon", "f8", ("y", "x"))
        vlon.units = "degrees_east"
        vlon.standard_name = "longitude"
        vlon.bounds = "lon_bnds"
        vlat = nc.createVariable("lat", "f8", ("y", "x"))
        vlat.units = "degrees_north"
        vlat.standard_name = "latitude"
        vlat.bounds = "lat_bnds"
        nc.createVariable("lon_bnds", "f8", ("y", "x", "nv4"))[:] = lon_b
        nc.createVariable("lat_bnds", "f8", ("y", "x", "nv4"))[:] = lat_b
        vr = nc.createVariable(varname, "f4", ("time", "y", "x"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        vr.coordinates = "lat lon"
        vr.units = units
        vlon[:] = lon
        vlat[:] = lat
        vr[:] = data.astype(np.float32)


def list_day_dirs(exp_root: Path, max_days: int | None) -> list[Path]:
    days = sorted(p for p in exp_root.iterdir() if p.is_dir() and DAY_RE.match(p.name))
    return days[:max_days] if max_days is not None else days


# --------------------------------------------------------------------------- #
# Stage 1 - target grid
# --------------------------------------------------------------------------- #


def make_target_grid(work: Path, raw_imerg: Path) -> Path:
    grid_txt = work / "grid.txt"
    sample = sorted(raw_imerg.glob("*.nc4"))[0]
    crop = work / "imerg_grid_sample.nc"
    work.mkdir(parents=True, exist_ok=True)
    lo, hi, la, ha = TARGET_BBOX
    run_cdo(f"-sellonlatbox,{lo},{hi},{la},{ha}", str(sample), str(crop))
    with grid_txt.open("w") as fh:
        proc = subprocess.run([CDO, "-s", "griddes", str(crop)], stdout=fh, text=True)
    if proc.returncode != 0:
        raise RuntimeError("cdo griddes failed")
    return grid_txt


def target_lonlat(work: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(work / "imerg_grid_sample.nc") as ds:
        return np.asarray(ds["lon"].values), np.asarray(ds["lat"].values)


# --------------------------------------------------------------------------- #
# Stage 2 - IMERG hourly (mean of S=H:00 and S=H:30 half-hours), start-labelled
# --------------------------------------------------------------------------- #


def index_imerg(raw_imerg: Path) -> dict[np.datetime64, Path]:
    index: dict[np.datetime64, Path] = {}
    for p in raw_imerg.glob("*.nc4"):
        m = IMERG_RE.search(p.name)
        if not m:
            continue
        dt = np.datetime64(datetime.strptime(m.group("date") + m.group("s"), "%Y%m%d%H%M%S"), "ns")
        index[dt] = p
    return index


def _imerg_hour(args):
    hour, p0, p30, lo, hi, la, ha = args
    vals = []
    for p in (p0, p30):
        if p is None:
            return hour, None
        with xr.open_dataset(p) as ds:
            da = ds["precipitation"].isel(time=0)
            da = da.sel(lon=slice(lo, hi), lat=slice(la, ha))
            v = np.asarray(da.values, dtype=np.float64)
            if da.dims[0] == "lon":   # IMERG is (lon, lat); want (lat, lon)
                v = v.T
        vals.append(np.where(v < 0, np.nan, v))
    return hour, np.nanmean(np.stack(vals), axis=0)


def build_imerg_hourly(work: Path, raw_imerg: Path, times: np.ndarray, workers: int) -> Path:
    out = work / "IMERG_hourly.nc"
    lon2d, lat2d = target_lonlat(work)
    lon1d = lon2d if lon2d.ndim == 1 else lon2d[0, :]
    lat1d = lat2d if lat2d.ndim == 1 else lat2d[:, 0]
    print(f"[imerg] indexing half-hour files under {raw_imerg} ...", flush=True)
    index = index_imerg(raw_imerg)
    lo, hi, la, ha = TARGET_BBOX
    tasks = [
        (i, index.get(t), index.get(t + np.timedelta64(30, "m")), lo, hi, la, ha)
        for i, t in enumerate(times)
    ]
    nlat, nlon = lat1d.size, lon1d.size
    data = np.full((len(times), nlat, nlon), np.nan, dtype=np.float32)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for hour, arr in pool.map(_imerg_hour, tasks, chunksize=24):
            if arr is not None:
                data[hour] = arr
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                print(f"[imerg] {done}/{len(tasks)} hours", flush=True)
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat1d, "lon": lon1d},
    )
    ds["precipitation"].attrs["units"] = "mm/hr"
    ds.to_netcdf(out, encoding={"precipitation": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}})
    print(f"[imerg] wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Stage 3 - radar conservative regrid (carry valid_time_mask)
# --------------------------------------------------------------------------- #


def regrid_radar(work: Path, radar_hourly: Path, grid_txt: Path) -> Path:
    out = work / "Radar_to_imerg.nc"
    local = work / "radar_hourly_source.nc"
    if not local.exists():
        subprocess.run(["cp", str(radar_hourly), str(local)], check=True)
    rain = work / "_radar_rain.nc"
    run_cdo(f"-remapcon,{grid_txt}", "-selname,rainfall_rate", str(local), str(rain))
    with xr.open_dataset(rain) as dr, xr.open_dataset(local) as ds:
        merged = dr[["rainfall_rate"]]
        if "valid_time_mask" in ds:
            merged = merged.assign(valid_time_mask=("time", np.asarray(ds["valid_time_mask"].values)))
        merged.to_netcdf(out)
    rain.unlink(missing_ok=True)
    print(f"[radar] wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Stage 4 - model hourly from FA deaccumulation + conservative regrid
# --------------------------------------------------------------------------- #


def _fa_window(sample_file: Path) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    import faxarray as fx

    with fx.open_dataset(str(sample_file), variables=["SURFPREC.EAU.CON"]) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lo, hi, la, ha = SOURCE_CROP_BBOX
    m = (lon >= lo) & (lon <= hi) & (lat >= la) & (lat <= ha)
    ys, xs = np.where(m)
    sy = slice(int(ys.min()), int(ys.max()) + 1)
    sx = slice(int(xs.min()), int(xs.max()) + 1)
    return sy, sx, lon[sy, sx], lat[sy, sx]


def _model_day(args):
    """Return (date_str, (24, ny, nx)) hourly total rain for one day, or None."""
    import faxarray as fx

    day_dir, sy, sx = args
    day_dir = Path(day_dir)
    acc = []
    try:
        for lead in range(25):
            with fx.open_dataset(str(day_dir / FA_FILE.format(lead=lead)), variables=list(SURFPREC_VARS)) as ds:
                cn = [k for k in ds.data_vars if "CON" in k.upper()][0]
                gn = [k for k in ds.data_vars if "GEC" in k.upper()][0]
                tot = (
                    np.asarray(ds[cn].isel(time=0).values, dtype=np.float64)
                    + np.asarray(ds[gn].isel(time=0).values, dtype=np.float64)
                )[sy, sx]
            acc.append(tot)
    except Exception as exc:  # noqa: BLE001
        return day_dir.name, None, str(exc)
    acc = np.stack(acc)                       # (25, ny, nx) accumulated
    hourly = np.diff(acc, axis=0)             # (24, ny, nx) -> rain in [H,H+1), label H
    hourly = np.where(hourly < 0, 0.0, hourly)
    return DAY_RE.match(day_dir.name).group(1), hourly.astype(np.float32), None


def build_model_hourly(work, fa_root, exp_key, label, fa_sub, times, grid_txt, max_days, workers) -> Path:
    out = work / f"{label}_to_imerg.nc"
    exp_root = fa_root / fa_sub / "untar-output"
    days = list_day_dirs(exp_root, max_days)
    if not days:
        raise RuntimeError(f"No FA day dirs under {exp_root}")
    sy, sx, lon, lat = _fa_window(days[0] / FA_FILE.format(lead=0))
    ny, nx = lon.shape
    native = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
    tindex = {np.datetime64(t, "h"): i for i, t in enumerate(times)}
    print(f"[{label}] {len(days)} days; FA window {ny}x{nx}; {workers} workers", flush=True)

    tasks = [(str(d), sy, sx) for d in days]
    done = 0
    warns: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_model_day, t) for t in tasks]
        for fut in as_completed(futs):
            date_str, hourly, err = fut.result()
            done += 1
            if err is not None or hourly is None:
                warns.append(f"{date_str}: {err}")
            else:
                day0 = np.datetime64(datetime.strptime(date_str, "%Y%m%d"), "h")
                for h in range(24):
                    idx = tindex.get(day0 + np.timedelta64(h, "h"))
                    if idx is not None:
                        native[idx] = hourly[h]
            if done % 50 == 0 or done == len(tasks):
                print(f"[{label}] {done}/{len(tasks)} days", flush=True)
    if warns:
        (work / f"{label}_warnings.txt").write_text("\n".join(warns) + "\n")

    native_nc = work / f"{label}_native_hourly.nc"
    write_curvilinear_nc(native_nc, lon=lon, lat=lat, times=times, data=native,
                         varname="total_rain", units="mm")
    run_cdo(f"-remapcon,{grid_txt}", str(native_nc), str(out))
    print(f"[{label}] wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Stage 5 - radar mask + common-valid time selection
# --------------------------------------------------------------------------- #


def _load_on_axis(path: Path, var: str, times: np.ndarray) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        da = ds[var].reindex(time=times)
    return da.load()


def radar_spatial_mask(work: Path, grid_txt: Path, method: str) -> np.ndarray:
    """Project the (already-masked) native radar footprint onto the target grid.

    method='conservative' : keep a target cell if ANY of its area overlaps the
                            radar's native 60%-coverage mask (cdo remapcon output).
    method='bilinear'     : keep a target cell only where a BILINEAR remap of the
                            radar is finite (matches the historical ~364-cell mask).
    The rainfall DATA is always conservative (remapcon); this only sets the mask.
    """
    if method == "bilinear":
        bfile = work / "Radar_bilinear_mask.nc"
        if not bfile.exists():
            run_cdo(f"-remapbil,{grid_txt}", "-selname,rainfall_rate",
                    str(work / "radar_hourly_source.nc"), str(bfile))
        src = bfile
    else:
        src = work / "Radar_to_imerg.nc"
    with xr.open_dataset(src) as ds:
        return np.isfinite(ds["rainfall_rate"].values).any(axis=0)


def build_common_valid(work: Path, times: np.ndarray, grid_txt: Path,
                       *, mask_method: str = "conservative",
                       out_subdir: str = "common-valid-time-production") -> dict[str, int]:
    common_dir = work / out_subdir
    common_dir.mkdir(parents=True, exist_ok=True)

    radar = _load_on_axis(work / "Radar_to_imerg.nc", "rainfall_rate", times)
    with xr.open_dataset(work / "Radar_to_imerg.nc") as ds:
        vtm = ds["valid_time_mask"].reindex(time=times).fillna(0).values if "valid_time_mask" in ds else None
    imerg = _load_on_axis(work / "IMERG_hourly.nc", "precipitation", times)
    models = {label: _load_on_axis(work / f"{label}_to_imerg.nc", "total_rain", times)
              for _, label, _ in MODEL_EXPERIMENTS if (work / f"{label}_to_imerg.nc").exists()}

    # radar spatial mask = native 60% mask projected onto the target grid
    spatial = radar_spatial_mask(work, grid_txt, mask_method)

    fields = {"Radar": radar, "IMERG": imerg, **models}
    fields = {k: v.where(xr.DataArray(spatial, dims=("lat", "lon"))) for k, v in fields.items()}

    def finite_times(da: xr.DataArray) -> np.ndarray:
        # a time is usable if it has >=1 finite cell within the radar mask
        return np.isfinite(da.values).reshape(da.shape[0], -1).sum(axis=1) > 0

    valid = {k: finite_times(v) for k, v in fields.items()}
    if vtm is not None:
        valid["Radar"] = valid["Radar"] & (np.asarray(vtm) > 0)

    common = np.ones(len(times), dtype=bool)
    for v in valid.values():
        common &= v

    counts = {f"valid:{k}": int(v.sum()) for k, v in valid.items()}
    counts["radar_valid_time_mask"] = int(np.sum(np.asarray(vtm) > 0)) if vtm is not None else -1
    counts["common_valid"] = int(common.sum())
    counts["radar_spatial_cells"] = int(spatial.sum())

    ctimes = times[common]
    for k, v in fields.items():
        sub = v.isel(time=np.where(common)[0])
        sub = sub.assign_coords(time=ctimes)
        sub.to_dataset(name="rainfall").to_netcdf(common_dir / f"{k}_common_valid.nc")
    return counts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_ROOT)
    p.add_argument("--raw-imerg", type=Path, default=DEFAULT_RAW_IMERG)
    p.add_argument("--fa-root", type=Path, default=DEFAULT_FA_ROOT)
    p.add_argument("--radar-hourly", type=Path, default=DEFAULT_RADAR_HOURLY)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--experiments", nargs="+", default=[k for k, _, _ in MODEL_EXPERIMENTS])
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--stages", nargs="+",
                   default=["grid", "imerg", "radar", "model", "common"],
                   choices=["grid", "imerg", "radar", "model", "common"])
    p.add_argument("--recompute", action="store_true",
                   help="Rebuild stage outputs even if they already exist.")
    p.add_argument("--mask-method", choices=["conservative", "bilinear"], default="conservative",
                   help="How to project the native radar mask onto the target grid "
                        "(rainfall data is always conservative).")
    p.add_argument("--common-subdir", default="common-valid-time-production",
                   help="Subdir under --work-dir for the common-valid output files.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    times = hourly_axis(args.start, args.end)
    print(f"hourly axis: {times[0]} .. {times[-1]}  ({len(times)} hours)", flush=True)

    if "grid" in args.stages:
        make_target_grid(work, args.raw_imerg)
        print(f"[grid] wrote {work / 'grid.txt'}", flush=True)
    grid_txt = work / "grid.txt"

    if "imerg" in args.stages:
        if args.recompute or not (work / "IMERG_hourly.nc").exists():
            build_imerg_hourly(work, args.raw_imerg, times, args.workers)
        else:
            print("[imerg] IMERG_hourly.nc exists, skip", flush=True)
    if "radar" in args.stages:
        if args.recompute or not (work / "Radar_to_imerg.nc").exists():
            regrid_radar(work, args.radar_hourly, grid_txt)
        else:
            print("[radar] Radar_to_imerg.nc exists, skip", flush=True)
    if "model" in args.stages:
        for key, label, fa_sub in MODEL_EXPERIMENTS:
            if key not in args.experiments:
                continue
            if not args.recompute and (work / f"{label}_to_imerg.nc").exists():
                print(f"[{label}] {label}_to_imerg.nc exists, skip", flush=True)
                continue
            build_model_hourly(work, args.fa_root, key, label, fa_sub, times,
                               grid_txt, args.max_days, args.workers)
    if "common" in args.stages:
        counts = build_common_valid(work, times, grid_txt,
                                    mask_method=args.mask_method,
                                    out_subdir=args.common_subdir)
        print("\n================ TIMESTEP COUNTS ================", flush=True)
        for k, v in counts.items():
            print(f"  {k:28s}: {v}", flush=True)
        (work / "timestep_counts.txt").write_text(
            "\n".join(f"{k},{v}" for k, v in counts.items()) + "\n")
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
