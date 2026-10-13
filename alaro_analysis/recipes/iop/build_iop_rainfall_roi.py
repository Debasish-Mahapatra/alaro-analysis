#!/usr/bin/env python3
"""Hourly rainfall over the whole ROI for the GoAmazon IOPs, models against IMERG.

IMERG and the four models (C1M, G1M, G2M, G2M-XCU) on the 0.1 deg IMERG grid over
the ROI (-67..-53 E, -10..4 N, 140 x 140 cells), for every hour of both IOPs. No
radar is involved, so there is no radar footprint and no radar valid-time mask;
the only mask is the hours where IMERG and all four models are complete.

The ROI keeps clear of the lateral boundaries: the model window read from FA (the
ROI plus a margin, so every edge cell of the target grid is fully covered by the
conservative remap) must lie inside the C zone, i.e. outside the 8-point
relaxation (I) zone and the extension (E) zone. ``fa_window`` raises otherwise.

Same deaccumulation, ``remapcon`` and IMERG half-hour averaging as the radar-box
set of ``build_iop_rainfall_common_valid``. Hours are UTC and labelled at the
start of the interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import (
    MODEL_SOURCES,
    hourly_axis,
    iop_days,
)
from alaro_analysis.workflows.precip_conservative_common_valid import (
    CDO,
    DEFAULT_RAW_IMERG,
    FA_FILE,
    _imerg_hour,
    _model_day,
    index_imerg,
    run_cdo,
    write_curvilinear_nc,
)

DEFAULT_WORK_DIR = CACHE_DIRS["rainfall"] / "roi"
ROI_BBOX = (-67.0, -53.0, -10.0, 4.0)  # lon_min, lon_max, lat_min, lat_max
# Model cells read around the ROI so the remap has full coverage of the edge cells.
SOURCE_MARGIN_DEG = 0.3
IMERG_FILE = "IMERG_hourly.nc"
COMMON_SUBDIR = "common-valid"
DATASETS = ("IMERG", *MODEL_SOURCES)


def fa_window(sample_file: Path, margin: float = SOURCE_MARGIN_DEG):
    """FA index window of the ROI plus ``margin``; it must sit inside the C zone."""
    import epygram
    import faxarray as fx

    with fx.open_dataset(str(sample_file), variables=["SURFPREC.EAU.CON"]) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lo, hi, la, ha = ROI_BBOX
    inside = (lon >= lo - margin) & (lon <= hi + margin) & (lat >= la - margin) & (lat <= ha + margin)
    ys, xs = np.where(inside)
    sy = slice(int(ys.min()), int(ys.max()) + 1)
    sx = slice(int(xs.min()), int(xs.max()) + 1)

    epygram.init_env()
    resource = epygram.formats.resource(str(sample_file), "r")
    try:
        dims = resource.readfield("SURFPREC.EAU.CON").geometry.dimensions
    finally:
        resource.close()
    c_zone = {
        axis: (int(dims[f"{axis}_CIoffset"]) + int(dims[f"{axis}_Iwidth"]),
               int(dims[f"{axis}_CIoffset"]) + int(dims[f"{axis}_CIzone"]) - int(dims[f"{axis}_Iwidth"]))
        for axis in ("X", "Y")
    }
    margins = {
        "south": sy.start - c_zone["Y"][0],
        "north": c_zone["Y"][1] - sy.stop,
        "west": sx.start - c_zone["X"][0],
        "east": c_zone["X"][1] - sx.stop,
    }
    if min(margins.values()) < 0:
        raise ValueError(f"ROI window reaches the relaxation zone: cells inside the C zone {margins}")
    return sy, sx, lon[sy, sx], lat[sy, sx], margins


def make_target_grid(work: Path, raw_imerg: Path) -> Path:
    """cdo grid description of the IMERG cells whose centres lie in the ROI."""
    grid_txt = work / "grid.txt"
    if grid_txt.exists():
        return grid_txt
    sample = min(raw_imerg.glob("*.nc4"))
    crop = work / "imerg_grid_sample.nc"
    lo, hi, la, ha = ROI_BBOX
    run_cdo(f"-sellonlatbox,{lo},{hi},{la},{ha}", str(sample), str(crop))
    import subprocess

    with grid_txt.open("w") as fh:
        subprocess.run([CDO, "-s", "griddes", str(crop)], stdout=fh, text=True, check=True)
    return grid_txt


def build_imerg(work: Path, raw_imerg: Path, times: np.ndarray, workers: int) -> Path:
    """IMERG hourly over the ROI: mean of the half-hours starting at H:00 and H:30."""
    out = work / IMERG_FILE
    index = index_imerg(raw_imerg)
    lo, hi, la, ha = ROI_BBOX
    tasks = [
        (i, index.get(t), index.get(t + np.timedelta64(30, "m")), lo, hi, la, ha)
        for i, t in enumerate(times.astype("datetime64[ns]"))
    ]
    with xr.open_dataset(work / "imerg_grid_sample.nc") as ds:
        lon, lat = np.asarray(ds["lon"].values), np.asarray(ds["lat"].values)
    data = np.full((times.size, lat.size, lon.size), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for hour, arr in pool.map(_imerg_hour, tasks, chunksize=24):
            if arr is not None:
                data[hour] = arr
    print(f"[IMERG] {int(np.isfinite(data).all(axis=(1, 2)).sum())}/{times.size} complete hours", flush=True)
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    ds["precipitation"].attrs["units"] = "mm/hr"
    ds.to_netcdf(out, encoding={"precipitation": {"zlib": True, "complevel": 4}})
    return out


def build_model(work: Path, label: str, days, times, grid_txt: Path, workers: int, attempts: int):
    """Deaccumulate one model's FA rain over the ROI window and remap it conservatively."""
    day_dirs = [MODEL_SOURCES[label][iop] / f"pf{day:%Y%m%d}" for iop, day in days]
    absent = [p for p in day_dirs if not (p / FA_FILE.format(lead=24)).is_file()]
    if absent:
        raise FileNotFoundError(f"{label}: {len(absent)} day(s) lack lead +0024, e.g. {absent[0]}")
    sy, sx, lon, lat, margins = fa_window(day_dirs[0] / FA_FILE.format(lead=0))
    index = {t: i for i, t in enumerate(times.astype("datetime64[h]"))}
    native = np.full((times.size, *lon.shape), np.nan, dtype=np.float32)
    print(f"[{label}] {len(day_dirs)} days, FA window {lon.shape}, cells inside the C zone "
          f"{margins}, {workers} workers", flush=True)

    pending = day_dirs
    for attempt in range(1, attempts + 1):
        failed = []
        # The FA reader occasionally aborts a worker (glibc heap corruption); the
        # next attempt redoes the days of the broken pool.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_model_day, (str(p), sy, sx)): p for p in pending}
            for future in as_completed(futures):
                try:
                    date_str, hourly, err = future.result()
                except Exception as exc:  # noqa: BLE001
                    date_str, hourly, err = None, None, repr(exc)
                if err is not None or hourly is None:
                    failed.append(futures[future])
                    continue
                day0 = np.datetime64(datetime.strptime(date_str, "%Y%m%d"), "h")
                for h in range(24):
                    native[index[day0 + np.timedelta64(h, "h")]] = hourly[h]
        print(f"[{label}] attempt {attempt}: {len(pending) - len(failed)}/{len(pending)} days", flush=True)
        if not failed:
            break
        pending = failed
    else:
        raise RuntimeError(f"{label}: {len(failed)} day(s) failed after {attempts} attempts")

    native_nc = work / f"{label}_native_hourly.nc"
    write_curvilinear_nc(native_nc, lon=lon, lat=lat, times=times, data=native,
                         varname="total_rain", units="mm")
    out = work / f"{label}_to_imerg.nc"
    run_cdo(f"-remapcon,{grid_txt}", str(native_nc), str(out))
    print(f"[{label}] wrote {out}", flush=True)
    return out


def load(work: Path, name: str, times: np.ndarray) -> np.ndarray:
    """(time, lat, lon) hourly rain of one dataset on the ROI grid, on ``times``."""
    path, var = (work / IMERG_FILE, "precipitation") if name == "IMERG" else (
        work / f"{name}_to_imerg.nc", "total_rain")
    with xr.open_dataset(path) as ds:
        return np.asarray(ds[var].reindex(time=times).values, dtype=np.float32)


def build_common_valid(work: Path, times: np.ndarray) -> dict[str, int]:
    """Keep the hours where every dataset is finite over the whole ROI; split by IOP."""
    fields = {name: load(work, name, times.astype("datetime64[ns]")) for name in DATASETS}
    complete = np.logical_and.reduce([np.isfinite(v).all(axis=(1, 2)) for v in fields.values()])
    with xr.open_dataset(work / "imerg_grid_sample.nc") as ds:
        lon, lat = np.asarray(ds["lon"].values), np.asarray(ds["lat"].values)
    hours = times.astype("datetime64[h]")
    counts = {f"complete_{name}": int(np.isfinite(v).all(axis=(1, 2)).sum()) for name, v in fields.items()}
    counts["common_valid"] = int(complete.sum())
    periods = {COMMON_SUBDIR: complete}
    for iop, (start, end) in GOAMAZON_IOPS.items():
        lo = np.datetime64(start, "h")
        hi = np.datetime64(end, "h") + np.timedelta64(23, "h")
        in_iop = (hours >= lo) & (hours <= hi)
        periods[f"{COMMON_SUBDIR}-iop{iop}"] = complete & in_iop
        counts[f"iop{iop}_hours"] = int(in_iop.sum())
        counts[f"iop{iop}_common_valid"] = int((complete & in_iop).sum())
    for subdir, keep in periods.items():
        target = work / subdir
        target.mkdir(parents=True, exist_ok=True)
        for name, values in fields.items():
            ds = xr.Dataset(
                {"rainfall": (("time", "lat", "lon"), values[keep])},
                coords={"time": times.astype("datetime64[ns]")[keep], "lat": lat, "lon": lon},
            )
            ds["rainfall"].attrs["units"] = "mm h-1"
            ds.attrs["domain"] = f"ROI lon {ROI_BBOX[0]}..{ROI_BBOX[1]}, lat {ROI_BBOX[2]}..{ROI_BBOX[3]}"
            ds.to_netcdf(target / f"{name}_common_valid.nc",
                         encoding={"rainfall": {"zlib": True, "complevel": 4}})
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--raw-imerg", type=Path, default=DEFAULT_RAW_IMERG)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_SOURCES), choices=list(MODEL_SOURCES))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing files.")
    return parse_configured_args(p, "build_iop_rainfall_roi", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(args.workers)
    days = iop_days()
    times = hourly_axis(days)
    print(f"{len(days)} IOP days, {times.size} hours, {times[0]} .. {times[-1]}", flush=True)

    grid_txt = make_target_grid(work, args.raw_imerg)
    if args.recompute or not (work / IMERG_FILE).exists():
        build_imerg(work, args.raw_imerg, times, workers)
    for label in args.experiments:
        if (work / f"{label}_to_imerg.nc").exists() and not args.recompute:
            print(f"[{label}] {label}_to_imerg.nc exists, skip", flush=True)
            continue
        build_model(work, label, days, times, grid_txt, workers, args.attempts)

    counts = build_common_valid(work, times)
    for key, value in counts.items():
        print(f"  {key:24s}: {value}", flush=True)
    (work / "timestep_counts.txt").write_text(
        "\n".join(f"{key},{value}" for key, value in counts.items()) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
