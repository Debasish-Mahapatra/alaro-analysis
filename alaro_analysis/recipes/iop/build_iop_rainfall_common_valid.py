#!/usr/bin/env python3
"""Common-valid hourly rainfall for the combined GoAmazon2014/5 IOPs.

Radar, IMERG and the four models (C1M, G1M, G2M, G2M-XCU) on the 0.1 deg IMERG
grid over the radar box, for IOP1 and IOP2 combined. Every dataset gets the radar
valid-time mask and the radar footprint (bilinear projection, as in the paper
figures), so all six are compared over the same hours and cells.

Radar and IMERG come from the 2014-15 conservative rebuild. The models are
rebuilt from raw FA with that rebuild's deaccumulation and ``remapcon``: G1M,
G2M and G2M-XCU from the IOP warm-restart runs, C1M from the continuous control
run on the same dates. Hours are UTC and labelled at the start of the interval.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, CONTROL_RUN, GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.common.paths import runs_path
from alaro_analysis.workflows.precip_conservative_common_valid import (
    FA_FILE,
    _fa_window,
    _model_day,
    build_common_valid,
    run_cdo,
    write_curvilinear_nc,
)

REFERENCE_DIR = runs_path("old-archive/rainfall-conservative-rebuild")
# Target grid, regridded radar with its valid_time_mask, IMERG, and the bilinear
# radar footprint, all 2014-15 hourly on the target grid.
REFERENCE_FILES = (
    "grid.txt",
    "imerg_grid_sample.nc",
    "Radar_to_imerg.nc",
    "IMERG_hourly.nc",
    "Radar_bilinear_mask.nc",
)
DEFAULT_WORK_DIR = CACHE_DIRS["rainfall"]
COMMON_SUBDIR = "common-valid-time-production-bilinearmask"
MODEL_SOURCES = {
    "C1M": {iop: CONTROL_RUN / "untar-output" for iop in GOAMAZON_IOPS},
    "G1M": {iop: IOP_RUNS / f"iop{iop}" / "graupel" / "untar-pf" for iop in GOAMAZON_IOPS},
    "G2M": {iop: IOP_RUNS / f"iop{iop}" / "2mom" / "untar-pf" for iop in GOAMAZON_IOPS},
    "G2M-XCU": {iop: IOP_RUNS / f"iop{iop}" / "NO3M" / "untar-pf" for iop in GOAMAZON_IOPS},
}


def iop_days() -> list[tuple[int, date]]:
    """Every (iop, day) of the campaign IOPs, inclusive."""
    days = []
    for iop, (start, end) in GOAMAZON_IOPS.items():
        day = start
        while day <= end:
            days.append((iop, day))
            day += timedelta(days=1)
    return days


def hourly_axis(days: list[tuple[int, date]]) -> np.ndarray:
    """Start-labelled UTC hours of the given days."""
    starts = np.array([np.datetime64(day, "h") for _, day in days])
    hours = starts[:, None] + np.arange(24).astype("timedelta64[h]")
    return hours.ravel().astype("datetime64[ns]")


def build_model(work: Path, label: str, days, times, grid_txt: Path, workers: int, attempts: int):
    """Deaccumulate one model's FA rain on the IOP days and remap it conservatively."""
    day_dirs = [MODEL_SOURCES[label][iop] / f"pf{day:%Y%m%d}" for iop, day in days]
    absent = [p for p in day_dirs if not (p / FA_FILE.format(lead=24)).is_file()]
    if absent:
        raise FileNotFoundError(f"{label}: {len(absent)} day(s) lack lead +0024, e.g. {absent[0]}")
    sy, sx, lon, lat = _fa_window(day_dirs[0] / FA_FILE.format(lead=0))
    index = {t: i for i, t in enumerate(times.astype("datetime64[h]"))}
    native = np.full((times.size, *lon.shape), np.nan, dtype=np.float32)
    print(f"[{label}] {len(day_dirs)} days, FA window {lon.shape}, {workers} workers", flush=True)

    pending = day_dirs
    for attempt in range(1, attempts + 1):
        failed = []
        # The FA reader occasionally aborts a worker (glibc heap corruption). The
        # broken pool fails its remaining days, which the next attempt redoes.
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
        done = len(pending) - len(failed)
        print(f"[{label}] attempt {attempt}: {done}/{len(pending)} days", flush=True)
        if not failed:
            break
        pending = failed
    else:
        raise RuntimeError(f"{label}: {len(failed)} day(s) failed after {attempts} attempts")

    native_nc = work / f"{label}_native_hourly.nc"
    write_curvilinear_nc(
        native_nc, lon=lon, lat=lat, times=times, data=native, varname="total_rain", units="mm"
    )
    out = work / f"{label}_to_imerg.nc"
    run_cdo(f"-remapcon,{grid_txt}", str(native_nc), str(out))
    print(f"[{label}] wrote {out}", flush=True)
    return out


def check_c1m_against_reference(work: Path, reference_dir: Path, times: np.ndarray) -> None:
    """C1M comes from the moved control run; it must match the 2014-15 rebuild."""
    new_path, old_path = work / "C1M_to_imerg.nc", reference_dir / "C1M_to_imerg.nc"
    if not (new_path.exists() and old_path.exists()):
        return
    with xr.open_dataset(new_path) as new, xr.open_dataset(old_path) as old:
        a = new["total_rain"].reindex(time=times).values
        b = old["total_rain"].reindex(time=times).values
    print(
        f"[check] C1M from amazon-control-run vs 2014-15 rebuild, IOP hours: "
        f"max|diff| = {np.nanmax(np.abs(a - b)):.3g} mm",
        flush=True,
    )


def per_iop_counts(work: Path) -> dict[str, int]:
    with xr.open_dataset(work / COMMON_SUBDIR / "Radar_common_valid.nc") as ds:
        common = np.asarray(ds["time"].values, dtype="datetime64[h]")
    with xr.open_dataset(work / "Radar_to_imerg.nc") as ds:
        times = np.asarray(ds["time"].values, dtype="datetime64[h]")
        radar_valid = np.asarray(ds["valid_time_mask"].values) > 0
    counts = {}
    for iop, (start, end) in GOAMAZON_IOPS.items():
        lo = np.datetime64(start, "h")
        hi = np.datetime64(end, "h") + np.timedelta64(23, "h")
        counts[f"iop{iop}_hours"] = 24 * ((end - start).days + 1)
        counts[f"iop{iop}_radar_valid"] = int(radar_valid[(times >= lo) & (times <= hi)].sum())
        counts[f"iop{iop}_common_valid"] = int(np.count_nonzero((common >= lo) & (common <= hi)))
    return counts


def split_by_iop(work: Path) -> dict[str, int]:
    """Write one copy of the common-valid set per IOP, for single-IOP figures."""
    source = work / COMMON_SUBDIR
    counts = {}
    for iop, (start, end) in GOAMAZON_IOPS.items():
        target = work / f"{COMMON_SUBDIR}-iop{iop}"
        target.mkdir(parents=True, exist_ok=True)
        lo = np.datetime64(start, "h")
        hi = np.datetime64(end, "h") + np.timedelta64(23, "h")
        for path in sorted(source.glob("*_common_valid.nc")):
            with xr.open_dataset(path) as ds:
                times = np.asarray(ds["time"].values, dtype="datetime64[h]")
                subset = ds.isel(time=np.flatnonzero((times >= lo) & (times <= hi))).load()
            subset.to_netcdf(target / path.name)
            counts[f"iop{iop}_split_hours"] = int(subset.sizes["time"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="2014-15 conservative rebuild holding the regridded radar and IMERG.",
    )
    p.add_argument(
        "--experiments", nargs="+", default=list(MODEL_SOURCES), choices=list(MODEL_SOURCES)
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing model files.")
    return parse_configured_args(p, "build_iop_rainfall_common_valid", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        if not (work / name).exists():
            shutil.copy2(args.reference_dir / name, work / name)
    grid_txt = work / "grid.txt"

    days = iop_days()
    times = hourly_axis(days)
    print(f"{len(days)} IOP days, {times.size} hours, {times[0]} .. {times[-1]}", flush=True)

    for label in args.experiments:
        if (work / f"{label}_to_imerg.nc").exists() and not args.recompute:
            print(f"[{label}] {label}_to_imerg.nc exists, skip", flush=True)
            continue
        build_model(
            work,
            label,
            days,
            times,
            grid_txt,
            workers=resolve_workers(args.workers),
            attempts=args.attempts,
        )
    check_c1m_against_reference(work, args.reference_dir, times)

    counts = build_common_valid(
        work, times, grid_txt, mask_method="bilinear", out_subdir=COMMON_SUBDIR
    )
    counts.update(per_iop_counts(work))
    counts.update(split_by_iop(work))
    print("\n================ TIMESTEP COUNTS ================", flush=True)
    for key, value in counts.items():
        print(f"  {key:28s}: {value}", flush=True)
    (work / "timestep_counts.txt").write_text(
        "\n".join(f"{key},{value}" for key, value in counts.items()) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
