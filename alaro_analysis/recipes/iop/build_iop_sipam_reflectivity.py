#!/usr/bin/env python3
"""SIPAM radar reflectivity on the 4 km model grid for the GoAmazon IOPs.

Every 12-min CAPPI volume (2..16 km) of the IOP days whose hour passed the radar
valid-time check of the rainfall verification (``Radar_to_imerg.nc``
``valid_time_mask``) is averaged in linear Z onto the cells of the native 4 km radar
domain window (``data.sipam_reflectivity``), so radar and model reflectivity are
compared at the same resolution. One file per day:
``<cache>/sipam/iop<n>/SIPAM_<YYYYMMDD>.nc``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.data.sipam_reflectivity import build_map, init_worker, process_day
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import campaign_days
from alaro_analysis.recipes.iop.build_iop_microphysics_fields import DEFAULT_WORK_DIR as FIELDS_DIR

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "sipam"
RADAR_VALID = CACHE_DIRS["rainfall"] / "Radar_to_imerg.nc"


def valid_hours() -> set[str]:
    with xr.open_dataset(RADAR_VALID) as ds:
        times = np.asarray(ds["time"].values, dtype="datetime64[h]")
        ok = np.asarray(ds["valid_time_mask"].values) > 0
    return {str(t).replace("-", "").replace("T", "")[:10] for t in times[ok]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=12, help="I/O bound: ~2.4 GB of raw volumes per day")
    return parse_configured_args(p, "build_iop_sipam_reflectivity", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with xr.open_dataset(FIELDS_DIR / "grid.nc") as grid:
        mapping = build_map(grid["lon"].values, grid["lat"].values, grid["inside"].values.astype(bool))
    hours = valid_hours()
    for iop in args.iops:
        tasks = []
        for day in campaign_days(iop):
            token = f"{day:%Y%m%d}"
            day_hours = sorted(h for h in hours if h.startswith(token))
            tasks.append((token, day_hours, str(args.work_dir / f"iop{iop}" / f"SIPAM_{token}.nc")))
        n_scans = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(mapping,)) as pool:
            for future in as_completed([pool.submit(process_day, t) for t in tasks]):
                day, n = future.result()
                n_scans += max(n, 0)
        print(f"[IOP{iop}] {len(tasks)} days, {n_scans} scans written -> {args.work_dir / f'iop{iop}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
