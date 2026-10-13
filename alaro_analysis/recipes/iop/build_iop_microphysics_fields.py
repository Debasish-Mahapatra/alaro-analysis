#!/usr/bin/env python3
"""Convert the microphysics fields of the GoAmazon IOP runs from raw FA to NetCDF.

One file per run and day, ``<cache>/fields/<LABEL>/iop<n>/<LABEL>_<YYYYMMDD>.nc``,
with the instantaneous hourly snapshots (leads +0001..+0024) on the window of the
native 4 km SIPAM radar domain (``data.radar_domain``; ``inside`` marks its 2960
cells). Only what the microphysics analyses need is converted
(``data.microphysics_fields``): simulated reflectivity, height, temperature,
pressure, humidity, the five hydrometeors, rain and droplet numbers (2-moment runs),
convective and stratiform precipitation fluxes, resolved w, surface precipitation of
the hour by type, 2 m temperature and humidity, CAPE and CIN. ``grid.nc`` holds the
window's lon/lat and mask.

C1M comes from the continuous control run, G1M, G2M and G2M-XCU from the IOP runs.
Needs the epygram environment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.microphysics_fields import convert_day, init_worker
from alaro_analysis.data.radar_domain import MASK_FILE, mask_window
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import campaign_days
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "fields"
# Each worker holds ~2 GB of ecTrans workspace plus a day of snapshots (~0.7 GB); two
# IOP jobs of 12 stay inside the 96 GB user cgroup (2 x 16 overflowed it).
FA_WORKERS = 12


def write_grid(work: Path, window) -> None:
    y0, y1, x0, x1, mask = window
    out = work / "grid.nc"
    if out.exists():
        return
    with xr.open_dataset(MASK_FILE) as ds:
        lon = ds["lon"].values[y0:y1, x0:x1]
        lat = ds["lat"].values[y0:y1, x0:x1]
    work.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"lon": (("y", "x"), lon), "lat": (("y", "x"), lat), "inside": (("y", "x"), mask.astype(np.int8))},
        attrs={"window_fa_index": f"y {y0}:{y1}, x {x0}:{x1}", "cells_inside": int(mask.sum())},
    ).to_netcdf(out)


def build(work: Path, iop: int, key: str, workers: int, attempts: int) -> None:
    label = EXPERIMENT_LABELS[key]
    root = MODEL_RUNS[key][iop]
    out_dir = work / label / f"iop{iop}"
    tasks = [(str(root), f"{day:%Y%m%d}", str(out_dir / f"{label}_{day:%Y%m%d}.nc"))
             for day in campaign_days(iop)]
    window = mask_window()
    pending = [t for t in tasks if not Path(t[2]).exists()]
    print(f"[IOP{iop} {label}] {len(tasks)} days, {len(pending)} to convert, {workers} workers", flush=True)
    for attempt in range(1, attempts + 1):
        if not pending:
            break
        failed, problems = [], []
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(window,)) as pool:
            futures = {pool.submit(convert_day, t): t for t in pending}
            for future in as_completed(futures):
                try:
                    day, n, day_problems = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker died; redo its day
                    failed.append(futures[future])
                    problems.append(f"{futures[future][1]}: {exc!r}")
                    continue
                if n == 0:
                    failed.append(futures[future])
                    problems.extend(day_problems)
        print(f"[IOP{iop} {label}] attempt {attempt}: {len(pending) - len(failed)}/{len(pending)} days"
              + (f"; last problem {problems[-1][:160]}" if problems else ""), flush=True)
        pending = failed
        workers = max(1, workers // 2) if failed else workers
    if pending:
        raise RuntimeError(f"[IOP{iop} {label}] {len(pending)} days failed")
    print(f"[IOP{iop} {label}] done -> {out_dir}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=4)
    return parse_configured_args(p, "build_iop_microphysics_fields", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = min(resolve_workers(args.workers), FA_WORKERS)
    write_grid(args.work_dir, mask_window())
    for iop in args.iops:
        for key in args.experiments:
            build(args.work_dir, iop, key, workers, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
