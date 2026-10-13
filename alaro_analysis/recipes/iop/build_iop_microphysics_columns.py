#!/usr/bin/env python3
"""Column microphysics statistics of the four runs for the GoAmazon IOPs.

Reads the converted fields (``build_iop_microphysics_fields``) and accumulates, over
the 2960 cells of the 4 km SIPAM radar domain and every hourly snapshot, the surface
drop size distribution, hydrometeor and precipitation-flux profiles, the fraction of
falling precipitation that reaches the ground, and cold-pool departures by rain rate
(``data.microphysics_columns``). Writes ``<cache>/columns/<LABEL>_iop<n>.npz``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.microphysics_columns import (
    COLD_RAIN_EDGES,
    DM_EDGES,
    LOGNW_EDGES,
    PROFILE_HEIGHTS_KM,
    RAIN_EDGES,
    SPECIES,
    empty_stats,
    init_worker,
    model_day,
)
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import campaign_days
from alaro_analysis.recipes.iop.build_iop_microphysics_fields import DEFAULT_WORK_DIR as FIELDS_DIR
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "columns"
TWO_MOMENT = {"2mom", "2mom-xcu"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--recompute", action="store_true")
    return parse_configured_args(p, "build_iop_microphysics_columns", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with xr.open_dataset(FIELDS_DIR / "grid.nc") as g:
        inside = g["inside"].values.astype(bool)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for iop in args.iops:
        for key in args.experiments:
            label = EXPERIMENT_LABELS[key]
            out = args.work_dir / f"{label}_iop{iop}.npz"
            if out.exists() and not args.recompute:
                print(f"[IOP{iop} {label}] cached", flush=True)
                continue
            files = [FIELDS_DIR / label / f"iop{iop}" / f"{label}_{d:%Y%m%d}.nc" for d in campaign_days(iop)]
            if not all(f.exists() for f in files):
                print(f"[IOP{iop} {label}] conversion incomplete, skip", flush=True)
                continue
            total = empty_stats()
            tasks = [(str(f), key in TWO_MOMENT) for f in files]
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(inside,)) as pool:
                for part in pool.map(model_day, tasks):
                    for k, v in part.items():
                        total[k] = total[k] + v
            save_npz(out, **total, profile_heights_km=PROFILE_HEIGHTS_KM, dm_edges=DM_EDGES,
                     lognw_edges=LOGNW_EDGES, rain_edges=RAIN_EDGES, cold_rain_edges=COLD_RAIN_EDGES,
                     species=np.array(SPECIES))
            print(f"[IOP{iop} {label}] {len(files)} days -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
