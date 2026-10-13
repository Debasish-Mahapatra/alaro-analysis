#!/usr/bin/env python3
"""Water budget of the four runs over the DDH box around T3, from the raw DDH files.

For every campaign day of each IOP and run, reads the day file (``DHFDLABOF+0024``) and
the 24 hourly lead files (``+0001``..``+0024``, cumulative from 00 UTC) with
``data.ddh_water_budget``. Writes ``<cache>/ddh_budget/<LABEL>_iop<n>.npz`` with

* ``day_<article>``: (days, levels) the day's budget, every article;
* ``hour_<article>``: (days, 24, levels) the budget of each UTC hour (lead h minus
  lead h-1), for the precipitation, evaporation, condensation and conversion articles.

Sources: G1M, G2M, G2M-XCU under ``IOP-runs/iop*/<exp>/untar-ddh``; C1M from the
continuous control run, extracted per campaign day by ``build_iop_closure_budgets``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.ddh_water_budget import FULL_LEVEL, HALF_LEVEL, read_budget
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import campaign_days

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "ddh_budget"
RUNS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M", "NO3M": "G2M-XCU"}
HOURLY = (
    "FQRPLS", "FQRPLC", "FQSPLS", "FQSPLC", "FQGPLS", "FQGPLC",
    "FQRPEVS", "FQRPEVC", "FQSPEVS", "FQSPEVC", "FQGPEVS", "FQGPEVC",
    "FQLCS", "FQLCC", "FQNCS", "FQNCC", "FQLPFPS", "FQNPFPS", "FQGPFPS", "FQVDIFT",
)


def day_dir(iop: int, experiment: str, day) -> Path:
    token = f"DDH{day:%Y%m%d}"
    if experiment == "control":
        return CACHE_DIRS["closure_budget"] / "control-ddh" / token
    return IOP_RUNS / f"iop{iop}" / experiment / "untar-ddh" / token


def read_day(folder: Path) -> tuple[dict | None, np.ndarray | None]:
    day = read_budget(folder / "DHFDLABOF+0024")
    if day is None:
        return None, None
    hourly = np.full((24, len(HOURLY), day["FQRPLS"].size), np.nan)
    previous = None
    for lead in range(1, 25):
        b = day if lead == 24 else read_budget(folder / f"DHFDLABOF+{lead:04d}", HOURLY)
        if b is None:
            return day, None
        current = np.stack([b[name] for name in HOURLY])
        hourly[lead - 1] = current if previous is None else current - previous
        previous = current
    return day, hourly


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--experiments", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--workers", type=int, default=32, help="threads; each read is an lfac subprocess")
    p.add_argument("--recompute", action="store_true")
    return parse_configured_args(p, "build_iop_water_budget", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for iop in args.iops:
        for experiment in args.experiments:
            label = RUNS[experiment]
            out = args.work_dir / f"{label}_iop{iop}.npz"
            if out.exists() and not args.recompute:
                print(f"[IOP{iop} {label}] cached", flush=True)
                continue
            days = campaign_days(iop)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(lambda d: read_day(day_dir(iop, experiment, d)), days))
            kept = [(d, day, hourly) for d, (day, hourly) in zip(days, results) if day is not None and hourly is not None]
            if not kept:
                print(f"[IOP{iop} {label}] no readable days", flush=True)
                continue
            arrays = {f"day_{name}": np.stack([k[1][name] for k in kept]) for name in HALF_LEVEL + FULL_LEVEL}
            hourly = np.stack([k[2] for k in kept])  # (days, 24, fields, levels)
            arrays.update({f"hour_{name}": hourly[:, :, i] for i, name in enumerate(HOURLY)})
            save_npz(out, **arrays, days=np.array([f"{k[0]:%Y%m%d}" for k in kept]))
            print(f"[IOP{iop} {label}] {len(kept)}/{len(days)} days -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
