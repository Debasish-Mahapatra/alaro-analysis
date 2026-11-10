#!/usr/bin/env python3
"""The four runs sampled like the T3 radar wind profiler, at the hours it observed.

Model side of both comparisons in the supervisor's note (``data.rwp_model_sample``):
mass flux of convective columns in 20 km boxes, each column split into 3MT updraft,
downdraft and the rest, only at levels with model echo (``MODEL_ECHO_DBZ``; 10 and 25 dBZ
as checks), for several definitions of a convective column, with the w histograms of the pieces (for the noise
test and the CFAD comparison) and of the resolved w (for the time-averaged profiler).

Snapshots are taken only where the profiler observed the hour that ends at them
(``build_iop_rwp_velocity``, at least half the profiles with data), so the IOP1 sample
starts on 5 March, when the profiler data start, and gaps are skipped in both IOPs.

Writes ``<cache>/rwp_model/iop<n>/<LABEL>.npz``, levels surface first.

``--box-km 60`` builds the supervisor's second scale instead: 60 km boxes over three
consecutive hourly snapshots (``fa_day_rwp_blocks``), for the 3 h windows the profiler
observed (on average at least half the profiles with data over the three hours), written
to ``<cache>/rwp_model_60km/iop<n>/<LABEL>.npz``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.convection_sections import N_LEVELS, build_window, init_worker
from alaro_analysis.data.rwp_model_sample import (
    HIST_KEYS,
    LEVEL_KEYS,
    SCALAR_KEYS,
    VARIANTS,
    BLOCK_SNAPSHOTS,
    add_sums,
    empty_sums,
    fa_day_rwp,
    fa_day_rwp_blocks,
)
from alaro_analysis.data.rwp_profiler import AVAILABLE_MIN, W_EDGES
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import FA_WORKERS, campaign_days
from alaro_analysis.recipes.iop.build_iop_rwp_mass_flux import DEFAULT_WORK_DIR as RWP_WORK_DIR
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS, NO_SCHEME

DEFAULT_WORK_DIR = CACHE_DIRS["convection"] / "rwp_model"
WORK_DIRS = {20: DEFAULT_WORK_DIR, 60: CACHE_DIRS["convection"] / "rwp_model_60km"}
FIELDS_DIR = CACHE_DIRS["microphysics"] / "fields"


def observed_leads(rwp_dir: Path, iop: int) -> dict[date, tuple[int, ...]]:
    """Day -> the leads whose preceding hour the profiler observed."""
    with np.load(rwp_dir / f"rwp_velocity_iop{iop}.npz") as cache:
        days = [date.fromisoformat(str(d)) for d in cache["days"]]
        available = np.asarray(cache["available"], float)
    return {day: tuple(int(h) + 1 for h in np.flatnonzero(row >= AVAILABLE_MIN))
            for day, row in zip(days, available)}


def observed_blocks(rwp_dir: Path, iop: int) -> dict[date, tuple[int, ...]]:
    """Day -> the 3 h blocks [3k, 3k + 3) UTC the profiler observed on average."""
    with np.load(rwp_dir / f"rwp_velocity_iop{iop}.npz") as cache:
        days = [date.fromisoformat(str(d)) for d in cache["days"]]
        available = np.asarray(cache["available"], float)
    blocks = available.reshape(len(days), -1, BLOCK_SNAPSHOTS).mean(axis=2)
    return {day: tuple(int(k) for k in np.flatnonzero(row >= AVAILABLE_MIN))
            for day, row in zip(days, blocks)}


def reduce_days(tasks, workers: int, attempts: int, window, worker=fa_day_rwp) -> tuple[dict, int, list[str]]:
    """Run the day workers, redoing the days a crashed worker took down (halving workers)."""
    total = empty_sums(N_LEVELS)
    problems: list[str] = []
    pending = list(tasks)
    reason = "unknown"
    ctx = mp.get_context("spawn")  # the workers read NetCDF, which does not survive fork here
    for attempt in range(1, attempts + 1):
        crashed = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                 initargs=(window,)) as pool:
            futures = {pool.submit(worker, task): task for task in pending}
            for future in as_completed(futures):
                try:
                    sums, _, day_problems = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker died, retry its day
                    crashed.append(futures[future])
                    reason = f"{futures[future][2]}: {exc!r}"
                    continue
                add_sums(total, sums)
                problems.extend(day_problems)
        print(f"  attempt {attempt}: {len(pending) - len(crashed)}/{len(pending)} days on "
              f"{workers} workers", flush=True)
        if not crashed:
            return total, total["snapshots"], problems
        print(f"  retrying {len(crashed)} days, last failure {reason[:200]}", flush=True)
        pending = crashed
        workers = max(1, workers // 2)
    raise RuntimeError(f"{len(crashed)} days failed after {attempts} attempts; {reason[:300]}")


def build(work: Path, rwp_dir: Path, iop: int, key: str, workers: int, attempts: int,
          box_km: int = 20) -> None:
    label = EXPERIMENT_LABELS[key]
    leads = (observed_leads if box_km == 20 else observed_blocks)(rwp_dir, iop)
    days = [day for day in campaign_days(iop) if leads.get(day)]
    tasks = [(str(MODEL_RUNS[key][iop]), str(FIELDS_DIR / label / f"iop{iop}" / f"{label}_{day:%Y%m%d}.nc"),
              f"{day:%Y%m%d}", key not in NO_SCHEME, leads[day]) for day in days]
    pool_workers = min(workers, FA_WORKERS)
    unit = "profiler hours" if box_km == 20 else "3 h windows"
    print(f"[IOP{iop} {label} {box_km} km] {len(days)} days, {sum(len(t[4]) for t in tasks)} {unit}, "
          f"{pool_workers} workers", flush=True)
    worker = fa_day_rwp if box_km == 20 else fa_day_rwp_blocks
    total, snapshots, problems = reduce_days(tasks, pool_workers, attempts, build_window(), worker)
    out = {"height_agl_m": total["height_agl_sum"][::-1] / max(snapshots, 1),
           "w_edges": W_EDGES, "snapshots": np.array(snapshots),
           "variants": np.array(list(VARIANTS))}
    for name in VARIANTS:
        for k in LEVEL_KEYS + HIST_KEYS:
            out[f"{name}__{k}"] = total[name][k][::-1]  # FA is top first; store surface first
        for k in SCALAR_KEYS:
            out[f"{name}__{k}"] = np.array(total[name][k])
    target = work / f"iop{iop}"
    target.mkdir(parents=True, exist_ok=True)
    save_npz(target / f"{label}.npz", **out)
    if problems:
        (target / f"{label}_problems.log").write_text("\n".join(problems) + "\n")
    print(f"[IOP{iop} {label}] {snapshots} snapshots, {len(problems)} problems -> {target}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=None, help="Default: by --box-km.")
    p.add_argument("--box-km", type=int, default=20, choices=sorted(WORK_DIRS),
                   help="20: 20 km boxes per snapshot; 60: 60 km boxes over three snapshots.")
    p.add_argument("--rwp-dir", type=Path, default=RWP_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing caches.")
    return parse_configured_args(p, "build_iop_rwp_model_sample", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    work = args.work_dir or WORK_DIRS[args.box_km]
    for iop in args.iops:
        for key in args.experiments:
            done = work / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}.npz"
            if done.exists() and not args.recompute:
                print(f"[IOP{iop} {EXPERIMENT_LABELS[key]}] cached, skip", flush=True)
                continue
            build(work, args.rwp_dir, iop, key, workers, args.attempts, args.box_km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
