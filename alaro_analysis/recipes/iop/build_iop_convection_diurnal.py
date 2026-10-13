#!/usr/bin/env python3
"""Diurnal height-hour sections of the convection diagnostics for the GoAmazon IOPs.

Per IOP and experiment, the domain mean over the SIPAM radar domain on the native
4 km grid (2960 cells, ``data.radar_domain``) of

    ud_extent  mean(max(UD_MESH_FRAC, 0))                         fraction
    ud_flux    mean(-min(UD_OMEGA, 0) * max(mesh, 0) / g)         kg m-2 s-1, up
    dd_extent  mean(max(DD_MESH_FRAC, 0))                         fraction
    dd_flux    mean( max(DD_OMEGA, 0) * max(mesh, 0) / g)         kg m-2 s-1, down
    w_up       mean(max(w, 0)), w = VERT.VELOCIT                  m s-1
    w_dn       mean(max(-w, 0))                                   m s-1
    cloud      mean(CLOUD_FRACTI)                                 fraction
    temp       mean(TEMPERATURE)                                  K, for the freezing line

binned by Amazon local hour (UTC-4) over the campaign IOP days, as in the paper's
updraft and downdraft figures. Every domain cell counts, with or without an
updraft. The draft fields are cleaned the way the 3MT physics cleans them on
entry (advection overshoots: mesh fraction < 0, sinking updraft omega, rising
downdraft omega all become 0). G2M-XCU is left out on purpose: it has no
parameterized updraft or downdraft.

All three runs are read from raw FA (C1M from the continuous control run, G1M and
G2M from the IOP runs' ``untar-pf``); the masked NetCDF archive is cut to the older
0.1 deg radar mask and is not used. FA stores model level S001 (top) first, so the
profiles are flipped and every cached array runs surface first, with its own height.

The reductions live in ``data.convection_sections``. Needs the epygram environment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, CONTROL_RUN, GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.convection_sections import (
    N_LEVELS,
    QUANTITIES,
    build_window,
    fa_day,
    fa_height,
    init_worker,
)

DEFAULT_WORK_DIR = CACHE_DIRS["convection"]
# Each FA worker holds about 2 GB of ecTrans workspace (VERT.VELOCIT and TEMPERATURE
# are spectral), so the pass stays at 16 workers inside the user memory cgroup (96 GB).
FA_WORKERS = 16
# Experiment -> IOP -> raw FA directory holding pfYYYYMMDD/pfABOFABOF+HHHH.
MODEL_RUNS = {
    "control": {iop: CONTROL_RUN / "untar-output" for iop in GOAMAZON_IOPS},
    "graupel": {iop: IOP_RUNS / f"iop{iop}" / "graupel" / "untar-pf" for iop in GOAMAZON_IOPS},
    "2mom": {iop: IOP_RUNS / f"iop{iop}" / "2mom" / "untar-pf" for iop in GOAMAZON_IOPS},
}


def campaign_days(iop: int) -> list[date]:
    start, end = GOAMAZON_IOPS[iop]
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def reduce_days(worker, tasks, workers: int, attempts: int, initargs=None, quantities=QUANTITIES):
    """Run the day workers, redoing the days a crashed worker took down.

    A whole pool dying means its workers were killed, and each FA worker holds
    about 2 GB of ecTrans workspace, so every retry halves the worker count. The
    failure reason is printed rather than swallowed.
    """
    sums = {q: np.zeros((N_LEVELS, 24)) for q in quantities}
    counts = {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in quantities}
    problems: list[str] = []
    n_files = 0
    pending = list(tasks)
    reason = "unknown"
    for attempt in range(1, attempts + 1):
        crashed = []
        pool_args = (
            {"initializer": init_worker, "initargs": (initargs,)} if initargs is not None else {}
        )
        with ProcessPoolExecutor(max_workers=workers, **pool_args) as pool:
            futures = {pool.submit(worker, task): task for task in pending}
            for future in as_completed(futures):
                try:
                    day_sums, day_counts, files, day_problems = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker died, retry its day
                    crashed.append(futures[future])
                    reason = f"{futures[future][1]}: {exc!r}"
                    continue
                for key in quantities:
                    sums[key] += day_sums[key]
                    counts[key] += day_counts[key]
                n_files += files
                problems.extend(day_problems)
        done = len(pending) - len(crashed)
        print(f"  attempt {attempt}: {done}/{len(pending)} days on {workers} workers", flush=True)
        if not crashed:
            break
        print(f"  retrying {len(crashed)} days, last failure {reason[:200]}", flush=True)
        pending = crashed
        workers = max(1, workers // 2)
    else:
        raise RuntimeError(
            f"{len(crashed)} days failed after {attempts} attempts; last failure {reason[:300]}"
        )
    return sums, counts, n_files, problems


def build(work: Path, iop: int, key: str, workers: int, attempts: int) -> None:
    root = MODEL_RUNS[key][iop]
    days = campaign_days(iop)
    label = EXPERIMENT_LABELS[key]
    first_day = f"{days[0]:%Y%m%d}"
    tasks = [(str(root), f"{day:%Y%m%d}") for day in days]

    window = build_window()
    init_worker(window)  # the parent needs the window for the height profile
    pool_workers = min(workers, FA_WORKERS)
    print(
        f"[IOP{iop} {label}] {len(days)} days from {root}, {int(window[4].sum())} domain cells, "
        f"{pool_workers} workers",
        flush=True,
    )
    sums, counts, n_files, problems = reduce_days(
        fa_day, tasks, pool_workers, attempts, initargs=window
    )

    out = work / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    for quantity in QUANTITIES:
        mean = np.full((N_LEVELS, 24), np.nan)
        seen = counts[quantity] > 0
        mean[seen] = sums[quantity][seen] / counts[quantity][seen]
        save_npz(
            out / f"{label}_{quantity}.npz",
            mean=mean,
            counts=counts[quantity],
            n_files=n_files,
        )
    height = fa_height(Path(root), first_day)
    save_npz(out / f"{label}_height.npz", height_m=height, n_files=n_files)
    if problems:
        (out / f"{label}_problems.log").write_text("\n".join(problems) + "\n")
    print(f"[IOP{iop} {label}] {n_files} files, {len(problems)} problems -> {out}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing caches.")
    return parse_configured_args(p, "build_iop_convection_diurnal", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        for key in args.experiments:
            done = args.work_dir / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}_height.npz"
            if done.exists() and not args.recompute:
                print(f"[IOP{iop} {EXPERIMENT_LABELS[key]}] cached, skip", flush=True)
                continue
            build(args.work_dir, iop, key, workers, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
