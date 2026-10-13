#!/usr/bin/env python3
"""Hydrometeor mass-fraction histograms for the GoAmazon IOPs, with no normalisation.

Per IOP and experiment, the plain number of grid-point-hours whose mass fraction
falls in each log10 bin at each model level, over the SIPAM radar domain on the
native 4 km grid (2960 cells, ``data.radar_domain``) on the campaign IOP days. Nothing is rescaled: heights are not stretched to 100 %
as a CFAD would, so heights, species and experiments stay comparable.

All four experiments are built from raw FA: C1M from the continuous control run,
G1M, G2M and G2M-XCU from the IOP runs' ``untar-pf`` (the masked NetCDF archive is
cut to the older 0.1 deg radar mask and is not used). C1M is the 2-ice control and
carries no prognostic graupel (its FA field is identically zero), so graupel is
skipped for it.

Each cache also holds the mean temperature profile, for the 0 degC line, and the
model-level heights. Needs the epygram environment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, CONTROL_RUN, GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.convection_sections import N_LEVELS, build_window, fa_height
from alaro_analysis.data.convection_sections import init_worker as init_fa_worker
from alaro_analysis.data.hydrometeor_histograms import LOG_EDGES, NBINS, fa_day, species_for

DEFAULT_WORK_DIR = CACHE_DIRS["hydrometeors"]
# Each FA worker holds about 2 GB of ecTrans workspace; the cgroup caps at 96 GB.
FA_WORKERS = 16
# Experiment -> IOP -> raw FA directory holding pfYYYYMMDD/pfABOFABOF+HHHH.
MODEL_RUNS = {
    "control": {iop: CONTROL_RUN / "untar-output" for iop in GOAMAZON_IOPS},
    "graupel": {iop: IOP_RUNS / f"iop{iop}" / "graupel" / "untar-pf" for iop in GOAMAZON_IOPS},
    "2mom": {iop: IOP_RUNS / f"iop{iop}" / "2mom" / "untar-pf" for iop in GOAMAZON_IOPS},
    "2mom-xcu": {iop: IOP_RUNS / f"iop{iop}" / "NO3M" / "untar-pf" for iop in GOAMAZON_IOPS},
}


def campaign_days(iop: int):
    start, end = GOAMAZON_IOPS[iop]
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def reduce_days(worker, tasks, species, workers: int, attempts: int, initargs=None):
    """Run the day workers, redoing the days a crashed worker took down.

    A whole pool dying means its workers were killed, so each retry halves the
    worker count and the failure reason is printed rather than swallowed.
    """
    counts = {name: np.zeros((N_LEVELS, NBINS), dtype=np.int64) for name in species}
    temp_sum = np.zeros(N_LEVELS)
    temp_count = np.zeros(N_LEVELS, dtype=np.int64)
    problems: list[str] = []
    n_files = 0
    pending = list(tasks)
    reason = "unknown"
    for attempt in range(1, attempts + 1):
        crashed = []
        pool_args = (
            {"initializer": init_fa_worker, "initargs": (initargs,)}
            if initargs is not None
            else {}
        )
        with ProcessPoolExecutor(max_workers=workers, **pool_args) as pool:
            futures = {pool.submit(worker, task): task for task in pending}
            for future in as_completed(futures):
                try:
                    day_counts, day_temp, day_temp_n, files, day_problems = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker died, retry its day
                    crashed.append(futures[future])
                    reason = f"{futures[future][1]}: {exc!r}"
                    continue
                for name in species:
                    counts[name] += day_counts[name]
                temp_sum += day_temp
                temp_count += day_temp_n
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
    return counts, temp_sum, temp_count, n_files, problems


def build(work: Path, iop: int, key: str, workers: int, attempts: int) -> None:
    root = MODEL_RUNS[key][iop]
    days = campaign_days(iop)
    label = EXPERIMENT_LABELS[key]
    first_day = f"{days[0]:%Y%m%d}"
    species = species_for(key)
    tasks = [(str(root), f"{day:%Y%m%d}", species) for day in days]

    window = build_window()
    init_fa_worker(window)  # the parent needs the window for the height profile
    pool_workers = min(workers, FA_WORKERS)
    print(
        f"[IOP{iop} {label}] {len(days)} days from {root}, {int(window[4].sum())} domain cells, "
        f"{pool_workers} workers, species={list(species)}",
        flush=True,
    )
    counts, temp_sum, temp_count, n_files, problems = reduce_days(
        fa_day,
        tasks,
        species,
        pool_workers,
        attempts,
        initargs=window,
    )

    temperature = np.full(N_LEVELS, np.nan)
    seen = temp_count > 0
    temperature[seen] = temp_sum[seen] / temp_count[seen]
    height = fa_height(Path(root), first_day)
    out = work / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    save_npz(
        out / f"{label}_histograms.npz",
        log_edges=LOG_EDGES,
        height_km=np.asarray(height, dtype=np.float64) / 1000.0,
        temperature_k=temperature,
        n_files=n_files,
        species=np.asarray(species),
        **{f"counts_{name}": counts[name] for name in species},
    )
    if problems:
        (out / f"{label}_problems.log").write_text("\n".join(problems) + "\n")
    total = sum(int(c.sum()) for c in counts.values())
    print(
        f"[IOP{iop} {label}] {n_files} files, {total:,} counted points, "
        f"{len(problems)} problems -> {out}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing caches.")
    return parse_configured_args(p, "build_iop_hydrometeor_histograms", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        for key in args.experiments:
            done = args.work_dir / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}_histograms.npz"
            if done.exists() and not args.recompute:
                print(f"[IOP{iop} {EXPERIMENT_LABELS[key]}] cached, skip", flush=True)
                continue
            build(args.work_dir, iop, key, workers, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
