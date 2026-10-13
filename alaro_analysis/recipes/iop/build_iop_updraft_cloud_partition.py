#!/usr/bin/env python3
"""Diurnal height-hour sections of updraft extent, mass flux and cloud cover, all four runs.

Same domain and binning as ``build_iop_convection_diurnal`` (domain mean over the
SIPAM radar domain on the native 4 km grid, every cell counting, binned by Amazon
local hour over the campaign days), with two changes:

1. The mass flux is the total upward flux of the column split into 3MT updraft,
   downdraft and environment (``data.convection_sections.column_fluxes``), the way
   the supervisor defines it: UD_OMEGA is the updraft velocity RELATIVE to its
   environment (ACCVUD; Gerard et al. 2009, eq. 4), DD_OMEGA is absolute (ACMODO)
   and the resolved VITESSE_VERT is the area mean of all three. The older
   ``ud_flux`` took UD_OMEGA as absolute and counted the updraft piece alone.
2. G2M-XCU is included. With 3MT off its column is one piece, so its flux is the
   resolved ascent; its parameterized quantities (updraft extent, updraft piece)
   are NaN, not zero.

Per IOP and run, cached as ``<LABEL>_partition_<quantity>.npz`` (87 levels x 24
local hours, with counts):

    ud_extent  mean(max(UD_MESH_FRAC, 0))                  fraction   (NaN for XCU)
    flux_up    total upward mass flux of the three pieces  kg m-2 s-1
    flux_par   the 3MT updraft piece alone                 kg m-2 s-1 (NaN for XCU)
    flux_res   rising resolved grid-mean motion alone      kg m-2 s-1
    cloud      mean(CLOUD_FRACTI)                          fraction
    temp       mean(TEMPERATURE)                           K, for the freezing line

Needs the epygram environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.convection_sections import (
    N_LEVELS,
    PARTITION_QUANTITIES,
    build_window,
    fa_day_partition,
    fa_height,
    init_worker,
)
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR,
    FA_WORKERS,
    campaign_days,
    reduce_days,
)
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS, NO_SCHEME


def build(work: Path, iop: int, key: str, workers: int, attempts: int) -> None:
    root = MODEL_RUNS[key][iop]
    days = campaign_days(iop)
    label = EXPERIMENT_LABELS[key]
    has_scheme = key not in NO_SCHEME
    tasks = [(str(root), f"{day:%Y%m%d}", has_scheme) for day in days]

    window = build_window()
    init_worker(window)  # the parent needs the window for the height profile
    pool_workers = min(workers, FA_WORKERS)
    print(
        f"[IOP{iop} {label}] {len(days)} days from {root}, {int(window[4].sum())} domain cells, "
        f"scheme={has_scheme}, {pool_workers} workers",
        flush=True,
    )
    sums, counts, n_files, problems = reduce_days(
        fa_day_partition, tasks, pool_workers, attempts, initargs=window,
        quantities=PARTITION_QUANTITIES,
    )

    out = work / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    for quantity in PARTITION_QUANTITIES:
        mean = np.full((N_LEVELS, 24), np.nan)
        seen = counts[quantity] > 0
        mean[seen] = sums[quantity][seen] / counts[quantity][seen]
        save_npz(
            out / f"{label}_partition_{quantity}.npz",
            mean=mean,
            counts=counts[quantity],
            n_files=n_files,
        )
    height = fa_height(Path(root), f"{days[0]:%Y%m%d}")
    save_npz(out / f"{label}_partition_height.npz", height_m=height, n_files=n_files)
    if problems:
        (out / f"{label}_partition_problems.log").write_text("\n".join(problems) + "\n")
    print(f"[IOP{iop} {label}] {n_files} files, {len(problems)} problems -> {out}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing caches.")
    return parse_configured_args(p, "build_iop_updraft_cloud_partition", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        for key in args.experiments:
            done = args.work_dir / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}_partition_height.npz"
            if done.exists() and not args.recompute:
                print(f"[IOP{iop} {EXPERIMENT_LABELS[key]}] cached, skip", flush=True)
                continue
            build(args.work_dir, iop, key, workers, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
