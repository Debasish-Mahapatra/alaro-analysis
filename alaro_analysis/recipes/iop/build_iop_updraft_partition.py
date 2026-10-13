#!/usr/bin/env python3
"""Upward and downward convective mass flux for the GoAmazon IOPs, all four runs.

Built for the comparison with the radar wind profiler (Giangrande et al. 2016), as
the observations define it:

1. Per column and level, on the instantaneous hourly snapshots, split the column
   into updraft, downdraft and environment the way the 3MT scheme defines them
   (``data.convection_sections.column_fluxes``): the updraft velocity UD_OMEGA is
   relative to its environment (ACCVUD; Gerard et al. 2009, eq. 4), the downdraft
   velocity DD_OMEGA is absolute (ACMODO), and the resolved VITESSE_VERT is the area
   mean of all three. Upward flux = rising part of each piece times its area; downward
   flux the same for sinking. Without 3MT (G2M-XCU) the column is one piece.
2. Keep only convective columns: rain >= 1 mm in the hour ending at the snapshot and,
   in runs with 3MT, a 3MT updraft (mesh fraction > 1e-3 at some level).
3. Average over 20 km boxes (5 x 5 cells, the observations' 1 h window at 5 m/s),
   dividing by the full box area, over the 116 boxes at least half inside the 4 km
   radar domain; keep only box-hours with a convective column.

Per IOP and run, binned by Amazon local hour (mean over the convective box-hours):

    box_up         upward mass flux, all three pieces                 kg m-2 s-1
    box_up_nodd    the same without the downdraft split
    box_down       downward mass flux, all three pieces (positive down)
    box_par_up     the 3MT updraft piece alone (its area x absolute speed)
    box_res_up     rising resolved grid-mean motion alone (no split)
    box_frequency  fraction of box-hours with convection

The upward flux is not box_par_up + box_res_up: the resolved motion already contains
the updraft (it is the area mean), and the updraft is compensated by sinking in the
rest of the column. Without a downdraft it equals the larger of the two.

Writes ``<LABEL>_<quantity>.npz`` into the convection cache. Needs the epygram environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.convection_sections import (
    BOX_QUANTITIES,
    N_LEVELS,
    build_window,
    fa_day_boxes,
    fa_height,
    init_worker,
)
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR,
    FA_WORKERS,
    MODEL_RUNS as SCHEME_RUNS,
    campaign_days,
    reduce_days,
)

MODEL_RUNS = {
    **SCHEME_RUNS,
    "2mom-xcu": {iop: IOP_RUNS / f"iop{iop}" / "NO3M" / "untar-pf" for iop in GOAMAZON_IOPS},
}
NO_SCHEME = {"2mom-xcu"}


def build(work: Path, iop: int, key: str, workers: int, attempts: int) -> None:
    root = MODEL_RUNS[key][iop]
    days = campaign_days(iop)
    label = EXPERIMENT_LABELS[key]
    has_scheme = key not in NO_SCHEME
    tasks = [(str(root), f"{day:%Y%m%d}", has_scheme) for day in days]
    window = build_window()
    init_worker(window)
    pool_workers = min(workers, FA_WORKERS)
    print(f"[IOP{iop} {label}] {len(days)} days from {root}, scheme={has_scheme}, "
          f"{pool_workers} workers", flush=True)
    sums, counts, n_files, problems = reduce_days(
        fa_day_boxes, tasks, pool_workers, attempts, initargs=window,
        quantities=BOX_QUANTITIES,
    )
    out = work / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    for quantity in BOX_QUANTITIES:
        mean = np.full((N_LEVELS, 24), np.nan)
        seen = counts[quantity] > 0
        mean[seen] = sums[quantity][seen] / counts[quantity][seen]
        save_npz(out / f"{label}_{quantity}.npz", mean=mean, counts=counts[quantity], n_files=n_files)
    if not (out / f"{label}_height.npz").exists():
        save_npz(out / f"{label}_height.npz", height_m=fa_height(Path(root), f"{days[0]:%Y%m%d}"),
                 n_files=n_files)
    if problems:
        (out / f"{label}_updraft_problems.log").write_text("\n".join(problems) + "\n")
    print(f"[IOP{iop} {label}] {n_files} files, {len(problems)} problems -> {out}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(MODEL_RUNS), choices=list(MODEL_RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Rebuild existing caches.")
    return parse_configured_args(p, "build_iop_updraft_partition", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        for key in args.experiments:
            done = args.work_dir / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}_box_up.npz"
            if done.exists() and not args.recompute:
                print(f"[IOP{iop} {EXPERIMENT_LABELS[key]}] cached, skip", flush=True)
                continue
            build(args.work_dir, iop, key, workers, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
