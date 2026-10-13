#!/usr/bin/env python3
"""Dominant hydrometeor species by height in the four runs, for the radar comparison.

Counterpart of ``build_iop_xband_hid``: where the X-band classifier labels each radar
gate, this labels each model cell with the species holding most of its precipitating
mass, counted only where simulated reflectivity exceeds the same 0 dBZ floor applied
to the radar, inside the radar domain mask of the model fields.

Writes ``<cache>/xband/model_species_<LABEL>_iop<N>.npz`` (height x species counts).
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.model_species import HEIGHT_EDGES_KM, SPECIES, species_counts_day

FIELDS = CACHE_DIRS["microphysics"] / "fields"
RUNS = ("C1M", "G1M", "G2M", "G2M-XCU")


def build(label: str, iop: int, work: Path, workers: int, inside_only: bool) -> None:
    days = sorted((FIELDS / label / f"iop{iop}").glob(f"{label}_*.nc"))
    if not days:
        print(f"[IOP{iop} {label}] no field files, skipped", flush=True)
        return
    tasks = [(str(path), inside_only) for path in days]
    counts = np.zeros((HEIGHT_EDGES_KM.size - 1, len(SPECIES)), dtype=np.int64)
    cells = 0
    problems: list[str] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for day_counts, n, problem in pool.map(species_counts_day, tasks):
            counts += day_counts
            cells += n
            if problem:
                problems.append(problem)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"model_species_{label}_iop{iop}.npz"
    save_npz(out, counts=counts, height_edges_km=HEIGHT_EDGES_KM,
             species=np.array(SPECIES), n_days=np.array([len(days)]), n_cells=np.array([cells]))
    share = counts.sum(axis=0) / max(counts.sum(), 1)
    print(f"[IOP{iop} {label}] {len(days)} days, {cells:,} detectable cells -> {out}\n  "
          + ", ".join(f"{s} {share[i]:.1%}" for i, s in enumerate(SPECIES)), flush=True)
    if problems:
        (work / f"model_species_{label}_iop{iop}_problems.log").write_text("\n".join(problems))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=CACHE_DIRS["microphysics"] / "xband")
    p.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--iops", nargs="+", type=int, default=[1])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--whole-window", action="store_true",
                   help="count the whole 75x74 crop, not just the radar-domain cells")
    return parse_configured_args(p, "build_iop_model_species", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        for label in args.runs:
            build(label, iop, args.work_dir, workers, not args.whole_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
