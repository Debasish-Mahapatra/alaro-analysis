#!/usr/bin/env python3
"""Vertical-velocity distributions of the ARM radar wind profiler at T3, per IOP.

The observed side of the vertical-velocity comparison (supervisor's note, section 2),
from the PI product behind Giangrande et al. (2016) (doi:10.5439/1440997):

* w histograms of convective echo, raw (6 s) and averaged over 30 s to 1 h, the
  averaging time standing for a length at the 5 m s-1 cell propagation speed
  (``data.rwp_profiler.BLOCK_SECONDS``); a block is averaged only inside convective echo;
* the retrieval noise, twice over: the spread of w in stratiform rain, where the true w
  is a few tenths of m s-1, and the white-noise part from consecutive 6 s samples;
* which UTC hours the profiler observed, so the model is sampled at the same hours.

Convective samples count where the reflectivity reaches ``--echo-dbz``, by default the
model's echo threshold (``data.rwp_model_sample.MODEL_ECHO_DBZ``).

Writes ``<cache>/rwp/rwp_velocity_iop<n>.npz``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.rwp_model_sample import MODEL_ECHO_DBZ
from alaro_analysis.data.rwp_profiler import BLOCK_SECONDS, W_EDGES, velocity_task
from alaro_analysis.recipes.iop.build_iop_rwp_mass_flux import (
    DEFAULT_WORK_DIR,
    RWP_ROOT,
    period_files,
)

SUMMED = ("hist_conv", "hist_strat", "lag_sq", "lag_n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rwp-root", type=Path, default=RWP_ROOT)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--echo-dbz", type=float, default=MODEL_ECHO_DBZ)
    p.add_argument("--workers", type=int, default=32)
    return parse_configured_args(p, "build_iop_rwp_velocity", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # netCDF/HDF5 does not survive fork on this machine
    for iop in args.iops:
        files = period_files(args.rwp_root, f"iop{iop}")
        rows, failed = [], []
        with ctx.Pool(workers) as pool:
            for row in pool.imap_unordered(velocity_task, [(str(f), args.echo_dbz) for f in files], 2):
                (failed if "error" in row else rows).append(row)
        rows.sort(key=lambda r: r["day"])
        out = {key: sum(r[key] for r in rows) for key in SUMMED}
        out.update(
            height_km=rows[0]["height_km"],
            w_edges=W_EDGES,
            block_seconds=np.asarray(BLOCK_SECONDS, dtype=float),
            days=np.array([r["day"] for r in rows]),
            available=np.stack([r["available"] for r in rows]),
            echo_dbz=np.array(args.echo_dbz),
        )
        path = args.work_dir / f"rwp_velocity_iop{iop}.npz"
        save_npz(path, **out)
        noise = np.sqrt(out["lag_sq"] / np.maximum(out["lag_n"], 1) / 2.0)
        observed = (out["available"] >= 0.5).sum()
        print(f"[IOP{iop}] {len(files)} files, {len(failed)} failed, {observed} observed hours; "
              f"white noise at 1-3 km {np.nanmedian(noise[(rows[0]['height_km'] > 1) & (rows[0]['height_km'] < 3)]):.2f} m/s",
              flush=True)
        print(f"[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
