#!/usr/bin/env python3
"""Observed convective mass flux at T3 from the ARM radar wind profiler.

Rebuilds the profiles of Giangrande et al. (2016) from the PI product
"Calibrated Radar Wind Profiler Precipitation Observations and Vertical Velocity
Retrievals" (doi:10.5439/1440997, ``maorwpcls.YYYYMMDD.cdf``), with the method in
``data.rwp_profiler``: convective echo, every sample counted, area fraction over the
whole window, mass flux Mc = rho (sigma_u v_u + sigma_d v_d), and windows kept when
convection is present for five minutes. One hour of profiles stands for a 20 km
domain and three hours for 60 km, at the paper's 5 m s-1 cell propagation speed.

Periods: ``full`` is every day on disk, for comparison with the paper's own figure 7;
``iop1`` (the profiler starts 5 March, so it covers the second half), ``iop2``,
``combined``, and ``wet2015``, the IOP1 calendar window a year later, when the
profiler ran for the whole wet season.

Writes ``<cache>/rwp/rwp_mass_flux_<period>_<window>h.npz``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from datetime import date
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.common.paths import runs_path
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.rwp_profiler import MIN_RETRIEVAL_COVERAGE, day_of, day_task

RWP_ROOT = runs_path("Obs/ARM_RWP/rwpcls")
DEFAULT_WORK_DIR = CACHE_DIRS["convection"] / "rwp"
PERIODS = {
    "full": None,
    "iop1": [GOAMAZON_IOPS[1]],
    "iop2": [GOAMAZON_IOPS[2]],
    "combined": [GOAMAZON_IOPS[1], GOAMAZON_IOPS[2]],
    "wet2015": [(date(2015, 2, 1), date(2015, 3, 31))],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rwp-root", type=Path, default=RWP_ROOT)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--periods", nargs="+", default=tuple(PERIODS), choices=tuple(PERIODS))
    p.add_argument("--window-hours", type=int, nargs="+", default=(1, 3))
    p.add_argument("--workers", type=int, default=None)
    return parse_configured_args(p, "build_iop_rwp_mass_flux", argv=argv)


def period_files(root: Path, period: str) -> list[Path]:
    files = sorted(root.glob("*/maorwpcls.*.cdf"))
    spans = PERIODS[period]
    if spans is None:
        return files
    keep = []
    for path in files:
        day = day_of(path)
        if any(first <= day <= last for first, last in spans):
            keep.append(path)
    return keep


def aggregate(rows: list[dict]) -> dict[str, np.ndarray]:
    """Ensemble profiles over the windows where convection is present."""
    height = rows[0]["height_km"]
    keep = [(r, r["present"]) for r in rows if r["present"].any()]
    sigma_up = np.concatenate([r["sigma_up"][m] for r, m in keep])
    sigma_down = np.concatenate([r["sigma_down"][m] for r, m in keep])
    flux_up = np.concatenate([r["flux_up"][m] for r, m in keep])
    flux_down = np.concatenate([r["flux_down"][m] for r, m in keep])
    sum_up = np.concatenate([r["sum_up"][m] for r, m in keep]).sum(axis=0)
    sum_down = np.concatenate([r["sum_down"][m] for r, m in keep]).sum(axis=0)
    n_up = np.concatenate([r["n_up"][m] for r, m in keep]).sum(axis=0)
    n_down = np.concatenate([r["n_down"][m] for r, m in keep]).sum(axis=0)
    n_conv = np.concatenate([r["n_conv"][m] for r, m in keep]).sum(axis=0)
    # Weighted-mean velocity: every retrieved sample counts once, as in section 4.1.
    v_up = np.divide(sum_up, n_up, out=np.full_like(sum_up, np.nan), where=n_up > 0)
    v_down = np.divide(sum_down, n_down, out=np.full_like(sum_down, np.nan), where=n_down > 0)
    coverage = np.divide(n_up + n_down, n_conv, out=np.zeros_like(n_conv), where=n_conv > 0)
    thin = coverage < MIN_RETRIEVAL_COVERAGE  # the melting layer, where nothing is retrieved
    out = {
        "height_km": height,
        "sigma_up": np.nanmean(sigma_up, axis=0),
        "sigma_down": np.nanmean(sigma_down, axis=0),
        "sigma_total": np.nanmean(sigma_up + sigma_down, axis=0),
        "v_up": v_up,
        "v_down": v_down,
        "flux_up": np.nanmean(flux_up, axis=0),
        "flux_down": np.nanmean(flux_down, axis=0),
        "flux_net": np.nanmean(flux_up + flux_down, axis=0),
        "n_up": n_up,
        "n_down": n_down,
        "n_conv": n_conv,
        "coverage": coverage,
        "n_windows": np.array([sigma_up.shape[0]]),
        "n_days": np.array([len({r["day"] for r, _ in keep})]),
    }
    for key in ("sigma_up", "sigma_down", "sigma_total", "v_up", "v_down",
                "flux_up", "flux_down", "flux_net"):
        out[key] = np.where(thin, np.nan, out[key])
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # netCDF/HDF5 does not survive fork on this machine
    for window in args.window_hours:
        for period in args.periods:
            files = period_files(args.rwp_root, period)
            if not files:
                print(f"[{period} {window}h] no files", flush=True)
                continue
            rows, failed = [], []
            with ctx.Pool(workers) as pool:
                for row in pool.imap_unordered(day_task, [(str(f), window) for f in files], 4):
                    (failed if row is None or "error" in row else rows).append(row)
            if not rows:
                print(f"[{period} {window}h] nothing readable", flush=True)
                continue
            rows.sort(key=lambda r: r["day"])
            result = aggregate(rows)
            out = args.work_dir / f"rwp_mass_flux_{period}_{window}h.npz"
            save_npz(out, **result)
            peak = int(np.nanargmax(result["flux_up"]))
            print(f"[{period} {window}h] {len(files)} days, {int(result['n_days'][0])} with "
                  f"convection, {int(result['n_windows'][0])} windows; "
                  f"peak upward flux {result['flux_up'][peak]:.3f} kg m-2 s-1 at "
                  f"{result['height_km'][peak]:.1f} km; max convective area fraction "
                  f"{np.nanmax(result['sigma_total']):.3f}; {len(failed)} files failed",
                  flush=True)
            print(f"[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
