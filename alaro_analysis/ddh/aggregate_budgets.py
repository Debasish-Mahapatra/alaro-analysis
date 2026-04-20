"""Aggregate per-day DDH .dta outputs into per-experiment × per-variable npz.

Input:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/{exp}/DDH20YYMMDD/{VAR}/
      {VAR}.DHFDLABOF+0024.{block}.dta    # 87 (pressure, value) pairs

Output:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/
      {exp}_{var}.npz
          pressure_hpa: (n_lev,)              absolute pressure, hPa
          blocks:       dict[str, array]      block -> (n_lev,) time-mean value
          n_days:       int
          days:         list[str]             contributing days

Values are in units defined by the fbl files (g/kg/day for the moisture budgets).

Usage:
  source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
  conda activate epygram
  python -m alaro_analysis.ddh.aggregate_budgets
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROCESSED_BASE = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed"
)
AGG_DIR = PROCESSED_BASE / "_aggregated"

EXPERIMENTS = ("control", "graupel", "2mom")
VARIABLES = ("QL", "QI", "QR", "QS", "QG", "QV", "UU", "VV")
LOG_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis/cache/logs")
N_WORKERS = 32


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def read_dta(path: Path, ycoor: str = "VZ") -> tuple[np.ndarray, np.ndarray]:
    """Return (coord, value) arrays from a 2-column .dta file.

    * ycoor == "VZ": first column is altitude in km (positive, top-first).
    * ycoor == "VP": first column is pressure in hPa with a display minus sign;
      we flip the sign.
    """
    arr = np.loadtxt(path)
    if ycoor == "VZ":
        coord = arr[:, 0]
    else:
        coord = -arr[:, 0]
    return coord.astype(np.float64), arr[:, 1].astype(np.float64)


_DTA_RE = re.compile(r"^(?P<var>\w+)\.DHFDLABOF\+\d+\.(?P<block>[^.]+)\.dta$")


def parse_day_var(var_dir: Path, ycoor: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Parse a single var_dir for one day; return block -> (coord, val) map."""
    if not (var_dir / "done.ok").exists():
        return None
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in var_dir.iterdir():
        m = _DTA_RE.match(f.name)
        if not m:
            continue
        block = m.group("block")
        try:
            out[block] = read_dta(f, ycoor=ycoor)
        except Exception:
            continue
    return out or None


# --------------------------------------------------------------------------
# Worker: aggregate for one (exp, var)
# --------------------------------------------------------------------------

def _aggregate_one(args):
    exp, var, lead, ycoor = args
    processed_root = PROCESSED_BASE / f"lead{lead}_{ycoor}"
    exp_dir = processed_root / exp
    if not exp_dir.exists():
        return (exp, var, 0, f"source {exp_dir} missing")
    day_dirs = sorted(d for d in exp_dir.iterdir()
                      if d.is_dir() and d.name.startswith("DDH20"))
    if not day_dirs:
        return (exp, var, 0, "no days")

    block_sums: dict[str, np.ndarray] = {}
    block_counts: dict[str, np.ndarray] = {}
    coord_sum: np.ndarray | None = None
    coord_count: np.ndarray | None = None
    days_used: list[str] = []

    for day_dir in day_dirs:
        res = parse_day_var(day_dir / var, ycoor)
        if res is None:
            continue
        for block, (coord, val) in res.items():
            if coord_sum is None:
                coord_sum = np.zeros_like(coord)
                coord_count = np.zeros_like(coord)
            if coord.shape != coord_sum.shape:
                continue
            fc = np.isfinite(coord)
            coord_sum[fc]  += coord[fc]
            coord_count[fc] += 1
            if block not in block_sums:
                block_sums[block] = np.zeros_like(val)
                block_counts[block] = np.zeros_like(val, dtype=np.int64)
            finite = np.isfinite(val)
            block_sums[block][finite] += val[finite]
            block_counts[block][finite] += 1
        days_used.append(day_dir.name)

    if coord_sum is None:
        return (exp, var, 0, "no readable day")

    mean_coord = np.where(coord_count > 0, coord_sum / np.maximum(coord_count, 1), np.nan)
    means: dict[str, np.ndarray] = {}
    for block, s in block_sums.items():
        c = block_counts[block]
        m = np.full_like(s, np.nan)
        nz = c > 0
        m[nz] = s[nz] / c[nz]
        means[block] = m

    out_dir = AGG_DIR / f"lead{lead}_{ycoor}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp}_{var}.npz"
    coord_key = "altitude_km" if ycoor == "VZ" else "pressure_hpa"
    np.savez_compressed(
        out_path,
        **{coord_key: mean_coord},
        days=np.array(days_used, dtype="U16"),
        **{f"block__{k}": v for k, v in means.items()},
    )
    return (exp, var, len(days_used), str(out_path))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                        choices=list(EXPERIMENTS))
    parser.add_argument("--variables", nargs="+", default=list(VARIABLES),
                        choices=list(VARIABLES))
    parser.add_argument("--lead", default="0024",
                        help="forecast lead (e.g. 0012 or 0024).  Matches the "
                             "directory name lead<LEAD>_<YCOOR> in DDH-processed.")
    parser.add_argument("--ycoor", default="VZ", choices=("VZ", "VP"))
    args = parser.parse_args()

    tasks = [(exp, var, args.lead, args.ycoor)
             for exp in args.experiments
             for var in args.variables
             if not (exp == "control" and var == "QG")]

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"aggregate_budgets_{args.lead}_{args.ycoor}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"aggregate_budgets (lead={args.lead} ycoor={args.ycoor}): "
                   f"{len(tasks)} (exp,var) pairs, {N_WORKERS} workers\n")
        logf.flush()
        with Pool(N_WORKERS) as pool:
            for exp, var, n_days, note in pool.imap_unordered(_aggregate_one, tasks):
                line = f"  {exp:<8} {var:<3}  {n_days:>4} days  {note}\n"
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        logf.write(f"DONE in {time.time() - t0:.1f}s\n")
    print(f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
