"""Aggregate per-day DDH budgets into 15-day (1-15 Mar 2014) npz, for the SUP
budget suite.

  * baseline control/graupel  -> <exp>_base15_<var>.npz  (restrict the existing
    per-day .dta, which span 2 years, to the 15 March days)
  * perturbations             -> <exp>_<var>.npz         (only 15 days exist)

Each npz mirrors the aggregate_budgets format: altitude_km, days, block__<name>.
Mean over days per block, finite-aware. QG only where the per-day dir exists
(control / control_* are 2-ice, no graupel).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PROC = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead0024_VZ")
AGG = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/lead0024_VZ")
DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]
SPECIES = ("QV", "QL", "QI", "QR", "QS", "QG")
_DTA = re.compile(r"^(?P<var>\w+)\.DHFDLABOF\+\d+\.(?P<block>[^.]+)\.dta$")

# (source_exp, day_list, output_prefix)
JOBS = [
    ("control", DAYS, "control_base15"),
    ("graupel", DAYS, "graupel_base15"),
    ("control_NIMELIT_1", DAYS, "control_NIMELIT_1"),
    ("graupel_NIMELIT_1", DAYS, "graupel_NIMELIT_1"),
    ("control_LNEBCV_F", DAYS, "control_LNEBCV_F"),
    ("graupel_LNEBCV_F", DAYS, "graupel_LNEBCV_F"),
]


def read_dta(p):
    a = np.loadtxt(p)
    return a[:, 0].astype(float), a[:, 1].astype(float)


def aggregate(exp, days, var):
    coord_s = coord_c = None
    bsum, bcnt = {}, {}
    used = []
    for day in days:
        vdir = PROC / exp / day / var
        if not (vdir / "done.ok").exists():
            continue
        got = False
        for f in vdir.iterdir():
            m = _DTA.match(f.name)
            if not m:
                continue
            try:
                z, v = read_dta(f)
            except Exception:
                continue
            if coord_s is None:
                coord_s = np.zeros_like(z); coord_c = np.zeros_like(z)
            if z.shape != coord_s.shape:
                continue
            fc = np.isfinite(z); coord_s[fc] += z[fc]; coord_c[fc] += 1
            blk = m.group("block")
            if blk not in bsum:
                bsum[blk] = np.zeros_like(v); bcnt[blk] = np.zeros_like(v)
            fv = np.isfinite(v); bsum[blk][fv] += v[fv]; bcnt[blk][fv] += 1
            got = True
        if got:
            used.append(day)
    if coord_s is None:
        return None
    alt = np.where(coord_c > 0, coord_s / np.maximum(coord_c, 1), np.nan)
    blocks = {f"block__{k}": np.where(bcnt[k] > 0, bsum[k] / np.maximum(bcnt[k], 1), np.nan)
              for k in bsum}
    return alt, np.array(used, dtype="U16"), blocks


def main():
    AGG.mkdir(parents=True, exist_ok=True)
    for exp, days, prefix in JOBS:
        for var in SPECIES:
            res = aggregate(exp, days, var)
            if res is None:
                continue
            alt, used, blocks = res
            out = AGG / f"{prefix}_{var}.npz"
            np.savez_compressed(out, altitude_km=alt, days=used, **blocks)
            print(f"{prefix} {var}: n_days={used.size} blocks={len(blocks)} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
