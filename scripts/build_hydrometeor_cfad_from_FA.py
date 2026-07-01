#!/usr/bin/env python3
"""Build hydrometeor mixing-ratio CFAD histograms from the RAW FA files
(full 480x480 model domain), not the cropped masked-netcdf.

Per file/species/level we histogram log10(mixing ratio) (q>0) over every grid
point into log-q bins, accumulating counts[level, bin].  Run in strided chunks
so each foreground piece stays short; the union of chunks k=0..N-1 is the full
file set (no time subsampling).

Usage:
    build_hydrometeor_cfad_from_FA.py <exp> <chunk_k> <n_chunks>
writes processed-data/paper6_hydrometeor_cfad_FA/<exp>_part{k}of{N}.npz
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import glob
import re
import sys
import time

import numpy as np

import epygram

try:
    epygram.init_env()
except Exception:
    pass

ALARO = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/paper6_hydrometeor_cfad_FA")
ALL_SPECIES = ("RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER")
NLEV = 87
NBINS = 60
LOG_EDGES = np.linspace(-11.0, -1.0, NBINS + 1)
LO = LOG_EDGES[0]
DBIN = (LOG_EDGES[-1] - LOG_EDGES[0]) / NBINS
WORKERS = 64
FH_RE = re.compile(r"\+(\d{4})$")

_SPECIES_FOR_WORKER: tuple[str, ...] = ALL_SPECIES


def _list_files(exp: str) -> list[str]:
    files = []
    for day in sorted(glob.glob(str(ALARO / exp / "untar-output" / "pf*"))):
        for f in sorted(glob.glob(day + "/*")):
            m = FH_RE.search(Path(f).name)
            if m and 0 <= int(m.group(1)) <= 23:  # drop +0024 duplicate valid time
                files.append(f)
    return files


def _process(path: str) -> dict:
    counts = {sp: np.zeros((NLEV, NBINS), dtype=np.int64) for sp in _SPECIES_FOR_WORKER}
    try:
        r = epygram.formats.resource(path, "r")
    except Exception:
        return counts
    try:
        for sp in _SPECIES_FOR_WORKER:
            cs = counts[sp]
            for i, lev in enumerate(range(1, NLEV + 1)):
                try:
                    fld = r.readfield(f"S{lev:03d}{sp}")
                except Exception:
                    continue
                if getattr(fld, "spectral", False):
                    fld.sp2gp()
                v = np.asarray(fld.getdata(), dtype=np.float64).ravel()
                v = v[np.isfinite(v) & (v > 0.0)]
                if v.size:
                    bi = np.floor((np.log10(v) - LO) / DBIN).astype(np.int64)
                    bi = bi[(bi >= 0) & (bi < NBINS)]
                    if bi.size:
                        cs[i] += np.bincount(bi, minlength=NBINS)
    finally:
        r.close()
    return counts


def main(argv: list[str]) -> int:
    global _SPECIES_FOR_WORKER
    exp, k, n = argv[0], int(argv[1]), int(argv[2])
    species = list(ALL_SPECIES)
    if exp == "control" and "GRAUPEL" in species:
        species.remove("GRAUPEL")  # 2-ice control: no graupel physics
    _SPECIES_FOR_WORKER = tuple(species)

    files = _list_files(exp)[k::n]
    nf = len(files)
    print(f"[{exp} chunk {k}/{n}] {nf} files, species={species}, {WORKERS} workers", flush=True)

    totals = {sp: np.zeros((NLEV, NBINS), dtype=np.int64) for sp in species}
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(_process, files, chunksize=4):
            for sp in species:
                totals[sp] += res[sp]
            done += 1
            if done % 500 == 0 or done == nf:
                print(f"[{exp} {k}/{n}] {done}/{nf} ({done/max(1e-9,time.time()-t0):.1f}/s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {f"counts_{sp}": totals[sp] for sp in species}
    payload["n_files"] = nf
    payload["log_edges"] = LOG_EDGES
    out = OUT / f"{exp}_part{k}of{n}.npz"
    np.savez(out, **payload)
    print(f"[{exp} {k}/{n}] done in {time.time()-t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
