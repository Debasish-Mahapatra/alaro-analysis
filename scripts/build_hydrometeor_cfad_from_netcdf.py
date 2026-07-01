#!/usr/bin/env python3
"""Build per-species hydrometeor mixing-ratio CFAD histograms from masked-netcdf.

For every experiment and species we accumulate, over all (subsampled) hourly
files and all grid points, a 2-D histogram counts[level, log10(q)-bin].  At plot
time each height row is normalised to 100 % (a CFAD) and the per-level median
mixing ratio is read off the cumulative distribution.

Writes processed-data/paper6_hydrometeor_cfad/<exp>_cfad.npz with one counts
array per species (shape [NLEV, NBINS]), the model-level heights (km), and the
log10 bin edges.  Subsampled in time by STRIDE for speed (a CFAD converges with
a tiny fraction of the 2-year sample).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

from alaro_analysis.common.spatial import apply_spatial_window_to_array, build_spatial_window
from alaro_analysis.data.discovery import collect_file_records
from alaro_analysis.workflows.hydrometeor import (
    as_time_level_yx,
    compute_geopotential_height_profile,
    read_field_array,
)

ALARO = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/paper6_hydrometeor_cfad")
SPECIES = ("RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER")
UTC_OFFSET = -4
WORKERS = 40
STRIDE = 1  # use every hourly file (no subsampling)
NLEV = 87
NBINS = 60
LOG_EDGES = np.linspace(-11.0, -1.0, NBINS + 1)  # log10(q) bin edges, 1e-11..1e-1
SW = build_spatial_window(None, None)  # full domain


def _hist_one(path: Path, variable: str) -> np.ndarray | None:
    """counts[NLEV, NBINS] for one file/species (log10 q, q>0, in range)."""
    try:
        arr = apply_spatial_window_to_array(
            as_time_level_yx(read_field_array(path, variable), path), SW, path
        )  # (T, L, Y, X)
    except Exception:
        return None
    nlev = min(arr.shape[1], NLEV)
    flat = np.moveaxis(arr[:, :nlev], 1, 0).reshape(nlev, -1).astype(np.float64)
    counts = np.zeros((NLEV, NBINS), dtype=np.int64)
    lo, dbin = LOG_EDGES[0], (LOG_EDGES[-1] - LOG_EDGES[0]) / NBINS
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(flat > 0.0, np.log10(flat), np.nan)
    bi = np.floor((lv - lo) / dbin)
    for lev in range(nlev):
        b = bi[lev]
        m = np.isfinite(b)
        b = b[m]
        b = b[(b >= 0) & (b < NBINS)].astype(np.int64)
        if b.size:
            counts[lev] += np.bincount(b, minlength=NBINS)
    return counts


def _worker(task: tuple[str, dict]) -> dict:
    _, paths = task
    out = {}
    for sp, p in paths.items():
        c = _hist_one(Path(p), sp)
        if c is not None:
            out[sp] = c
    return out


def build_experiment(exp: str) -> None:
    base = ALARO / exp / "masked-netcdf"
    records = collect_file_records(base / "RAIN", None, None, UTC_OFFSET)
    records = records[::STRIDE]
    species = [s for s in SPECIES if (base / s).is_dir()]
    # C1M is the 2-ice control: graupel is not a prognostic species -> skip it.
    if exp == "control" and "GRAUPEL" in species:
        species.remove("GRAUPEL")

    tasks = []
    for _, rain_path in records:
        day, name = rain_path.parent.name, rain_path.name
        paths = {sp: str(base / sp / day / name) for sp in species}
        tasks.append((exp, paths))
    n = len(tasks)
    print(f"[{exp}] {n} files (stride {STRIDE}), species={species}, {WORKERS} workers", flush=True)

    totals = {sp: np.zeros((NLEV, NBINS), dtype=np.int64) for sp in species}
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(_worker, tasks, chunksize=8):
            for sp, c in res.items():
                totals[sp] += c
            done += 1
            if done % 1000 == 0 or done == n:
                print(f"[{exp}] {done}/{n} ({done/max(1e-9, time.time()-t0):.0f}/s)", flush=True)

    height_m, _ = compute_geopotential_height_profile(
        base / "GEOPOTENTIEL", "GEOPOTENTIEL", None, None, UTC_OFFSET, "first", SW
    )
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {f"counts_{sp}": totals[sp] for sp in species}
    payload["height_km"] = np.asarray(height_m, dtype=np.float64) / 1000.0
    payload["log_edges"] = LOG_EDGES
    payload["n_files"] = n
    np.savez(OUT / f"{exp}_cfad.npz", **payload)
    print(f"[{exp}] done in {time.time()-t0:.0f}s -> {OUT/f'{exp}_cfad.npz'}", flush=True)


def main(argv: list[str]) -> int:
    for exp in (argv or ("control", "graupel", "2mom")):
        build_experiment(exp)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
