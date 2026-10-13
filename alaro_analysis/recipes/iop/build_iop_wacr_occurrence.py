#!/usr/bin/env python3
"""Hydrometeor occurrence above T3 in September 2014: ARM W-band cloud radar against the runs.

See ``data.wacr_occurrence`` for the method. The WACR files cover 31 Aug - 30 Sep 2014
(IOP2). Observations and runs use the same hours: the radar-valid hours with SIPAM scans
and WACR profiles for at least half the hour. The runs are sampled in the 5x5 cells
(20 km) around T3 at their hourly snapshots. Writes ``<cache>/wacr/<dataset>_T3.npz``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.microphysics_columns import PROFILE_HEIGHTS_KM
from alaro_analysis.data.wacr_occurrence import (
    BOX,
    MIN_HEIGHT_KM,
    Q_THRESHOLDS,
    RAIN_DBZ_2KM,
    SCAN_S,
    WACR_GLOB,
    WACR_ROOT,
    empty_model_stats,
    init_model,
    init_wacr,
    model_day,
    t3_cell,
    wacr_file,
)
from alaro_analysis.recipes.iop.build_iop_microphysics_fields import DEFAULT_WORK_DIR as FIELDS_DIR
from alaro_analysis.recipes.iop.build_iop_sipam_reflectivity import DEFAULT_WORK_DIR as SIPAM_DIR
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "wacr"
IOP = 2
MIN_PROFILES = 900  # of ~1780 per hour


def wacr_days() -> list[str]:
    return sorted({p.name.split(".")[2] for p in WACR_ROOT.glob(WACR_GLOB)})


def sipam_rain_slots(days: list[str], t3: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """12-min slots with a SIPAM scan and whether it shows 20 dBZ at 2 km in the 3x3 cells around T3."""
    j, i = t3
    keys, rain = [], []
    for day in days:
        path = SIPAM_DIR / f"iop{IOP}" / f"SIPAM_{day}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            dbz = ds["dbz"].sel(height=2000.0).values[:, j - 1:j + 2, i - 1:i + 2]
            secs = ds["time"].values.astype("datetime64[s]").astype(np.int64)
        keys.append(secs // SCAN_S)
        rain.append(np.nanmax(np.where(np.isfinite(dbz), dbz, -99.0), axis=(1, 2)) >= RAIN_DBZ_2KM)
    return np.concatenate(keys), np.concatenate(rain)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--datasets", nargs="+", default=["WACR", *MODEL_RUNS])
    p.add_argument("--workers", type=int, default=8, help="I/O bound: 28 GB of spectra files")
    p.add_argument("--recompute", action="store_true")
    return parse_configured_args(p, "build_iop_wacr_occurrence", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(FIELDS_DIR / "grid.nc") as g:
        t3 = t3_cell(g["lon"].values, g["lat"].values)
    days = wacr_days()
    obs_path = args.work_dir / "WACR_T3.npz"
    if "WACR" in args.datasets and (args.recompute or not obs_path.exists()):
        slot_keys, slot_rain = sipam_rain_slots(days, t3)
        merged: dict[tuple[int, int], list] = {}
        files = sorted(str(p) for p in WACR_ROOT.glob(WACR_GLOB))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_wacr,
                                 initargs=(slot_keys, slot_rain)) as pool:
            for part in pool.map(wacr_file, files, chunksize=4):
                for key, (s, n) in part.items():
                    acc = merged.setdefault(key, [np.zeros(PROFILE_HEIGHTS_KM.size), 0])
                    acc[0] += s
                    acc[1] += n
        hours = np.array(sorted({h for h, _ in merged}), dtype=np.int64)
        det = np.zeros((2, hours.size, PROFILE_HEIGHTS_KM.size))
        n = np.zeros((2, hours.size))
        for (h, r), (s, c) in merged.items():
            k = np.searchsorted(hours, h)
            det[r, k], n[r, k] = s, c
        save_npz(obs_path, hours=hours, det=det, n=n, profile_heights_km=PROFILE_HEIGHTS_KM,
                 t3=np.array(t3), min_height_km=np.array(MIN_HEIGHT_KM))
        print(f"[WACR] {len(files)} files, {hours.size} hours, {int(n.sum())} profiles "
              f"({int(n[1].sum())} under rain) -> {obs_path}", flush=True)
    with np.load(obs_path) as obs:
        hours_ok = obs["hours"][obs["n"].sum(axis=0) >= MIN_PROFILES]
    hour_set = frozenset(int(h) for h in hours_ok)
    for key in args.datasets:
        if key == "WACR":
            continue
        label = EXPERIMENT_LABELS[key]
        out = args.work_dir / f"{label}_T3.npz"
        if out.exists() and not args.recompute:
            print(f"[{label}] cached", flush=True)
            continue
        # a day file holds the snapshots 01..24 UTC, so hour h lives in the file of h - 1 h
        model_days = sorted({datetime.utcfromtimestamp((int(h) - 1) * 3600).strftime("%Y%m%d") for h in hour_set})
        files = [FIELDS_DIR / label / f"iop{IOP}" / f"{label}_{d}.nc" for d in model_days]
        if not all(f.exists() for f in files):
            print(f"[{label}] conversion incomplete, skip", flush=True)
            continue
        total = empty_model_stats()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_model,
                                 initargs=(t3, hour_set)) as pool:
            for part in pool.map(model_day, [str(f) for f in files]):
                for k, v in part.items():
                    total[k] = total[k] + v
        save_npz(out, **total, q_thresholds=np.array(Q_THRESHOLDS), profile_heights_km=PROFILE_HEIGHTS_KM,
                 n_hours=np.array(len(hour_set)), box_cells=np.array(2 * BOX + 1))
        print(f"[{label}] {len(files)} days, {int(total['n'].sum())} columns -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
