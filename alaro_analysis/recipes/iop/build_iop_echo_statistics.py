#!/usr/bin/env python3
"""Radar-echo statistics of SIPAM and of the four runs' simulated reflectivity, GoAmazon IOPs.

Same grid (4 km), same altitudes (2..16 km), same hours and same cells for all five
datasets (``analysis.radar_echo``):

* hours: the radar-valid hours of the rainfall verification; SIPAM uses every 12-min
  scan of those hours, the models their instantaneous snapshot at the hour;
* columns: only precipitating ones (reaching 20 dBZ somewhere), which removes the
  dry-season clear-air echo of the boundary layer;
* cells: statistics only in the 60-110 km range ring of the radar domain. Inside 40-60 km
  the radar's cone of silence removes echoes above ~8 km (checked on 20 IOP1 days:
  echo occurrence at 10 km 0.9 % at 20-40 km against ~5 % beyond 60 km); beyond 110 km
  the volumes are unreliable (Giangrande et al. 2016). The Steiner background may use
  the whole radar domain.

The model reflectivity is the model's own diagnostic (REFLEC_DBZ, routine GPPRS0D),
interpolated in linear Z to the radar altitudes. Writes
``<cache>/echo_stats/<dataset>_iop<n>.npz``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.analysis.radar_echo import HEIGHTS_KM, Z_IWC_T, empty_stats
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.echo_statistics import add, init_worker, model_day, sipam_day
from alaro_analysis.data.sipam_reflectivity import RADAR_LAT, RADAR_LON
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import campaign_days
from alaro_analysis.recipes.iop.build_iop_microphysics_fields import DEFAULT_WORK_DIR as FIELDS_DIR
from alaro_analysis.recipes.iop.build_iop_sipam_reflectivity import DEFAULT_WORK_DIR as SIPAM_DIR
from alaro_analysis.recipes.iop.build_iop_sipam_reflectivity import valid_hours
from alaro_analysis.recipes.iop.build_iop_updraft_partition import MODEL_RUNS

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "echo_stats"
RANGE_KM = (60.0, 110.0)


def domains() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(FIELDS_DIR / "grid.nc") as g:
        lon, lat, inside = g["lon"].values, g["lat"].values, g["inside"].values.astype(bool)
    r = np.hypot((lon - RADAR_LON) * 111.0 * np.cos(np.radians(RADAR_LAT)), (lat - RADAR_LAT) * 111.0)
    return inside, inside & (r >= RANGE_KM[0]) & (r <= RANGE_KM[1]), r


def temperature_profile_c(iop: int) -> np.ndarray:
    """IOP-mean radar-domain temperature (degC) at the radar heights, mean of C1M, G1M and G2M."""
    profiles = []
    for label in ("C1M", "G1M", "G2M"):
        t = CACHE_DIRS["convection"] / f"iop{iop}" / f"{label}_temp.npz"
        h = CACHE_DIRS["convection"] / f"iop{iop}" / f"{label}_height.npz"
        if t.exists() and h.exists():
            temp = np.nanmean(np.load(t)["mean"], axis=1) - 273.15
            z = np.load(h)["height_m"] / 1000.0
            profiles.append(np.interp(HEIGHTS_KM, z, temp))
    return np.mean(profiles, axis=0)


def run(tasks, worker, doms, workers: int, t_profile_c=None) -> dict:
    total = empty_stats()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(doms, t_profile_c, Z_IWC_T)) as pool:
        for part in pool.map(worker, tasks):
            add(total, part)
    return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--datasets", nargs="+", default=["SIPAM", *MODEL_RUNS])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--recompute", action="store_true")
    return parse_configured_args(p, "build_iop_echo_statistics", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inside, ring, _ = domains()
    doms = (inside, ring)
    hours = frozenset(valid_hours())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for iop in args.iops:
        for key in args.datasets:
            label = "SIPAM" if key == "SIPAM" else EXPERIMENT_LABELS[key]
            out = args.work_dir / f"{label}_iop{iop}.npz"
            if out.exists() and not args.recompute:
                print(f"[IOP{iop} {label}] cached", flush=True)
                continue
            days = [f"{d:%Y%m%d}" for d in campaign_days(iop)]
            if key == "SIPAM":
                tasks = [str(p) for d in days if (p := SIPAM_DIR / f"iop{iop}" / f"SIPAM_{d}.nc").exists()]
                stats = run(tasks, sipam_day, doms, args.workers, temperature_profile_c(iop))
            else:
                files = [FIELDS_DIR / label / f"iop{iop}" / f"{label}_{d}.nc" for d in days]
                missing = [f for f in files if not f.exists()]
                if missing:
                    print(f"[IOP{iop} {label}] {len(missing)} converted days missing, skip", flush=True)
                    continue
                stats = run([(str(f), hours) for f in files], model_day, doms, args.workers,
                            temperature_profile_c(iop))
            save_npz(out, **stats, range_km=np.array(RANGE_KM), n_stat_cells=np.array(int(ring.sum())),
                     t_profile_c=temperature_profile_c(iop))
            print(f"[IOP{iop} {label}] {int(stats['n_maps'][0])} maps -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
