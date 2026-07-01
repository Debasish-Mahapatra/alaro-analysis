#!/usr/bin/env python3
"""Recompute updraft-extent / updraft-flux / cloud-cover diurnal-mean profiles
DIRECTLY from the masked-netcdf files (no reliance on the pre-built caches).

For every experiment this reads, per hourly file, the full-domain mean vertical
profile of:
  * updraft extent      = mean(UD_MESH_FRAC)
  * updraft mass flux   = mean( where(mesh>0, (-UD_OMEGA*mesh)/g, 0) )
  * cloud cover         = mean(CLOUD_FRACTI)
  * temperature         = mean(TEMPERATURE)            (for the freezing line)
then bins by Amazon local hour (UTC-4) and averages over the full 2-year run.
The reduction is identical to hydrometeor.py's compute_diurnal_profile /
compute_updraft_derived_profile_from_files (reused here per file).

Writes fresh npz to processed-data/paper5_from_netcdf/<exp>_<quantity>.npz
(plus <exp>_height.npz).  32 parallel workers.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.common.spatial import (
    apply_spatial_window_to_array,
    build_spatial_window,
)
from alaro_analysis.data.dataset_io import nanmean_with_count, read_vertical_profile
from alaro_analysis.data.discovery import collect_file_records
from alaro_analysis.workflows.hydrometeor import (
    as_time_level_yx,
    compute_geopotential_height_profile,
    read_field_array,
)

ALARO = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/paper5_from_netcdf")
EXPS = ("control", "graupel", "2mom")
UTC_OFFSET = -4
WORKERS = 32
SW = build_spatial_window(None, None)  # full domain (tag "full-domain")
QUANTITIES = ("extent", "flux", "cloud", "temp")


def _worker(task: tuple[str, str, str, str, int]):
    omega_s, mesh_s, cloud_s, temp_s, local_hour = task
    ext = flux = cloud = temp = None
    try:
        omega_p, mesh_p = Path(omega_s), Path(mesh_s)
        omega = apply_spatial_window_to_array(
            as_time_level_yx(read_field_array(omega_p, "UD_OMEGA"), omega_p), SW, omega_p
        )
        mesh = apply_spatial_window_to_array(
            as_time_level_yx(read_field_array(mesh_p, "UD_MESH_FRAC"), mesh_p), SW, mesh_p
        )
        ext, _ = nanmean_with_count(mesh, axis=(0, 2, 3))
        flux, _ = nanmean_with_count(
            np.where(mesh > 0, (-omega * mesh) / G, 0.0), axis=(0, 2, 3)
        )
    except Exception:
        ext = flux = None
    try:
        cloud, _ = read_vertical_profile(Path(cloud_s), "CLOUD_FRACTI", spatial_window=SW, compact_match=True)
    except Exception:
        cloud = None
    try:
        temp, _ = read_vertical_profile(Path(temp_s), "TEMPERATURE", spatial_window=SW, compact_match=True)
    except Exception:
        temp = None
    return (
        local_hour,
        None if ext is None else np.asarray(ext, dtype=np.float64),
        None if flux is None else np.asarray(flux, dtype=np.float64),
        None if cloud is None else np.asarray(cloud, dtype=np.float64),
        None if temp is None else np.asarray(temp, dtype=np.float64),
    )


def build_experiment(exp: str) -> None:
    base = ALARO / exp / "masked-netcdf"
    records = collect_file_records(base / "UD_OMEGA", None, None, UTC_OFFSET)
    tasks: list[tuple[str, str, str, str, int]] = []
    for local_hour, omega in records:
        day, name = omega.parent.name, omega.name
        tasks.append(
            (
                str(omega),
                str(base / "UD_MESH_FRAC" / day / name),
                str(base / "CLOUD_FRACTI" / day / name),
                str(base / "TEMPERATURE" / day / name),
                local_hour,
            )
        )
    n = len(tasks)
    print(f"[{exp}] {n} hourly files; reducing with {WORKERS} workers ...", flush=True)

    nlev: int | None = None
    acc: dict[str, list[np.ndarray]] | None = None
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for local_hour, ext, flux, cloud, temp in pool.map(_worker, tasks, chunksize=16):
            profiles = {"extent": ext, "flux": flux, "cloud": cloud, "temp": temp}
            if nlev is None:
                for p in profiles.values():
                    if p is not None:
                        nlev = int(p.size)
                        break
                if nlev is None:
                    done += 1
                    continue
                acc = {
                    k: [np.zeros((nlev, 24)), np.zeros((nlev, 24), dtype=np.int64)]
                    for k in QUANTITIES
                }
            for k, p in profiles.items():
                if p is None or p.size != nlev:
                    continue
                finite = np.isfinite(p)
                acc[k][0][finite, local_hour] += p[finite]
                acc[k][1][finite, local_hour] += 1
            done += 1
            if done % 2000 == 0 or done == n:
                rate = done / max(1e-9, time.time() - t0)
                print(f"[{exp}] {done}/{n} files ({rate:.0f}/s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for k in QUANTITIES:
        sums, counts = acc[k]
        mean = np.full_like(sums, np.nan)
        nz = counts > 0
        mean[nz] = sums[nz] / counts[nz]
        np.savez(OUT / f"{exp}_{k}.npz", mean=mean, counts=counts, n_files=n)

    height_m, _ = compute_geopotential_height_profile(
        base / "GEOPOTENTIEL", "GEOPOTENTIEL", None, None, UTC_OFFSET, "first", SW
    )
    np.savez(OUT / f"{exp}_height.npz", height_m=np.asarray(height_m, dtype=np.float64))
    print(f"[{exp}] done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


def main(argv: list[str]) -> int:
    exps = tuple(argv) if argv else EXPS
    for exp in exps:
        build_experiment(exp)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
