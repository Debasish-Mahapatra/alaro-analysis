#!/usr/bin/env python3
"""Diurnal-mean height-hour sections of downdraft and resolved-vertical-velocity
diagnostics, computed DIRECTLY from the raw ALARO FA output.

The masked-netcdf archive is not touched: every profile below is read out of
``untar-output/pf<YYYYMMDD>/pfABOFABOF+<HHHH>`` with epygram, cropped to the
same radar window the netcdf archive uses (y=[186:261), x=[189:261), 2822
points inside the SIPAM radar mask) and reduced to a vertical profile per file.

Quantities (all domain means over the 2822 radar-mask cells):

  parameterized downdraft (DD_MESH_FRAC / DD_OMEGA, both gridpoint fields).
  DD_OMEGA is positive for subsidence in the ALARO convention, so the mass
  flux below is positive downward:
    extent      mean(DD_MESH_FRAC)                              fraction
    flux        mean(where(mesh>0, DD_OMEGA*mesh/g, 0))         kg m-2 s-1

  resolved vertical velocity (VERT.VELOCIT = w, m/s, positive upward,
  spectral in FA so every level is transformed with sp2gp) with the upward
  motion masked out, i.e. keeping only w < 0, and reported as a positive
  downward speed:
    w_dn        mean(max(-w, 0))                                m s-1
    w_dn_frac   mean(w < 0)                                     fraction
    w_dn_flux   mean(where(w<0, -rho*w, 0)), rho = p/(Rd*T)     kg m-2 s-1
    w_dn_cond   mean of -w over the w<0 cells only              m s-1

  temp          mean(TEMPERATURE)                               K (freezing line)

Profiles are binned by Amazon local hour (UTC-4) and averaged over the full
two-year run, exactly as build_updraft_cloud_diurnal_from_netcdf.py does for
the netcdf archive.  Level index 0 is S001 (model top); the paired height array
comes from GEOPOTENTIEL/g of the first file of each experiment, and the plotting
script sorts by height.

Writes processed-data/paper_downdraft_from_FA/<exp>_<quantity>.npz plus
<exp>_height.npz.  32 parallel workers, one day-directory per task.

Requires the epygram conda environment:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
"""
from __future__ import annotations

import os

# One thread per worker: the ecTrans spectral transform would otherwise
# oversubscribe the machine 32 times over.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
ALARO = RUNS_ROOT / "ALARO"
MASK_FILE = RUNS_ROOT / "mask" / "Radar_mask_latlon.nc"
OUT = RUNS_ROOT / "processed-data" / "paper_downdraft_from_FA"

EXPS = ("control", "graupel", "2mom")
UTC_OFFSET = -4
# 32 by default (project convention).  This workflow is only partly CPU-bound:
# the ecTrans spectral transform leaves each worker at ~72% CPU while it waits on
# GPFS, so DOWNDRAFT_FA_WORKERS can raise the count when the machine is idle.
# Each worker holds ~1.9 GB (ecTrans workspace) and the user cgroup caps at 96 GB,
# so stay at or below 40 -- 64 workers gets OOM-killed.
WORKERS = int(os.environ.get("DOWNDRAFT_FA_WORKERS", "32"))
N_LEVELS = 87
G = 9.80665
RD = 287.05

# Same ROI bbox the netcdf conversion used, intersected with the radar mask.
ROI_WEST, ROI_EAST, ROI_SOUTH, ROI_NORTH = -67.0, -53.0, -10.0, 4.0
MASK_VAR = "radar_mask"
MASK_THRESHOLD = 0.5
EXPECTED_MASK_POINTS = 2822

QUANTITIES = (
    "extent",
    "flux",
    "w_dn",
    "w_dn_frac",
    "w_dn_flux",
    "w_dn_cond",
    "temp",
)

_WINDOW: tuple[int, int, int, int, np.ndarray] | None = None


def fa_path(exp: str, day: str, hour: int) -> Path:
    return ALARO / exp / "untar-output" / day / f"pfABOFABOF+{hour:04d}"


def _open(path: Path):
    import epygram

    return epygram.formats.resource(str(path), "r")


def read_gridpoint(resource, name: str) -> np.ndarray:
    field = resource.readfield(name)
    if field.spectral:
        field.sp2gp()
    return np.asarray(field.getdata(), dtype=np.float64)


def build_window() -> tuple[int, int, int, int, np.ndarray]:
    """Crop window + boolean radar mask on the raw FA grid.

    Reproduces converter/pipeline.compute_crop_and_spatial_mask: ROI bbox AND
    the regridded radar mask, cropped to the bounding box of the kept points.
    """
    import epygram
    import xarray as xr

    from alaro_analysis.converter.pipeline import _regrid_mask_to_model

    epygram.init_env()
    sample = fa_path(EXPS[0], sorted(p.name for p in (ALARO / EXPS[0] / "untar-output").iterdir())[0], 0)
    resource = _open(sample)
    try:
        geometry = resource.readfield("S001DD_MESH_FRAC").geometry
        lon, lat = (np.asarray(a, dtype=np.float64) for a in geometry.get_lonlat_grid())
    finally:
        resource.close()

    radar, _ = _regrid_mask_to_model(
        mask_file=MASK_FILE,
        target_lat=xr.DataArray(lat, dims=("y", "x")),
        target_lon=xr.DataArray(lon, dims=("y", "x")),
        mask_var=MASK_VAR,
        mask_lat_name=None,
        mask_lon_name=None,
        mask_threshold=MASK_THRESHOLD,
    )
    roi = (lon >= ROI_WEST) & (lon <= ROI_EAST) & (lat >= ROI_SOUTH) & (lat <= ROI_NORTH)
    spatial = roi & np.asarray(radar, dtype=bool)
    ys, xs = np.where(spatial)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mask = spatial[y0:y1, x0:x1]
    if int(mask.sum()) != EXPECTED_MASK_POINTS:
        raise RuntimeError(
            f"radar mask has {int(mask.sum())} points, expected {EXPECTED_MASK_POINTS}"
        )
    return y0, y1, x0, x1, mask


def _init_worker(window: tuple[int, int, int, int, np.ndarray]) -> None:
    global _WINDOW
    import epygram

    epygram.init_env()
    _WINDOW = window


def read_stack(resource, fa_name: str) -> np.ndarray:
    """(N_LEVELS, n_masked_points) array for one FA field, S001 first."""
    y0, y1, x0, x1, mask = _WINDOW
    out = np.empty((N_LEVELS, int(mask.sum())), dtype=np.float64)
    for k in range(N_LEVELS):
        out[k] = read_gridpoint(resource, f"S{k + 1:03d}{fa_name}")[y0:y1, x0:x1][mask]
    return out


def profiles_from_file(path: Path) -> dict[str, np.ndarray]:
    resource = _open(path)
    try:
        mesh = read_stack(resource, "DD_MESH_FRAC")
        omega = read_stack(resource, "DD_OMEGA")
        w = read_stack(resource, "VERT.VELOCIT")
        temp = read_stack(resource, "TEMPERATURE")
        pres = read_stack(resource, "PRESSURE")
    finally:
        resource.close()

    # Mask out the upward motion: keep w < 0, reported as a positive down-speed.
    down = w < 0.0
    w_dn = np.where(down, -w, 0.0)
    rho = pres / (RD * temp)
    n_dn = down.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = np.where(n_dn > 0, w_dn.sum(axis=1) / np.maximum(n_dn, 1), np.nan)
    return {
        "extent": np.nanmean(mesh, axis=1),
        # DD_OMEGA > 0 for subsidence, so this is a positive downward mass flux.
        "flux": np.nanmean(np.where(mesh > 0.0, (omega * mesh) / G, 0.0), axis=1),
        "w_dn": np.nanmean(w_dn, axis=1),
        "w_dn_frac": down.mean(axis=1).astype(np.float64),
        "w_dn_flux": np.nanmean(rho * w_dn, axis=1),
        "w_dn_cond": cond,
        "temp": np.nanmean(temp, axis=1),
    }


def process_day(task: tuple[str, str]) -> tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, ...]]:
    exp, day = task
    sums = {q: np.zeros((N_LEVELS, 24), dtype=np.float64) for q in QUANTITIES}
    counts = {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in QUANTITIES}
    warnings: list[str] = []
    n_files = 0
    for hour in range(24):
        path = fa_path(exp, day, hour)
        if not path.exists():
            warnings.append(f"{exp} {day} +{hour:04d}: missing")
            continue
        try:
            profiles = profiles_from_file(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{exp} {day} +{hour:04d}: {exc}")
            continue
        n_files += 1
        local_hour = (hour + UTC_OFFSET) % 24
        for q, profile in profiles.items():
            finite = np.isfinite(profile)
            sums[q][finite, local_hour] += profile[finite]
            counts[q][finite, local_hour] += 1
    return exp, n_files, sums, counts, tuple(warnings)


def height_profile(exp: str, day: str) -> np.ndarray:
    resource = _open(fa_path(exp, day, 0))
    try:
        return read_stack(resource, "GEOPOTENTIEL").mean(axis=1) / G
    finally:
        resource.close()


def main(argv: list[str]) -> int:
    exps = tuple(argv) if argv else EXPS
    window = build_window()
    print(
        f"radar window y=[{window[0]}:{window[1]}) x=[{window[2]}:{window[3]}) "
        f"{int(window[4].sum())} points",
        flush=True,
    )

    days = {exp: sorted(p.name for p in (ALARO / exp / "untar-output").iterdir() if p.is_dir()) for exp in exps}
    tasks = [(exp, day) for exp in exps for day in days[exp]]
    print(f"{len(tasks)} day-tasks over {exps} with {WORKERS} workers", flush=True)

    sums = {exp: {q: np.zeros((N_LEVELS, 24)) for q in QUANTITIES} for exp in exps}
    counts = {exp: {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in QUANTITIES} for exp in exps}
    n_files = dict.fromkeys(exps, 0)
    all_warnings: list[str] = []

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    _init_worker(window)  # parent needs _WINDOW for height_profile
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(window,)) as pool:
        for exp, nf, day_sums, day_counts, warnings in pool.map(process_day, tasks, chunksize=1):
            n_files[exp] += nf
            for q in QUANTITIES:
                sums[exp][q] += day_sums[q]
                counts[exp][q] += day_counts[q]
            all_warnings.extend(warnings)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / max(1e-9, elapsed)
                eta = (len(tasks) - done) / max(1e-9, rate)
                print(
                    f"{done}/{len(tasks)} days ({rate * 3600:.0f} days/h) "
                    f"elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min, "
                    f"warnings={len(all_warnings)}",
                    flush=True,
                )

    for exp in exps:
        for q in QUANTITIES:
            mean = np.full((N_LEVELS, 24), np.nan)
            nz = counts[exp][q] > 0
            mean[nz] = sums[exp][q][nz] / counts[exp][q][nz]
            np.savez(
                OUT / f"{exp}_{q}.npz",
                mean=mean,
                counts=counts[exp][q],
                n_files=n_files[exp],
            )
        np.savez(
            OUT / f"{exp}_height.npz",
            height_m=height_profile(exp, days[exp][0]),
            n_files=n_files[exp],
        )
        print(f"[{exp}] {n_files[exp]} files -> {OUT}", flush=True)

    if all_warnings:
        (OUT / "warnings.log").write_text("\n".join(all_warnings) + "\n")
        print(f"{len(all_warnings)} warnings -> {OUT / 'warnings.log'}", flush=True)
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
