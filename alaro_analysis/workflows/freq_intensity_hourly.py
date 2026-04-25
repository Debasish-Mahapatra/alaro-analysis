"""Build (hour, |omega|) 2D histograms per experiment, parameterized updrafts only.
Writes /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_only.npz for the plotting step."""
from pathlib import Path
from multiprocessing import Pool
import re
import numpy as np
import xarray as xr

DATA = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
EXPERIMENTS = ("control", "graupel", "2mom")
UTC_OFFSET = -4
HOUR_RE = re.compile(r"\+(\d{4})\.nc$")

# log-spaced intensity bins: 0.1 .. 100 Pa/s, 60 bins
BIN_EDGES = np.logspace(-1, 2, 61)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
N_INT = BIN_EDGES.size - 1


def _read(p):
    with xr.open_dataset(p, decode_times=False) as ds:
        return ds[list(ds.data_vars)[0]].values[0].astype(np.float64)


def _process_day(args):
    exp, day_name = args
    base = DATA / exp / "masked-netcdf"
    H = np.zeros((24, N_INT), dtype=np.int64)
    ncells = np.zeros(24, dtype=np.int64)   # total finite cells per hour, for normalising

    for f in sorted((base / "UD_OMEGA" / day_name).iterdir()):
        if not f.name.endswith(".nc"):
            continue
        m = HOUR_RE.search(f.name)
        if not m:
            continue
        lead = int(m.group(1))
        if lead >= 24:
            continue
        hour = (lead + UTC_OFFSET) % 24

        try:
            om = _read(base / "UD_OMEGA" / day_name / f.name)
            me = _read(base / "UD_MESH_FRAC" / day_name / f.name)
        except Exception:
            continue

        L = min(om.shape[0], me.shape[0])
        om, me = om[:L], me[:L]

        # Parameterized updrafts: mesh > 0 AND upward (omega < 0).
        mask = np.isfinite(om) & np.isfinite(me) & (me > 0) & (om < 0)
        if mask.any():
            H[hour] += np.histogram(np.abs(om[mask]), bins=BIN_EDGES)[0]
        # Denominator: finite cells at this hour (to normalise frequency)
        ncells[hour] += int(np.sum(np.isfinite(om)))

    return exp, H, ncells


def main():
    tasks = []
    for exp in EXPERIMENTS:
        for d in sorted(x.name for x in (DATA / exp / "masked-netcdf" / "UD_OMEGA").iterdir()
                        if x.is_dir() and x.name.startswith("pf")):
            tasks.append((exp, d))
    print(f"tasks: {len(tasks)} (days x experiments)", flush=True)

    agg = {e: [np.zeros((24, N_INT), dtype=np.int64),
               np.zeros(24, dtype=np.int64)] for e in EXPERIMENTS}

    with Pool(32) as pool:
        for i, (exp, H, n) in enumerate(pool.imap_unordered(_process_day, tasks), 1):
            agg[exp][0] += H
            agg[exp][1] += n
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} days", flush=True)

    np.savez("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_only.npz",
             bin_edges=BIN_EDGES,
             bin_centers=BIN_CENTERS,
             **{f"{e}_hist": agg[e][0] for e in EXPERIMENTS},
             **{f"{e}_ncells": agg[e][1] for e in EXPERIMENTS})
    for e in EXPERIMENTS:
        H, n = agg[e]
        print(f"{e}: total counts={H.sum():,}  total finite cells={n.sum():,}", flush=True)
    print("saved /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_only.npz", flush=True)


if __name__ == "__main__":
    main()
