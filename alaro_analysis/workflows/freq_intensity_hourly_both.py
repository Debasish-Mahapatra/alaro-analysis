"""Build (hour, |omega|) histograms for BOTH parameterized-only and TOTAL-updraft
definitions, per experiment.  Writes /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_and_total_nh.npz.

TOTAL reconstruction reuses hydrometeor._build_total_updraft_omega_and_sigma
so the definition matches TOTAL_UPDRAFT_INTENSITY exactly."""
from pathlib import Path
from multiprocessing import Pool
import re
import numpy as np
import xarray as xr

from alaro_analysis.workflows.hydrometeor import _build_total_updraft_omega_and_sigma

DATA = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
EXPERIMENTS = ("control", "graupel", "2mom")
UTC_OFFSET = -4
HOUR_RE = re.compile(r"\+(\d{4})\.nc$")
RD = 287.05

BIN_EDGES = np.logspace(-1, 2, 61)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
N_INT = BIN_EDGES.size - 1


def _read(p):
    with xr.open_dataset(p, decode_times=False) as ds:
        return ds[list(ds.data_vars)[0]].values[0].astype(np.float64)


def _process_day(args):
    exp, day_name = args
    base = DATA / exp / "masked-netcdf"
    H_ud   = np.zeros((24, N_INT), dtype=np.int64)
    H_tot  = np.zeros((24, N_INT), dtype=np.int64)
    ncells = np.zeros(24, dtype=np.int64)

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
            om = _read(base / "UD_OMEGA"      / day_name / f.name)
            me = _read(base / "UD_MESH_FRAC"  / day_name / f.name)
            w  = _read(base / "VERT_VELOCIT"  / day_name / f.name)
            p  = _read(base / "PRESSURE"      / day_name / f.name)
            t  = _read(base / "TEMPERATURE"   / day_name / f.name)
        except Exception:
            continue

        L = min(om.shape[0], me.shape[0], w.shape[0], p.shape[0], t.shape[0])
        om, me, w, p, t = (a[:L] for a in (om, me, w, p, t))
        if np.nanmax(p) < 2000.0:
            p = p * 100.0
        rho = p / (RD * t)

        # UD-only: |omega_UD| where mesh > 0 AND upward
        mask_ud = np.isfinite(om) & np.isfinite(me) & (me > 0) & (om < 0)
        if mask_ud.any():
            H_ud[hour] += np.histogram(np.abs(om[mask_ud]), bins=BIN_EDGES)[0]

        # TOTAL: |omega_tot| where sigma_tot > 0 (reuses hydrometeor logic)
        om_tot, sig_tot = _build_total_updraft_omega_and_sigma(om, me, w, rho)
        mask_tot = np.isfinite(om_tot) & np.isfinite(sig_tot) & (sig_tot > 0)
        if mask_tot.any():
            H_tot[hour] += np.histogram(np.abs(om_tot[mask_tot]), bins=BIN_EDGES)[0]

        ncells[hour] += int(np.sum(np.isfinite(om)))

    return exp, H_ud, H_tot, ncells


def main():
    tasks = []
    for exp in EXPERIMENTS:
        for d in sorted(x.name for x in (DATA / exp / "masked-netcdf" / "UD_OMEGA").iterdir()
                        if x.is_dir() and x.name.startswith("pf")):
            tasks.append((exp, d))
    print(f"tasks: {len(tasks)}", flush=True)

    agg = {e: [np.zeros((24, N_INT), dtype=np.int64),
               np.zeros((24, N_INT), dtype=np.int64),
               np.zeros(24, dtype=np.int64)] for e in EXPERIMENTS}

    with Pool(32) as pool:
        for i, (exp, Hu, Ht, n) in enumerate(pool.imap_unordered(_process_day, tasks), 1):
            agg[exp][0] += Hu
            agg[exp][1] += Ht
            agg[exp][2] += n
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    np.savez("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_and_total_nh.npz",
             bin_edges=BIN_EDGES, bin_centers=BIN_CENTERS,
             **{f"{e}_ud":     agg[e][0] for e in EXPERIMENTS},
             **{f"{e}_tot":    agg[e][1] for e in EXPERIMENTS},
             **{f"{e}_ncells": agg[e][2] for e in EXPERIMENTS})
    for e in EXPERIMENTS:
        Hu, Ht, n = agg[e]
        print(f"{e}: UD counts={Hu.sum():,}  TOT counts={Ht.sum():,}  "
              f"finite cells={n.sum():,}", flush=True)
    print("saved /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity_hourly/ud_and_total_nh.npz", flush=True)


if __name__ == "__main__":
    main()
