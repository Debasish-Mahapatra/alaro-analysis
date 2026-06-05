"""Frequency vs intensity histograms: parameterized |omega_UD| vs resolved
|rho g w|.  Writes counts to /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity/all_levels_pooled.npz for the plotting step."""
from pathlib import Path
import numpy as np
import xarray as xr
import sys

from alaro_analysis.common.parallel import imap_unordered_progress

DATA = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
EXPERIMENTS = ("control", "graupel", "2mom")
RD, G = 287.05, 9.80665
BIN_EDGES = np.logspace(-2, 2.5, 80)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def _read(p):
    with xr.open_dataset(p, decode_times=False) as ds:
        return ds[list(ds.data_vars)[0]].values[0].astype(np.float64)


def _process_day(args):
    exp, day_name = args
    base = DATA / exp / "masked-netcdf"
    hist_param = np.zeros(BIN_EDGES.size - 1, dtype=np.int64)
    hist_reslv = np.zeros(BIN_EDGES.size - 1, dtype=np.int64)
    for f in sorted((base / "UD_OMEGA" / day_name).iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            om = _read(base / "UD_OMEGA" / day_name / f.name)
            me = _read(base / "UD_MESH_FRAC" / day_name / f.name)
            w  = _read(base / "VERT_VELOCIT" / day_name / f.name)
            p  = _read(base / "PRESSURE" / day_name / f.name)
            t  = _read(base / "TEMPERATURE" / day_name / f.name)
        except Exception:
            continue
        if np.nanmax(p) < 2000.0:
            p *= 100.0
        L = min(om.shape[0], me.shape[0], w.shape[0], p.shape[0], t.shape[0])
        om, me, w, p, t = (a[:L] for a in (om, me, w, p, t))
        rho = p / (RD * t)
        m_param = np.isfinite(om) & np.isfinite(me) & (me > 0) & (om < 0)
        if m_param.any():
            hist_param += np.histogram(np.abs(om[m_param]), bins=BIN_EDGES)[0]
        m_reslv = np.isfinite(w) & np.isfinite(rho) & (w > 0)
        if m_reslv.any():
            hist_reslv += np.histogram(rho[m_reslv] * G * w[m_reslv], bins=BIN_EDGES)[0]
    return exp, hist_param, hist_reslv


def main():
    tasks = []
    for exp in EXPERIMENTS:
        days = sorted(d.name for d in (DATA / exp / "masked-netcdf" / "UD_OMEGA").iterdir()
                      if d.is_dir() and d.name.startswith("pf"))
        for d in days:
            tasks.append((exp, d))
    print(f"tasks: {len(tasks)} (days x experiments)", flush=True)

    agg = {exp: [np.zeros(BIN_EDGES.size - 1, dtype=np.int64),
                 np.zeros(BIN_EDGES.size - 1, dtype=np.int64)]
           for exp in EXPERIMENTS}

    for exp, hp, hr in imap_unordered_progress(_process_day, tasks, desc="days"):
        agg[exp][0] += hp
        agg[exp][1] += hr

    np.savez("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity/all_levels_pooled.npz",
             bin_edges=BIN_EDGES,
             bin_centers=BIN_CENTERS,
             **{f"{e}_param": agg[e][0] for e in EXPERIMENTS},
             **{f"{e}_reslv": agg[e][1] for e in EXPERIMENTS})
    for e in EXPERIMENTS:
        print(f"{e}: param n={agg[e][0].sum():,}  resolved n={agg[e][1].sum():,}",
              flush=True)
    print("saved /gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/freq_intensity/all_levels_pooled.npz", flush=True)


if __name__ == "__main__":
    main()
