"""Extract annual-mean temperature profile and altitude from DDH files.

Reads VTT0 (initial temperature, K) and a ddhb -Y VZ profile from each of the
730 DHFDLABOF+0024 files per experiment, averages across days, and saves:

  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/
      temperature_{exp}.npz
          altitude_km:   (n_lev,)     altitude in km, surface first not - we
                                      keep the top-first ordering from DDH/ddhb
          temperature_k: (n_lev,)     time-mean temperature in K
          n_days:        int

Usage:
  conda activate epygram
  python -m alaro_analysis.ddh.extract_temperature
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import epygram

UNTAR_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-untar")
AGG_DIR    = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated")
ALT_SRC    = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead0024_VZ")
LOG_PATH   = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/alaro-analysis/cache/logs/extract_temperature.log")
EXPERIMENTS = ("control", "graupel", "2mom")
N_WORKERS = 32


CP_DRY = 1004.5  # J/kg/K

def _read_temperature_k(path: Path) -> np.ndarray | None:
    """Read temperature in K from one DDH LFA file.

    DDH stores VCT0 = cp * T * (dp/g) (enthalpy per unit area) and VPP0 = dp/g
    (mass per unit area).  Thus T_K = VCT0 / VPP0 / cp.
    """
    try:
        r = epygram.formats.resource(str(path), "r", fmt="LFA")
        fields = r.listfields()
        if "VCT0" not in fields or "VPP0" not in fields:
            r.close()
            return None
        vct = np.asarray(r.readfield("VCT0").getdata(), dtype=np.float64).ravel()
        vpp = np.asarray(r.readfield("VPP0").getdata(), dtype=np.float64).ravel()
        r.close()
        with np.errstate(divide="ignore", invalid="ignore"):
            t_k = vct / np.where(vpp > 0, vpp, np.nan) / CP_DRY
        return t_k
    except Exception:
        return None


def _process_day(args):
    exp, day_dir = args
    ddh_file = day_dir / "DHFDLABOF+0024"
    return _read_temperature_k(ddh_file)


def _mean_altitude(exp: str) -> np.ndarray | None:
    """Average altitude profile across all aggregated .dta VQLM files."""
    root = ALT_SRC / exp
    if not root.exists():
        return None
    altitudes: list[np.ndarray] = []
    for day_dir in sorted(root.iterdir()):
        vqlm = day_dir / "QL" / "QL.DHFDLABOF+0024.VQLM.dta"
        if not vqlm.exists():
            continue
        arr = np.loadtxt(vqlm)
        altitudes.append(arr[:, 0])   # altitude in km, no sign flip needed
    if not altitudes:
        return None
    # Alignments: all days use the same number of model levels.
    min_len = min(a.size for a in altitudes)
    stacked = np.stack([a[:min_len] for a in altitudes], axis=0)
    return np.mean(stacked, axis=0)


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    epygram.init_env()

    with open(LOG_PATH, "w") as logf:
        t0 = time.time()
        for exp in EXPERIMENTS:
            exp_untar_root = UNTAR_ROOT / exp / "output"
            day_dirs = sorted(d for d in exp_untar_root.iterdir()
                              if d.is_dir() and d.name.startswith("DDH20"))
            logf.write(f"{exp}: {len(day_dirs)} day dirs\n"); logf.flush()

            tasks = [(exp, d) for d in day_dirs]
            sums = None
            counts = None
            with Pool(N_WORKERS) as pool:
                for t in pool.imap_unordered(_process_day, tasks):
                    if t is None:
                        continue
                    if sums is None:
                        sums = t.copy().astype(np.float64)
                        counts = np.ones_like(sums)
                    else:
                        if t.shape != sums.shape:
                            continue
                        mask = np.isfinite(t)
                        sums[mask] += t[mask]
                        counts[mask] += 1
            if sums is None:
                logf.write(f"  {exp}: no VTT0 read\n"); continue
            n_days = int(counts.max())
            temp_k = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

            alt_km = _mean_altitude(exp)
            if alt_km is None:
                logf.write(f"  {exp}: WARN no altitude source (run +0024 VZ first)\n")
                alt_km = np.full_like(temp_k, np.nan)

            n = min(len(alt_km), len(temp_k))
            out = AGG_DIR / f"temperature_{exp}.npz"
            AGG_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                altitude_km=alt_km[:n],
                temperature_k=temp_k[:n],
                n_days=n_days,
            )
            logf.write(f"  {exp}: T range {np.nanmin(temp_k):.1f}-{np.nanmax(temp_k):.1f} K, "
                       f"alt range {np.nanmin(alt_km):.2f}-{np.nanmax(alt_km):.2f} km, "
                       f"n_days={n_days} -> {out}\n")
            logf.flush()
        logf.write(f"DONE in {time.time() - t0:.1f}s\n")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
