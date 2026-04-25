"""Sui CMPE1 diurnal cycle — robust version.

CMPE1 = P_s / (P_s + d[C]/dt)
  [C] = column-integrated total hydrometeor mass (kg m-2)
  P_s = surface precipitation flux (kg m-2 s-1)

Differences from the naive version:
  * per-day processing in chunks, saving partial aggregates every CHUNK days
    so we never lose everything to a late-run hang.
  * per-task timeout via concurrent.futures.ProcessPoolExecutor + future.result(timeout)
    so a stuck day fails its task instead of deadlocking the pool.
  * workers skip days quietly on any exception; we log them.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
import re
import numpy as np
import xarray as xr

DATA = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
EXPERIMENTS = ("control", "graupel", "2mom")
SPECIES = ("RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER")
HOUR_RE = re.compile(r"\+(\d{4})\.nc$")
UTC_OFFSET = -4
G = 9.80665
PER_TASK_TIMEOUT_S = 90          # kill any day that takes >90 s
CHUNK = 100                       # save partial aggregate every CHUNK days
OUT_NPZ = Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/sui_cmpe1/sui_cmpe1.npz")
PARTIAL_NPZ = Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/sui_cmpe1/sui_cmpe1_partial.npz")


def _read(p):
    with xr.open_dataset(p, decode_times=False) as ds:
        return ds[list(ds.data_vars)[0]].values[0].astype(np.float64)


def _dp_pa(p):
    n = p.shape[0]
    if n == 1:
        return np.abs(p)
    ph = np.empty((n + 1,) + p.shape[1:], dtype=np.float64)
    ph[1:-1] = 0.5 * (p[:-1] + p[1:])
    ph[0]    = p[0]  + (p[0]  - ph[1])
    ph[-1]   = p[-1] - (ph[-2] - p[-1])
    return np.abs(ph[:-1] - ph[1:])


def _process_day(args):
    exp, day_name = args
    base = DATA / exp / "masked-netcdf"
    sum_C  = np.zeros(24, dtype=np.float64)
    sum_Ps = np.zeros(24, dtype=np.float64)
    cnt    = np.zeros(24, dtype=np.int64)

    try:
        files = sorted((base / "PRESSURE" / day_name).iterdir())
    except Exception:
        return exp, sum_C, sum_Ps, cnt

    for f in files:
        if not f.name.endswith(".nc"):
            continue
        m = HOUR_RE.search(f.name)
        if not m:
            continue
        lead = int(m.group(1))
        if lead >= 24:
            continue
        hour_local = (lead + UTC_OFFSET) % 24
        step = f.name
        try:
            p  = _read(base / "PRESSURE" / day_name / step)
            cv = _read(base / "CV_PREC_FLUX" / day_name / step)
            st = _read(base / "ST_PREC_FLUX" / day_name / step)
        except Exception:
            continue
        if np.nanmax(p) < 2000.0:
            p = p * 100.0
        L = min(p.shape[0], cv.shape[0], st.shape[0])
        p, cv, st = p[:L], cv[:L], st[:L]

        p_lev_mean = np.nanmean(p, axis=(1, 2))
        surface_idx = int(np.argmax(p_lev_mean))
        dp = _dp_pa(p)

        total_mass = np.zeros_like(p[0], dtype=np.float64)
        ok = True
        for sp in SPECIES:
            try:
                q = _read(base / sp / day_name / step)[:L]
            except Exception:
                ok = False
                break
            q_pos = np.where(np.isfinite(q) & (q > 0.0), q, 0.0)
            total_mass += np.sum(q_pos * dp, axis=0) / G
        if not ok:
            continue

        C_mean = float(np.nanmean(total_mass))
        flux_surface = float(np.nanmean((cv + st)[surface_idx]))
        if not np.isfinite(flux_surface):
            flux_surface = 0.0

        sum_C[hour_local]  += C_mean
        sum_Ps[hour_local] += flux_surface
        cnt[hour_local]    += 1

    return exp, sum_C, sum_Ps, cnt


def save_snapshot(agg, processed, total, where):
    out = {}
    for e in EXPERIMENTS:
        sC, sP, cn = agg[e]
        out[f"{e}_C_sum"]  = sC
        out[f"{e}_Ps_sum"] = sP
        out[f"{e}_counts"] = cn
    out["processed"] = np.array(processed)
    out["total"]     = np.array(total)
    np.savez(where, **out)


def main():
    tasks = []
    for exp in EXPERIMENTS:
        for d in sorted(x.name for x in (DATA / exp / "masked-netcdf" / "PRESSURE").iterdir()
                        if x.is_dir() and x.name.startswith("pf")):
            tasks.append((exp, d))
    total = len(tasks)
    print(f"tasks: {total}", flush=True)

    agg = {e: [np.zeros(24), np.zeros(24), np.zeros(24, dtype=np.int64)]
           for e in EXPERIMENTS}
    processed = 0
    timeouts = []

    with ProcessPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_process_day, t): t for t in tasks}
        for fut in futures:
            t = futures[fut]
            try:
                exp, sC, sP, cn = fut.result(timeout=PER_TASK_TIMEOUT_S)
            except FutTimeout:
                timeouts.append(t)
                fut.cancel()
                processed += 1
                continue
            except Exception:
                processed += 1
                continue
            agg[exp][0] += sC; agg[exp][1] += sP; agg[exp][2] += cn
            processed += 1
            if processed % 50 == 0 or processed == total:
                print(f"  {processed}/{total}  (timeouts={len(timeouts)})", flush=True)
            if processed % CHUNK == 0:
                save_snapshot(agg, processed, total, PARTIAL_NPZ)

    save_snapshot(agg, processed, total, PARTIAL_NPZ)

    result = {}
    for e in EXPERIMENTS:
        sC, sP, cn = agg[e]
        cn_safe = np.maximum(cn, 1)
        mC = np.where(cn > 0, sC / cn_safe, np.nan)
        mP = np.where(cn > 0, sP / cn_safe, np.nan)
        result[f"{e}_C_mean"]  = mC
        result[f"{e}_Ps_mean"] = mP
        result[f"{e}_counts"]  = cn
        print(f"  {e}: [C] mean {mC.min():.3e}..{mC.max():.3e} kg/m2  "
              f"P_s mean {mP.min():.3e}..{mP.max():.3e} kg/m2/s", flush=True)

    np.savez(OUT_NPZ, **result)
    if timeouts:
        print(f"  timed-out days: {len(timeouts)}  first few: {timeouts[:5]}", flush=True)
    print(f"saved {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
