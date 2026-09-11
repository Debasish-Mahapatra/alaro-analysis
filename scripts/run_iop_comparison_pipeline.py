#!/mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python
"""Incremental IOP comparison driver.

This driver owns cache scheduling and completion checks. Rendering is
delegated to ``run_iop_rainfall_updraft.plot_pair``. It never launches until
the caller does so explicitly, allowing the legacy cache writer to be stopped
at a safe point first.
"""
import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import run_iop_rainfall_updraft as rain

ROOT = rain.ROOT
OUT = rain.OUT
LOG = rain.LOG
PERIODS = rain.PERIODS
EXPS = rain.EXPS
WORKERS = 32
NUMERIC = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")


def expected_dates(period):
    a, b = PERIODS[period]
    return [a + timedelta(days=n) for n in range((b - a).days + 1)]


def cache_path(root, d):
    source_key = f"{root.parent.parent.name}_{root.parent.name}"
    return rain.CACHE / source_key / f"{d:%Y%m%d}.npz"


def cache_valid(path):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            rain_shape = z["rain"].shape
            hourly_shape = z["hourly"].shape
            return len(rain_shape) == 3 and hourly_shape == (24,) + rain_shape
    except Exception:
        return False


def task(root, d):
    p = cache_path(root, d)
    if cache_valid(p):
        return "cached", str(p)
    if p.exists():
        quarantine = rain.LOG.parent / "recovered-corrupt-cache" / p.parent.name
        quarantine.mkdir(parents=True, exist_ok=True)
        stale = quarantine / (p.name + f".bad.{os.getpid()}")
        shutil.move(str(p), str(stale))
    rain.reduce_day(root, d)
    return "reduced", str(p)


def schedule(period):
    dates = expected_dates(period)
    roots = []
    iop_key = f"iop{period}"
    for exp in EXPS:
        roots.append((rain.IOP / iop_key / exp / "untar-output", "warm"))
        roots.append((rain.CONT / exp / "untar-output", "continuous"))
    jobs = []
    for root, _kind in roots:
        available = rain.available(root, dates[0], dates[-1])
        for d in available:
            if not cache_valid(cache_path(root, d)):
                jobs.append((root, d))
    if not jobs:
        return {"period": period, "scheduled": 0, "completed": 0, "errors": []}
    errors = []
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(task, root, d) for root, d in jobs]
        for (root, d), future in zip(jobs, futures):
            try:
                future.result()
                completed += 1
            except Exception as exc:
                errors.append({"root": str(root), "date": d.isoformat(), "error": repr(exc)})
    return {"period": period, "scheduled": len(jobs), "completed": completed, "errors": errors}


def pair_ready(period, exp):
    dates = expected_dates(period)
    key = f"iop{period}"
    warm = rain.IOP / key / exp / "untar-output"
    cont = rain.CONT / exp / "untar-output"
    if not all((warm / f"pf{d:%Y%m%d}" / ".untar_done").is_file() for d in dates):
        return False, dates, [], []
    if not all(rain.required(warm, d) and rain.required(cont, d) for d in dates):
        return False, dates, [], []
    other = [np.load(cache_path(warm, d), allow_pickle=False) for d in dates]
    continuous = [np.load(cache_path(cont, d), allow_pickle=False) for d in dates]
    if not all(cache_valid(cache_path(warm, d)) and cache_valid(cache_path(cont, d)) for d in dates):
        for z in other + continuous:
            z.close()
        return False, dates, [], []
    return True, dates, other, continuous


def output_valid(period, exp):
    base = OUT / f"iop{period}_{exp}_smoke"
    pngs = [base.with_name(base.name + suffix) for suffix in ("_rainfall_maps.png", "_rainfall_diurnal.png", "_updraft.png")]
    txts = [base.with_name(base.name + suffix) for suffix in ("_rainfall_maps.txt", "_rainfall_diurnal.txt", "_updraft.txt")]
    if not all(p.is_file() and p.stat().st_size > 0 for p in pngs + txts):
        return False
    try:
        from PIL import Image
        if not all(Image.open(p).info.get("dpi", (0, 0))[0] >= 449 for p in pngs):
            return False
    except Exception:
        return False
    return all(NUMERIC.search(p.read_text(errors="replace")) for p in txts)


def atomic_json(path, data):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_once(status):
    status["schedule"] = [schedule(i) for i in (1, 2)]
    status["pairs"] = []
    for i in (1, 2):
        # A period is renderable only when every experiment has every planned
        # day. Caches are warmed independently, but no partial experiment or
        # shortened common intersection may produce a figure.
        period_complete = all(
            all((rain.IOP / f"iop{i}" / exp / "untar-output" / f"pf{d:%Y%m%d}" / ".untar_done").is_file()
                and rain.required(rain.IOP / f"iop{i}" / exp / "untar-output", d)
                and cache_valid(cache_path(rain.IOP / f"iop{i}" / exp / "untar-output", d))
                and cache_valid(cache_path(rain.CONT / exp / "untar-output", d))
                for d in expected_dates(i))
            for exp in EXPS)
        for exp in EXPS:
            ready, dates, other, cont = pair_ready(i, exp)
            ready = ready and period_complete
            rec = {"period": i, "experiment": exp, "expected_days": len(dates), "ready": ready, "rendered": False}
            if ready and not output_valid(i, exp):
                try:
                    OUT.mkdir(parents=True, exist_ok=True)
                    rain.plot_pair(i, exp, dates, other, cont)
                    if not output_valid(i, exp):
                        raise RuntimeError("plot_pair returned without six valid outputs")
                    marker = OUT / f"iop{i}_{exp}_pair_complete.json"
                    atomic_json(marker, {"period": i, "experiment": exp, "days": [d.isoformat() for d in dates], "time": time.time()})
                    rec["rendered"] = True
                except Exception as exc:
                    rec["error"] = repr(exc)
            elif ready:
                rec["rendered"] = True
            status["pairs"].append(rec)
    status["complete_pairs"] = sum(1 for x in status["pairs"] if x["rendered"])
    status["expected_pairs"] = 6
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    rain.epygram.init_env()
    LOG.mkdir(parents=True, exist_ok=True)
    status = {"started": time.time(), "mode": "watch" if args.watch else "once"}
    while True:
        status = run_once(status)
        status["updated"] = time.time()
        atomic_json(LOG / "iop_comparison_pipeline_status.json", status)
        if status["complete_pairs"] == 6 or not args.watch:
            atomic_json(LOG / "iop_comparison_pipeline_final.json", status)
            return 0 if status["complete_pairs"] == 6 else 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
