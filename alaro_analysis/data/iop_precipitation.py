"""Iop precipitation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from alaro_analysis.common.iop_settings import (
    CACHE,
    CONT,
    EXPS,
    IOP,
    LOG,
    NLEV,
    PERIODS,
    VARS,
    WORKERS,
    G,
)
from alaro_analysis.data.cache import save_npz


def fa(path):
    import epygram

    return epygram.formats.resource(str(path), "r")


def read(r, name):
    f = r.readfield(name)
    if getattr(f, "spectral", False):
        f.sp2gp()
    return np.asarray(f.getdata(), dtype=np.float64)


def fpath(root, d, h):
    return root / f"pf{d:%Y%m%d}" / f"pfABOFABOF+{h:04d}"


def required(root, d, rain=True):
    hs = range(25) if rain else range(1, 25)
    return all(fpath(root, d, h).is_file() for h in hs)


def available(root, start, end):
    out = []
    d = start
    while d <= end:
        p = root / f"pf{d:%Y%m%d}"
        # Continuous untar trees predate the marker convention; their
        # complete 25-file day is the completion evidence.  IOP extraction
        # must provide .untar_done so an in-progress day is never compared.
        continuous = root.parent.parent.name == "ALARO"
        if required(root, d) and (continuous or (p / ".untar_done").exists()):
            out.append(d)
        d += timedelta(days=1)
    return out


def preflight():
    LOG.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now().isoformat()
    rec = {"timestamp": stamp, "periods": {}}
    for i, (a, b) in PERIODS.items():
        key = f"iop{i}"
        rec["periods"][key] = {}
        for e in EXPS + ("control",):
            roots = (
                [IOP / key / e / "untar-output"] if e != "control" else [CONT / e / "untar-output"]
            )
            got = available(roots[0], a, b)
            rec["periods"][key][e] = {
                "available": [x.isoformat() for x in got],
                "missing": [
                    (a + timedelta(days=j)).isoformat()
                    for j in range((b - a).days + 1)
                    if a + timedelta(days=j) not in got
                ],
            }
    (LOG / "iop_rainfall_updraft_manifest.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    return rec


def reduce_day(root, d):
    # Include source tree in the cache key: continuous and IOP both contain
    # ``<experiment>/untar-output`` and must never share reductions.
    source_key = f"{root.parent.parent.name}_{root.parent.name}"
    cache = CACHE / source_key / f"{d:%Y%m%d}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        old = dict(np.load(cache, allow_pickle=False))
        if old.get("hourly", np.empty(0)).shape == (24,) + old.get("rain", np.empty(0)).shape:
            return old
        print(f"invalidating old cache with bad hourly shape: {cache}", flush=True)
    rain = []
    prof = {k: [] for k in ("extent", "flux", "cloud", "temp")}
    height = []
    for h in range(25):
        r = fa(fpath(root, d, h))
        try:
            x = sum(read(r, v) for v in VARS)
            rain.append(x)
            if h > 0:
                ext = []
                flux = []
                cloud = []
                temp = []
                z = []
                for lev in range(1, NLEV + 1):
                    mesh = read(r, f"S{lev:03d}UD_MESH_FRAC")
                    omg = read(r, f"S{lev:03d}UD_OMEGA")
                    cl = read(r, f"S{lev:03d}CLOUD_FRACTI")
                    te = read(r, f"S{lev:03d}TEMPERATURE")
                    gg = read(r, f"S{lev:03d}GEOPOTENTIEL")
                    finite = np.isfinite(mesh) & np.isfinite(omg)
                    ext.append(np.nanmean(mesh))
                    flux.append(
                        np.nanmean(
                            np.where(finite, np.where(mesh > 0, -omg * mesh / G, 0.0), np.nan)
                        )
                    )
                    cloud.append(np.nanmean(cl))
                    temp.append(np.nanmean(te))
                    z.append(np.nanmean(gg) / G)
                for k, v in (("extent", ext), ("flux", flux), ("cloud", cloud), ("temp", temp)):
                    prof[k].append(v)
                height.append(z)
        finally:
            r.close()
    a = np.stack(rain)
    out = {
        "rain": a[24] - a[0],
        "hourly": np.diff(a, axis=0),
        "height": np.nanmean(height, axis=0),
        "schema": np.array("rain_updraft_v2"),
    }
    for k, v in prof.items():
        out[k] = np.stack(v)
    save_npz(cache, **out)
    return out


def _reduce_task(task):
    root, dstr = task
    return str(root), dstr, reduce_day(Path(root), date.fromisoformat(dstr))["rain"].shape


def warm_continuous(key, available):
    tasks = []
    for e in EXPS:
        root = CONT / e / "untar-output"
        tasks.extend((str(root), dstr) for dstr in available)
    if not tasks:
        return
    print(f"{key}: warming {len(tasks)} continuous date caches with {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        done = 0
        for _ in pool.map(_reduce_task, tasks, chunksize=1):
            done += 1
            if done % 5 == 0 or done == len(tasks):
                print(f"{key}: continuous cache {done}/{len(tasks)}", flush=True)


def initialize_fa_environment():
    """Initialize the optional FA stack only when an IOP job starts."""
    import epygram

    epygram.init_env()
