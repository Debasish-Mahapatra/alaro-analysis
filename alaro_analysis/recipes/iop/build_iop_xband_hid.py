#!/usr/bin/env python3
"""Observed hydrometeor types over IOP1, from the CHUVA X-band dual-pol radar at T3.

The radar is the only polarimetric instrument covering IOP1, so it is the only
observation that can say which hydrometeor species were actually present - the
question the graupel runs exist to answer.

Per volume: Z, ZDR, KDP and RhoHV are read from the level-1 GAMIC files, gates with
RhoHV <= 0.8, Z <= 0 dBZ or range > 60 km are dropped, and the CSU fuzzy-logic scheme
(Dolan & Rutledge 2009) is run with its X-band membership functions and the
temperature profile of the nearest ARM radiosonde at the same site. Counts are binned
by height in 0.5 km layers and summed over the IOP.

Writes ``<cache>/xband/xband_hid_iop<N>.npz``: counts (height x class), the volumes
and gates used, the days covered and the height edges. Needs no model data.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.xband_hid import CLASS_NAMES, HEIGHT_EDGES_KM, process_volume

RADAR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Obs/CHUVA/xband_t3_level1")
SONDES = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ARM/radiosond_data/maosondewnpnM1.b1")
PYLIBS = Path(__file__).resolve().parents[3] / "cache" / "pylibs"
SONDE_LEVELS = 300  # thinned profile, enough for a melting level


def sonde_profiles(days: list[date]) -> list[tuple[datetime, np.ndarray, np.ndarray]]:
    """(launch time, height km, temperature C) of every sonde on those days."""
    from netCDF4 import Dataset

    wanted = {f"{day:%Y%m%d}" for day in days}
    out = []
    for path in sorted(SONDES.glob("maosondewnpnM1.b1.*.cdf")):
        token = path.name.split(".")[2]
        if token not in wanted:
            continue
        try:
            with Dataset(path) as data:
                alt = np.asarray(data["alt"][:], dtype=float) / 1000.0
                temp = np.asarray(data["tdry"][:], dtype=float)
        except Exception:  # noqa: BLE001 - a broken sonde is simply skipped
            continue
        good = np.isfinite(alt) & np.isfinite(temp) & (temp > -120.0) & (temp < 60.0)
        if good.sum() < 50:
            continue
        alt, temp = alt[good], temp[good]
        order = np.argsort(alt)
        alt, temp = alt[order], temp[order]
        step = max(1, alt.size // SONDE_LEVELS)
        stamp = datetime.strptime(f"{token}{path.name.split('.')[3]}", "%Y%m%d%H%M%S")
        out.append((stamp, alt[::step], temp[::step]))
    return out


def nearest_profile(profiles, stamp: datetime):
    return min(profiles, key=lambda p: abs((p[0] - stamp).total_seconds()))


def volume_time(path: Path) -> datetime:
    token = path.name.split("-")[1].split(".")[0]      # 117BRX-20140215010008.HDF5
    return datetime.strptime(token, "%Y%m%d%H%M%S")


def build(iop: int, work: Path, workers: int, zdr_offset: float) -> None:
    start, end = GOAMAZON_IOPS[iop]
    days = [d for d in sorted(p.name for p in RADAR.iterdir() if p.is_dir())
            if start.isoformat() <= d <= end.isoformat()]
    if not days:
        print(f"[IOP{iop}] no radar days on disk, skipped", flush=True)
        return
    day_dates = [date.fromisoformat(d) for d in days]
    profiles = sonde_profiles(day_dates)
    if not profiles:
        raise RuntimeError("no usable radiosondes for these days")
    volumes = [p for day in days for p in sorted((RADAR / day).glob("*.HDF5"))]
    tasks = []
    for path in volumes:
        _, alt, temp = nearest_profile(profiles, volume_time(path))
        tasks.append((str(path), alt, temp, zdr_offset))
    print(f"[IOP{iop}] {len(days)} days, {len(volumes)} volumes, {len(profiles)} sondes, "
          f"{workers} workers", flush=True)

    counts = np.zeros((HEIGHT_EDGES_KM.size - 1, len(CLASS_NAMES)), dtype=np.int64)
    gates = 0
    problems: list[str] = []
    # fork plus HDF5 has corrupted the heap on this machine; spawn keeps workers clean
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for i, (hist, n, problem) in enumerate(pool.map(process_volume, tasks, chunksize=4), 1):
            counts += hist
            gates += n
            if problem:
                problems.append(problem)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)} volumes, {gates:,} classified gates", flush=True)

    work.mkdir(parents=True, exist_ok=True)
    tag = "" if zdr_offset == 0.0 else "_zdrcal"
    out = work / f"xband_hid_iop{iop}{tag}.npz"
    save_npz(
        out, counts=counts, height_edges_km=HEIGHT_EDGES_KM,
        classes=np.array([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]),
        days=np.array(days), n_volumes=np.array([len(volumes)]), n_gates=np.array([gates]),
    )
    share = counts.sum(axis=0) / max(counts.sum(), 1)
    summary = ", ".join(
        f"{CLASS_NAMES[i]} {share[i - 1]:.1%}" for i in sorted(CLASS_NAMES) if share[i - 1] > 0.01
    )
    print(f"[IOP{iop}] {gates:,} gates, {len(problems)} bad volumes -> {out}\n  {summary}",
          flush=True)
    if problems:
        (work / f"xband_hid_iop{iop}_problems.log").write_text("\n".join(problems[:500]) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=CACHE_DIRS["microphysics"] / "xband")
    p.add_argument("--iops", nargs="+", type=int, default=[1])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--zdr-offset", type=float, default=0.0,
                   help="system ZDR bias in dB, subtracted before classifying "
                        "(+0.50 dB measured from the birdbath scans over IOP1)")
    return parse_configured_args(p, "build_iop_xband_hid", argv=argv)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(PYLIBS))   # csu_radartools lives beside the repo, not in the env
    args = parse_args(argv)
    workers = resolve_workers(args.workers)
    for iop in args.iops:
        build(iop, args.work_dir, workers, args.zdr_offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
