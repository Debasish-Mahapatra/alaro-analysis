#!/usr/bin/env python3
"""DDH condensation budgets for the GoAmazon IOPs, for all four experiments.

Runs the ``ddhb`` QV budget on every campaign IOP day and aggregates it into the
schema ``ddh.plot_condensation_partition.load_partition`` reads: ``altitude_km``,
``block__condcv``, ``block__condrs`` and the day list. The DDH mean temperature
profile (T = VCT0/VPP0/cp) is stored alongside, for the 0 degC line of the figure.

Sources: G1M, G2M and G2M-XCU are already extracted under ``IOP-runs/iop*/<exp>/
untar-ddh``; C1M's DDH exists only as one tarball per day under
``amazon-control-run/control-DDH/output``, so its campaign days are extracted here
first (idempotent - a day with its +0024 file is left alone).

G2M-XCU (NO3M) runs under the 2mom ddhb runtime and has no convective
condensation at all: it carries no parameterized convection.

Day and night budgets as well: the DDH files accumulate from the 00 UTC start, so a
window is the difference of two leads (``ddht -cDIFFE_EC2_EC1``). With Amazon local
time at UTC-4, day (06-18 LT) is 10-22 UTC and night is 00-10 UTC plus 22-24 UTC,
combined by duration. The three windows add back to the 24 h budget to within 0.1 %.

This recipe imports epygram but never matplotlib: the LFA reader can abort when
both live in one process.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import CACHE_DIRS, CONTROL_RUN, GOAMAZON_IOPS, IOP_RUNS
from alaro_analysis.common.paths import data_path, runs_path
from alaro_analysis.data.cache import save_npz

DEFAULT_WORK_DIR = CACHE_DIRS["closure_budget"]
TOOLBOX = data_path("ddhtoolbox")
RUNTIME_ROOT = runs_path("old-archive/DDH-processed/lead0024_VZ/_runtime")
CONTROL_TARBALLS = CONTROL_RUN.parent / "control-DDH" / "output"
LEAD_FILE = "DHFDLABOF+0024"
BLOCKS = ("condcv", "condrs")
# Window -> (start lead file or None for the run start, end lead file, hours).
WINDOWS = {
    "day": ("DHFDLABOF+0010", "DHFDLABOF+0022", 12),
    "night_early": (None, "DHFDLABOF+0010", 10),
    "night_late": ("DHFDLABOF+0022", "DHFDLABOF+0024", 2),
}
HALVES = {"day": ("day",), "night": ("night_early", "night_late")}
CP_DRY = 1004.0
# Experiment directory -> ddhb runtime. NO3M is a 2-moment variant, so it uses the
# 2mom budget lists, as the earlier IOP closure workflow did.
RUNTIMES = {"control": "control", "graupel": "graupel", "2mom": "2mom", "NO3M": "2mom"}
LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M", "NO3M": "G2M-XCU"}


def campaign_days(iop: int) -> list[date]:
    start, end = GOAMAZON_IOPS[iop]
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def source_day(work: Path, iop: int, experiment: str, day: date) -> Path:
    """Directory holding that day's DDH files, extracting the C1M tarball if needed."""
    token = f"DDH{day:%Y%m%d}"
    if experiment != "control":
        return IOP_RUNS / f"iop{iop}" / experiment / "untar-ddh" / token
    target = work / "control-ddh" / token
    if (target / LEAD_FILE).is_file():
        return target
    tarball = CONTROL_TARBALLS / f"{token}.tar.gz"
    if not tarball.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    # The tarballs hold bare DHFDLABOF+* members, so each day needs its own folder.
    with tarfile.open(tarball) as archive:
        archive.extractall(target)
    return target


def ddhb_env(experiment: str) -> dict[str, str]:
    runtime = RUNTIME_ROOT / RUNTIMES[experiment]
    lists = runtime / "ddh_budget_lists"
    env = os.environ.copy()
    env.update(
        DDHTOOLBOX=str(TOOLBOX),
        DDH_PLOT="",
        DDHB_BPS=str(lists),
        DDHI_LIST=str(lists / "alaro" / "conversion_list"),
        PATH=f"{TOOLBOX}/tools:{TOOLBOX}/tools/lfa:{TOOLBOX}/tools/.dd2gr/src:" + env["PATH"],
    )
    return env


def run_day(work: Path, iop: int, experiment: str, day: date, recompute: bool) -> str:
    """Run the QV budget of one day and keep its condensation blocks."""
    out = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
    wanted = [out / f"QV.{LEAD_FILE}.{block}.dta" for block in BLOCKS]
    if all(p.is_file() for p in wanted) and not recompute:
        return "cached"
    source = source_day(work, iop, experiment, day)
    if not (source / LEAD_FILE).is_file():
        return "missing-source"
    out.mkdir(parents=True, exist_ok=True)
    scratch = work / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddhb_", dir=str(scratch)) as tmp:
        completed = subprocess.run(
            ["ddhb", "-v", "alaro/QV", "-i", LEAD_FILE, "-Y", "VZ", "-r", tmp],
            cwd=source,
            env=ddhb_env(experiment),
            capture_output=True,
            text=True,
        )
        budget = Path(tmp) / "budget.alaro"
        if completed.returncode or not budget.is_dir():
            (out / "ddhb.log").write_text(completed.stdout[-4000:] + completed.stderr[-4000:])
            return "failed"
        for block in BLOCKS:
            name = f"QV.{LEAD_FILE}.{block}.dta"
            if not (budget / name).is_file():
                return f"no-{block}"
            shutil.copy2(budget / name, out / name)
    return "ok"


def run_day_windows(work: Path, iop: int, experiment: str, day: date, recompute: bool) -> str:
    """QV budgets of the day and night windows of one day (see WINDOWS)."""
    out = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
    wanted = [out / f"QV.{window}.{block}.dta" for window in WINDOWS for block in BLOCKS]
    if all(p.is_file() for p in wanted) and not recompute:
        return "cached"
    source = source_day(work, iop, experiment, day)
    needed = {f for start, end, _ in WINDOWS.values() for f in (start, end) if f}
    if not all((source / f).is_file() for f in needed):
        return "missing-source"
    out.mkdir(parents=True, exist_ok=True)
    scratch = work / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env = ddhb_env(experiment)
    with tempfile.TemporaryDirectory(prefix="ddhw_", dir=str(scratch)) as tmp:
        tmp = Path(tmp)
        for name in needed:  # short local names keep the Fortran tools' paths short
            (tmp / name).symlink_to(source / name)
        for window, (start, end, _) in WINDOWS.items():
            if start is None:
                (tmp / window).symlink_to(source / end)
            else:
                done = subprocess.run(
                    ["ddht", "-cDIFFE_EC2_EC1", f"-2{end}", f"-1{start}", f"-s{window}"],
                    cwd=tmp, env=env, capture_output=True, text=True,
                )
                if done.returncode or not (tmp / window).is_file():
                    (out / f"ddht_{window}.log").write_text(done.stdout[-4000:] + done.stderr[-4000:])
                    return f"ddht-failed-{window}"
            result = tmp / f"r_{window}"
            done = subprocess.run(
                ["ddhb", "-v", "alaro/QV", "-i", window, "-Y", "VZ", "-r", str(result)],
                cwd=tmp, env=env, capture_output=True, text=True,
            )
            budget = result / "budget.alaro"
            if done.returncode or not budget.is_dir():
                (out / f"ddhb_{window}.log").write_text(done.stdout[-4000:] + done.stderr[-4000:])
                return f"ddhb-failed-{window}"
            for block in BLOCKS:
                name = f"QV.{window}.{block}.dta"
                if not (budget / name).is_file():
                    return f"no-{block}-{window}"
                shutil.copy2(budget / name, out / name)
    return "ok"


def run_day_all(work: Path, iop: int, experiment: str, day: date, recompute: bool) -> str:
    status = run_day(work, iop, experiment, day, recompute)
    windows = run_day_windows(work, iop, experiment, day, recompute)
    return status if windows in ("ok", "cached") else f"{status}/{windows}"


def extract_days(work: Path, iop: int, experiment: str, days, workers: int, recompute: bool):
    """Run every day of one experiment; ddhb is a subprocess, so threads are enough."""
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_day_all, work, iop, experiment, day, recompute): day for day in days
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
    return counts


def temperature_profile(day_dirs: list[Path]) -> np.ndarray | None:
    """Day-mean DDH temperature profile in K, from T = VCT0/VPP0/cp."""
    import epygram

    total, count = None, None
    for folder in day_dirs:
        path = folder / LEAD_FILE
        if not path.is_file():
            continue
        try:
            resource = epygram.formats.resource(str(path), "r", fmt="LFA")
            fields = resource.listfields()
            if "VCT0" not in fields or "VPP0" not in fields:
                resource.close()
                continue
            vct = np.asarray(resource.readfield("VCT0").getdata(), float).ravel()
            vpp = np.asarray(resource.readfield("VPP0").getdata(), float).ravel()
            resource.close()
        except Exception:  # noqa: BLE001 - a corrupt day must not stop the mean
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            temperature = vct / np.where(vpp > 0, vpp, np.nan) / CP_DRY
        if total is None:
            total = np.zeros_like(temperature)
            count = np.zeros_like(temperature)
        finite = np.isfinite(temperature)
        total[finite] += temperature[finite]
        count[finite] += 1
    if total is None:
        return None
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def aggregate(work: Path, iop: int, experiment: str, days: list[date]) -> Path | None:
    """Mean the per-day blocks into one npz in the load_partition schema."""
    rows: dict[str, list[np.ndarray]] = {block: [] for block in BLOCKS}
    altitude = None
    used: list[str] = []
    sources: list[Path] = []
    for day in days:
        folder = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
        paths = {block: folder / f"QV.{LEAD_FILE}.{block}.dta" for block in BLOCKS}
        if not all(path.is_file() for path in paths.values()):
            continue
        for block, path in paths.items():
            table = np.loadtxt(path)
            altitude = table[:, 0].astype(float) if altitude is None else altitude
            rows[block].append(table[:, 1].astype(float))
        used.append(day.isoformat())
        sources.append(source_day(work, iop, experiment, day))
    if not used:
        return None
    payload = {
        "altitude_km": altitude,
        "days": np.asarray(used),
        **{f"block__{block}": np.nanmean(rows[block], axis=0) for block in BLOCKS},
    }
    temperature = temperature_profile(sources)
    if temperature is not None:
        payload["temperature_k"] = temperature
    out = work / "aggregates" / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{experiment}_QV.npz"
    save_npz(path, **payload)
    print(
        f"[IOP{iop} {LABELS[experiment]}] aggregated {len(used)}/{len(days)} days -> {path}",
        flush=True,
    )
    return path


def aggregate_halves(work: Path, iop: int, experiment: str, days: list[date]) -> None:
    """Day and night means, in the same schema as the 24 h aggregate.

    Night is the duration-weighted mean of its two windows; the 0 degC temperature
    profile is the 24 h one (the DDH files carry no per-window mean temperature).
    """
    out = work / "aggregates" / f"iop{iop}"
    whole = out / f"{experiment}_QV.npz"
    temperature = None
    if whole.is_file():
        with np.load(whole, allow_pickle=True) as cache:
            temperature = cache["temperature_k"] if "temperature_k" in cache.files else None
    for half, windows in HALVES.items():
        rows: dict[str, list[np.ndarray]] = {block: [] for block in BLOCKS}
        altitude, used = None, []
        for day in days:
            folder = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
            paths = {(w, b): folder / f"QV.{w}.{b}.dta" for w in windows for b in BLOCKS}
            if not all(p.is_file() for p in paths.values()):
                continue
            hours = sum(WINDOWS[w][2] for w in windows)
            for block in BLOCKS:
                tables = [np.loadtxt(paths[(w, block)]) for w in windows]
                altitude = tables[0][:, 0].astype(float) if altitude is None else altitude
                rows[block].append(
                    sum(WINDOWS[w][2] * t[:, 1] for w, t in zip(windows, tables)) / hours
                )
            used.append(day.isoformat())
        if not used:
            continue
        payload = {
            "altitude_km": altitude,
            "days": np.asarray(used),
            "window_utc": np.asarray(
                [f"{w}:{WINDOWS[w][0] or 'start'}->{WINDOWS[w][1]}" for w in windows]
            ),
            **{f"block__{block}": np.nanmean(rows[block], axis=0) for block in BLOCKS},
        }
        if temperature is not None:
            payload["temperature_k"] = temperature
        path = out / f"{experiment}_QV_{half}.npz"
        save_npz(path, **payload)
        print(f"[IOP{iop} {LABELS[experiment]}] {half}: {len(used)}/{len(days)} days -> {path}",
              flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(RUNTIMES), choices=list(RUNTIMES))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--recompute", action="store_true", help="Rerun ddhb for existing days.")
    return parse_configured_args(p, "build_iop_closure_budgets", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for iop in args.iops:
        days = campaign_days(iop)
        for experiment in args.experiments:
            counts = extract_days(
                args.work_dir, iop, experiment, days, args.workers, args.recompute
            )
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"[IOP{iop} {LABELS[experiment]}] {len(days)} days: {summary}", flush=True)
            aggregate(args.work_dir, iop, experiment, days)
            aggregate_halves(args.work_dir, iop, experiment, days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
