#!/usr/bin/env python3
"""Warm-rain water held below 3 km, from the IOP DDH, for all four runs.

``build_iop_closure_budgets`` keeps only the QV condensation blocks, so the warm
reservoir of the paper's closure figure (cloud water plus rain water held below
3 km) has no IOP cache. Two quantities are cached per IOP and experiment:

    <work>/aggregates/iop<N>/<experiment>_QLQR.npz
        altitude_km
        content_gkg    cloud + rain water content, g/kg, the mean of the budget
                       window's initial and final state: (VQL0+VQL1+VQR0+VQR1)
                       / 2 / VPP0 * 1000, read from the DDH LFA
        block__VQLM, block__VQRM   the ddhb budget blocks of the same days
        days

**The two are not the same thing.** ``VQLM``/``VQRM`` are "EAU CONDENS : MEAN
TENDENCY" in g/kg/day (ddhtoolbox ``conversion_list``), i.e. how the warm water
changes over the budget window, not how much is there; single days go negative.
The published figure's "warm reservoir" bar integrates those tendencies.
``content_gkg`` is the actual water content and is what the rebuilt figure uses.

ddhb is a subprocess (about 1.7 s per budget), so the days run on threads.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import GOAMAZON_IOPS
from alaro_analysis.data.cache import save_npz
from alaro_analysis.recipes.iop.build_iop_closure_budgets import (
    DEFAULT_WORK_DIR,
    LABELS,
    LEAD_FILE,
    campaign_days,
    ddhb_env,
    source_day,
)

# LFA fields of the warm water content: layer mass of cloud water and rain at the
# start and end of the budget window, and the layer air mass to divide by.
CONTENT_FIELDS = ("VQL0", "VQL1", "VQR0", "VQR1", "VPP0")

# DDH variable -> the block holding its mean mass mixing ratio
WARM_BLOCKS = {"QL": "VQLM", "QR": "VQRM"}
EXPERIMENTS = tuple(LABELS)


def run_day(work: Path, iop: int, experiment: str, day: date, recompute: bool) -> str:
    """Run the QL and QR budgets of one day and keep their mass blocks."""
    out = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
    wanted = {
        variable: out / f"{variable}.{LEAD_FILE}.{block}.dta"
        for variable, block in WARM_BLOCKS.items()
    }
    if all(path.is_file() for path in wanted.values()) and not recompute:
        return "cached"
    source = source_day(work, iop, experiment, day)
    if not (source / LEAD_FILE).is_file():
        return "missing-source"
    out.mkdir(parents=True, exist_ok=True)
    scratch = work / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for variable, block in WARM_BLOCKS.items():
        if wanted[variable].is_file() and not recompute:
            continue
        with tempfile.TemporaryDirectory(prefix="ddhb_", dir=str(scratch)) as tmp:
            completed = subprocess.run(
                ["ddhb", "-v", f"alaro/{variable}", "-i", LEAD_FILE, "-Y", "VZ", "-r", tmp],
                cwd=source,
                env=ddhb_env(experiment),
                capture_output=True,
                text=True,
            )
            name = f"{variable}.{LEAD_FILE}.{block}.dta"
            produced = Path(tmp) / "budget.alaro" / name
            if completed.returncode or not produced.is_file():
                (out / f"ddhb_{variable}.log").write_text(
                    completed.stdout[-4000:] + completed.stderr[-4000:]
                )
                return f"failed-{variable}"
            shutil.copy2(produced, wanted[variable])
    return "ok"


def content_profile(day_dirs: list[Path]) -> np.ndarray | None:
    """Day-mean cloud + rain water content (g/kg), from the DDH LFA state fields.

    DDH stores layer masses, so the mixing ratio of a layer is its water mass over
    the air mass VPP0; the budget window's initial and final states are averaged.
    """
    import epygram

    total = count = None
    for folder in day_dirs:
        path = folder / LEAD_FILE
        if not path.is_file():
            continue
        try:
            resource = epygram.formats.resource(str(path), "r", fmt="LFA")
            available = resource.listfields()
            if any(name not in available for name in CONTENT_FIELDS):
                resource.close()
                continue
            values = {
                name: np.asarray(resource.readfield(name).getdata(), float).ravel()
                for name in CONTENT_FIELDS
            }
            resource.close()
        except Exception:  # noqa: BLE001 - a corrupt day must not stop the mean
            continue
        water = 0.5 * (values["VQL0"] + values["VQL1"] + values["VQR0"] + values["VQR1"])
        air = values["VPP0"]
        with np.errstate(divide="ignore", invalid="ignore"):
            content = np.where(air > 0, water / air, np.nan) * 1000.0
        if total is None:
            total = np.zeros_like(content)
            count = np.zeros_like(content)
        finite = np.isfinite(content)
        total[finite] += content[finite]
        count[finite] += 1
    if total is None:
        return None
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def aggregate(work: Path, iop: int, experiment: str, days: list[date]) -> Path | None:
    """Mean the per-day warm blocks and water content into one npz."""
    rows: dict[str, list[np.ndarray]] = {block: [] for block in WARM_BLOCKS.values()}
    altitude = None
    used: list[str] = []
    sources: list[Path] = []
    for day in days:
        folder = work / f"iop{iop}" / experiment / f"DDH{day:%Y%m%d}"
        paths = {
            block: folder / f"{variable}.{LEAD_FILE}.{block}.dta"
            for variable, block in WARM_BLOCKS.items()
        }
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
    out = work / "aggregates" / f"iop{iop}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{experiment}_QLQR.npz"
    payload = {
        "altitude_km": altitude,
        "days": np.asarray(used),
        **{f"block__{block}": np.nanmean(rows[block], axis=0) for block in rows},
    }
    content = content_profile(sources)
    if content is not None:
        payload["content_gkg"] = content
    save_npz(path, **payload)
    print(f"[IOP{iop} {LABELS[experiment]}] aggregated {len(used)}/{len(days)} days -> {path}",
          flush=True)
    return path


def build(work: Path, iop: int, experiment: str, workers: int, recompute: bool) -> None:
    days = campaign_days(iop)
    outcomes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_day, work, iop, experiment, day, recompute): day for day in days
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items()))
    print(f"[IOP{iop} {LABELS[experiment]}] {len(days)} days: {summary}", flush=True)
    aggregate(work, iop, experiment, days)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--recompute", action="store_true", help="Rerun existing budgets.")
    return parse_configured_args(p, "build_iop_warm_reservoir", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for iop in args.iops:
        for experiment in args.experiments:
            build(args.work_dir, iop, experiment, args.workers, args.recompute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
