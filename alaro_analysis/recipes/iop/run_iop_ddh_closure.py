#!/usr/bin/env python3
"""Run the required QV/QL/QR DDHB budgets for all complete IOP DDH days."""

import os
import shutil
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.paths import data_path, runs_path

ROOT = runs_path("")
TB = data_path("ddhtoolbox")
RUNTIME = ROOT / "DDH-processed" / "lead0024_VZ" / "_runtime"
OUT = ROOT / "iop-runs" / "analysis" / "ddh"
PERIODS = {1: (date(2014, 1, 15), date(2014, 2, 28)), 2: (date(2014, 7, 29), date(2014, 8, 31))}
EXPS = ("graupel", "2mom", "NO3M")
VARS = ("QV", "QL", "QR")
REQ = {
    "QV": ("QV.DHFDLABOF+0024.condcv.dta", "QV.DHFDLABOF+0024.condrs.dta"),
    "QL": ("QL.DHFDLABOF+0024.VQLM.dta",),
    "QR": ("QR.DHFDLABOF+0024.VQRM.dta",),
}


def run(iop, exp, d):
    src = ROOT / "iop-runs" / f"iop{iop}" / exp / "untar-output" / f"DDH{d:%Y%m%d}"
    out = OUT / f"iop{iop}" / exp / f"DDH{d:%Y%m%d}"
    if not (src / "DHFDLABOF+0024").is_file():
        return "missing-source"
    out.mkdir(parents=True, exist_ok=True)
    if all((out / n).is_file() for ns in REQ.values() for n in ns):
        return "done"
    env = os.environ.copy()
    env.update(DDHTOOLBOX=str(TB), DDH_PLOT="")
    rt = RUNTIME / ("graupel" if exp == "graupel" else "2mom")
    env["DDHB_BPS"] = str(rt / "ddh_budget_lists")
    env["DDHI_LIST"] = str(rt / "ddh_budget_lists" / "alaro" / "conversion_list")
    env["PATH"] = f"{TB}/tools:{TB}/tools/lfa:{TB}/tools/.dd2gr/src:" + env["PATH"]
    for var in VARS:
        with tempfile.TemporaryDirectory(prefix="iopddh_") as tmp:
            log = out / f"{var}.run.log"
            with log.open("w") as fh:
                rc = subprocess.run(
                    ["ddhb", "-v", f"alaro/{var}", "-i", "DHFDLABOF+0024", "-Y", "VZ", "-r", tmp],
                    cwd=src,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                ).returncode
            budget = Path(tmp) / "budget.alaro"
            if rc or not budget.is_dir():
                return f"failed-{var}"
            for p in budget.glob("*.dta"):
                shutil.copy2(p, out / p.name)
    return "ok"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--iop", type=int, choices=(1, 2))
    a = parse_configured_args(ap, "run_iop_ddh_closure", argv=None)
    for iop in (a.iop,) if a.iop else (1, 2):
        start, end = PERIODS[iop]
        d = start
        while d <= end:
            for exp in EXPS:
                print(iop, exp, d, run(iop, exp, d), flush=True)
            d += timedelta(days=1)


if __name__ == "__main__":
    main()
