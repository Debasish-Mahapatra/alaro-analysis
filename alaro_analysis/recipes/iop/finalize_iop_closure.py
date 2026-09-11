"""Wait for complete DDHB extraction, aggregate exact periods, and plot."""

import subprocess
import sys
import time
from datetime import date, timedelta

from alaro_analysis.common.paths import runs_path

ROOT = runs_path("iop-runs/analysis/ddh")
PER = {1: (date(2014, 1, 15), date(2014, 2, 28)), 2: (date(2014, 7, 29), date(2014, 8, 31))}
EX = ("graupel", "2mom", "NO3M")
REQ = (
    "QV.DHFDLABOF+0024.condcv.dta",
    "QV.DHFDLABOF+0024.condrs.dta",
    "QL.DHFDLABOF+0024.VQLM.dta",
    "QR.DHFDLABOF+0024.VQRM.dta",
)


def ready(i, e):
    a, b = PER[i]
    d = a
    n = 0
    while d <= b:
        p = ROOT / f"iop{i}" / e / f"DDH{d:%Y%m%d}"
        n += all((p / x).is_file() for x in REQ)
        d += timedelta(days=1)
    return n == (b - a).days + 1


def main():
    while not all(ready(i, e) for i in (1, 2) for e in EX):
        print("waiting for complete DDHB periods", flush=True)
        time.sleep(30)
    py = sys.executable
    module = "alaro_analysis.recipes.iop.make_iop_closure_budget"
    for i in (1, 2):
        subprocess.run([py, "-m", module, "--aggregate", "--iop", str(i)], check=True)
    subprocess.run([py, "-m", module], check=True)


if __name__ == "__main__":
    main()
