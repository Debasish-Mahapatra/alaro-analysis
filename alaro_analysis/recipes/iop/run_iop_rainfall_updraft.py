#!/usr/bin/env python3
"""Native-grid IOP versus continuous rainfall and updraft workflow.

This deliberately reads the raw FA stream (rather than a cropped converter
product), and caches one complete date/experiment reduction at a time.  It is
safe to rerun while IOP extraction is in progress: only dates with
``.untar_done`` and all required lead files are admitted to a comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.iop_settings import (
    CACHE,
    CONT,
    EXPS,
    FH,
    IOP,
    LABEL,
    LOG,
    NLEV,
    OFFSET,
    OUT,
    PERIODS,
    ROOT,
    VARS,
    WORKERS,
    G,
)
from alaro_analysis.common.paths import runs_path
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.iop_precipitation import (
    _reduce_task,
    available,
    fa,
    fpath,
    initialize_fa_environment,
    preflight,
    read,
    reduce_day,
    required,
    warm_continuous,
)
from alaro_analysis.plotting.iop_precipitation import plot_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--period", type=int, choices=(1, 2))
    ap.add_argument("--interval", type=int, default=300)
    args = parse_configured_args(ap, "run_iop_rainfall_updraft", argv=None)
    initialize_fa_environment()
    while True:
        manifest = preflight()
        if args.run:
            for i in (args.period,) if args.period else (1, 2):
                a, b = PERIODS[i]
                key = f"iop{i}"
                if args.watch:
                    # Warm the continuous cache immediately, independent of extraction.
                    warm_continuous(key, manifest["periods"][key]["control"]["available"])
                for e in EXPS:
                    ir = IOP / key / e / "untar-output"
                    cr = CONT / e / "untar-output"
                    # Every plotted quantity uses the identical date intersection of
                    # all three paired IOP experiments and the shared continuous run.
                    common = set(manifest["periods"][key]["control"]["available"])
                    for ee in EXPS:
                        common &= set(manifest["periods"][key][ee]["available"])
                    dates = sorted(date.fromisoformat(x) for x in common)
                    other = [reduce_day(ir, d) for d in dates]
                    cont = [reduce_day(cr, d) for d in dates]
                    expected = (b - a).days + 1
                    if len(dates) == expected:
                        plot_pair(i, e, dates, other, cont)
                    elif dates:
                        print(
                            f"{key}: holding {len(dates)}/{expected} common dates; no figures",
                            flush=True,
                        )
        if not args.watch:
            return
        print(f"watch: waiting {args.interval}s for complete IOP extraction", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
