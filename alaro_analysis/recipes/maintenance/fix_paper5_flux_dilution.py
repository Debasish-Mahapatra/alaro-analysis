#!/usr/bin/env python3
"""Undo the radar-mask dilution in the cached figure-5 updraft mass flux.

build_updraft_cloud_diurnal_from_netcdf.py used to evaluate

    np.where(mesh > 0, (-omega * mesh) / g, 0.0)

on the padded 75x72 netcdf crop.  Only 2822 of those 5400 cells are inside the
SIPAM radar mask; the rest are NaN, and `NaN > 0` is False, so they entered the
mean as zeros.  The following nanmean therefore divided by 5400 rather than
2822 and every cached flux came out low by exactly 75*72/2822 = 1.9135364989.

That the factor is exact (not approximate) was checked on 12 files spanning
both years and all three experiments: after the buggy `where` the array has
exactly 5400 finite cells at every level, and good/buggy matched 5400/2822 to
4e-16.  So the cache can be rescaled instead of recomputed -- the result is
bit-identical to rerunning the now-fixed builder.

Rescales <exp>_flux.npz in place (originals are in uncorrected_flux_backup/)
and stamps the correction into the npz so it cannot be applied twice.
"""

from __future__ import annotations

import numpy as np

from alaro_analysis.common.paths import runs_path
from alaro_analysis.data.cache import save_npz

CACHE = runs_path("processed-data/paper5_from_netcdf")
EXPS = ("control", "graupel", "2mom")
N_CROP_CELLS = 75 * 72
N_MASK_POINTS = 2822
FACTOR = N_CROP_CELLS / N_MASK_POINTS
STAMP = "radar_mask_dilution_corrected"


def main() -> int:
    for exp in EXPS:
        path = CACHE / f"{exp}_flux.npz"
        with np.load(path) as d:
            payload = {k: d[k] for k in d.files}
        if STAMP in payload:
            print(f"[{exp}] already corrected (factor {float(payload[STAMP]):.10f}) -- skipped")
            continue
        before = float(np.nanmax(payload["mean"]))
        payload["mean"] = np.asarray(payload["mean"], dtype=np.float64) * FACTOR
        payload[STAMP] = np.array([FACTOR], dtype=np.float64)
        save_npz(path, **payload)
        print(
            f"[{exp}] flux max {before:.6g} -> {float(np.nanmax(payload['mean'])):.6g} "
            f"kg m-2 s-1  (x{FACTOR:.10f})"
        )
    print(f"\nCorrected {CACHE}; rerun make_paper_updraft_cloud_relative_panel.py to re-render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
