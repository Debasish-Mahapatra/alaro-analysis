#!/usr/bin/env python3
"""Observed raindrop size distributions near Manaus for the wet and dry seasons of the IOPs.

The ARM laser disdrometer (maoldS10, strict QC merge, 1-min spectra) runs from
25 Sep 2014 to Dec 2015, so it covers IOP2 for three weeks and IOP1 not at all. The
seasonal regimes are therefore matched by calendar instead of by year:

    wet  ~ IOP1:  1 Feb - 31 Mar 2015
    dry  ~ IOP2: 15 Aug - 15 Oct of 2014 and 2015
    iop2_2014:   25 Sep - 15 Oct 2014, the exact IOP2 overlap (a check on the dry set)
    dry_2015:    15 Aug - 15 Oct 2015, the IOP2 window a year later, when the
                 instrument ran for the whole season

Same QC (rain rate > 0.1 mm/h, Z < 55 dBZ, >= 5 consecutive minutes), the same
empirical parameters as the paper's DSD figure (Dm = M4/M3, log10 Nw from LWC and Dm,
instrument spectrum truncated at 0.312 mm) and the same histograms and rain-rate law
as the model statistics (``data.microphysics_columns``). Writes
``<cache>/disdrometer/disdrometer_<season>.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.dsd import apply_disdrometer_qc, empirical_dsd_parameters
from alaro_analysis.common.iop_settings import CACHE_DIRS
from alaro_analysis.common.observation_settings import CACHE_DIR, OBS_ZIP
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.disdrometer import extract_strict_obs
from alaro_analysis.data.microphysics_columns import DM_EDGES, LOGNW_EDGES, RAIN_EDGES, _rain_rate

DEFAULT_WORK_DIR = CACHE_DIRS["microphysics"] / "disdrometer"
SEASONS = {
    "wet": [("2015-02-01", "2015-03-31")],
    "dry": [("2014-08-15", "2014-10-15"), ("2015-08-15", "2015-10-15")],
    "iop2_2014": [("2014-09-25", "2014-10-15")],
    "dry_2015": [("2015-08-15", "2015-10-15")],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    return parse_configured_args(p, "build_iop_disdrometer_seasons", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = extract_strict_obs(OBS_ZIP, CACHE_DIR)
    with Dataset(target) as ds:
        nd = np.asarray(ds["number_density_drops"][:], dtype=float)
        widths = np.asarray(ds["class_size_width"][:], dtype=float)
        centers = np.asarray(ds["particle_size"][:], dtype=float)
        m3 = np.asarray(ds["moment3"][:], dtype=float)
        m4 = np.asarray(ds["moment4"][:], dtype=float)
        m5 = np.asarray(ds["moment5"][:], dtype=float)
        rr = np.asarray(ds["precip_rate"][:], dtype=float)
        z = np.asarray(ds["equivalent_radar_reflectivity"][:], dtype=float)
        tv = ds["time"]
        times = np.array(num2date(tv[:], tv.units, only_use_cftime_datetimes=False), dtype="datetime64[m]")
    qc = apply_disdrometer_qc(precip_rate_mm_h=rr, reflectivity_dbz=z)
    par = empirical_dsd_parameters(number_density=nd, bin_centers_mm=centers, bin_widths_mm=widths,
                                   moment3=m3, moment4=m4, moment5=m5)
    ok = qc & np.isfinite(par["dm_mm"]) & (par["dm_mm"] > 0) & np.isfinite(par["log_nw"]) & (par["lwc_g_m3"] > 0)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for season, spans in SEASONS.items():
        sel = np.zeros_like(ok)
        for first, last in spans:
            sel |= (times >= np.datetime64(first)) & (times < np.datetime64(last) + np.timedelta64(1, "D"))
        use = ok & sel
        dm, lognw, lwc = par["dm_mm"][use], par["log_nw"][use], par["lwc_g_m3"][use]
        r = _rain_rate(lwc, dm)
        save_npz(
            args.work_dir / f"disdrometer_{season}.npz",
            dsd_hist=np.histogram2d(dm, lognw, [DM_EDGES, LOGNW_EDGES])[0],
            dsd_dm_by_rr=np.histogram2d(r, dm, [RAIN_EDGES, DM_EDGES])[0],
            dsd_n=np.array([dm.size]),
            n_minutes_in_season=np.array([int(sel.sum())]),
            rain_mm=np.array([float(np.nansum(np.where(use, rr, 0.0)) / 60.0)]),
            dm_edges=DM_EDGES, lognw_edges=LOGNW_EDGES, rain_edges=RAIN_EDGES,
        )
        print(f"[{season}] {dm.size} QC'd rain minutes of {int(sel.sum())}; median Dm {np.median(dm):.2f} mm, "
              f"median log10 Nw {np.median(lognw):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
