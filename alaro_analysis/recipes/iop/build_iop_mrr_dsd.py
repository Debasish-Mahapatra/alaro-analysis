#!/usr/bin/env python3
"""Rain drop size distributions from the CHUVA micro rain radar, per GoAmazon IOP.

The ARM laser disdrometer only starts on 25 Sep 2014, so IOP1 has no surface drop
spectra. The CHUVA MRR at Manacapuru UEA ran through both IOPs and measures a drop
spectrum per range gate every minute, which fills that gap.

Everything downstream of the spectrum is the disdrometer path, unchanged: the same QC
(``common.dsd.apply_disdrometer_qc``), the same Dm, Nw and LWC
(``common.dsd.empirical_dsd_parameters``), the same 0.312 mm truncation, the same
histogram edges as ``build_iop_disdrometer_seasons`` and
``data.microphysics_columns``. Only the instrument differs.

MRR specifics:

* the level-1 files are the manufacturer's processed product; ``mrrN`` is a number
  density per METRE of diameter, so it is divided by 1000 to get m-3 mm-1 (verified
  against the file's own LWC: the rebuilt LWC matches to about 4 %)
* one gate is taken, by default gate 3 at about 800 m, the lowest height clear of the
  instrument's near-field; the surface is not observed
* moments are integrated over the measured bins above 0.312 mm

Writes ``<cache>/mrr/mrr_iop<N>.npz`` in the same schema the disdrometer caches use.
"""

from __future__ import annotations

import argparse
import bz2
import re
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.dsd import apply_disdrometer_qc, empirical_dsd_parameters
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS
from alaro_analysis.data.cache import save_npz
from alaro_analysis.data.microphysics_columns import DM_EDGES, LOGNW_EDGES, RAIN_EDGES

CHUVA = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Obs/CHUVA")
# IOP -> directory of daily AveData files (the two campaigns sit in separate trees)
SOURCES = {
    1: CHUVA / "mrr/st_manacapuru_uea/AveData",
    2: CHUVA / "mrr_iop2/st_manacapuru_uea/AveData",
}
IMPROTOO = Path(__file__).resolve().parents[3] / "cache" / "pylibs"
GATE = 3                 # about 800 m above the site
ND_PER_METRE = 1000.0    # mrrN is per metre of diameter
DMIN_MM = 0.312          # the Parsivel's minimum detectable diameter
RAIN_MIN_MM_H = 0.1      # same floor as the disdrometer QC


def _rain_rate(lwc_g_m3: np.ndarray, dm_mm: np.ndarray) -> np.ndarray:
    """Atlas et al. (1973) mass-weighted fall speed, as the other DSD caches use."""
    speed = 9.65 - 10.3 * np.exp(-0.6 * dm_mm)
    return 3.6 * lwc_g_m3 * np.maximum(speed, 0.0)


def read_day(path: Path, gate: int) -> dict[str, np.ndarray] | None:
    """One day of MRR spectra at one gate: N(D) [m-3 mm-1], bin centres, rain rate, Z."""
    sys.path.insert(0, str(IMPROTOO))
    import IMProToo

    with tempfile.TemporaryDirectory(prefix="mrr_", dir=str(path.parent)) as tmp:
        plain = Path(tmp) / path.name.replace(".bz2", "")
        try:
            with bz2.open(path, "rb") as fh, plain.open("wb") as out:
                shutil.copyfileobj(fh, out)
            data = IMProToo.mrrProcessedData(str(plain))
        except Exception as exc:  # noqa: BLE001 - a bad day must not stop the pass
            print(f"  {path.name}: unreadable ({exc})", flush=True)
            return None
    fill = lambda a: np.ma.filled(np.asarray(a, dtype=float), np.nan)  # noqa: E731
    nd = fill(data.mrrN)[:, gate, :] / ND_PER_METRE
    diameters = fill(data.mrrD)[:, gate, :]
    if not np.isfinite(diameters).any():
        return None
    return {
        "nd": nd,
        "diameters": diameters,
        "rain": fill(data.mrrRR)[:, gate],
        "reflectivity": fill(data.mrrCapitalZ)[:, gate],
    }


def moments(nd: np.ndarray, diameters: np.ndarray) -> tuple[np.ndarray, ...]:
    """M3, M4, M5 over the bins above DMIN_MM, and the widths used."""
    centres = np.nanmedian(diameters, axis=0)
    widths = np.gradient(centres)
    keep = np.isfinite(nd) & np.isfinite(diameters) & (diameters >= DMIN_MM) & (nd > 0.0)
    weight = np.where(keep, nd * widths[None, :], 0.0)
    powers = [np.nansum(weight * np.where(keep, diameters, 0.0) ** k, axis=1) for k in (3, 4, 5)]
    return (*powers, centres, widths, np.where(keep, nd, 0.0))


def build(iop: int, work: Path, gate: int) -> None:
    start, end = GOAMAZON_IOPS[iop]
    files = sorted(SOURCES[iop].rglob("*.mrr.bz2"))
    used: list[date] = []
    chunks: list[dict[str, np.ndarray]] = []
    for path in files:
        match = re.search(r"(\d{8})\d{6}\.mrr", path.name)
        if not match:
            continue
        token = match.group(1)
        day = date(int(token[:4]), int(token[4:6]), int(token[6:]))
        if not start <= day <= end:
            continue
        payload = read_day(path, gate)
        if payload is None:
            print(f"  {day}: unreadable, skipped", flush=True)
            continue
        chunks.append(payload)
        used.append(day)
    if not chunks:
        raise RuntimeError(f"no MRR days inside IOP{iop}")

    nd = np.concatenate([c["nd"] for c in chunks])
    diameters = np.concatenate([c["diameters"] for c in chunks])
    rain = np.concatenate([c["rain"] for c in chunks])
    reflectivity = np.concatenate([c["reflectivity"] for c in chunks])
    m3, m4, m5, centres, widths, nd_kept = moments(nd, diameters)
    parameters = empirical_dsd_parameters(
        number_density=nd_kept, bin_centers_mm=centres, bin_widths_mm=widths,
        moment3=m3, moment4=m4, moment5=m5,
    )
    qc = apply_disdrometer_qc(precip_rate_mm_h=rain, reflectivity_dbz=reflectivity)
    ok = (
        qc
        & np.isfinite(parameters["dm_mm"]) & (parameters["dm_mm"] > 0)
        & np.isfinite(parameters["log_nw"]) & (parameters["lwc_g_m3"] > 0)
    )
    dm, log_nw, lwc = parameters["dm_mm"][ok], parameters["log_nw"][ok], parameters["lwc_g_m3"][ok]
    rate = _rain_rate(lwc, dm)

    work.mkdir(parents=True, exist_ok=True)
    out = work / f"mrr_iop{iop}.npz"
    save_npz(
        out,
        dsd_hist=np.histogram2d(dm, log_nw, [DM_EDGES, LOGNW_EDGES])[0],
        dsd_dm_by_rr=np.histogram2d(rate, dm, [RAIN_EDGES, DM_EDGES])[0],
        dsd_n=np.array([dm.size]),
        n_minutes_in_season=np.array([int(rain.size)]),
        rain_mm=np.array([float(np.nansum(np.where(ok, rain, 0.0)) / 60.0)]),
        days=np.array([d.isoformat() for d in used]),
        gate=np.array([gate]),
        dm_edges=DM_EDGES, lognw_edges=LOGNW_EDGES, rain_edges=RAIN_EDGES,
    )
    print(
        f"[IOP{iop}] {len(used)} days, {rain.size} minutes, {dm.size} QC'd rain minutes; "
        f"median Dm {np.median(dm):.2f} mm, median log10 Nw {np.median(log_nw):.2f} -> {out}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=CACHE_DIRS["microphysics"] / "mrr")
    p.add_argument("--iops", nargs="+", type=int, default=list(GOAMAZON_IOPS))
    p.add_argument("--gate", type=int, default=GATE, help="MRR range gate (3 = about 800 m)")
    return parse_configured_args(p, "build_iop_mrr_dsd", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for iop in args.iops:
        build(iop, args.work_dir, args.gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
