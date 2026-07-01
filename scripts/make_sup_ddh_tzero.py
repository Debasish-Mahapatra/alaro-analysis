"""Cache the DDH 0 C level (VCT0/VPP0/cp) per experiment, 15-day mean.

epygram-only (NO matplotlib): the LFA C library can corrupt the heap when used
in the same process as matplotlib rendering, so freezing levels are computed here
and cached; the plotting scripts just read the npz.

Output: processed-data/data/ddh_tzero_km.npz  (key = experiment name -> 0C km)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import epygram

from alaro_analysis.ddh.io import CP_DRY, UNTAR_ROOT, load_budget

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/ddh_tzero_km.npz")
DDH_DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]
EXPS = ["control", "graupel", "control_NIMELIT_1", "graupel_NIMELIT_1",
        "control_LNEBCV_F", "graupel_LNEBCV_F"]


def freezing_km(exp, alt):
    root = UNTAR_ROOT / exp / "output"
    acc = cnt = None
    for d in DDH_DAYS:
        f = root / d / "DHFDLABOF+0024"
        if not f.exists():
            continue
        try:
            r = epygram.formats.resource(str(f), "r", fmt="LFA")
            fl = r.listfields()
            if "VCT0" not in fl or "VPP0" not in fl:
                r.close(); continue
            vct = np.asarray(r.readfield("VCT0").getdata(), float).ravel()
            vpp = np.asarray(r.readfield("VPP0").getdata(), float).ravel()
            r.close()
        except Exception:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = vct / np.where(vpp > 0, vpp, np.nan) / CP_DRY
        if acc is None:
            acc = np.zeros_like(t); cnt = np.zeros_like(t)
        m = np.isfinite(t); acc[m] += t[m]; cnt[m] += 1
    if acc is None:
        return np.nan
    t = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    z = np.asarray(alt, float)
    n = min(z.size, t.size); z, d = z[:n], t[:n] - 273.15
    ok = np.isfinite(z) & np.isfinite(d); z, d = z[ok], d[ok]
    if z.size < 2:
        return np.nan
    o = np.argsort(z); z, d = z[o], d[o]
    cr = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if cr.size == 0:
        return np.nan
    i = int(cr[0]); w = d[i] / (d[i] - d[i + 1])
    return float(z[i] + w * (z[i + 1] - z[i]))


def main():
    alt = load_budget("control_base15", "QV", lead="0024")["altitude_km"]
    out = {}
    for e in EXPS:
        out[e] = freezing_km(e, alt)
        print(f"{e}: 0C = {out[e]:.2f} km", flush=True)
    np.savez(OUT, **out)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
