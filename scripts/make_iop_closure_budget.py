#!/usr/bin/env python3
"""Closure-budget comparison for the six IOP/continuous pairs.

The script reads the date-complete DDH aggregates made by the IOP DDH
workflow.  It intentionally discovers aggregates below ``iop-runs/analysis``
so it remains usable while the extraction workflow is writing its final cache
layout.  Continuous data are read from the established DDH-processed cache.
For each IOP and microphysics experiment a three-panel, 450 dpi figure is
written (continuous, IOP, and IOP-minus-continuous), together with a text file
containing all plotted profiles, dates, units, and source paths.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
IOP_ROOT = ROOT / "iop-runs"
CONT_AGG = ROOT / "DDH-processed" / "_aggregated"
OUT = IOP_ROOT / "figures"
RAIN_CACHE = IOP_ROOT / "analysis" / "cache"
DPI = 450
PERIODS = {1: (date(2014, 1, 15), date(2014, 2, 28)),
           2: (date(2014, 7, 29), date(2014, 8, 31))}
EXPS = ("graupel", "2mom", "NO3M")
LABEL = {"graupel": "G1M", "2mom": "G2M", "NO3M": "G2M-XCU"}
COLORS = {"continuous": "#1f77b4", "IOP": "#d62728", "delta": "#2ca02c"}


def dates_for(iop: int):
    a, b = PERIODS[iop]
    return [a + timedelta(days=n) for n in range((b - a).days + 1)]


def aggregate_path(iop: int | None, exp: str, var: str) -> Path | None:
    """Find one aggregate, accepting the cache naming variants in use."""
    if iop is None:
        p = CONT_AGG / "lead0024_VZ" / f"{exp}_{var}.npz"
        return p if p.exists() else None
    roots = [IOP_ROOT / "analysis", IOP_ROOT]
    names = (f"{exp}_{var}.npz", f"iop{iop}_{exp}_{var}.npz",
             f"{exp}_iop{iop}_{var}.npz")
    for root in roots:
        for name in names:
            found = list(root.rglob(name))
            if found:
                # Prefer a lead0024_VZ aggregate when several products exist.
                found.sort(key=lambda p: ("lead0024_VZ" not in str(p), len(str(p))))
                return found[0]
    return None


def read_aggregate(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        out = {k: np.asarray(d[k]) for k in d.files}
    for required in ("altitude_km",):
        if required not in out:
            raise KeyError(f"{path} lacks {required}")
    return out


def aggregate_iop_ddh(iop: int, exp: str) -> Path | None:
    """Aggregate validated per-day DDHB files into the normal NPZ schema."""
    root = IOP_ROOT / "analysis" / "ddh" / f"iop{iop}" / exp
    days = dates_for(iop)
    rows = {"QV": {"condcv": [], "condrs": []},
            "QL": {"VQLM": []}, "QR": {"VQRM": []}}
    used = []
    for d in days:
        dd = root / f"DDH{d:%Y%m%d}"
        names = {"QV": ("QV.DHFDLABOF+0024.condcv.dta", "QV.DHFDLABOF+0024.condrs.dta"),
                 "QL": ("QL.DHFDLABOF+0024.VQLM.dta",),
                 "QR": ("QR.DHFDLABOF+0024.VQRM.dta",)}
        if not all((dd / n).is_file() for ns in names.values() for n in ns):
            continue
        for var, ns in names.items():
            for n, block in zip(ns, rows[var]):
                arr = np.loadtxt(dd / n)
                if var == "QV" and not used:
                    z = arr[:, 0].astype(float)
                rows[var][block].append(arr[:, 1].astype(float))
        used.append(d.isoformat())
    # Final products must cover the exact inclusive IOP period.  Partial
    # aggregates are useful only for debugging and must never feed figures.
    if len(used) != len(days):
        return None
    outdir = IOP_ROOT / "analysis" / "ddh" / f"iop{iop}" / exp / "lead0024_VZ"
    outdir.mkdir(parents=True, exist_ok=True)
    for var, blocks in rows.items():
        payload = {"altitude_km": z, "days": np.asarray(used)}
        payload.update({f"block__{k}": np.nanmean(v, axis=0) for k, v in blocks.items()})
        path = outdir / f"{exp}_{var}.npz"; np.savez(path, **payload)
    return outdir / f"{exp}_QV.npz"


def load_budget(iop: int | None, exp: str) -> dict:
    qv = aggregate_path(iop, exp, "QV")
    ql = aggregate_path(iop, exp, "QL")
    qr = aggregate_path(iop, exp, "QR")
    if not qv or not ql or not qr:
        missing = [v for v, p in (("QV", qv), ("QL", ql), ("QR", qr)) if p is None]
        raise FileNotFoundError(f"missing {iop=} {exp=} aggregate(s): {missing}")
    qvd, qld, qrd = read_aggregate(qv), read_aggregate(ql), read_aggregate(qr)
    z = qvd["altitude_km"].astype(float)
    # Aggregators use block__ names; tolerate the raw block names as well.
    def block(d, name):
        return np.asarray(d.get("block__" + name, d.get(name)), float)
    conv = np.maximum(-block(qvd, "condcv"), 0.0)
    resolved = np.maximum(-block(qvd, "condrs"), 0.0)
    ql_v = block(qld, "VQLM")
    qr_v = block(qrd, "VQRM")
    n = min(z.size, conv.size, resolved.size, ql_v.size, qr_v.size)
    return {"z": z[:n], "conv": conv[:n], "resolved": resolved[:n],
            "total": conv[:n] + resolved[:n], "warm": ql_v[:n] + qr_v[:n],
            "sources": [str(qv), str(ql), str(qr)]}


def find_daily_cache(iop: int | None, exp: str, d: date) -> Path | None:
    stamp = d.strftime("%Y%m%d")
    # run_iop_rainfall_updraft.py uses these source keys.
    keys = ((f"iop{iop}_{exp}", f"iop{iop}_{exp}_untar-output", exp)
            if iop is not None else (f"ALARO_{exp}", exp))
    for key in keys:
        p = RAIN_CACHE / key / f"{stamp}.npz"
        if p.exists():
            return p
    hits = list(RAIN_CACHE.rglob(f"{stamp}.npz"))
    hits = [p for p in hits if (f"iop{iop}" in str(p) if iop is not None else f"ALARO_{exp}" in str(p)) and exp in str(p)]
    return hits[0] if hits else None


def mean_updraft(iop: int | None, exp: str, days: list[date]):
    paths = [find_daily_cache(iop, exp, d) for d in days]
    paths = [p for p in paths if p]
    if not paths:
        return None
    rows = [dict(np.load(p, allow_pickle=False)) for p in paths]
    z = np.nanmean([r["height"] for r in rows], axis=0) / 1000.0
    # Rain/updraft cache stores hour x level profiles; closure figure uses
    # the matched date/time mean profile.
    flux = np.nanmean([r["flux"] for r in rows], axis=(0, 1))
    temp = np.nanmean([r["temp"] for r in rows], axis=(0, 1))
    return z, flux, temp, paths


def freeze(z, t):
    order = np.argsort(z); z, t = np.asarray(z)[order], np.asarray(t)[order]
    x = t - 273.15
    ix = np.where(np.sign(x[:-1]) != np.sign(x[1:]))[0]
    if not ix.size: return np.nan
    i = int(ix[0]); return float(z[i] + x[i] / (x[i] - x[i + 1]) * (z[i + 1] - z[i]))


def column(z, v, top=3.0):
    m = np.isfinite(z) & np.isfinite(v); z, v = np.asarray(z)[m], np.asarray(v)[m]
    o = np.argsort(z); z, v = z[o], v[o]
    if z.size < 2 or z[0] > 0 or z[-1] < top: return np.nan
    zz = np.r_[0., z[(z > 0) & (z < top)], top]
    return float(np.trapezoid(np.interp(zz, z, v), zz))


def draw(iop, exp, days, cont, warm, updraft, cont_updraft=None):
    if updraft is None or cont_updraft is None:
        raise FileNotFoundError("paired PF updraft flux/temperature caches are required")
    z = cont["z"]; iz = np.interp(z, warm["z"], warm["total"])
    total = warm["total"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 8), sharey=True, constrained_layout=True)
    panels = [(cont, f"continuous ({LABEL[exp]})", COLORS["continuous"], cont_updraft),
              (warm, f"IOP ({LABEL[exp]})", COLORS["IOP"], updraft)]
    delta = {"z": warm["z"], **{k: warm[k] - cont[k] for k in ("conv", "resolved", "total")}}
    twin_axes = []
    for j, (d, title, color, profile) in enumerate(panels):
        ax = axes[j]
        ax.plot(d["total"], d["z"], color="black", lw=2, label="total condensation")
        ax.plot(d["conv"], d["z"], color="#d62728", ls="-.", label="convective")
        ax.plot(d["resolved"], d["z"], color="#1f77b4", ls="--", label="resolved")
        ax.set_title(title); ax.set_xlabel("condensation (g kg$^{-1}$ day$^{-1}$)")
        ax.set_ylim(0, 10); ax.grid(alpha=.25)
        if j == 0: ax.set_ylabel("altitude (km)")
        if profile is not None:
            uz, uf, ut, _ = profile
            ax2 = ax.twiny(); ax2.plot(uf, uz, color="#009e73", lw=1.5)
            ax2.set_xlabel("updraft flux (kg m$^{-2}$ s$^{-1}$)", color="#009e73")
            ax2.tick_params(axis="x", colors="#009e73")
            twin_axes.append(ax2)
        z0 = freeze(profile[0], profile[2])
        if np.isfinite(z0): ax.axhline(z0, color="#555", ls=":", lw=1)
        zpk = d["z"][np.nanargmax(d["total"])]
        ax.axhspan(0, zpk, color="#BFC7D5", alpha=.18, zorder=0)
        ax.axhline(zpk, color="#555", ls=(0, (5, 3)), lw=1)
    cond_max = max(np.nanmax(cont["total"]), np.nanmax(warm["total"])) * 1.05
    flux_max = max(np.nanmax(cont_updraft[1]), np.nanmax(updraft[1])) * 1.05
    for ax in axes: ax.set_xlim(0, cond_max)
    for ax in twin_axes: ax.set_xlim(0, flux_max)
    handles = [Line2D([], [], color="black", lw=2, label="total condensation"),
               Line2D([], [], color="#d62728", ls="-.", label="convective part"),
               Line2D([], [], color="#1f77b4", ls="--", label="resolved part"),
               Line2D([], [], color="#009e73", label="updraft mass flux"),
               Line2D([], [], color="#555", ls=(0, (5, 3)), label="condensation peak"),
               Line2D([], [], color="#555", ls=":", label="0°C isotherm")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    outdir = OUT / f"iop{iop}"; outdir.mkdir(parents=True, exist_ok=True)
    stem = f"closure_budget_{exp}_vs_continuous_{exp}"
    png = outdir / f"{stem}.png"; fig.savefig(png, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    txt = outdir / f"{stem}.txt"
    with txt.open("w") as f:
        f.write(f"IOP {iop}, {exp} ({LABEL[exp]}) versus continuous {exp}\n")
        f.write(f"Dates: {days[0]} through {days[-1]} inclusive; n={len(days)}; lead=0024\n")
        f.write("Condensation: g kg^-1 day^-1; altitude: km; warm reservoir: g kg^-1; updraft flux: kg m^-2 s^-1\n")
        f.write("Temperature/freezing level uses matched full-grid PF cache temperature profile.\n")
        f.write("Continuous sources:\n" + "\n".join(cont["sources"]) + "\nIOP sources:\n" + "\n".join(warm["sources"]) + "\n")
        f.write("\nlabel,altitude_km,convective,resolved,total,warm_reservoir\n")
        for lab, d in (("continuous", cont), ("IOP", warm), ("delta", delta)):
            for vals in zip(d["z"], d["conv"], d["resolved"], d["total"], d.get("warm", np.full_like(d["z"], np.nan))):
                f.write(lab + "," + ",".join(f"{float(x):.8g}" for x in vals) + "\n")
        f.write(f"\ncontinuous_column_gkgday_km,{column(cont['z'],cont['total']):.8g}\nIOP_column_gkgday_km,{column(warm['z'],warm['total']):.8g}\n")
    return png, txt


def run(args):
    for iop in ((args.iop,) if args.iop else (1, 2)):
        days = dates_for(iop)
        for exp in EXPS:
            while True:
                try:
                    # Each IOP is compared with the same continuous
                    # experiment.  NO3M has no continuous DDH product.
                    if exp == "NO3M":
                        print(f"skip iop={iop} exp=NO3M: no continuous NO3M DDH product")
                        break
                    cont = load_budget(None, exp); warm = load_budget(iop, exp)
                    break
                except (FileNotFoundError, KeyError) as e:
                    if not args.wait: raise
                    print(f"waiting for iop={iop} exp={exp}: {e}", flush=True); time.sleep(args.poll)
            if exp == "NO3M":
                continue
            up = mean_updraft(iop, exp, days)
            cup = mean_updraft(None, exp, days)
            # Temperature is diagnostic metadata; profiles still plot if the
            # updraft reduction is temporarily unavailable.
            paths = draw(iop, exp, days, cont, warm, up, cup)
            print("wrote", *paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iop", type=int, choices=(1, 2)); ap.add_argument("--wait", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="aggregate validated per-day DDHB files")
    ap.add_argument("--poll", type=int, default=60); args = ap.parse_args()
    if args.aggregate:
        for i in ((args.iop,) if args.iop else (1, 2)):
            for e in EXPS: print("aggregate", i, e, aggregate_iop_ddh(i, e))
    else: run(args)


if __name__ == "__main__": main()
