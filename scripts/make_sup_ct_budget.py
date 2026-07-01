"""SUP CT (total-condensate) scheme budget for the closure experiments.

The CT budget splits the heating into the 3MT convection scheme (micro-cv) and
the resolved microphysics scheme (micro-rs).  CRITICAL: the CT budget-list (FBL)
differs by configuration - C1M (control, 2-ice) uses CT3.fbl-2ice, G1M (graupel,
3-ice) uses CT.fbl-3ice - because the graupel terms only exist in the 3-ice
list.  This script forces the correct FBL per run, extracts CT via ddhb for the
15 days, aggregates, and plots the 6 runs.

Input +0024 DDH: baseline from DDH-0024-only/<cfg>/output/<day>_DHFDLABOF+0024;
perturbations from DDH-untar/<exp>/output/<day>/DHFDLABOF+0024.

Output: SUP/budgets/budget_ct_scheme.png  (2 panels: convection 3MT, microphysics)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

RUNS = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
TOOLBOX = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ddhtoolbox")
RT_BASE = RUNS / "DDH-processed/lead0024_VZ/_runtime"
DDH0024 = RUNS / "DDH-0024-only"
UNTAR = RUNS / "DDH-untar"
CACHE = RUNS / "DDH-processed/_sup_ct"
OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/budgets_by_variable")
TZERO_NPZ = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/ddh_tzero_km.npz")
DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]
COMPONENTS = ["micro-cv", "micro-rs"]
COMP_TITLE = {"micro-cv": "convection scheme (3MT)", "micro-rs": "microphysics scheme"}
COMP_COLOR = {"micro-cv": "#d95f02", "micro-rs": "#7570b3"}
CONFIGS_CT = {
    "C1M": {"baseline": "control_base", "NIMELIT = 1": "control_NIMELIT_1", "LNEBCV = .F.": "control_LNEBCV_F"},
    "G1M": {"baseline": "graupel_base", "NIMELIT = 1": "graupel_NIMELIT_1", "LNEBCV = .F.": "graupel_LNEBCV_F"},
}
LEAD = "0024"
Z_MAX = 20.0
DPI = 450

FBL = {"control": TOOLBOX / "ddh_budget_lists/alaro/CT3.fbl-2ice",
       "graupel": TOOLBOX / "ddh_budget_lists/alaro/CT.fbl-3ice"}
C1M, G1M = "#d62728", "#1f77b4"

# id, runtime type (2ice=control / 3ice=graupel), source kind, source exp, label, colour, ls
EXPS = [
    ("control_base",      "control", "base", "control",           "C1M baseline",   C1M, "-"),
    ("graupel_base",      "graupel", "base", "graupel",           "G1M baseline",   G1M, "-"),
    ("control_NIMELIT_1", "control", "pert", "control_NIMELIT_1", "C1M NIMELIT 1",  C1M, "--"),
    ("graupel_NIMELIT_1", "graupel", "pert", "graupel_NIMELIT_1", "G1M NIMELIT 1",  G1M, "--"),
    ("control_LNEBCV_F",  "control", "pert", "control_LNEBCV_F",  "C1M LNEBCV .F.", C1M, ":"),
    ("graupel_LNEBCV_F",  "graupel", "pert", "graupel_LNEBCV_F",  "G1M LNEBCV .F.", G1M, ":"),
]


def prepare_runtimes():
    """Copy each base runtime and overwrite alaro/CT.fbl with the correct FBL."""
    for rt in ("control", "graupel"):
        dst = CACHE / "_runtime" / rt / "ddh_budget_lists"
        if not dst.exists():
            shutil.copytree(RT_BASE / rt / "ddh_budget_lists", dst)
        shutil.copy2(FBL[rt], dst / "alaro" / "CT.fbl")


def input_path(kind, src, day):
    if kind == "base":
        return DDH0024 / src / "output" / f"{day}_DHFDLABOF+{LEAD}"
    return UNTAR / src / "output" / day / f"DHFDLABOF+{LEAD}"


def extract(task):
    exp_id, rt, kind, src, day = task
    out_dir = CACHE / exp_id / day
    done = out_dir / "done.ok"
    if done.exists() and all((out_dir / f"CT.DHFDLABOF+{LEAD}.{c}.dta").exists() for c in COMPONENTS):
        return f"SKIP {exp_id} {day}"
    src_file = input_path(kind, src, day)
    if not src_file.exists():
        return f"MISS {exp_id} {day} ({src_file})"
    out_dir.mkdir(parents=True, exist_ok=True)
    bps = CACHE / "_runtime" / rt / "ddh_budget_lists"
    env = os.environ.copy()
    env["DDHTOOLBOX"] = str(TOOLBOX)
    env["DDHB_BPS"] = str(bps)
    env["DDHI_LIST"] = str(bps / "alaro" / "conversion_list")
    env.pop("DDH_PLOT", None)
    env["PATH"] = f"{TOOLBOX/'tools'}:{TOOLBOX/'tools'/'lfa'}:{TOOLBOX/'tools'/'.dd2gr'/'src'}:{env.get('PATH','')}"
    with tempfile.TemporaryDirectory(prefix="ct_in_") as idir, \
         tempfile.TemporaryDirectory(prefix="ct_wk_") as wdir:
        link = Path(idir) / f"DHFDLABOF+{LEAD}"
        os.symlink(str(src_file), str(link))
        cmd = ["ddhb", "-v", "alaro/CT", "-i", f"DHFDLABOF+{LEAD}", "-Y", "VZ", "-r", str(wdir)]
        p = subprocess.run(cmd, cwd=idir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            return f"FAIL {exp_id} {day} rc={p.returncode}"
        bud = Path(wdir) / "budget.alaro"
        miss = [c for c in COMPONENTS if not (bud / f"CT.DHFDLABOF+{LEAD}.{c}.dta").exists()]
        if miss:
            return f"NODTA {exp_id} {day} missing={miss}"
        for c in COMPONENTS:
            shutil.copy2(bud / f"CT.DHFDLABOF+{LEAD}.{c}.dta", out_dir / f"CT.DHFDLABOF+{LEAD}.{c}.dta")
        done.write_text("ok\n")
    return f"OK {exp_id} {day}"


def read_dta(p):
    a = np.loadtxt(p)
    return a[:, 0].astype(float), a[:, 1].astype(float)


def aggregate(exp_id):
    out = {}
    for c in COMPONENTS:
        zs = zc = vs = vc = None
        for day in DAYS:
            f = CACHE / exp_id / day / f"CT.DHFDLABOF+{LEAD}.{c}.dta"
            if not f.exists():
                continue
            z, v = read_dta(f)
            if zs is None:
                zs = np.zeros_like(z); zc = np.zeros_like(z); vs = np.zeros_like(v); vc = np.zeros_like(v)
            mz = np.isfinite(z); zs[mz] += z[mz]; zc[mz] += 1
            mv = np.isfinite(v); vs[mv] += v[mv]; vc[mv] += 1
        if zs is None:
            out[c] = (None, None)
        else:
            out[c] = (np.where(zc > 0, zs / np.maximum(zc, 1), np.nan),
                      np.where(vc > 0, vs / np.maximum(vc, 1), np.nan))
    return out


def gfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def main():
    prepare_runtimes()
    tasks = [(e[0], e[1], e[2], e[3], d) for e in EXPS for d in DAYS]
    print(f"extracting CT: {len(tasks)} (exp,day) tasks", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(extract, t): t for t in tasks}
        ok = 0
        for fu in as_completed(futs):
            r = fu.result()
            if not r.startswith(("OK", "SKIP")):
                print(r, flush=True)
            else:
                ok += 1
        print(f"extracted/cached {ok}/{len(tasks)}", flush=True)

    tzf = np.load(TZERO_NPZ)
    TZ = {k: float(tzf[k]) for k in tzf.files}

    def tz_for(eid):
        return TZ.get({"control_base": "control", "graupel_base": "graupel"}.get(eid, eid), np.nan)

    agg = {e[0]: aggregate(e[0]) for e in EXPS}
    OUT.mkdir(parents=True, exist_ok=True)

    configs = ["C1M", "G1M"]
    cols = ["baseline", "NIMELIT = 1", "LNEBCV = .F."]
    allv = []
    for config in configs:
        for exp in cols:
            eid = CONFIGS_CT[config][exp]
            for c in COMPONENTS:
                z, v = agg[eid][c]
                if z is not None:
                    m = (z >= 0) & (z <= Z_MAX); allv.append(v[m])
    vals = np.concatenate([a[np.isfinite(a)] for a in allv]) if allv else np.array([0.0, 1.0])
    lo = min(0.0, float(vals.min())); hi = max(0.0, float(vals.max()))
    pad = 0.06 * (hi - lo) if hi > lo else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.8), sharex=True, sharey=True, squeeze=False)
    for i, config in enumerate(configs):
        for j, exp in enumerate(cols):
            ax = axes[i, j]
            eid = CONFIGS_CT[config][exp]
            for c in COMPONENTS:
                z, v = agg[eid][c]
                if z is None:
                    continue
                m = (z >= 0) & (z <= Z_MAX)
                ax.plot(v[m], z[m], color=COMP_COLOR[c], lw=2.1, label=COMP_TITLE[c])
            z0 = tz_for(eid)
            if np.isfinite(z0):
                ax.axhline(z0, color="0.3", ls=":", lw=1.0, alpha=0.8)
            ax.axvline(0, color="k", lw=0.6, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.set_title(f"{config} CT — {exp}")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(0, Z_MAX)
            gfmt(ax)
    for i in range(2):
        axes[i, 0].set_ylabel("altitude (km)")
    for j in range(3):
        axes[1, j].set_xlabel("heating rate (K day$^{-1}$)")
    handles = [Line2D([], [], color=COMP_COLOR[c], lw=2.1, label=COMP_TITLE[c]) for c in COMPONENTS]
    handles.append(Line2D([], [], color="0.3", ls=":", lw=1.0, label="0 $^\\circ$C isotherm"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("CT budget (total condensate)", fontsize=15, y=0.997)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    p = OUT / "budget_CT.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
