"""SUP figures: water-vapour negativity correction (QV.neg) profiles.

The DDH QV budget carries a "negativity correction" block (the model's
negative-humidity fixer), a water-vapour sink in g kg-1 day-1.  These figures
show its 15-day-mean (1-15 Mar 2014) vertical profile for C1M (control) and G1M
(graupel), comparing the baseline against each single-switch perturbation:

  qv_neg_correction_NIMELIT.png : baseline (NIMELIT = 2) vs NIMELIT = 1
  qv_neg_correction_LNEBCV.png  : baseline (LNEBCV = .T.) vs LNEBCV = .F.

Perturbation profiles come from the aggregated npz (block__neg, n_days=15);
the baseline 15-day mean is aggregated here from the per-day QV.neg .dta files
(the base15 aggregates kept only condcv/condrs, so neg is rebuilt directly).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

AGG = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/lead0024_VZ")
PROC = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead0024_VZ")
OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/qv_negativity_correction")
DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]   # 1-15 Mar 2014

DPI = 450
Z_MAX = 10.0
X_MIN, X_MAX = -0.28, 0.01           # fixed so both figures compare 1:1
BASE_COLOR = "0.45"
PERT_COLOR = "#1f77b4"

FIGURES = [
    ("qv_neg_correction_NIMELIT", [
        ("C1M", "control", "control_NIMELIT_1", "NIMELIT = 2", "NIMELIT = 1"),
        ("G1M", "graupel", "graupel_NIMELIT_1", "NIMELIT = 2", "NIMELIT = 1"),
    ]),
    ("qv_neg_correction_LNEBCV", [
        ("C1M", "control", "control_LNEBCV_F", "LNEBCV = .T.", "LNEBCV = .F."),
        ("G1M", "graupel", "graupel_LNEBCV_F", "LNEBCV = .T.", "LNEBCV = .F."),
    ]),
]


def read_dta(p):
    a = np.loadtxt(p)
    return a[:, 0].astype(float), a[:, 1].astype(float)


def baseline_neg(exp):
    """15-day-mean QV.neg profile, aggregated from the per-day .dta files."""
    s = c = zs = zc = None
    for d in DAYS:
        f = PROC / exp / d / "QV" / "QV.DHFDLABOF+0024.neg.dta"
        if not f.exists():
            continue
        z, v = read_dta(f)
        if s is None:
            s, c = np.zeros_like(v), np.zeros_like(v)
            zs, zc = np.zeros_like(z), np.zeros_like(z)
        m = np.isfinite(v); s[m] += v[m]; c[m] += 1
        zm = np.isfinite(z); zs[zm] += z[zm]; zc[zm] += 1
    z = np.where(zc > 0, zs / np.maximum(zc, 1), np.nan)
    v = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return z, v


def pert_neg(exp):
    """15-day-mean QV.neg profile from the aggregated npz."""
    d = np.load(AGG / f"{exp}_QV.npz", allow_pickle=True)
    return np.asarray(d["altitude_km"], float), np.asarray(d["block__neg"], float)


def clip_sort(z, v):
    m = np.isfinite(z) & np.isfinite(v) & (z >= 0) & (z <= Z_MAX)
    z, v = z[m], v[m]
    o = np.argsort(z)
    return z[o], v[o]


def gfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def make_fig(png, panels):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True)
    cols = {}
    for ax, (cfg, base, pert, blab, plab) in zip(axes, panels):
        zb, vb = clip_sort(*baseline_neg(base))
        zp, vp = clip_sort(*pert_neg(pert))
        ax.plot(vb, zb, color=BASE_COLOR, lw=2.4, label=blab)
        ax.plot(vp, zp, color=PERT_COLOR, lw=2.4, label=plab)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, Z_MAX)
        ax.set_title(cfg)
        ax.set_xlabel("QV negativity correction (g kg$^{-1}$ day$^{-1}$)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)
        gfmt(ax)
        cols[f"{cfg}_{blab}_z_km"] = zb
        cols[f"{cfg}_{blab}_g_kg_day"] = vb
        cols[f"{cfg}_{plab}_z_km"] = zp
        cols[f"{cfg}_{plab}_g_kg_day"] = vp
    axes[0].set_ylabel("altitude (km)")
    fig.suptitle("Water-vapour negativity correction", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{png}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    write_data(cols, f"{png}.txt")


def write_data(cols, name):
    (OUT / "data_txt").mkdir(parents=True, exist_ok=True)
    txt = OUT / "data_txt" / name
    keys = list(cols.keys())
    n = max((len(cols[k]) for k in keys), default=0)
    with open(txt, "w") as f:
        f.write("# QV negativity correction (g/kg/day) vs altitude (km); "
                "15-day mean 1-15 Mar 2014; C1M=control, G1M=graupel.\n")
        f.write("\t".join(keys) + "\n")
        for i in range(n):
            f.write("\t".join(f"{cols[k][i]:.6g}" if i < len(cols[k]) else ""
                              for k in keys) + "\n")
    print(f"wrote {txt}")


def main():
    for png, panels in FIGURES:
        make_fig(png, panels)


if __name__ == "__main__":
    main()
