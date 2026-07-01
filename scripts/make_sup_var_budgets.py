"""SUP per-variable budget figures (one figure per species, per config).

Layout requested: one figure per budget variable (QV, QL, QI, QR, QS, QG), and
within it 2x2 subplots pairing each perturbation with the baseline:

    [ baseline      | NIMELIT = 1   ]
    [ baseline      | LNEBCV = .F.  ]

Each subplot is the FULL budget of that variable for that one run: every process
term (condensation, evaporation, autoconversion, precip flux, turbulence,
dynamics, negativity correction) as a vertical profile.  C1M and G1M are separate
figures.  15-day mean (1-15 Mar 2014).

Run: python make_sup_var_budgets.py [VAR ...]   (default: all species present)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.ddh.io import BLOCK_COLORS, load_budget, pretty_block_label
from alaro_analysis.ddh.plot_style import (
    DEFAULT_COLOR, pathway_from_block, pathway_line_attributes,
)

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/budgets_by_variable")
TZERO_NPZ = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/ddh_tzero_km.npz")
DPI = 450
Z_MAX = 20.0

# config -> (label, {experiment-role: (load_key, temp_exp)})
CONFIGS = {
    "C1M": {
        "baseline":   ("control_base15",    "control"),
        "NIMELIT = 1":("control_NIMELIT_1", "control_NIMELIT_1"),
        "LNEBCV = .F.":("control_LNEBCV_F", "control_LNEBCV_F"),
    },
    "G1M": {
        "baseline":   ("graupel_base15",    "graupel"),
        "NIMELIT = 1":("graupel_NIMELIT_1", "graupel_NIMELIT_1"),
        "LNEBCV = .F.":("graupel_LNEBCV_F", "graupel_LNEBCV_F"),
    },
}
VAR_TITLE = {"QV": "water vapour", "QL": "cloud liquid", "QI": "cloud ice",
             "QR": "rain", "QS": "snow", "QG": "graupel"}

# 0 C level per experiment, precomputed once up front: all the epygram LFA reads
# MUST finish before any matplotlib rendering (interleaving the two C libraries
# corrupts the heap -> SIGABRT).
FREEZE: dict[str, float] = {}

# Repo colour convention (alaro_analysis.ddh.plot_style / io.BLOCK_COLORS):
# one bright colour per physical process, linestyle separating convective (-.)
# from resolved (--).  TERM_ORDER fixes the legend order.
TERM_ORDER = ["cond-cv", "cond-rs", "evap-cv", "evap-rs", "auto-cv", "auto-rs",
              "prec-cv", "prec-rs", "turconv", "turdiff", "dynam", "neg"]
# QV writes blocks without the hyphen; map them onto the canonical names.
CANON = {"condcv": "cond-cv", "condrs": "cond-rs", "evapcv": "evap-cv", "evaprs": "evap-rs"}
TERM_LW = 1.9


def gfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def terms_present(blocks):
    """Return [(label, colour, linestyle, alpha, profile)] in the repo convention."""
    canon_blocks = {CANON.get(k, k): v for k, v in blocks.items()}
    out = []
    for name in TERM_ORDER:
        if name not in canon_blocks:
            continue
        color = BLOCK_COLORS.get(name, DEFAULT_COLOR)
        ls, alpha = pathway_line_attributes(pathway_from_block(name))
        out.append((pretty_block_label(name), color, ls, alpha, canon_blocks[name]))
    return out


def common_xlim(profiles):
    vals = [np.asarray(p, float)[np.isfinite(p)] for p in profiles]
    vals = [v for v in vals if v.size]
    if not vals:
        return (-1.0, 1.0)
    c = np.concatenate(vals)
    lo = min(0.0, float(c.min())); hi = max(0.0, float(c.max()))
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


COLS = ["baseline", "NIMELIT = 1", "LNEBCV = .F."]


def make_figure(var):
    """One figure per variable: rows = config (C1M, G1M), cols = experiment.

    Each subplot shows every tendency term of the variable's budget.
    """
    configs = [c for c in ("C1M", "G1M") if not (var == "QG" and c == "C1M")]
    grid = {}
    all_profiles = []
    for i, config in enumerate(configs):
        for j, exp in enumerate(COLS):
            key, temp_exp = CONFIGS[config][exp]
            b = load_budget(key, var, lead="0024")
            if b is None:
                grid[(i, j)] = None
                continue
            present = terms_present(b["blocks"])
            grid[(i, j)] = (present, b["altitude_km"], FREEZE.get(temp_exp, np.nan))
            all_profiles += [t[-1] for t in present]
    if not all_profiles:
        return
    xlim = common_xlim(all_profiles)

    nrows = len(configs)
    fig, axes = plt.subplots(nrows, 3, figsize=(15.0, 4.7 * nrows + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    labels_seen = {}
    for i, config in enumerate(configs):
        for j, exp in enumerate(COLS):
            ax = axes[i, j]
            item = grid[(i, j)]
            if item is None:
                ax.set_visible(False); continue
            present, z, z0 = item
            for label, color, ls, alpha, prof in present:
                h, = ax.plot(prof, z, color=color, ls=ls, lw=TERM_LW, alpha=alpha, label=label)
                labels_seen[label] = h
            if np.isfinite(z0):
                ax.axhline(z0, color="0.3", ls=":", lw=1.0, alpha=0.8)
            ax.axvline(0, color="k", lw=0.6, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.set_title(f"{config} {var} — {exp}")
            ax.set_xlim(*xlim)
            ax.set_ylim(0, Z_MAX)
            gfmt(ax)
    for i in range(nrows):
        axes[i, 0].set_ylabel("altitude (km)")
    for j in range(3):
        axes[nrows - 1, j].set_xlabel(r"rate (g kg$^{-1}$ day$^{-1}$)")
    handles = list(labels_seen.values()) + [
        plt.Line2D([], [], color="0.3", ls=":", lw=1.0, label="0 $^\\circ$C isotherm")]
    labels = list(labels_seen.keys()) + ["0 $^\\circ$C isotherm"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{var} budget ({VAR_TITLE.get(var, var)})", fontsize=15, y=0.997)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"budget_{var}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def load_freezing():
    """Read the precomputed 0 C levels (built by make_sup_ddh_tzero.py).

    Kept out of this process on purpose: the epygram LFA reader can corrupt the
    heap when used alongside matplotlib, so freezing levels are cached separately.
    """
    if not TZERO_NPZ.exists():
        raise SystemExit(f"missing {TZERO_NPZ}; run make_sup_ddh_tzero.py first")
    d = np.load(TZERO_NPZ)
    for k in d.files:
        FREEZE[k] = float(d[k])
        print(f"  0C {k}: {FREEZE[k]:.2f} km", flush=True)


def main():
    vars_req = sys.argv[1:] or ["QV", "QL", "QI", "QR", "QS", "QG"]
    load_freezing()
    for var in vars_req:
        make_figure(var)


if __name__ == "__main__":
    main()
