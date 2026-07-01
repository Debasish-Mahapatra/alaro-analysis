"""SUP closure-budget figures for the LNEBCV and NIMELIT perturbation runs.

A DDH-only version of the main "closure budget compression" figure (Plot 7),
for the two perturbation pairs, control vs graupel only (no G2M):

  * LNEBCV : control_LNEBCV_F  vs graupel_LNEBCV_F  (LNEBCV .T.->.F.)
  * NIMELIT: control_NIMELIT_1 vs graupel_NIMELIT_1 (NIMELIT 2->1)

Everything is derived from the DDH budgets of these 15-day (1-15 Mar 2014) runs:
condensation partition (QV.condcv / QV.condrs), warm reservoir (QL.VQLM+QR.VQRM)
and the 0 C isotherm (T = VCT0/VPP0/cp from the DDH LFA).  The FA-derived updraft
mass flux of the original figure is intentionally omitted (it is not DDH data).

Outputs one folder per switch under microphysics-paper/SUP/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import epygram

from alaro_analysis.ddh.io import AGG_DIR, CP_DRY, UNTAR_ROOT
from alaro_analysis.ddh.plot_condensation_partition import (
    column_integrate,
    load_partition,
)

DPI = 450
SUP_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP")
Z_MAX = 10.0
FLUX_XMAX = 1.2e-2
COND_XMAX = 3.0          # fixed condensation x-axis (g/kg/day) so all closure figures compare 1:1

HEIGHT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/geopotential/2years")
UPDRAFT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data/updraft_flux/radar")

TOTAL_COLOR = "#111111"
CONVECTIVE_COLOR = "#D62728"
RESOLVED_COLOR = "#1F77B4"
UPDRAFT_COLOR = "#009E73"
PEAK_COLOR = "#4D4D4D"
FREEZING_COLOR = "#555555"

# control = red, graupel = blue (EXPERIMENT_COLORS), used for the summary bars
BAR_COLOR = {"C1M": "#d62728", "G1M": "#1f77b4"}
PANEL_LABELS = ("(a)", "(b)", "(c)")

DDH_DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]   # 1-15 Mar 2014


def _pert(exp, label):
    """Perturbation panel: its own 15-day aggregation / untar / updraft cache."""
    return {"label": label, "part_exp": exp, "untar_exp": exp,
            "days": None, "updraft_key": exp}


def _base(base, label):
    """Baseline panel: original run restricted to the same 15 days."""
    return {"label": label, "part_exp": f"{base}_base15", "untar_exp": base,
            "days": DDH_DAYS, "updraft_key": base}


def _lbl(spec, label):
    """Copy a panel spec with a different panel title."""
    spec = dict(spec)
    spec["label"] = label
    return spec


FIGURES = [
    {"folder": "closure_budget_compression_LNEBCV",
     "png": "closure_budget_compression_LNEBCV", "subtitle": "LNEBCV = .F.",
     "panels": [_pert("control_LNEBCV_F", "C1M"), _pert("graupel_LNEBCV_F", "G1M")]},
    {"folder": "closure_budget_compression_LNEBCV",
     "png": "closure_budget_compression_LNEBCV_baseline", "subtitle": "LNEBCV = .T.",
     "panels": [_base("control", "C1M"), _base("graupel", "G1M")]},
    {"folder": "closure_budget_compression_NIMELIT",
     "png": "closure_budget_compression_NIMELIT", "subtitle": "NIMELIT = 1",
     "panels": [_pert("control_NIMELIT_1", "C1M"), _pert("graupel_NIMELIT_1", "G1M")]},
    {"folder": "closure_budget_compression_NIMELIT",
     "png": "closure_budget_compression_NIMELIT_baseline", "subtitle": "NIMELIT = 2",
     "panels": [_base("control", "C1M"), _base("graupel", "G1M")]},

    # per-experiment comparisons: the two configs of one experiment, side by side
    {"folder": "closure_budget_compression_LNEBCV",
     "png": "closure_budget_compression_C1M_LNEBCV", "subtitle": "C1M",
     "panels": [_lbl(_base("control", "C1M"), "LNEBCV = .T."),
                _lbl(_pert("control_LNEBCV_F", "C1M"), "LNEBCV = .F.")]},
    {"folder": "closure_budget_compression_LNEBCV",
     "png": "closure_budget_compression_G1M_LNEBCV", "subtitle": "G1M",
     "panels": [_lbl(_base("graupel", "G1M"), "LNEBCV = .T."),
                _lbl(_pert("graupel_LNEBCV_F", "G1M"), "LNEBCV = .F.")]},
    {"folder": "closure_budget_compression_NIMELIT",
     "png": "closure_budget_compression_C1M_NIMELIT", "subtitle": "C1M",
     "panels": [_lbl(_pert("control_NIMELIT_1", "C1M"), "NIMELIT = 1"),
                _lbl(_base("control", "C1M"), "NIMELIT = 2")]},
    {"folder": "closure_budget_compression_NIMELIT",
     "png": "closure_budget_compression_G1M_NIMELIT", "subtitle": "G1M",
     "panels": [_lbl(_pert("graupel_NIMELIT_1", "G1M"), "NIMELIT = 1"),
                _lbl(_base("graupel", "G1M"), "NIMELIT = 2")]},
]


def freezing_level_km(altitude_km, temperature_k):
    """Lowest 0 C crossing of a temperature profile."""
    z = np.asarray(altitude_km, float)
    t = np.asarray(temperature_k, float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    if z.size < 2:
        return np.nan
    order = np.argsort(z)
    z, t = z[order], t[order]
    d = t - 273.15
    cross = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if cross.size == 0:
        return np.nan
    i = int(cross[0])
    w = d[i] / (d[i] - d[i + 1])
    return float(z[i] + w * (z[i + 1] - z[i]))


def layer_integral(altitude_km, values, z_bottom=0.0, z_top=3.0):
    """Integrate a profile between two heights, inserting boundary values."""
    z = np.asarray(altitude_km, float)
    v = np.asarray(values, float)
    ok = np.isfinite(z) & np.isfinite(v)
    z, v = z[ok], v[ok]
    if z.size < 2:
        return np.nan
    order = np.argsort(z)
    z, v = z[order], v[order]
    if z_top <= z_bottom or z_top < z[0] or z_bottom > z[-1]:
        return np.nan
    inside = (z > z_bottom) & (z < z_top)
    z_l = np.r_[z_bottom, z[inside], z_top]
    v_l = np.interp(z_l, z, v)
    return float(np.trapezoid(v_l, z_l))


def warm_reservoir_0_3(exp):
    """0-3 km column of QL(VQLM)+QR(VQRM) liquid+rain content (g/kg/km)."""
    ql = np.load(AGG_DIR / "lead0024_VZ" / f"{exp}_QL.npz", allow_pickle=True)
    qr = np.load(AGG_DIR / "lead0024_VZ" / f"{exp}_QR.npz", allow_pickle=True)
    z = np.asarray(ql["altitude_km"], float)
    vqlm = np.asarray(ql["block__VQLM"], float)
    vqrm = np.asarray(qr["block__VQRM"], float)
    return layer_integral(z, vqlm + vqrm, 0.0, 3.0)


def mean_temperature_profile(exp, day_names=None):
    """Day-mean T(z) [K] from the DDH LFA files (T = VCT0/VPP0/cp).

    day_names restricts to those DDH<date> dirs (used for the baseline runs,
    whose untarred DDH spans two years); None uses every DDH20* dir present.
    """
    root = UNTAR_ROOT / exp / "output"
    if day_names is not None:
        days = [root / d for d in day_names if (root / d).is_dir()]
    else:
        days = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("DDH20"))
    acc, cnt = None, None
    for d in days:
        f = d / "DHFDLABOF+0024"
        if not f.exists():
            continue
        try:
            r = epygram.formats.resource(str(f), "r", fmt="LFA")
            flds = r.listfields()
            if "VCT0" not in flds or "VPP0" not in flds:
                r.close()
                continue
            vct = np.asarray(r.readfield("VCT0").getdata(), float).ravel()
            vpp = np.asarray(r.readfield("VPP0").getdata(), float).ravel()
            r.close()
        except Exception:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = vct / np.where(vpp > 0, vpp, np.nan) / CP_DRY
        if acc is None:
            acc = np.zeros_like(t)
            cnt = np.zeros_like(t)
        m = np.isfinite(t)
        acc[m] += t[m]
        cnt[m] += 1
    if acc is None:
        return None
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def load_updraft(exp):
    """Domain/time-mean updraft mass-flux profile (kg/m2/s) vs height (km)."""
    base = "control" if exp.startswith("control") else "graupel"
    h = np.asarray(np.load(HEIGHT_DIR / f"{base}_full-domain_height_profile_first.npz",
                           allow_pickle=True)["height_m"], float) / 1000.0
    fx = np.asarray(np.load(UPDRAFT_DIR / f"{exp}_updraft_radar.npz",
                            allow_pickle=True)["mean_flux"], float)
    n = min(h.size, fx.size)
    h, fx = h[:n], fx[:n]
    order = np.argsort(h)
    return h[order], fx[order]


def gather(spec):
    """All quantities needed for one panel (perturbation or baseline)."""
    part = load_partition(spec["part_exp"], lead="0024")
    z = part.altitude_km
    total = part.total_gkgday
    finite = np.isfinite(z) & np.isfinite(total)
    z_peak = float(z[finite][np.argmax(total[finite])]) if finite.any() else np.nan
    temp = mean_temperature_profile(spec["untar_exp"], spec.get("days"))
    z_freeze = freezing_level_km(z, temp) if temp is not None else np.nan
    col_total = column_integrate(total, z)
    u_h, u_flux = load_updraft(spec["updraft_key"])
    return {
        "altitude_km": z,
        "convective": part.convective_gkgday,
        "resolved": part.resolved_gkgday,
        "total": total,
        "z_peak": z_peak,
        "z_freeze": z_freeze,
        "col_total": col_total,
        "compression": col_total / max(z_peak, 1e-6),
        "u_h": u_h,
        "u_flux": u_flux,
    }


def panel_label(ax, text):
    ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=14, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      boxstyle="round,pad=0.18"), zorder=20)


def make_figure(cfg):
    data = {p["label"]: gather(p) for p in cfg["panels"]}
    labels = [p["label"] for p in cfg["panels"]]

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16})
    fig = plt.figure(figsize=(10.5, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 0.30],
                          hspace=0.45, wspace=0.16,
                          left=0.09, right=0.96, top=0.80, bottom=0.10)
    fig.suptitle(cfg["subtitle"], fontsize=16, y=0.965)

    for i, lab in enumerate(labels):
        d = data[lab]
        ax = fig.add_subplot(gs[0, i])
        ax.axhspan(0, d["z_peak"], color="#BFC7D5", alpha=0.18, zorder=0)
        ax.plot(d["total"], d["altitude_km"], color=TOTAL_COLOR, lw=2.8,
                label="total condensation")
        ax.plot(d["convective"], d["altitude_km"], color=CONVECTIVE_COLOR, lw=2.0,
                ls="-.", alpha=0.9, label="convection-scheme part")
        ax.plot(d["resolved"], d["altitude_km"], color=RESOLVED_COLOR, lw=2.0,
                ls="--", alpha=0.95, label="resolved-microphysics part")
        ax.axhline(d["z_peak"], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), alpha=0.9)
        ax.axhline(d["z_freeze"], color=FREEZING_COLOR, lw=1.2, ls=":", alpha=0.95)

        ax2 = ax.twiny()
        ax2.plot(d["u_flux"], d["u_h"], color=UPDRAFT_COLOR, lw=2.1, alpha=0.95)
        ax2.set_xlim(0, FLUX_XMAX)
        ax2.set_xlabel(r"updraft mass flux (kg m$^{-2}$ s$^{-1}$)",
                       color=UPDRAFT_COLOR, labelpad=8)
        ax2.tick_params(axis="x", colors=UPDRAFT_COLOR)
        ax2.spines["top"].set_color(UPDRAFT_COLOR)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        ax.set_xlim(0, COND_XMAX)
        ax.set_ylim(0, Z_MAX)
        ax.set_title(lab, pad=32)
        ax.set_xlabel(r"condensation (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if i == 0:
            ax.set_ylabel("altitude (km)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(alpha=0.25)
        panel_label(ax, PANEL_LABELS[i])

    style_handles = [
        Line2D([0], [0], color=TOTAL_COLOR, lw=2.8, label="total condensation"),
        Line2D([0], [0], color=CONVECTIVE_COLOR, lw=2.0, ls="-.",
               label="convection-scheme part"),
        Line2D([0], [0], color=RESOLVED_COLOR, lw=2.0, ls="--",
               label="resolved-microphysics part"),
        Line2D([0], [0], color=UPDRAFT_COLOR, lw=2.1, label="updraft mass flux"),
        Line2D([0], [0], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)),
               label="condensation peak height"),
        Line2D([0], [0], color=FREEZING_COLOR, lw=1.2, ls=":",
               label=r"0 $^{\circ}$C isotherm"),
    ]
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=style_handles, loc="center", ncol=3, fontsize=11,
                  frameon=False, handlelength=2.6, columnspacing=1.8)

    out_dir = SUP_ROOT / cfg["folder"]
    (out_dir / "data_txt").mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{cfg['png']}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")

    txt = out_dir / "data_txt" / f"{cfg['png']}.txt"
    with open(txt, "w") as f:
        f.write(f"SUP closure-budget figure: {cfg['subtitle']}\n")
        f.write("DDH condensation + FA updraft mass flux; 1-15 Mar 2014; lead 0024.\n\n")
        f.write("label,experiment,column_total_gkgday_km,peak_height_km,"
                "compression_gkgday,freezing_level_km\n")
        for p in cfg["panels"]:
            d = data[p["label"]]
            f.write(f"{p['label']},{p['part_exp']},{d['col_total']:.6f},{d['z_peak']:.6f},"
                    f"{d['compression']:.6f},{d['z_freeze']:.6f}\n")
        f.write("\nlabel,altitude_km,convective_gkgday,resolved_gkgday,total_gkgday\n")
        for p in cfg["panels"]:
            d = data[p["label"]]
            for zz, cc, rr, tt in zip(d["altitude_km"], d["convective"],
                                      d["resolved"], d["total"]):
                f.write(f"{p['label']},{zz:.6f},{cc:.6e},{rr:.6e},{tt:.6e}\n")
        f.write("\nlabel,height_km,updraft_mass_flux_kg_m2_s\n")
        for p in cfg["panels"]:
            d = data[p["label"]]
            for hh, ff in zip(d["u_h"], d["u_flux"]):
                f.write(f"{p['label']},{hh:.6f},{ff:.6e}\n")
    print(f"wrote {txt}")


def main():
    epygram.init_env()
    for cfg in FIGURES:
        make_figure(cfg)


if __name__ == "__main__":
    main()
