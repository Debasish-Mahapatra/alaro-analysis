"""SUP budget suite for the closure experiments (15-day, 1-15 Mar 2014).

Six runs compared in every panel: baseline C1M/G1M, plus the NIMELIT=1 and
LNEBCV=.F. perturbations of each.  Colour = microphysics config (C1M red,
G1M blue); line style = switch (baseline solid, NIMELIT dashed, LNEBCV dotted).

Figures (SUP/budgets/):
  budget_condensation.png     total / convective / resolved condensation
  budget_evap_sublim.png      rain evap, snow sublim, graupel sublim
  budget_precip_flux.png      rain / snow / graupel precipitation flux
  budget_species_storage.png  QL, QI, QR, QS, QG mean mixing ratios

The CT (total-condensate, 3MT-vs-microphysics) budget is built separately by
make_sup_ct_budget.py because it needs the 2-ice/3-ice CT FBL split.

DDH budgets are the model's domain average (not re-masked); 0 C isotherm from
the DDH temperature (VCT0/VPP0/cp), per experiment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import epygram
from matplotlib.lines import Line2D

from alaro_analysis.ddh.io import AGG_DIR, CP_DRY, UNTAR_ROOT, load_budget
from alaro_analysis.ddh.plot_budgets import (
    condensation_profile, evap_sublim_profile, precip_profile,
    species_storage_profile,
)

OUT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/SUP/budgets")
DPI = 450
Z_MAX = 15.0
DDH_DAYS = [f"DDH201403{d:02d}" for d in range(1, 16)]
C1M, G1M = "#d62728", "#1f77b4"

# (load_key, temp_exp, label, colour, linestyle)
EXPS = [
    ("control_base15",    "control",           "C1M baseline",  C1M, "-"),
    ("graupel_base15",    "graupel",           "G1M baseline",  G1M, "-"),
    ("control_NIMELIT_1", "control_NIMELIT_1", "C1M NIMELIT 1", C1M, "--"),
    ("graupel_NIMELIT_1", "graupel_NIMELIT_1", "G1M NIMELIT 1", G1M, "--"),
    ("control_LNEBCV_F",  "control_LNEBCV_F",  "C1M LNEBCV .F.", C1M, ":"),
    ("graupel_LNEBCV_F",  "graupel_LNEBCV_F",  "G1M LNEBCV .F.", G1M, ":"),
]


def gfmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def mean_temperature_profile(exp):
    """Day-mean DDH T(z) [K] over the 15 days (T = VCT0/VPP0/cp from LFA)."""
    root = UNTAR_ROOT / exp / "output"
    acc = cnt = None
    for d in DDH_DAYS:
        f = root / d / "DHFDLABOF+0024"
        if not f.exists():
            continue
        try:
            r = epygram.formats.resource(str(f), "r", fmt="LFA")
            flds = r.listfields()
            if "VCT0" not in flds or "VPP0" not in flds:
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
        return None
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def freezing_km(alt_km, t_k):
    if t_k is None:
        return np.nan
    z = np.asarray(alt_km, float); t = np.asarray(t_k, float)
    n = min(z.size, t.size); z, t = z[:n], t[:n]
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    if z.size < 2:
        return np.nan
    o = np.argsort(z); z, t = z[o], t[o]
    d = t - 273.15
    cr = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if cr.size == 0:
        return np.nan
    i = int(cr[0]); w = d[i] / (d[i] - d[i + 1])
    return float(z[i] + w * (z[i + 1] - z[i]))


def load_all():
    data = {}
    for key, temp_exp, label, color, ls in EXPS:
        cache = {}
        for var in ("QV", "QL", "QI", "QR", "QS", "QG"):
            b = load_budget(key, var, lead="0024")
            if b is not None:
                cache[var] = b
        t = mean_temperature_profile(temp_exp)
        z = cache["QL"]["altitude_km"]
        data[key] = {"cache": cache, "label": label, "color": color, "ls": ls,
                     "z": z, "z0": freezing_km(z, t)}
    return data


def _draw_freeze(ax, data):
    for key, *_ in EXPS:
        d = data[key]
        if np.isfinite(d["z0"]):
            ax.axhline(d["z0"], color=d["color"], ls=d["ls"], lw=0.8, alpha=0.5, zorder=1)


def _xlim(arrays):
    vals = np.concatenate([np.asarray(a, float)[np.isfinite(a)] for a in arrays if a is not None and np.isfinite(a).any()])
    lo = min(0.0, float(vals.min())); hi = max(0.0, float(vals.max()))
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _finish(fig, axes, data, xlim):
    for ax in np.atleast_1d(axes).ravel():
        ax.axvline(0, color="k", lw=0.6, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, Z_MAX)
        ax.set_xlabel(r"rate (g kg$^{-1}$ day$^{-1}$)")
        gfmt(ax)
    np.atleast_1d(axes).ravel()[0].set_ylabel("altitude (km)")
    handles = [Line2D([], [], color=c, ls=s, lw=2.0, label=l) for _, _, l, c, s in EXPS]
    handles.append(Line2D([], [], color="0.4", ls="-", lw=0.8, alpha=0.6, label="0 $^\\circ$C isotherm"))
    np.atleast_1d(axes).ravel()[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()


def fig_three(keyfn, titles, panel_keys, png, suptitle, storage=False):
    fig, axes = plt.subplots(1, len(titles), figsize=(5.0 * len(titles), 6.2),
                             sharex=True, sharey=True)
    series = {k: keyfn(data[k]["cache"]) for k, *_ in EXPS} if not storage else None
    all_vals = []
    for ax, title, pk in zip(axes, titles, panel_keys):
        for key, *_ in EXPS:
            d = data[key]
            if storage:
                prof = species_storage_profile(d["cache"], pk)
                z = d["cache"][pk]["altitude_km"] if pk in d["cache"] else None
            else:
                prof = series[key].get(pk)
                z = d["z"]
            if prof is None or z is None:
                continue
            ax.plot(prof, z, color=d["color"], ls=d["ls"], lw=2.0)
            all_vals.append(prof)
        _draw_freeze(ax, data)
        ax.set_title(title)
    xlim = _xlim(all_vals)
    if storage:
        for ax in axes:
            ax.set_xlabel(r"mixing ratio (g kg$^{-1}$)")
    _finish(fig, axes, data, xlim)
    if storage:
        for ax in axes:
            ax.set_xlabel(r"mixing ratio (g kg$^{-1}$)")
    fig.suptitle(suptitle, fontsize=14, y=1.00)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / png
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main():
    global data
    data = load_all()
    for k, *_ , in EXPS:
        print(f"  {k}: 0C={data[k]['z0']:.2f} km  vars={sorted(data[k]['cache'])}")
    fig_three(condensation_profile,
              ("total condensation", "convective condensation", "resolved condensation"),
              ("cond_total", "cond_total_cv", "cond_total_rs"),
              "budget_condensation.png", "Condensation budget")
    fig_three(evap_sublim_profile,
              ("rain evaporation", "snow sublimation", "graupel sublimation"),
              ("rain_evap_total", "snow_evap_total", "graupel_evap_total"),
              "budget_evap_sublim.png", "Evaporation / sublimation budget")
    fig_three(precip_profile,
              ("rain flux", "snow flux", "graupel flux"),
              ("rain_prec_total", "snow_prec_total", "graupel_prec_total"),
              "budget_precip_flux.png", "Precipitation-flux budget")
    # NB: the DDH "VQxM" storage term swings sign day-to-day (it is a signed
    # storage tendency, not a mixing ratio), so a "species storage" mixing-ratio
    # panel would be mislabeled - intentionally omitted from this budget suite.


if __name__ == "__main__":
    main()
