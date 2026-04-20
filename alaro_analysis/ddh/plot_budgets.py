"""Plot microphysics budget profiles from aggregated DDH output.

Uses altitude (km) as the vertical axis (0-20 km) and marks the annual-mean
0 C isotherm as a dashed horizontal line per experiment.

Inputs:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/
      lead0024_VZ/{exp}_{var}.npz          time-mean budgets on altitude axis
      temperature_{exp}.npz                annual-mean temperature + altitude

Outputs:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/DDH-figures/
      1_condensation_profile.png
      2_evap_sublim_by_species.png
      3_condense_vs_evap.png
      4_precip_flux_by_species.png
      5_column_budget_bars.png
      6_evap_over_condensation_ratio.png
      7_species_storage_profile.png

Usage:
  conda activate epygram
  python -m alaro_analysis.ddh.plot_budgets [--lead 0024]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .io import (
    EXP_COLORS, EXPERIMENTS, FIG_DIR, T_FREEZE_K, Z_MAX_KM,
    draw_freeze_lines, freezing_level_km, load_budget, load_temperature,
    set_altitude_axis, tick_formatter,
)


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

def condensation_profile(exp_cache: dict) -> dict[str, np.ndarray]:
    out = {}
    ql = exp_cache.get("QL")
    qi = exp_cache.get("QI")
    if ql is None or qi is None:
        return out
    for name, blocks, var in (
        ("cond_cv_liquid", ("cond-cv",), ql),
        ("cond_rs_liquid", ("cond-rs",), ql),
        ("cond_cv_ice",    ("cond-cv",), qi),
        ("cond_rs_ice",    ("cond-rs",), qi),
    ):
        b = var["blocks"]
        if all(block in b for block in blocks):
            out[name] = sum(b[x] for x in blocks)
    if "cond_cv_liquid" in out and "cond_cv_ice" in out:
        out["cond_total_cv"] = out["cond_cv_liquid"] + out["cond_cv_ice"]
    if "cond_rs_liquid" in out and "cond_rs_ice" in out:
        out["cond_total_rs"] = out["cond_rs_liquid"] + out["cond_rs_ice"]
    if "cond_total_cv" in out and "cond_total_rs" in out:
        out["cond_total"] = out["cond_total_cv"] + out["cond_total_rs"]
    return out


def evap_sublim_profile(exp_cache: dict) -> dict[str, np.ndarray]:
    out = {}
    for sp, var in (("rain", "QR"), ("snow", "QS"), ("graupel", "QG")):
        v = exp_cache.get(var)
        if v is None:
            continue
        b = v["blocks"]
        if "evap-cv" in b and "evap-rs" in b:
            out[f"{sp}_evap_cv"]    = b["evap-cv"]
            out[f"{sp}_evap_rs"]    = b["evap-rs"]
            out[f"{sp}_evap_total"] = b["evap-cv"] + b["evap-rs"]
    tot_keys = [k for k in out if k.endswith("_total")]
    if tot_keys:
        out["evap_sublim_all"] = np.sum([out[k] for k in tot_keys], axis=0)
    return out


def precip_profile(exp_cache: dict) -> dict[str, np.ndarray]:
    out = {}
    for sp, var in (("rain", "QR"), ("snow", "QS"), ("graupel", "QG")):
        v = exp_cache.get(var)
        if v is None:
            continue
        b = v["blocks"]
        if "prec-cv" in b and "prec-rs" in b:
            out[f"{sp}_prec_cv"]    = b["prec-cv"]
            out[f"{sp}_prec_rs"]    = b["prec-rs"]
            out[f"{sp}_prec_total"] = b["prec-cv"] + b["prec-rs"]
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_total_condensation(all_data, temps, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    titles = ("Total condensation", "Convective condensation", "Stratiform condensation")
    keys = ("cond_total", "cond_total_cv", "cond_total_rs")
    for ax, title, key in zip(axes, titles, keys):
        for exp, label in EXPERIMENTS.items():
            c = condensation_profile(all_data[exp])
            if key not in c:
                continue
            z = all_data[exp]["QL"]["altitude_km"]
            ax.plot(c[key], z, color=EXP_COLORS[exp], lw=2, label=label)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
    set_altitude_axis(axes[0])
    axes[0].legend(fontsize=9)
    fig.suptitle("Vertical profile of condensation rate",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def plot_evap_sublim_by_species(all_data, temps, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for ax, (sp, sp_label) in zip(axes, (
        ("rain", "Rain evaporation"),
        ("snow", "Snow sublimation"),
        ("graupel", "Graupel sublimation"),
    )):
        for exp, label in EXPERIMENTS.items():
            ev = evap_sublim_profile(all_data[exp])
            key = f"{sp}_evap_total"
            if key not in ev:
                continue
            z = all_data[exp]["QR"]["altitude_km"]
            ax.plot(ev[key], z, color=EXP_COLORS[exp], lw=2, label=label)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(sp_label)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
    set_altitude_axis(axes[0])
    axes[0].legend(fontsize=9)
    fig.suptitle("Falling-precipitation re-evaporation / sublimation profile",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def plot_condense_vs_evap(all_data, temps, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for ax, (exp, label) in zip(axes, EXPERIMENTS.items()):
        z = all_data[exp]["QL"]["altitude_km"]
        c = condensation_profile(all_data[exp])
        e = evap_sublim_profile(all_data[exp])
        if "cond_total" in c:
            ax.plot(c["cond_total"], z, color="#1f77b4", lw=2,
                    label="Condensation (source)")
        if "evap_sublim_all" in e:
            ax.plot(e["evap_sublim_all"], z, color="#d62728", lw=2,
                    label="Evap + sublim (back to vapour)")
        z0 = freezing_level_km(temps.get(exp, {}))
        if np.isfinite(z0):
            ax.axhline(z0, color="k", lw=1.0, ls="--", alpha=0.85,
                       label=r"0 $^{\circ}$C isotherm")
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(label)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        ax.legend(loc="lower right", fontsize=9)
    set_altitude_axis(axes[0])
    fig.suptitle("Condensation vs re-evaporation / sublimation per level",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def plot_precip_per_species(all_data, temps, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for ax, (sp, title) in zip(axes, (
        ("rain", "Rain flux"),
        ("snow", "Snow flux"),
        ("graupel", "Graupel flux"),
    )):
        for exp, label in EXPERIMENTS.items():
            p = precip_profile(all_data[exp])
            key = f"{sp}_prec_total"
            if key not in p:
                continue
            z = all_data[exp]["QR"]["altitude_km"]
            ax.plot(p[key], z, color=EXP_COLORS[exp], lw=2, label=label)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
    set_altitude_axis(axes[0])
    axes[0].legend(fontsize=9)
    fig.suptitle("Per-species precipitation flux rate per level",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def column_integrate(profile, alt_km):
    """Integrate profile [g/kg/day] over altitude using dz weighting.

    Approximate conversion: assumes density ~ 1 kg/m^3 averaged.  For a
    meaningful scale we use hydrostatic dp/g weighting of a reference
    atmosphere instead -- computed via dp ~ rho*g*dz.  Here we just use a
    layer-midpoint rule with dz in km and return relative units (g/kg day km);
    interpreted as a "column content rate".
    """
    z = np.asarray(alt_km, dtype=np.float64)
    # Ensure ascending
    if z[0] > z[-1]:
        z = z[::-1]
        profile = profile[::-1]
    dz = np.diff(z)
    mid = 0.5 * (profile[:-1] + profile[1:])
    return float(np.nansum(mid * dz))


def plot_column_budget_bars(all_data, temps, path):
    labels, cond_vals, evap_vals, ev_r, ev_s, ev_g = [], [], [], [], [], []
    for exp, lbl in EXPERIMENTS.items():
        z = all_data[exp]["QL"]["altitude_km"]
        c = condensation_profile(all_data[exp])
        e = evap_sublim_profile(all_data[exp])
        labels.append(lbl)
        cond_vals.append(column_integrate(c.get("cond_total", np.zeros_like(z)), z))
        evap_vals.append(column_integrate(e.get("evap_sublim_all", np.zeros_like(z)), z))
        ev_r.append(column_integrate(e.get("rain_evap_total",    np.zeros_like(z)), z))
        ev_s.append(column_integrate(e.get("snow_evap_total",    np.zeros_like(z)), z))
        ev_g.append(column_integrate(e.get("graupel_evap_total", np.zeros_like(z)), z))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, cond_vals, w, label="Condensation", color="#1f77b4")
    bars2 = ax1.bar(x + w/2, evap_vals, w, label="Evap + sublim", color="#d62728")
    for bars, vals in [(bars1, cond_vals), (bars2, evap_vals)]:
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width()/2, v,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"Column rate (g kg$^{-1}$ day$^{-1}$ km)")
    ax1.set_title("Column-integrated condensation and re-evap/sublim")
    ax1.grid(alpha=0.3, axis="y"); ax1.legend()

    w2 = 0.25
    ax2.bar(x - w2, ev_r, w2, label="Rain evap",    color="#1f77b4")
    ax2.bar(x,       ev_s, w2, label="Snow sublim",  color="#17becf")
    ax2.bar(x + w2, ev_g, w2, label="Graupel sublim", color="#9467bd")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"Column rate (g kg$^{-1}$ day$^{-1}$ km)")
    ax2.set_title("Column-integrated re-evap/sublim per species")
    ax2.grid(alpha=0.3, axis="y"); ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def plot_precip_efficiency(all_data, temps, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, ys, labels = [], [], []
    for i, (exp, lbl) in enumerate(EXPERIMENTS.items()):
        z = all_data[exp]["QL"]["altitude_km"]
        c = condensation_profile(all_data[exp])
        e = evap_sublim_profile(all_data[exp])
        if "cond_total" not in c or "evap_sublim_all" not in e:
            continue
        cc = column_integrate(c["cond_total"], z)
        ee = column_integrate(e["evap_sublim_all"], z)
        r = ee / cc if cc > 0 else np.nan
        xs.append(i); ys.append(r); labels.append(lbl)
        ax.bar(i, r, color=EXP_COLORS[exp], width=0.6)
        ax.text(i, r, f"{r:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Evap+sublim / Condensation  (column integrated)")
    ax.set_title("Fraction of condensate re-evaporated / re-sublimated aloft")
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(tick_formatter())
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def plot_species_storage(all_data, temps, path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
    species = (("QL", "Cloud liquid"), ("QI", "Cloud ice"),
               ("QR", "Rain"), ("QS", "Snow"), ("QG", "Graupel"))
    mean_key = {"QL": "VQLM", "QI": "VQIM", "QR": "VQRM",
                "QS": "VQSM", "QG": "VQGM"}
    for ax, (var, title) in zip(axes, species):
        for exp, lbl in EXPERIMENTS.items():
            v = all_data[exp].get(var)
            if v is None:
                continue
            key = mean_key[var]
            if key not in v["blocks"]:
                if var == "QI" and "VQNM" in v["blocks"]:
                    key = "VQNM"
                else:
                    continue
            ax.plot(v["blocks"][key], v["altitude_km"],
                    color=EXP_COLORS[exp], lw=2, label=lbl)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(r"Mean mixing ratio (g kg$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
    set_altitude_axis(axes[0])
    axes[-1].legend(fontsize=9)
    fig.suptitle("Time-mean mixing ratio per species",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lead", default="0024")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_data, temps = {}, {}
    for exp in EXPERIMENTS:
        all_data[exp] = {}
        for var in ("QL", "QI", "QR", "QS", "QG", "QV", "UU", "VV"):
            d = load_budget(exp, var, lead=args.lead)
            if d is not None:
                all_data[exp][var] = d
        temps[exp] = load_temperature(exp)
        z0 = freezing_level_km(temps[exp])
        print(f"  {exp:<8} 0 C isotherm: {z0:.2f} km"
              if np.isfinite(z0) else f"  {exp:<8} no T data")

    plot_total_condensation(all_data, temps, FIG_DIR / "1_condensation_profile.png")
    plot_evap_sublim_by_species(all_data, temps, FIG_DIR / "2_evap_sublim_by_species.png")
    plot_condense_vs_evap(all_data, temps, FIG_DIR / "3_condense_vs_evap.png")
    plot_precip_per_species(all_data, temps, FIG_DIR / "4_precip_flux_by_species.png")
    plot_column_budget_bars(all_data, temps, FIG_DIR / "5_column_budget_bars.png")
    plot_precip_efficiency(all_data, temps, FIG_DIR / "6_evap_over_condensation_ratio.png")
    plot_species_storage(all_data, temps, FIG_DIR / "7_species_storage_profile.png")


if __name__ == "__main__":
    main()
