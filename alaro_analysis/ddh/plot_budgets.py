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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import (
    EXP_COLORS, EXPERIMENTS, FIG_DIR, T_FREEZE_K, Z_MAX_KM,
    draw_freeze_lines, freezing_level_km, load_budget, load_temperature,
    set_altitude_axis, tick_formatter,
)
from .plot_style import process_color


CONDENSATION_PROFILE_SUBDIR = "condensation_profile"
CONDENSATION_PROFILE_FILENAME = "1_condensation_profile.png"
EVAP_SUBLIM_SUBDIR = "evap_sublim_by_species"
EVAP_SUBLIM_FILENAME = "2_evap_sublim_by_species.png"
CONDENSE_VS_EVAP_SUBDIR = "condense_vs_evap"
CONDENSE_VS_EVAP_FILENAME = "3_condense_vs_evap.png"
PRECIP_FLUX_SUBDIR = "precip_flux_by_species"
PRECIP_FLUX_FILENAME = "4_precip_flux_by_species.png"
SPECIES_STORAGE_SUBDIR = "species_storage_profile"
SPECIES_STORAGE_FILENAME = "7_species_storage_profile.png"
DATA_TXT_SUBDIR = "data_txt"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
PANEL_TITLE_FONTSIZE = 16
PANEL_LABEL_FONTSIZE = 14


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


def species_storage_profile(exp_cache: dict, var: str) -> np.ndarray | None:
    """Return the mean mixing-ratio profile used by the storage plot."""
    mean_key = {"QL": "VQLM", "QI": "VQIM", "QR": "VQRM",
                "QS": "VQSM", "QG": "VQGM"}
    v = exp_cache.get(var)
    if v is None:
        return None
    key = mean_key[var]
    if key not in v["blocks"]:
        if var == "QI" and "VQNM" in v["blocks"]:
            key = "VQNM"
        else:
            return None
    return v["blocks"][key]


def _add_panel_label(ax, index: int) -> None:
    label = PANEL_LABELS[index] if index < len(PANEL_LABELS) else f"({index + 1})"
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.6,
            "boxstyle": "round,pad=0.18",
        },
        zorder=10,
    )


def _format_txt_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _txt_path_for_plot(path: Path) -> Path:
    return path.parent / DATA_TXT_SUBDIR / f"{path.stem}.txt"


def _common_xlim(
    profiles: list[np.ndarray],
    *,
    pad_fraction: float = 0.06,
) -> tuple[float, float]:
    """Return one x-axis range from a list of profile arrays."""
    values = []
    for profile in profiles:
        finite = np.asarray(profile, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    if not values:
        return (-1.0, 1.0)

    combined = np.concatenate(values)
    xmin = min(0.0, float(np.min(combined)))
    xmax = max(0.0, float(np.max(combined)))
    span = xmax - xmin
    if not np.isfinite(span) or span <= 0.0:
        return (-1.0, 1.0)
    pad = span * pad_fraction
    return (xmin - pad, xmax + pad)


def _legend_upper_right(ax) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9)


def common_condense_vs_evap_xlim(
    all_data,
    *,
    pad_fraction: float = 0.06,
) -> tuple[float, float]:
    """Return one x-axis range for all condensation-vs-evap panels."""
    values = []
    for exp in EXPERIMENTS:
        c = condensation_profile(all_data[exp])
        e = evap_sublim_profile(all_data[exp])
        for profile in (c.get("cond_total"), e.get("evap_sublim_all")):
            if profile is None:
                continue
            finite = np.asarray(profile, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                values.append(finite)
    return _common_xlim(values, pad_fraction=pad_fraction)


def common_total_condensation_xlim(all_data) -> tuple[float, float]:
    values = []
    for exp in EXPERIMENTS:
        c = condensation_profile(all_data[exp])
        for key in ("cond_total", "cond_total_cv", "cond_total_rs"):
            if key in c:
                values.append(c[key])
    return _common_xlim(values)


def common_evap_sublim_xlim(all_data) -> tuple[float, float]:
    values = []
    for exp in EXPERIMENTS:
        ev = evap_sublim_profile(all_data[exp])
        for key in ("rain_evap_total", "snow_evap_total", "graupel_evap_total"):
            if key in ev:
                values.append(ev[key])
    return _common_xlim(values)


def common_precip_xlim(all_data) -> tuple[float, float]:
    values = []
    for exp in EXPERIMENTS:
        p = precip_profile(all_data[exp])
        for key in ("rain_prec_total", "snow_prec_total", "graupel_prec_total"):
            if key in p:
                values.append(p[key])
    return _common_xlim(values)


def common_species_storage_xlim(all_data) -> tuple[float, float]:
    values = []
    for exp in EXPERIMENTS:
        for var in ("QL", "QI", "QR", "QS", "QG"):
            profile = species_storage_profile(all_data[exp], var)
            if profile is not None:
                values.append(profile)
    return _common_xlim(values)


def _write_csv_section(
    fh,
    title: str,
    columns: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'-' * len(title)}\n")
    fh.write(",".join(columns) + "\n")
    for row in rows:
        fh.write(",".join(_format_txt_value(value) for value in row) + "\n")
    fh.write("\n")


def _write_profile_header(
    fh,
    title: str,
    path: Path,
    *,
    description: str,
    units: str,
    xlim: tuple[float, float],
) -> None:
    fh.write(f"{title}\n")
    fh.write(f"{'=' * len(title)}\n")
    fh.write(f"Figure: {path}\n")
    fh.write(f"{description}\n")
    fh.write(f"Units: {units}.\n\n")
    fh.write(
        "Shared x-axis limits: "
        f"{_format_txt_value(xlim[0])}, {_format_txt_value(xlim[1])}.\n\n"
    )


def _write_freezing_section(fh, temps) -> None:
    _write_csv_section(
        fh,
        "Freezing-level data",
        ("experiment", "label", "z_freeze_km"),
        [
            (exp, label, freezing_level_km(temps.get(exp, {})))
            for exp, label in EXPERIMENTS.items()
        ],
    )


def write_total_condensation_txt(all_data, temps, path: Path, xlim: tuple[float, float]) -> Path:
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("Total condensation", "cond_total"),
        ("Convective condensation", "cond_total_cv"),
        ("Stratiform condensation", "cond_total_rs"),
    )
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_profile_header(
            fh,
            f"DDH Condensation Profile Plot Data: {path.stem}",
            path,
            description="Variables: total, convective, and stratiform condensation profiles.",
            units="g kg-1 day-1",
            xlim=xlim,
        )
        _write_freezing_section(fh, temps)
        for panel_title, key in panels:
            for exp, label in EXPERIMENTS.items():
                c = condensation_profile(all_data[exp])
                if key not in c:
                    continue
                z = all_data[exp]["QL"]["altitude_km"]
                n_days = all_data[exp]["QL"].get("n_days", "")
                rows = [(z[i], c[key][i], n_days) for i in range(z.size)]
                _write_csv_section(
                    fh,
                    f"{panel_title} - {label} profile data",
                    ("altitude_km", "rate_gkgday", "n_days"),
                    rows,
                )
    print(f"  txt: {txt_path}")
    return txt_path


def write_evap_sublim_txt(all_data, temps, path: Path, xlim: tuple[float, float]) -> Path:
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("Rain evaporation", "rain_evap_total"),
        ("Snow sublimation", "snow_evap_total"),
        ("Graupel sublimation", "graupel_evap_total"),
    )
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_profile_header(
            fh,
            f"DDH Evaporation/Sublimation Plot Data: {path.stem}",
            path,
            description="Variables: falling-precipitation evaporation and sublimation profiles by species.",
            units="g kg-1 day-1",
            xlim=xlim,
        )
        _write_freezing_section(fh, temps)
        for panel_title, key in panels:
            for exp, label in EXPERIMENTS.items():
                ev = evap_sublim_profile(all_data[exp])
                if key not in ev:
                    continue
                z = all_data[exp]["QR"]["altitude_km"]
                n_days = all_data[exp]["QR"].get("n_days", "")
                rows = [(z[i], ev[key][i], n_days) for i in range(z.size)]
                _write_csv_section(
                    fh,
                    f"{panel_title} - {label} profile data",
                    ("altitude_km", "rate_gkgday", "n_days"),
                    rows,
                )
    print(f"  txt: {txt_path}")
    return txt_path


def write_precip_flux_txt(all_data, temps, path: Path, xlim: tuple[float, float]) -> Path:
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("Rain flux", "rain_prec_total"),
        ("Snow flux", "snow_prec_total"),
        ("Graupel flux", "graupel_prec_total"),
    )
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_profile_header(
            fh,
            f"DDH Precipitation Flux Plot Data: {path.stem}",
            path,
            description="Variables: per-species precipitation flux-rate profiles.",
            units="g kg-1 day-1",
            xlim=xlim,
        )
        _write_freezing_section(fh, temps)
        for panel_title, key in panels:
            for exp, label in EXPERIMENTS.items():
                p = precip_profile(all_data[exp])
                if key not in p:
                    continue
                z = all_data[exp]["QR"]["altitude_km"]
                n_days = all_data[exp]["QR"].get("n_days", "")
                rows = [(z[i], p[key][i], n_days) for i in range(z.size)]
                _write_csv_section(
                    fh,
                    f"{panel_title} - {label} profile data",
                    ("altitude_km", "rate_gkgday", "n_days"),
                    rows,
                )
    print(f"  txt: {txt_path}")
    return txt_path


def write_species_storage_txt(all_data, temps, path: Path, xlim: tuple[float, float]) -> Path:
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("QL", "Cloud liquid"),
        ("QI", "Cloud ice"),
        ("QR", "Rain"),
        ("QS", "Snow"),
        ("QG", "Graupel"),
    )
    with txt_path.open("w", encoding="utf-8") as fh:
        _write_profile_header(
            fh,
            f"DDH Species Storage Plot Data: {path.stem}",
            path,
            description="Variables: time-mean hydrometeor mixing-ratio profiles by species.",
            units="g kg-1",
            xlim=xlim,
        )
        _write_freezing_section(fh, temps)
        for var, panel_title in panels:
            for exp, label in EXPERIMENTS.items():
                profile = species_storage_profile(all_data[exp], var)
                if profile is None:
                    continue
                z = all_data[exp][var]["altitude_km"]
                n_days = all_data[exp][var].get("n_days", "")
                rows = [(z[i], profile[i], n_days) for i in range(z.size)]
                _write_csv_section(
                    fh,
                    f"{panel_title} - {label} profile data",
                    ("altitude_km", "mixing_ratio_gkg", "n_days"),
                    rows,
                )
    print(f"  txt: {txt_path}")
    return txt_path


def write_condense_vs_evap_txt(all_data, temps, path: Path) -> Path:
    """Write the exact profile data used by the condensation-vs-evap figure."""
    txt_path = _txt_path_for_plot(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"DDH Condensation vs Evaporation Plot Data: {path.stem}"
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {path}\n")
        fh.write("Variables: condensation source and evaporation/sublimation back to vapour.\n")
        fh.write("Units: g kg-1 day-1.\n\n")
        xlim = common_condense_vs_evap_xlim(all_data)
        fh.write(f"Shared x-axis limits: {_format_txt_value(xlim[0])}, {_format_txt_value(xlim[1])}.\n\n")

        _write_csv_section(
            fh,
            "Freezing-level data",
            ("experiment", "label", "z_freeze_km"),
            [
                (exp, label, freezing_level_km(temps.get(exp, {})))
                for exp, label in EXPERIMENTS.items()
            ],
        )

        for exp, label in EXPERIMENTS.items():
            z = all_data[exp]["QL"]["altitude_km"]
            c = condensation_profile(all_data[exp])
            e = evap_sublim_profile(all_data[exp])
            cond = c.get("cond_total", np.full_like(z, np.nan, dtype=np.float64))
            evap = e.get("evap_sublim_all", np.full_like(z, np.nan, dtype=np.float64))
            n_days = all_data[exp]["QL"].get("n_days", "")
            rows = [
                (
                    z[i],
                    cond[i],
                    evap[i],
                    n_days,
                )
                for i in range(z.size)
            ]
            _write_csv_section(
                fh,
                f"{label} profile data",
                (
                    "altitude_km",
                    "condensation_source_gkgday",
                    "evap_sublim_back_to_vapour_gkgday",
                    "n_days",
                ),
                rows,
            )
    print(f"  txt: {txt_path}")
    return txt_path


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_total_condensation(all_data, temps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_total_condensation_xlim(all_data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    titles = ("Total condensation", "Convective condensation", "Stratiform condensation")
    keys = ("cond_total", "cond_total_cv", "cond_total_rs")
    for index, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        for exp, label in EXPERIMENTS.items():
            c = condensation_profile(all_data[exp])
            if key not in c:
                continue
            z = all_data[exp]["QL"]["altitude_km"]
            ax.plot(c[key], z, color=EXP_COLORS[exp], lw=2, label=label)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(title, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        _add_panel_label(ax, index)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        _legend_upper_right(ax)
    set_altitude_axis(axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")
    write_total_condensation_txt(all_data, temps, path, xlim)


def plot_evap_sublim_by_species(all_data, temps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_evap_sublim_xlim(all_data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    for index, (ax, (sp, sp_label)) in enumerate(zip(axes, (
        ("rain", "Rain evaporation"),
        ("snow", "Snow sublimation"),
        ("graupel", "Graupel sublimation"),
    ))):
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
        ax.set_title(sp_label, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        _add_panel_label(ax, index)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        _legend_upper_right(ax)
    set_altitude_axis(axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")
    write_evap_sublim_txt(all_data, temps, path, xlim)


def plot_condense_vs_evap(all_data, temps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_condense_vs_evap_xlim(all_data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    for index, (ax, (exp, label)) in enumerate(zip(axes, EXPERIMENTS.items())):
        z = all_data[exp]["QL"]["altitude_km"]
        c = condensation_profile(all_data[exp])
        e = evap_sublim_profile(all_data[exp])
        if "cond_total" in c:
            ax.plot(c["cond_total"], z, color=process_color("Condensation"), lw=2,
                    label="Condensation (source)")
        if "evap_sublim_all" in e:
            ax.plot(e["evap_sublim_all"], z, color=process_color("Evaporation"), lw=2,
                    label="Evap + sublim (back to vapour)")
        z0 = freezing_level_km(temps.get(exp, {}))
        if np.isfinite(z0):
            ax.axhline(z0, color="k", lw=1.0, ls="--", alpha=0.85,
                       label=r"0 $^{\circ}$C isotherm")
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(label, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        _add_panel_label(ax, index)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        ax.legend(loc="upper right", fontsize=9)
    set_altitude_axis(axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")
    write_condense_vs_evap_txt(all_data, temps, path)


def plot_precip_per_species(all_data, temps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_precip_xlim(all_data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    for index, (ax, (sp, title)) in enumerate(zip(axes, (
        ("rain", "Rain flux"),
        ("snow", "Snow flux"),
        ("graupel", "Graupel flux"),
    ))):
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
        ax.set_title(title, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        _add_panel_label(ax, index)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Rate (g kg$^{-1}$ day$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        _legend_upper_right(ax)
    set_altitude_axis(axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")
    write_precip_flux_txt(all_data, temps, path, xlim)


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
    bars1 = ax1.bar(
        x - w/2,
        cond_vals,
        w,
        label="Condensation",
        color=process_color("Condensation"),
    )
    bars2 = ax1.bar(
        x + w/2,
        evap_vals,
        w,
        label="Evap + sublim",
        color=process_color("Evaporation"),
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    xlim = common_species_storage_xlim(all_data)
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharex=True, sharey=True)
    species = (("QL", "Cloud liquid"), ("QI", "Cloud ice"),
               ("QR", "Rain"), ("QS", "Snow"), ("QG", "Graupel"))
    for index, (ax, (var, title)) in enumerate(zip(axes, species)):
        for exp, lbl in EXPERIMENTS.items():
            profile = species_storage_profile(all_data[exp], var)
            if profile is None:
                continue
            ax.plot(profile, all_data[exp][var]["altitude_km"],
                    color=EXP_COLORS[exp], lw=2, label=lbl)
        draw_freeze_lines(ax, temps)
        ax.axvline(0, color="k", lw=0.6, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.set_title(title, color="black", fontsize=PANEL_TITLE_FONTSIZE)
        _add_panel_label(ax, index)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"Mean mixing ratio (g kg$^{-1}$)")
        ax.xaxis.set_major_formatter(tick_formatter())
        _legend_upper_right(ax)
    set_altitude_axis(axes[0])
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")
    write_species_storage_txt(all_data, temps, path, xlim)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(lead: str = "0024", plot: str = "all") -> list:
    """Produce the main budget figures.  Returns output paths in order."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_data, temps = {}, {}
    for exp in EXPERIMENTS:
        all_data[exp] = {}
        for var in ("QL", "QI", "QR", "QS", "QG", "QV", "UU", "VV"):
            d = load_budget(exp, var, lead=lead)
            if d is not None:
                all_data[exp][var] = d
        temps[exp] = load_temperature(exp)
        z0 = freezing_level_km(temps[exp])
        print(f"  {exp:<8} 0 C isotherm: {z0:.2f} km"
              if np.isfinite(z0) else f"  {exp:<8} no T data")

    plan = [
        (
            "condensation-profile",
            Path(CONDENSATION_PROFILE_SUBDIR) / CONDENSATION_PROFILE_FILENAME,
            plot_total_condensation,
        ),
        (
            "evap-sublim-by-species",
            Path(EVAP_SUBLIM_SUBDIR) / EVAP_SUBLIM_FILENAME,
            plot_evap_sublim_by_species,
        ),
        (
            "condense-vs-evap",
            Path(CONDENSE_VS_EVAP_SUBDIR) / CONDENSE_VS_EVAP_FILENAME,
            plot_condense_vs_evap,
        ),
        (
            "precip-flux-by-species",
            Path(PRECIP_FLUX_SUBDIR) / PRECIP_FLUX_FILENAME,
            plot_precip_per_species,
        ),
        ("column-budget-bars", "5_column_budget_bars.png", plot_column_budget_bars),
        (
            "evap-over-condensation-ratio",
            "6_evap_over_condensation_ratio.png",
            plot_precip_efficiency,
        ),
        (
            "species-storage-profile",
            Path(SPECIES_STORAGE_SUBDIR) / SPECIES_STORAGE_FILENAME,
            plot_species_storage,
        ),
    ]
    if plot == "profile-set":
        selected = {
            "condensation-profile",
            "evap-sublim-by-species",
            "precip-flux-by-species",
            "species-storage-profile",
        }
        plan = [item for item in plan if item[0] in selected]
    elif plot != "all":
        plan = [item for item in plan if item[0] == plot]
        if not plan:
            raise ValueError(f"unknown plot selection: {plot}")

    out_paths = []
    for _key, name, fn in plan:
        path = FIG_DIR / name
        fn(all_data, temps, path)
        out_paths.append(path)
    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lead", default="0024")
    parser.add_argument(
        "--plot",
        choices=(
            "all",
            "profile-set",
            "condensation-profile",
            "evap-sublim-by-species",
            "condense-vs-evap",
            "precip-flux-by-species",
            "species-storage-profile",
            "column-budget-bars",
            "evap-over-condensation-ratio",
        ),
        default="all",
        help="Select one plot to regenerate, or all plots.",
    )
    args = parser.parse_args()
    run(lead=args.lead, plot=args.plot)


if __name__ == "__main__":
    main()
