#!/usr/bin/env python3
"""Warm-rain condensation budget (paper closure figure), rebuilt for all four runs.

One panel per configuration: the DDH total condensation split into its
convection-scheme (3MT) and resolved-microphysics parts, with the upward mass flux
on a twin axis, the height of peak condensation (dashed) and the 0 degC level
(dotted). The summary metrics of the published figure are written to the data text
rather than drawn as bars.

Three changes from the published version:

1. The mass flux is the total upward flux of the column split into 3MT updraft,
   downdraft and environment (``data.convection_sections.column_fluxes``), cached by
   ``build_iop_updraft_cloud_partition``. The published figure used the 3MT updraft
   area times UD_OMEGA, which treats UD_OMEGA as an absolute velocity; it is the
   velocity relative to the updraft's environment (ACCVUD; Gerard et al. 2009, eq. 4).
2. G2M-XCU is included. It runs without 3MT, so its convective condensation is zero
   and its upward flux is the resolved ascent of the single-piece column; the last
   metric, convective condensation per unit flux, is therefore zero for it.
3. The period is the GoAmazon IOPs, not the continuous 2-year run: only C1M still has
   2 years of raw FA on this machine.

Condensation and the warm reservoir come from the DDH aggregates of
``build_iop_closure_budgets`` (QV) and ``build_iop_warm_reservoir`` (QL, QR).
Written beside the published figure under a different name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS, IOP_PERIODS
from alaro_analysis.common.paths import data_path
from alaro_analysis.recipes.iop.make_iop_closure_budget_figures import (
    CONVECTIVE_COLOR,
    FREEZING_COLOR,
    PEAK_COLOR,
    RESOLVED_COLOR,
    TOTAL_COLOR,
    UPDRAFT_COLOR,
    Z_MAX_KM,
    column_integral,
    freezing_level_km,
    load_condensation,
    peak_height_km,
)

OUTPUT_DIR = data_path("microphysics-paper/7.DDH-closure_budget_compression")
FIGURE_STEM = "closure_budget_compression_column-split_iops"
PERIOD_NAMES = {"combined": "iop1+iop2", "iop1": "iop1", "iop2": "iop2"}
# DDH experiment key -> label; the convection caches are keyed by label.
EXPERIMENTS = ("control", "graupel", "2mom", "NO3M")
CACHE_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M", "NO3M": "G2M-XCU"}
RESERVOIR_TOP_KM = 3.0
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")


def load_flux(convection_work: Path, iops: tuple[int, ...], label: str, quantity: str):
    """Count-weighted all-hours mean of a cached partition flux, and its height (km)."""
    total = weight = height = None
    for iop in iops:
        with np.load(convection_work / f"iop{iop}" / f"{label}_partition_{quantity}.npz") as c:
            mean = np.asarray(c["mean"], float)
            counts = np.asarray(c["counts"], float)
        valid = np.isfinite(mean) & (counts > 0)
        contribution = np.where(valid, mean * counts, 0.0)
        counted = np.where(valid, counts, 0.0)
        total = contribution if total is None else total + contribution
        weight = counted if weight is None else weight + counted
        with np.load(convection_work / f"iop{iop}" / f"{label}_partition_height.npz") as c:
            height = np.asarray(c["height_m"], float) / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(weight.sum(axis=1) > 0, total.sum(axis=1) / weight.sum(axis=1), np.nan)
    return height, profile


def load_reservoir(work: Path, iops: tuple[int, ...], experiment: str, key="content_gkg"):
    """Day-weighted mean warm-water profile and its altitude (km).

    ``content_gkg`` is the cloud + rain water content (g/kg). ``tendency`` is the
    ddhb VQLM+VQRM pair the published figure used, which is a mean tendency in
    g/kg/day, not a content; it is kept only for comparison.
    """
    total, days, altitude = None, 0, None
    for iop in iops:
        path = work / "aggregates" / f"iop{iop}" / f"{experiment}_QLQR.npz"
        with np.load(path, allow_pickle=True) as cache:
            n = int(np.asarray(cache["days"]).size)
            altitude = np.asarray(cache["altitude_km"], float)
            if key == "tendency":
                profile = np.asarray(cache["block__VQLM"], float) + np.asarray(
                    cache["block__VQRM"], float
                )
            else:
                profile = np.asarray(cache[key], float)
        total = profile * n if total is None else total + profile * n
        days += n
    return altitude, total / max(days, 1)


def layer_integral(altitude_km, values, z_bottom=0.0, z_top=RESERVOIR_TOP_KM) -> float:
    """Trapezoidal integral of a profile between two heights, in km."""
    z, v = np.asarray(altitude_km, float), np.asarray(values, float)
    ok = np.isfinite(z) & np.isfinite(v)
    z, v = z[ok], v[ok]
    order = np.argsort(z)
    z, v = z[order], v[order]
    if z.size < 2:
        return float("nan")
    grid = np.r_[z_bottom, z[(z > z_bottom) & (z < z_top)], z_top]
    return float(np.trapezoid(np.interp(grid, z, v), grid))


def mean_flux(height_km, flux, z_bottom=0.0, z_top=RESERVOIR_TOP_KM) -> float:
    """Level mean of the flux profile inside a layer, as the published figure takes it."""
    z, v = np.asarray(height_km, float), np.asarray(flux, float)
    inside = np.isfinite(z) & np.isfinite(v) & (z >= z_bottom) & (z <= z_top)
    return float(np.mean(v[inside])) if inside.any() else float("nan")


def gather(work: Path, convection_work: Path, iops: tuple[int, ...]) -> dict:
    panels = {}
    for experiment in EXPERIMENTS:
        data = load_condensation(work, iops, experiment)
        label = CACHE_LABELS[experiment]
        data["label"] = label
        data["z_peak"] = peak_height_km(data["altitude_km"], data["total"])
        data["z_freeze"] = freezing_level_km(data["altitude_km"], data["temperature_k"])
        data["column"] = column_integral(data["altitude_km"], data["total"])
        data["flux_up"] = load_flux(convection_work, iops, label, "flux_up")
        data["flux_par"] = (
            None if experiment == "NO3M" else load_flux(convection_work, iops, label, "flux_par")
        )
        z_res, content = load_reservoir(work, iops, experiment)
        data["reservoir"] = layer_integral(z_res, content)
        data["reservoir_tendency"] = layer_integral(
            *load_reservoir(work, iops, experiment, "tendency")
        )
        data["conv_0_3"] = layer_integral(data["altitude_km"], data["convective"])
        data["flux_0_3"] = mean_flux(*data["flux_up"])
        data["conv_per_flux"] = data["conv_0_3"] / data["flux_0_3"]
        panels[experiment] = data
    return panels


def draw(panels: dict, png: Path, dpi: int) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 18})
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5.4 * len(EXPERIMENTS), 9.4), sharey=True
    )
    cond_max = max(
        np.nanmax(d["total"][d["altitude_km"] <= Z_MAX_KM]) for d in panels.values()
    ) * 1.08
    flux_max = max(np.nanmax(d["flux_up"][1]) for d in panels.values()) * 1.08

    for index, experiment in enumerate(EXPERIMENTS):
        data = panels[experiment]
        ax = axes[index]
        z = data["altitude_km"]
        if np.isfinite(data["z_peak"]):
            ax.axhspan(0, data["z_peak"], color="#BFC7D5", alpha=0.18, zorder=0)
        ax.plot(data["total"], z, color=TOTAL_COLOR, lw=2.8)
        ax.plot(data["convective"], z, color=CONVECTIVE_COLOR, lw=2.0, ls="-.", alpha=0.9)
        ax.plot(data["resolved"], z, color=RESOLVED_COLOR, lw=2.0, ls="--", alpha=0.95)
        if np.isfinite(data["z_peak"]):
            ax.axhline(data["z_peak"], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), alpha=0.9)
        if np.isfinite(data["z_freeze"]):
            ax.axhline(data["z_freeze"], color=FREEZING_COLOR, lw=1.2, ls=":", alpha=0.95)
        ax.set_xlim(0, cond_max)
        ax.set_ylim(0, Z_MAX_KM)
        ax.set_title(data["label"], pad=34)
        ax.set_xlabel(r"condensation (g kg$^{-1}$ day$^{-1}$)")
        if index == 0:
            ax.set_ylabel("altitude (km)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.text(0.95, 0.97, PANEL_LABELS[index], transform=ax.transAxes, ha="right", va="top",
                fontweight="bold")

        twin = ax.twiny()
        height, flux = data["flux_up"]
        twin.plot(flux, height, color=UPDRAFT_COLOR, lw=2.1, alpha=0.95)
        twin.set_xlim(0, flux_max)
        twin.set_ylim(0, Z_MAX_KM)
        twin.set_xlabel(r"upward mass flux (kg m$^{-2}$ s$^{-1}$)", color=UPDRAFT_COLOR)
        twin.tick_params(axis="x", colors=UPDRAFT_COLOR)
        twin.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        if experiment == "NO3M":
            twin.text(0.5, 0.90, "no 3MT: resolved ascent only", transform=twin.transAxes,
                      ha="center", va="top", fontsize=12, color=UPDRAFT_COLOR)

    handles = [
        Line2D([0], [0], color=TOTAL_COLOR, lw=2.8, label="total condensation"),
        Line2D([0], [0], color=CONVECTIVE_COLOR, lw=2.0, ls="-.", label="convection-scheme part"),
        Line2D([0], [0], color=RESOLVED_COLOR, lw=2.0, ls="--", label="resolved-microphysics part"),
        Line2D([0], [0], color=UPDRAFT_COLOR, lw=2.1, label="upward mass flux"),
        Line2D([0], [0], color=PEAK_COLOR, lw=1.2, ls=(0, (6, 3)), label="condensation peak height"),
        Line2D([0], [0], color=FREEZING_COLOR, lw=1.2, ls=":", label="0 °C isotherm"),
    ]
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.17, wspace=0.10)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3,
               fontsize=LEGEND_FONTSIZE, frameon=False)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_txt(txt: Path, panels: dict, iops: tuple[int, ...]) -> None:
    lines = [
        "Warm-rain condensation budget, column-split upward mass flux, all four runs.",
        f"Period: GoAmazon2014/5 {' and '.join(f'IOP{i}' for i in iops)}; "
        + ", ".join(f"IOP{i}: {GOAMAZON_IOPS[i][0]} to {GOAMAZON_IOPS[i][1]}" for i in iops),
        "Condensation: DDH QV budget at lead +0024 (condcv/condrs), day mean over the "
        "campaign days.",
        "Warm reservoir: cloud + rain water CONTENT below 3 km, (VQL0+VQL1+VQR0+VQR1)/2/VPP0 "
        "from the DDH LFA state, in g/kg. The published figure instead integrated the ddhb "
        "blocks QL.VQLM + QR.VQRM, which the ddhtoolbox conversion list defines as "
        "'EAU CONDENS : MEAN TENDENCY' in g/kg/day - a tendency, not a content. Its value is "
        "reported below as warm_tendency_0_3km for comparison.",
        "Upward mass flux: total rising mass flux of the column split into 3MT updraft, "
        "downdraft and environment, domain mean over the 2960-cell 4 km radar domain, "
        "all hours. flux_par is the 3MT updraft piece alone, for reference.",
        "G2M-XCU runs without 3MT: convective condensation is zero and the upward flux is "
        "the resolved ascent of the single-piece column.",
        "",
        "summary metrics",
        "label,days,column_total_gkgday_km,z_peak_km,compression_gkgday,"
        "warm_reservoir_0_3km_gkg_km,warm_tendency_0_3km_gkgday_km,"
        "convective_condensation_0_3km_gkgday_km,"
        "mean_upward_flux_0_3km_kg_m2_s,convective_condensation_per_flux",
    ]
    for experiment in EXPERIMENTS:
        d = panels[experiment]
        lines.append(
            f"{d['label']},{d['days']},{d['column']:.6g},{d['z_peak']:.6g},"
            f"{d['column'] / max(d['z_peak'], 1e-6):.6g},{d['reservoir']:.6g},"
            f"{d['reservoir_tendency']:.6g},"
            f"{d['conv_0_3']:.6g},{d['flux_0_3']:.6g},{d['conv_per_flux']:.6g}"
        )
    for experiment in EXPERIMENTS:
        d = panels[experiment]
        lines += ["", f"{d['label']} condensation profile",
                  "altitude_km,total_gkgday,convective_gkgday,resolved_gkgday"]
        for z, t, c, r in zip(d["altitude_km"], d["total"], d["convective"], d["resolved"]):
            lines.append(f"{z:.4f},{t:.6g},{c:.6g},{r:.6g}")
        height, flux = d["flux_up"]
        par = d["flux_par"][1] if d["flux_par"] is not None else np.full_like(flux, np.nan)
        lines += ["", f"{d['label']} upward mass flux profile",
                  "height_km,flux_up_kg_m2_s,flux_parameterized_kg_m2_s"]
        for z, f_up, f_par in zip(height, flux, par):
            lines.append(f"{z:.4f},{f_up:.6g},{'' if not np.isfinite(f_par) else f'{f_par:.6g}'}")
    txt.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=CACHE_DIRS["closure_budget"])
    p.add_argument("--convection-work-dir", type=Path, default=CACHE_DIRS["convection"])
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_paper_closure_budget_compression_panel", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        panels = gather(args.work_dir, args.convection_work_dir, iops)
        name = f"{FIGURE_STEM}_{PERIOD_NAMES[key]}"
        png = args.output_dir / f"{name}_450dpi.png"
        txt = args.output_dir / f"{name}_data.txt"
        draw(panels, png, args.dpi)
        write_txt(txt, panels, iops)
        print(f"[saved] {png}\n[saved] {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
