#!/usr/bin/env python3
"""Microphysics figures of the GoAmazon IOP runs against SIPAM and the disdrometer.

Supplementary folder S04 (``iop_settings.supplementary_files``), every figure with IOP1
(wet season) in the top row and IOP2 (dry season) in the bottom row:

    m1_reflectivity_cfad      reflectivity CFADs, SIPAM and the four runs
    m2_echo_occurrence        echo occurrence profiles and 40 dBZ echo-top heights
    m3_convective_stratiform  Steiner convective / stratiform area by local hour, rain shares
    m4_rain_dsd               surface Dm - log10 Nw, disdrometer and the four runs
    m5_hydrometeor_profiles   domain-mean cloud ice, snow, graupel and rain
    m6_ice_water_content      ice water content of stratiform columns, radar (Z, T) and runs
    m7_ice_cloud_height       hydrometeor occurrence above T3, ARM W-band cloud radar (IOP2 only)
    m8_water_budget           the runs' own (DDH) condensation, evaporation and precipitation
    m9_ice_by_column_type     ice water content of all cells, and at 8 km by the column below
    m10_boundary_layer_evaporation  sonde biases of the boundary layer against the runs' evaporation; cold pools
    m11_rain_production       precipitation efficiency and what graupel, two-moment rain and 3MT change

Inputs: ``build_iop_echo_statistics`` (radar and simulated reflectivity on the same
4 km grid, same hours, 60-110 km from the radar, precipitating columns only),
``build_iop_microphysics_columns``, ``build_iop_disdrometer_seasons`` and
``build_iop_wacr_occurrence``. Each figure has its plotted numbers in ``data/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from alaro_analysis.analysis.radar_echo import DBZ_EDGES, HEIGHTS_KM, OCCURRENCE_DBZ, TOP_DBZ
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import CACHE_DIRS, FIGURES, GOAMAZON_IOPS, supplementary_files, write_figure_index

SUPPLEMENT_NUMBER = 4
PERIOD = "iop1_iop2"
CACHE = CACHE_DIRS["microphysics"]
CONVECTION = CACHE_DIRS["convection"]
OBS = ("SIPAM", "black")
RUNS = (("C1M", "#d62728"), ("G1M", "#1f77b4"), ("G2M", "#2ca02c"), ("G2M-XCU", "#9467bd"))
ROWS = {1: "IOP1 (wet season)", 2: "IOP2 (dry season)"}
LETTERS = "abcdefghijklmnop"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})


def g_formatter():
    return mticker.FuncFormatter(lambda v, _: f"{v:g}")


def letter(ax, i, color="black"):
    ax.text(0.03, 0.97, f"({LETTERS[i]})", transform=ax.transAxes, ha="left", va="top", fontsize=14,
            fontweight="bold", color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5})


def load(name: str):
    path = Path(name)
    return dict(np.load(path)) if path.exists() else None


def echo(label: str, iop: int):
    return load(CACHE / "echo_stats" / f"{label}_iop{iop}.npz")


def columns(label: str, iop: int):
    return load(CACHE / "columns" / f"{label}_iop{iop}.npz")


def freezing_level_km(iop: int) -> float:
    """0 degC height of the radar-domain mean temperature (mean over the runs that have it)."""
    levels = []
    for label in ("C1M", "G1M", "G2M"):
        t, h = CONVECTION / f"iop{iop}" / f"{label}_temp.npz", CONVECTION / f"iop{iop}" / f"{label}_height.npz"
        if not (t.exists() and h.exists()):
            continue
        temp = np.nanmean(np.load(t)["mean"], axis=1)
        z = np.load(h)["height_m"] / 1000.0
        d = temp - 273.15
        k = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
        if k.size:
            i = int(k[0])
            levels.append(z[i] + d[i] / (d[i] - d[i + 1]) * (z[i + 1] - z[i]))
    return float(np.mean(levels)) if levels else float("nan")


def finish(fig, png: Path, jpg: Path, dpi: int) -> None:
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    convert_to_jpg(png, jpg, dpi=dpi)
    png.unlink()
    print(f"[saved] {jpg}", flush=True)


def header(title: str, lines: list[str]) -> list[str]:
    out = [title, "=" * len(title)]
    for iop, (start, end) in GOAMAZON_IOPS.items():
        out.append(f"IOP{iop}: {start} to {end} ({ROWS[iop]}).")
    return out + lines + [""]


RADAR_NOTE = [
    "Reflectivity: SIPAM S-band CAPPIs (2..16 km) averaged in linear Z onto the 4 km model cells; model = its",
    "own diagnostic REFLEC_DBZ (GPPRS0D: scheme rain DSD, fixed ICE3-type snow/graupel/ice laws), interpolated",
    "in linear Z to the same heights. Same radar-valid hours (SIPAM every 12-min scan, models the hourly",
    "snapshot), same cells: 60-110 km from the radar (cone of silence inside 60 km). Only precipitating",
    "columns (reaching 20 dBZ somewhere) count, which removes the dry-season clear-air echo; echo >= 10 dBZ.",
]


# --------------------------------------------------------------------------- m1 CFADs
def make_cfad(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m1_reflectivity_cfad", PERIOD)
    datasets = [OBS, *RUNS]
    fig, axes = plt.subplots(2, 5, figsize=(24, 11), sharex=True, sharey=True)
    norm = mcolors.LogNorm(0.05, 20)
    centers = 0.5 * (DBZ_EDGES[1:] + DBZ_EDGES[:-1])
    lines = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        zfrz = freezing_level_km(iop)
        for c, (label, _) in enumerate(datasets):
            ax = axes[r, c]
            s = echo(label, iop)
            if s is None:
                ax.text(0.5, 0.5, "not built", transform=ax.transAxes, ha="center")
                continue
            counts = s["cfad"][0]
            per_level = counts.sum(axis=1)
            freq = 100.0 * counts / np.maximum(per_level[:, None], 1) / np.diff(DBZ_EDGES)[None]
            freq = np.where(per_level[:, None] >= 0.01 * per_level.max(), freq, np.nan)
            mesh = ax.pcolormesh(DBZ_EDGES, np.append(HEIGHTS_KM - 0.5, HEIGHTS_KM[-1] + 0.5), freq,
                                 cmap="viridis", norm=norm, shading="flat")
            cdf = np.cumsum(counts, axis=1) / np.maximum(per_level[:, None], 1)
            med = np.array([np.interp(0.5, cdf[k], centers) if per_level[k] else np.nan for k in range(cdf.shape[0])])
            p90 = np.array([np.interp(0.9, cdf[k], centers) if per_level[k] else np.nan for k in range(cdf.shape[0])])
            ok = per_level >= 0.01 * per_level.max()
            ax.plot(med[ok], HEIGHTS_KM[ok], color="white", lw=2.2)
            ax.plot(p90[ok], HEIGHTS_KM[ok], color="white", lw=2.2, ls="--")
            if np.isfinite(zfrz):
                ax.axhline(zfrz, color="0.85", lw=1.2, ls=":")
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"{ROWS[iop]}\nheight (km)")
            if r == 1:
                ax.set_xlabel("reflectivity (dBZ)")
            ax.set_xlim(10, 60)
            ax.set_ylim(1.5, 16.5)
            letter(ax, r * 5 + c, color="black")
            lines.append(f"IOP{iop},{label}," + ",".join(f"{v:.2f}" for v in med) + "," + ",".join(f"{v:.2f}" for v in p90))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.94, bottom=0.12, wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.65])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("frequency at each height (% per dBZ)")
    cb.ax.yaxis.set_major_formatter(g_formatter())
    handles = [Line2D([0], [0], color="0.25", lw=2.2, label="median"),
               Line2D([0], [0], color="0.25", lw=2.2, ls="--", label="90th percentile"),
               Line2D([0], [0], color="0.5", lw=1.2, ls=":", label=r"0 $^{\circ}$C level")]
    legend_below(fig, axes, handles, ncol=3)
    finish(fig, png, jpg, dpi)
    text = header("m1_reflectivity_cfad", RADAR_NOTE + [
        "Each height row normalised to 100 % (CFAD); rows with < 1 % of the busiest row's echoes masked.",
        "Lines: median and 90th percentile reflectivity per height (white in the figure).",
        "",
        "iop,dataset," + ",".join(f"median_{h:g}km" for h in HEIGHTS_KM) + "," + ",".join(f"p90_{h:g}km" for h in HEIGHTS_KM),
    ]) + lines
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m2 occurrence and tops
def make_occurrence(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m2_echo_occurrence", PERIOD)
    datasets = [OBS, *RUNS]
    fig, axes = plt.subplots(2, 3, figsize=(18, 13), sharey=True)
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        zfrz = freezing_level_km(iop)
        for label, color in datasets:
            s = echo(label, iop)
            if s is None:
                continue
            n = s["n_cells"][0]
            lw = 3.2 if label == "SIPAM" else 2.2
            for c, t in enumerate((10.0, 30.0)):
                i = OCCURRENCE_DBZ.index(t)
                occ = 100.0 * s["occurrence"][i] / n
                axes[r, c].plot(np.where(occ > 0, occ, np.nan), HEIGHTS_KM, color=color, lw=lw)
                rows.append(f"IOP{iop},{label},occurrence_ge{t:g}dBZ_percent," + ",".join(f"{v:.4f}" for v in occ))
            tops = 100.0 * s["tops"][TOP_DBZ.index(40.0)] / n
            axes[r, 2].plot(np.where(tops > 0, tops, np.nan), HEIGHTS_KM, color=color, lw=lw, marker="o", ms=4)
            rows.append(f"IOP{iop},{label},echo_top_40dBZ_percent_per_km," + ",".join(f"{v:.5f}" for v in tops))
        for c in range(3):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(g_formatter())
            if np.isfinite(zfrz):
                ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
            ax.grid(alpha=0.3, which="both")
            ax.set_ylim(1.5, 16.5)
            letter(ax, r * 3 + c)
        axes[r, 0].set_ylabel(f"{ROWS[iop]}\nheight (km)")
    axes[0, 0].set_title(r"echo $\geq$ 10 dBZ")
    axes[0, 1].set_title(r"echo $\geq$ 30 dBZ")
    axes[0, 2].set_title("40 dBZ echo-top height")
    for c, xl in enumerate(("cells with echo (%)", "cells with echo (%)", "cells with the top at this height (%)")):
        axes[1, c].set_xlabel(xl)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.08, hspace=0.18)
    handles = [Line2D([0], [0], color=c, lw=3.2 if l == "SIPAM" else 2.2, label=l) for l, c in datasets]
    handles.append(Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"0 $^{\circ}$C level"))
    legend_below(fig, axes, handles, ncol=6)
    finish(fig, png, jpg, dpi)
    text = header("m2_echo_occurrence", RADAR_NOTE + [
        "Occurrence = share of cell-maps (all cells, all maps) with echo above the threshold at each height.",
        "Echo top = highest of the 15 levels with Z >= 40 dBZ in a column, share of cell-maps per 1 km level.",
        "",
        "iop,dataset,quantity," + ",".join(f"{h:g}km" for h in HEIGHTS_KM),
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m3 convective / stratiform
def make_partition(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m3_convective_stratiform", PERIOD)
    datasets = [OBS, *RUNS]
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    hours = np.arange(24)
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        amounts = []
        for label, color in datasets:
            s = echo(label, iop)
            if s is None:
                continue
            ch = s["class_hour"]
            lw = 3.2 if label == "SIPAM" else 2.2
            conv = 100.0 * ch[0] / np.maximum(ch[2], 1)
            strat = 100.0 * ch[1] / np.maximum(ch[2], 1)
            axes[r, 0].plot(hours, conv, color=color, lw=lw)
            axes[r, 1].plot(hours, strat, color=color, lw=lw)
            per_cell = 24.0 / (s["n_maps"][0] * s["n_stat_cells"])  # summed mm/h -> mm/day per cell
            zr = s["zr_rain_hour"].sum(axis=1) * per_cell
            own = zr if label == "SIPAM" else s["sfc_rain_hour"].sum(axis=1) * per_cell
            amounts.append((label, color, own[0], own[1]))
            rows.append(f"IOP{iop},{label},{100*ch[0].sum()/ch[2].sum():.3f},{100*ch[1].sum()/ch[2].sum():.3f},"
                        f"{zr[0]:.3f},{zr[1]:.3f},{zr[2]:.3f},{own[0]:.3f},{own[1]:.3f},{own[2]:.3f},"
                        + ",".join(f"{v:.3f}" for v in conv) + "," + ",".join(f"{v:.3f}" for v in strat))
        x = np.arange(len(amounts))
        ax = axes[r, 2]
        ax.bar(x - 0.2, [a[2] for a in amounts], width=0.38, color=[a[1] for a in amounts])
        ax.bar(x + 0.2, [a[3] for a in amounts], width=0.38, color=[a[1] for a in amounts], alpha=0.45,
               hatch="//", edgecolor="white")
        ax.set_xticks(x, [a[0] for a in amounts])
        ax.set_ylabel(r"rain (mm day$^{-1}$)")
        ax.grid(alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(g_formatter())
        for c in range(2):
            axes[r, c].set_xlim(0, 23)
            axes[r, c].set_xticks(range(0, 24, 3))
            axes[r, c].grid(alpha=0.3)
            axes[r, c].yaxis.set_major_formatter(g_formatter())
        axes[r, 0].set_ylabel(f"{ROWS[iop]}\ncells (%)")
        for c in range(3):
            letter(axes[r, c], r * 3 + c)
    axes[0, 0].set_title("convective echo")
    axes[0, 1].set_title("stratiform echo")
    axes[0, 2].set_title("rain by echo type")
    for c in range(2):
        axes[1, c].set_xlabel("local hour (UTC-4)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.18)
    handles = [Line2D([0], [0], color=c, lw=3.2 if l == "SIPAM" else 2.2, label=l) for l, c in datasets]
    handles += [Patch(facecolor="0.4", label="convective"),
                Patch(facecolor="0.4", alpha=0.45, hatch="//", edgecolor="white", label="stratiform")]
    legend_below(fig, axes, handles, ncol=4)
    finish(fig, png, jpg, dpi)
    text = header("m3_convective_stratiform", RADAR_NOTE + [
        "Classification at 3 km, Steiner et al. (1995) adapted to the 4 km grid (analysis.radar_echo).",
        "Area = share of all cells (precipitating or not) classified convective / stratiform, by local hour.",
        "Rain by echo type (mm/day, mean over the ring cells and the radar-valid hours): SIPAM from its 2 km",
        "reflectivity with the SIPAM Z-R law (Z = 174.8 R^1.56); runs from their own instantaneous rain rate at",
        "the lowest model level (precipitation flux), in the columns their reflectivity classifies. Some model",
        "rain falls from columns that never reach 20 dBZ (no class): the 'all' columns include it.",
        "The same Z-R law applied to the model reflectivity ('ZR' columns) shows how the radar would see the runs.",
        "",
        "iop,dataset,convective_area_percent,stratiform_area_percent,ZR_convective_mm_day,ZR_stratiform_mm_day,"
        "ZR_all_mm_day,rain_convective_mm_day,rain_stratiform_mm_day,rain_all_mm_day,"
        "convective_area_by_hour_0..23,stratiform_area_by_hour_0..23",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m4 rain DSD
# Bringi et al. (2003, J. Atmos. Sci. 60, 354) maritime-like and continental-like convective clusters.
BRINGI_MARITIME = (1.5, 1.75, 4.0, 4.5)  # Dm min, max (mm), log10 Nw min, max
BRINGI_CONTINENTAL = (2.0, 2.75, 3.0, 3.5)


def _median_2d(hist, xedges, yedges):
    xc, yc = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    n = hist.sum()
    if n == 0:
        return np.nan, np.nan
    return (np.interp(0.5, np.cumsum(hist.sum(axis=1)) / n, xc),
            np.interp(0.5, np.cumsum(hist.sum(axis=0)) / n, yc))


def make_dsd(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m4_rain_dsd", PERIOD)
    obs_name = {1: "wet", 2: "dry"}
    datasets = [("disdrometer", "black"), *RUNS]
    fig, axes = plt.subplots(2, 6, figsize=(30, 11.5))
    norm = mcolors.LogNorm(0.01, 5)
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        obs = load(CACHE / "disdrometer" / f"disdrometer_{obs_name[iop]}.npz")
        obs_med = _median_2d(obs["dsd_hist"], obs["dm_edges"], obs["lognw_edges"]) if obs else (np.nan, np.nan)
        for c, (label, color) in enumerate(datasets):
            ax = axes[r, c]
            s = obs if label == "disdrometer" else columns(label, iop)
            if s is None:
                ax.text(0.5, 0.5, "not built", transform=ax.transAxes, ha="center")
                continue
            h = s["dsd_hist"]
            n = h.sum()
            mesh = ax.pcolormesh(s["dm_edges"], s["lognw_edges"], (100.0 * h / max(n, 1)).T,
                                 cmap="viridis", norm=norm, shading="flat")
            for (x0, x1, y0, y1), name in ((BRINGI_MARITIME, "maritime"), (BRINGI_CONTINENTAL, "continental")):
                ax.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                          ec="0.15", lw=1.8))
                if r == 0 and c == 0:
                    ax.text(x1 + 0.05, y1, name, fontsize=11, color="0.15", va="top")
            med = _median_2d(h, s["dm_edges"], s["lognw_edges"])
            ax.plot(*obs_med, marker="*", ms=18, color="black", mec="white", mew=1.5, ls="none")
            ax.plot(*med, marker="o", ms=11, color=color if label != "disdrometer" else "black",
                    mec="white", mew=1.5, ls="none")
            ax.set_xlim(0.3, 3.5)
            ax.set_ylim(1.5, 5.5)
            if r == 0:
                ax.set_title(label if label != "disdrometer" else "disdrometer")
            if c == 0:
                ax.set_ylabel(f"{ROWS[iop]}\n" + r"log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
            if r == 1:
                ax.set_xlabel(r"D$_m$ (mm)")
            letter(ax, r * 6 + c)
            mu = s["dsd_mu_sum"][0] / s["dsd_n"][0] if "dsd_mu_sum" in s and s["dsd_n"][0] else np.nan
            rows.append(f"IOP{iop},{label},{int(n)},{med[0]:.3f},{med[1]:.3f},{mu:.2f}")
        # median Dm by rain rate
        ax = axes[r, 5]
        for label, color in datasets:
            s = obs if label == "disdrometer" else columns(label, iop)
            if s is None:
                continue
            hb = s["dsd_dm_by_rr"]
            dmc = 0.5 * (s["dm_edges"][1:] + s["dm_edges"][:-1])
            rc = np.sqrt(s["rain_edges"][1:] * s["rain_edges"][:-1])
            med = [np.interp(0.5, np.cumsum(row) / row.sum(), dmc) if row.sum() >= 30 else np.nan for row in hb]
            ax.plot(rc, med, color=color, lw=3.2 if label == "disdrometer" else 2.2, marker="o", ms=5)
            rows.append(f"IOP{iop},{label},median_Dm_by_rain_rate," + ",".join(f"{v:.3f}" for v in med))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(g_formatter())
        ax.set_ylim(0.5, 3.0)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylabel(r"median D$_m$ (mm)")
        if r == 0:
            ax.set_title(r"D$_m$ against rain rate")
        if r == 1:
            ax.set_xlabel(r"rain rate (mm h$^{-1}$)")
        letter(ax, r * 6 + 5)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.94, bottom=0.12, wspace=0.28, hspace=0.14)
    cax = fig.add_axes([0.915, 0.2, 0.01, 0.65])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("share of rain samples (%)")
    cb.ax.yaxis.set_major_formatter(g_formatter())
    handles = [Line2D([0], [0], marker="*", ms=16, color="black", ls="none", label="disdrometer median"),
               Line2D([0], [0], marker="o", ms=10, color="0.4", ls="none", label="panel median"),
               Patch(facecolor="none", edgecolor="0.3", lw=1.8, label="Bringi et al. (2003) convective clusters")]
    handles += [Line2D([0], [0], color=c, lw=3.2 if l == "disdrometer" else 2.2, label=l) for l, c in datasets]
    legend_below(fig, axes, handles, ncol=4)
    finish(fig, png, jpg, dpi)
    text = header("m4_rain_dsd", [
        "Disdrometer: ARM laser disdrometer (maoldS10, strict QC, 1-min spectra) near Manacapuru. It starts on",
        "25 Sep 2014, so seasons are matched by calendar: wet = 1 Feb-31 Mar 2015 (for IOP1), dry = 15 Aug-15 Oct",
        "2014 and 2015 (for IOP2); the exact IOP2 overlap 25 Sep-15 Oct 2014 alone gives the same medians.",
        "QC: rain rate > 0.1 mm/h, Z < 55 dBZ, >= 5 consecutive minutes; Dm = M4/M3, Nw from LWC and Dm.",
        "Models: lowest model level, every rainy cell (qr > 1e-7) of the 4 km radar domain and every hourly",
        "snapshot, with each scheme's own size distribution (1-moment Abel & Boutle 2012, mu = 0; 2-moment gamma,",
        "ALARO variable mu), truncated at the disdrometer's 0.312 mm; same QC thresholds.",
        "Joint frequency = share of all samples per (Dm, log10 Nw) bin. Rain rate for the last column from LWC",
        "and Dm with the Atlas et al. (1973) mass-weighted fall speed, the same for all datasets.",
        "",
        "iop,dataset,n_samples,median_Dm_mm,median_log10Nw,mean_mu  /  iop,dataset,median_Dm_by_rain_rate(bins 0.1-0.3-1-2-5-10-20-50-200 mm/h)",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m5 hydrometeor profiles
SPECIES_PANELS = (("qi", "cloud ice"), ("qs", "snow"), ("qg", "graupel"), ("qr", "rain"))


def make_hydrometeors(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m5_hydrometeor_profiles", PERIOD)
    fig, axes = plt.subplots(2, 4, figsize=(24, 13), sharey=True)
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        zfrz = freezing_level_km(iop)
        for label, color in RUNS:
            s = columns(label, iop)
            if s is None:
                continue
            h = s["profile_heights_km"]
            n = s["n_profile"][0]
            sp = {name: s["species_sum"][k] / n * 1000.0 for k, name in enumerate(s["species"])}  # g/kg
            for c, (name, _) in enumerate(SPECIES_PANELS):
                axes[r, c].plot(sp[name], h, color=color, lw=2.2)
                rows.append(f"IOP{iop},{label},{name}_g_per_kg," + ",".join(f"{v:.5f}" for v in sp[name]))
        for c in range(4):
            ax = axes[r, c]
            ax.set_ylim(0, 16)
            ax.set_xlim(left=0)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.xaxis.set_major_formatter(g_formatter())
            if np.isfinite(zfrz):
                ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
            letter(ax, r * 4 + c)
        axes[r, 0].set_ylabel(f"{ROWS[iop]}\nheight (km)")
    for c, (_, title) in enumerate(SPECIES_PANELS):
        axes[0, c].set_title(title)
        axes[1, c].set_xlabel(r"mass fraction (g kg$^{-1}$)")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.12, hspace=0.18)
    handles = [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in RUNS]
    handles.append(Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"0 $^{\circ}$C level"))
    legend_below(fig, axes, handles, ncol=5)
    finish(fig, png, jpg, dpi)
    text = header("m5_hydrometeor_profiles", [
        "Model only, 4 km radar domain (2960 cells), every hourly snapshot of the IOP: domain-mean mass fraction",
        "(g/kg) of each hydrometeor at fixed heights (0.25 km steps). C1M has no graupel.",
        "",
        "iop,dataset,quantity," + ",".join(f"{h:g}km" for h in columns_heights()),
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


def columns_heights() -> np.ndarray:
    from alaro_analysis.data.microphysics_columns import PROFILE_HEIGHTS_KM

    return PROFILE_HEIGHTS_KM


# --------------------------------------------------------------------------- DDH water budget (m8, m10, m11)
EVAP_LAYERS_KM = ((0.0, 3.0, "below 3 km", "#fdae6b"), (3.0, 7.0, "3-7 km", "#e6550d"),
                  (7.0, 20.0, "above 7 km", "#a63603"))
CONDENSATION_COLOR = "#636363"
PRECIPITATION_COLOR = "#3182bd"
LOW_LEVEL_KM = 3.0


def ddh(label: str, iop: int):
    return load(CACHE / "ddh_budget" / f"{label}_iop{iop}.npz")


def ddh_heights_km(b: dict) -> np.ndarray:
    from alaro_analysis.data.ddh_water_budget import full_level_heights

    return full_level_heights(b["day_VPP0"].mean(axis=0), b["day_VCT0"].mean(axis=0),
                              b["day_VQV0"].mean(axis=0)) / 1000.0


def ddh_summary(b: dict) -> dict:
    """Day-mean column budget (mm/day) of one run and IOP over the DDH box."""
    from alaro_analysis.data.ddh_water_budget import (CONDENSATION, EVAPORATION, FORMATION, PRECIPITATION,
                                                      PRECIPITATION_3MT, column_total, per_level)

    z = ddh_heights_km(b)
    evap_level = per_level(b, EVAPORATION).mean(axis=0)
    out = {
        "P": column_total(b, PRECIPITATION).mean(),
        "P_3MT": column_total(b, PRECIPITATION_3MT).mean(),
        "C": column_total(b, CONDENSATION).mean(),
        "A": column_total(b, FORMATION).mean(),
        "Ev": column_total(b, EVAPORATION).mean(),
        "E_sfc": -b["day_FQVDIFT"][:, -1].mean(),
        "MC": b["day_TQVDIVFLUHOR"].sum(axis=1).mean(),
        "z": z,
        "evap_level": evap_level,
        "evap_layers": [evap_level[(z >= lo) & (z < hi)].sum() for lo, hi, _, _ in EVAP_LAYERS_KM],
    }
    out["PE"] = out["P"] / out["C"]
    hourly_evap = per_level(b, EVAPORATION, prefix="hour_")[..., z < LOW_LEVEL_KM].sum(axis=-1)  # (days, 24)
    hourly_rain = column_total(b, PRECIPITATION, prefix="hour_")
    out["low_evap_hour"] = hourly_evap.mean(axis=0)
    out["rain_hour"] = hourly_rain.mean(axis=0)
    out["low_evap"] = hourly_evap.sum(axis=1).mean()
    return out


def local_order() -> tuple[np.ndarray, np.ndarray]:
    """DDH hour index i covers i..i+1 UTC; local start hour and the order that sorts it."""
    lh = (np.arange(24) + UTC_OFFSET_H) % 24
    return lh, np.argsort(lh)


UTC_OFFSET_H = -4


def make_water_budget(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m8_water_budget", PERIOD)
    fig, axes = plt.subplots(2, 3, figsize=(24, 13), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    rows = []
    lh, order = local_order()
    for r, iop in enumerate(GOAMAZON_IOPS):
        zfrz = freezing_level_km(iop)
        ax = axes[r, 0]
        labels = []
        for i, (label, color) in enumerate(RUNS):
            b = ddh(label, iop)
            if b is None:
                continue
            s = ddh_summary(b)
            labels.append(label)
            x = len(labels) - 1
            ax.bar(x - 0.27, s["C"], width=0.25, color=CONDENSATION_COLOR)
            bottom = 0.0
            for (lo, hi, name, c), v in zip(EVAP_LAYERS_KM, s["evap_layers"]):
                ax.bar(x, v, width=0.25, bottom=bottom, color=c)
                bottom += v
            ax.bar(x + 0.27, s["P"], width=0.25, color=PRECIPITATION_COLOR)
            ax.text(x + 0.27, s["P"] + 0.3, f"{s['PE']:.2f}", ha="center", va="bottom", fontsize=12)
            axes[r, 1].plot(np.where(s["z"] < 16, s["evap_level"] / np.maximum(np.gradient(-s["z"]), 1e-6), np.nan),
                            s["z"], color=color, lw=2.2)
            ratio = np.where(s["rain_hour"] > 0.02, s["low_evap_hour"] / np.maximum(s["rain_hour"], 1e-9), np.nan)
            axes[r, 2].plot(lh[order], ratio[order], color=color, lw=2.2, marker="o", ms=4)
            rows.append(f"IOP{iop},{label},P,{s['P']:.3f},P_3MT,{s['P_3MT']:.3f},condensation,{s['C']:.3f},"
                        f"formation,{s['A']:.3f},evaporation,{s['Ev']:.3f},evap_below3km,{s['evap_layers'][0]:.3f},"
                        f"evap_3to7km,{s['evap_layers'][1]:.3f},evap_above7km,{s['evap_layers'][2]:.3f},"
                        f"P_over_C,{s['PE']:.3f},Ev_over_formation,{s['Ev'] / s['A']:.3f},"
                        f"surface_evaporation,{s['E_sfc']:.3f},moisture_convergence,{s['MC']:.3f},"
                        f"low_evap_over_P,{s['low_evap'] / s['P']:.3f}")
            rows.append(f"IOP{iop},{label},low_evap_by_local_hour_0..23_mm_h," + ",".join(f"{v:.3f}" for v in s["low_evap_hour"][order]))
            rows.append(f"IOP{iop},{label},rain_by_local_hour_0..23_mm_h," + ",".join(f"{v:.3f}" for v in s["rain_hour"][order]))
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel(f"{ROWS[iop]}\n" + r"mm day$^{-1}$")
        ax.grid(alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(g_formatter())
        ax = axes[r, 1]
        ax.set_ylim(0, 16)
        ax.grid(alpha=0.3)
        ax.axvline(0, color="0.5", lw=1)
        ax.xaxis.set_major_formatter(g_formatter())
        if np.isfinite(zfrz):
            ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
        ax.set_ylabel("height (km)")
        ax = axes[r, 2]
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(g_formatter())
        ax.set_ylabel("evaporation below 3 km / surface precipitation")
        for c in range(3):
            letter(axes[r, c], r * 3 + c)
    axes[0, 0].set_title("water budget of the DDH box")
    axes[0, 1].set_title("evaporation of precipitation")
    axes[0, 2].set_title("evaporation below 3 km per unit rain")
    axes[1, 1].set_xlabel(r"mm day$^{-1}$ km$^{-1}$")
    axes[1, 2].set_xlabel("local hour (UTC-4)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.28, hspace=0.18)
    handles = [Patch(facecolor=CONDENSATION_COLOR, label="condensation"),
               *[Patch(facecolor=c, label=f"evaporation {name}") for _, _, name, c in EVAP_LAYERS_KM],
               Patch(facecolor=PRECIPITATION_COLOR, label="surface precipitation (number: / condensation)")]
    handles += [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in RUNS]
    handles.append(Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"0 $^{\circ}$C level"))
    legend_below(fig, axes, handles, ncol=5)
    finish(fig, png, jpg, dpi)
    text = header("m8_water_budget", [
        "The models' own water budget (DDH), every campaign day, over the DDH box of about 150 km x 150 km centred",
        "on T3 (61.27-59.93 W, 3.88-2.54 S). Day totals in mm/day (= kg m-2 day-1).",
        "condensation = net condensation + deposition to cloud water and cloud ice (3MT + resolved);",
        "formation = conversion of cloud water and ice to rain, snow and graupel; evaporation = evaporation and",
        "sublimation of the falling rain, snow and graupel, summed over the three (the melting layer books snow",
        "losses against rain gains, so only the sum is water returned to vapour); surface precipitation = rain +",
        "snow + graupel at the ground. The number on the blue bar is surface precipitation / condensation.",
        "Water closes to ~5 %: formation - evaporation - precipitation = -0.3 to -0.55 mm/day.",
        "Profile: evaporation per km. Last column: evaporation below 3 km / surface precipitation by local hour",
        "(DDH hourly differences; hours with < 0.02 mm/h of rain not drawn).",
        "",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


SOUNDING_DATA = FIGURES / "06_sounding_profiles" / "data"


def sounding_bias(iop: int, quantity: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Model minus ARM sonde profiles of figure 6 (``Temperature`` or ``Relative humidity``)."""
    path = SOUNDING_DATA / f"sounding_profiles_iop{iop}.txt"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(quantity))
    header_row = lines[start + 2].split(",")
    out_rows = []
    for l in lines[start + 3:]:
        if not l or not l[0].isdigit():
            break
        out_rows.append([float(v) for v in l.split(",")])
    data = np.array(out_rows)
    result = {}
    for label, _ in RUNS:
        col = header_row.index(f"{label}-ARM")
        result[label] = (data[:, 0], data[:, col])
    return result


def make_boundary_layer(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m10_boundary_layer_evaporation", PERIOD)
    fig, axes = plt.subplots(2, 4, figsize=(28, 13))
    rows = []
    hours = np.arange(24)
    for r, iop in enumerate(GOAMAZON_IOPS):
        t_bias = sounding_bias(iop, "Temperature")
        rh_bias = sounding_bias(iop, "Relative humidity")
        for label, color in RUNS:
            if label in t_bias:
                z, v = t_bias[label]
                axes[r, 0].plot(v[z <= 3], z[z <= 3], color=color, lw=2.2, marker="o", ms=4)
            if label in rh_bias:
                z, v = rh_bias[label]
                axes[r, 1].plot(v[z <= 3], z[z <= 3], color=color, lw=2.2, marker="o", ms=4)
            b = ddh(label, iop)
            c = columns(label, iop)
            if b is None or c is None or label not in t_bias:
                continue
            s = ddh_summary(b)
            ratio = s["low_evap"] / s["P"]
            z, v = t_bias[label]
            k = int(np.argmin(np.abs(z - 0.5)))
            zr, vr = rh_bias[label]
            axes[r, 2].plot(ratio, v[k], marker="o", ms=14, color=color, mec="white", mew=1.5, ls="none")
            offset = 14 * ([l for l, _ in RUNS].index(label) % 3) - 10
            axes[r, 2].annotate(f"{vr[k]:+.0f} % RH", (ratio, v[k]), textcoords="offset points", xytext=(10, offset),
                                fontsize=12, color="0.2")
            cp = c["coldpool_hour"][0] / np.maximum(c["coldpool_hour"][1], 1)
            ok = c["coldpool_hour"][1] >= 1000
            axes[r, 3].plot(hours[ok], cp[ok], color=color, lw=2.2, marker="o", ms=4)
            t2m = c["t2m_hour"][0] / np.maximum(c["t2m_hour"][1], 1) - 273.15
            rows.append(f"IOP{iop},{label},low_evap_over_P,{ratio:.3f},T_bias_0.5km,{v[k]:.3f},RH_bias_0.5km,{vr[k]:.3f},"
                        f"domain_mean_T2m_C,{np.nanmean(t2m):.3f},coldpool_dT_14-19LT,{np.nanmean(cp[14:20]):.3f}")
            rows.append(f"IOP{iop},{label},coldpool_dT_by_local_hour," + ",".join(f"{v_:.3f}" for v_ in cp))
            rows.append(f"IOP{iop},{label},domain_mean_T2m_by_local_hour_C," + ",".join(f"{v_:.3f}" for v_ in t2m))
        for c_ in range(2):
            ax = axes[r, c_]
            ax.axvline(0, color="0.5", lw=1)
            ax.set_ylim(0, 3.1)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(g_formatter())
        axes[r, 0].set_ylabel(f"{ROWS[iop]}\nheight (km)")
        ax = axes[r, 2]
        ax.axhline(0, color="0.5", lw=1)
        xs = [l.get_xdata()[0] for l in ax.lines if len(l.get_xdata()) == 1]
        ys = [l.get_ydata()[0] for l in ax.lines if len(l.get_ydata()) == 1]
        if xs:
            ax.set_xlim(min(xs) - 0.03, max(xs) + 0.06)
            ax.set_ylim(min(ys + [0.0]) - 0.25, max(ys + [0.0]) + 0.25)
        ax.grid(alpha=0.3)
        ax.set_ylabel("model - sonde temperature at 0.5 km (K)")
        ax.xaxis.set_major_formatter(g_formatter())
        ax = axes[r, 3]
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
        ax.set_ylabel("2 m temperature under rain - dry median (K)")
        for c_ in range(4):
            letter(axes[r, c_], r * 4 + c_)
    axes[0, 0].set_title("temperature bias against the sondes")
    axes[0, 1].set_title("relative humidity bias against the sondes")
    axes[0, 2].set_title("boundary-layer bias and evaporation")
    axes[0, 3].set_title("cold pools")
    axes[1, 0].set_xlabel("model - sonde (K)")
    axes[1, 1].set_xlabel("model - sonde (% RH)")
    axes[1, 2].set_xlabel("evaporation below 3 km / surface precipitation")
    axes[1, 3].set_xlabel("local hour (UTC-4)")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.18)
    handles = [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in RUNS]
    legend_below(fig, axes, handles, ncol=4)
    finish(fig, png, jpg, dpi)
    text = header("m10_boundary_layer_evaporation", [
        "Columns 1-2: model minus ARM MAO radiosonde, 0-3 km, the profiles of figure 6 (runs sampled at the launch",
        "times, 290 / 305 launches; data from 06_sounding_profiles/data/sounding_profiles_iop<n>.txt).",
        "Column 3: that temperature bias at 0.5 km against the runs' own evaporation of falling precipitation",
        "below 3 km divided by their surface precipitation (DDH box around T3, m8); the label gives the RH bias.",
        "Surface fluxes and radiation of the DDH box are the same in all runs within ~3 W/m2 (data below), so",
        "the runs differ in their boundary layer through the evaporation of their precipitation.",
        "Column 4: 2 m temperature of cells with >= 1 mm in the hour minus the hour's median over non-raining",
        "cells, by local hour, 4 km radar domain (hours with < 1000 such cells not drawn). No observations. It",
        "mixes evaporative cooling with cloud shading by day and cloud greenhouse warming at night (positive).",
        "",
    ]) + rows + [""] + surface_energy_rows()
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


def surface_energy_rows() -> list[str]:
    rows = ["iop,run,sensible_heat_W_m2,net_solar_surface_W_m2,net_thermal_surface_W_m2,latent_heat_W_m2"]
    for iop in GOAMAZON_IOPS:
        for label, _ in RUNS:
            b = ddh(label, iop)
            if b is None or "day_FCTRSO" not in b:
                continue
            per_s = 1.0 / 86400.0
            rows.append(f"IOP{iop},{label},{-b['day_FCTDIFT'][:, -1].mean() * per_s:.1f},"
                        f"{b['day_FCTRSO'][:, -1].mean() * per_s:.1f},{b['day_FCTRTH'][:, -1].mean() * per_s:.1f},"
                        f"{-b['day_FQVDIFT'][:, -1].mean() * per_s * 2.5e6:.1f}")
    return rows


SWITCHES = (("graupel", "C1M", "G1M"), ("two-moment rain", "G1M", "G2M"), ("3MT off", "G2M", "G2M-XCU"))


def make_rain_production(root: Path, dpi: int) -> None:
    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m11_rain_production", PERIOD)
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    rows = []
    summary = {(label, iop): ddh_summary(b) for iop in GOAMAZON_IOPS for label, _ in RUNS
               if (b := ddh(label, iop)) is not None}
    markers = {1: "o", 2: "s"}
    ax = axes[0, 0]
    for (label, iop), s in summary.items():
        color = dict(RUNS)[label]
        ax.plot(s["C"], s["P"], marker=markers[iop], ms=13, color=color, mec="white", mew=1.5, ls="none")
    cmax = max(s["C"] for s in summary.values()) * 1.1
    for pe in (0.5, 0.6):
        ax.plot([0, cmax], [0, pe * cmax], color="0.6", lw=1, ls="--")
        ax.text(cmax, pe * cmax, f" {pe:g}", va="center", fontsize=12, color="0.4")
    ax.set_xlim(0, cmax * 1.08)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"condensation (mm day$^{-1}$)")
    ax.set_ylabel(r"surface precipitation (mm day$^{-1}$)")
    ax.set_title("precipitation against condensation")
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    width = 0.13
    for j, iop in enumerate(GOAMAZON_IOPS):
        for i, (name, a, b_) in enumerate(SWITCHES):
            if (a, iop) not in summary or (b_, iop) not in summary:
                continue
            sa, sb = summary[(a, iop)], summary[(b_, iop)]
            d_cond = (sb["C"] - sa["C"]) * sa["PE"]
            d_eff = sb["C"] * (sb["PE"] - sa["PE"])
            x = i + (j - 0.5) * 0.45
            ax.bar(x - width, d_cond, width=width, color=CONDENSATION_COLOR, hatch="//" if iop == 2 else None,
                   edgecolor="white")
            ax.bar(x, d_eff, width=width, color="#e6550d", hatch="//" if iop == 2 else None, edgecolor="white")
            ax.bar(x + width, sb["P"] - sa["P"], width=width, color=PRECIPITATION_COLOR,
                   hatch="//" if iop == 2 else None, edgecolor="white")
            rows.append(f"IOP{iop},{name},{a}->{b_},dP,{sb['P'] - sa['P']:.3f},from_condensation,{d_cond:.3f},"
                        f"from_efficiency,{d_eff:.3f},dC,{sb['C'] - sa['C']:.3f},dPE,{sb['PE'] - sa['PE']:.3f},"
                        f"dMC,{sb['MC'] - sa['MC']:.3f},dE_sfc,{sb['E_sfc'] - sa['E_sfc']:.3f}")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(range(len(SWITCHES)), [f"{n}\n({a} to {b_})" for n, a, b_ in SWITCHES])
    ax.set_ylabel(r"change in surface precipitation (mm day$^{-1}$)")
    ax.set_title("what each switch changes")
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(g_formatter())
    ax = axes[1, 0]
    for (label, iop), s in summary.items():
        color = dict(RUNS)[label]
        ax.plot(s["MC"] + s["E_sfc"], s["P"], marker=markers[iop], ms=13, color=color, mec="white", mew=1.5, ls="none")
    lim = max(max(s["MC"] + s["E_sfc"], s["P"]) for s in summary.values()) * 1.1
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"moisture convergence + surface evaporation (mm day$^{-1}$)")
    ax.set_ylabel(r"surface precipitation (mm day$^{-1}$)")
    ax.set_title("where the water comes from")
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    from alaro_analysis.data.ddh_water_budget import CONDENSATION, per_level
    for label, color in RUNS:
        b = ddh(label, 1)
        if b is None:
            continue
        z = ddh_heights_km(b)
        cond = per_level(b, CONDENSATION).mean(axis=0) / np.maximum(np.gradient(-z), 1e-6)
        ax.plot(cond, z, color=color, lw=2.2)
        rows.append(f"IOP1,{label},condensation_mm_day_km,(heights km)" + ",".join(f"{v:.2f}:{c:.3f}" for v, c in zip(z, cond) if v < 16))
    zfrz = freezing_level_km(1)
    if np.isfinite(zfrz):
        ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
    ax.axvline(0, color="0.5", lw=1)
    ax.set_ylim(0, 16)
    ax.set_xlabel(r"condensation (mm day$^{-1}$ km$^{-1}$)")
    ax.set_ylabel("height (km)")
    ax.set_title("condensation profile, IOP1 (wet season)")
    ax.grid(alpha=0.3)
    for i, ax in enumerate(axes.flat):
        letter(ax, i)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1, wspace=0.25, hspace=0.25)
    handles = [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in RUNS]
    handles += [Line2D([0], [0], marker=m, ms=12, color="0.4", ls="none", label=ROWS[i]) for i, m in markers.items()]
    handles += [Patch(facecolor=CONDENSATION_COLOR, label="from the change in condensation"),
                Patch(facecolor="#e6550d", label="from the change in precipitation efficiency"),
                Patch(facecolor=PRECIPITATION_COLOR, label="total change"),
                Patch(facecolor="0.7", hatch="//", edgecolor="white", label="IOP2 (hatched)")]
    legend_below(fig, axes, handles, ncol=4)
    finish(fig, png, jpg, dpi)
    text = header("m11_rain_production", [
        "DDH box around T3, day-mean budgets of every campaign day (m8).",
        "Precipitation efficiency PE = surface precipitation / condensation (cloud water + ice, 3MT + resolved).",
        "Change between two runs a -> b: dP = (C_b - C_a) PE_a + C_b (PE_b - PE_a), exact.",
        "Moisture convergence = column sum of the horizontal moisture-flux convergence into the box; with the",
        "surface evaporation it balances the precipitation (the change of precipitable water is < 0.1 mm/day).",
        "",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m6 ice water content
# The coarse radar field starts at 0 dBZ, i.e. ~0.03 g m-3 of ice at -10 degC with the Hogan relation;
# both distributions are cut there so the medians compare like with like.
IWC_FLOOR_LOG = -1.5


def make_iwc(root: Path, dpi: int) -> None:
    from alaro_analysis.analysis.radar_echo import IWC_THRESHOLDS, LOG_IWC_EDGES, Z_IWC_T

    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m6_ice_water_content", PERIOD)
    datasets = [OBS, *RUNS]
    fig, axes = plt.subplots(2, 6, figsize=(30, 11.5))
    norm = mcolors.LogNorm(0.05, 20)
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        for c, (label, color) in enumerate(datasets):
            ax = axes[r, c]
            s = echo(label, iop)
            if s is None or "iwc_occurrence_all_cells" not in s or s["iwc_hist"][1].sum() == 0:
                ax.text(0.5, 0.5, "not built", transform=ax.transAxes, ha="center")
                continue
            counts = s["iwc_hist"][1].copy()  # stratiform columns, where the aggregate Z-IWC relation applies
            counts[:, LOG_IWC_EDGES[:-1] < IWC_FLOOR_LOG] = 0.0  # below what the radar can see (0 dBZ)
            per_level = counts.sum(axis=1)
            freq = 100.0 * counts / np.maximum(per_level[:, None], 1) / np.diff(LOG_IWC_EDGES)[None] * 0.1
            freq = np.where(per_level[:, None] >= 0.01 * per_level.max(), freq, np.nan)
            mesh = ax.pcolormesh(LOG_IWC_EDGES, np.append(HEIGHTS_KM - 0.5, HEIGHTS_KM[-1] + 0.5), freq,
                                 cmap="viridis", norm=norm, shading="flat")
            centers = 0.5 * (LOG_IWC_EDGES[1:] + LOG_IWC_EDGES[:-1])
            cdf = np.cumsum(counts, axis=1) / np.maximum(per_level[:, None], 1)
            med = np.array([np.interp(0.5, cdf[k], centers) if per_level[k] else np.nan for k in range(cdf.shape[0])])
            ok = per_level >= 0.01 * per_level.max()
            ax.plot(med[ok], HEIGHTS_KM[ok], color="white", lw=2.2)
            ax.set_xlim(IWC_FLOOR_LOG, 1)
            ax.set_ylim(4.5, 16.5)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{10**v:g}"))
            if r == 0:
                ax.set_title(label if label != "SIPAM" else "SIPAM (from Z and T)")
            if c == 0 and r == 0:
                ax.text(0.97, 0.03, "stratiform columns", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, color="white")
            if c == 0:
                ax.set_ylabel(f"{ROWS[iop]}\nheight (km)")
            if r == 1:
                ax.set_xlabel(r"ice water content (g m$^{-3}$)")
            letter(ax, r * 6 + c)
            n = s["n_cells"][0]
            occ = 100.0 * s["iwc_occurrence"][1] / n
            occ_cells = 100.0 * s["iwc_occurrence_all_cells"] / n
            j = IWC_THRESHOLDS.index(0.1)
            axes[r, 5].plot(np.where(occ[j] > 0, occ[j], np.nan), HEIGHTS_KM, color=color,
                            lw=3.2 if label == "SIPAM" else 2.2)
            rows.append(f"IOP{iop},{label},median_log10_iwc," + ",".join(f"{v:.3f}" for v in med))
            for jj, thr in enumerate(IWC_THRESHOLDS):
                rows.append(f"IOP{iop},{label},all_cells_occurrence_iwc_ge{thr:g}_percent," + ",".join(f"{v:.4f}" for v in occ_cells[jj]))
                rows.append(f"IOP{iop},{label},stratiform_occurrence_iwc_ge{thr:g}_percent," + ",".join(f"{v:.4f}" for v in occ[jj]))
                occ_all = 100.0 * s["iwc_occurrence"][0][jj] / n
                rows.append(f"IOP{iop},{label},all_precipitating_occurrence_iwc_ge{thr:g}_percent," + ",".join(f"{v:.4f}" for v in occ_all))
        ax = axes[r, 5]
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(g_formatter())
        ax.set_ylim(4.5, 16.5)
        ax.grid(alpha=0.3, which="both")
        if r == 0:
            ax.set_title(r"ice water content $\geq$ 0.1 g m$^{-3}$")
        if r == 1:
            ax.set_xlabel("cells (%)")
        letter(ax, r * 6 + 5)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.94, bottom=0.12, wspace=0.25, hspace=0.14)
    cax = fig.add_axes([0.915, 0.2, 0.01, 0.65])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("frequency at each height (% per 0.1 decade)")
    cb.ax.yaxis.set_major_formatter(g_formatter())
    handles = [Line2D([0], [0], color="0.25", lw=2.2, label="median")]
    handles += [Line2D([0], [0], color=c, lw=3.2 if l == "SIPAM" else 2.2, label=l) for l, c in datasets]
    legend_below(fig, axes, handles, ncol=6)
    finish(fig, png, jpg, dpi)
    a, b, c_, d = Z_IWC_T
    text = header("m6_ice_water_content", RADAR_NOTE + [
        f"Radar IWC from Z and T, Hogan et al. (2006) 3 GHz: log10 IWC = {b} Z {c_:+} T {d:+} (IWC g/m3, Z dBZ, T degC);",
        "fitted to stratiform aggregates, so the figure shows STRATIFORM columns only (Steiner class at 3 km);",
        "the data text also gives all precipitating columns and all cells (m9). T = IOP-mean radar-domain model",
        "temperature at each",
        "height; only heights with T <= -5 degC. Model IWC = rho * (snow + graupel + cloud ice). Same cells, hours",
        "and precipitating-column filter as the reflectivity figures. Both distributions cut at 10^-1.5 g/m3",
        "(~0 dBZ, what the coarse radar field can see) before medians are taken.",
        "",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m9 where the ice is
ICE_HEIGHT_KM = 8.0
COLUMN_TYPES = (("stratiform columns", 1.0, None),
                ("convective and other precipitating columns", 0.55, None),
                ("non-precipitating columns", 0.3, "//"))


def make_ice_location(root: Path, dpi: int) -> None:
    from alaro_analysis.analysis.radar_echo import IWC_THRESHOLDS

    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m9_ice_by_column_type", PERIOD)
    datasets = [OBS, *RUNS]
    j = IWC_THRESHOLDS.index(0.1)
    k = int(np.argmin(np.abs(HEIGHTS_KM - ICE_HEIGHT_KM)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), gridspec_kw={"width_ratios": [1, 1.25]})
    rows = []
    for r, iop in enumerate(GOAMAZON_IOPS):
        zfrz = freezing_level_km(iop)
        bars = []
        for label, color in datasets:
            s = echo(label, iop)
            if s is None or "iwc_occurrence_all_cells" not in s:
                continue
            n = s["n_cells"][0]
            cells = 100.0 * s["iwc_occurrence_all_cells"][j] / n
            precipitating = 100.0 * s["iwc_occurrence"][0][j] / n
            stratiform = 100.0 * s["iwc_occurrence"][1][j] / n
            axes[r, 0].plot(np.where(cells > 0, cells, np.nan), HEIGHTS_KM, color=color,
                            lw=3.2 if label == "SIPAM" else 2.2)
            parts = (stratiform[k], precipitating[k] - stratiform[k], cells[k] - precipitating[k])
            bars.append((label, color, parts))
            rows.append(f"IOP{iop},{label},all_cells_iwc_ge0.1_percent," + ",".join(f"{v:.4f}" for v in cells))
            rows.append(f"IOP{iop},{label},at_{ICE_HEIGHT_KM:g}km_stratiform_convective_other_nonprecipitating_percent,"
                        + ",".join(f"{v:.3f}" for v in parts))
        ax = axes[r, 0]
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(g_formatter())
        ax.set_ylim(4.5, 16.5)
        ax.grid(alpha=0.3, which="both")
        if np.isfinite(zfrz):
            ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
        ax.set_ylabel(f"{ROWS[iop]}\nheight (km)")
        ax = axes[r, 1]
        x = np.arange(len(bars))
        bottom = np.zeros(len(bars))
        for t, (_, alpha, hatch) in enumerate(COLUMN_TYPES):
            vals = np.array([b[2][t] for b in bars])
            ax.bar(x, vals, bottom=bottom, width=0.62, color=[b[1] for b in bars], alpha=alpha, hatch=hatch,
                   edgecolor="white", linewidth=1.5)
            bottom += vals
        ax.set_xticks(x, [b[0] for b in bars])
        ax.set_ylabel(f"cells at {ICE_HEIGHT_KM:g} km (%)")
        ax.grid(alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(g_formatter())
        for c in range(2):
            letter(axes[r, c], r * 2 + c)
    axes[0, 0].set_title(r"ice water content $\geq$ 0.1 g m$^{-3}$, all cells")
    axes[0, 1].set_title(rf"ice water content $\geq$ 0.1 g m$^{{-3}}$ at {ICE_HEIGHT_KM:g} km, by column")
    axes[1, 0].set_xlabel("cells (%)")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.18)
    handles = [Line2D([0], [0], color=c, lw=3.2 if l == "SIPAM" else 2.2, label=l) for l, c in datasets]
    handles += [Patch(facecolor="0.3", alpha=alpha, hatch=hatch, edgecolor="white", label=name)
                for name, alpha, hatch in COLUMN_TYPES]
    handles.append(Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"0 $^{\circ}$C level"))
    legend_below(fig, axes, handles, ncol=4)
    finish(fig, png, jpg, dpi)
    text = header("m9_ice_by_column_type", RADAR_NOTE[:4] + [
        "Ice water content as in m6 (radar: Hogan et al. 2006 from Z and the IOP-mean model T; models: rho * (snow",
        "+ graupel + cloud ice)), threshold 0.1 g/m3 (~5 dBZ, above the coarse radar field's 0 dBZ floor).",
        "Left: share of ALL cells of the 60-110 km ring, with no precipitating-column filter and no classification,",
        "so it does not depend on the models' rain and snow reflectivity laws.",
        f"Right, at {ICE_HEIGHT_KM:g} km: the same cells split by the column below: stratiform (Steiner class at 3 km),",
        "convective and other precipitating (column reaches 20 dBZ, not stratiform), non-precipitating (column never",
        "reaches 20 dBZ: anvils, ice above light rain).",
        "",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- m7 ice-cloud height (WACR)
def _centroid(occ: np.ndarray, h: np.ndarray, bottom_km: float = 6.0) -> float:
    sel = (h >= bottom_km) & np.isfinite(occ)
    return float(np.sum(occ[sel] * h[sel]) / np.sum(occ[sel])) if np.sum(occ[sel]) > 0 else np.nan


def make_ice_cloud(root: Path, dpi: int) -> None:
    from alaro_analysis.data.wacr_occurrence import MIN_HEIGHT_KM

    png, jpg, txt = supplementary_files(root, SUPPLEMENT_NUMBER, "m7_ice_cloud_height", "iop2")
    obs = load(CACHE / "wacr" / "WACR_T3.npz")
    if obs is None:
        print("[m7] WACR statistics not built", flush=True)
        return
    h = obs["profile_heights_km"]
    hours = obs["n"].sum(axis=0) >= 900
    occ_obs = 100.0 * obs["det"][0, hours].sum(axis=0) / obs["n"][0, hours].sum()
    occ_obs = np.where(h >= MIN_HEIGHT_KM, occ_obs, np.nan)
    n_days = np.unique(obs["hours"][hours] // 24).size
    zfrz = freezing_level_km(2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(occ_obs, h, color="black", lw=3.2)
    obs_centroid = _centroid(occ_obs, h)
    rows = [f"WACR,{int(hours.sum())}_hours,{int(obs['n'][0, hours].sum())}_profiles,centroid_km,{obs_centroid:.2f},"
            "occurrence_percent," + ",".join(f"{v:.3f}" for v in occ_obs)]
    markers = ("v", "o", "^")
    centroids = []
    for label, color in RUNS:
        s = load(CACHE / "wacr" / f"{label}_T3.npz")
        if s is None:
            continue
        q = s["q_thresholds"]
        occ = 100.0 * s["occ"][0] / s["n"][0]
        mid = len(q) // 2
        ax.plot(np.where(h >= MIN_HEIGHT_KM, occ[mid], np.nan), h, color=color, lw=2.2)
        cents = [_centroid(np.where(h >= MIN_HEIGHT_KM, o, np.nan), h) for o in occ]
        centroids.append((label, color, cents))
        for k, thr in enumerate(q):
            rows.append(f"{label},{int(s['n'][0])}_columns,threshold_kg_per_kg,{thr:g},centroid_km,{cents[k]:.2f},"
                        "occurrence_percent," + ",".join(f"{v:.3f}" for v in occ[k]))
        rows.append(f"{label},ice_only_ge{q[mid]:g},occurrence_percent,"
                    + ",".join(f"{v:.3f}" for v in 100.0 * s["occ_ice"][0] / s["n"][0]))
    if np.isfinite(zfrz):
        ax.axhline(zfrz, color="0.4", lw=1.2, ls=":")
    ax.set_ylim(MIN_HEIGHT_KM, 16)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(g_formatter())
    ax.set_xlabel("profiles with hydrometeors (%)")
    ax.set_ylabel("height (km)")
    ax.set_title("hydrometeor occurrence above T3")
    letter(ax, 0)
    ax = axes[1]
    x = np.arange(len(centroids))
    for i, (label, color, cents) in enumerate(centroids):
        for k, m in enumerate(markers):
            ax.plot(i + (k - 1) * 0.18, cents[k], marker=m, ms=12, color=color, mec="white", mew=1.2, ls="none")
    ax.axhline(obs_centroid, color="black", lw=3.2)
    ax.text(-0.5, obs_centroid + 0.04, "WACR", ha="left", va="bottom", fontsize=13)
    lows = [v for _, _, cents in centroids for v in cents if np.isfinite(v)]
    ax.set_ylim(min(lows + [obs_centroid]) - 0.3, max(lows + [obs_centroid]) + 0.4)
    ax.set_xticks(x, [c[0] for c in centroids])
    ax.set_xlim(-0.6, len(centroids) - 0.4)
    ax.set_ylabel("mean height of hydrometeors above 6 km (km)")
    ax.set_title("height of the ice-cloud layer")
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(g_formatter())
    letter(ax, 1)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12, wspace=0.28)
    handles = [Line2D([0], [0], color="black", lw=3.2, label="WACR")]
    handles += [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c, _ in centroids]
    handles += [Line2D([0], [0], marker=m, ms=11, color="0.4", ls="none", label=f"model threshold {t:g} kg/kg")
                for m, t in zip(markers, (1e-6, 1e-5, 1e-4))]
    handles.append(Line2D([0], [0], color="0.4", lw=1.2, ls=":", label=r"0 $^{\circ}$C level"))
    legend_below(fig, axes, handles, ncol=5)
    finish(fig, png, jpg, dpi)
    text = header("m7_ice_cloud_height", [
        "IOP2 only: ARM W-band cloud radar (WACR, 95 GHz, vertically pointing) at T3, Manacapuru, 31 Aug-30 Sep",
        "2014 (the copy on disk). Its hydrometeor mask (spectra recorded where the radar's cloud mask detected",
        "a target, every 2 s and 30 m) gives the share of profiles with a hydrometeor at each height (0.25 km",
        "bins). Below 4 km the mask also flags insects (50-80 % of profiles in the boundary layer): not used.",
        f"Hours: radar-valid hours with SIPAM scans and >= 900 WACR profiles ({int(hours.sum())} hours on {n_days} days);",
        "runs at their hourly snapshots in the 5x5 cells (20 km) around T3, same hours.",
        "Only profiles without rain above the site (SIPAM, or the model's own reflectivity, < 20 dBZ at 2 km in",
        "the 3x3 cells around the column): in rain the W-band beam is attenuated.",
        "Runs: a level counts when the total condensate (cloud water + ice + rain + snow + graupel) reaches the",
        "threshold (lines: 1e-5 kg/kg); markers in (b) for 1e-6, 1e-5, 1e-4 kg/kg. Above 6 km it is all ice.",
        "Mean height = occurrence-weighted mean height above 6 km. The WACR loses sensitivity with range, so",
        "its upper-level occurrence is if anything too low.",
        "",
    ]) + rows
    txt.write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--figures", nargs="+",
                   default=["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11"])
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_microphysics_story", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    makers = {"m1": make_cfad, "m2": make_occurrence, "m3": make_partition, "m4": make_dsd,
              "m5": make_hydrometeors, "m6": make_iwc, "m7": make_ice_cloud, "m8": make_water_budget,
              "m9": make_ice_location, "m10": make_boundary_layer, "m11": make_rain_production}
    for key in args.figures:
        makers[key](args.output_dir, args.dpi)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
