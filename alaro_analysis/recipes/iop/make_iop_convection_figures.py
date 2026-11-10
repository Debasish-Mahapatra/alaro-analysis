#!/usr/bin/env python3
"""Updraft and downdraft figures for the GoAmazon IOPs, from the cached diurnal sections.

One figure per period, for C1M_nodiaggrp, C1M_diaggrp, G1M and G2M (G2M-XCU has no parameterized updraft
or downdraft, so it is left out): the supplementary updraft and downdraft profiles, six
panels of all-hours mean profiles per experiment (updraft and downdraft extent, mass
flux and resolved velocity).

The diurnal height sections (IOP figures 6 and 7) are drawn by
``make_iop_draft_sections``, with the column-split convective mass flux and all four
runs.

Sections come from ``build_iop_convection_diurnal``; periods are IOP1+IOP2
combined (count weighted), IOP1 and IOP2. Every hour of the campaign days counts,
so there is no observational time mask here; the file counts are in the data text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_DISPLAY, EXPERIMENT_LABELS
from alaro_analysis.common.experiments import shown_runs
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    PLOT_TOP_KM,
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR,
    QUANTITIES,
)

EXPS = shown_runs(("control-nodiag", "control", "graupel", "2mom"))
FIGURE_NAMES = {"profiles": "updraft_downdraft_profiles"}
FIGURE_STEMS = {"profiles": "updraft_downdraft_profiles"}
LABELS = {
    "ud_extent": ("Updraft extent", "Fraction", True),
    "ud_flux": ("Updraft mass flux", "kg m$^{-2}$ s$^{-1}$", False),
    "w_up": ("Resolved updraft velocity", "m s$^{-1}$", False),
    "dd_extent": ("Downdraft extent", "Fraction", True),
    "dd_flux": ("Downdraft mass flux", "kg m$^{-2}$ s$^{-1}$", False),
    "w_dn": ("Resolved downdraft velocity", "m s$^{-1}$", False),
    "cloud": ("Cloud cover", "Fraction", True),
}
PLAIN_UNITS = {
    "Fraction": "fraction",
    "kg m$^{-2}$ s$^{-1}$": "kg m-2 s-1",
    "m s$^{-1}$": "m s-1",
}
PROFILE_PANELS = ("ud_extent", "ud_flux", "w_up", "dd_extent", "dd_flux", "w_dn")
MAX_HEIGHT_KM = 16.0
REF_FRACTION = 0.05  # mask the relative bias where the reference is below 5 % of its p98
TICK_FS, AXIS_FS, HEADER_FS, LETTER_FS = 19, 21, 18, 16
CBAR_LABEL_FS, CBAR_TICK_FS, LEGEND_FS = 19, 16, 14


def merge_sections(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Count-weighted mean of one quantity over several IOPs.

    Each part is ``(mean, counts)`` on the same (level, hour) grid; cells no IOP
    sampled come back NaN.
    """
    total = None
    weight = None
    for mean, counts in parts:
        mean = np.asarray(mean, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.float64)
        if total is None:
            total = np.zeros_like(mean)
            weight = np.zeros_like(mean)
        valid = np.isfinite(mean) & (counts > 0)
        total[valid] += mean[valid] * counts[valid]
        weight[valid] += counts[valid]
    out = np.full_like(total, np.nan)
    seen = weight > 0
    out[seen] = total[seen] / weight[seen]
    return out


def load_period(work: Path, iops: tuple[int, ...]) -> dict:
    """Cached sections of every experiment for one period, cropped to MAX_HEIGHT_KM."""
    data: dict = {"sections": {}, "files": {}}
    for exp in EXPS:
        label = EXPERIMENT_LABELS[exp]
        sections, files = {}, 0
        for quantity in QUANTITIES:
            parts = []
            for iop in iops:
                with np.load(work / f"iop{iop}" / f"{label}_{quantity}.npz") as cache:
                    parts.append((cache["mean"], cache["counts"]))
                    files = max(files, int(cache["n_files"]))
            sections[quantity] = merge_sections(parts)
        with np.load(work / f"iop{iops[0]}" / f"{label}_height.npz") as cache:
            height_km = np.asarray(cache["height_m"], dtype=np.float64) / 1000.0
        data["sections"][exp] = sections
        data["files"][exp] = files * len(iops)
        data.setdefault("height_km", height_km)

    keep = np.isfinite(data["height_km"]) & (data["height_km"] <= MAX_HEIGHT_KM)
    data["height_km"] = data["height_km"][keep]
    for exp in EXPS:
        data["sections"][exp] = {k: v[keep, :] for k, v in data["sections"][exp].items()}
    return data


def relative_bias(numerator: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """100 * (model - ref) / ref, masked where the reference is negligibly small."""
    ref = np.asarray(reference, dtype=np.float64)
    positive = ref[np.isfinite(ref) & (ref > 0.0)]
    threshold = REF_FRACTION * float(np.percentile(positive, 98.0)) if positive.size else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * numerator / np.where(ref != 0.0, ref, np.nan)
    return np.where(np.isfinite(ref) & (ref > threshold), rel, np.nan)


def hour_mean(section: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(section, axis=1)


def freezing_km(temp_section: np.ndarray, height_km: np.ndarray) -> float:
    """Height where the all-hours mean temperature crosses 0 degC."""
    profile = hour_mean(temp_section) - 273.15
    finite = np.isfinite(profile) & np.isfinite(height_km)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    z, t = height_km[finite], profile[finite]
    crossings = np.flatnonzero((t[:-1] > 0.0) & (t[1:] <= 0.0))
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[0])
    return float(z[i] + (0.0 - t[i]) * (z[i + 1] - z[i]) / (t[i + 1] - t[i]))


def figure_outputs(root: Path, kind: str, period: str) -> tuple[Path, Path, Path]:
    return figure_files(root, FIGURE_NAMES[kind], FIGURE_STEMS[kind], period)


def period_header(data: dict, iops: tuple[int, ...]) -> list[str]:
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785)."]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, {(end - start).days + 1} days.")
    files = ", ".join(f"{EXPERIMENT_DISPLAY[e]} {data['files'][e]}" for e in EXPS)
    lines += [
        f"Hourly files reduced: {files}. Every hour of the campaign days is used; "
        "no observational time mask applies here.",
        "Domain mean over the SIPAM radar domain on the native 4 km grid (2960 cells, "
        "ALARO-RUNS/mask/Radar_mask_4km_native.nc); every cell counts, with or without a draft. "
        "Hour is Amazon local time (UTC-4).",
        "All runs from raw FA: C1M from the continuous control run, G1M and G2M from the IOP "
        "warm-restart runs.",
        "G2M-XCU is not shown: it has no parameterized updraft or downdraft.",
        "Draft fields cleaned as the 3MT physics does on entry (advection overshoots): mesh "
        "fraction < 0 -> 0, UD_OMEGA > 0 -> 0, DD_OMEGA < 0 -> 0.",
        "Updraft mass flux = mean(-min(UD_OMEGA, 0) * max(mesh, 0) / g), positive upward.",
        "Downdraft mass flux = mean(max(DD_OMEGA, 0) * max(mesh, 0) / g), positive downward.",
        "Resolved velocities from VERT.VELOCIT: w_up = mean(max(w, 0)), w_dn = mean(max(-w, 0)).",
    ]
    return lines


def draw_profiles(data: dict, png: Path, dpi: int) -> dict[str, dict[str, np.ndarray]]:
    """Six panels of all-hours mean profiles, one curve per experiment."""
    height = data["height_km"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.0), sharey=True)
    curves: dict[str, dict[str, np.ndarray]] = {}
    freezing = {exp: freezing_km(data["sections"][exp]["temp"], height) for exp in EXPS}

    for index, quantity in enumerate(PROFILE_PANELS):
        ax = axes[index // 3, index % 3]
        title, unit, _ = LABELS[quantity]
        curves[quantity] = {}
        for exp in EXPS:
            profile = hour_mean(data["sections"][exp][quantity])
            curves[quantity][exp] = profile
            ax.plot(
                profile,
                height,
                color=EXPERIMENT_COLORS[exp],
                lw=2.0,
                label=EXPERIMENT_DISPLAY[exp],
            )
        for exp in EXPS:
            if np.isfinite(freezing[exp]):
                ax.axhline(freezing[exp], color=EXPERIMENT_COLORS[exp], lw=1.1, ls=":", alpha=0.9)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel(unit, fontsize=13)
        ax.set_ylim(0.0, PLOT_TOP_KM)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.tick_params(axis="both", labelsize=12)
        # Small-valued panels otherwise crowd their tick labels into each other.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(
            0.0,
            1.015,
            f"({'abcdef'[index]})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    for row in range(2):
        axes[row, 0].set_ylabel("Height above mean sea level (km)", fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = list(handles) + [plt.Line2D([0], [0], color="0.4", lw=1.1, ls=":")]
    labels = list(labels) + ["0 °C height"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.11, wspace=0.16, hspace=0.26)
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    curves["freezing"] = freezing
    return curves


def write_profile_txt(txt: Path, data: dict, curves: dict, header: list[str]) -> None:
    height = data["height_km"]
    title = "Updraft and downdraft profiles"
    lines = [title, "=" * len(title), *header, "", "All-hours mean profiles.", ""]
    lines += ["0 degC height (km)", "------------------", "series,z_freeze_km"]
    lines += [
        f"{EXPERIMENT_DISPLAY[e]},{curves['freezing'][e]:.6g}"
        if np.isfinite(curves["freezing"][e])
        else f"{EXPERIMENT_DISPLAY[e]},nan"
        for e in EXPS
    ]
    for quantity in PROFILE_PANELS:
        name, unit, _ = LABELS[quantity]
        heading = f"{name} [{PLAIN_UNITS[unit]}]"
        lines += ["", heading, "-" * len(heading)]
        lines.append(",".join(["height_km", *[EXPERIMENT_DISPLAY[e] for e in EXPS]]))
        for i, z in enumerate(height):
            row = [f"{z:.4f}"]
            row += [
                "nan" if not np.isfinite(curves[quantity][e][i]) else f"{curves[quantity][e][i]:.6g}"
                for e in EXPS
            ]
            lines.append(",".join(row))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_convection_figures", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = load_period(args.work_dir, iops)
        header = period_header(data, iops)
        root = args.output_dir

        png, jpg, txt = figure_outputs(root, "profiles", key)
        curves = draw_profiles(data, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_profile_txt(txt, data, curves, header)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)

    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
