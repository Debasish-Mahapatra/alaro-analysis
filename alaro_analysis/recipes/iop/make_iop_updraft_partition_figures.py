#!/usr/bin/env python3
"""Closure budget of IOP1 against IOP2 with the convective upward mass flux.

Supplementary figure S03 (``iop_settings.supplementary_files``): rows IOP1 and IOP2,
one column per run (C1M, G1M, G2M, G2M-XCU), 24 h means. Condensation is the DDH QV
budget of figure 11 (``build_iop_closure_budgets``). The mass-flux axis carries the
convective profiles of ``build_iop_updraft_partition``, sampled as the radar wind
profiler samples (Giangrande et al. 2016): convective columns only, 20 km boxes
divided by their full area, mean over the box-hours with convection.

    total upward       each column split into 3MT updraft, downdraft and environment
                       (updraft velocity relative, downdraft absolute, grid mean =
                       area mean of the three); rising part of each piece x its area
    3MT updraft        the updraft piece alone
    resolved ascent    rising grid-mean motion alone, no split

The total is not the sum of the other two: the grid-mean motion already contains the
updraft, and the updraft is compensated by sinking around it. G2M-XCU has no 3MT, so
its total is its resolved ascent. The downward flux is in the data text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    supplementary_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_closure_budgets import DEFAULT_WORK_DIR, LABELS
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR as CONVECTION_WORK_DIR,
)
from alaro_analysis.recipes.iop.make_iop_closure_budget_figures import (
    EXPERIMENTS,
    UPDRAFT_COLOR,
    Z_MAX_KM,
    column_integral,
    draw_rows,
    freezing_level_km,
    load_condensation,
    peak_height_km,
)

SUPPLEMENT_NUMBER = 3
STEM = "closure_budget_updraft_partition"
PERIOD = "iop1_iop2"
# Total in the updraft green; the two single pieces solid in their own colours, so
# they cannot be mistaken for the dashed red and blue condensation parts.
LINES = {
    "box_up": {"lw": 2.8, "ls": "-"},
    "box_par_up": {"color": "#E69F00", "lw": 2.0, "ls": "-"},
    "box_res_up": {"color": "#8C510A", "lw": 2.0, "ls": "-"},
}
LINE_LABELS = {
    "box_up": "total upward mass flux",
    "box_par_up": "3MT updraft",
    "box_res_up": "resolved ascent",
}
TEXT_QUANTITIES = ("box_up", "box_up_nodd", "box_down", "box_par_up", "box_res_up")


def box_hours(convection_work: Path, iop: int, label: str) -> tuple[int, int]:
    """(convective box-hours, all box-hours) of one run and IOP."""
    with np.load(convection_work / f"iop{iop}" / f"{label}_box_frequency.npz") as cache:
        mean, counts = np.asarray(cache["mean"], float)[0], np.asarray(cache["counts"], float)[0]
    return int(round(np.nansum(mean * counts))), int(counts.sum())


def load_profile(convection_work: Path, iops: tuple[int, ...], label: str, quantity: str):
    """Hour-count-weighted mean profile (kg m-2 s-1) over all local hours, and height (km)."""
    total = weight = height = None
    for iop in iops:
        with np.load(convection_work / f"iop{iop}" / f"{label}_{quantity}.npz") as cache:
            mean = np.asarray(cache["mean"], float)
            counts = np.asarray(cache["counts"], float)
        valid = np.isfinite(mean) & (counts > 0)
        part, n = np.where(valid, mean * counts, 0.0), np.where(valid, counts, 0.0)
        total = part if total is None else total + part
        weight = n if weight is None else weight + n
        with np.load(convection_work / f"iop{iop}" / f"{label}_height.npz") as cache:
            height = np.asarray(cache["height_m"], float) / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(weight.sum(axis=1) > 0, total.sum(axis=1) / weight.sum(axis=1), np.nan)
    return height, profile


def gather(work: Path, convection_work: Path, iop: int) -> dict:
    panels = {}
    for experiment in EXPERIMENTS:
        data = load_condensation(work, (iop,), experiment)
        data["z_peak"] = peak_height_km(data["altitude_km"], data["total"])
        data["z_freeze"] = freezing_level_km(data["altitude_km"], data["temperature_k"])
        data["column"] = column_integral(data["altitude_km"], data["total"])
        label = LABELS[experiment]
        flux = {q: load_profile(convection_work, (iop,), label, q) for q in TEXT_QUANTITIES}
        data["box_hours"] = box_hours(convection_work, iop, label)
        data["flux"] = flux
        data["updraft"] = None
        # parts first, total last, so the total stays visible where they coincide (G2M-XCU)
        data["updraft_lines"] = [(flux[key][0], flux[key][1], LINES[key]) for key in reversed(LINES)]
        panels[experiment] = data
    return panels


def flux_summary(height_km: np.ndarray, flux: np.ndarray) -> tuple[float, float, float]:
    """Peak flux, its height and the 0-Z_MAX_KM column mean of one profile."""
    ok = np.isfinite(flux) & (height_km <= Z_MAX_KM)
    if not ok.any() or not np.nanmax(flux[ok]) > 0:
        return 0.0, float("nan"), 0.0
    i = int(np.nanargmax(np.where(ok, flux, -np.inf)))
    return float(flux[i]), float(height_km[i]), column_integral(height_km, flux) / Z_MAX_KM


def write_txt(txt: Path, rows: dict, jpg: Path) -> None:
    lines = [f"{STEM}_{PERIOD}", "=" * len(f"{STEM}_{PERIOD}"), f"Figure: {jpg}"]
    for iop in GOAMAZON_IOPS:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(f"IOP{iop}: {start} to {end} inclusive, 24 h means.")
    lines += [
        "Condensation: DDH QV budget (g kg-1 day-1), as figure 11, over the DDH domain (all columns).",
        "Mass flux (kg m-2 s-1), sampled like the radar wind profiler (Giangrande et al. 2016):",
        "  - instantaneous hourly snapshots (leads +0001..+0024), split per column and level before "
        "any averaging;",
        "  - column = 3MT updraft (UD_MESH_FRAC, UD_OMEGA = velocity relative to its environment, "
        "ACCVUD / Gerard et al. 2009 eq. 4),",
        "    downdraft (DD_MESH_FRAC, DD_OMEGA = absolute velocity, ACMODO, inside the updraft's "
        "environment) and the rest;",
        "    VITESSE_VERT = area mean of the three. up = rising part of each piece x its area; "
        "down = sinking part (up - down = -omega/g);",
        "  - convective columns only: rain >= 1 mm in the hour ending at the snapshot and, with 3MT, "
        "an updraft (mesh > 1e-3 at some level);",
        "  - 20 km boxes (5 x 5 cells), 116 boxes at least half inside the 4 km radar domain, each "
        "divided by its full area;",
        "  - mean over the box-hours with at least one convective column.",
        "up_nodd = up without the downdraft split; par_up = 3MT updraft piece alone; res_up = rising "
        "grid-mean motion alone (no split).",
        "Figure lines: up (total), par_up (3MT updraft), res_up (resolved ascent). "
        "Column = mean over 0-10 km.",
        "",
        "iop,run,convective_box_hours,all_box_hours,convective_fraction",
    ]
    for row_label, panels in rows.items():
        for experiment, data in panels.items():
            conv, total = data["box_hours"]
            lines.append(f"{row_label},{LABELS[experiment]},{conv},{total},{conv / max(total, 1):.4f}")
    lines += ["", "iop,run,quantity,peak_flux,peak_height_km,column_mean_flux"]
    names = {"box_up": "up", "box_up_nodd": "up_nodd", "box_down": "down",
             "box_par_up": "par_up", "box_res_up": "res_up"}
    for row_label, panels in rows.items():
        for experiment, data in panels.items():
            for key in TEXT_QUANTITIES:
                peak, z, column = flux_summary(*data["flux"][key])
                lines.append(f"{row_label},{LABELS[experiment]},{names[key]},{peak:.5f},{z:.2f},{column:.5f}")
    lines += ["", "iop,run,column_condensation_g_kg_day_km,peak_height_km,zero_degC_km"]
    for row_label, panels in rows.items():
        for experiment, data in panels.items():
            lines.append(f"{row_label},{LABELS[experiment]},{data['column']:.3f},{data['z_peak']:.2f},"
                         f"{data['z_freeze']:.2f}")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--convection-work-dir", type=Path, default=CONVECTION_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_updraft_partition_figures", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = {f"IOP{iop}": gather(args.work_dir, args.convection_work_dir, iop) for iop in GOAMAZON_IOPS}
    png, jpg, txt = supplementary_files(args.output_dir, SUPPLEMENT_NUMBER, STEM, PERIOD)
    handles = [Line2D([0], [0], label=LINE_LABELS[k], **{"color": UPDRAFT_COLOR, **LINES[k]})
               for k in LINES]
    draw_rows(rows, png, args.dpi, updraft_handles=handles)
    convert_to_jpg(png, jpg, dpi=args.dpi)
    png.unlink()
    write_txt(txt, rows, jpg)
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
