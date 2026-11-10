#!/usr/bin/env python3
"""Errors of the runs' diurnal cycle of rain against the radar and IMERG (supplementary).

Companion of the mean diurnal cycle figure, on the same common-valid hourly sets
(``build_iop_rainfall_common_valid``: radar valid-time mask and radar footprint on
every dataset, 0.1 deg cells, hours labelled at the start of the interval, local time
UTC-4). Three panels:

a) the runs' mean diurnal cycle minus the radar's, with the radar's 95 % sampling range
   (1.96 standard errors of its hourly mean over the days) and IMERG minus radar as the
   spread between the two observations;
b) the mean-square error of the mean diurnal cycle against the radar, split exactly into
   the offset of the means, the amplitude difference and the shape and timing remainder
   (``analysis.diurnal_errors.mse_decomposition``), RMSE on top (the same against IMERG
   is in the data file);
c) the shift of the time of maximum of the 24-h harmonic against the radar in every cell
   of the footprint, so a lag of the domain mean can be told from one of a few cells.
The harmonic amplitudes and times of the domain mean are in the data file.

Writes the figure per period (combined, iop1, iop2) with its data file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from alaro_analysis.analysis.diurnal_errors import (
    CENTRES,
    first_harmonic,
    mean_cycle,
    mse_decomposition,
    wrap_hours,
)
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.experiments import run_display, shown_runs
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import FIGURES, IOP_PERIODS, figure_files, write_figure_index
from alaro_analysis.recipes.iop.build_iop_rainfall_common_valid import DEFAULT_WORK_DIR
from alaro_analysis.recipes.iop.make_iop_rainfall_verification import (
    PERIOD_LABELS,
    PERIODS,
    UTC_OFFSET_HOURS,
    period_header,
)
from alaro_analysis.workflows import precip_diurnal_cycle as diurnal_wf

FIGURE_NAME = "diurnal_cycle_errors"
STYLE = {cfg.label: cfg for cfg in diurnal_wf.DATASETS}
OBS = ("Radar", "IMERG")
# Figure name -> file stem of every run shown
RUN_FILES = {run_display(stem): stem
             for stem in shown_runs(("C1M_nodiaggrp", "C1M", "G1M", "G2M", "G2M-XCU"))}
RUNS = tuple(RUN_FILES)
FILES = {"Radar": "Radar", "IMERG": "IMERG", **RUN_FILES}
# The three parts of the error, named by what they mean for the rain
PARTS = (("offset", "Too much or too little rain overall (offset)", "#ffd92f"),
         ("amplitude", "Afternoon peak too strong or too weak (amplitude)", "#66c2a5"),
         ("shape", "Rain at the wrong hours (timing and shape)", "#fc8d62"))
BAND = {"color": "deepskyblue", "alpha": 0.3}  # the radar's sampling range, as in the mean-cycle figure
TICK, AXIS, TITLE, LEGEND = 13, 15, 16, 14


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_diurnal_cycle_errors", argv=argv)


def load(data_dir: Path) -> dict:
    """Hourly rain of every dataset on the cells all of them have, and local hour/day."""
    rain, times = {}, None
    for label, stem in FILES.items():
        with xr.open_dataset(data_dir / f"{stem}_common_valid.nc") as ds:
            rain[label] = np.asarray(ds["rainfall"].values, dtype=np.float64)
            times = np.asarray(ds["time"].values, dtype="datetime64[h]")
    cells = np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in rain.values()])
    local = times + np.timedelta64(UTC_OFFSET_HOURS, "h")
    return {
        "cells": {k: v[:, cells] for k, v in rain.items()},
        "local_hour": local.astype("int64") % 24,
        "local_day": local.astype("datetime64[D]"),
    }


def diagnose(data: dict) -> dict:
    hours = data["local_hour"]
    domain = {k: v.mean(axis=1) for k, v in data["cells"].items()}
    cycles = {k: mean_cycle(v, hours) for k, v in domain.items()}
    # radar sampling error of each hourly mean: day-to-day spread / sqrt(days)
    se = np.array([domain["Radar"][hours == h].std(ddof=1) / np.sqrt((hours == h).sum())
                   for h in range(24)])
    cell_cycles = {k: mean_cycle(v, hours) for k, v in data["cells"].items()}
    cell_peak = {k: first_harmonic(c)[1] for k, c in cell_cycles.items()}
    out = {"cycles": cycles, "radar_se": se, "harmonic": {k: first_harmonic(c) for k, c in cycles.items()},
           "shift": {k: wrap_hours(cell_peak[k] - cell_peak["Radar"]) for k in ("IMERG",) + RUNS},
           "n_cells": data["cells"]["Radar"].shape[1], "n_hours": hours.size,
           "decomposition": {}}
    for reference in OBS:
        for k in ("IMERG",) + RUNS:
            if k != reference:
                out["decomposition"][(k, reference)] = mse_decomposition(cycles[k], cycles[reference])
    out["hourly_rmse"] = {k: float(np.sqrt(np.mean((domain[k] - domain["Radar"]) ** 2)))
                          for k in ("IMERG",) + RUNS}
    return out


def panel_differences(ax, d: dict) -> None:
    band = 1.96 * d["radar_se"]
    ax.fill_between(CENTRES, -band, band, lw=0, zorder=0, **BAND)
    ax.axhline(0.0, color="black", lw=1.0)
    for k in ("IMERG",) + RUNS:
        style = STYLE[k]
        ax.plot(CENTRES, d["cycles"][k] - d["cycles"]["Radar"], color=style.color,
                ls=style.linestyle, lw=2.4 if k == "IMERG" else 2.2)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Local time (UTC-4)", fontsize=AXIS)
    ax.set_ylabel("Difference (mm h$^{-1}$)", fontsize=AXIS)
    ax.set_title("Rain of each dataset minus the radar, by hour of the day", fontsize=TITLE)
    ax.grid(alpha=0.3)


def panel_decomposition(ax, d: dict) -> None:
    keys = ("IMERG",) + RUNS
    for x, k in enumerate(keys):
        parts = d["decomposition"][(k, "Radar")]
        bottom = 0.0
        for key, _, colour in PARTS:
            ax.bar(x, parts[key], 0.65, bottom=bottom, color=colour, edgecolor="white", lw=0.8)
            bottom += parts[key]
        ax.text(x, bottom, f"{parts['rmse']:.2f}", ha="center", va="bottom", fontsize=TICK)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=TICK, rotation=25, ha="right")
    ax.set_ylabel("Mean-square error vs radar (mm$^2$ h$^{-2}$)", fontsize=AXIS)
    ax.set_title("Error of the mean diurnal cycle", fontsize=TITLE)
    ax.text(0.02, 0.97, "number on each bar: RMSE (mm h$^{-1}$)", transform=ax.transAxes,
            fontsize=TICK - 1, va="top")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)


def panel_shift(ax, d: dict) -> None:
    keys = ("IMERG",) + RUNS
    values = [d["shift"][k] for k in keys]
    parts = ax.boxplot(values, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                       medianprops={"color": "black", "lw": 2})
    for patch, k in zip(parts["boxes"], keys):
        if k == "IMERG":  # hollow, as its dotted line in (a)
            patch.set_facecolor("white")
            patch.set_edgecolor(STYLE[k].color)
            patch.set_hatch("..")
        else:
            patch.set_facecolor(STYLE[k].color)
            patch.set_alpha(0.85)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(keys, fontsize=TICK, rotation=25, ha="right")
    ax.set_ylabel("Peak shift vs radar (h, + = later)", fontsize=AXIS)
    ax.set_title("Rain-peak timing in each 0.1\u00b0 cell", fontsize=TITLE)
    ax.grid(axis="y", alpha=0.3)


def write_text(path: Path, d: dict, header: list[str], jpg: Path) -> None:
    lines = [f"{path.stem}", "=" * len(path.stem), *header,
             f"Figure: {jpg}",
             f"{d['n_hours']} common-valid hours, {d['n_cells']} cells of 0.1 deg. Local time UTC-4; hour h is",
             "the interval [h, h+1). Domain mean over the cells first, then mean by local hour.",
             "Harmonic: 24-h harmonic fitted at the bin centres h+0.5; time of maximum in local hours.",
             "MSE decomposition: MSE = offset^2 + (sd_m - sd_o)^2 + 2 sd_m sd_o (1 - r) over the 24 hourly means.",
             "Cell shift: time of maximum of each cell's own cycle minus the radar's, wrapped to [-12, 12) h;",
             "boxes 25-75 %, whiskers 5-95 %.",
             "", "mean diurnal cycle (mm/h)",
             "local_hour," + ",".join(OBS + RUNS) + ",radar_95pct_half_range"]
    for h in range(24):
        lines.append(f"{h}," + ",".join(f"{d['cycles'][k][h]:.5f}" for k in OBS + RUNS)
                     + f",{1.96 * d['radar_se'][h]:.5f}")
    lines += ["", "decomposition of the mean-cycle MSE",
              "dataset,reference,rmse_mm_h,mse,offset,amplitude,shape_and_timing,correlation"]
    for (k, reference), p in d["decomposition"].items():
        lines.append(f"{k},{reference},{p['rmse']:.4f},{p['mse']:.6f},{p['offset']:.6f},"
                     f"{p['amplitude']:.6f},{p['shape']:.6f},{p['r']:.3f}")
    lines += ["", "24-h harmonic of the domain-mean cycle", "dataset,amplitude_mm_h,time_of_maximum_LT"]
    for k in OBS + RUNS:
        a, t = d["harmonic"][k]
        lines.append(f"{k},{float(a):.4f},{float(t):.2f}")
    lines += ["", "shift of the time of maximum against the radar, over the cells (h)",
              "dataset,p5,p25,median,p75,p95"]
    for k, v in d["shift"].items():
        q = np.percentile(v, [5, 25, 50, 75, 95])
        lines.append(f"{k}," + ",".join(f"{x:+.2f}" for x in q))
    lines += ["", "for scale: RMSE of every hourly domain mean against the radar (mm/h)",
              "dataset,rmse_all_hours"] + [f"{k},{v:.4f}" for k, v in d["hourly_rmse"].items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(d: dict, period: str, png: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(15.5, 12.0))
    grid = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.30)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    panel_differences(axes[0], d)
    panel_decomposition(axes[1], d)
    panel_shift(axes[2], d)
    for ax, letter in zip(axes, "abc"):
        ax.tick_params(labelsize=TICK)
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, fontsize=TITLE, fontweight="bold",
                va="bottom")
    handles = [plt.Line2D([], [], color=STYLE[k].color, ls=STYLE[k].linestyle, lw=2.6, label=k)
               for k in OBS + RUNS]
    handles += [Patch(label="Radar 95 % sampling range", **BAND)]
    handles += [Patch(facecolor=c, label=lab) for _, lab, c in PARTS]
    legend = legend_below(fig, np.array(axes), handles, ncol=3, fontsize=LEGEND,
                          title=PERIOD_LABELS[period])
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontsize(LEGEND + 1)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for period in args.periods:
        data_dir = args.work_dir / PERIODS[period]
        d = diagnose(load(data_dir))
        png, jpg, txt = figure_files(args.figure_root, FIGURE_NAME, FIGURE_NAME, period)
        make_figure(d, period, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_text(txt, d, period_header(data_dir, IOP_PERIODS[period]), jpg)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.figure_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
