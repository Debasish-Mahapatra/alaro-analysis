#!/usr/bin/env python3
"""Convective mass flux of the runs against the ARM radar wind profiler at T3.

The profiler side comes from ``build_iop_rwp_mass_flux``: Giangrande et al. (2016)
eq. (1), Mc = rho (sigma_u v_u + sigma_d v_d), over the 1-hour windows with at least
five minutes of convective echo, which stand for a 20 km domain at the 5 m s-1 cell
propagation speed of Amazon convection.

The model side comes from ``build_iop_updraft_partition``: the same quantity over
20 km boxes (5 x 5 cells of the 4 km radar domain) that hold a convective column, with
each column split into 3MT updraft, downdraft and environment before any averaging.

Both sides divide by the whole domain, so they are the same quantity. What differs is
the threshold that puts a sample in the ensemble: the profiler needs five minutes of
convective echo in the hour (at least 8 % of the window), the model one convective
column in the box (4 % of its area). The model's own convective coverage is written
next to the fluxes so the reader can see how much of any gap that accounts for.

Writes supplementary figure 5 and its data file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_COLORS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import (
    FIGURES,
    GOAMAZON_IOPS,
    supplementary_files,
    write_figure_index,
)
from alaro_analysis.recipes.iop.build_iop_closure_budgets import LABELS
from alaro_analysis.recipes.iop.build_iop_convection_diurnal import (
    DEFAULT_WORK_DIR as CONVECTION_WORK_DIR,
)
from alaro_analysis.recipes.iop.build_iop_rwp_mass_flux import (
    DEFAULT_WORK_DIR as RWP_WORK_DIR,
)
from alaro_analysis.recipes.iop.make_iop_closure_budget_figures import EXPERIMENTS
from alaro_analysis.recipes.iop.make_iop_updraft_partition_figures import box_hours, load_profile

SUPPLEMENT_NUMBER = 5
STEM = "convective_mass_flux_vs_profiler"
PERIOD = "iop1_iop2"
WINDOW_HOURS = 1  # the 20 km domain of the model boxes
Z_MAX_KM = 16.0
OBS_COLOR = "#111111"
# The catalogue keys the no-3MT run as "2mom-xcu"; the IOP recipes call it NO3M.
COLOR_KEY = {"control": "control", "graupel": "graupel", "2mom": "2mom", "NO3M": "2mom-xcu"}
TICK, AXIS, TITLE, LEGEND = 15, 17, 19, 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rwp-dir", type=Path, default=RWP_WORK_DIR)
    p.add_argument("--convection-dir", type=Path, default=CONVECTION_WORK_DIR)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_rwp_mass_flux_comparison", argv=argv)


def load_observed(rwp_dir: Path, iop: int) -> dict[str, np.ndarray]:
    with np.load(rwp_dir / f"rwp_mass_flux_iop{iop}_{WINDOW_HOURS}h.npz") as cache:
        return {key: np.asarray(cache[key], float) for key in cache.files}


def panel(ax, obs: dict, model: dict, title: str) -> None:
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.plot(obs["flux_up"], obs["height_km"], color=OBS_COLOR, lw=3.0, zorder=5)
    ax.plot(obs["flux_down"], obs["height_km"], color=OBS_COLOR, lw=3.0, ls="--", zorder=5)
    for experiment, (height, up, down) in model.items():
        colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
        ax.plot(up, height, color=colour, lw=2.0)
        ax.plot(-down, height, color=colour, lw=2.0, ls="--")
    ax.set_ylim(0, Z_MAX_KM)
    ax.set_title(title, fontsize=TITLE)
    ax.set_xlabel("Mass flux (kg m$^{-2}$ s$^{-1}$)", fontsize=AXIS)
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.25)


def gather(args) -> dict:
    out = {}
    for iop in (1, 2):
        obs = load_observed(args.rwp_dir, iop)
        model = {}
        for experiment in EXPERIMENTS:
            label = LABELS[experiment]
            height, up = load_profile(args.convection_dir, (iop,), label, "box_up")
            _, down = load_profile(args.convection_dir, (iop,), label, "box_down")
            model[experiment] = (height, up, down)
        out[iop] = {"obs": obs, "model": model,
                    "box_hours": {e: box_hours(args.convection_dir, iop, LABELS[e])
                                  for e in EXPERIMENTS}}
    return out


def peak(height: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(values) & (height <= Z_MAX_KM)
    if not ok.any():
        return float("nan"), float("nan")
    index = np.nanargmax(np.where(ok, values, -np.inf))
    return float(values[index]), float(height[index])


def write_text(data: dict, path: Path) -> None:
    lines = [
        STEM, "=" * len(STEM),
        "Convective mass flux (kg m-2 s-1) over a 20 km domain, upward and downward,",
        "both positive. Observations: ARM 1290 MHz radar wind profiler at T3, PI product",
        "doi:10.5439/1440997, by the method of Giangrande et al. 2016 (JGR 121, 12891),",
        "their eq. (1): Mc = rho (sigma_u v_u + sigma_d v_d), where sigma is the share of",
        "the whole window covered by convective updraft or downdraft and v the mean",
        "velocity there. One hour of profiles stands for 20 km at the 5 m s-1 cell",
        "propagation speed. A window enters the ensemble with five minutes of convective",
        "echo. No velocity is retrieved in the melting layer, so 4-5 km is blank.",
        "Model: the same quantity over the 20 km boxes of the 4 km radar domain that hold",
        "a convective column (rain >= 1 mm/h and, with 3MT, an updraft), each column split",
        "into updraft, downdraft and environment before averaging, divided by the full box.",
        "",
    ]
    for iop in (1, 2):
        first, last = GOAMAZON_IOPS[iop]
        obs = data[iop]["obs"]
        lines += [
            f"IOP{iop}: {first} to {last} inclusive.",
            f"  profiler: {int(obs['n_days'][0])} days with convection, "
            f"{int(obs['n_windows'][0])} hourly windows in the ensemble.",
        ]
        for experiment in EXPERIMENTS:
            conv, total = data[iop]["box_hours"][experiment]
            lines.append(f"  {LABELS[experiment]:<8} {conv} convective box-hours of {total} "
                         f"({conv / max(total, 1):.3f})")
        lines.append("")
    lines += ["iop,dataset,quantity,peak,peak_height_km,value_2km,value_6km,value_10km"]
    for iop in (1, 2):
        obs = data[iop]["obs"]
        rows = [("profiler", obs["height_km"], obs["flux_up"], -obs["flux_down"])]
        for experiment, (height, up, down) in data[iop]["model"].items():
            rows.append((LABELS[experiment], height, up, down))
        for name, height, up, down in rows:
            for quantity, values in (("up", up), ("down", down)):
                value, z = peak(height, values)
                at = [float(np.interp(target, height, np.nan_to_num(values, nan=0.0)))
                      for target in (2.0, 6.0, 10.0)]
                lines.append(f"IOP{iop},{name},{quantity},{value:.4f},{z:.2f},"
                             + ",".join(f"{v:.4f}" for v in at))
    lines.append("")
    lines += ["height_km," + ",".join(
        f"IOP{iop}_{name}_{q}" for iop in (1, 2)
        for name in ["profiler"] + [LABELS[e] for e in EXPERIMENTS] for q in ("up", "down"))]
    grid = np.arange(0.25, Z_MAX_KM + 0.01, 0.25)
    columns = []
    for iop in (1, 2):
        obs = data[iop]["obs"]
        series = [(obs["height_km"], obs["flux_up"], -obs["flux_down"])]
        series += [(h, u, d) for h, u, d in data[iop]["model"].values()]
        for height, up, down in series:
            for values in (up, down):
                columns.append(np.interp(grid, height, values, left=np.nan, right=np.nan))
    for row, z in enumerate(grid):
        lines.append(f"{z:.2f}," + ",".join(f"{c[row]:.5f}" for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = gather(args)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.6), sharey=True)
    for ax, iop in zip(axes, (1, 2)):
        panel(ax, data[iop]["obs"], data[iop]["model"], f"IOP{iop}")
    axes[0].set_ylabel("Height (km)", fontsize=AXIS)
    handles = [plt.Line2D([], [], color=OBS_COLOR, lw=3.0, label="Radar wind profiler")]
    handles += [plt.Line2D([], [], color=EXPERIMENT_COLORS[COLOR_KEY[e]], lw=2.0, label=LABELS[e])
                for e in EXPERIMENTS]
    handles += [plt.Line2D([], [], color="0.35", lw=2.0, label="upward"),
                plt.Line2D([], [], color="0.35", lw=2.0, ls="--", label="downward")]
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.17, wspace=0.08)
    legend_below(fig, axes, handles, ncol=4, fontsize=LEGEND)
    png, jpg, txt = supplementary_files(args.figure_root, SUPPLEMENT_NUMBER, STEM, PERIOD)
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    convert_to_jpg(png, jpg, dpi=args.dpi)
    png.unlink()
    write_text(data, txt)
    write_figure_index(args.figure_root)
    print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
