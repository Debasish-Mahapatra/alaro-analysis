#!/usr/bin/env python3
"""Convective mass flux of the runs against the ARM radar wind profiler at T3.

The profiler side comes from ``build_iop_rwp_mass_flux``: Giangrande et al. (2016)
eq. (1), Mc = rho (sigma_u v_u + sigma_d v_d), over the 1-hour windows with at least
five minutes of convective echo, which stand for a 20 km domain at the 5 m s-1 cell
propagation speed of Amazon convection.

The model side comes from ``build_iop_updraft_partition``: the same quantity over
20 km boxes (5 x 5 cells of the 4 km radar domain) that hold a convective column, with
each column split into 3MT updraft, downdraft and environment before any averaging.

Both sides divide by the whole domain, so they are the same quantity, and both are split
the way the paper's figure 7 splits them: the area fraction sigma of rising (sinking)
convective air, the mass flux Mc, and the velocity v = Mc / (rho sigma) that links them.
The model's velocity uses the profiler's own density profile (the interpolated sounding
in its files), so the two velocities differ only through Mc and sigma.

The threshold that puts a sample in the ensemble differs (five minutes of convective echo
in the profiler's hour, one convective column in a model box), so the convective share of
the window or box is written next to the profiles: if the model's ensemble were the less
convective one, its smaller flux could be a sampling artefact.

Writes supplementary figure 5 (3 rows x 2 IOPs) and its data file.
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


def interp_rho(obs: dict):
    """Density on any height (km), from the profiler's interpolated sounding."""
    ok = np.isfinite(obs["rho"])
    return lambda z: np.interp(z, obs["height_km"][ok], obs["rho"][ok])


# Row -> (observed up, observed down, model up, model down, sign of down, x label)
ROWS = (
    ("v_up_eff", "v_down_eff", "v_up", "v_down", -1.0, "Velocity (m s$^{-1}$)"),
    ("sigma_up", "sigma_down", "sigma_up", "sigma_down", -1.0, "Area fraction"),
    ("flux_up", "flux_down", "flux_up", "flux_down", -1.0, "Mass flux (kg m$^{-2}$ s$^{-1}$)"),
)


def panel(ax, obs: dict, model: dict, row: tuple, title: str | None) -> None:
    o_up, o_down, m_up, m_down, sign, xlabel = row
    ax.axvline(0.0, color="0.6", lw=0.8)
    # The profiler's sinking quantities are stored with their sign (velocity, flux) or
    # positive (area); draw every sinking quantity on the negative side.
    obs_down = -np.abs(obs[o_down])
    ax.plot(obs[o_up], obs["height_km"], color=OBS_COLOR, lw=3.0, zorder=5)
    ax.plot(obs_down, obs["height_km"], color=OBS_COLOR, lw=3.0, ls="--", zorder=5)
    for experiment, entry in model.items():
        colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
        ax.plot(entry[m_up], entry["height_km"], color=colour, lw=2.0)
        ax.plot(sign * np.abs(entry[m_down]), entry["height_km"], color=colour, lw=2.0, ls="--")
    ax.set_ylim(0, Z_MAX_KM)
    if title:
        ax.set_title(title, fontsize=TITLE)
    ax.set_xlabel(xlabel, fontsize=AXIS)
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
            _, area_up = load_profile(args.convection_dir, (iop,), label, "box_up_area")
            _, area_down = load_profile(args.convection_dir, (iop,), label, "box_down_area")
            _, conv = load_profile(args.convection_dir, (iop,), label, "box_conv_frac")
            model[experiment] = {"height_km": height, "flux_up": up, "flux_down": down,
                                 "sigma_up": area_up, "sigma_down": area_down,
                                 "conv_share": float(np.nanmean(conv))}
        rho = interp_rho(obs)
        for entry in model.values():
            r = rho(entry["height_km"])
            with np.errstate(invalid="ignore", divide="ignore"):
                entry["v_up"] = entry["flux_up"] / (r * entry["sigma_up"])
                entry["v_down"] = entry["flux_down"] / (r * entry["sigma_down"])
        with np.errstate(invalid="ignore", divide="ignore"):
            obs["v_up_eff"] = obs["flux_up"] / (obs["rho"] * obs["sigma_up"])
            obs["v_down_eff"] = -obs["flux_down"] / (obs["rho"] * obs["sigma_down"])
        out[iop] = {"obs": obs, "model": model,
                    "box_hours": {e: box_hours(args.convection_dir, iop, LABELS[e])
                                  for e in EXPERIMENTS}}
    return out


def write_text(data: dict, path: Path) -> None:
    lines = [
        STEM, "=" * len(STEM),
        "Convective vertical velocity, area fraction and mass flux over a 20 km domain.",
        "Observations: ARM 1290 MHz radar wind profiler at T3, PI product doi:10.5439/1440997,",
        "by the method of Giangrande et al. 2016 (JGR 121, 12891), their eq. (1):",
        "Mc = rho (sigma_u v_u + sigma_d v_d), sigma = share of the whole window covered by",
        "convective rising (sinking) air, v = mean velocity there. One hour of profiles",
        "stands for 20 km at 5 m/s cell propagation; a window enters the ensemble with five",
        "minutes of convective echo. No velocity is retrieved in the melting layer (4-5 km).",
        "Model: the same over the 20 km boxes of the 4 km radar domain that hold a convective",
        "column (rain >= 1 mm/h and, with 3MT, an updraft), each column split into 3MT",
        "updraft, downdraft and environment before averaging, divided by the full box.",
        "Velocity on both sides = Mc / (rho sigma), with rho from the profiler's sounding.",
        "",
    ]
    for iop in (1, 2):
        first, last = GOAMAZON_IOPS[iop]
        obs = data[iop]["obs"]
        lines += [
            f"IOP{iop}: {first} to {last} inclusive.",
            f"  profiler: {int(obs['n_days'][0])} days with convection, "
            f"{int(obs['n_windows'][0])} hourly windows; convective share of the window "
            f"{obs['convective_column_fraction'][0]:.3f}",
        ]
        for experiment, entry in data[iop]["model"].items():
            conv, total = data[iop]["box_hours"][experiment]
            lines.append(f"  {LABELS[experiment]:<8} {conv} convective box-hours of {total}; "
                         f"convective share of the box {entry['conv_share']:.3f}")
        lines.append("")
    heights = (2.0, 3.0, 6.0, 8.0, 10.0)
    lines += ["iop,dataset,quantity," + ",".join(f"z{z:g}km" for z in heights)]
    for iop in (1, 2):
        obs = data[iop]["obs"]
        rows = [("profiler", obs["height_km"], {"sigma_up": obs["sigma_up"],
                 "v_up": obs["v_up_eff"], "flux_up": obs["flux_up"],
                 "sigma_down": obs["sigma_down"], "v_down": -obs["v_down_eff"],
                 "flux_down": -obs["flux_down"]})]
        rows += [(LABELS[e], m["height_km"], m) for e, m in data[iop]["model"].items()]
        for name, height, entry in rows:
            for quantity in ("sigma_up", "v_up", "flux_up", "sigma_down", "v_down", "flux_down"):
                values = np.asarray(entry[quantity], float)
                ok = np.isfinite(values)
                at = [float(np.interp(z, height[ok], values[ok])) if ok.any() else float("nan")
                      for z in heights]
                lines.append(f"IOP{iop},{name},{quantity}," + ",".join(f"{v:.4f}" for v in at))
    lines.append("")
    lines.append("Sinking quantities are positive (area, downward velocity, downward flux).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = gather(args)
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 19.0), sharey=True)
    for r, row in enumerate(ROWS):
        for c, iop in enumerate((1, 2)):
            panel(axes[r, c], data[iop]["obs"], data[iop]["model"], row,
                  f"IOP{iop}" if r == 0 else None)
        axes[r, 0].set_ylabel("Height (km)", fontsize=AXIS)
        # same x range in both IOPs, so the columns compare by eye
        lo = min(axes[r, 0].get_xlim()[0], axes[r, 1].get_xlim()[0])
        hi = max(axes[r, 0].get_xlim()[1], axes[r, 1].get_xlim()[1])
        for ax in axes[r]:
            ax.set_xlim(lo, hi)
    for ax, letter in zip(axes.flat, "abcdef"):
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, va="top",
                fontsize=TITLE, fontweight="bold")
    handles = [plt.Line2D([], [], color=OBS_COLOR, lw=3.0, label="Radar wind profiler")]
    handles += [plt.Line2D([], [], color=EXPERIMENT_COLORS[COLOR_KEY[e]], lw=2.0, label=LABELS[e])
                for e in EXPERIMENTS]
    handles += [plt.Line2D([], [], color="0.35", lw=2.0, label="upward"),
                plt.Line2D([], [], color="0.35", lw=2.0, ls="--", label="downward")]
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.08, wspace=0.08, hspace=0.22)
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
