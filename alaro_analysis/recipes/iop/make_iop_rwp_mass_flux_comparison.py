#!/usr/bin/env python3
"""Convective mass flux of the runs against the ARM radar wind profiler at T3.

The profiler side comes from ``build_iop_rwp_mass_flux``: Giangrande et al. (2016)
eq. (1), Mc = rho (sigma_u v_u + sigma_d v_d), over the windows with at least five
minutes of convective echo. At the 5 m s-1 cell propagation speed of Amazon convection a
1 h window stands for a 20 km domain and a 3 h window for 60 km.

The model side comes from ``build_iop_rwp_model_sample``: the same quantity over the
boxes of the 4 km radar domain that hold a convective column, each column split into 3MT
updraft, downdraft and environment before any averaging, divided by the full box, at the
hours the profiler observed and only at levels with model echo (>= 10 dBZ), since the
profiler sees nothing outside echo. The two scales of the supervisor's note:

* ``20km`` (IOP figure): 20 km boxes, one snapshot, against the 1 h windows;
* ``60km`` (supplementary): 60 km boxes over three consecutive hourly snapshots, against
  the 3 h windows (``--box-km 60`` of the model sample); a box-window counts when convective
  columns cover at least five minutes' worth of it, the profiler's own rule.

The figure compares the upward and the downward flux as Giangrande et al. (2016) do: the
model as simulated against the profiler, with no error added to either. The profiler splits
its samples into rising and sinking at w = 0 with no threshold, and each sample carries a
retrieval error, which adds to both its upward and its downward flux and leaves the net flux
alone. The text file therefore also gives the model with the profiler's own error, measured
from the data (``noise_profile``) and added to every model piece analytically
(``data.rwp_model_sample.noisy_fluxes``), and the net flux (up minus down).

The profiler is left out below 2 km: its w is the Doppler velocity corrected by a fall
speed from the reflectivity, and the reflectivity reads low there in heavy rain (receiver
saturation), so w comes out too negative (Wang et al. 2020 give 1.5 km; Gupta et al. 2024
mask below 2 km). The grey band is the 95 % range of the profiler mean when whole days
are resampled: an IOP has only 16-29 days with convection.

One figure per period (combined IOPs, IOP1, IOP2); the combined model sums both IOPs'
box statistics, each with its own IOP's profiler error. The text file adds the other
convective-column definitions, a 1.5 m s-1 error, the area fractions and velocities.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_COLORS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import (
    PLOT_TOP_KM,
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.data.rwp_model_sample import VARIANTS, noisy_fluxes
from alaro_analysis.data.rwp_profiler import histogram_quantiles
from alaro_analysis.recipes.iop.build_iop_closure_budgets import DISPLAY, LABELS
from alaro_analysis.recipes.iop.build_iop_rwp_mass_flux import (
    DEFAULT_WORK_DIR as RWP_WORK_DIR,
)
from alaro_analysis.recipes.iop.build_iop_rwp_model_sample import WORK_DIRS as MODEL_WORK_DIRS
from alaro_analysis.recipes.iop.make_iop_closure_budget_figures import EXPERIMENTS

# scale -> (figure folder name = stem, profiler window in hours, model box in km, description)
SCALES = {
    "20km": ("convective_mass_flux_vs_profiler", 1, 20,
             "20 km boxes (5 x 5 cells), one snapshot, against 1 h windows"),
    "60km": ("convective_mass_flux_vs_profiler_60km", 3, 60,
             "60 km boxes (15 x 15 cells) over three consecutive hourly snapshots, against 3 h windows"),
}
PERIOD_LABELS = {"combined": "Combined IOPs", "iop1": "IOP1", "iop2": "IOP2"}
# Convective columns as the profiler classifies its echo: Steiner-convective at 3 km in the
# model's own reflectivity (with 3MT active), echo levels only. Their share of a box matches
# the profiler's convective share of its hour (0.32-0.34 vs 0.33 in IOP1, 0.27-0.32 vs 0.29
# in IOP2); rain >= 1 mm in the hour, the first definition, takes in stratiform rain too
# (0.34-0.47). The rain thresholds are in the text file.
VARIANT = "steiner"
SUPERVISOR_NOISE = 1.5  # m s-1, the error quoted in the note
TEXT_HEIGHTS = (1.0, 2.0, 3.0, 6.0, 8.0, 10.0)
Z_MAX_KM = 16.0
OBS_COLOR = "#111111"
# The catalogue keys the no-3MT run as "2mom-xcu"; the IOP recipes call it NO3M.
COLOR_KEY = {"control_nodiaggrp": "control-nodiag", "control": "control", "graupel": "graupel", "2mom": "2mom",
             "NO3M": "2mom-xcu"}
TICK, AXIS, TITLE, LEGEND = 17, 19, 20, 18
MIN_SAMPLES = 1000  # gates with fewer samples give no noise estimate
OBS_MIN_KM = 2.0  # below this the profiler's w is biased low (see the docstring)
N_BOOT = 2000
BAND_COLOR = "0.80"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rwp-dir", type=Path, default=RWP_WORK_DIR)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--scales", nargs="+", default=list(SCALES), choices=list(SCALES))
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    return parse_configured_args(p, "make_iop_rwp_mass_flux_comparison", argv=argv)


def load_observed(rwp_dir: Path, period: str, window_hours: int) -> dict[str, np.ndarray]:
    with np.load(rwp_dir / f"rwp_mass_flux_{period}_{window_hours}h.npz") as cache:
        return {key: np.asarray(cache[key], float) for key in cache.files if key != "days"}


def load_model(model_dir: Path, iop: int, label: str) -> dict[str, np.ndarray]:
    with np.load(model_dir / f"iop{iop}" / f"{label}.npz") as cache:
        return {key: np.asarray(cache[key]) for key in cache.files}


def _running_median(values: np.ndarray, half: int = 2) -> np.ndarray:
    out = np.full_like(values, np.nan)
    for i in range(values.size):
        window = values[max(0, i - half): i + half + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return np.where(np.isfinite(values), out, np.nan)


def noise_profile(rwp_dir: Path, iop: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(height km, error m s-1, parts) of the profiler's per-sample velocity error.

    Two estimates, the larger one kept at each gate: the spread of w in stratiform rain
    (robust: interquartile range / 1.349), where the true w is a few tenths of m s-1, and
    the white noise of consecutive 6 s samples in convective echo. Below the melting layer
    the stratiform spread is the larger (0.7-1.1 m s-1), above it the white noise
    (0.8-1.3 m s-1). Both are lower bounds of a sort: the first misses errors peculiar to
    convective echo, the second misses errors that persist from sample to sample.
    """
    with np.load(rwp_dir / f"rwp_velocity_iop{iop}.npz") as cache:
        height = np.asarray(cache["height_km"], float)
        strat = np.asarray(cache["hist_strat"], float)
        lag_sq, lag_n = np.asarray(cache["lag_sq"], float), np.asarray(cache["lag_n"], float)
    quartiles = histogram_quantiles(strat, [0.25, 0.75])
    robust = np.where(strat.sum(axis=1) >= MIN_SAMPLES, (quartiles[:, 1] - quartiles[:, 0]) / 1.349, np.nan)
    white = np.where(lag_n >= MIN_SAMPLES, np.sqrt(lag_sq / np.maximum(lag_n, 1) / 2.0), np.nan)
    noise = _running_median(np.fmax(robust, white))
    ok = np.isfinite(noise)
    filled = np.interp(height, height[ok], noise[ok])  # across the melting layer and above
    return height, filled, {"stratiform": robust, "white": white}


def model_profiles(entries: list[tuple[dict, object]], variant: str) -> dict[str, np.ndarray]:
    """Box-mean profiles of one run and variant over one or more IOPs.

    ``entries`` is [(model cache, noise in m s-1), ...], one per IOP: each IOP's profiler
    error is added to its own pieces, then the box sums of the IOPs are added up.
    """
    totals = {k: 0.0 for k in ("up", "down", "area_up", "area_down", "mass", "area", "boxes")}
    for entry, noise in entries:
        mass, area = entry[f"{variant}__hist_mass"], entry[f"{variant}__hist_area"]
        up, down, area_up, area_down = noisy_fluxes(mass, area, noise)
        if np.all(np.asarray(noise) == 0):  # exact sums, free of the 0.1 m s-1 binning
            up, down = entry[f"{variant}__up"], entry[f"{variant}__down"]
        for key, value in (("up", up), ("down", down), ("area_up", area_up), ("area_down", area_down),
                           ("mass", mass.sum(axis=1)), ("area", area.sum(axis=1)),
                           ("boxes", float(entry[f"{variant}__boxes"]))):
            totals[key] = totals[key] + value
    boxes = max(totals["boxes"], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = totals["mass"] / totals["area"]
        return {
            "height_km": np.mean([e["height_agl_m"] for e, _ in entries], axis=0) / 1000.0,
            "flux_up": totals["up"] / boxes, "flux_down": totals["down"] / boxes,
            "flux_net": (totals["up"] - totals["down"]) / boxes,
            "sigma_up": totals["area_up"] / boxes, "sigma_down": totals["area_down"] / boxes,
            "v_up": totals["up"] / (rho * totals["area_up"]),
            "v_down": totals["down"] / (rho * totals["area_down"]),
        }


def observed_band(obs: dict, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """95 % range of the profiler's mean upward, downward and net flux, whole days resampled."""
    rng = np.random.default_rng(seed)
    up, down = obs["window_flux_up"], -obs["window_flux_down"]
    day = obs["window_day"].astype(int)
    n_days = int(day.max()) + 1
    members = [np.flatnonzero(day == d) for d in range(n_days)]
    draws = {"flux_up": [], "flux_down": [], "flux_net": []}
    with warnings.catch_warnings():  # the melting layer is all NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(N_BOOT):
            pick = np.concatenate([members[d] for d in rng.integers(0, n_days, n_days)])
            draws["flux_up"].append(np.nanmean(up[pick], axis=0))
            draws["flux_down"].append(np.nanmean(down[pick], axis=0))
            draws["flux_net"].append(np.nanmean(up[pick] - down[pick], axis=0))
        low_ok = obs["height_km"] >= OBS_MIN_KM
        return {key: tuple(np.where(low_ok, q, np.nan) for q in np.nanpercentile(np.array(v), [2.5, 97.5], axis=0))
                for key, v in draws.items()}


def observed_profiles(obs: dict) -> dict[str, np.ndarray]:
    """Profiler profiles, left out below OBS_MIN_KM, with their day-resampling band."""
    ok = obs["height_km"] >= OBS_MIN_KM
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "flux_up": obs["flux_up"], "flux_down": -obs["flux_down"], "flux_net": obs["flux_net"],
            "sigma_up": obs["sigma_up"], "sigma_down": obs["sigma_down"],
            "v_up": obs["flux_up"] / (obs["rho"] * obs["sigma_up"]),
            "v_down": -obs["flux_down"] / (obs["rho"] * obs["sigma_down"]),
        }
    out = {key: np.where(ok, value, np.nan) for key, value in out.items()}
    out["height_km"] = obs["height_km"]
    out["band"] = observed_band(obs)
    return out


def gather(args, scale: str, period: str) -> dict:
    _, window_hours, box_km, _ = SCALES[scale]
    iops = IOP_PERIODS[period]
    obs = load_observed(args.rwp_dir, period, window_hours)
    noise = {iop: noise_profile(args.rwp_dir, iop) for iop in iops}
    runs = {}
    for experiment in EXPERIMENTS:
        entries = {iop: load_model(MODEL_WORK_DIRS[box_km], iop, LABELS[experiment]) for iop in iops}
        measured = {}
        for iop, entry in entries.items():
            z_noise, values, _ = noise[iop]
            measured[iop] = np.interp(entry["height_agl_m"] / 1000.0, z_noise, values)
        runs[experiment] = {
            "entries": entries,
            "profiles": {(variant, level): model_profiles(
                [(entries[iop], {"none": 0.0, "measured": measured[iop], "supervisor": SUPERVISOR_NOISE}[level])
                 for iop in iops], variant)
                for variant in VARIANTS for level in ("none", "measured", "supervisor")},
        }
    observed_days = {}
    for iop in iops:
        with np.load(args.rwp_dir / f"rwp_velocity_iop{iop}.npz") as cache:
            observed_days[iop] = (str(cache["days"][0]), str(cache["days"][-1]),
                                  int((np.asarray(cache["available"]) >= 0.5).sum()))
    return {"obs": obs, "observed": observed_profiles(obs), "runs": runs, "noise": noise,
            "observed_days": observed_days, "iops": iops, "scale": scale, "period": period}


def panel(ax, observed: dict, runs: dict, kind: str, title: str | None, xlabel: str) -> None:
    ax.axvline(0.0, color="0.6", lw=0.8)
    z = observed["height_km"]
    band = observed["band"]
    if kind == "net":
        ax.fill_betweenx(z, *band["flux_net"], color=BAND_COLOR, lw=0, zorder=1)
        ax.plot(observed["flux_net"], z, color=OBS_COLOR, lw=3.0, zorder=5)
    else:
        ax.fill_betweenx(z, *band["flux_up"], color=BAND_COLOR, lw=0, zorder=1)
        ax.fill_betweenx(z, *(-b for b in band["flux_down"][::-1]), color=BAND_COLOR, lw=0, zorder=1)
        ax.plot(observed["flux_up"], z, color=OBS_COLOR, lw=3.0, zorder=5)
        ax.plot(-observed["flux_down"], z, color=OBS_COLOR, lw=3.0, ls="--", zorder=5)
    level = "measured" if kind == "noisy" else "none"
    for experiment, run in runs.items():
        prof = run["profiles"][(VARIANT, level)]
        colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
        if kind == "net":
            ax.plot(prof["flux_net"], prof["height_km"], color=colour, lw=2.0)
        else:
            ax.plot(prof["flux_up"], prof["height_km"], color=colour, lw=2.0)
            ax.plot(-prof["flux_down"], prof["height_km"], color=colour, lw=2.0, ls="--")
    ax.set_ylim(0, PLOT_TOP_KM)
    if title:
        ax.set_title(title, fontsize=TITLE)
    ax.set_xlabel(xlabel, fontsize=AXIS)
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.25)


def _at(height: np.ndarray, values: np.ndarray) -> list[float]:
    ok = np.isfinite(values) & np.isfinite(height)
    if not ok.any():
        return [float("nan")] * len(TEXT_HEIGHTS)
    order = np.argsort(height[ok])
    h, v = height[ok][order], np.asarray(values)[ok][order]
    return [float(np.interp(z, h, v)) if h[0] <= z <= h[-1] and np.min(np.abs(h - z)) < 0.5
            else float("nan") for z in TEXT_HEIGHTS]


def write_text(data: dict, path: Path) -> None:
    stem, window_hours, box_km, description = SCALES[data["scale"]]
    obs = data["obs"]
    title = f"{stem}_{data['period']}"
    lines = [
        title, "=" * len(title),
        f"Convective mass flux at the {box_km} km scale, ARM radar wind profiler at T3 against the runs,",
        f"{PERIOD_LABELS[data['period']]}. Model: {description}.",
        "Profiler: PI product doi:10.5439/1440997, method of Giangrande et al. 2016 eq. (1),",
        f"Mc = rho (sigma_u v_u + sigma_d v_d) over {window_hours} h windows ({box_km} km at 5 m/s) with",
        "5 min of convective echo; no retrieval in the melting layer (about 3.5-5 km); left out below",
        f"{OBS_MIN_KM:g} km, where the reflectivity-based fall-speed correction biases w low (receiver",
        "saturation in heavy rain; Wang et al. 2020, Gupta et al. 2024). Band: 2.5-97.5 % of the",
        f"profiler mean over {N_BOOT} resamplings of whole days.",
        "Model: boxes of the 4 km radar domain holding a convective column, each column split into 3MT",
        "updraft, downdraft and environment, divided by the full box, at the hours the profiler observed",
        "(IOP1 from 5 March), levels with model echo >= 10 dBZ only."
        + (" A 60 km box-window counts when its convective columns, averaged over the three snapshots,"
           " cover at least 5/180 of it (the profiler's five minutes of convective echo in three hours);"
           " its flux is the mean over the three snapshots." if box_km == 60 else ""),
        f"Figure: variant '{VARIANT}', the model as simulated (noise none); the model with the measured",
        "profiler error and the net flux are in this file only. The combined IOPs add up both IOPs'",
        "box sums, each IOP with its own profiler error.",
        "Variants: rain1_all = rain >= 1 mm/h and 3MT active, every level (the former definition);",
        "rain1/rain5/rain10 = rain >= 1/5/10 mm/h, 3MT active, echo levels; steiner = Steiner-",
        "convective at 3 km, 3MT active, echo levels. G2M-XCU has no 3MT: rain/Steiner only.",
        "Noise: none, measured (profile below), supervisor (1.5 m/s at every level).",
        "Upward and downward quantities are positive; v = Mc / (rho sigma).",
        "",
        f"profiler: {int(obs['n_days'][0])} days with convection, {int(obs['n_windows'][0])} windows, "
        f"convective share of the window {obs['convective_column_fraction'][0]:.3f}",
    ]
    for iop in data["iops"]:
        first, last = GOAMAZON_IOPS[iop]
        z_noise, noise, parts = data["noise"][iop]
        day0, day1, hours = data["observed_days"][iop]
        lines += [
            f"IOP{iop}: {first} to {last}; profiler files {day0} to {day1}, {hours} observed hours.",
            "  measured error (m/s) at " + ", ".join(f"{z:g}" for z in TEXT_HEIGHTS) + " km: "
            + ", ".join(f"{v:.2f}" for v in _at(z_noise, noise))
            + "  [stratiform spread " + ", ".join(f"{v:.2f}" for v in _at(z_noise, parts["stratiform"]))
            + "; white noise " + ", ".join(f"{v:.2f}" for v in _at(z_noise, parts["white"])) + "]",
        ]
        for experiment, run in data["runs"].items():
            entry = run["entries"][iop]
            for variant in VARIANTS:
                boxes = float(entry[f"{variant}__boxes"])
                share = float(entry[f"{variant}__conv_share"]) / max(boxes, 1.0)
                lines.append(f"  {DISPLAY[experiment]:<13} {variant:<9} {int(entry['snapshots'])} snapshots, "
                             f"{int(boxes)} convective box-snapshots of "
                             f"{int(float(entry[f'{variant}__boxes_all']))}, convective share of the box {share:.3f}")
    lines += ["", "dataset,variant,noise,quantity," + ",".join(f"z{z:g}km" for z in TEXT_HEIGHTS)]
    quantities = ("flux_net", "flux_up", "flux_down", "sigma_up", "sigma_down", "v_up", "v_down")
    observed = data["observed"]
    for q in quantities:
        lines.append(f"profiler,-,-,{q}," + ",".join(f"{v:.4f}" for v in _at(observed["height_km"], observed[q])))
    for q, (low, high) in observed["band"].items():
        for name, values in (("p2.5", low), ("p97.5", high)):
            lines.append(f"profiler,{name},-,{q}," + ",".join(f"{v:.4f}" for v in _at(observed["height_km"], values)))
    for experiment, run in data["runs"].items():
        for (variant, level), prof in run["profiles"].items():
            for q in quantities:
                lines.append(f"{DISPLAY[experiment]},{variant},{level},{q},"
                             + ",".join(f"{v:.4f}" for v in _at(prof["height_km"], prof[q])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PANELS = (("raw", "Convective mass flux (kg m$^{-2}$ s$^{-1}$)"),)


def draw(data: dict, png: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(8.0 * len(PANELS), 9.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (kind, xlabel), letter in zip(axes, PANELS, "ab"):
        panel(ax, data["observed"], data["runs"], kind, None, xlabel)
        if len(PANELS) > 1:
            ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, va="top",
                    fontsize=TITLE, fontweight="bold")
    axes[0].set_ylabel("Height (km)", fontsize=AXIS)
    # all panels on one x range, so they read against each other by eye
    lims = [ax.get_xlim() for ax in axes]
    for ax in axes:
        ax.set_xlim(min(l[0] for l in lims), max(l[1] for l in lims))
    handles = [plt.Line2D([], [], color=OBS_COLOR, lw=3.0, label="Radar wind profiler"),
               Patch(color=BAND_COLOR, label="profiler, 95 % range over days")]
    handles += [plt.Line2D([], [], color=EXPERIMENT_COLORS[COLOR_KEY[e]], lw=2.0, label=DISPLAY[e])
                for e in EXPERIMENTS]
    handles += [plt.Line2D([], [], color="0.35", lw=2.0, label="upward"),
                plt.Line2D([], [], color="0.35", lw=2.0, ls="--", label="downward")]
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.12, wspace=0.08)
    legend = legend_below(fig, list(axes), handles, ncol=2 * len(PANELS), fontsize=LEGEND,
                          title=PERIOD_LABELS[data["period"]], title_fontsize=LEGEND)
    legend.get_title().set_fontweight("bold")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for scale in args.scales:
        stem, _, box_km, _ = SCALES[scale]
        if not all((MODEL_WORK_DIRS[box_km] / f"iop{iop}" / f"{LABELS[e]}.npz").exists()
                   for iop in GOAMAZON_IOPS for e in EXPERIMENTS):
            print(f"[{scale}] model sample not built yet, skip", flush=True)
            continue
        for period in args.periods:
            data = gather(args, scale, period)
            png, jpg, txt = figure_files(args.figure_root, stem, stem, period)
            draw(data, png, args.dpi)
            convert_to_jpg(png, jpg, dpi=args.dpi)
            png.unlink()
            write_text(data, txt)
            print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.figure_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
