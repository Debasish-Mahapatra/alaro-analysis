#!/usr/bin/env python3
"""Vertical velocity of the runs against the ARM radar wind profiler at T3.

Section 2 of the supervisor's note. Velocity distributions depend on the scale they are
sampled at: the profiler sees ~0.5-1 km for 6 s, the model resolves features 4-7 grid
lengths across (15-30 km). So raw profiler distributions are not set against grid-point
ones. One figure per period (combined IOPs, IOP1, IOP2), two rows:

Row 1, the model as the profiler sees it (note 2a): each convective column enters as three
pieces (3MT updraft, downdraft, rest) weighted by their area, at levels with model echo
(>= 10 dBZ), with the profiler's measured error added to every piece, against the raw 6 s
profiler samples in convective echo: (a) share of samples with w > 3 m s-1, (b) with
w < -3 m s-1, (c) 95th and 5th percentiles.

Row 2: (d) the profiler as the model sees it (note 2b): profiler w averaged over 13 min
(one 4 km cell at 5 m s-1) and 30 min (9 km), inside convective echo only, against the
resolved (grid-mean) w of the convective columns, 99th and 1st percentiles; (e, f) the
scale plot (note 2c): the profiler's 99th percentile and its mass-flux-weighted mean
updraft against the averaging length L = 5 m s-1 x time, below and above the melting
layer, with the runs' resolved w at their effective scale and the 3MT bulk updraft at the
small-scale end, against the flux-weighted mean of the profiler's updrafts.

The profiler is left out below 2 km (its w is biased low there, see
``make_iop_rwp_mass_flux_comparison``), above 12 km (Wang et al. 2020; it sees convective
echo there for only a few hours), and at heights with fewer than ``MIN_RAW_HOURS`` hours of
convective echo or ``MIN_AVERAGED`` independent averaging windows. The combined IOPs pool
both IOPs' histograms, the model's each with its own IOP's profiler error.

Model: ``build_iop_rwp_model_sample`` (convective-column variant as in
make_iop_rwp_mass_flux_comparison); profiler: ``build_iop_rwp_velocity``. Heights are above
ground, in 200 m bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import ndtr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_COLORS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg, legend_below
from alaro_analysis.common.iop_settings import PLOT_TOP_KM, FIGURES, IOP_PERIODS, figure_files, write_figure_index
from alaro_analysis.data.rwp_model_sample import add_noise
from alaro_analysis.data.rwp_profiler import (
    PROPAGATION_MS,
    SAMPLE_SECONDS,
    exceedance,
    flux_weighted_updraft,
    histogram_quantiles,
)
from alaro_analysis.recipes.iop.build_iop_closure_budgets import DISPLAY, LABELS
from alaro_analysis.recipes.iop.build_iop_rwp_mass_flux import DEFAULT_WORK_DIR as RWP_WORK_DIR
from alaro_analysis.recipes.iop.build_iop_rwp_model_sample import DEFAULT_WORK_DIR as MODEL_WORK_DIR
from alaro_analysis.recipes.iop.make_iop_closure_budget_figures import EXPERIMENTS
from alaro_analysis.recipes.iop.make_iop_rwp_mass_flux_comparison import (
    AXIS,
    COLOR_KEY,
    LEGEND,
    OBS_COLOR,
    OBS_MIN_KM,
    PERIOD_LABELS,
    TICK,
    TITLE,
    VARIANT,
    load_model,
    noise_profile,
)

FIGURE_NAME = "vertical_velocity_vs_profiler"
Z_CENTRES = np.round(np.arange(0.1, 16.0, 0.2), 1)  # 200 m bins above ground
Z_TOP = 12.0  # the profiler's top (OBS_MAX_KM); the runs above it have nothing to meet
OBS_MAX_KM = 12.0
GAP_KM = {"obs": 0.3, "model": 1.0}  # no interpolation across larger gaps (the melting layer)
THRESHOLD = 3.0  # m s-1
PERCENTILES = (0.05, 0.95)
TAILS = (0.01, 0.99)
AVERAGING_S = (800, 1800)  # 4 km and 9 km at 5 m s-1
LAYERS = ((2.0, 3.5), (6.0, 9.0))  # below and above the melting layer
GRID_KM = 4.0
EFFECTIVE = (4, 7)  # grid lengths the model really resolves
MIN_WINDOWS = 10  # independent profiler windows in a layer for a point of the scale plot
MIN_RAW_HOURS = 3.0  # hours of convective echo for a height's raw distribution
MIN_AVERAGED = 30  # independent averaging windows for a height's averaged distribution
OBS_GREY = "0.55"
ERROR_GREY = "0.45"
TEXT_HEIGHTS = (2.0, 3.0, 6.0, 8.0, 10.0, 12.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rwp-dir", type=Path, default=RWP_WORK_DIR)
    p.add_argument("--model-dir", type=Path, default=MODEL_WORK_DIR)
    p.add_argument("--figure-root", type=Path, default=FIGURES)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    return parse_configured_args(p, "make_iop_rwp_velocity_figures", argv=argv)


def binned_counts(hist: np.ndarray, z_km: np.ndarray, max_gap: float, min_total: float = 0.0) -> np.ndarray:
    """(height, bin) counts linearly interpolated, as distributions, to ``Z_CENTRES``.

    Each level is normalised, interpolated and scaled back by the interpolated total, so
    counts of several sources add up to their pooled distribution. Levels with no more
    than ``min_total`` samples are skipped; a centre farther than ``max_gap`` from the
    levels around it stays NaN, so nothing is invented across the melting layer.
    """
    total = hist.sum(axis=1)
    ok = (total > max(min_total, 0.0)) & np.isfinite(z_km)
    z, pdf, tot = z_km[ok], hist[ok] / total[ok][:, None], total[ok]
    order = np.argsort(z)
    z, pdf, tot = z[order], pdf[order], tot[order]
    out = np.full((Z_CENTRES.size, hist.shape[1]), np.nan)
    for i, centre in enumerate(Z_CENTRES):
        j = np.searchsorted(z, centre)
        if j == 0 or j == z.size:
            continue
        lo, hi = z[j - 1], z[j]
        if centre - lo > max_gap or hi - centre > max_gap:
            continue
        f = (centre - lo) / (hi - lo)
        out[i] = ((1.0 - f) * pdf[j - 1] + f * pdf[j]) * ((1.0 - f) * tot[j - 1] + f * tot[j])
    return out


def normalised(counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / counts.sum(axis=1, keepdims=True)


def observed_mask() -> np.ndarray:
    return (Z_CENTRES >= OBS_MIN_KM) & (Z_CENTRES <= OBS_MAX_KM)


def layer_sum(hist: np.ndarray, z_km: np.ndarray, layer) -> np.ndarray:
    keep = (z_km >= layer[0]) & (z_km <= layer[1])
    return hist[keep].sum(axis=0)


def load(args, period: str) -> dict:
    """Profiler histograms summed over the period's IOPs, and the runs' caches per IOP."""
    iops = IOP_PERIODS[period]
    obs, noise, runs = {}, {}, {e: {} for e in EXPERIMENTS}
    for iop in iops:
        with np.load(args.rwp_dir / f"rwp_velocity_iop{iop}.npz") as cache:
            for key in ("hist_conv",):
                obs[key] = obs.get(key, 0) + np.asarray(cache[key], float)
            obs["height_km"] = np.asarray(cache["height_km"], float)
            obs["block_seconds"] = np.asarray(cache["block_seconds"], float)
        z_noise, values, _ = noise_profile(args.rwp_dir, iop)
        noise[iop] = (z_noise, values)
        for experiment in EXPERIMENTS:
            entry = load_model(args.model_dir, iop, LABELS[experiment])
            z = entry["height_agl_m"] / 1000.0
            runs[experiment][iop] = {"entry": entry, "z": z, "noise": np.interp(z, z_noise, values)}
    return {"obs": obs, "noise": noise, "runs": runs, "iops": iops, "period": period}


def model_counts(runs: dict, key: str, noisy: bool) -> np.ndarray:
    """A run's (height, bin) counts of histogram ``key``, pooled over its IOPs."""
    total = 0.0
    for run in runs.values():
        hist = run["entry"][f"{VARIANT}__{key}"].astype(float)
        if noisy:
            hist = add_noise(hist, run["noise"])
        total = total + np.nan_to_num(binned_counts(hist, run["z"], GAP_KM["model"]))
    return np.where(np.asarray(total).sum(axis=1, keepdims=True) > 0, total, np.nan)


def _stats(hist: np.ndarray) -> dict[str, float]:
    if hist.sum() <= 0:
        return {"p95": np.nan, "p99": np.nan, "flux": np.nan}
    pct = histogram_quantiles(hist, (0.95, 0.99))
    return {"p95": float(pct[0]), "p99": float(pct[1]), "flux": float(flux_weighted_updraft(hist))}


def _at(values: np.ndarray) -> str:
    return ",".join(f"{float(values[np.argmin(np.abs(Z_CENTRES - z))]):.4f}" for z in TEXT_HEIGHTS)


def _profile_axes(ax, xlabel: str) -> None:
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_ylim(0, PLOT_TOP_KM)
    ax.set_xlabel(xlabel, fontsize=AXIS)
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.25)


def draw(data: dict, png: Path, dpi: int) -> list[str]:
    obs = data["obs"]
    keep = observed_mask()
    lines = []
    fig = plt.figure(figsize=(21.0, 16.0))
    grid = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=0.96, bottom=0.12, wspace=0.22, hspace=0.30)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)])
    for c in (1, 2):
        axes[0, c].sharey(axes[0, 0])
    axes[1, 0].sharey(axes[0, 0])

    # Row 1: the model as the profiler sees it.
    raw = obs["hist_conv"][0]
    obs_raw = normalised(binned_counts(raw, obs["height_km"], GAP_KM["obs"], MIN_RAW_HOURS * 3600 / SAMPLE_SECONDS))
    obs_raw[~keep] = np.nan
    up, down = exceedance(obs_raw, THRESHOLD)
    pct = histogram_quantiles(obs_raw, PERCENTILES)
    noise = np.mean([np.interp(Z_CENTRES, *data["noise"][iop]) for iop in data["iops"]], axis=0)
    alone = 100 * (1.0 - ndtr(THRESHOLD / noise))
    ax_up, ax_down, ax_pct = axes[0]
    ax_up.plot(100 * up, Z_CENTRES, color=OBS_COLOR, lw=3.0, zorder=5)
    ax_down.plot(100 * down, Z_CENTRES, color=OBS_COLOR, lw=3.0, zorder=5)
    ax_pct.plot(pct[:, 1], Z_CENTRES, color=OBS_COLOR, lw=3.0, zorder=5)
    ax_pct.plot(pct[:, 0], Z_CENTRES, color=OBS_COLOR, lw=3.0, ls="--", zorder=5)
    for ax in (ax_up, ax_down):
        ax.plot(alone, Z_CENTRES, color=ERROR_GREY, lw=1.6, ls=":")
    lines += ["profiler,-,w>3 %," + _at(100 * up), "profiler,-,w<-3 %," + _at(100 * down),
              "profiler,-,p5," + _at(pct[:, 0]), "profiler,-,p95," + _at(pct[:, 1]),
              "error alone,measured,w>3 %," + _at(alone), "error alone,measured,sd," + _at(noise)]
    for experiment, runs in data["runs"].items():
        colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
        for noisy in (True, False):
            pdf = normalised(model_counts(runs, "hist_area", noisy))
            m_up, m_down = exceedance(pdf, THRESHOLD)
            m_pct = histogram_quantiles(pdf, PERCENTILES)
            if noisy:
                ax_up.plot(100 * m_up, Z_CENTRES, color=colour, lw=2.2)
                ax_down.plot(100 * m_down, Z_CENTRES, color=colour, lw=2.2)
                ax_pct.plot(m_pct[:, 1], Z_CENTRES, color=colour, lw=2.2)
                ax_pct.plot(m_pct[:, 0], Z_CENTRES, color=colour, lw=2.2, ls="--")
            tag = f"{DISPLAY[experiment]},{'measured' if noisy else 'none'}"
            lines += [f"{tag},w>3 %," + _at(100 * m_up), f"{tag},w<-3 %," + _at(100 * m_down),
                      f"{tag},p5," + _at(m_pct[:, 0]), f"{tag},p95," + _at(m_pct[:, 1])]
    _profile_axes(ax_up, "Samples with w > 3 m s$^{-1}$ (%)")
    _profile_axes(ax_down, "Samples with w < $-$3 m s$^{-1}$ (%)")
    _profile_axes(ax_pct, "5th and 95th percentile of w (m s$^{-1}$)")
    top = max(ax_up.get_xlim()[1], ax_down.get_xlim()[1])
    for ax in (ax_up, ax_down):
        ax.set_xlim(0, top)
    ax_up.set_ylabel("Height above ground (km)", fontsize=AXIS)

    # (d) The profiler as the model sees it.
    ax_avg = axes[1, 0]
    blocks = list(obs["block_seconds"].astype(int))
    for seconds, colour in zip(AVERAGING_S, (OBS_COLOR, OBS_GREY)):
        i = blocks.index(seconds)
        pdf = normalised(binned_counts(obs["hist_conv"][i], obs["height_km"], GAP_KM["obs"],
                                       MIN_AVERAGED * seconds / SAMPLE_SECONDS))
        pdf[~keep] = np.nan
        q = histogram_quantiles(pdf, TAILS)
        ax_avg.plot(q[:, 1], Z_CENTRES, color=colour, lw=3.0, zorder=5)
        ax_avg.plot(q[:, 0], Z_CENTRES, color=colour, lw=3.0, ls="--", zorder=5)
        lines += [f"profiler {seconds}s,-,p1," + _at(q[:, 0]), f"profiler {seconds}s,-,p99," + _at(q[:, 1])]
    for experiment, runs in data["runs"].items():
        colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
        q = histogram_quantiles(normalised(model_counts(runs, "hist_grid", False)), TAILS)
        ax_avg.plot(q[:, 1], Z_CENTRES, color=colour, lw=2.2)
        ax_avg.plot(q[:, 0], Z_CENTRES, color=colour, lw=2.2, ls="--")
        lines += [f"{DISPLAY[experiment]} resolved,none,p1," + _at(q[:, 0]),
                  f"{DISPLAY[experiment]} resolved,none,p99," + _at(q[:, 1])]
    _profile_axes(ax_avg, "1st and 99th percentile of w (m s$^{-1}$)")
    ax_avg.set_ylabel("Height above ground (km)", fontsize=AXIS)

    # (e, f) The scale plot.
    length = PROPAGATION_MS * obs["block_seconds"] / 1000.0
    eff = (EFFECTIVE[0] * GRID_KM, EFFECTIVE[1] * GRID_KM)
    eff_mid = float(np.sqrt(eff[0] * eff[1]))
    lines.append("")
    lines.append("layer,dataset,L_km,p95,p99,flux_weighted_updraft,n_indep")
    for ax, layer in zip(axes[1, 1:], LAYERS):
        z_obs = obs["height_km"]
        series = {"p99": [], "flux": []}
        for i, seconds in enumerate(obs["block_seconds"]):
            hist = layer_sum(obs["hist_conv"][i], z_obs, layer)
            gates = ((z_obs >= layer[0]) & (z_obs <= layer[1])).sum()
            n_indep = hist.sum() / max(gates, 1) * SAMPLE_SECONDS / seconds
            stats = _stats(hist) if n_indep >= MIN_WINDOWS else {"p95": np.nan, "p99": np.nan, "flux": np.nan}
            series["p99"].append(stats["p99"])
            series["flux"].append(stats["flux"])
            lines.append(f"{layer[0]:g}-{layer[1]:g}km,profiler,{length[i]:.3f},{stats['p95']:.3f},"
                         f"{stats['p99']:.3f},{stats['flux']:.3f},{n_indep:.0f}")
        ax.plot(length, series["p99"], color=OBS_COLOR, lw=2.5, marker="o", ms=8, zorder=5)
        ax.plot(length, series["flux"], color=OBS_GREY, lw=2.5, marker="^", ms=8, zorder=5)
        ax.axvspan(eff[0], eff[1], color="0.9", zorder=0)
        for j, (experiment, runs) in enumerate(data["runs"].items()):
            colour = EXPERIMENT_COLORS[COLOR_KEY[experiment]]
            hist = sum(layer_sum(r["entry"][f"{VARIANT}__hist_grid"].astype(float), r["z"], layer)
                       for r in runs.values())
            stats = _stats(hist)
            ax.plot([eff_mid * (1.0 + 0.08 * (j - 1.5))], [stats["p99"]], color=colour, marker="o", ms=11,
                    ls="none", markeredgecolor="k", markeredgewidth=0.7, zorder=6)
            w1 = sum(r["entry"][f"{VARIANT}__ud_w1"][(r["z"] >= layer[0]) & (r["z"] <= layer[1])].sum()
                     for r in runs.values())
            w2 = sum(r["entry"][f"{VARIANT}__ud_w2"][(r["z"] >= layer[0]) & (r["z"] <= layer[1])].sum()
                     for r in runs.values())
            bulk = float(w2 / w1) if w1 > 0 else np.nan
            if np.isfinite(bulk):
                ax.plot([length[0] * (1.0 + 0.12 * (j - (len(EXPERIMENTS) - 1) / 2))], [bulk], color=colour, marker="*", ms=17,
                        ls="none", markeredgecolor="k", markeredgewidth=0.7, zorder=6)
            lines.append(f"{layer[0]:g}-{layer[1]:g}km,{DISPLAY[experiment]} resolved,{eff_mid:.1f},"
                         f"{stats['p95']:.3f},{stats['p99']:.3f},{stats['flux']:.3f},-")
            lines.append(f"{layer[0]:g}-{layer[1]:g}km,{DISPLAY[experiment]} 3MT bulk updraft,{length[0]:.3f},"
                         f"-,-,{bulk:.3f},-")
        ax.set_xscale("log")
        ax.set_xlabel("Averaging length (km)", fontsize=AXIS)
        ax.set_ylabel(f"w (m s$^{{-1}}$), {layer[0]:g}-{layer[1]:g} km", fontsize=AXIS)
        ax.tick_params(labelsize=TICK)
        ax.grid(alpha=0.25, which="both")
    top = max(ax.get_ylim()[1] for ax in axes[1, 1:])
    for ax in axes[1, 1:]:
        ax.set_ylim(0, top)

    titles = ("Strong updrafts", "Strong downdrafts", "Tails of w",
              "Averaged profiler vs resolved w", "Below the melting layer", "Above the melting layer")
    for ax, letter, title in zip(axes.flat, "abcdef", titles):
        ax.text(0.0, 1.01, f"({letter})", transform=ax.transAxes, va="bottom", ha="left",
                fontsize=TITLE, fontweight="bold")
        ax.set_title(title, fontsize=TITLE - 2, pad=8)
    handles = [Line2D([], [], color=OBS_COLOR, lw=3.0, label="Radar wind profiler (6 s; 13 min in d)"),
               Line2D([], [], color=OBS_GREY, lw=3.0, label="Profiler, 30 min average (d)"),
               Line2D([], [], color=ERROR_GREY, lw=1.6, ls=":", label="Profiler error alone"),
               Line2D([], [], color="0.3", lw=2.2, label="95th / 99th percentile"),
               Line2D([], [], color="0.3", lw=2.2, ls="--", label="5th / 1st percentile")]
    handles += [Line2D([], [], color=EXPERIMENT_COLORS[COLOR_KEY[e]], lw=2.2, label=DISPLAY[e]) for e in EXPERIMENTS]
    handles += [Line2D([], [], color=OBS_COLOR, lw=2.5, marker="o", ms=8, label="Profiler 99th percentile (e, f)"),
                Line2D([], [], color=OBS_GREY, lw=2.5, marker="^", ms=8, label="Profiler flux-weighted updraft (e, f)"),
                Line2D([], [], color="white", marker="o", ms=11, markeredgecolor="k", ls="none",
                       label="Runs: resolved w, 99th percentile"),
                Line2D([], [], color="white", marker="*", ms=17, markeredgecolor="k", ls="none",
                       label="Runs: 3MT bulk updraft"),
                Patch(color="0.9", label="Model effective scale (4-7 cells)")]
    legend = legend_below(fig, axes, handles, ncol=3, fontsize=LEGEND, title=PERIOD_LABELS[data["period"]],
                          title_fontsize=LEGEND)
    legend.get_title().set_fontweight("bold")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return lines


def header(data: dict) -> list[str]:
    title = f"{FIGURE_NAME}_{data['period']}"
    return [
        title, "=" * len(title),
        f"Vertical velocity against the ARM radar wind profiler at T3, {PERIOD_LABELS[data['period']]}.",
        "(a-c) the model as the profiler sees it: convective columns ('" + VARIANT + "'), echo levels,",
        "pieces (3MT updraft, downdraft, rest) weighted by area, measured profiler error added; the",
        "profiler's raw 6 s samples in convective echo (classes convection and weak convection).",
        "(d) profiler w averaged over 800 s (4 km at 5 m/s) and 1800 s (9 km) inside convective echo",
        "(windows half filled with convective echo, sliding by 6 s) against the resolved grid-mean w",
        "of the convective columns, no error added.",
        "(e, f) profiler 95th/99th percentile and flux-weighted mean updraft sum(w^2)/sum(w) over",
        "w > 0, pooled over the layer, against L = 5 m/s x averaging time (6 s .. 1 h); runs: the same",
        f"of the resolved w at the effective scale {EFFECTIVE[0]}-{EFFECTIVE[1]} grid lengths, and the 3MT bulk",
        "updraft sum(sig rho w^2)/sum(sig rho w) of the absolute 3MT updraft velocity.",
        f"Profiler left out below {OBS_MIN_KM:g} km (w biased low), above {OBS_MAX_KM:g} km, at heights with less",
        f"than {MIN_RAW_HOURS:g} h of convective echo (raw) or {MIN_AVERAGED} independent windows (averaged), and",
        f"scale-plot points with fewer than {MIN_WINDOWS} independent windows. IOPs: "
        + ", ".join(f"IOP{i}" for i in data["iops"]) + "; combined = histograms pooled.",
        "",
        "dataset,noise,statistic," + ",".join(f"z{z:g}km" for z in TEXT_HEIGHTS),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for period in args.periods:
        data = load(args, period)
        png, jpg, txt = figure_files(args.figure_root, FIGURE_NAME, FIGURE_NAME, period)
        lines = draw(data, png, args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        txt.write_text("\n".join(header(data) + lines) + "\n", encoding="utf-8")
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.figure_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
