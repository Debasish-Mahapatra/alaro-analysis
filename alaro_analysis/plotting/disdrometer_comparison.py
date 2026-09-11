"""Disdrometer comparison. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS


def _as_datetime(values: np.ndarray) -> list[datetime]:
    return [v.astype("datetime64[s]").astype(datetime) for v in values]


def plot_timeseries(
    out: Path,
    valid_times: np.ndarray,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    dates = _as_datetime(valid_times)
    ax.plot(dates, np.where(obs > 0.0, obs, np.nan), color="#111111", lw=1.6, label="Obs")
    for exp, values in model.items():
        ax.plot(
            dates,
            np.where(values > 0.0, values, np.nan),
            color=EXPERIMENT_COLORS[exp],
            lw=1.4,
            label=EXPERIMENT_LABELS[exp],
            alpha=0.9,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_title(f"Near-surface rain number concentration, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="upper right", frameon=False, ncol=4)
    fig.savefig(out, dpi=450)
    plt.close(fig)


def plot_scatter(
    out: Path,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(model), figsize=(14, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    positives = [obs[obs > 0.0]]
    positives.extend(v[v > 0.0] for v in model.values())
    combined = np.concatenate([v[np.isfinite(v)] for v in positives if v.size])
    if combined.size:
        lo = max(1.0e-3, float(np.nanpercentile(combined, 1)) * 0.5)
        hi = float(np.nanpercentile(combined, 99)) * 2.0
    else:
        lo, hi = 1.0e-2, 1.0e4
    for ax, (exp, values) in zip(axes, model.items(), strict=True):
        mask = np.isfinite(obs) & np.isfinite(values) & (obs > 0.0) & (values > 0.0)
        ax.scatter(
            obs[mask],
            values[mask],
            s=18,
            alpha=0.65,
            color=EXPERIMENT_COLORS[exp],
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(EXPERIMENT_LABELS[exp])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Observed (m$^{-3}$)")
    axes[0].set_ylabel("Model (m$^{-3}$)")
    fig.suptitle(f"Disdrometer vs lowest model level rain number, {lead_text}")
    fig.tight_layout()
    fig.savefig(out, dpi=450)
    plt.close(fig)


def plot_distribution(
    out: Path,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    series = {"Obs": obs, **{EXPERIMENT_LABELS[e]: v for e, v in model.items()}}
    finite = np.concatenate(
        [v[np.isfinite(v) & (v > 0.0)] for v in series.values() if np.asarray(v).size]
    )
    if finite.size:
        bins = np.logspace(
            math.log10(max(1.0e-3, float(np.nanpercentile(finite, 1)) * 0.5)),
            math.log10(float(np.nanpercentile(finite, 99)) * 2.0),
            34,
        )
    else:
        bins = np.logspace(-2, 4, 34)
    for label, values in series.items():
        values = np.asarray(values)
        clean = values[np.isfinite(values) & (values > 0.0)]
        if clean.size == 0:
            continue
        color = (
            "#111111"
            if label == "Obs"
            else EXPERIMENT_COLORS[next(exp for exp in model if EXPERIMENT_LABELS[exp] == label)]
        )
        ax.hist(
            clean,
            bins=bins,
            histtype="step",
            density=True,
            lw=1.7,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_ylabel("Density")
    ax.set_title(f"Matched rain number distributions, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out, dpi=450)
    plt.close(fig)
