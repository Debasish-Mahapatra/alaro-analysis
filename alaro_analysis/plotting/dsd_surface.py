"""Dsd surface. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS

DEFAULT_FIGURE_DPI = 450


PANEL_ORDER = ("obs", "control", "graupel", "2mom")


PANEL_GRID_POSITIONS = {"obs": (0, 0), "control": (0, 1), "graupel": (1, 0), "2mom": (1, 1)}


def _format_count(n: int) -> str:
    return f"n={n:,}"


def _compute_obs_contour(
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    sigma_bins: float = 1.5,
    contour_level_pct: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]] | None:
    """Smoothed obs frequency field + a single inclusive contour level."""
    from scipy.ndimage import gaussian_filter

    if obs_x.size == 0:
        return None
    h, _, _ = np.histogram2d(obs_x, obs_y, bins=[x_edges, y_edges])
    if h.sum() == 0:
        return None
    freq_pct = 100.0 * h / h.sum()
    smoothed = gaussian_filter(freq_pct, sigma=sigma_bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, smoothed.T, [contour_level_pct]


def plot_2x2_with_marginals(
    out_path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    x_label: str,
    *,
    title: str = "",
    bins: int = 60,
) -> None:
    """2x2 joint density panels, each with bottom-X and left-Y marginals.

    The marginals overlay all four datasets so they read as constant
    references.  The panel's own dataset is drawn thicker on top.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            x_los.append(float(np.nanpercentile(sx, 0.5)))
            x_his.append(float(np.nanpercentile(sx, 99.5)))
        if sy.size:
            y_los.append(float(np.nanpercentile(sy, 0.5)))
            y_his.append(float(np.nanpercentile(sy, 99.5)))
    if not x_los or not y_los:
        raise RuntimeError("No samples available to plot 2x2+marginals")
    x_lo = max(0.2, min(x_los))
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    obs_x = samples["obs"][x_field]
    obs_y = samples["obs"]["log_nw"]
    contour = _compute_obs_contour(obs_x, obs_y, x_edges, y_edges)

    color_for_name = {
        "obs": "#111111",
        "control": "#d62728",
        "graupel": "#1f77b4",
        "2mom": "#2ca02c",
    }
    label_for_name = {
        "obs": "Obs",
        "control": EXPERIMENT_LABELS.get("control", "C1M"),
        "graupel": EXPERIMENT_LABELS.get("graupel", "G1M"),
        "2mom": EXPERIMENT_LABELS.get("2mom", "G2M"),
    }

    x_pdf: dict[str, np.ndarray] = {}
    y_pdf: dict[str, np.ndarray] = {}
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            xh, _ = np.histogram(sx, bins=x_edges, density=True)
            x_pdf[name] = xh
        else:
            x_pdf[name] = np.zeros(bins)
        if sy.size:
            yh, _ = np.histogram(sy, bins=y_edges, density=True)
            y_pdf[name] = yh
        else:
            y_pdf[name] = np.zeros(bins)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(13.5, 12.0))
    outer = fig.add_gridspec(
        2, 2, hspace=0.20, wspace=0.18, left=0.06, right=0.94, top=0.93, bottom=0.05
    )

    last_im = None
    for name in PANEL_ORDER:
        i, j = PANEL_GRID_POSITIONS[name]
        inner = outer[i, j].subgridspec(
            2,
            2,
            width_ratios=[1, 4],
            height_ratios=[4, 1],
            hspace=0.04,
            wspace=0.04,
        )
        ax_y = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[0, 1], sharey=ax_y)
        ax_x = fig.add_subplot(inner[1, 1], sharex=ax_main)

        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        n = int(sx.size)
        if n == 0:
            ax_main.set_title(f"{label_for_name[name]} (n=0)")
            continue
        h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
        h = h / max(1.0, h.sum())
        with np.errstate(divide="ignore"):
            shown = np.log10(np.where(h > 0, h, np.nan))
        ax_main.set_facecolor("white")
        im = ax_main.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, shading="auto")
        last_im = im
        if contour is not None:
            xc, yc, field, levels = contour
            try:
                ax_main.contour(xc, yc, field, levels=levels, colors="black", linewidths=1.2)
            except ValueError:
                pass
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(y_lo, y_hi)
        ax_main.set_title(f"{label_for_name[name]} (n={n:,})")
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        plt.setp(ax_main.get_yticklabels(), visible=False)

        for ds_name in PANEL_ORDER:
            color = color_for_name[ds_name]
            lw = 2.4 if ds_name == name else 1.0
            alpha = 1.0 if ds_name == name else 0.7
            zorder = 5 if ds_name == name else 2
            ax_x.plot(
                x_centers,
                x_pdf[ds_name],
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )
            ax_y.plot(
                y_pdf[ds_name],
                y_centers,
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )
        ax_x.set_xlim(x_lo, x_hi)
        ax_x.set_ylim(bottom=0.0)
        ax_x.grid(True, alpha=0.3)
        ax_x.set_xlabel(x_label)
        ax_x.set_ylabel("PDF")

        ax_y.set_ylim(y_lo, y_hi)
        ax_y.set_xlim(left=0.0)
        ax_y.invert_xaxis()
        ax_y.grid(True, alpha=0.3)
        ax_y.set_ylabel("log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
        ax_y.set_xlabel("PDF")

    legend_handles = [
        plt.Line2D([], [], color=color_for_name[n], lw=2.0, label=label_for_name[n])
        for n in PANEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    if last_im is not None:
        cbar_ax = fig.add_axes([0.96, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("log$_{10}$ frequency")
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if title:
        fig.suptitle(title, y=0.965)
    fig.savefig(out_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_2x2_joint_density(
    out_path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    x_label: str,
    *,
    title: str,
    bins: int = 60,
) -> None:
    """Plot a 2x2 joint density of (x_field, log_nw) for obs + 3 model runs.

    ``samples`` must hold keys ``obs``, ``control``, ``graupel``, ``2mom``.
    The QC'd observed core contour (0.5 % smoothed frequency) is overlaid
    on each of the three model panels.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Take per-dataset percentiles and span the union, so 1-mom's collapsed
    # log Nw line at log10(N0) doesn't fall outside the obs-driven range.
    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            x_los.append(float(np.nanpercentile(sx, 0.5)))
            x_his.append(float(np.nanpercentile(sx, 99.5)))
        if sy.size:
            y_los.append(float(np.nanpercentile(sy, 0.5)))
            y_his.append(float(np.nanpercentile(sy, 99.5)))
    if not x_los or not y_los:
        raise RuntimeError("No samples available to plot 2x2 joint density")

    x_lo = max(0.2, min(x_los))
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)

    obs_x = samples["obs"][x_field]
    obs_y = samples["obs"]["log_nw"]
    contour = _compute_obs_contour(obs_x, obs_y, x_edges, y_edges)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 9.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    for name in PANEL_ORDER:
        i, j = PANEL_GRID_POSITIONS[name]
        ax = axes[i, j]
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        label = "Obs" if name == "obs" else EXPERIMENT_LABELS.get(name, name)
        if sx.size == 0:
            ax.set_title(f"{label} (n=0)")
            continue
        h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
        h = h / max(1.0, h.sum())
        with np.errstate(divide="ignore"):
            shown = np.log10(np.where(h > 0, h, np.nan))
        ax.set_facecolor("white")
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("white")
        im = ax.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, shading="auto")
        ax.set_title(f"{label} ({_format_count(sx.size)})")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        last_im = im
        if contour is not None:
            xc, yc, field, levels = contour
            try:
                ax.contour(
                    xc,
                    yc,
                    field,
                    levels=levels,
                    colors="black",
                    linewidths=1.4,
                )
            except ValueError:
                pass
    for ax in axes[1, :]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
    fig.suptitle(title)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, location="right")
        cbar.set_label("log$_{10}$ frequency")
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.savefig(out_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
