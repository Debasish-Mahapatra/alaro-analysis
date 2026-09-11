"""Updraft hydrometeor."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.updraft_hydrometeor import _compute_mean, _marginal_profile
from alaro_analysis.common.updraft_hydrometeor_settings import (
    EXP_LINE_COLORS,
    EXPERIMENTS,
    FS,
    FS_LEG,
    FS_MARG,
    FS_TICK,
    FS_TITLE,
    HYDRO_SHORT,
    STRATIFY_CONFIGS,
)


def plot_binned_means(
    results: dict[str, dict],
    hydrometeors: list[str],
    experiments: list[str],
    output_path: Path,
    stratify: str = "flux",
    coarse_edges: np.ndarray | None = None,
):
    """
    Analysis 1: Binned means at constant updraft.

    Stratify the data by coarse bins of a chosen updraft variable and compare
    mean hydrometeor content across schemes within each bin.  The bin acts as a
    control variable so that remaining differences are purely microphysical.

    Parameters
    ----------
    results : dict
        ``{experiment: result_dict}`` as returned by `accumulate_histograms`
        or `load_cache`.
    stratify : str
        Which variable was used for stratification
        (``"flux"``, ``"extent"``, ``"intensity"``).
    coarse_edges : array, optional
        Edges of the coarse bins.  Defaults taken from ``STRATIFY_CONFIGS``.
    """
    cfg = STRATIFY_CONFIGS[stratify]
    if coarse_edges is None:
        coarse_edges = cfg["coarse_edges"]
    x_label = cfg["label"]
    x_unit = cfg["unit"]

    n_coarse = len(coarse_edges) - 1
    n_vars = len(hydrometeors)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows), squeeze=False)

    bar_width = 0.8 / len(experiments)

    for i, hvar in enumerate(hydrometeors):
        r_idx, c_idx = divmod(i, ncols)
        ax = axes[r_idx, c_idx]
        x_positions = np.arange(n_coarse)

        for j, exp in enumerate(experiments):
            r = results[exp]
            fine_bins = r["x_bins"]
            s = r["sums"].get(hvar)
            c = r["counts"].get(hvar)
            if s is None:
                continue

            # Aggregate fine bins into coarse bins
            # s and c have shape (n_fine_x, n_height) — sum over height first
            s_x = np.nansum(s, axis=1)  # shape (n_fine_x,)
            c_x = np.nansum(c, axis=1)

            fine_centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])
            coarse_means = np.full(n_coarse, np.nan)
            for k in range(n_coarse):
                mask = (fine_centers >= coarse_edges[k]) & (fine_centers < coarse_edges[k + 1])
                total_s = s_x[mask].sum()
                total_c = c_x[mask].sum()
                if total_c > 0:
                    coarse_means[k] = total_s / total_c

            ax.bar(
                x_positions + j * bar_width,
                coarse_means,
                width=bar_width,
                color=EXP_LINE_COLORS[exp],
                label=EXPERIMENTS[exp],
                edgecolor="k",
                linewidth=0.5,
            )

        label = HYDRO_SHORT.get(hvar, hvar)
        ax.set_ylabel(f"Mean {label} (kg/kg)", fontsize=14)
        ax.set_xticks(x_positions + bar_width * (len(experiments) - 1) / 2)
        ax.set_xticklabels(
            [f"{coarse_edges[k]:.2g}–{coarse_edges[k + 1]:.2g}" for k in range(n_coarse)],
            fontsize=12,
        )
        ax.set_xlabel(f"{x_label} bin ({x_unit})", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(
            f"{label} — binned means at constant {x_label.lower()}", fontsize=16, fontweight="bold"
        )
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplot(s)
    for i in range(n_vars, nrows * ncols):
        r_idx, c_idx = divmod(i, ncols)
        axes[r_idx, c_idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved binned-means figure: {output_path}")
    plt.close(fig)


def plot_regression_slopes(
    results: dict[str, dict],
    hydrometeors: list[str],
    experiments: list[str],
    output_path: Path,
    stratify: str = "flux",
    regression_range: tuple[float, float] | None = None,
):
    """
    Analysis 2: Regression slope of hydrometeor content vs a stratification variable.

    For each scheme, fit  Hydrometeor = a + b × X  using the fine-bin marginal
    means (height-collapsed).  The slope *b* measures microphysical efficiency:
    how much condensate is retained per unit of dynamical work.  Slope
    differences *are* the microphysical fingerprint.

    Parameters
    ----------
    results : dict
        ``{experiment: result_dict}`` as returned by `accumulate_histograms`
        or `load_cache`.
    stratify : str
        Which variable was used for stratification
        (``"flux"``, ``"extent"``, ``"intensity"``).
    regression_range : tuple, optional
        (min, max) x-values to include in the regression.
        Defaults taken from ``STRATIFY_CONFIGS``.
    """
    cfg = STRATIFY_CONFIGS[stratify]
    x_label = cfg["label"]
    x_unit = cfg["unit"]
    if regression_range is None:
        regression_range = cfg["regression_range"]

    n_vars = len(hydrometeors)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows), squeeze=False)

    slope_table: dict[str, dict[str, float]] = {}  # hvar -> {exp: slope}

    for i, hvar in enumerate(hydrometeors):
        r_idx, c_idx = divmod(i, ncols)
        ax = axes[r_idx, c_idx]
        slope_table[hvar] = {}

        for exp in experiments:
            r = results[exp]
            fine_bins = r["x_bins"]
            s = r["sums"].get(hvar)
            c = r["counts"].get(hvar)
            if s is None:
                continue

            # Height-collapsed mean vs x
            s_x = np.nansum(s, axis=1)
            c_x = np.nansum(c, axis=1)
            fine_centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])

            with np.errstate(invalid="ignore"):
                mean_vs_x = np.where(c_x > 0, s_x / c_x, np.nan)

            # Select valid range for regression
            sel = (
                (fine_centers >= regression_range[0])
                & (fine_centers <= regression_range[1])
                & np.isfinite(mean_vs_x)
            )
            x = fine_centers[sel]
            y = mean_vs_x[sel]
            if len(x) < 3:
                continue

            # Weighted linear regression (weight by sample count)
            w = c_x[sel]
            coeffs = np.polyfit(x, y, 1, w=np.sqrt(w))
            slope, intercept = coeffs
            slope_table[hvar][exp] = slope

            # Plot data and fit
            ax.scatter(
                x,
                y,
                s=12,
                color=EXP_LINE_COLORS[exp],
                alpha=0.5,
                zorder=2,
            )
            x_fit = np.linspace(regression_range[0], regression_range[1], 100)
            ax.plot(
                x_fit,
                intercept + slope * x_fit,
                color=EXP_LINE_COLORS[exp],
                linewidth=2.5,
                label=f"{EXPERIMENTS[exp]}  slope={slope:.2e}",
                zorder=3,
            )

        label = HYDRO_SHORT.get(hvar, hvar)
        ax.set_ylabel(f"Mean {label} (kg/kg)", fontsize=14)
        ax.set_xlabel(f"{x_label} ({x_unit})", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(
            f"{label} — regression vs {x_label.lower()}",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(*regression_range)

    # Hide unused subplot(s)
    for i in range(n_vars, nrows * ncols):
        r_idx, c_idx = divmod(i, ncols)
        axes[r_idx, c_idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved regression-slopes figure: {output_path}")
    plt.close(fig)

    # Print slope summary
    print(f"\n=== Regression slopes (hydrometeor per unit {x_label.lower()}) ===")
    for hvar in hydrometeors:
        print(f"\n  {HYDRO_SHORT.get(hvar, hvar)}:")
        for exp, sl in slope_table.get(hvar, {}).items():
            print(f"    {EXPERIMENTS[exp]:8s}  slope = {sl:.4e}")
    print()


def plot_figure(
    results: dict[str, dict],
    hydrometeors: list[str],
    experiments: list[str],
    output_path: Path,
    stratify: str = "flux",
):
    """
    Figure-9-style multi-panel plot using contourf (no smoothing).
    Each panel gets its own vertical colorbar. No triangles. Clean tick formatting.
    """
    import cmaps
    from matplotlib.ticker import MaxNLocator

    nrows = len(hydrometeors)
    ncols = len(experiments)

    # ---- Global rcParams — BIG fonts everywhere ----
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.labelsize": FS,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEG,
        }
    )

    # Figure dimensions
    cell_w, cell_h = 8.5, 6.5
    fig = plt.figure(figsize=(cell_w * ncols + 3.0, cell_h * nrows + 3.0))

    outer = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.55,
        hspace=0.50,
        left=0.07,
        right=0.93,
        top=0.97,
        bottom=0.05,
    )

    for row, hvar in enumerate(hydrometeors):
        # Pre-compute marginals for ALL experiments
        marginals = {}
        for exp in experiments:
            r = results[exp]
            s = r["sums"].get(hvar)
            c = r["counts"].get(hvar)
            if s is None:
                marginals[exp] = (None, None)
                continue
            marginals[exp] = (
                _marginal_profile(s, c, axis=0),  # profile vs height
                _marginal_profile(s, c, axis=1),  # distribution vs flux
            )

        for col, exp in enumerate(experiments):
            r = results[exp]
            flux_bins = r.get("x_bins", r.get("flux_bins"))
            h_bins = r["h_bins"]
            freq = r["freq"]
            nf, nh = len(flux_bins) - 1, len(h_bins) - 1
            f_centers = 0.5 * (flux_bins[:-1] + flux_bins[1:])
            h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

            s = r["sums"].get(hvar, np.zeros((nf, nh)))
            c = r["counts"].get(hvar, np.zeros((nf, nh)))
            mean_hydro = _compute_mean(s, c)

            # Per-panel colour limits
            pos = mean_hydro[np.isfinite(mean_hydro) & (mean_hydro > 0)]
            if len(pos) == 0:
                vmin, vmax = 1e-10, 1e-3
            else:
                vmin = max(np.percentile(pos, 2), 1e-12)
                vmax = np.percentile(pos, 98)

            n_levels = 20
            cf_levels = np.geomspace(vmin, vmax, n_levels)

            # ---- inner 2x2 grid ----
            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer[row, col],
                width_ratios=[5, 1.2],
                height_ratios=[5, 1.2],
                wspace=0.05,
                hspace=0.05,
            )

            # ============ MAIN PANEL — contourf ============
            ax = fig.add_subplot(inner[0, 0])
            plot_data = np.where(np.isfinite(mean_hydro), mean_hydro, 0.0)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Log scale.*")
                cf = ax.contourf(
                    f_centers,
                    h_centers,
                    plot_data.T,
                    levels=cf_levels,
                    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                    cmap=cmaps.WhiteBlueGreenYellowRed,
                )

            # Frequency contours (powers of 10)
            fmax = np.nanmax(freq)
            if fmax > 10:
                clevels = 10.0 ** np.arange(1, int(np.log10(fmax)) + 1)
                clevels = clevels[clevels <= fmax]
                if len(clevels):
                    ax.contour(
                        f_centers,
                        h_centers,
                        freq.T,
                        levels=clevels,
                        colors="k",
                        linewidths=0.8,
                        alpha=0.45,
                    )

            ax.set_xlim(flux_bins[0], flux_bins[-1])
            ax.set_ylim(0, 18)
            ytick_vals = np.arange(0, 20, 2)
            ax.set_yticks(ytick_vals)
            ax.set_xticklabels([])
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=FS_TICK, length=6, width=1.2)

            if col == 0:
                ax.set_yticklabels([str(int(v)) for v in ytick_vals], fontsize=FS_TICK)
                ax.set_ylabel("Height (km)", fontsize=FS)
            else:
                ax.set_yticklabels([])

            label = HYDRO_SHORT.get(hvar, hvar)
            ax.set_title(f"{EXPERIMENTS[exp]}  -  {label}", fontsize=FS_TITLE, fontweight="bold")

            # ============ RIGHT MARGINAL (profile) ============
            ax_r = fig.add_subplot(inner[0, 1])
            ax_r.set_ylim(0, 18)
            for other_exp in experiments:
                prof, _ = marginals[other_exp]
                if prof is None:
                    continue
                is_self = other_exp == exp
                ax_r.plot(
                    prof,
                    h_centers,
                    color=EXP_LINE_COLORS[other_exp],
                    linewidth=2.5 if is_self else 1.2,
                    linestyle="-" if is_self else "--",
                    alpha=1.0 if is_self else 0.7,
                    zorder=3 if is_self else 1,
                    label=EXPERIMENTS[other_exp],
                )
            ax_r.set_yticklabels([])
            ax_r.set_yticks(ytick_vals)
            ax_r.tick_params(axis="x", labelsize=FS_MARG, labelbottom=False, labeltop=True)
            ax_r.xaxis.set_major_locator(MaxNLocator(2))
            ax_r.set_xlim(left=0)
            ax_r.grid(False)
            if row == 0 and col == ncols - 1:
                ax_r.legend(fontsize=FS_LEG, loc="upper right", framealpha=0.8)

            # ============ BOTTOM MARGINAL (flux distribution) ============
            ax_b = fig.add_subplot(inner[1, 0])
            ax_b.set_xlim(flux_bins[0], flux_bins[-1])
            for other_exp in experiments:
                _, fdist = marginals[other_exp]
                if fdist is None:
                    continue
                is_self = other_exp == exp
                ax_b.plot(
                    f_centers,
                    fdist,
                    color=EXP_LINE_COLORS[other_exp],
                    linewidth=2.5 if is_self else 1.2,
                    linestyle="-" if is_self else "--",
                    alpha=1.0 if is_self else 0.7,
                    zorder=3 if is_self else 1,
                )
            ax_b.tick_params(axis="both", labelsize=FS_MARG)
            ax_b.yaxis.set_major_locator(MaxNLocator(3))
            ax_b.set_ylim(bottom=0)
            ax_b.grid(False)
            if col > 0:
                ax_b.set_yticklabels([])
            else:
                ax_b.set_ylabel(f"{label}\n(kg kg$^{{-1}}$)", fontsize=FS_MARG)

            _cfg = STRATIFY_CONFIGS.get(stratify, STRATIFY_CONFIGS["flux"])
            ax_b.set_xlabel(f"{_cfg['label']} ({_cfg['unit']})", fontsize=FS)

            # ============ EMPTY CORNER ============
            ax_e = fig.add_subplot(inner[1, 1])
            ax_e.axis("off")

            # ============ COLORBAR — per panel, vertical, next to right marginal ====
            fig.canvas.draw()
            pos_r = ax_r.get_position()
            cax = fig.add_axes(
                [
                    pos_r.x1 + 0.006,
                    pos_r.y0,
                    0.012,
                    pos_r.height,
                ]
            )
            cb = fig.colorbar(cf, cax=cax, orientation="vertical")
            # Explicit power-of-10 ticks
            import math

            log_lo = math.floor(math.log10(vmin))
            log_hi = math.ceil(math.log10(vmax))
            pow10_ticks = [10**p for p in range(log_lo, log_hi + 1) if vmin <= 10**p <= vmax]
            if len(pow10_ticks) >= 2:
                cb.set_ticks(pow10_ticks)
                cb.set_ticklabels([f"$10^{{{int(math.log10(t))}}}$" for t in pow10_ticks])
            cb.ax.tick_params(labelsize=FS_MARG)
            cb.set_label(f"{label} (kg/kg)", fontsize=FS_MARG, rotation=270, labelpad=18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close(fig)
