"""
Joint updraft-intensity vs height distributions of hydrometeor mixing ratios.

Inspired by Figure 9 of Van Weverberg et al. (2024, QJRMS):
  - x-axis: updraft vertical velocity w (m/s)
  - y-axis: height (km)
  - colour shading: mean hydrometeor mixing ratio in each (w, height) bin
  - contours: absolute frequency (number of grid-point samples)
  - marginal panels: vertical profile (right) and velocity distribution (bottom)

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.updraft_hydrometeor
"""

from __future__ import annotations

import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import cmaps
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")

EXPERIMENTS = {
    "control": "C1M",
    "graupel": "G1M",
    "2mom":    "G2M",
}

HYDROMETEORS = ["RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER"]

HYDRO_LABELS = {
    "RAIN":         "Rain (kg/kg)",
    "SNOW":         "Snow (kg/kg)",
    "GRAUPEL":      "Graupel (kg/kg)",
    "LIQUID_WATER": "Cloud liquid water (kg/kg)",
    "SOLID_WATER":  "Cloud ice (kg/kg)",
}

# Binning — height axis shared by all stratifications
H_BINS = np.linspace(0.0, 20.0, 101)       # 0.2 km bins (in km)

# Stratification configurations: bins + labels for each x-axis variable
STRATIFY_CONFIGS = {
    "flux": {
        "bins": np.linspace(-0.1, 3.0, 156),      # ~0.02 kg/m²/s
        "label": "Updraft flux",
        "unit": r"kg m$^{-2}$ s$^{-1}$",
        "coarse_edges": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "regression_range": (0.1, 2.5),
    },
    "extent": {
        "bins": np.linspace(0.0, 1.0, 101),        # 0.01 fraction
        "label": "Updraft extent",
        "unit": "fraction",
        "coarse_edges": np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        "regression_range": (0.02, 0.8),
    },
    "intensity": {
        "bins": np.linspace(0.0, 150.0, 151),      # 1.0 Pa/s
        "label": "Updraft intensity",
        "unit": "Pa/s",
        "coarse_edges": np.array([0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0]),
        "regression_range": (5.0, 120.0),
    },
}

# Backwards-compatible alias
FLUX_BINS = STRATIFY_CONFIGS["flux"]["bins"]

# Subsample: set to None to use all, or an int for max days
MAX_DAYS: int | None = None
# Only use timesteps with lead time >= 3h to skip spin-up
MIN_LEAD_HOUR = 3

# Physical constants
G = 9.80665

# Output
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures")
CACHE_DIR  = Path("/gpfs/me01/me/CLIMATE/CLIMATE/deba/alaro-analysis/cache/updraft_hydrometeor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_days(exp_dir: Path, variable: str) -> list[Path]:
    """Return sorted day directories for a variable."""
    var_dir = exp_dir / "masked-netcdf" / variable
    if not var_dir.exists():
        return []
    days = sorted(d for d in var_dir.iterdir() if d.is_dir() and d.name.startswith("pf"))
    return days


def list_steps(day_dir: Path, min_lead_hour: int = 3) -> list[Path]:
    """Return step files with lead time >= min_lead_hour."""
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        # filename like pfABOFABOF+0012.nc -> lead = 12
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= min_lead_hour:
            steps.append(f)
    return steps


def read_field(filepath: Path) -> np.ndarray:
    """Read a 4D (time, level, y, x) field, squeeze time."""
    with xr.open_dataset(filepath, decode_times=False) as ds:
        var_name = list(ds.data_vars)[0]
        return ds[var_name].values[0]  # (level, y, x)


def _compute_stratify_variable(omega, mesh, stratify):
    """Compute the x-axis variable from raw omega and mesh fields."""
    if stratify == "flux":
        return (-omega * mesh) / G
    if stratify == "extent":
        return mesh.copy()
    if stratify == "intensity":
        return np.abs(omega)
    raise ValueError(f"Unknown stratify mode: {stratify}")


def _process_day(args):
    """Process a single day directory. Called by multiprocessing workers."""
    exp_dir, day_dir, hydrometeors, min_lead_hour, stratify, x_bins = args
    day_name = day_dir.name
    steps = list_steps(day_dir, min_lead_hour)
    if not steps:
        return None

    nx = len(x_bins) - 1
    nh = len(H_BINS) - 1

    sums   = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    freq   = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            omega  = read_field(exp_dir / "masked-netcdf" / "UD_OMEGA" / day_name / step_name)
            mesh   = read_field(exp_dir / "masked-netcdf" / "UD_MESH_FRAC" / day_name / step_name)
            height = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
        except Exception:
            continue

        x_var = _compute_stratify_variable(omega, mesh, stratify)
        h_km = height / 1000.0

        # Mask: only where updraft is active (mesh > 0) and valid
        mask = (mesh > 0) & np.isfinite(x_var) & np.isfinite(h_km)
        x_flat = x_var[mask]
        h_flat = h_km[mask]

        if len(x_flat) == 0:
            continue

        x_idx = np.digitize(x_flat, x_bins) - 1
        h_idx = np.digitize(h_flat, H_BINS) - 1
        valid = (x_idx >= 0) & (x_idx < nx) & (h_idx >= 0) & (h_idx < nh)
        x_idx = x_idx[valid]
        h_idx = h_idx[valid]

        if len(x_idx) == 0:
            continue

        np.add.at(freq, (x_idx, h_idx), 1.0)

        for hvar in hydrometeors:
            try:
                hydro = read_field(exp_dir / "masked-netcdf" / hvar / day_name / step_name)
            except Exception:
                continue
            h_flat_var = hydro[mask][valid]
            h_flat_var = np.maximum(h_flat_var, 0.0)
            np.add.at(sums[hvar], (x_idx, h_idx), h_flat_var)
            np.add.at(counts[hvar], (x_idx, h_idx), 1.0)

        n_files += 1

    return {"sums": sums, "counts": counts, "freq": freq, "n_files": n_files}


N_WORKERS = 8


def accumulate_histograms(
    experiment: str,
    hydrometeors: list[str],
    max_days: int | None = None,
    stratify: str = "flux",
) -> dict:
    """
    Accumulate 2D histograms using multiprocessing over days.

    Parameters
    ----------
    stratify : str
        Stratification variable: ``"flux"``, ``"extent"``, or ``"intensity"``.
    """
    x_bins = STRATIFY_CONFIGS[stratify]["bins"]
    exp_dir = DATA_ROOT / experiment

    nx = len(x_bins) - 1
    nh = len(H_BINS) - 1

    days = list_days(exp_dir, "UD_OMEGA")
    if max_days is not None:
        days = days[:max_days]

    # Build task list — pass stratify mode and bins to workers
    tasks = [(exp_dir, d, hydrometeors, MIN_LEAD_HOUR, stratify, x_bins) for d in days]

    print(f"  {experiment} [{stratify}]: processing {len(days)} days with {N_WORKERS} workers...",
          flush=True)

    sums   = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    freq   = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0

    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_day, tasks)):
            if result is None:
                continue
            freq += result["freq"]
            n_files += result["n_files"]
            for h in hydrometeors:
                sums[h]   += result["sums"][h]
                counts[h] += result["counts"][h]
            if (i + 1) % 100 == 0:
                print(f"  {experiment} [{stratify}]: {i+1}/{len(days)} days done, "
                      f"{n_files} files", flush=True)

    print(f"  {experiment} [{stratify}]: DONE - {n_files} files from {len(days)} days", flush=True)

    return {
        "sums": sums,
        "counts": counts,
        "freq": freq,
        "n_files": n_files,
        "x_bins": x_bins,
        "h_bins": H_BINS,
        "stratify": stratify,
    }


def save_cache(experiment: str, result: dict, stratify: str = "flux"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if stratify == "flux" else f"_{stratify}"
    outpath = CACHE_DIR / f"{experiment}{suffix}.npz"
    save_dict = {
        "freq": result["freq"],
        "n_files": np.array(result["n_files"]),
        "x_bins": result["x_bins"],
        "h_bins": H_BINS,
        "stratify": np.array(stratify),
    }
    for hvar in result["sums"]:
        save_dict[f"sum_{hvar}"] = result["sums"][hvar]
        save_dict[f"cnt_{hvar}"] = result["counts"][hvar]
    np.savez_compressed(outpath, **save_dict)
    print(f"  Saved cache: {outpath}")


def load_cache(experiment: str, stratify: str = "flux") -> dict | None:
    suffix = "" if stratify == "flux" else f"_{stratify}"
    path = CACHE_DIR / f"{experiment}{suffix}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    hydros = [k.replace("sum_", "") for k in data.files if k.startswith("sum_")]
    # Backwards-compatible: old caches stored "flux_bins" instead of "x_bins"
    x_bins = data["x_bins"] if "x_bins" in data.files else data["flux_bins"]
    return {
        "sums":   {h: data[f"sum_{h}"] for h in hydros},
        "counts": {h: data[f"cnt_{h}"] for h in hydros},
        "freq":   data["freq"],
        "n_files": int(data["n_files"]),
        "x_bins": x_bins,
        "h_bins": data["h_bins"],
        "stratify": stratify,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Nice short labels for the hydrometeors (used in panel titles & colorbars)
HYDRO_SHORT = {
    "RAIN":         "Rain",
    "SNOW":         "Snow",
    "GRAUPEL":      "Graupel",
    "LIQUID_WATER": "Liquid water",
    "SOLID_WATER":  "Ice",
}


def _compute_mean(s, c):
    """Element-wise sum/count → mean, NaN where count==0."""
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _marginal_profile(s, c, axis):
    """Weighted-mean marginal along *axis* (0→profile over height, 1→over flux)."""
    with np.errstate(invalid="ignore"):
        total_s = np.nansum(s, axis=axis)
        total_c = np.nansum(c, axis=axis)
        return np.where(total_c > 0, total_s / total_c, np.nan)


# Fixed experiment colours for marginal lines
EXP_LINE_COLORS = {
    "control": "#d62728",   # red
    "graupel": "#1f77b4",   # blue
    "2mom":    "#2ca02c",   # green
}

# Big font size constant — used EVERYWHERE
FS = 22        # base font (axis labels)
FS_TICK = 20   # tick labels
FS_TITLE = 24  # panel titles
FS_CBAR = 20   # colorbar label + ticks
FS_MARG = 16   # marginal axis labels/ticks
FS_LEG = 16    # legend


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
    x_unit  = cfg["unit"]

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
            [f"{coarse_edges[k]:.2g}–{coarse_edges[k+1]:.2g}" for k in range(n_coarse)],
            fontsize=12,
        )
        ax.set_xlabel(f"{x_label} bin ({x_unit})", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{label} — binned means at constant {x_label.lower()}",
                      fontsize=16, fontweight="bold")
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
    x_unit  = cfg["unit"]
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
                x, y, s=12, color=EXP_LINE_COLORS[exp], alpha=0.5, zorder=2,
            )
            x_fit = np.linspace(regression_range[0], regression_range[1], 100)
            ax.plot(
                x_fit, intercept + slope * x_fit,
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
            fontsize=16, fontweight="bold",
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
):
    """
    Figure-9-style multi-panel plot using contourf (no smoothing).
    Each panel gets its own vertical colorbar. No triangles. Clean tick formatting.
    """
    from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nrows = len(hydrometeors)
    ncols = len(experiments)

    # ---- Global rcParams — BIG fonts everywhere ----
    plt.rcParams.update({
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
    })

    # Figure dimensions
    cell_w, cell_h = 8.5, 6.5
    fig = plt.figure(figsize=(cell_w * ncols + 3.0, cell_h * nrows + 3.0))

    outer = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        wspace=0.55,
        hspace=0.50,
        left=0.07, right=0.93, top=0.97, bottom=0.05,
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
                2, 2,
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
                    f_centers, h_centers, plot_data.T,
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
                        f_centers, h_centers, freq.T,
                        levels=clevels, colors="k",
                        linewidths=0.8, alpha=0.45,
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
                is_self = (other_exp == exp)
                ax_r.plot(
                    prof, h_centers,
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
                is_self = (other_exp == exp)
                ax_b.plot(
                    f_centers, fdist,
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

            ax_b.set_xlabel(r"Updraft flux (kg m$^{-2}$ s$^{-1}$)", fontsize=FS)

            # ============ EMPTY CORNER ============
            ax_e = fig.add_subplot(inner[1, 1])
            ax_e.axis("off")

            # ============ COLORBAR — per panel, vertical, next to right marginal ====
            fig.canvas.draw()
            pos_r = ax_r.get_position()
            cax = fig.add_axes([
                pos_r.x1 + 0.006,
                pos_r.y0,
                0.012,
                pos_r.height,
            ])
            cb = fig.colorbar(cf, cax=cax, orientation="vertical")
            # Explicit power-of-10 ticks
            import math
            log_lo = math.floor(math.log10(vmin))
            log_hi = math.ceil(math.log10(vmax))
            pow10_ticks = [10**p for p in range(log_lo, log_hi + 1)
                          if vmin <= 10**p <= vmax]
            if len(pow10_ticks) >= 2:
                cb.set_ticks(pow10_ticks)
                cb.set_ticklabels([f"$10^{{{int(math.log10(t))}}}$" for t in pow10_ticks])
            cb.ax.tick_params(labelsize=FS_MARG)
            cb.set_label(f"{label} (kg/kg)", fontsize=FS_MARG, rotation=270, labelpad=18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Updraft-hydrometeor joint distributions")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument("--hydrometeors", nargs="+", default=HYDROMETEORS)
    parser.add_argument("--max-days", type=int, default=MAX_DAYS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stratify", nargs="+",
                        default=list(STRATIFY_CONFIGS.keys()),
                        choices=list(STRATIFY_CONFIGS.keys()),
                        help="Stratification variable(s): flux, extent, intensity")
    args = parser.parse_args()

    stratifications = args.stratify

    for strat in stratifications:
        print(f"\n{'='*60}")
        print(f"  Stratification: {strat}")
        print(f"{'='*60}")

        results = {}
        for exp in args.experiments:
            print(f"Processing {exp} [{strat}]...")
            cached = None if args.no_cache else load_cache(exp, stratify=strat)
            if cached is not None:
                print(f"  Loaded from cache")
                results[exp] = cached
            else:
                r = accumulate_histograms(exp, args.hydrometeors,
                                          max_days=args.max_days, stratify=strat)
                save_cache(exp, r, stratify=strat)
                results[exp] = r

        outpath = Path(args.output) if args.output else OUTPUT_DIR / f"updraft_hydrometeor_{strat}.png"

        # Joint distribution figure (only for flux — original layout)
        if strat == "flux":
            plot_figure(results, args.hydrometeors, args.experiments, outpath)

        # Analysis 1: Binned means
        binned_path = OUTPUT_DIR / f"updraft_hydrometeor_{strat}_binned_means.png"
        plot_binned_means(results, args.hydrometeors, args.experiments, binned_path,
                          stratify=strat)

        # Analysis 2: Regression slopes
        regr_path = OUTPUT_DIR / f"updraft_hydrometeor_{strat}_regression_slopes.png"
        plot_regression_slopes(results, args.hydrometeors, args.experiments, regr_path,
                               stratify=strat)


if __name__ == "__main__":
    main()
