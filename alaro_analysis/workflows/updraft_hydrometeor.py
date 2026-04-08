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

# Binning
FLUX_BINS = np.linspace(-0.1, 3.0, 156)    # ~0.02 kg/m²/s bins
H_BINS = np.linspace(0.0, 20.0, 101)       # 0.2 km bins (in km)

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


def _process_day(args):
    """Process a single day directory. Called by multiprocessing workers."""
    exp_dir, day_dir, hydrometeors, min_lead_hour = args
    day_name = day_dir.name
    steps = list_steps(day_dir, min_lead_hour)
    if not steps:
        return None

    nf = len(FLUX_BINS) - 1
    nh = len(H_BINS) - 1

    sums   = {h: np.zeros((nf, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nf, nh), dtype=np.float64) for h in hydrometeors}
    freq   = np.zeros((nf, nh), dtype=np.float64)
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            omega  = read_field(exp_dir / "masked-netcdf" / "UD_OMEGA" / day_name / step_name)
            mesh   = read_field(exp_dir / "masked-netcdf" / "UD_MESH_FRAC" / day_name / step_name)
            height = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
        except Exception:
            continue

        # Updraft mass flux: M_u = sigma_u * (-omega_u) / g  [kg/m²/s]
        flux = (-omega * mesh) / G
        h_km = height / 1000.0

        # Mask: only where updraft is active (mesh > 0) and valid
        mask = (mesh > 0) & np.isfinite(flux) & np.isfinite(h_km)
        flux_flat = flux[mask]
        h_flat = h_km[mask]

        if len(flux_flat) == 0:
            continue

        f_idx = np.digitize(flux_flat, FLUX_BINS) - 1
        h_idx = np.digitize(h_flat, H_BINS) - 1
        valid = (f_idx >= 0) & (f_idx < nf) & (h_idx >= 0) & (h_idx < nh)
        f_idx = f_idx[valid]
        h_idx = h_idx[valid]

        if len(f_idx) == 0:
            continue

        np.add.at(freq, (f_idx, h_idx), 1.0)

        for hvar in hydrometeors:
            try:
                hydro = read_field(exp_dir / "masked-netcdf" / hvar / day_name / step_name)
            except Exception:
                continue
            h_flat_var = hydro[mask][valid]
            h_flat_var = np.maximum(h_flat_var, 0.0)
            np.add.at(sums[hvar], (f_idx, h_idx), h_flat_var)
            np.add.at(counts[hvar], (f_idx, h_idx), 1.0)

        n_files += 1

    return {"sums": sums, "counts": counts, "freq": freq, "n_files": n_files}


N_WORKERS = 8


def accumulate_histograms(
    experiment: str,
    hydrometeors: list[str],
    max_days: int | None = None,
) -> dict:
    """
    Accumulate 2D histograms using multiprocessing over days.
    """
    exp_dir = DATA_ROOT / experiment

    nf = len(FLUX_BINS) - 1
    nh = len(H_BINS) - 1

    days = list_days(exp_dir, "UD_OMEGA")
    if max_days is not None:
        days = days[:max_days]

    # Build task list
    tasks = [(exp_dir, d, hydrometeors, MIN_LEAD_HOUR) for d in days]

    print(f"  {experiment}: processing {len(days)} days with {N_WORKERS} workers...", flush=True)

    sums   = {h: np.zeros((nf, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nf, nh), dtype=np.float64) for h in hydrometeors}
    freq   = np.zeros((nf, nh), dtype=np.float64)
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
                print(f"  {experiment}: {i+1}/{len(days)} days done, {n_files} files", flush=True)

    print(f"  {experiment}: DONE - {n_files} files from {len(days)} days", flush=True)

    return {
        "sums": sums,
        "counts": counts,
        "freq": freq,
        "n_files": n_files,
        "flux_bins": FLUX_BINS,
        "h_bins": H_BINS,
    }


def save_cache(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    outpath = CACHE_DIR / f"{experiment}.npz"
    save_dict = {
        "freq": result["freq"],
        "n_files": np.array(result["n_files"]),
        "flux_bins": FLUX_BINS,
        "h_bins": H_BINS,
    }
    for hvar in result["sums"]:
        save_dict[f"sum_{hvar}"] = result["sums"][hvar]
        save_dict[f"cnt_{hvar}"] = result["counts"][hvar]
    np.savez_compressed(outpath, **save_dict)
    print(f"  Saved cache: {outpath}")


def load_cache(experiment: str) -> dict | None:
    path = CACHE_DIR / f"{experiment}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    hydros = [k.replace("sum_", "") for k in data.files if k.startswith("sum_")]
    return {
        "sums":   {h: data[f"sum_{h}"] for h in hydros},
        "counts": {h: data[f"cnt_{h}"] for h in hydros},
        "freq":   data["freq"],
        "n_files": int(data["n_files"]),
        "flux_bins": data["flux_bins"],
        "h_bins": data["h_bins"],
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
            flux_bins = r["flux_bins"]
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
    args = parser.parse_args()

    results = {}
    for exp in args.experiments:
        print(f"Processing {exp}...")
        cached = None if args.no_cache else load_cache(exp)
        if cached is not None:
            print(f"  Loaded from cache")
            results[exp] = cached
        else:
            r = accumulate_histograms(exp, args.hydrometeors, max_days=args.max_days)
            save_cache(exp, r)
            results[exp] = r

    outpath = Path(args.output) if args.output else OUTPUT_DIR / "updraft_hydrometeor_joint.png"
    plot_figure(results, args.hydrometeors, args.experiments, outpath)


if __name__ == "__main__":
    main()
