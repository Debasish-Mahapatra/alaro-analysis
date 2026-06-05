"""
Quick test: joint vertical-velocity (w, m/s) vs height distribution of hydrometeors.

Unlike the main script which uses updraft flux (sigma * omega / g),
this converts omega (Pa/s) to true vertical velocity w = -omega / (rho * g),
where rho = P / (Rd * T).

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.updraft_hydrometeor_w
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures")

HYDROMETEORS = ["RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER"]
HYDRO_SHORT = {
    "RAIN": "Rain", "SNOW": "Snow", "GRAUPEL": "Graupel",
    "LIQUID_WATER": "Liquid water", "SOLID_WATER": "Ice",
}

# Physical constants
G  = 9.80665      # m/s²
RD = 287.05       # J/(kg·K), gas constant for dry air

# Binning: w in m/s (positive = upward)
W_BINS = np.linspace(-0.5, 10.0, 106)   # ~0.1 m/s bins
H_BINS = np.linspace(0.0, 20.0, 101)    # 0.2 km bins

MIN_LEAD_HOUR = 3
N_DAYS = 3

# Fonts
FS = 18
FS_TICK = 16
FS_TITLE = 20
FS_CBAR = 14
FS_MARG = 14
FS_LEG = 13

EXP_LINE_COLORS = {
    "control": "#d62728",
    "graupel": "#1f77b4",
    "2mom":    "#2ca02c",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_field(filepath: Path) -> np.ndarray:
    with xr.open_dataset(filepath, decode_times=False) as ds:
        var_name = list(ds.data_vars)[0]
        return ds[var_name].values[0]  # (level, y, x)


def list_steps(day_dir: Path) -> list[Path]:
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= MIN_LEAD_HOUR:
            steps.append(f)
    return steps


# ---------------------------------------------------------------------------
# Accumulate
# ---------------------------------------------------------------------------

def accumulate(experiment: str = "control") -> dict:
    exp_dir = DATA_ROOT / experiment
    nc_dir = exp_dir / "masked-netcdf"

    # Find days
    omega_dir = nc_dir / "UD_OMEGA"
    days = sorted(d for d in omega_dir.iterdir() if d.is_dir() and d.name.startswith("pf"))
    days = days[:N_DAYS]

    nw = len(W_BINS) - 1
    nh = len(H_BINS) - 1

    sums   = {h: np.zeros((nw, nh), dtype=np.float64) for h in HYDROMETEORS}
    counts = {h: np.zeros((nw, nh), dtype=np.float64) for h in HYDROMETEORS}
    freq   = np.zeros((nw, nh), dtype=np.float64)
    n_files = 0

    for di, day in enumerate(days):
        day_name = day.name
        steps = list_steps(day)
        for step_file in steps:
            step_name = step_file.name
            try:
                omega  = read_field(nc_dir / "UD_OMEGA" / day_name / step_name)
                mesh   = read_field(nc_dir / "UD_MESH_FRAC" / day_name / step_name)
                height = read_field(nc_dir / "GEOPOTENTIEL" / day_name / step_name)
                pressure = read_field(nc_dir / "PRESSURE" / day_name / step_name)
                temperature = read_field(nc_dir / "TEMPERATURE" / day_name / step_name)
            except Exception as e:
                print(f"  skip {day_name}/{step_name}: {e}")
                continue

            # Convert omega (Pa/s) to w (m/s): w = -omega / (rho * g)
            rho = pressure / (RD * temperature)
            w = -omega / (rho * G)

            h_km = height / 1000.0

            # Only where updraft is active
            mask = (mesh > 0) & np.isfinite(w) & np.isfinite(h_km)
            w_flat = w[mask]
            h_flat = h_km[mask]

            if len(w_flat) == 0:
                continue

            w_idx = np.digitize(w_flat, W_BINS) - 1
            h_idx = np.digitize(h_flat, H_BINS) - 1
            valid = (w_idx >= 0) & (w_idx < nw) & (h_idx >= 0) & (h_idx < nh)
            w_idx = w_idx[valid]
            h_idx = h_idx[valid]

            if len(w_idx) == 0:
                continue

            np.add.at(freq, (w_idx, h_idx), 1.0)

            for hvar in HYDROMETEORS:
                try:
                    hydro = read_field(nc_dir / hvar / day_name / step_name)
                except Exception:
                    continue
                h_vals = hydro[mask][valid]
                h_vals = np.maximum(h_vals, 0.0)
                np.add.at(sums[hvar], (w_idx, h_idx), h_vals)
                np.add.at(counts[hvar], (w_idx, h_idx), 1.0)

            n_files += 1

        print(f"  day {di+1}/{len(days)} done, {n_files} files total", flush=True)

    return {
        "sums": sums, "counts": counts, "freq": freq,
        "n_files": n_files, "w_bins": W_BINS, "h_bins": H_BINS,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _compute_mean(s, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(c > 0, s / c, np.nan)
    return m


def _marginal_profile(s, c, axis):
    with np.errstate(divide="ignore", invalid="ignore"):
        total_s = np.nansum(s, axis=axis)
        total_c = np.nansum(c, axis=axis)
        return np.where(total_c > 0, total_s / total_c, np.nan)


def plot(result: dict, output_path: Path):
    from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation

    w_bins = result["w_bins"]
    h_bins = result["h_bins"]
    freq = result["freq"]
    w_centers = 0.5 * (w_bins[:-1] + w_bins[1:])
    h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

    hydros = [h for h in HYDROMETEORS if result["sums"].get(h) is not None]
    nrows = len(hydros)

    plt.rcParams.update({
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
    })

    fig = plt.figure(figsize=(10, 6.5 * nrows))
    outer = gridspec.GridSpec(
        nrows, 1, figure=fig,
        hspace=0.45, left=0.10, right=0.88, top=0.97, bottom=0.05,
    )

    for row, hvar in enumerate(hydros):
        s = result["sums"][hvar]
        c = result["counts"][hvar]
        mean_h = _compute_mean(s, c)

        pos = mean_h[np.isfinite(mean_h) & (mean_h > 0)]
        if len(pos) == 0:
            continue
        vmin = max(np.percentile(pos, 2), 1e-12)
        vmax = np.percentile(pos, 98)
        cf_levels = np.geomspace(vmin, vmax, 20)

        # Marginals
        prof = _marginal_profile(s, c, axis=0)   # vs height
        wdist = _marginal_profile(s, c, axis=1)  # vs w

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[row],
            width_ratios=[5, 1.2], height_ratios=[5, 1.2],
            wspace=0.05, hspace=0.05,
        )

        # --- Main panel ---
        ax = fig.add_subplot(inner[0, 0])
        plot_data = np.where(np.isfinite(mean_h), mean_h, 0.0)
        cf = ax.contourf(
            w_centers, h_centers, plot_data.T,
            levels=cf_levels,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis",
        )

        fmax = np.nanmax(freq)
        if fmax > 10:
            clevels = 10.0 ** np.arange(1, int(np.log10(fmax)) + 1)
            clevels = clevels[clevels <= fmax]
            if len(clevels):
                ax.contour(
                    w_centers, h_centers, freq.T,
                    levels=clevels, colors="k", linewidths=0.8, alpha=0.45,
                )

        ax.set_xlim(w_bins[0], w_bins[-1])
        ax.set_ylim(0, 18)
        yticks = np.arange(0, 20, 2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(v)) for v in yticks], fontsize=FS_TICK)
        ax.set_ylabel("Height (km)", fontsize=FS)
        ax.set_xticklabels([])
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=FS_TICK, length=6, width=1.2)

        label = HYDRO_SHORT.get(hvar, hvar)
        ax.set_title(f"C1M  -  {label}", fontsize=FS_TITLE, fontweight="bold")

        # --- Right marginal ---
        ax_r = fig.add_subplot(inner[0, 1])
        ax_r.set_ylim(0, 18)
        ax_r.set_yticks(yticks)
        ax_r.set_yticklabels([])
        ax_r.plot(prof, h_centers, color=EXP_LINE_COLORS["control"], lw=2)
        ax_r.tick_params(axis="x", labelsize=FS_MARG, labelbottom=False, labeltop=True)
        ax_r.xaxis.set_major_locator(MaxNLocator(2))
        ax_r.set_xlim(left=0)
        ax_r.grid(False)

        # --- Bottom marginal ---
        ax_b = fig.add_subplot(inner[1, 0])
        ax_b.set_xlim(w_bins[0], w_bins[-1])
        ax_b.plot(w_centers, wdist, color=EXP_LINE_COLORS["control"], lw=2)
        ax_b.tick_params(axis="both", labelsize=FS_MARG)
        ax_b.yaxis.set_major_locator(MaxNLocator(3))
        ax_b.set_ylim(bottom=0)
        ax_b.grid(False)
        ax_b.set_xlabel(r"Vertical velocity $w$ (m s$^{-1}$)", fontsize=FS)

        # --- Empty corner ---
        ax_e = fig.add_subplot(inner[1, 1])
        ax_e.axis("off")

        # --- Colorbar ---
        fig.canvas.draw()
        pos_r = ax_r.get_position()
        cax = fig.add_axes([pos_r.x1 + 0.008, pos_r.y0, 0.012, pos_r.height])
        cb = fig.colorbar(cf, cax=cax, orientation="vertical")
        cb.set_label(f"{label} (kg kg$^{{-1}}$)", fontsize=FS_CBAR)
        cb.ax.tick_params(labelsize=FS_CBAR)
        cb.ax.yaxis.set_major_formatter(
            LogFormatterSciNotation(base=10, labelOnlyBase=True)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Accumulating 3 days of control (w in m/s)...", flush=True)
    result = accumulate("control")
    print(f"Done: {result['n_files']} files", flush=True)

    out = OUTPUT_DIR / "test_updraft_w_control_3days.png"
    plot(result, out)
