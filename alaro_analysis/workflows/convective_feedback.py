"""
Convective feedback diagnostics: loading, downdrafts, and moisture recycling.

Three hypotheses for how microphysics feeds back on convective dynamics:

  H1 — Condensate loading:  heavy hydrometeors weigh down updrafts.
        Test: updraft intensity conditioned on total condensate mass.

  H2 — Downdraft feedback:  graupel evaporation drives stronger downdrafts
        that suppress subsequent convection.
        Test: downdraft flux / intensity / extent comparison across schemes.

  H3 — Cumulative moisture misplacement:  graupel evaporates above the BL,
        depositing moisture that doesn't recycle, leaving the next day drier.
        Test: pre-convection (06–09 LT) thermodynamic profiles.

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.convective_feedback
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration (shared with updraft_hydrometeor)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures")
CACHE_DIR = Path(
    "/gpfs/me01/me/CLIMATE/CLIMATE/deba/alaro-analysis/cache/convective_feedback"
)

EXPERIMENTS = {
    "control": "C1M",
    "graupel": "G1M",
    "2mom":    "G2M",
}

EXP_COLORS = {
    "control": "#d62728",
    "graupel": "#1f77b4",
    "2mom":    "#2ca02c",
}

HYDROMETEORS = ["RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER"]

G = 9.80665
MIN_LEAD_HOUR = 3
N_WORKERS = 24


# ---------------------------------------------------------------------------
# Shared I/O helpers (same as updraft_hydrometeor)
# ---------------------------------------------------------------------------

def list_days(exp_dir: Path, variable: str) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / variable
    if not var_dir.exists():
        return []
    return sorted(d for d in var_dir.iterdir() if d.is_dir() and d.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int = 3) -> list[Path]:
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= min_lead_hour:
            steps.append(f)
    return steps


def list_steps_in_range(
    day_dir: Path, lead_min: int, lead_max: int,
) -> list[Path]:
    """Return step files with lead_min <= lead <= lead_max."""
    steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead_min <= lead <= lead_max:
            steps.append(f)
    return steps


def read_field(filepath: Path) -> np.ndarray:
    """Read a 4D (time, level, y, x) field, squeeze time → (level, y, x)."""
    with xr.open_dataset(filepath, decode_times=False) as ds:
        var_name = list(ds.data_vars)[0]
        return ds[var_name].values[0]


# =========================================================================
# H1 — CONDENSATE LOADING
# =========================================================================
#
# Question: at the same total condensate loading, do G1M updrafts differ
# from C1M/G2M?  If yes → loading is NOT the cause (something else is).
# If no → loading IS the mechanism.
#
# Approach: for each grid point where updraft is active, compute:
#   x = total condensate = QL + QI + QR + QS + QG  (kg/kg)
#   y = updraft intensity = |UD_OMEGA|  (Pa/s)
# Bin by x, compute mean y in each bin → regression.
# =========================================================================

LOADING_BINS = np.linspace(0.0, 0.01, 101)  # 0–10 g/kg in 0.1 g/kg steps
H_BINS = np.linspace(0.0, 20.0, 101)


def _h1_process_day(args):
    """Accumulate updraft intensity binned by condensate loading × height."""
    exp_dir, day_dir, min_lead_hour = args
    day_name = day_dir.name
    steps = list_steps(day_dir, min_lead_hour)
    if not steps:
        return None

    nx = len(LOADING_BINS) - 1
    nh = len(H_BINS) - 1

    # We accumulate updraft intensity (|omega|) and flux binned by loading
    sums_intensity = np.zeros((nx, nh), dtype=np.float64)
    counts         = np.zeros((nx, nh), dtype=np.float64)
    sums_flux      = np.zeros((nx, nh), dtype=np.float64)
    freq           = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            omega  = read_field(exp_dir / "masked-netcdf" / "UD_OMEGA"     / day_name / step_name)
            mesh   = read_field(exp_dir / "masked-netcdf" / "UD_MESH_FRAC" / day_name / step_name)
            height = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
        except Exception:
            continue

        # Total condensate loading
        total_condensate = np.zeros_like(omega)
        for hvar in HYDROMETEORS:
            try:
                hfield = read_field(exp_dir / "masked-netcdf" / hvar / day_name / step_name)
                total_condensate += np.maximum(hfield, 0.0)
            except Exception:
                pass

        h_km = height / 1000.0
        mask = (mesh > 0) & np.isfinite(omega) & np.isfinite(h_km) & np.isfinite(total_condensate)

        tc_flat = total_condensate[mask]
        h_flat  = h_km[mask]
        omega_abs_flat = np.abs(omega[mask])
        flux_flat = (-omega[mask] * mesh[mask]) / G

        if len(tc_flat) == 0:
            continue

        x_idx = np.digitize(tc_flat, LOADING_BINS) - 1
        h_idx = np.digitize(h_flat, H_BINS) - 1
        valid = (x_idx >= 0) & (x_idx < nx) & (h_idx >= 0) & (h_idx < nh)
        x_idx = x_idx[valid]
        h_idx = h_idx[valid]

        if len(x_idx) == 0:
            continue

        np.add.at(freq, (x_idx, h_idx), 1.0)
        np.add.at(sums_intensity, (x_idx, h_idx), omega_abs_flat[valid])
        np.add.at(sums_flux, (x_idx, h_idx), flux_flat[valid])
        np.add.at(counts, (x_idx, h_idx), 1.0)
        n_files += 1

    return {
        "sums_intensity": sums_intensity,
        "sums_flux": sums_flux,
        "counts": counts,
        "freq": freq,
        "n_files": n_files,
    }


def accumulate_h1(experiment: str, max_days: int | None = None) -> dict:
    exp_dir = DATA_ROOT / experiment
    days = list_days(exp_dir, "UD_OMEGA")
    if max_days is not None:
        days = days[:max_days]

    tasks = [(exp_dir, d, MIN_LEAD_HOUR) for d in days]
    print(f"  H1 {experiment}: processing {len(days)} days ...", flush=True)

    nx = len(LOADING_BINS) - 1
    nh = len(H_BINS) - 1
    sums_intensity = np.zeros((nx, nh), dtype=np.float64)
    sums_flux      = np.zeros((nx, nh), dtype=np.float64)
    counts         = np.zeros((nx, nh), dtype=np.float64)
    freq           = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0

    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_h1_process_day, tasks)):
            if res is None:
                continue
            sums_intensity += res["sums_intensity"]
            sums_flux      += res["sums_flux"]
            counts         += res["counts"]
            freq           += res["freq"]
            n_files        += res["n_files"]
            if (i + 1) % 100 == 0:
                print(f"  H1 {experiment}: {i+1}/{len(days)} days", flush=True)

    print(f"  H1 {experiment}: DONE — {n_files} files", flush=True)
    return {
        "sums_intensity": sums_intensity,
        "sums_flux": sums_flux,
        "counts": counts,
        "freq": freq,
        "n_files": n_files,
        "x_bins": LOADING_BINS,
        "h_bins": H_BINS,
    }


def save_h1(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"h1_{experiment}.npz"
    np.savez_compressed(
        path,
        sums_intensity=result["sums_intensity"],
        sums_flux=result["sums_flux"],
        counts=result["counts"],
        freq=result["freq"],
        n_files=np.array(result["n_files"]),
        x_bins=result["x_bins"],
        h_bins=result["h_bins"],
    )
    print(f"  Saved: {path}")


def load_h1(experiment: str) -> dict | None:
    path = CACHE_DIR / f"h1_{experiment}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {
        "sums_intensity": d["sums_intensity"],
        "sums_flux": d["sums_flux"],
        "counts": d["counts"],
        "freq": d["freq"],
        "n_files": int(d["n_files"]),
        "x_bins": d["x_bins"],
        "h_bins": d["h_bins"],
    }


def plot_h1(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H1 plot: updraft intensity & flux vs total condensate loading.

    Two rows: (a) mean |omega| vs loading,  (b) mean updraft flux vs loading.
    Height-collapsed (marginal over all heights).
    Left column: regression, Right column: binned means.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)

    loading_coarse = np.array([0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01])

    for row, (target, ylabel) in enumerate([
        ("intensity", r"Mean updraft intensity |$\omega$| (Pa/s)"),
        ("flux", r"Mean updraft flux (kg m$^{-2}$ s$^{-1}$)"),
    ]):
        ax_reg = axes[row, 0]
        ax_bar = axes[row, 1]

        sums_key = f"sums_{target}"
        bar_width = 0.8 / len(experiments)
        n_coarse = len(loading_coarse) - 1
        x_positions = np.arange(n_coarse)

        for j, exp in enumerate(experiments):
            r = results[exp]
            fine_bins = r["x_bins"]
            s = np.nansum(r[sums_key], axis=1)
            c = np.nansum(r["counts"], axis=1)
            fine_centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])

            with np.errstate(invalid="ignore"):
                mean_y = np.where(c > 0, s / c, np.nan)

            # Regression
            sel = np.isfinite(mean_y) & (fine_centers > 0.0002) & (fine_centers < 0.008)
            x, y = fine_centers[sel], mean_y[sel]
            if len(x) >= 3:
                w = c[sel]
                slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
                ax_reg.scatter(x * 1000, y, s=12, color=EXP_COLORS[exp], alpha=0.5)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax_reg.plot(
                    x_fit * 1000, intercept + slope * x_fit,
                    color=EXP_COLORS[exp], linewidth=2.5,
                    label=f"{EXPERIMENTS[exp]}  slope={slope:.1f}",
                )

            # Binned means
            coarse_means = np.full(n_coarse, np.nan)
            for k in range(n_coarse):
                mask = (fine_centers >= loading_coarse[k]) & (fine_centers < loading_coarse[k + 1])
                ts, tc = s[mask].sum(), c[mask].sum()
                if tc > 0:
                    coarse_means[k] = ts / tc
            ax_bar.bar(
                x_positions + j * bar_width, coarse_means,
                width=bar_width, color=EXP_COLORS[exp],
                label=EXPERIMENTS[exp], edgecolor="k", linewidth=0.5,
            )

        ax_reg.set_xlabel("Total condensate (g/kg)", fontsize=14)
        ax_reg.set_ylabel(ylabel, fontsize=14)
        ax_reg.legend(fontsize=11, loc="upper left")
        ax_reg.grid(alpha=0.3)
        ax_reg.set_title(
            f"H1: {target.capitalize()} vs loading — regression",
            fontsize=16, fontweight="bold",
        )

        ax_bar.set_xticks(x_positions + bar_width * (len(experiments) - 1) / 2)
        ax_bar.set_xticklabels(
            [f"{loading_coarse[k]*1000:.1f}–{loading_coarse[k+1]*1000:.1f}"
             for k in range(n_coarse)], fontsize=11,
        )
        ax_bar.set_xlabel("Condensate loading bin (g/kg)", fontsize=14)
        ax_bar.set_ylabel(ylabel, fontsize=14)
        ax_bar.legend(fontsize=11)
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_title(
            f"H1: {target.capitalize()} vs loading — binned means",
            fontsize=16, fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H1 figure: {output_path}")
    plt.close(fig)


# =========================================================================
# H2 — DOWNDRAFT FEEDBACK
# =========================================================================
#
# Compare downdraft diagnostics (flux, intensity, extent) across schemes.
# Downdraft flux = (-DD_OMEGA × DD_MESH_FRAC) / g  (but DD_OMEGA > 0 for
# downdrafts in ALARO convention, so flux = (DD_OMEGA × DD_MESH_FRAC) / g).
#
# Approach: compute diurnal profiles of downdraft properties, compare.
# Also: downdraft properties conditioned on updraft flux bins (same event).
# =========================================================================

DD_H_BINS = np.linspace(0.0, 20.0, 101)


def _h2_process_day(args):
    """Accumulate downdraft flux, intensity, extent profiles by hour and height."""
    exp_dir, day_dir = args
    day_name = day_dir.name
    all_steps = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= MIN_LEAD_HOUR:
            all_steps.append((lead, f))

    if not all_steps:
        return None

    nh = len(DD_H_BINS) - 1

    # Accumulate by lead hour: sums and counts for flux, intensity, extent
    # Also accumulate updraft flux alongside for conditioning
    hourly = {}

    for lead, step_file in all_steps:
        step_name = step_file.name
        try:
            dd_omega = read_field(exp_dir / "masked-netcdf" / "DD_OMEGA"     / day_name / step_name)
            dd_mesh  = read_field(exp_dir / "masked-netcdf" / "DD_MESH_FRAC" / day_name / step_name)
            height   = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
        except Exception:
            continue

        h_km = height / 1000.0

        # Downdraft mask: where downdraft is active
        dd_active = (dd_mesh > 0) & np.isfinite(dd_omega) & np.isfinite(h_km)

        if not np.any(dd_active):
            continue

        # Downdraft flux: DD_OMEGA is positive for subsidence in ALARO
        # so flux = (dd_omega * dd_mesh) / g
        dd_flux = np.where(dd_active, (np.abs(dd_omega) * dd_mesh) / G, 0.0)
        dd_intensity = np.where(dd_active, np.abs(dd_omega), np.nan)

        # Profile by height: average over (y, x)
        if lead not in hourly:
            hourly[lead] = {
                "flux_sum": np.zeros(nh, dtype=np.float64),
                "flux_cnt": np.zeros(nh, dtype=np.float64),
                "intensity_sum": np.zeros(nh, dtype=np.float64),
                "intensity_cnt": np.zeros(nh, dtype=np.float64),
                "extent_sum": np.zeros(nh, dtype=np.float64),
                "extent_cnt": np.zeros(nh, dtype=np.float64),
                "n": 0,
            }

        for lev in range(dd_omega.shape[0]):
            h_val = np.nanmean(h_km[lev])
            if not np.isfinite(h_val):
                continue
            h_idx = np.digitize(h_val, DD_H_BINS) - 1
            if h_idx < 0 or h_idx >= nh:
                continue

            # Extent: fraction of grid points with active downdraft
            lev_mask = dd_active[lev]
            n_total = np.sum(np.isfinite(dd_omega[lev]))
            if n_total == 0:
                continue
            extent_val = np.sum(lev_mask) / n_total

            hourly[lead]["extent_sum"][h_idx] += extent_val
            hourly[lead]["extent_cnt"][h_idx] += 1.0

            # Mean flux and intensity where active
            if np.any(lev_mask):
                flux_mean = np.nanmean(dd_flux[lev][lev_mask])
                int_mean  = np.nanmean(dd_intensity[lev][lev_mask])
                if np.isfinite(flux_mean):
                    hourly[lead]["flux_sum"][h_idx] += flux_mean
                    hourly[lead]["flux_cnt"][h_idx] += 1.0
                if np.isfinite(int_mean):
                    hourly[lead]["intensity_sum"][h_idx] += int_mean
                    hourly[lead]["intensity_cnt"][h_idx] += 1.0

        hourly[lead]["n"] += 1

    return hourly


def accumulate_h2(experiment: str, max_days: int | None = None) -> dict:
    exp_dir = DATA_ROOT / experiment
    days = list_days(exp_dir, "DD_OMEGA")
    if max_days is not None:
        days = days[:max_days]

    tasks = [(exp_dir, d) for d in days]
    print(f"  H2 {experiment}: processing {len(days)} days ...", flush=True)

    nh = len(DD_H_BINS) - 1
    # 24 hours × nh height bins
    profiles = {}
    for var in ("flux", "intensity", "extent"):
        profiles[f"{var}_sum"] = np.zeros((24, nh), dtype=np.float64)
        profiles[f"{var}_cnt"] = np.zeros((24, nh), dtype=np.float64)
    n_files = 0

    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_h2_process_day, tasks)):
            if res is None:
                continue
            for lead, data in res.items():
                hour = lead % 24
                for var in ("flux", "intensity", "extent"):
                    profiles[f"{var}_sum"][hour] += data[f"{var}_sum"]
                    profiles[f"{var}_cnt"][hour] += data[f"{var}_cnt"]
                n_files += data["n"]
            if (i + 1) % 100 == 0:
                print(f"  H2 {experiment}: {i+1}/{len(days)} days", flush=True)

    print(f"  H2 {experiment}: DONE — {n_files} files", flush=True)
    profiles["n_files"] = n_files
    profiles["h_bins"] = DD_H_BINS
    return profiles


def save_h2(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"h2_{experiment}.npz"
    save_dict = {k: np.array(v) for k, v in result.items()}
    np.savez_compressed(path, **save_dict)
    print(f"  Saved: {path}")


def load_h2(experiment: str) -> dict | None:
    path = CACHE_DIR / f"h2_{experiment}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def plot_h2(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H2 plot: downdraft diagnostics comparison.

    3 columns (flux, intensity, extent) × 2 rows:
      Top row: height profile (averaged over all hours)
      Bottom row: diurnal cycle (averaged over 0–6 km, the low-level downdraft layer)
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 11), squeeze=False)

    for col, (var, ylabel_prof, ylabel_diur) in enumerate([
        ("flux", r"Downdraft flux (kg m$^{-2}$ s$^{-1}$)", r"Downdraft flux (kg m$^{-2}$ s$^{-1}$)"),
        ("intensity", r"Downdraft intensity |$\omega$| (Pa/s)", r"|$\omega$| (Pa/s)"),
        ("extent", "Downdraft extent (fraction)", "Extent (fraction)"),
    ]):
        ax_prof = axes[0, col]
        ax_diur = axes[1, col]

        for exp in experiments:
            r = results[exp]
            h_bins = r["h_bins"]
            h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
            s = r[f"{var}_sum"]  # (24, nh)
            c = r[f"{var}_cnt"]

            # Height profile: average over all hours
            s_all = np.nansum(s, axis=0)
            c_all = np.nansum(c, axis=0)
            with np.errstate(invalid="ignore"):
                profile = np.where(c_all > 0, s_all / c_all, np.nan)

            ax_prof.plot(
                profile, h_centers,
                color=EXP_COLORS[exp], linewidth=2.5, label=EXPERIMENTS[exp],
            )

            # Diurnal cycle: average over 0–6 km
            low_mask = h_centers <= 6.0
            s_low = np.nansum(s[:, low_mask], axis=1)
            c_low = np.nansum(c[:, low_mask], axis=1)
            with np.errstate(invalid="ignore"):
                diurnal = np.where(c_low > 0, s_low / c_low, np.nan)

            ax_diur.plot(
                np.arange(24), diurnal,
                color=EXP_COLORS[exp], linewidth=2.5, label=EXPERIMENTS[exp],
            )

        ax_prof.set_ylabel("Height (km)", fontsize=14)
        ax_prof.set_xlabel(ylabel_prof, fontsize=14)
        ax_prof.set_ylim(0, 18)
        ax_prof.legend(fontsize=12)
        ax_prof.grid(alpha=0.3)
        ax_prof.set_title(f"H2: Downdraft {var} profile", fontsize=16, fontweight="bold")

        ax_diur.set_xlabel("Hour (UTC)", fontsize=14)
        ax_diur.set_ylabel(ylabel_diur, fontsize=14)
        ax_diur.set_xlim(0, 23)
        ax_diur.set_xticks(np.arange(0, 24, 3))
        ax_diur.legend(fontsize=12)
        ax_diur.grid(alpha=0.3)
        ax_diur.set_title(
            f"H2: Downdraft {var} diurnal (0–6 km)", fontsize=16, fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H2 figure: {output_path}")
    plt.close(fig)


# =========================================================================
# H3 — PRE-CONVECTION MOISTURE PROFILES
# =========================================================================
#
# Compare morning (lead hours 6–9, roughly 06–09 LT) profiles of:
#   - Specific humidity (HUMI.SPECIFI)
#   - Relative humidity (HUMI.RELATIVE)
#   - Temperature (TEMPERATURE)
#
# If G1M is already drier each morning before convection starts, the
# feedback is cumulative across days (moisture misplacement), not within-
# event (loading/downdrafts).
# =========================================================================

THERMO_VARS = {
    "HUMI.SPECIFI": {"label": "Specific humidity", "unit": "g/kg", "scale": 1000.0},
    "HUMI.RELATIVE": {"label": "Relative humidity", "unit": "%", "scale": 100.0},
    "TEMPERATURE": {"label": "Temperature", "unit": "K", "scale": 1.0},
}
PRECONV_LEADS = (6, 7, 8, 9)  # lead hours corresponding to ~06–09 LT


def _h3_process_day(args):
    """Accumulate pre-convection thermodynamic profiles for one day."""
    exp_dir, day_dir = args
    day_name = day_dir.name

    nh = len(H_BINS) - 1
    sums = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    counts = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    n = 0

    for lead in PRECONV_LEADS:
        step_name = None
        for f in day_dir.iterdir():
            if f.name.endswith(".nc"):
                try:
                    l = int(f.stem.split("+")[1])
                except (IndexError, ValueError):
                    continue
                if l == lead:
                    step_name = f.name
                    break

        if step_name is None:
            continue

        try:
            height = read_field(
                exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name
            )
        except Exception:
            continue

        h_km = height / 1000.0

        for var in THERMO_VARS:
            try:
                field = read_field(exp_dir / "masked-netcdf" / var / day_name / step_name)
            except Exception:
                continue

            # Level-by-level spatial mean
            for lev in range(field.shape[0]):
                h_val = np.nanmean(h_km[lev])
                if not np.isfinite(h_val):
                    continue
                h_idx = np.digitize(h_val, H_BINS) - 1
                if h_idx < 0 or h_idx >= nh:
                    continue
                val = np.nanmean(field[lev])
                if np.isfinite(val):
                    sums[var][h_idx] += val
                    counts[var][h_idx] += 1.0

        n += 1

    if n == 0:
        return None
    return {"sums": sums, "counts": counts, "n": n}


def accumulate_h3(experiment: str, max_days: int | None = None) -> dict:
    exp_dir = DATA_ROOT / experiment
    days = list_days(exp_dir, "GEOPOTENTIEL")
    if max_days is not None:
        days = days[:max_days]

    tasks = [(exp_dir, d) for d in days]
    print(f"  H3 {experiment}: processing {len(days)} days ...", flush=True)

    nh = len(H_BINS) - 1
    sums   = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    counts = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    n_files = 0

    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_h3_process_day, tasks)):
            if res is None:
                continue
            for v in THERMO_VARS:
                sums[v]   += res["sums"][v]
                counts[v] += res["counts"][v]
            n_files += res["n"]
            if (i + 1) % 100 == 0:
                print(f"  H3 {experiment}: {i+1}/{len(days)} days", flush=True)

    print(f"  H3 {experiment}: DONE — {n_files} files", flush=True)
    return {
        "sums": sums,
        "counts": counts,
        "n_files": n_files,
        "h_bins": H_BINS,
    }


def save_h3(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"h3_{experiment}.npz"
    save_dict = {
        "n_files": np.array(result["n_files"]),
        "h_bins": result["h_bins"],
    }
    for v in THERMO_VARS:
        save_dict[f"sum_{v}"] = result["sums"][v]
        save_dict[f"cnt_{v}"] = result["counts"][v]
    np.savez_compressed(path, **save_dict)
    print(f"  Saved: {path}")


def load_h3(experiment: str) -> dict | None:
    path = CACHE_DIR / f"h3_{experiment}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    variables = [k.replace("sum_", "") for k in d.files if k.startswith("sum_")]
    return {
        "sums":   {v: d[f"sum_{v}"] for v in variables},
        "counts": {v: d[f"cnt_{v}"] for v in variables},
        "n_files": int(d["n_files"]),
        "h_bins": d["h_bins"],
    }


def plot_h3(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H3 plot: pre-convection (06–09 LT) thermodynamic profiles.

    Top row: absolute profiles for each variable.
    Bottom row: anomalies relative to C1M.
    """
    variables = list(THERMO_VARS.keys())
    ncols = len(variables)
    fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 12), squeeze=False)

    # Compute profiles
    profiles = {}
    for exp in experiments:
        r = results[exp]
        h_bins = r["h_bins"]
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
        profiles[exp] = {}
        for v in variables:
            s = r["sums"][v]
            c = r["counts"][v]
            with np.errstate(invalid="ignore"):
                prof = np.where(c > 0, s / c, np.nan)
            cfg = THERMO_VARS[v]
            profiles[exp][v] = prof * cfg["scale"]

    ref_exp = experiments[0]  # control

    for col, var in enumerate(variables):
        cfg = THERMO_VARS[var]
        ax_abs = axes[0, col]
        ax_anom = axes[1, col]

        for exp in experiments:
            prof = profiles[exp][var]
            ax_abs.plot(
                prof, h_centers,
                color=EXP_COLORS[exp], linewidth=2.5, label=EXPERIMENTS[exp],
            )

            # Anomaly vs control
            if exp != ref_exp:
                ref = profiles[ref_exp][var]
                anom = prof - ref
                ax_anom.plot(
                    anom, h_centers,
                    color=EXP_COLORS[exp], linewidth=2.5, label=f"{EXPERIMENTS[exp]} − C1M",
                )

        ax_abs.set_ylabel("Height (km)", fontsize=14)
        ax_abs.set_xlabel(f"{cfg['label']} ({cfg['unit']})", fontsize=14)
        ax_abs.set_ylim(0, 18)
        ax_abs.legend(fontsize=12)
        ax_abs.grid(alpha=0.3)
        ax_abs.set_title(
            f"H3: Pre-convection {cfg['label']}", fontsize=16, fontweight="bold",
        )

        ax_anom.set_ylabel("Height (km)", fontsize=14)
        ax_anom.set_xlabel(f"$\\Delta$ {cfg['label']} ({cfg['unit']})", fontsize=14)
        ax_anom.set_ylim(0, 18)
        ax_anom.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax_anom.legend(fontsize=12)
        ax_anom.grid(alpha=0.3)
        ax_anom.set_title(
            f"H3: {cfg['label']} anomaly vs C1M", fontsize=16, fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H3 figure: {output_path}")
    plt.close(fig)


# =========================================================================
# H4 — MOISTURE FLUX DIVERGENCE PROFILE
# =========================================================================
#
# Same large-scale circulation acts on different moisture profiles.
# G1M deposits moisture at 600-800 hPa (graupel evaporation).
# C1M/G2M deposit moisture at the surface (rain).
#
# If there is mean divergence aloft and convergence in the BL (typical
# tropical convective regime), then:
#   G1M: more moisture at 600-800 hPa × divergence = net EXPORT
#   C1M: more moisture in BL × convergence = net RETENTION
#
# Diagnostics:
#   (a) Wind divergence profile:  div = ∂u/∂x + ∂v/∂y
#   (b) Moisture flux divergence: ∂(qu)/∂x + ∂(qv)/∂y
#   (c) Moisture profile (q)
#   (d) Effective moisture tendency: q × div  (the "divergence acting on
#       moisture" term — shows where divergence exports more moisture)
#
# Variables: WIND.U.PHYS, WIND.V.PHYS, HUMI.SPECIFI, GEOPOTENTIEL
# Grid: ~4 km Lambert conformal, lat/lon in NetCDF coords.
# =========================================================================

H4_H_BINS = np.linspace(0.0, 20.0, 101)


def _estimate_grid_spacing(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dx, dy) arrays in metres from the lat/lon coordinates.

    dx and dy vary with latitude; returned arrays have shape (ny, nx)
    matching the spatial grid (broadcast as needed).
    """
    with xr.open_dataset(filepath, decode_times=False) as ds:
        lat = ds["lat"].values  # (ny, nx)
        lon = ds["lon"].values

    # Central differences in degrees, converted to metres
    # dy: along y-axis (column-wise)
    dlat_dy = np.gradient(lat, axis=0)  # degrees per grid cell in y
    dy = dlat_dy * 111_000.0            # metres

    # dx: along x-axis (row-wise), scaled by cos(lat)
    dlon_dx = np.gradient(lon, axis=1)
    dx = dlon_dx * 111_000.0 * np.cos(np.radians(lat))

    return np.abs(dx), np.abs(dy)


def _h4_process_day(args):
    """
    Accumulate wind divergence and moisture flux divergence profiles.

    For each timestep, compute at every model level:
      - div   = ∂u/∂x + ∂v/∂y                (wind divergence)
      - mf_div = ∂(qu)/∂x + ∂(qv)/∂y         (moisture flux divergence)
      - q_div  = q × div                       (divergence acting on moisture)
      - q                                       (specific humidity)

    Then bin by height and accumulate domain-mean profiles.
    """
    exp_dir, day_dir, min_lead_hour, dx, dy = args
    day_name = day_dir.name
    steps = list_steps(day_dir, min_lead_hour)
    if not steps:
        return None

    nh = len(H4_H_BINS) - 1

    # Accumulators: profile sums and counts
    acc = {
        k: np.zeros(nh, dtype=np.float64)
        for k in ("div_sum", "div_cnt",
                   "mfdiv_sum", "mfdiv_cnt",
                   "qdiv_sum", "qdiv_cnt",
                   "q_sum", "q_cnt")
    }
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            u = read_field(exp_dir / "masked-netcdf" / "WIND.U.PHYS"   / day_name / step_name)
            v = read_field(exp_dir / "masked-netcdf" / "WIND.V.PHYS"   / day_name / step_name)
            q = read_field(exp_dir / "masked-netcdf" / "HUMI.SPECIFI"  / day_name / step_name)
            h = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL"  / day_name / step_name)
        except Exception:
            continue

        h_km = h / 1000.0

        # Compute divergence fields level by level
        nlev = u.shape[0]
        for lev in range(nlev):
            u_lev = u[lev]
            v_lev = v[lev]
            q_lev = q[lev]
            h_lev = h_km[lev]

            # Mean height for this level → bin index
            h_val = np.nanmean(h_lev)
            if not np.isfinite(h_val):
                continue
            h_idx = np.digitize(h_val, H4_H_BINS) - 1
            if h_idx < 0 or h_idx >= nh:
                continue

            # Finite-difference derivatives (NaN-safe via nanmean at the end)
            du_dx = np.gradient(u_lev, axis=1) / dx   # ∂u/∂x
            dv_dy = np.gradient(v_lev, axis=0) / dy   # ∂v/∂y

            div = du_dx + dv_dy  # wind divergence (s⁻¹)

            qu = q_lev * u_lev
            qv = q_lev * v_lev
            dqu_dx = np.gradient(qu, axis=1) / dx
            dqv_dy = np.gradient(qv, axis=0) / dy

            mf_div = dqu_dx + dqv_dy  # moisture flux divergence (kg/kg/s)
            q_times_div = q_lev * div  # divergence acting on moisture

            # Domain means (ignoring NaN from masked points)
            div_mean = np.nanmean(div)
            mf_mean  = np.nanmean(mf_div)
            qd_mean  = np.nanmean(q_times_div)
            q_mean   = np.nanmean(q_lev)

            if np.isfinite(div_mean):
                acc["div_sum"][h_idx] += div_mean
                acc["div_cnt"][h_idx] += 1.0
            if np.isfinite(mf_mean):
                acc["mfdiv_sum"][h_idx] += mf_mean
                acc["mfdiv_cnt"][h_idx] += 1.0
            if np.isfinite(qd_mean):
                acc["qdiv_sum"][h_idx] += qd_mean
                acc["qdiv_cnt"][h_idx] += 1.0
            if np.isfinite(q_mean):
                acc["q_sum"][h_idx] += q_mean
                acc["q_cnt"][h_idx] += 1.0

        n_files += 1

    acc["n_files"] = n_files
    return acc


def accumulate_h4(experiment: str, max_days: int | None = None) -> dict:
    exp_dir = DATA_ROOT / experiment
    days = list_days(exp_dir, "WIND.U.PHYS")
    if max_days is not None:
        days = days[:max_days]

    # Pre-compute grid spacing from one sample file
    sample_step = list_steps(days[0], MIN_LEAD_HOUR)[0]
    sample_path = exp_dir / "masked-netcdf" / "WIND.U.PHYS" / days[0].name / sample_step.name
    dx, dy = _estimate_grid_spacing(sample_path)

    tasks = [(exp_dir, d, MIN_LEAD_HOUR, dx, dy) for d in days]
    print(f"  H4 {experiment}: processing {len(days)} days ...", flush=True)

    nh = len(H4_H_BINS) - 1
    totals = {
        k: np.zeros(nh, dtype=np.float64)
        for k in ("div_sum", "div_cnt", "mfdiv_sum", "mfdiv_cnt",
                   "qdiv_sum", "qdiv_cnt", "q_sum", "q_cnt")
    }
    n_files = 0

    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_h4_process_day, tasks)):
            if res is None:
                continue
            for k in totals:
                totals[k] += res[k]
            n_files += res["n_files"]
            if (i + 1) % 100 == 0:
                print(f"  H4 {experiment}: {i+1}/{len(days)} days", flush=True)

    print(f"  H4 {experiment}: DONE — {n_files} files", flush=True)
    totals["n_files"] = n_files
    totals["h_bins"] = H4_H_BINS
    return totals


def save_h4(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"h4_{experiment}.npz"
    save_dict = {k: np.array(v) for k, v in result.items()}
    np.savez_compressed(path, **save_dict)
    print(f"  Saved: {path}")


def load_h4(experiment: str) -> dict | None:
    path = CACHE_DIR / f"h4_{experiment}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _compute_freezing_levels(experiments: list[str]) -> dict[str, float]:
    """
    Compute the mean freezing level (km) for each experiment from H3 cache.

    Interpolates the mean temperature profile to find T = 273.15 K.
    Returns {experiment: freezing_level_km}.
    """
    freezing = {}
    for exp in experiments:
        h3 = load_h3(exp)
        if h3 is None:
            continue
        h_bins = h3["h_bins"]
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
        s = h3["sums"]["TEMPERATURE"]
        c = h3["counts"]["TEMPERATURE"]
        with np.errstate(invalid="ignore"):
            t_profile = np.where(c > 0, s / c, np.nan)

        # Find where temperature crosses 273.15 K (scanning upward)
        valid = np.isfinite(t_profile)
        h_v = h_centers[valid]
        t_v = t_profile[valid]
        if len(t_v) < 2:
            continue
        for k in range(len(t_v) - 1):
            if t_v[k] >= 273.15 and t_v[k + 1] < 273.15:
                # Linear interpolation
                frac = (273.15 - t_v[k]) / (t_v[k + 1] - t_v[k])
                freezing[exp] = float(h_v[k] + frac * (h_v[k + 1] - h_v[k]))
                break
    return freezing


def _add_freezing_level(ax, freezing_levels: dict[str, float], experiments: list[str]):
    """Draw a horizontal dashed line for each experiment's freezing level."""
    for exp in experiments:
        if exp not in freezing_levels:
            continue
        fl = freezing_levels[exp]
        ax.axhline(
            fl, color=EXP_COLORS[exp], linewidth=1.5, linestyle=":",
            alpha=0.7,
        )
    # Single label annotation using the mean freezing level
    if freezing_levels:
        mean_fl = np.mean(list(freezing_levels.values()))
        ax.annotate(
            f"0 °C ≈ {mean_fl:.1f} km",
            xy=(1.0, mean_fl), xycoords=("axes fraction", "data"),
            fontsize=10, ha="right", va="bottom", color="0.3",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )


def plot_h4(results: dict[str, dict], experiments: list[str], output_path: Path):
    """
    H4 plot: moisture flux divergence profiles.

    4 panels:
      (a) Wind divergence ∂u/∂x + ∂v/∂y — should be ~same for all schemes
      (b) Moisture profile q — differs between schemes
      (c) q × div — divergence acting on moisture (where schemes differ)
      (d) Full moisture flux divergence ∂(qu)/∂x + ∂(qv)/∂y

    Positive = divergence (moisture export), Negative = convergence (moisture import).
    """
    freezing_levels = _compute_freezing_levels(experiments)

    fig, axes = plt.subplots(1, 4, figsize=(28, 8), squeeze=False)

    panel_configs = [
        ("div",   r"Wind divergence (s$^{-1}$)",
         "Wind divergence profile\n(same dynamics?)"),
        ("q",     "Specific humidity (g/kg)",
         "Moisture profile\n(where is the moisture?)"),
        ("qdiv",  r"q $\times$ div (kg kg$^{-1}$ s$^{-1}$)",
         "Divergence $\\times$ moisture\n(where is moisture exported?)"),
        ("mfdiv", r"$\nabla \cdot (q\mathbf{v})$ (kg kg$^{-1}$ s$^{-1}$)",
         "Full moisture flux divergence"),
    ]

    for col, (var, xlabel, title) in enumerate(panel_configs):
        ax = axes[0, col]

        for exp in experiments:
            r = results[exp]
            h_bins = r["h_bins"]
            h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

            s = r[f"{var}_sum"]
            c = r[f"{var}_cnt"]
            with np.errstate(invalid="ignore"):
                profile = np.where(c > 0, s / c, np.nan)

            # Scale q to g/kg for readability
            if var == "q":
                profile = profile * 1000.0

            ax.plot(
                profile, h_centers,
                color=EXP_COLORS[exp], linewidth=2.5, label=EXPERIMENTS[exp],
            )

        ax.set_ylabel("Height (km)", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, 18)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

        # Vertical zero line for divergence panels
        if var != "q":
            ax.axvline(0, color="k", linewidth=0.8, linestyle="--")

        # Freezing level
        _add_freezing_level(ax, freezing_levels, experiments)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    print(f"Saved H4 figure: {output_path}")
    plt.close(fig)

    # Also plot anomalies relative to control
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), squeeze=False)
    ref_exp = experiments[0]

    for col, (var, xlabel, title) in enumerate([
        ("q",     "$\\Delta$ Specific humidity (g/kg)",
         "Moisture anomaly vs C1M"),
        ("qdiv",  r"$\Delta$ q $\times$ div (kg kg$^{-1}$ s$^{-1}$)",
         "Divergence $\\times$ moisture\nanomaly vs C1M"),
        ("mfdiv", r"$\Delta$ $\nabla \cdot (q\mathbf{v})$ (kg kg$^{-1}$ s$^{-1}$)",
         "Moisture flux divergence\nanomaly vs C1M"),
    ]):
        ax = axes2[0, col]
        r_ref = results[ref_exp]
        h_bins = r_ref["h_bins"]
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])

        s_ref = r_ref[f"{var}_sum"]
        c_ref = r_ref[f"{var}_cnt"]
        with np.errstate(invalid="ignore"):
            prof_ref = np.where(c_ref > 0, s_ref / c_ref, np.nan)
        if var == "q":
            prof_ref = prof_ref * 1000.0

        for exp in experiments:
            if exp == ref_exp:
                continue
            r = results[exp]
            s = r[f"{var}_sum"]
            c = r[f"{var}_cnt"]
            with np.errstate(invalid="ignore"):
                prof = np.where(c > 0, s / c, np.nan)
            if var == "q":
                prof = prof * 1000.0

            anom = prof - prof_ref
            ax.plot(
                anom, h_centers,
                color=EXP_COLORS[exp], linewidth=2.5,
                label=f"{EXPERIMENTS[exp]} − C1M",
            )

        ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
        _add_freezing_level(ax, freezing_levels, experiments)
        ax.set_ylabel("Height (km)", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, 18)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    anom_path = output_path.with_name(output_path.stem + "_anomaly.png")
    fig2.tight_layout()
    fig2.savefig(anom_path, dpi=450, bbox_inches="tight")
    print(f"Saved H4 anomaly figure: {anom_path}")
    plt.close(fig2)


# =========================================================================
# Main
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convective feedback diagnostics")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument("--hypotheses", nargs="+", default=["h1", "h2", "h3", "h4"],
                        choices=["h1", "h2", "h3", "h4"])
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if "h1" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H1: Condensate loading")
        print("=" * 60)
        h1_results = {}
        for exp in args.experiments:
            cached = None if args.no_cache else load_h1(exp)
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h1_results[exp] = cached
            else:
                r = accumulate_h1(exp, max_days=args.max_days)
                save_h1(exp, r)
                h1_results[exp] = r
        plot_h1(h1_results, args.experiments, OUTPUT_DIR / "convective_feedback_h1_loading.png")

    if "h2" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H2: Downdraft feedback")
        print("=" * 60)
        h2_results = {}
        for exp in args.experiments:
            cached = None if args.no_cache else load_h2(exp)
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h2_results[exp] = cached
            else:
                r = accumulate_h2(exp, max_days=args.max_days)
                save_h2(exp, r)
                h2_results[exp] = r
        plot_h2(h2_results, args.experiments, OUTPUT_DIR / "convective_feedback_h2_downdrafts.png")

    if "h3" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H3: Pre-convection moisture")
        print("=" * 60)
        h3_results = {}
        for exp in args.experiments:
            cached = None if args.no_cache else load_h3(exp)
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h3_results[exp] = cached
            else:
                r = accumulate_h3(exp, max_days=args.max_days)
                save_h3(exp, r)
                h3_results[exp] = r
        plot_h3(h3_results, args.experiments, OUTPUT_DIR / "convective_feedback_h3_moisture.png")

    if "h4" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H4: Moisture flux divergence")
        print("=" * 60)
        h4_results = {}
        for exp in args.experiments:
            cached = None if args.no_cache else load_h4(exp)
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h4_results[exp] = cached
            else:
                r = accumulate_h4(exp, max_days=args.max_days)
                save_h4(exp, r)
                h4_results[exp] = r
        plot_h4(h4_results, args.experiments, OUTPUT_DIR / "convective_feedback_h4_moist_divergence.png")


if __name__ == "__main__":
    main()
