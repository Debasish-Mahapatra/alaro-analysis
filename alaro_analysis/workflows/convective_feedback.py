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
N_WORKERS = 8


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
# Main
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convective feedback diagnostics")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument("--hypotheses", nargs="+", default=["h1", "h2", "h3"],
                        choices=["h1", "h2", "h3"])
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


if __name__ == "__main__":
    main()
