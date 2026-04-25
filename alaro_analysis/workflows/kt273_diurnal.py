"""
Per-diagnostic diurnal line plots for 0 C isotherm (KT273*) diagnostics
plus boundary-layer theta_e gradient.

One figure per diagnostic (not stacked), three lines per figure
(C1M / G1M / G2M), x = hour of day (UTC), y = domain-mean value.
Supports full 2-year and per-season breakdowns.

Diagnostics
-----------
Raw KT273 scalars (domain nanmean per step):
    KT273GRAUPEL, KT273RAIN, KT273SNOW,
    KT273LIQUID_WATE, KT273SOLID_WATER,
    KT273DD_OMEGA, KT273DD_MESH_FRA,
    KT273UD_OMEGA, KT273UD_MESH_FRA,
    KT273HUMI.SPECIF

Derived from KT273:
    DD_MASS_FLUX = -(KT273DD_OMEGA * KT273DD_MESH_FRA) / g   [kg m-2 s-1]
    UD_MASS_FLUX = -(KT273UD_OMEGA * KT273UD_MESH_FRA) / g   [kg m-2 s-1]

Derived from full 3-D state (T, q, p, gz):
    BL_THETAE_GRAD = (theta_e at 1500 m AGL - theta_e at lowest level) / 1500 m
                     [K / km]   (positive => stable above BL)

Usage
-----
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.kt273_diurnal [--max-days N] [--force]
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from alaro_analysis.analysis.derived import compute_theta_e_field
from alaro_analysis.common.constants import EXPERIMENT_COLORS as EXP_COLORS, G, SEASONS

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/kt273_diurnal")
CACHE_DIR = Path(
    "/gpfs/me01/me/CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/kt273_diurnal"
)

EXPERIMENTS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
# EXP_COLORS imported from common.constants.EXPERIMENT_COLORS above.

MIN_LEAD_HOUR = 0
N_WORKERS = 34
BL_TOP_M = 1500.0   # reference AGL height for theta_e gradient
UTC_OFFSET_HOURS = -4   # Amazon local time is UTC-4


# Variable short key -> (cached dir name, y-axis label, unit)
KT273_SCALARS: dict[str, tuple[str, str, str]] = {
    "graupel":     ("KT273GRAUPEL",     "Graupel mixing ratio at 0 C",    "kg/kg"),
    "rain":        ("KT273RAIN",        "Rain mixing ratio at 0 C",       "kg/kg"),
    "snow":        ("KT273SNOW",        "Snow mixing ratio at 0 C",       "kg/kg"),
    "liquid":      ("KT273LIQUID_WATE", "Cloud liquid water at 0 C",      "kg/kg"),
    "solid":       ("KT273SOLID_WATER", "Cloud ice at 0 C",               "kg/kg"),
    "dd_omega":    ("KT273DD_OMEGA",    "Downdraft omega at 0 C",         "Pa/s"),
    "dd_mesh":     ("KT273DD_MESH_FRA", "Downdraft mesh fraction at 0 C", "-"),
    "ud_omega":    ("KT273UD_OMEGA",    "Updraft omega at 0 C",           "Pa/s"),
    "ud_mesh":     ("KT273UD_MESH_FRA", "Updraft mesh fraction at 0 C",   "-"),
    "humi":        ("KT273HUMI.SPECIF", "Specific humidity at 0 C",       "kg/kg"),
}

DERIVED_LABELS: dict[str, tuple[str, str]] = {
    "dd_mass_flux":  ("Downdraft mass flux at 0 C",
                      r"kg m$^{-2}$ s$^{-1}$"),
    "ud_mass_flux":  ("Updraft mass flux at 0 C",
                      r"kg m$^{-2}$ s$^{-1}$"),
    "bl_thetae_grad": (
        r"BL $\theta_e$ gradient  ($\theta_e$(1.5 km AGL) - $\theta_e$(surf)) / 1.5 km",
        "K/km",
    ),
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def list_days(exp_dir: Path, variable: str) -> list[Path]:
    root = exp_dir / "masked-netcdf" / variable
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("pf"))


def _day_month(day_dir: Path) -> int | None:
    """Extract month number from a pfYYYYMMDD directory name."""
    name = day_dir.name
    if not name.startswith("pf") or len(name) != 10:
        return None
    try:
        return int(name[6:8])
    except ValueError:
        return None


def list_steps(day_dir: Path) -> list[tuple[int, Path]]:
    steps: list[tuple[int, Path]] = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= MIN_LEAD_HOUR:
            steps.append((lead, f))
    return steps


def _read_arr(filepath: Path) -> np.ndarray:
    """Read a single-variable NetCDF; return full-shape array (time, level, y, x)
    or (time, y, x) squeezed as appropriate.  Returns np.nan array if missing."""
    with xr.open_dataset(filepath, decode_times=False) as ds:
        var_name = list(ds.data_vars)[0]
        return ds[var_name].values


# ---------------------------------------------------------------------------
# Per-day accumulator
# ---------------------------------------------------------------------------

def _process_day(args):
    exp_dir, day_dir = args
    day_name = day_dir.name
    steps = list_steps(day_dir)
    if not steps:
        return None

    # Accumulators: per-diagnostic (24,) sums + counts
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for key in list(KT273_SCALARS) + list(DERIVED_LABELS):
        sums[key] = np.zeros(24, dtype=np.float64)
        counts[key] = np.zeros(24, dtype=np.int64)

    for lead, step_file in steps:
        step_name = step_file.name
        hour = (lead + UTC_OFFSET_HOURS) % 24   # local hour (Amazon UTC-4)

        # ---- KT273 scalar diagnostics: domain-mean of each cached field
        kt_vals: dict[str, float] = {}
        for short_key, (dirname, _label, _unit) in KT273_SCALARS.items():
            fp = exp_dir / "masked-netcdf" / dirname / day_name / step_name
            if not fp.exists():
                kt_vals[short_key] = np.nan
                continue
            try:
                arr = _read_arr(fp)
                vals = arr[np.isfinite(arr)]
                kt_vals[short_key] = float(np.mean(vals)) if vals.size else np.nan
            except Exception:
                kt_vals[short_key] = np.nan

        # ---- Derived mass fluxes (requires both omega and mesh)
        def _mass_flux(omega_key: str, mesh_key: str) -> float:
            om = kt_vals.get(omega_key, np.nan)
            me = kt_vals.get(mesh_key, np.nan)
            if not (np.isfinite(om) and np.isfinite(me)):
                return np.nan
            return -(om * me) / G

        dd_flux = _mass_flux("dd_omega", "dd_mesh")
        ud_flux = _mass_flux("ud_omega", "ud_mesh")

        # ---- BL theta_e gradient: needs TEMPERATURE, HUMI.SPECIFI, PRESSURE, GEOPOTENTIEL
        bl_grad = np.nan
        try:
            t   = _read_arr(exp_dir / "masked-netcdf" / "TEMPERATURE"  / day_name / step_name)
            q   = _read_arr(exp_dir / "masked-netcdf" / "HUMI.SPECIFI" / day_name / step_name)
            p   = _read_arr(exp_dir / "masked-netcdf" / "PRESSURE"     / day_name / step_name)
            gz  = _read_arr(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
        except Exception:
            t = q = p = gz = None

        if t is not None and q is not None and p is not None and gz is not None:
            # Drop leading time dim if present → (L, y, x)
            if t.ndim == 4:
                t = t[0]; q = q[0]; p = p[0]; gz = gz[0]

            # Guard against all-NaN files
            if np.isfinite(t).any() and np.isfinite(q).any() and np.isfinite(p).any():
                p_pa = p if np.nanmax(p) >= 2000.0 else p * 100.0
                theta_e_lyx = compute_theta_e_field(t, q, p_pa)   # (L, y, x)

                theta_e_prof = np.nanmean(theta_e_lyx, axis=(1, 2))  # (L,)
                z_prof       = np.nanmean(gz,         axis=(1, 2))   # (L,) m AGL ~ geopotential m

                # Identify surface (lowest z) and 1500 m level via linear interp
                valid = np.isfinite(theta_e_prof) & np.isfinite(z_prof)
                if valid.sum() >= 2:
                    z_v = z_prof[valid]
                    th_v = theta_e_prof[valid]
                    order = np.argsort(z_v)
                    z_s = z_v[order]
                    th_s = th_v[order]
                    z_surf = float(z_s[0])
                    th_surf = float(th_s[0])
                    z_top = z_surf + BL_TOP_M
                    if z_s[-1] >= z_top:
                        th_top = float(np.interp(z_top, z_s, th_s))
                        bl_grad = (th_top - th_surf) / (BL_TOP_M / 1000.0)  # K/km

        # ---- Accumulate into hour bins
        for key, val in kt_vals.items():
            if np.isfinite(val):
                sums[key][hour]   += val
                counts[key][hour] += 1
        for key, val in (("dd_mass_flux", dd_flux),
                          ("ud_mass_flux", ud_flux),
                          ("bl_thetae_grad", bl_grad)):
            if np.isfinite(val):
                sums[key][hour]   += val
                counts[key][hour] += 1

    return {"sums": sums, "counts": counts}


# ---------------------------------------------------------------------------
# Compute / cache / load (per experiment, per period)
# ---------------------------------------------------------------------------

def accumulate(
    experiment: str,
    period_key: str,
    allowed_months: tuple[int, ...] | None,
    max_days: int | None,
) -> dict:
    exp_dir = DATA_ROOT / experiment
    # Use any always-present var to enumerate days.
    all_days = list_days(exp_dir, "PRESSURE")
    if allowed_months is not None:
        days = [d for d in all_days if _day_month(d) in allowed_months]
    else:
        days = list(all_days)
    if max_days is not None:
        days = days[:max_days]
    if not days:
        raise RuntimeError(f"No days for {experiment} in {period_key}")

    print(f"  kt273 {experiment}/{period_key}: {len(days)} days", flush=True)
    tasks = [(exp_dir, d) for d in days]

    all_keys = list(KT273_SCALARS) + list(DERIVED_LABELS)
    agg_sums   = {k: np.zeros(24, dtype=np.float64) for k in all_keys}
    agg_counts = {k: np.zeros(24, dtype=np.int64)   for k in all_keys}
    n_used = 0

    with Pool(N_WORKERS) as pool:
        for idx, res in enumerate(pool.imap_unordered(_process_day, tasks)):
            if res is None:
                continue
            for k in all_keys:
                agg_sums[k]   += res["sums"][k]
                agg_counts[k] += res["counts"][k]
            n_used += 1
            if (idx + 1) % 100 == 0:
                print(f"  kt273 {experiment}/{period_key}: {idx+1}/{len(days)}", flush=True)

    print(f"  kt273 {experiment}/{period_key}: DONE {n_used} days", flush=True)
    return {
        "sums":   agg_sums,
        "counts": agg_counts,
        "n_days": n_used,
    }


def _finalize(sums: dict[str, np.ndarray], counts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, s in sums.items():
        c = counts[k]
        mean = np.full(s.shape, np.nan, dtype=np.float64)
        nz = c > 0
        mean[nz] = s[nz] / c[nz]
        out[k] = mean
    return out


def save_cache(experiment: str, period_key: str, result: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"kt273_{experiment}_{period_key}.npz"
    flat: dict[str, np.ndarray] = {}
    for k, v in result["sums"].items():
        flat[f"sum__{k}"] = v
    for k, v in result["counts"].items():
        flat[f"cnt__{k}"] = v
    flat["n_days"] = np.array(result["n_days"])
    np.savez_compressed(path, **flat)
    print(f"  saved {path}")
    return path


def load_cache(experiment: str, period_key: str) -> dict | None:
    path = CACHE_DIR / f"kt273_{experiment}_{period_key}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    sums:   dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for k in d.files:
        if k.startswith("sum__"):
            sums[k[len("sum__"):]] = d[k]
        elif k.startswith("cnt__"):
            counts[k[len("cnt__"):]] = d[k]
    return {"sums": sums, "counts": counts, "n_days": int(d["n_days"])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _figure_filename(diag_key: str, period_key: str) -> str:
    base = diag_key  # keys are already lowercase and clean
    if period_key == "full_2yr":
        return f"{base}_diurnal.png"
    return f"{base}_diurnal_{period_key}.png"


def _plot_one(diag_key: str, label: str, unit: str,
              finals: dict[str, dict[str, np.ndarray]],
              period_key: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(24)
    for exp, exp_label in EXPERIMENTS.items():
        y = finals.get(exp, {}).get(diag_key)
        if y is None:
            continue
        ax.plot(hours, y, color=EXP_COLORS[exp], lw=2.0, marker="o", ms=3,
                label=exp_label)
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlabel("Hour (Amazon UTC-4)")
    ylabel = f"{label}  [{unit}]" if unit and unit != "-" else label
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_path = OUTPUT_DIR / _figure_filename(diag_key, period_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {out_path}")


def plot_all(finals_by_period: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    for period_key, finals in finals_by_period.items():
        for short_key, (_dir, label, unit) in KT273_SCALARS.items():
            _plot_one(short_key, label, unit, finals, period_key)
        for short_key, (label, unit) in DERIVED_LABELS.items():
            _plot_one(short_key, label, unit, finals, period_key)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_periods(modes: list[str], seasons: list[str]) -> list[tuple[str, tuple[int, ...] | None]]:
    specs: list[tuple[str, tuple[int, ...] | None]] = []
    if "full" in modes:
        specs.append(("full_2yr", None))
    if "seasonal" in modes:
        for s in seasons:
            if s not in SEASONS:
                raise ValueError(f"Unknown season {s!r}; valid: {list(SEASONS)}")
            specs.append((s, tuple(SEASONS[s]["months"])))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists.")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                        choices=list(EXPERIMENTS))
    parser.add_argument("--analysis-modes", nargs="+",
                        default=("full",),
                        choices=("full", "seasonal"))
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS.keys()))
    args = parser.parse_args()

    periods = _build_periods(list(args.analysis_modes), list(args.seasons))

    finals_by_period: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for period_key, months in periods:
        finals: dict[str, dict[str, np.ndarray]] = {}
        for exp in args.experiments:
            cached = None if args.force else load_cache(exp, period_key)
            if cached is None:
                cached = accumulate(exp, period_key, months, args.max_days)
                save_cache(exp, period_key, cached)
            else:
                print(f"  kt273 {exp}/{period_key}: loaded cache ({cached['n_days']} days)")
            finals[exp] = _finalize(cached["sums"], cached["counts"])
        finals_by_period[period_key] = finals

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_all(finals_by_period)


if __name__ == "__main__":
    main()
