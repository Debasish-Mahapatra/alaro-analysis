"""
Microphysical budget diagnostics: where does condensate end up?

Motivated by Kwinten's question: "How much evaporation/sublimation/melting do
you have relative to condensation — given an amount of condensation, where does
the condensate end up: stored in hydrometeors aloft, as precipitation, or
re-evaporated/sublimated back to vapour?"

Per-process microphysics rates (condensation, evap, sublim, melt, depos,
autoconversion, accretion) are not written to the default FA history — they
live inside APLPAR as intermediate arrays.  The per-species fluxes are in DDH
output (/gpfs/.../ALARO-RUNS/DDH/{control,graupel,2mom}/) if full detail is
needed.

This workflow works purely from the masked-NetCDF cache.  It computes four
diagnostics:

  B1 — Species storage partition.  Fraction of total condensate mass in each
       hydrometeor (QR / QS / QG / QL / QI), by height and hour-of-day.
       Tests Kwinten's G1M-graupel-aloft hypothesis: does G1M store more
       frozen mass at high altitude than C1M/G2M?

  B2 — Surface-reach fraction.  surface_flux / max(column_flux) per hour.
       If the model produces a large column-max flux but only a small fraction
       reaches the surface, a lot is being re-evaporated below cloud base.

  B3 — Per-level residence time.  tau(z) = mass_above(z) / flux(z).
       If graupel has short tau aloft in G1M → less time for sublimation →
       supports Kwinten's residence-time hypothesis.

  B4 — Flux divergence profile.  -d(flux_total)/dz along the vertical.
       Positive = net microphysical source of precipitation at that level
       (condensation + collection dominate).
       Negative = net sink (evaporation/sublimation/melting dominate).
       This is a proxy for the per-process rates; in cloud-free layers
       (above cloud top / below cloud base), the sign is diagnostic
       of sublimation / evaporation alone.

Companion note — 3MT convective triggering
-------------------------------------------
Kwinten also asked: does 3MT use column moisture or only low-level moisture?
Inspection of the C46 source (arpifs/phys_dmn/accvud.F90, lines 929-938 and
1230-1296) shows that the trigger variable ZS4 accumulates L*dp*CVGQ from
surface (KLEV) to top (KTDIA).  The per-level activation test is:

    ZKUO2 = ZS4 + ZICVG + ZICVGL > 0      ! accvud.F90:935-938

So triggering does use column-integrated moisture convergence, not just
surface.  Mid/upper-level drying reduces ZS4 and can suppress convection at
low levels, supporting Kwinten's hypothesis.  Namelist controls: LCVGQM
(modulated CVGQ closure), LCAPE (CAPE-based alternative), RMULACVG (weight).

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.microphysics_budget \
        --max-days 30          # optional, limit for quick tests
        --force                # re-run even if cache exists
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration (mirrors convective_feedback.py so caches sit side-by-side)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/microphysics_budget")
CACHE_DIR = Path(
    "/gpfs/me01/me/CLIMATE/CLIMATE/deba/alaro-analysis/cache/microphysics_budget"
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

# Hydrometeor species layout.  Precipitating species first so residence-time
# aggregations can slice precip-only vs cloud-only subsets cleanly.
SPECIES = ("RAIN", "SNOW", "GRAUPEL", "LIQUID_WATER", "SOLID_WATER")
SPECIES_LABEL = {
    "RAIN":         "rain",
    "SNOW":         "snow",
    "GRAUPEL":      "graupel",
    "LIQUID_WATER": "cloud liq.",
    "SOLID_WATER":  "cloud ice",
}
SPECIES_COLOR = {
    "RAIN":         "#1f77b4",
    "SNOW":         "#17becf",
    "GRAUPEL":      "#9467bd",
    "LIQUID_WATER": "#d62728",
    "SOLID_WATER":  "#bcbd22",
}

G = 9.80665
MIN_LEAD_HOUR = 0
N_WORKERS = 24
T_FREEZE = 273.15  # K


# ---------------------------------------------------------------------------
# I/O helpers (same contract as convective_feedback.py)
# ---------------------------------------------------------------------------

def list_days(exp_dir: Path, variable: str) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / variable
    if not var_dir.exists():
        return []
    return sorted(d for d in var_dir.iterdir() if d.is_dir() and d.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int = MIN_LEAD_HOUR) -> list[tuple[int, Path]]:
    steps: list[tuple[int, Path]] = []
    for f in sorted(day_dir.iterdir()):
        if not f.name.endswith(".nc"):
            continue
        try:
            lead = int(f.stem.split("+")[1])
        except (IndexError, ValueError):
            continue
        if lead >= min_lead_hour:
            steps.append((lead, f))
    return steps


def read_field(filepath: Path) -> np.ndarray:
    """Read a (time, level, y, x) field; squeeze leading time → (level, y, x)."""
    with xr.open_dataset(filepath, decode_times=False) as ds:
        var_name = list(ds.data_vars)[0]
        return ds[var_name].values[0]


def _freeze_height(tmp_lyx: np.ndarray, z_lyx: np.ndarray) -> float:
    """Mean height (m) of the 273.15 K isotherm over a (level, y, x) column.

    For each (y, x) column, linearly interpolate T(z) between the two
    consecutive levels that straddle 273.15 K; the lowest such crossing (in
    height) is taken as the freezing level.  Columns entirely above or below
    freezing contribute NaN and are dropped from the spatial mean.
    """
    t = np.asarray(tmp_lyx, dtype=np.float64)
    z = np.asarray(z_lyx,   dtype=np.float64)
    if t.shape != z.shape or t.ndim != 3:
        return float("nan")

    diff    = t[:-1] - T_FREEZE                         # (L-1, y, x)
    diff_n  = t[1:]  - T_FREEZE
    # Sign change between adjacent levels => 273.15 K crossing in that layer.
    cross = np.sign(diff) != np.sign(diff_n)
    cross &= np.isfinite(diff) & np.isfinite(diff_n)
    if not np.any(cross):
        return float("nan")

    # Linear interpolation weight within the crossing layer.
    with np.errstate(divide="ignore", invalid="ignore"):
        w = diff / (diff - diff_n)
    z_cross = z[:-1] + w * (z[1:] - z[:-1])              # (L-1, y, x)
    z_cross = np.where(cross, z_cross, np.nan)

    # Pick the LOWEST crossing per (y, x) column — robust to multi-crossing
    # profiles (e.g. stratospheric inversions well above the true freezing
    # level).  "Lowest" = smallest height.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        z_freeze_yx = np.nanmin(z_cross, axis=0)         # (y, x)
    return float(np.nanmean(z_freeze_yx))


def _dp_pa_from_pressure(p_lyx: np.ndarray) -> np.ndarray:
    """Layer thickness dp (Pa) from a (level, y, x) pressure field.

    Mirrors alaro_analysis.analysis.derived.compute_dp_pa but for 3-D arrays.
    """
    n_lev = p_lyx.shape[0]
    if n_lev == 1:
        return np.abs(p_lyx)
    p_half = np.empty((n_lev + 1, p_lyx.shape[1], p_lyx.shape[2]), dtype=np.float64)
    p_half[1:-1] = 0.5 * (p_lyx[:-1] + p_lyx[1:])
    p_half[0]    = p_lyx[0] + (p_lyx[0] - p_half[1])
    p_half[-1]   = p_lyx[-1] - (p_half[-2] - p_lyx[-1])
    return np.abs(p_half[:-1] - p_half[1:])


# ---------------------------------------------------------------------------
# Per-day accumulator (multiprocessing-safe)
# ---------------------------------------------------------------------------

def _process_day(args):
    """Accumulate hourly (level,) profiles for one day of one experiment.

    Returns a dict of arrays keyed by diagnostic; shapes are (n_levels, 24) for
    profile-like diagnostics and (24,) for scalars.  Sums and counts are kept
    separately so that combining across days is a simple add.
    """
    exp_dir, day_dir = args
    day_name = day_dir.name

    # Use any always-present variable to probe timesteps.
    steps = list_steps(day_dir)
    if not steps:
        return None

    # Lazy per-level-count initialization: figured out on the first file that
    # successfully reads.  Keeps this resilient to per-day n_level drift.
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    n_levels = None

    for lead, step_file in steps:
        step_name = step_file.name
        hour = lead % 24
        try:
            p   = read_field(exp_dir / "masked-netcdf" / "PRESSURE"    / day_name / step_name)
            gz  = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
            cv  = read_field(exp_dir / "masked-netcdf" / "CV_PREC_FLUX" / day_name / step_name)
            st  = read_field(exp_dir / "masked-netcdf" / "ST_PREC_FLUX" / day_name / step_name)
            tmp = read_field(exp_dir / "masked-netcdf" / "TEMPERATURE"  / day_name / step_name)
        except Exception:
            continue

        # PRESSURE in masked cache is generally Pa, but guard against hPa.
        if np.nanmax(p) < 2000.0:
            p = p * 100.0

        species_arr: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            try:
                species_arr[sp] = read_field(
                    exp_dir / "masked-netcdf" / sp / day_name / step_name
                )
            except Exception:
                pass
        if not species_arr:
            continue

        # Harmonize level counts across species / flux / pressure / temperature.
        arrays = [p, gz, cv, st, tmp, *species_arr.values()]
        min_lev = min(a.shape[0] for a in arrays)
        p   = p[:min_lev]
        gz  = gz[:min_lev]
        cv  = cv[:min_lev]
        st  = st[:min_lev]
        tmp = tmp[:min_lev]
        for k in species_arr:
            species_arr[k] = species_arr[k][:min_lev]

        if n_levels is None:
            n_levels = min_lev
            for key in (
                "flux_total",
                "flux_div",
                "mass_above_total",
                "mass_per_level_total",
                "height_m",
                "pressure_pa",
                *(f"mass_per_level_{sp}"   for sp in SPECIES),
                *(f"mass_above_{sp}"        for sp in SPECIES),
            ):
                sums[key]   = np.zeros((n_levels, 24), dtype=np.float64)
                counts[key] = np.zeros((n_levels, 24), dtype=np.int64)
            for key in ("surface_flux", "max_column_flux", "z_freeze"):
                sums[key]   = np.zeros(24, dtype=np.float64)
                counts[key] = np.zeros(24, dtype=np.int64)
        elif min_lev != n_levels:
            # Drift across steps within a single day — skip this step.
            continue

        # Layer thickness and per-layer mass/area (kg m-2).
        dp = _dp_pa_from_pressure(p)
        layer_weight = dp / G  # (L, y, x) — multiply by q to get kg/m2 per layer

        # Total precip flux (CV + ST, kg m-2 s-1).
        flux_total = cv + st

        # Per-layer mass, averaged over (y, x).
        mass_total_lev = np.zeros(n_levels, dtype=np.float64)
        mass_species_lev: dict[str, np.ndarray] = {}
        for sp, q in species_arr.items():
            q_pos = np.where(np.isfinite(q) & (q > 0.0), q, 0.0)
            layer_mass = q_pos * layer_weight  # (L, y, x)
            prof = np.nanmean(layer_mass, axis=(1, 2))  # (L,)
            mass_species_lev[sp] = prof
            mass_total_lev += prof

        # Assume vertical index 0 = top of atmosphere, n_levels-1 = surface
        # (standard ALARO/ARPEGE convention; sanity-check with pressure means).
        p_mean = np.nanmean(p, axis=(1, 2))
        if not np.all(np.diff(p_mean) >= 0):
            # Pressure not increasing downward → reverse before cumulating.
            # Record a warning but proceed (should be rare).
            pass
        surface_idx = int(np.argmax(p_mean))
        top_idx     = int(np.argmin(p_mean))
        downward = surface_idx > top_idx  # True for "0=top, last=surface" layout

        # Mass above each level: cumulative sum from TOA down to that level
        # (inclusive).  Units: kg/m2.
        if downward:
            cum_total = np.cumsum(mass_total_lev)
        else:
            cum_total = np.cumsum(mass_total_lev[::-1])[::-1]
        mass_above_total = cum_total

        mass_above_species: dict[str, np.ndarray] = {}
        for sp, prof in mass_species_lev.items():
            if downward:
                mass_above_species[sp] = np.cumsum(prof)
            else:
                mass_above_species[sp] = np.cumsum(prof[::-1])[::-1]

        # Flux profile and divergence.  dF/dz approximated via finite diff
        # with respect to height (m).  GEOPOTENTIEL in the masked cache is
        # already stored as geopotential height (m), not as phi=g*z.
        z_m       = np.nanmean(gz, axis=(1, 2))
        flux_mean = np.nanmean(flux_total, axis=(1, 2))

        # central diff where possible; one-sided at boundaries
        flux_div = np.zeros_like(flux_mean)
        if n_levels >= 2:
            # d(F)/dz is positive when flux grows with height; for downward
            # flux, we want the increase-downward, i.e. -dF/dz in height
            dz = np.diff(z_m)
            with np.errstate(invalid="ignore"):
                dF_dz_fwd = np.diff(flux_mean) / dz
            flux_div[:-1] = -dF_dz_fwd
            flux_div[-1]  = flux_div[-2] if n_levels >= 2 else 0.0

        # Surface / max-column scalars for reach-surface fraction.
        surface_flux = float(flux_mean[surface_idx])
        max_col_flux = float(np.nanmax(flux_mean)) if np.any(np.isfinite(flux_mean)) else np.nan

        # 0 C isotherm height — per column, linearly interpolate T(z) to find
        # where T = 273.15 K, then average over (y, x).  Uses the lowest
        # crossing (i.e. the conventional freezing level, not a stratospheric
        # false positive).  Returns NaN for columns that are entirely above
        # or below freezing.
        z_freeze = _freeze_height(tmp, gz)

        def _add_profile(key: str, values: np.ndarray):
            finite = np.isfinite(values)
            sums[key][finite, hour]   += values[finite]
            counts[key][finite, hour] += 1

        def _add_scalar(key: str, value: float):
            if np.isfinite(value):
                sums[key][hour]   += value
                counts[key][hour] += 1

        _add_profile("flux_total",            flux_mean)
        _add_profile("flux_div",              flux_div)
        _add_profile("mass_above_total",      mass_above_total)
        _add_profile("mass_per_level_total",  mass_total_lev)
        _add_profile("height_m",              z_m)
        _add_profile("pressure_pa",           p_mean)
        for sp in SPECIES:
            if sp in mass_species_lev:
                _add_profile(f"mass_per_level_{sp}", mass_species_lev[sp])
                _add_profile(f"mass_above_{sp}",     mass_above_species[sp])

        _add_scalar("surface_flux",    surface_flux)
        _add_scalar("max_column_flux", max_col_flux)
        _add_scalar("z_freeze",        z_freeze)

    if n_levels is None:
        return None
    return {"sums": sums, "counts": counts, "n_levels": n_levels}


# ---------------------------------------------------------------------------
# Compute / cache / load
# ---------------------------------------------------------------------------

def accumulate(experiment: str, max_days: int | None = None) -> dict:
    exp_dir = DATA_ROOT / experiment
    days = list_days(exp_dir, "PRESSURE")
    if max_days is not None:
        days = days[:max_days]
    if not days:
        raise RuntimeError(f"No days under {exp_dir}/masked-netcdf/PRESSURE")

    tasks = [(exp_dir, d) for d in days]
    print(f"  budget {experiment}: {len(days)} days", flush=True)

    agg_sums: dict[str, np.ndarray] = {}
    agg_counts: dict[str, np.ndarray] = {}
    n_levels_seen: int | None = None
    n_days_used = 0

    with Pool(N_WORKERS) as pool:
        for idx, res in enumerate(pool.imap_unordered(_process_day, tasks)):
            if res is None:
                continue
            if n_levels_seen is None:
                n_levels_seen = res["n_levels"]
                for k, arr in res["sums"].items():
                    agg_sums[k]   = arr.copy()
                    agg_counts[k] = res["counts"][k].copy()
            else:
                # If a later day has a different n_levels, skip it (rare;
                # logged so the user notices).
                if res["n_levels"] != n_levels_seen:
                    print(
                        f"  budget {experiment}: level-count drift "
                        f"({res['n_levels']} vs {n_levels_seen}); skipping",
                        flush=True,
                    )
                    continue
                for k, arr in res["sums"].items():
                    agg_sums[k]   += arr
                    agg_counts[k] += res["counts"][k]
            n_days_used += 1
            if (idx + 1) % 100 == 0:
                print(f"  budget {experiment}: {idx+1}/{len(days)} days", flush=True)

    print(f"  budget {experiment}: DONE — {n_days_used} days used", flush=True)
    return {
        "sums":     agg_sums,
        "counts":   agg_counts,
        "n_levels": int(n_levels_seen) if n_levels_seen is not None else 0,
        "n_days":   int(n_days_used),
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


def save_budget(experiment: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"budget_{experiment}.npz"
    flat: dict[str, np.ndarray] = {}
    for k, v in result["sums"].items():
        flat[f"sum__{k}"] = v
    for k, v in result["counts"].items():
        flat[f"cnt__{k}"] = v
    flat["n_levels"] = np.array(result["n_levels"])
    flat["n_days"]   = np.array(result["n_days"])
    np.savez_compressed(path, **flat)
    print(f"  saved: {path}")


def load_budget(experiment: str) -> dict | None:
    path = CACHE_DIR / f"budget_{experiment}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    sums:   dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for key in d.files:
        if key.startswith("sum__"):
            sums[key[len("sum__"):]] = d[key]
        elif key.startswith("cnt__"):
            counts[key[len("cnt__"):]] = d[key]
    return {
        "sums":     sums,
        "counts":   counts,
        "n_levels": int(d["n_levels"]),
        "n_days":   int(d["n_days"]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

HOUR_BAND_DIURNAL = slice(0, 24)         # full diurnal cycle (default)
HOUR_BAND_DAYTIME = slice(12, 19)        # 12–18 LT convection window (unused)
HOUR_BAND_DEFAULT = HOUR_BAND_DIURNAL
HOUR_BAND_LABEL   = "24-hour mean"


def _mean_over_hours(profile_lh: np.ndarray, hours: slice) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(profile_lh[:, hours], axis=1)


def _mean_scalar_over_hours(vec: np.ndarray, hours: slice) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return float(np.nanmean(vec[hours]))


def _draw_freeze_line(ax, finals: dict[str, dict[str, np.ndarray]]):
    """Draw a dashed horizontal line at the per-experiment mean 0 C isotherm.

    Colored the same as each experiment so it's obvious which line belongs to
    which model; a single annotation labels them as "0 C isotherm".
    """
    for exp in finals:
        z0 = _mean_scalar_over_hours(finals[exp]["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(z0 / 1000.0, color=EXP_COLORS[exp], lw=1.0, ls="--",
                       alpha=0.7, zorder=1)
    # Legend hint: add a neutral dashed line entry so readers can identify.
    ax.plot([], [], color="k", lw=1.0, ls="--", alpha=0.7, label=r"0 $^\circ$C isotherm")


def plot_b1_species_fraction(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B1 — fractional storage partition by species."""
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 6.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (exp, label) in zip(axes, EXPERIMENTS.items()):
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        total = _mean_over_hours(f["mass_per_level_total"], HOUR_BAND_DEFAULT)
        total_safe = np.where(total > 0, total, np.nan)

        # Build fractional columns per species; stacked-area-style plot.
        sp_fracs: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            prof = _mean_over_hours(f[f"mass_per_level_{sp}"], HOUR_BAND_DEFAULT)
            sp_fracs[sp] = prof / total_safe

        order = list(SPECIES)
        lower = np.zeros_like(z_km)
        for sp in order:
            upper = lower + np.nan_to_num(sp_fracs[sp], nan=0.0)
            ax.fill_betweenx(
                z_km, lower, upper,
                color=SPECIES_COLOR[sp], alpha=0.85,
                label=SPECIES_LABEL[sp],
            )
            lower = upper

        # 0 C isotherm for this experiment only (panel per experiment).
        z0 = _mean_scalar_over_hours(f["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(z0 / 1000.0, color="k", lw=1.2, ls="--", alpha=0.85,
                       label=r"0 $^\circ$C isotherm")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel("Fraction of total condensate mass")
        ax.set_title(label)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Species partition of column condensate mass",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b2_surface_reach(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B2 — surface-reach fraction vs hour (surface flux / max column flux)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        surf = f["surface_flux"]
        col  = f["max_column_flux"]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(col > 0, surf / col, np.nan)
        ax.plot(np.arange(24), frac, color=EXP_COLORS[exp], lw=2, label=label)
    ax.set_xlim(0, 23)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Surface flux / max column flux")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Fraction of precipitation flux reaching the surface")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


FLUX_FLOOR = 1e-8  # kg m-2 s-1; below this, residence time is meaningless


def plot_b3_residence_time(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B3 — per-level residence time tau(z) = mass_above(z) / flux(z)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    ax_total, ax_frozen = axes

    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        mass_tot = _mean_over_hours(f["mass_above_total"], HOUR_BAND_DEFAULT)
        flux     = _mean_over_hours(f["flux_total"],       HOUR_BAND_DEFAULT)
        flux_ok = flux > FLUX_FLOOR
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_total = np.where(flux_ok, mass_tot / flux, np.nan)

        # Frozen-species tau: use snow + graupel mass / total flux as proxy.
        mass_snow    = _mean_over_hours(f.get("mass_above_SNOW",    np.zeros_like(flux)), HOUR_BAND_DEFAULT)
        mass_graupel = _mean_over_hours(f.get("mass_above_GRAUPEL", np.zeros_like(flux)), HOUR_BAND_DEFAULT)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_snow    = np.where(flux_ok, mass_snow    / flux, np.nan)
            tau_graupel = np.where(flux_ok, mass_graupel / flux, np.nan)

        ax_total.plot(tau_total, z_km, color=EXP_COLORS[exp], lw=2, label=label)
        ax_frozen.plot(tau_snow,    z_km, color=EXP_COLORS[exp], lw=2, ls="-",  label=f"{label} snow")
        ax_frozen.plot(tau_graupel, z_km, color=EXP_COLORS[exp], lw=2, ls="--", label=f"{label} graupel")

    for ax in axes:
        _draw_freeze_line(ax, finals)
        ax.set_xscale("log")
        ax.set_xlim(1e1, 1e6)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel(r"Residence time $\tau$ (s)")
        ax.grid(which="both", alpha=0.3)
    ax_total.set_ylabel("Height (km)")
    ax_total.set_title("All condensate")
    ax_frozen.set_title("Snow (solid) vs graupel (dashed)")
    ax_total.legend(loc="upper right")
    ax_frozen.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        r"Per-level condensate residence time $\tau(z) = m_{>z} / F(z)$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b4_flux_divergence(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """B4 — -dF/dz profile.  Positive = source, negative = sink (evap/sublim)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    ax_raw, ax_norm = axes

    for exp, label in EXPERIMENTS.items():
        f = finals[exp]
        z_km  = _mean_over_hours(f["height_m"],  HOUR_BAND_DEFAULT) / 1000.0
        fdiv  = _mean_over_hours(f["flux_div"],  HOUR_BAND_DEFAULT)
        surf  = _mean_scalar_over_hours(f["surface_flux"], HOUR_BAND_DEFAULT)

        ax_raw.plot(fdiv * 1e6, z_km, color=EXP_COLORS[exp], lw=2, label=label)
        if np.isfinite(surf) and surf > 0:
            ax_norm.plot(fdiv / surf * 1000.0, z_km, color=EXP_COLORS[exp], lw=2, label=label)

    for ax in axes:
        _draw_freeze_line(ax, finals)
        ax.axvline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_ylim(0.0, 18.0)
        ax.grid(alpha=0.3)
    ax_raw.set_ylabel("Height (km)")
    ax_raw.set_xlabel(r"$-dF/dz$ ($10^{-6}$ kg m$^{-3}$ s$^{-1}$)")
    ax_raw.set_title("Raw divergence")
    ax_norm.set_xlabel(r"$-dF/dz \, / \, F_{\mathrm{surf}}$ ($10^{-3}$ m$^{-1}$)")
    ax_norm.set_title("Normalized by surface flux")
    ax_raw.legend(loc="upper right")
    fig.suptitle(
        "Precipitation flux divergence  |  "
        "positive = net source, negative = net sink",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


def plot_b1b_mass_above_species(finals: dict[str, dict[str, np.ndarray]], output_path: Path):
    """Companion to B1 — absolute column mass-above per species."""
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 6.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (exp, label) in zip(axes, EXPERIMENTS.items()):
        f = finals[exp]
        z_km = _mean_over_hours(f["height_m"], HOUR_BAND_DEFAULT) / 1000.0
        for sp in SPECIES:
            prof = _mean_over_hours(f[f"mass_above_{sp}"], HOUR_BAND_DEFAULT)
            ax.plot(prof * 1000.0, z_km, color=SPECIES_COLOR[sp], lw=2, label=SPECIES_LABEL[sp])

        z0 = _mean_scalar_over_hours(f["z_freeze"], HOUR_BAND_DEFAULT)
        if np.isfinite(z0):
            ax.axhline(z0 / 1000.0, color="k", lw=1.2, ls="--", alpha=0.85,
                       label=r"0 $^\circ$C isotherm")

        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e2)
        ax.set_ylim(0.0, 18.0)
        ax.set_xlabel(r"Mass above level (g m$^{-2}$)")
        ax.set_title(label)
        ax.grid(which="both", alpha=0.3)

    axes[0].set_ylabel("Height (km)")
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Integrated condensate mass above height, per species",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit days per experiment (for quick tests).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if per-experiment cache exists.")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                        choices=list(EXPERIMENTS),
                        help="Subset of experiments to process.")
    args = parser.parse_args()

    finals: dict[str, dict[str, np.ndarray]] = {}
    for exp in args.experiments:
        cached = None if args.force else load_budget(exp)
        if cached is None:
            cached = accumulate(exp, max_days=args.max_days)
            save_budget(exp, cached)
        else:
            print(f"  budget {exp}: loaded cache ({cached['n_days']} days)")
        finals[exp] = _finalize(cached["sums"], cached["counts"])

    # Need all three experiments present for the comparison plots; fall back
    # to whichever subset is loaded if user asked for fewer.
    if set(finals) != set(EXPERIMENTS):
        print(f"  note: plotting with subset {list(finals)} only", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_b1_species_fraction(finals,  OUTPUT_DIR / "B1_species_fraction.png")
    plot_b1b_mass_above_species(finals, OUTPUT_DIR / "B1b_mass_above_species.png")
    plot_b2_surface_reach(finals,     OUTPUT_DIR / "B2_surface_reach_fraction.png")
    plot_b3_residence_time(finals,    OUTPUT_DIR / "B3_residence_time.png")
    plot_b4_flux_divergence(finals,   OUTPUT_DIR / "B4_flux_divergence.png")


if __name__ == "__main__":
    main()
