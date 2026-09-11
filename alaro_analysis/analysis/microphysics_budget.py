"""Microphysics budget."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from alaro_analysis.common.microphysics_budget_settings import (
    CACHE_DIR,
    DATA_ROOT,
    N_WORKERS,
    SPECIES,
    T_FREEZE,
    G,
)
from alaro_analysis.common.parallel import imap_unordered_progress
from alaro_analysis.data.cache import load_cache as read_npz_payload
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.masked_fields import list_days, list_steps, read_field


def _freeze_height(tmp_lyx: np.ndarray, z_lyx: np.ndarray) -> float:
    """Mean height (m) of the 273.15 K isotherm over a (level, y, x) column.

    For each (y, x) column, linearly interpolate T(z) between the two
    consecutive levels that straddle 273.15 K; the lowest such crossing (in
    height) is taken as the freezing level.  Columns entirely above or below
    freezing contribute NaN and are dropped from the spatial mean.
    """
    t = np.asarray(tmp_lyx, dtype=np.float64)
    z = np.asarray(z_lyx, dtype=np.float64)
    if t.shape != z.shape or t.ndim != 3:
        return float("nan")

    diff = t[:-1] - T_FREEZE  # (L-1, y, x)
    diff_n = t[1:] - T_FREEZE
    # Sign change between adjacent levels => 273.15 K crossing in that layer.
    cross = np.sign(diff) != np.sign(diff_n)
    cross &= np.isfinite(diff) & np.isfinite(diff_n)
    if not np.any(cross):
        return float("nan")

    # Linear interpolation weight within the crossing layer.
    with np.errstate(divide="ignore", invalid="ignore"):
        w = diff / (diff - diff_n)
    z_cross = z[:-1] + w * (z[1:] - z[:-1])  # (L-1, y, x)
    z_cross = np.where(cross, z_cross, np.nan)

    # Pick the LOWEST crossing per (y, x) column — robust to multi-crossing
    # profiles (e.g. stratospheric inversions well above the true freezing
    # level).  "Lowest" = smallest height.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        z_freeze_yx = np.nanmin(z_cross, axis=0)  # (y, x)
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
    p_half[0] = p_lyx[0] + (p_lyx[0] - p_half[1])
    p_half[-1] = p_lyx[-1] - (p_half[-2] - p_lyx[-1])
    return np.abs(p_half[:-1] - p_half[1:])


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
            p = read_field(exp_dir / "masked-netcdf" / "PRESSURE" / day_name / step_name)
            gz = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
            cv = read_field(exp_dir / "masked-netcdf" / "CV_PREC_FLUX" / day_name / step_name)
            st = read_field(exp_dir / "masked-netcdf" / "ST_PREC_FLUX" / day_name / step_name)
            tmp = read_field(exp_dir / "masked-netcdf" / "TEMPERATURE" / day_name / step_name)
        except Exception:
            continue

        # PRESSURE in masked cache is generally Pa, but guard against hPa.
        if np.nanmax(p) < 2000.0:
            p = p * 100.0

        species_arr: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            try:
                species_arr[sp] = read_field(exp_dir / "masked-netcdf" / sp / day_name / step_name)
            except Exception:
                pass
        if not species_arr:
            continue

        # Harmonize level counts across species / flux / pressure / temperature.
        arrays = [p, gz, cv, st, tmp, *species_arr.values()]
        min_lev = min(a.shape[0] for a in arrays)
        p = p[:min_lev]
        gz = gz[:min_lev]
        cv = cv[:min_lev]
        st = st[:min_lev]
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
                *(f"mass_per_level_{sp}" for sp in SPECIES),
                *(f"mass_above_{sp}" for sp in SPECIES),
            ):
                sums[key] = np.zeros((n_levels, 24), dtype=np.float64)
                counts[key] = np.zeros((n_levels, 24), dtype=np.int64)
            for key in ("surface_flux", "max_column_flux", "z_freeze"):
                sums[key] = np.zeros(24, dtype=np.float64)
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
        top_idx = int(np.argmin(p_mean))
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
        z_m = np.nanmean(gz, axis=(1, 2))
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
            flux_div[-1] = flux_div[-2] if n_levels >= 2 else 0.0

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
            sums[key][finite, hour] += values[finite]
            counts[key][finite, hour] += 1

        def _add_scalar(key: str, value: float):
            if np.isfinite(value):
                sums[key][hour] += value
                counts[key][hour] += 1

        _add_profile("flux_total", flux_mean)
        _add_profile("flux_div", flux_div)
        _add_profile("mass_above_total", mass_above_total)
        _add_profile("mass_per_level_total", mass_total_lev)
        _add_profile("height_m", z_m)
        _add_profile("pressure_pa", p_mean)
        for sp in SPECIES:
            if sp in mass_species_lev:
                _add_profile(f"mass_per_level_{sp}", mass_species_lev[sp])
                _add_profile(f"mass_above_{sp}", mass_above_species[sp])

        _add_scalar("surface_flux", surface_flux)
        _add_scalar("max_column_flux", max_col_flux)
        _add_scalar("z_freeze", z_freeze)

    if n_levels is None:
        return None
    return {"sums": sums, "counts": counts, "n_levels": n_levels}


def accumulate(
    experiment: str,
    max_days: int | None = None,
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    exp_dir = data_root / experiment
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
    for idx, res in enumerate(imap_unordered_progress(_process_day, tasks, workers=workers)):
        if res is None:
            continue
        if n_levels_seen is None:
            n_levels_seen = res["n_levels"]
            for k, arr in res["sums"].items():
                agg_sums[k] = arr.copy()
                agg_counts[k] = res["counts"][k].copy()
        else:
            if res["n_levels"] != n_levels_seen:
                print(
                    f"  budget {experiment}: level-count drift ({res['n_levels']} vs {n_levels_seen}); skipping",
                    flush=True,
                )
                continue
            for k, arr in res["sums"].items():
                agg_sums[k] += arr
                agg_counts[k] += res["counts"][k]
        n_days_used += 1
        if (idx + 1) % 100 == 0:
            print(f"  budget {experiment}: {idx + 1}/{len(days)} days", flush=True)
    print(f"  budget {experiment}: DONE — {n_days_used} days used", flush=True)
    return {
        "sums": agg_sums,
        "counts": agg_counts,
        "n_levels": int(n_levels_seen) if n_levels_seen is not None else 0,
        "n_days": int(n_days_used),
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


def save_budget(
    experiment: str,
    result: dict,
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"budget_{experiment}.npz"
    flat: dict[str, np.ndarray] = {}
    for k, v in result["sums"].items():
        flat[f"sum__{k}"] = v
    for k, v in result["counts"].items():
        flat[f"cnt__{k}"] = v
    flat["n_levels"] = np.array(result["n_levels"])
    flat["n_days"] = np.array(result["n_days"])
    save_npz_compressed(path, **flat, **context.arrays(path) if context else {})
    print(f"  saved: {path}")


def load_budget(
    experiment: str, *, cache_dir: Path = CACHE_DIR, context: CacheContext | None = None
) -> dict | None:
    path = cache_dir / f"budget_{experiment}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    d = read_npz_payload(path)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for key in d.keys():
        if key.startswith("sum__"):
            sums[key[len("sum__") :]] = d[key]
        elif key.startswith("cnt__"):
            counts[key[len("cnt__") :]] = d[key]
    return {
        "sums": sums,
        "counts": counts,
        "n_levels": int(d["n_levels"]),
        "n_days": int(d["n_days"]),
    }


def _mean_over_hours(profile_lh: np.ndarray, hours: slice) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(profile_lh[:, hours], axis=1)


def _mean_scalar_over_hours(vec: np.ndarray, hours: slice) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return float(np.nanmean(vec[hours]))
