"""Convective feedback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.convective_feedback_settings import (
    CACHE_DIR,
    DATA_ROOT,
    DD_H_BINS,
    H4_H_BINS,
    H_BINS,
    HYDROMETEORS,
    LOADING_BINS,
    MIN_LEAD_HOUR,
    N_WORKERS,
    PRECONV_LEADS,
    THERMO_VARS,
    G,
)
from alaro_analysis.common.parallel import imap_unordered_progress
from alaro_analysis.data.cache import load_cache as read_npz_payload
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.masked_fields import list_days, list_steps, read_field


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
    counts = np.zeros((nx, nh), dtype=np.float64)
    sums_flux = np.zeros((nx, nh), dtype=np.float64)
    freq = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            omega = read_field(exp_dir / "masked-netcdf" / "UD_OMEGA" / day_name / step_name)
            mesh = read_field(exp_dir / "masked-netcdf" / "UD_MESH_FRAC" / day_name / step_name)
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
        h_flat = h_km[mask]
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


def accumulate_h1(
    experiment: str,
    max_days: int | None = None,
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    exp_dir = data_root / experiment
    days = list_days(exp_dir, "UD_OMEGA")
    if max_days is not None:
        days = days[:max_days]
    tasks = [(exp_dir, d, MIN_LEAD_HOUR) for d in days]
    print(f"  H1 {experiment}: processing {len(days)} days ...", flush=True)
    nx = len(LOADING_BINS) - 1
    nh = len(H_BINS) - 1
    sums_intensity = np.zeros((nx, nh), dtype=np.float64)
    sums_flux = np.zeros((nx, nh), dtype=np.float64)
    counts = np.zeros((nx, nh), dtype=np.float64)
    freq = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0
    for i, res in enumerate(imap_unordered_progress(_h1_process_day, tasks, workers=workers)):
        if res is None:
            continue
        sums_intensity += res["sums_intensity"]
        sums_flux += res["sums_flux"]
        counts += res["counts"]
        freq += res["freq"]
        n_files += res["n_files"]
        if (i + 1) % 100 == 0:
            print(f"  H1 {experiment}: {i + 1}/{len(days)} days", flush=True)
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


def save_h1(
    experiment: str,
    result: dict,
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"h1_{experiment}.npz"
    save_npz_compressed(
        path,
        sums_intensity=result["sums_intensity"],
        sums_flux=result["sums_flux"],
        counts=result["counts"],
        freq=result["freq"],
        n_files=np.array(result["n_files"]),
        x_bins=result["x_bins"],
        h_bins=result["h_bins"],
        **context.arrays(path) if context else {},
    )
    print(f"  Saved: {path}")


def load_h1(
    experiment: str, *, cache_dir: Path = CACHE_DIR, context: CacheContext | None = None
) -> dict | None:
    path = cache_dir / f"h1_{experiment}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    d = read_npz_payload(path)
    return {
        "sums_intensity": d["sums_intensity"],
        "sums_flux": d["sums_flux"],
        "counts": d["counts"],
        "freq": d["freq"],
        "n_files": int(d["n_files"]),
        "x_bins": d["x_bins"],
        "h_bins": d["h_bins"],
    }


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
            dd_omega = read_field(exp_dir / "masked-netcdf" / "DD_OMEGA" / day_name / step_name)
            dd_mesh = read_field(exp_dir / "masked-netcdf" / "DD_MESH_FRAC" / day_name / step_name)
            height = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
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
                int_mean = np.nanmean(dd_intensity[lev][lev_mask])
                if np.isfinite(flux_mean):
                    hourly[lead]["flux_sum"][h_idx] += flux_mean
                    hourly[lead]["flux_cnt"][h_idx] += 1.0
                if np.isfinite(int_mean):
                    hourly[lead]["intensity_sum"][h_idx] += int_mean
                    hourly[lead]["intensity_cnt"][h_idx] += 1.0

        hourly[lead]["n"] += 1

    return hourly


def accumulate_h2(
    experiment: str,
    max_days: int | None = None,
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    exp_dir = data_root / experiment
    days = list_days(exp_dir, "DD_OMEGA")
    if max_days is not None:
        days = days[:max_days]
    tasks = [(exp_dir, d) for d in days]
    print(f"  H2 {experiment}: processing {len(days)} days ...", flush=True)
    nh = len(DD_H_BINS) - 1
    profiles = {}
    for var in ("flux", "intensity", "extent"):
        profiles[f"{var}_sum"] = np.zeros((24, nh), dtype=np.float64)
        profiles[f"{var}_cnt"] = np.zeros((24, nh), dtype=np.float64)
    n_files = 0
    for i, res in enumerate(imap_unordered_progress(_h2_process_day, tasks, workers=workers)):
        if res is None:
            continue
        for lead, data in res.items():
            hour = lead % 24
            for var in ("flux", "intensity", "extent"):
                profiles[f"{var}_sum"][hour] += data[f"{var}_sum"]
                profiles[f"{var}_cnt"][hour] += data[f"{var}_cnt"]
            n_files += data["n"]
        if (i + 1) % 100 == 0:
            print(f"  H2 {experiment}: {i + 1}/{len(days)} days", flush=True)
    print(f"  H2 {experiment}: DONE — {n_files} files", flush=True)
    profiles["n_files"] = n_files
    profiles["h_bins"] = DD_H_BINS
    return profiles


def save_h2(
    experiment: str,
    result: dict,
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"h2_{experiment}.npz"
    save_dict = {k: np.array(v) for k, v in result.items()}
    save_npz_compressed(path, **save_dict, **context.arrays(path) if context else {})
    print(f"  Saved: {path}")


def load_h2(
    experiment: str, *, cache_dir: Path = CACHE_DIR, context: CacheContext | None = None
) -> dict | None:
    path = cache_dir / f"h2_{experiment}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    d = read_npz_payload(path)
    return {k: d[k] for k in d.keys()}


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
            height = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
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


def accumulate_h3(
    experiment: str,
    max_days: int | None = None,
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    exp_dir = data_root / experiment
    days = list_days(exp_dir, "GEOPOTENTIEL")
    if max_days is not None:
        days = days[:max_days]
    tasks = [(exp_dir, d) for d in days]
    print(f"  H3 {experiment}: processing {len(days)} days ...", flush=True)
    nh = len(H_BINS) - 1
    sums = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    counts = {v: np.zeros(nh, dtype=np.float64) for v in THERMO_VARS}
    n_files = 0
    for i, res in enumerate(imap_unordered_progress(_h3_process_day, tasks, workers=workers)):
        if res is None:
            continue
        for v in THERMO_VARS:
            sums[v] += res["sums"][v]
            counts[v] += res["counts"][v]
        n_files += res["n"]
        if (i + 1) % 100 == 0:
            print(f"  H3 {experiment}: {i + 1}/{len(days)} days", flush=True)
    print(f"  H3 {experiment}: DONE — {n_files} files", flush=True)
    return {"sums": sums, "counts": counts, "n_files": n_files, "h_bins": H_BINS}


def save_h3(
    experiment: str,
    result: dict,
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"h3_{experiment}.npz"
    save_dict = {"n_files": np.array(result["n_files"]), "h_bins": result["h_bins"]}
    for v in THERMO_VARS:
        save_dict[f"sum_{v}"] = result["sums"][v]
        save_dict[f"cnt_{v}"] = result["counts"][v]
    save_npz_compressed(path, **save_dict, **context.arrays(path) if context else {})
    print(f"  Saved: {path}")


def load_h3(
    experiment: str, *, cache_dir: Path = CACHE_DIR, context: CacheContext | None = None
) -> dict | None:
    path = cache_dir / f"h3_{experiment}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    d = read_npz_payload(path)
    variables = [k.replace("sum_", "") for k in d.keys() if k.startswith("sum_")]
    return {
        "sums": {v: d[f"sum_{v}"] for v in variables},
        "counts": {v: d[f"cnt_{v}"] for v in variables},
        "n_files": int(d["n_files"]),
        "h_bins": d["h_bins"],
    }


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
    dy = dlat_dy * 111_000.0  # metres

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
        for k in (
            "div_sum",
            "div_cnt",
            "mfdiv_sum",
            "mfdiv_cnt",
            "qdiv_sum",
            "qdiv_cnt",
            "q_sum",
            "q_cnt",
        )
    }
    n_files = 0

    for step_file in steps:
        step_name = step_file.name
        try:
            u = read_field(exp_dir / "masked-netcdf" / "WIND.U.PHYS" / day_name / step_name)
            v = read_field(exp_dir / "masked-netcdf" / "WIND.V.PHYS" / day_name / step_name)
            q = read_field(exp_dir / "masked-netcdf" / "HUMI.SPECIFI" / day_name / step_name)
            h = read_field(exp_dir / "masked-netcdf" / "GEOPOTENTIEL" / day_name / step_name)
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
            du_dx = np.gradient(u_lev, axis=1) / dx  # ∂u/∂x
            dv_dy = np.gradient(v_lev, axis=0) / dy  # ∂v/∂y

            div = du_dx + dv_dy  # wind divergence (s⁻¹)

            qu = q_lev * u_lev
            qv = q_lev * v_lev
            dqu_dx = np.gradient(qu, axis=1) / dx
            dqv_dy = np.gradient(qv, axis=0) / dy

            mf_div = dqu_dx + dqv_dy  # moisture flux divergence (kg/kg/s)
            q_times_div = q_lev * div  # divergence acting on moisture

            # Domain means (ignoring NaN from masked points)
            div_mean = np.nanmean(div)
            mf_mean = np.nanmean(mf_div)
            qd_mean = np.nanmean(q_times_div)
            q_mean = np.nanmean(q_lev)

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


def accumulate_h4(
    experiment: str,
    max_days: int | None = None,
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    exp_dir = data_root / experiment
    days = list_days(exp_dir, "WIND.U.PHYS")
    if max_days is not None:
        days = days[:max_days]
    sample_step = list_steps(days[0], MIN_LEAD_HOUR)[0]
    sample_path = exp_dir / "masked-netcdf" / "WIND.U.PHYS" / days[0].name / sample_step.name
    dx, dy = _estimate_grid_spacing(sample_path)
    tasks = [(exp_dir, d, MIN_LEAD_HOUR, dx, dy) for d in days]
    print(f"  H4 {experiment}: processing {len(days)} days ...", flush=True)
    nh = len(H4_H_BINS) - 1
    totals = {
        k: np.zeros(nh, dtype=np.float64)
        for k in (
            "div_sum",
            "div_cnt",
            "mfdiv_sum",
            "mfdiv_cnt",
            "qdiv_sum",
            "qdiv_cnt",
            "q_sum",
            "q_cnt",
        )
    }
    n_files = 0
    for i, res in enumerate(imap_unordered_progress(_h4_process_day, tasks, workers=workers)):
        if res is None:
            continue
        for k in totals:
            totals[k] += res[k]
        n_files += res["n_files"]
        if (i + 1) % 100 == 0:
            print(f"  H4 {experiment}: {i + 1}/{len(days)} days", flush=True)
    print(f"  H4 {experiment}: DONE — {n_files} files", flush=True)
    totals["n_files"] = n_files
    totals["h_bins"] = H4_H_BINS
    return totals


def save_h4(
    experiment: str,
    result: dict,
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"h4_{experiment}.npz"
    save_dict = {k: np.array(v) for k, v in result.items()}
    save_npz_compressed(path, **save_dict, **context.arrays(path) if context else {})
    print(f"  Saved: {path}")


def load_h4(
    experiment: str, *, cache_dir: Path = CACHE_DIR, context: CacheContext | None = None
) -> dict | None:
    path = cache_dir / f"h4_{experiment}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    d = read_npz_payload(path)
    return {k: d[k] for k in d.keys()}


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
