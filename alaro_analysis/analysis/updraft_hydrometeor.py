"""Updraft hydrometeor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.common.parallel import imap_unordered_progress
from alaro_analysis.common.updraft_hydrometeor_settings import (
    CACHE_DIR,
    DATA_ROOT,
    H_BINS,
    MIN_LEAD_HOUR,
    N_WORKERS,
    STRATIFY_CONFIGS,
    G,
)
from alaro_analysis.data.cache import load_cache as read_npz_payload
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.masked_fields import list_days, list_steps, read_field


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

    sums = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
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


def accumulate_histograms(
    experiment: str,
    hydrometeors: list[str],
    max_days: int | None = None,
    stratify: str = "flux",
    *,
    workers: int = N_WORKERS,
    data_root: Path = DATA_ROOT,
) -> dict:
    """
    Accumulate 2D histograms using multiprocessing over days.

    Parameters
    ----------
    stratify : str
        Stratification variable: ``"flux"``, ``"extent"``, or ``"intensity"``.
    """
    x_bins = STRATIFY_CONFIGS[stratify]["bins"]
    exp_dir = data_root / experiment
    nx = len(x_bins) - 1
    nh = len(H_BINS) - 1
    days = list_days(exp_dir, "UD_OMEGA")
    if max_days is not None:
        days = days[:max_days]
    tasks = [(exp_dir, d, hydrometeors, MIN_LEAD_HOUR, stratify, x_bins) for d in days]
    print(
        f"  {experiment} [{stratify}]: processing {len(days)} days with {workers} workers...",
        flush=True,
    )
    sums = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    counts = {h: np.zeros((nx, nh), dtype=np.float64) for h in hydrometeors}
    freq = np.zeros((nx, nh), dtype=np.float64)
    n_files = 0
    for i, result in enumerate(imap_unordered_progress(_process_day, tasks, workers=workers)):
        if result is None:
            continue
        freq += result["freq"]
        n_files += result["n_files"]
        for h in hydrometeors:
            sums[h] += result["sums"][h]
            counts[h] += result["counts"][h]
        if (i + 1) % 100 == 0:
            print(
                f"  {experiment} [{stratify}]: {i + 1}/{len(days)} days done, {n_files} files",
                flush=True,
            )
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


def save_cache(
    experiment: str,
    result: dict,
    stratify: str = "flux",
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if stratify == "flux" else f"_{stratify}"
    outpath = cache_dir / f"{experiment}{suffix}.npz"
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
    save_npz_compressed(outpath, **save_dict, **context.arrays(outpath) if context else {})
    print(f"  Saved cache: {outpath}")


def load_cache(
    experiment: str,
    stratify: str = "flux",
    *,
    cache_dir: Path = CACHE_DIR,
    context: CacheContext | None = None,
) -> dict | None:
    suffix = "" if stratify == "flux" else f"_{stratify}"
    path = cache_dir / f"{experiment}{suffix}.npz"
    if not (path.exists() and (context is None or context.valid(path))):
        return None
    data = read_npz_payload(path)
    hydros = [k.replace("sum_", "") for k in data.keys() if k.startswith("sum_")]
    x_bins = data["x_bins"] if "x_bins" in data.keys() else data["flux_bins"]
    return {
        "sums": {h: data[f"sum_{h}"] for h in hydros},
        "counts": {h: data[f"cnt_{h}"] for h in hydros},
        "freq": data["freq"],
        "n_files": int(data["n_files"]),
        "x_bins": x_bins,
        "h_bins": data["h_bins"],
        "stratify": stratify,
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
