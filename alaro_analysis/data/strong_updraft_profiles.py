"""Strong updraft profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.strong_profile_settings import H_BINS_KM, ISOTHERMS_C, STEP_RE, TEMP_VAR
from alaro_analysis.common.study_specs import RAIN_PROFILE, ProfileSpec
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.cache_context import CacheContext


@dataclass(frozen=True)
class RainProfile:
    experiment: str
    h_bins_km: np.ndarray
    rain_kg_kg: np.ndarray
    rain_count: np.ndarray
    temperature_k: np.ndarray
    temperature_count: np.ndarray
    strong_sample_count: np.ndarray
    isotherm_heights_km: dict[float, float]
    n_days_seen: int
    n_files_seen: int
    n_files_used: int
    n_files_with_strong_updraft: int
    missing_files: int
    source: str


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cache_path(
    args: argparse.Namespace, experiment: str, *, spec: ProfileSpec = RAIN_PROFILE
) -> Path:
    day_tag = "all-days" if args.max_days is None else f"{args.max_days}days"
    lead_tag = f"lead{args.min_lead_hour}_to{args.max_lead_hour}"
    return (
        args.cache_dir
        / f"{experiment}_strong_updraft_{spec.slug}_omega{slug_float(args.min_updraft_pa_s)}_mesh{slug_float(args.min_updraft_mesh_frac)}_{lead_tag}_{day_tag}.npz"
    )


def list_days(exp_dir: Path) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / "UD_OMEGA"
    if not var_dir.exists():
        return []
    return sorted(p for p in var_dir.iterdir() if p.is_dir() and p.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int, max_lead_hour: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(day_dir.glob("*.nc")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        lead = int(match.group(1))
        if min_lead_hour <= lead <= max_lead_hour:
            out.append(path)
    return out


def read_field(path: Path, variable: str) -> np.ndarray:
    with Dataset(path) as ds:
        var_name = variable if variable in ds.variables else None
        if var_name is None:
            data_vars = [
                name
                for name, var in ds.variables.items()
                if var.dimensions and name not in {"time", "level", "x", "y", "lat", "lon"}
            ]
            if not data_vars:
                raise KeyError(f"No data variable found in {path}")
            var_name = data_vars[0]
        raw = ds.variables[var_name][:]

    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected (level, y, x) after squeeze for {path}, got {arr.shape}")
    return arr


def _empty_accumulators() -> dict[str, Any]:
    n_height = len(H_BINS_KM) - 1
    return {
        "rain_sum": np.zeros(n_height, dtype=np.float64),
        "rain_count": np.zeros(n_height, dtype=np.int64),
        "temperature_sum": np.zeros(n_height, dtype=np.float64),
        "temperature_count": np.zeros(n_height, dtype=np.int64),
        "strong_sample_count": np.zeros(n_height, dtype=np.int64),
        "n_files_seen": 0,
        "n_files_used": 0,
        "n_files_with_strong_updraft": 0,
        "missing_files": 0,
    }


def process_day(
    task: tuple[Path, Path, int, int, float, float], *, spec: ProfileSpec = RAIN_PROFILE
) -> dict[str, Any]:
    if len(task) == 7:
        task, spec = (task[:6], task[6])
    exp_dir, day_dir, min_lead_hour, max_lead_hour, min_updraft_pa_s, min_mesh_frac = task
    data_dir = exp_dir / "masked-netcdf"
    out = _empty_accumulators()
    for omega_path in list_steps(day_dir, min_lead_hour, max_lead_hour):
        day_name = omega_path.parent.name
        file_name = omega_path.name
        out["n_files_seen"] += 1
        mesh_path = data_dir / "UD_MESH_FRAC" / day_name / file_name
        height_path = data_dir / "GEOPOTENTIEL" / day_name / file_name
        rain_path = data_dir / spec.variable / day_name / file_name
        temp_path = data_dir / TEMP_VAR / day_name / file_name
        if not all((path.exists() for path in (mesh_path, height_path, rain_path, temp_path))):
            out["missing_files"] += 1
            continue
        try:
            omega = read_field(omega_path, "UD_OMEGA")
            mesh = read_field(mesh_path, "UD_MESH_FRAC")
            height_km = read_field(height_path, "GEOPOTENTIEL") / 1000.0
            rain = read_field(rain_path, spec.variable)
            temperature = read_field(temp_path, TEMP_VAR)
        except Exception:
            out["missing_files"] += 1
            continue
        strong_mask = (
            np.isfinite(omega)
            & np.isfinite(mesh)
            & np.isfinite(height_km)
            & (mesh > min_mesh_frac)
            & (-omega >= min_updraft_pa_s)
        )
        out["n_files_used"] += 1
        if not np.any(strong_mask):
            continue
        height_flat = height_km[strong_mask]
        h_idx = np.digitize(height_flat, H_BINS_KM) - 1
        valid_height = (h_idx >= 0) & (h_idx < len(H_BINS_KM) - 1)
        h_idx = h_idx[valid_height]
        if h_idx.size == 0:
            continue
        out["n_files_with_strong_updraft"] += 1
        np.add.at(out["strong_sample_count"], h_idx, 1)
        rain_values = rain[strong_mask][valid_height]
        rain_finite = np.isfinite(rain_values)
        if np.any(rain_finite):
            rain_values = np.maximum(rain_values[rain_finite], 0.0)
            np.add.at(out["rain_sum"], h_idx[rain_finite], rain_values)
            np.add.at(out["rain_count"], h_idx[rain_finite], 1)
        temp_values = temperature[strong_mask][valid_height]
        temp_finite = np.isfinite(temp_values)
        if np.any(temp_finite):
            np.add.at(out["temperature_sum"], h_idx[temp_finite], temp_values[temp_finite])
            np.add.at(out["temperature_count"], h_idx[temp_finite], 1)
    return out


def combine_day_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = _empty_accumulators()
    for result in results:
        for key in (
            "rain_sum",
            "rain_count",
            "temperature_sum",
            "temperature_count",
            "strong_sample_count",
        ):
            out[key] += result[key]
        for key in (
            "n_files_seen",
            "n_files_used",
            "n_files_with_strong_updraft",
            "missing_files",
        ):
            out[key] += result[key]
    return out


def mean_from_sum_count(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(sums.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def isotherm_height_km(
    height_km: np.ndarray,
    temperature_k: np.ndarray,
    target_c: float,
) -> float:
    temperature_c = temperature_k - 273.15
    finite = np.isfinite(height_km) & np.isfinite(temperature_c)
    z = height_km[finite]
    t = temperature_c[finite]
    if z.size < 2:
        return float("nan")
    order = np.argsort(z)
    z = z[order]
    t = t[order]
    diff = t - target_c
    for i in range(diff.size - 1):
        if diff[i] == 0:
            return float(z[i])
        if diff[i] * diff[i + 1] <= 0 and t[i] != t[i + 1]:
            frac = (target_c - t[i]) / (t[i + 1] - t[i])
            return float(z[i] + frac * (z[i + 1] - z[i]))
    return float("nan")


def build_profile(
    args: argparse.Namespace, experiment: str, *, spec: ProfileSpec = RAIN_PROFILE
) -> RainProfile:
    exp_dir = args.data_root / experiment
    days = list_days(exp_dir)
    if args.max_days is not None:
        days = days[: args.max_days]
    if not days:
        raise RuntimeError(f"No day directories found for {experiment}: {exp_dir}")
    print(
        f"[{experiment}] processing {len(days)} days with {args.workers} workers; lead {args.min_lead_hour}..{args.max_lead_hour}; -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s",
        flush=True,
    )
    tasks = [
        (
            exp_dir,
            day,
            args.min_lead_hour,
            args.max_lead_hour,
            args.min_updraft_pa_s,
            args.min_updraft_mesh_frac,
            spec,
        )
        for day in days
    ]
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            results.append(process_day(task, spec=spec))
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"[{experiment}] {idx}/{len(tasks)} days", flush=True)
    else:
        with get_context("fork").Pool(processes=args.workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(process_day, tasks), start=1):
                results.append(result)
                if idx % args.progress_every == 0 or idx == len(tasks):
                    used = sum((r["n_files_used"] for r in results))
                    strong = sum((r["n_files_with_strong_updraft"] for r in results))
                    print(
                        f"[{experiment}] {idx}/{len(tasks)} days; {used} files read; {strong} with strong updrafts",
                        flush=True,
                    )
    combined = combine_day_results(results)
    rain = mean_from_sum_count(combined["rain_sum"], combined["rain_count"])
    temperature = mean_from_sum_count(combined["temperature_sum"], combined["temperature_count"])
    centers = 0.5 * (H_BINS_KM[:-1] + H_BINS_KM[1:])
    isotherms = {target: isotherm_height_km(centers, temperature, target) for target in ISOTHERMS_C}
    return RainProfile(
        experiment=experiment,
        h_bins_km=H_BINS_KM.copy(),
        rain_kg_kg=rain,
        rain_count=combined["rain_count"],
        temperature_k=temperature,
        temperature_count=combined["temperature_count"],
        strong_sample_count=combined["strong_sample_count"],
        isotherm_heights_km=isotherms,
        n_days_seen=len(days),
        n_files_seen=combined["n_files_seen"],
        n_files_used=combined["n_files_used"],
        n_files_with_strong_updraft=combined["n_files_with_strong_updraft"],
        missing_files=combined["missing_files"],
        source=str(exp_dir / "masked-netcdf"),
    )


def save_cache(path: Path, profile: RainProfile, *, context: CacheContext | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(list(profile.isotherm_heights_km), dtype=np.float64)
    heights = np.asarray([profile.isotherm_heights_km[t] for t in targets], dtype=np.float64)
    save_npz_compressed(
        path,
        experiment=np.asarray(profile.experiment),
        h_bins_km=profile.h_bins_km,
        rain_kg_kg=profile.rain_kg_kg,
        rain_count=profile.rain_count,
        temperature_k=profile.temperature_k,
        temperature_count=profile.temperature_count,
        strong_sample_count=profile.strong_sample_count,
        isotherm_targets_c=targets,
        isotherm_heights_km=heights,
        n_days_seen=np.asarray(profile.n_days_seen, dtype=np.int64),
        n_files_seen=np.asarray(profile.n_files_seen, dtype=np.int64),
        n_files_used=np.asarray(profile.n_files_used, dtype=np.int64),
        n_files_with_strong_updraft=np.asarray(profile.n_files_with_strong_updraft, dtype=np.int64),
        missing_files=np.asarray(profile.missing_files, dtype=np.int64),
        source=np.asarray(profile.source),
        **(context.arrays(path) if context else {}),
    )


def load_cache(path: Path) -> RainProfile:
    with np.load(path, allow_pickle=False) as data:
        targets = np.asarray(data["isotherm_targets_c"], dtype=np.float64)
        heights = np.asarray(data["isotherm_heights_km"], dtype=np.float64)
        return RainProfile(
            experiment=str(data["experiment"]),
            h_bins_km=np.asarray(data["h_bins_km"], dtype=np.float64),
            rain_kg_kg=np.asarray(data["rain_kg_kg"], dtype=np.float64),
            rain_count=np.asarray(data["rain_count"], dtype=np.int64),
            temperature_k=np.asarray(data["temperature_k"], dtype=np.float64),
            temperature_count=np.asarray(data["temperature_count"], dtype=np.int64),
            strong_sample_count=np.asarray(data["strong_sample_count"], dtype=np.int64),
            isotherm_heights_km={
                float(target): float(height) for target, height in zip(targets, heights)
            },
            n_days_seen=int(data["n_days_seen"]),
            n_files_seen=int(data["n_files_seen"]),
            n_files_used=int(data["n_files_used"]),
            n_files_with_strong_updraft=int(data["n_files_with_strong_updraft"]),
            missing_files=int(data["missing_files"]),
            source=str(data["source"]),
        )


def get_profile(
    args: argparse.Namespace, experiment: str, *, spec: ProfileSpec = RAIN_PROFILE
) -> RainProfile:
    path = cache_path(args, experiment, spec=spec)
    root = args.data_root / experiment / "masked-netcdf"
    context = CacheContext.from_netcdf(
        "strong_updraft_profile",
        {**vars(args), "variable": spec.variable, "height_bins": H_BINS_KM},
        [
            root / variable
            for variable in (spec.variable, TEMP_VAR, "UD_OMEGA", "UD_MESH_FRAC", "GEOPOTENTIEL")
        ],
    )
    if context.valid(path) and (not args.recompute):
        print(f"[{experiment}] using cache {path}", flush=True)
        return load_cache(path)
    profile = build_profile(args, experiment, spec=spec)
    save_cache(path, profile, context=context)
    print(f"[{experiment}] saved cache {path}", flush=True)
    return profile
