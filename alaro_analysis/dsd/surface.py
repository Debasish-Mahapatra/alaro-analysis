"""Surface. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import RD, RUNS_ROOT
from alaro_analysis.common.dsd import (
    DEFAULT_QC_MIN_RUN_MINUTES,
    DEFAULT_QC_RAIN_RATE_MIN,
    DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    apply_disdrometer_qc,
    empirical_dsd_parameters,
    mp_diagnostics_from_n0_lambda,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
    mp_lambda_from_lwc_nt,
)
from alaro_analysis.data.disdrometer import extract_strict_obs
from alaro_analysis.data.masks import DomainMask

PROCESSED_DIR = RUNS_ROOT / "processed-data" / "disdrometer_dsd"


ANALYTICS_DIR = PROCESSED_DIR / "analytics"


FIGURE_DIR = RUNS_ROOT / "figures" / "disdrometer_dsd"


MASK_FILE = RUNS_ROOT / "mask" / "Radar_mask_latlon.nc"


NETCDF_ROOT = RUNS_ROOT / "ALARO"


NETCDF_BOTTOM_LEVEL_INDEX = 0  # in the converted netCDF, level=0 is the surface


OBS_PARAMETERS = ("dm_mm", "d0_mm", "sigma_m_mm", "log_nw", "lwc_g_m3", "nt_m3")


PF_DAY_RE = re.compile(r"^pf(\d{8})$")


PF_FILE_RE = re.compile(r"^pfABOFABOF\+(\d{4})\.nc$")


def discover_netcdf_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path = NETCDF_ROOT,
    max_days: int | None = None,
    reference_var: str = "RAIN",
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    base = netcdf_root / experiment / "masked-netcdf"
    ref_dir = base / reference_var
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing variable folder: {ref_dir}")
    needed_vars = ["RAIN", "TEMPERATURE", "PRESSURE"]
    if experiment == "2mom":
        needed_vars.append("PNR")
    day_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]
    lead_set = set(leads) if leads is not None else None
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for day_dir in day_dirs:
        day_match = PF_DAY_RE.match(day_dir.name)
        if not day_match:
            continue
        init_dt = datetime.strptime(day_match.group(1), "%Y%m%d")
        for path in sorted(day_dir.iterdir()):
            file_match = PF_FILE_RE.match(path.name)
            if not file_match:
                continue
            lead = int(file_match.group(1))
            if lead_set is not None and lead not in lead_set:
                continue
            paths: dict[str, Path] = {}
            ok = True
            for var in needed_vars:
                candidate = base / var / day_dir.name / path.name
                if not candidate.exists():
                    ok = False
                    break
                paths[var] = candidate
            if not ok:
                continue
            valid_dt = init_dt + timedelta(hours=lead)
            records.append((np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths))
    records.sort(key=lambda x: x[0])
    return records


_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_bottom_level_mean(
    path: Path,
    var: str,
    *,
    level_index: int = NETCDF_BOTTOM_LEVEL_INDEX,
) -> tuple[float, int]:
    with Dataset(path) as ds:
        data = ds.variables[var]
        if data.ndim == 4:
            field = np.asarray(data[0, level_index], dtype=np.float64)
        elif data.ndim == 3:
            field = np.asarray(data[level_index], dtype=np.float64)
        elif data.ndim == 2:
            field = np.asarray(data[:], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected ndim for {var} in {path}: {data.ndim}")
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    masked = np.where(_WORKER_MASK, field, np.nan)
    finite = np.isfinite(masked)
    if not finite.any():
        return float("nan"), 0
    return float(np.nanmean(masked[finite])), int(finite.sum())


def _process_timestep_task(
    task: tuple[np.datetime64, np.datetime64, int, str, dict[str, str], float, str, float],
) -> tuple[dict[str, Any], list[str]]:
    valid_time, init_time, lead, experiment, paths, min_qr, onemom_closure, n0_fixed = task
    warnings: list[str] = []
    row: dict[str, Any] = {
        "valid_time": valid_time,
        "init_time": init_time,
        "lead_hours": lead,
        "experiment": experiment,
        "qr_kgkg": np.nan,
        "n_per_kg": np.nan,
        "temperature_k": np.nan,
        "pressure_pa": np.nan,
        "rho_air_kg_m3": np.nan,
        "valid_cells": 0,
    }
    for key in (
        "lwc_g_m3",
        "nt_m3",
        "lambda_per_mm",
        "n0_per_m3_mm",
        "dm_mm",
        "d0_mm",
        "sigma_m_mm",
        "log_nw",
    ):
        row[key] = np.nan

    try:
        qr_mean, n_cells = _read_bottom_level_mean(Path(paths["RAIN"]), "RAIN")
        temp_mean, _ = _read_bottom_level_mean(Path(paths["TEMPERATURE"]), "TEMPERATURE")
        pres_mean, _ = _read_bottom_level_mean(Path(paths["PRESSURE"]), "PRESSURE")
        if experiment == "2mom":
            n_kg, _ = _read_bottom_level_mean(Path(paths["PNR"]), "PNR")
        else:
            n_kg = np.nan  # 1-mom uses fixed-N0 closure below
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"WARNING {experiment} {valid_time} +{lead:04d}: {exc}")
        return row, warnings

    rho = (
        pres_mean / (RD * temp_mean)
        if (np.isfinite(pres_mean) and np.isfinite(temp_mean) and temp_mean > 0)
        else np.nan
    )
    row.update(
        qr_kgkg=qr_mean,
        temperature_k=temp_mean,
        pressure_pa=pres_mean,
        rho_air_kg_m3=rho,
        valid_cells=n_cells,
    )

    if not np.isfinite(qr_mean) or qr_mean < min_qr or not np.isfinite(rho) or rho <= 0.0:
        return row, warnings

    if experiment == "2mom":
        if not np.isfinite(n_kg) or n_kg <= 0.0:
            return row, warnings
        row["n_per_kg"] = n_kg
        diag = mp_from_q_n_per_kg(
            q_r_kgkg=np.asarray([qr_mean]),
            n_r_per_kg=np.asarray([n_kg]),
            rho_air_kg_m3=np.asarray([rho]),
        )
    else:
        if onemom_closure == "abel_boutle":
            diag = mp_from_q_abel_boutle(
                q_r_kgkg=np.asarray([qr_mean]),
                rho_air_kg_m3=np.asarray([rho]),
            )
        else:
            diag = mp_from_q_fixed_n0(
                q_r_kgkg=np.asarray([qr_mean]),
                rho_air_kg_m3=np.asarray([rho]),
                n0_per_m3_mm=n0_fixed,
            )
        # diagnostic n_kg = N0/lambda/rho_air for sanity / record
        if np.isfinite(diag["nt_m3"][0]) and rho > 0:
            row["n_per_kg"] = float(diag["nt_m3"][0]) / rho

    for key in (
        "lwc_g_m3",
        "nt_m3",
        "lambda_per_mm",
        "n0_per_m3_mm",
        "dm_mm",
        "d0_mm",
        "sigma_m_mm",
        "log_nw",
    ):
        row[key] = float(diag[key][0])
    return row, warnings


def gather_experiment_time_series(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    *,
    min_qr: float,
    onemom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    tasks = [
        (
            rec[0],
            rec[1],
            rec[2],
            experiment,
            {var: str(path) for var, path in rec[3].items()},
            min_qr,
            onemom_closure,
            n0_fixed,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} timesteps", flush=True)
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        _init_worker(domain_mask.mask)
        for idx, task in enumerate(tasks, 1):
            row, warnings = _process_timestep_task(task)
            rows.append(row)
            for w in warnings:
                print(w, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        ) as pool:
            for idx, (row, warnings) in enumerate(
                pool.imap_unordered(_process_timestep_task, tasks),
                1,
            ):
                rows.append(row)
                for w in warnings:
                    print(w, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)
    rows.sort(key=lambda r: np.datetime64(r["valid_time"]))
    return rows


def time_series_to_samples(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    samples: dict[str, list[float]] = {key: [] for key in OBS_PARAMETERS}
    for row in rows:
        dm = row.get("dm_mm", np.nan)
        d0 = row.get("d0_mm", np.nan)
        sigma = row.get("sigma_m_mm", np.nan)
        log_nw = row.get("log_nw", np.nan)
        lwc = row.get("lwc_g_m3", np.nan)
        nt = row.get("nt_m3", np.nan)
        if not all(np.isfinite([dm, d0, sigma, log_nw, lwc, nt])):
            continue
        if dm <= 0 or d0 <= 0 or lwc <= 0 or nt <= 0:
            continue
        samples["dm_mm"].append(dm)
        samples["d0_mm"].append(d0)
        samples["sigma_m_mm"].append(sigma)
        samples["log_nw"].append(log_nw)
        samples["lwc_g_m3"].append(lwc)
        samples["nt_m3"].append(nt)
    return {key: np.asarray(v, dtype=np.float32) for key, v in samples.items()}


@dataclass(frozen=True)
class ObsBundle:
    path_a: dict[str, np.ndarray]
    path_b: dict[str, np.ndarray]
    qc_kept: int
    qc_total: int


def read_observation_samples(
    obs_zip: Path,
    cache_dir: Path,
    *,
    min_rr_mm_h: float = DEFAULT_QC_RAIN_RATE_MIN,
    max_z_dbz: float = DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    min_run_minutes: int = DEFAULT_QC_MIN_RUN_MINUTES,
) -> ObsBundle:
    target = extract_strict_obs(obs_zip, cache_dir)
    with Dataset(target) as ds:
        nd = np.asarray(ds["number_density_drops"][:], dtype=float)
        widths = np.asarray(ds["class_size_width"][:], dtype=float)
        centers = np.asarray(ds["particle_size"][:], dtype=float)
        m3 = np.asarray(ds["moment3"][:], dtype=float)
        m4 = np.asarray(ds["moment4"][:], dtype=float)
        m5 = np.asarray(ds["moment5"][:], dtype=float)
        rr = np.asarray(ds["precip_rate"][:], dtype=float)
        z = np.asarray(ds["equivalent_radar_reflectivity"][:], dtype=float)

    qc_total = int(rr.size)
    qc_mask = apply_disdrometer_qc(
        precip_rate_mm_h=rr,
        reflectivity_dbz=z,
        min_rr_mm_h=min_rr_mm_h,
        max_z_dbz=max_z_dbz,
        min_run_minutes=min_run_minutes,
    )
    qc_kept = int(qc_mask.sum())

    # Path A — empirical DSD parameters direct from the QC'd raw N(D)
    parts_a = empirical_dsd_parameters(
        number_density=nd,
        bin_centers_mm=centers,
        bin_widths_mm=widths,
        moment3=m3,
        moment4=m4,
        moment5=m5,
    )
    valid_a = (
        qc_mask
        & np.isfinite(parts_a["dm_mm"])
        & (parts_a["dm_mm"] > 0.0)
        & np.isfinite(parts_a["d0_mm"])
        & (parts_a["d0_mm"] > 0.0)
        & np.isfinite(parts_a["lwc_g_m3"])
        & (parts_a["lwc_g_m3"] > 0.0)
        & np.isfinite(parts_a["nt_m3"])
        & (parts_a["nt_m3"] > 0.0)
        & np.isfinite(parts_a["log_nw"])
    )
    path_a = {
        "dm_mm": parts_a["dm_mm"][valid_a].astype(np.float32),
        "d0_mm": parts_a["d0_mm"][valid_a].astype(np.float32),
        "sigma_m_mm": parts_a["sigma_m_mm"][valid_a].astype(np.float32),
        "log_nw": parts_a["log_nw"][valid_a].astype(np.float32),
        "lwc_g_m3": parts_a["lwc_g_m3"][valid_a].astype(np.float32),
        "nt_m3": parts_a["nt_m3"][valid_a].astype(np.float32),
    }

    # Path B — Marshall-Palmer projection from observed (LWC, Nt)
    lwc_b = parts_a["lwc_g_m3"]
    nt_b = parts_a["nt_m3"]
    lam = mp_lambda_from_lwc_nt(lwc_b, nt_b)
    n0 = nt_b * lam
    diag_b = mp_diagnostics_from_n0_lambda(n0, lam)
    valid_b = (
        qc_mask
        & np.isfinite(diag_b["dm_mm"])
        & (diag_b["dm_mm"] > 0.0)
        & np.isfinite(diag_b["d0_mm"])
        & (diag_b["d0_mm"] > 0.0)
        & np.isfinite(diag_b["log_nw"])
        & np.isfinite(lwc_b)
        & (lwc_b > 0.0)
    )
    path_b = {
        "dm_mm": diag_b["dm_mm"][valid_b].astype(np.float32),
        "d0_mm": diag_b["d0_mm"][valid_b].astype(np.float32),
        "sigma_m_mm": diag_b["sigma_m_mm"][valid_b].astype(np.float32),
        "log_nw": diag_b["log_nw"][valid_b].astype(np.float32),
        "lwc_g_m3": diag_b["lwc_g_m3"][valid_b].astype(np.float32),
        "nt_m3": diag_b["nt_m3"][valid_b].astype(np.float32),
    }

    return ObsBundle(path_a=path_a, path_b=path_b, qc_kept=qc_kept, qc_total=qc_total)
