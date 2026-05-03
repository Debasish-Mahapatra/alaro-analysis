"""Disdrometer vs ALARO domain-mean DSD comparison (Marshall-Palmer framework).

Both observations and the model runs are reduced to the same Marshall-Palmer
DSD parameter set (D_m, D_0, log_10 N_w, sigma_m), with two parallel obs
reductions:

* **Path A (empirical obs)**: D_m = M_4/M_3, D_0 from the cumulative
  mass-50% crossing, log_10 N_w from LWC and the empirical D_m.
  No DSD shape forced on the observations.
* **Path B (MP-projected obs)**: fit Marshall-Palmer (N_0, lambda) per
  rainy QC'd minute from observed (LWC, N_t).  All parameters come from
  the analytic MP relations.

For the model side both paths use the same MP closure:

* **2-moment (G2M)** uses prognostic q_r and PNR -> (N_0, lambda).
* **1-moment (C1M, G1M)** uses fixed N_0 = 8e6 m^-4 (Marshall & Palmer
  1948), with lambda solved from q_r alone.

QC of the observations follows ``disdro.tex`` Section 1: R > 0.1 mm/h,
Z_e < 55 dBZ, and a minimum of five consecutive rainy minutes.

Outputs:

* Path A figures ``figures/disdrometer_dsd/dsd_pathA_*_<tag>.png``
* Path B figures ``figures/disdrometer_dsd/dsd_pathB_*_<tag>.png``
* Per-experiment domain-mean time series under ``processed-data/``
* Sample arrays per dataset/path in NPZ for reuse
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    DEFAULT_QC_MIN_RUN_MINUTES,
    DEFAULT_QC_RAIN_RATE_MIN,
    DEFAULT_QC_REFLECTIVITY_MAX_DBZ,
    MP_FIXED_N0_PER_M3_MM,
    apply_disdrometer_qc,
    empirical_dsd_parameters,
    mp_diagnostics_from_n0_lambda,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
    mp_lambda_from_lwc_nt,
)
from alaro_analysis.converter.pipeline import _regrid_mask_to_model
from alaro_analysis.workflows.disdrometer_comparison import (
    CACHE_DIR,
    OBS_ZIP,
    RUNS_ROOT,
    extract_strict_obs,
    lead_label,
    parse_lead_selection,
)


PROCESSED_DIR = RUNS_ROOT / "processed-data" / "disdrometer_dsd"
ANALYTICS_DIR = PROCESSED_DIR / "analytics"
FIGURE_DIR = RUNS_ROOT / "figures" / "disdrometer_dsd"
MASK_FILE = RUNS_ROOT / "mask" / "Radar_mask_latlon.nc"
NETCDF_ROOT = RUNS_ROOT / "ALARO"
DEFAULT_FIGURE_DPI = 450
NETCDF_BOTTOM_LEVEL_INDEX = 0  # in the converted netCDF, level=0 is the surface

OBS_PARAMETERS = ("dm_mm", "d0_mm", "sigma_m_mm", "log_nw", "lwc_g_m3", "nt_m3")
PANEL_ORDER = ("obs", "control", "graupel", "2mom")
PANEL_GRID_POSITIONS = {"obs": (0, 0), "control": (0, 1), "graupel": (1, 0), "2mom": (1, 1)}

PF_DAY_RE = re.compile(r"^pf(\d{8})$")
PF_FILE_RE = re.compile(r"^pfABOFABOF\+(\d{4})\.nc$")


@dataclass(frozen=True)
class DomainMask:
    mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    selected_var: str

    @property
    def n_cells(self) -> int:
        return int(self.mask.sum())


def build_domain_mask_from_netcdf(
    sample_netcdf: Path,
    mask_file: Path,
    *,
    mask_var: str | None = None,
    mask_threshold: float = 0.5,
) -> DomainMask:
    with Dataset(sample_netcdf) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    target_lat = xr.DataArray(lat, dims=("y", "x"))
    target_lon = xr.DataArray(lon, dims=("y", "x"))
    mask_array, selected = _regrid_mask_to_model(
        mask_file=mask_file,
        target_lat=target_lat,
        target_lon=target_lon,
        mask_var=mask_var,
        mask_lat_name=None,
        mask_lon_name=None,
        mask_threshold=mask_threshold,
    )
    return DomainMask(
        mask=np.asarray(mask_array, dtype=bool),
        lat=lat,
        lon=lon,
        selected_var=selected,
    )


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
            records.append(
                (np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths)
            )
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
    for key in ("lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
                "dm_mm", "d0_mm", "sigma_m_mm", "log_nw"):
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

    rho = pres_mean / (RD * temp_mean) if (
        np.isfinite(pres_mean) and np.isfinite(temp_mean) and temp_mean > 0
    ) else np.nan
    row.update(
        qr_kgkg=qr_mean,
        temperature_k=temp_mean,
        pressure_pa=pres_mean,
        rho_air_kg_m3=rho,
        valid_cells=n_cells,
    )

    if (
        not np.isfinite(qr_mean) or qr_mean < min_qr
        or not np.isfinite(rho) or rho <= 0.0
    ):
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

    for key in ("lwc_g_m3", "nt_m3", "lambda_per_mm", "n0_per_m3_mm",
                "dm_mm", "d0_mm", "sigma_m_mm", "log_nw"):
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
            rec[0], rec[1], rec[2], experiment,
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


def write_experiment_time_series_netcdf(
    path: Path,
    rows: list[dict[str, Any]],
    experiment: str,
    domain_mask: DomainMask,
    *,
    onemom_closure: str,
    n0_fixed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    valid_times = np.asarray(
        [np.datetime64(r["valid_time"]) for r in rows], dtype="datetime64[s]"
    )
    init_times = np.asarray(
        [np.datetime64(r["init_time"]) for r in rows], dtype="datetime64[s]"
    )
    leads = np.asarray([r["lead_hours"] for r in rows], dtype=np.int16)
    epoch = np.datetime64("1970-01-01T00:00:00", "s")

    def col(key: str, dtype=np.float32) -> np.ndarray:
        return np.asarray([r.get(key, np.nan) for r in rows], dtype=dtype)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n)
        v = ds.createVariable("valid_time", "f8", ("time",))
        v.units = "seconds since 1970-01-01 00:00:00"
        v.calendar = "proleptic_gregorian"
        v[:] = (valid_times - epoch) / np.timedelta64(1, "s")
        v = ds.createVariable("init_time", "f8", ("time",))
        v.units = "seconds since 1970-01-01 00:00:00"
        v.calendar = "proleptic_gregorian"
        v[:] = (init_times - epoch) / np.timedelta64(1, "s")
        v = ds.createVariable("lead_hours", "i2", ("time",))
        v.units = "h"
        v[:] = leads

        for key, units, long_name in (
            ("qr_kgkg", "kg/kg", "Domain-mean rain mixing ratio"),
            ("n_per_kg", "1/kg", "Domain-mean rain number concentration per kg"),
            ("temperature_k", "K", "Domain-mean temperature"),
            ("pressure_pa", "Pa", "Domain-mean pressure"),
            ("rho_air_kg_m3", "kg/m^3", "Domain-mean air density"),
            ("lwc_g_m3", "g/m^3", "Liquid water content (Marshall-Palmer)"),
            ("nt_m3", "1/m^3", "Total drop number concentration (Marshall-Palmer)"),
            ("lambda_per_mm", "1/mm", "Marshall-Palmer slope parameter"),
            ("n0_per_m3_mm", "1/(m^3 mm)", "Marshall-Palmer intercept parameter"),
            ("dm_mm", "mm", "Mass-weighted mean diameter Dm = 4/lambda"),
            ("d0_mm", "mm", "Median volume diameter D0 = 3.67/lambda"),
            ("sigma_m_mm", "mm", "Mass-spectrum width sigma_m = 2/lambda"),
            ("log_nw", "log10(m^-3 mm^-1)", "log10(N_w) = log10(N_0)"),
        ):
            var = ds.createVariable(key, "f4", ("time",), fill_value=np.float32(np.nan))
            var.units = units
            var.long_name = long_name
            var[:] = col(key)
        cells = ds.createVariable("valid_cells", "i4", ("time",))
        cells.long_name = "Number of finite masked cells used for the domain average"
        cells[:] = col("valid_cells", dtype=np.int32)

        ds.createDimension("y", domain_mask.lat.shape[0])
        ds.createDimension("x", domain_mask.lat.shape[1])
        latv = ds.createVariable("lat", "f4", ("y", "x"), zlib=True, complevel=1)
        latv.units = "degrees_north"
        latv[:] = domain_mask.lat.astype(np.float32)
        lonv = ds.createVariable("lon", "f4", ("y", "x"), zlib=True, complevel=1)
        lonv.units = "degrees_east"
        lonv[:] = domain_mask.lon.astype(np.float32)
        maskv = ds.createVariable("radar_mask", "i1", ("y", "x"), zlib=True, complevel=1)
        maskv.long_name = "Radar mask remapped to the masked-netcdf grid"
        maskv[:] = domain_mask.mask.astype(np.int8)

        ds.experiment = experiment
        if experiment == "2mom":
            ds.dsd_closure = "marshall_palmer_2mom"
        elif onemom_closure == "abel_boutle":
            ds.dsd_closure = "abel_boutle_2012"
        else:
            ds.dsd_closure = "marshall_palmer_fixed_n0"
        ds.n0_fixed_per_m3_mm = float(n0_fixed)
        ds.description = (
            "Domain-mean ALARO bottom-layer rain DSD parameters under a "
            "Marshall-Palmer closure, derived from the QC'd masked netCDF outputs."
        )


# ---------------------------------------------------------------------------
# Observation reduction
# ---------------------------------------------------------------------------


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
        & np.isfinite(parts_a["dm_mm"]) & (parts_a["dm_mm"] > 0.0)
        & np.isfinite(parts_a["d0_mm"]) & (parts_a["d0_mm"] > 0.0)
        & np.isfinite(parts_a["lwc_g_m3"]) & (parts_a["lwc_g_m3"] > 0.0)
        & np.isfinite(parts_a["nt_m3"]) & (parts_a["nt_m3"] > 0.0)
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
        & np.isfinite(diag_b["dm_mm"]) & (diag_b["dm_mm"] > 0.0)
        & np.isfinite(diag_b["d0_mm"]) & (diag_b["d0_mm"] > 0.0)
        & np.isfinite(diag_b["log_nw"])
        & np.isfinite(lwc_b) & (lwc_b > 0.0)
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


# ---------------------------------------------------------------------------
# Plotting helpers (2x2 with obs-contour overlay)
# ---------------------------------------------------------------------------


def _format_count(n: int) -> str:
    return f"n={n:,}"


def _compute_obs_contour(
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    sigma_bins: float = 1.5,
    contour_level_pct: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]] | None:
    """Smoothed obs frequency field + a single inclusive contour level."""
    if obs_x.size == 0:
        return None
    h, _, _ = np.histogram2d(obs_x, obs_y, bins=[x_edges, y_edges])
    if h.sum() == 0:
        return None
    freq_pct = 100.0 * h / h.sum()
    smoothed = gaussian_filter(freq_pct, sigma=sigma_bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, smoothed.T, [contour_level_pct]


def plot_2x2_with_marginals(
    out_path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    x_label: str,
    *,
    title: str = "",
    bins: int = 60,
) -> None:
    """2x2 joint density panels, each with bottom-X and left-Y marginals.

    The marginals overlay all four datasets so they read as constant
    references.  The panel's own dataset is drawn thicker on top.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            x_los.append(float(np.nanpercentile(sx, 0.5)))
            x_his.append(float(np.nanpercentile(sx, 99.5)))
        if sy.size:
            y_los.append(float(np.nanpercentile(sy, 0.5)))
            y_his.append(float(np.nanpercentile(sy, 99.5)))
    if not x_los or not y_los:
        raise RuntimeError("No samples available to plot 2x2+marginals")
    x_lo = max(0.2, min(x_los))
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    obs_x = samples["obs"][x_field]
    obs_y = samples["obs"]["log_nw"]
    contour = _compute_obs_contour(obs_x, obs_y, x_edges, y_edges)

    color_for_name = {
        "obs": "#111111",
        "control": "#d62728",
        "graupel": "#1f77b4",
        "2mom": "#2ca02c",
    }
    label_for_name = {
        "obs": "Obs",
        "control": EXPERIMENT_LABELS.get("control", "C1M"),
        "graupel": EXPERIMENT_LABELS.get("graupel", "G1M"),
        "2mom": EXPERIMENT_LABELS.get("2mom", "G2M"),
    }

    x_pdf: dict[str, np.ndarray] = {}
    y_pdf: dict[str, np.ndarray] = {}
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            xh, _ = np.histogram(sx, bins=x_edges, density=True)
            x_pdf[name] = xh
        else:
            x_pdf[name] = np.zeros(bins)
        if sy.size:
            yh, _ = np.histogram(sy, bins=y_edges, density=True)
            y_pdf[name] = yh
        else:
            y_pdf[name] = np.zeros(bins)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(13.5, 12.0))
    outer = fig.add_gridspec(
        2, 2, hspace=0.20, wspace=0.18, left=0.06, right=0.94, top=0.93, bottom=0.05
    )

    last_im = None
    for name in PANEL_ORDER:
        i, j = PANEL_GRID_POSITIONS[name]
        inner = outer[i, j].subgridspec(
            2, 2,
            width_ratios=[1, 4],
            height_ratios=[4, 1],
            hspace=0.04, wspace=0.04,
        )
        ax_y = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[0, 1], sharey=ax_y)
        ax_x = fig.add_subplot(inner[1, 1], sharex=ax_main)

        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        n = int(sx.size)
        if n == 0:
            ax_main.set_title(f"{label_for_name[name]} (n=0)")
            continue
        h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
        h = h / max(1.0, h.sum())
        with np.errstate(divide="ignore"):
            shown = np.log10(np.where(h > 0, h, np.nan))
        ax_main.set_facecolor("white")
        im = ax_main.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, shading="auto")
        last_im = im
        if contour is not None:
            xc, yc, field, levels = contour
            try:
                ax_main.contour(xc, yc, field, levels=levels, colors="black", linewidths=1.2)
            except ValueError:
                pass
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(y_lo, y_hi)
        ax_main.set_title(f"{label_for_name[name]} (n={n:,})")
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        plt.setp(ax_main.get_yticklabels(), visible=False)

        for ds_name in PANEL_ORDER:
            color = color_for_name[ds_name]
            lw = 2.4 if ds_name == name else 1.0
            alpha = 1.0 if ds_name == name else 0.7
            zorder = 5 if ds_name == name else 2
            ax_x.plot(
                x_centers, x_pdf[ds_name],
                color=color, linewidth=lw, alpha=alpha, zorder=zorder,
            )
            ax_y.plot(
                y_pdf[ds_name], y_centers,
                color=color, linewidth=lw, alpha=alpha, zorder=zorder,
            )
        ax_x.set_xlim(x_lo, x_hi)
        ax_x.set_ylim(bottom=0.0)
        ax_x.grid(True, alpha=0.3)
        ax_x.set_xlabel(x_label)
        ax_x.set_ylabel("PDF")

        ax_y.set_ylim(y_lo, y_hi)
        ax_y.set_xlim(left=0.0)
        ax_y.invert_xaxis()
        ax_y.grid(True, alpha=0.3)
        ax_y.set_ylabel("log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
        ax_y.set_xlabel("PDF")

    legend_handles = [
        plt.Line2D([], [], color=color_for_name[n], lw=2.0, label=label_for_name[n])
        for n in PANEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=False,
    )
    if last_im is not None:
        cbar_ax = fig.add_axes([0.96, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("log$_{10}$ frequency")
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if title:
        fig.suptitle(title, y=0.965)
    fig.savefig(out_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_2x2_joint_density(
    out_path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    x_label: str,
    *,
    title: str,
    bins: int = 60,
) -> None:
    """Plot a 2x2 joint density of (x_field, log_nw) for obs + 3 model runs.

    ``samples`` must hold keys ``obs``, ``control``, ``graupel``, ``2mom``.
    The QC'd observed core contour (0.5 % smoothed frequency) is overlaid
    on each of the three model panels.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Take per-dataset percentiles and span the union, so 1-mom's collapsed
    # log Nw line at log10(N0) doesn't fall outside the obs-driven range.
    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            x_los.append(float(np.nanpercentile(sx, 0.5)))
            x_his.append(float(np.nanpercentile(sx, 99.5)))
        if sy.size:
            y_los.append(float(np.nanpercentile(sy, 0.5)))
            y_his.append(float(np.nanpercentile(sy, 99.5)))
    if not x_los or not y_los:
        raise RuntimeError("No samples available to plot 2x2 joint density")

    x_lo = max(0.2, min(x_los))
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)

    obs_x = samples["obs"][x_field]
    obs_y = samples["obs"]["log_nw"]
    contour = _compute_obs_contour(obs_x, obs_y, x_edges, y_edges)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.0, 9.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    for name in PANEL_ORDER:
        i, j = PANEL_GRID_POSITIONS[name]
        ax = axes[i, j]
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        label = "Obs" if name == "obs" else EXPERIMENT_LABELS.get(name, name)
        if sx.size == 0:
            ax.set_title(f"{label} (n=0)")
            continue
        h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
        h = h / max(1.0, h.sum())
        with np.errstate(divide="ignore"):
            shown = np.log10(np.where(h > 0, h, np.nan))
        ax.set_facecolor("white")
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("white")
        im = ax.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, shading="auto")
        ax.set_title(f"{label} ({_format_count(sx.size)})")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        last_im = im
        if contour is not None:
            xc, yc, field, levels = contour
            try:
                ax.contour(
                    xc, yc, field,
                    levels=levels,
                    colors="black",
                    linewidths=1.4,
                )
            except ValueError:
                pass
    for ax in axes[1, :]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
    fig.suptitle(title)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, location="right")
        cbar.set_label("log$_{10}$ frequency")
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.savefig(out_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_samples_npz(path: Path, datasets: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, samples in datasets.items():
        for key, values in samples.items():
            payload[f"{name}__{key}"] = values
    np.savez_compressed(path, **payload)


def run_dsd(args: argparse.Namespace) -> dict[str, Path]:
    experiments = tuple(args.experiments)
    leads = parse_lead_selection(args.lead)
    output_tag = f"{lead_label(leads)}"

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.analytics_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observation_samples(
        args.obs_zip, args.cache_dir,
        min_rr_mm_h=args.obs_min_precip,
        max_z_dbz=args.obs_max_dbz,
        min_run_minutes=args.obs_min_run_minutes,
    )
    print(
        f"obs QC: kept {obs.qc_kept:,} / {obs.qc_total:,} minutes "
        f"(R>{args.obs_min_precip} mm/h, Ze<{args.obs_max_dbz} dBZ, "
        f">={args.obs_min_run_minutes}-min runs)",
        flush=True,
    )
    print(
        f"  Path A (empirical) samples: {obs.path_a['dm_mm'].size:,}; "
        f"Path B (MP-projected) samples: {obs.path_b['dm_mm'].size:,}",
        flush=True,
    )

    sample_records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] | None = None
    for exp in experiments:
        recs = discover_netcdf_records(exp, leads, args.netcdf_root, args.max_days)
        if recs:
            sample_records = recs
            break
    if not sample_records:
        raise RuntimeError("No netCDF records found for any experiment")

    sample_path = sample_records[0][3]["RAIN"]
    print(f"building radar mask from {args.mask_file} (sample {sample_path})", flush=True)
    domain_mask = build_domain_mask_from_netcdf(
        sample_path,
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
    )
    print(f"  mask cells kept: {domain_mask.n_cells} / {domain_mask.mask.size}", flush=True)

    model_samples: dict[str, dict[str, np.ndarray]] = {}
    for exp in experiments:
        print(f"gathering domain-mean MP DSD for {exp}", flush=True)
        records = discover_netcdf_records(exp, leads, args.netcdf_root, args.max_days)
        rows = gather_experiment_time_series(
            exp, records, domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            n0_fixed=args.n0_fixed,
            workers=max(1, int(args.workers)),
            progress_every=args.progress_every,
            tasks_per_child=args.tasks_per_child,
        )
        nc_path = args.processed_dir / f"disdrometer_dsd_mp_{exp}_{output_tag}.nc"
        if rows:
            write_experiment_time_series_netcdf(
                nc_path, rows, exp, domain_mask,
                onemom_closure=args.onemom_closure,
                n0_fixed=args.n0_fixed,
            )
            print(f"  [{exp}] wrote {nc_path}", flush=True)
        samples = time_series_to_samples(rows)
        model_samples[exp] = samples
        print(f"  [{exp}] kept {samples['dm_mm'].size:,} rainy timesteps", flush=True)

    samples_a = {"obs": obs.path_a, **model_samples}
    samples_b = {"obs": obs.path_b, **model_samples}
    npz_path = args.processed_dir / f"disdrometer_dsd_samples_{output_tag}.npz"
    save_samples_npz(
        npz_path,
        {
            "obs_pathA": obs.path_a,
            "obs_pathB": obs.path_b,
            **{exp: model_samples[exp] for exp in experiments},
        },
    )
    print(f"sample arrays saved to {npz_path}", flush=True)

    fig_paths: dict[str, Path] = {}
    if not args.no_plots:
        for path_label, samples in (("pathA", samples_a), ("pathB", samples_b)):
            obs_kind = "empirical" if path_label == "pathA" else "MP-projected"
            for x_field, x_label, suffix in (
                ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
                ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
            ):
                title = (
                    f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field=='d0_mm' else 'D$_m$'}, "
                    f"obs {obs_kind}"
                )
                out = args.figure_dir / f"dsd_{path_label}_{suffix}_{output_tag}.png"
                plot_2x2_joint_density(
                    out_path=out,
                    samples=samples,
                    x_field=x_field,
                    x_label=x_label,
                    title=title,
                )
                fig_paths[f"{path_label}_{suffix}"] = out
                print(f"  rendered {out}", flush=True)

    outputs = {"samples_npz": npz_path, **fig_paths}
    for label, path in outputs.items():
        if path.exists():
            print(f"{label}: {path}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Disdrometer vs ALARO domain-mean DSD comparison under a "
            "Marshall-Palmer closure (model side) and two parallel obs "
            "reductions (empirical and MP-projected)."
        )
    )
    parser.add_argument("--lead", default="all")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--min-qr",
        type=float,
        default=1.0e-7,
        help="Minimum domain-mean rain mixing ratio (kg/kg) to keep a timestep.",
    )
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
        help="DSD closure for the 1-moment runs: 'abel_boutle' (Abel & Boutle 2012, "
             "the formula in arpifs/adiab/gpprs0d.F90) or 'fixed_n0' (classical Marshall-Palmer).",
    )
    parser.add_argument(
        "--n0-fixed",
        type=float,
        default=MP_FIXED_N0_PER_M3_MM,
        help="Marshall-Palmer N0 (1/(m^3 mm)) used when --onemom-closure=fixed_n0 "
             "(default 8000, ie 8e6 m^-4).",
    )
    parser.add_argument("--obs-min-precip", type=float, default=DEFAULT_QC_RAIN_RATE_MIN)
    parser.add_argument("--obs-max-dbz", type=float, default=DEFAULT_QC_REFLECTIVITY_MAX_DBZ)
    parser.add_argument("--obs-min-run-minutes", type=int, default=DEFAULT_QC_MIN_RUN_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--analytics-dir", type=Path, default=ANALYTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dsd(args)


if __name__ == "__main__":
    main()
