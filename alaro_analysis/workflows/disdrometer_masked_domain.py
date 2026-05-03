from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from multiprocessing import get_context
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS, EXPERIMENTS, RD
from alaro_analysis.converter.pipeline import _regrid_mask_to_model
from alaro_analysis.workflows.disdrometer_comparison import (
    ANALYTICS_DIR,
    CACHE_DIR,
    DEFAULT_LAT,
    DEFAULT_LON,
    FIGURE_DIR,
    MODEL_ROOT,
    OBS_ZIP,
    PROCESSED_DIR,
    RUNS_ROOT,
    BOTTOM_LEVEL,
    ObservationWindow,
    average_observation_window,
    discover_model_records,
    lead_label,
    parse_lead_selection,
    rain_mean_volume_diameter_mm,
    read_observations,
    save_summary_csv,
    single_moment_rain_number_per_kg,
    summarize_experiment,
    title_lead_text,
)


MASK_FILE = RUNS_ROOT / "mask" / "Radar_mask_latlon.nc"
MASKED_NETCDF_DIR = PROCESSED_DIR / "masked-lowest-layer"
PF_KEY_SEP = "->"

_WORKER_CROP: tuple[int, int, int, int] | None = None
_WORKER_MASK: np.ndarray | None = None


@dataclass(frozen=True)
class DomainMask:
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    mask: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    selected_var: str

    @property
    def crop(self) -> tuple[int, int, int, int]:
        return (self.y_start, self.y_stop, self.x_start, self.x_stop)


def _init_masked_worker(mask: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    import epygram

    epygram.init_env()
    global _WORKER_CROP, _WORKER_MASK
    _WORKER_CROP = crop
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_field_2d(resource: Any, fid: str) -> np.ndarray:
    field = resource.readfield(fid)
    if getattr(field, "spectral", False):
        field.sp2gp()
    return np.asarray(field.getdata(), dtype=np.float64)


def _read_lonlat_from_fa(path: Path, fid: str) -> tuple[np.ndarray, np.ndarray]:
    import epygram

    epygram.init_env()
    resource = epygram.formats.resource(str(path), "r")
    try:
        field = resource.readfield(fid)
        if getattr(field, "spectral", False):
            field.sp2gp()
        lon, lat = field.geometry.get_lonlat_grid()
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    finally:
        resource.close()


def build_domain_mask(
    sample_fa: Path,
    mask_file: Path,
    mask_var: str | None = None,
    mask_threshold: float = 0.5,
    reference_fid: str = f"S{BOTTOM_LEVEL:03d}RAIN",
) -> DomainMask:
    lon, lat = _read_lonlat_from_fa(sample_fa, reference_fid)
    target_lon = xr.DataArray(lon, dims=("y", "x"))
    target_lat = xr.DataArray(lat, dims=("y", "x"))
    full_mask, selected_var = _regrid_mask_to_model(
        mask_file=mask_file,
        target_lat=target_lat,
        target_lon=target_lon,
        mask_var=mask_var,
        mask_lat_name=None,
        mask_lon_name=None,
        mask_threshold=mask_threshold,
    )
    ys, xs = np.where(full_mask)
    if ys.size == 0:
        raise ValueError("Radar mask selects no model grid cells")
    y_start, y_stop = int(ys.min()), int(ys.max()) + 1
    x_start, x_stop = int(xs.min()), int(xs.max()) + 1
    crop_mask = full_mask[y_start:y_stop, x_start:x_stop]
    return DomainMask(
        y_start=y_start,
        y_stop=y_stop,
        x_start=x_start,
        x_stop=x_stop,
        mask=crop_mask,
        lon=lon[y_start:y_stop, x_start:x_stop],
        lat=lat[y_start:y_stop, x_start:x_stop],
        selected_var=selected_var,
    )


def _crop_and_mask(values: np.ndarray) -> np.ndarray:
    if _WORKER_CROP is None or _WORKER_MASK is None:
        raise RuntimeError("Worker mask was not initialized")
    y_start, y_stop, x_start, x_stop = _WORKER_CROP
    cropped = np.asarray(values[y_start:y_stop, x_start:x_stop], dtype=np.float64)
    return np.where(_WORKER_MASK, cropped, np.nan)


def _finite_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _read_experiment_domain(
    fa_path: Path,
    experiment: str,
    level: int,
) -> tuple[dict[str, Any], np.ndarray]:
    import epygram

    rain_fid = f"S{level:03d}RAIN"
    temp_fid = f"S{level:03d}TEMPERATURE"
    pressure_fid = f"S{level:03d}PRESSURE"
    pnr_fid = f"S{level:03d}PNR"
    dmean_fid = f"S{level:03d}DMEANR"

    resource = epygram.formats.resource(str(fa_path), "r")
    try:
        rain = np.maximum(_crop_and_mask(_read_field_2d(resource, rain_fid)), 0.0)
        temp = _crop_and_mask(_read_field_2d(resource, temp_fid))
        pressure = _crop_and_mask(_read_field_2d(resource, pressure_fid))
        rho_air = pressure / (RD * temp)

        method = "derived_equilibrium_diameter"
        if experiment == "2mom":
            try:
                rain_number_kg = np.maximum(_crop_and_mask(_read_field_2d(resource, pnr_fid)), 0.0)
                method = "prognostic_pnr"
            except Exception:
                rain_number_kg = single_moment_rain_number_per_kg(rain)
        else:
            rain_number_kg = single_moment_rain_number_per_kg(rain)

        rain_number_m3 = rain_number_kg * rho_air

        try:
            dmean_mm = _crop_and_mask(_read_field_2d(resource, dmean_fid)) * 1000.0
            dmean_mm = np.where(np.isfinite(dmean_mm) & (dmean_mm > 0.0), dmean_mm, np.nan)
        except Exception:
            dmean_mm = rain_mean_volume_diameter_mm(rain, rain_number_kg)
            if experiment != "2mom":
                dmean_mm = np.where((rain > 0.0) & np.isfinite(dmean_mm), dmean_mm, np.nan)

        row = {
            f"{experiment}_rain_number_m3": _finite_mean(rain_number_m3),
            f"{experiment}_rain_number_kg": _finite_mean(rain_number_kg),
            f"{experiment}_rain_mixing_ratio_kgkg": _finite_mean(rain),
            f"{experiment}_dmean_mm": _finite_mean(dmean_mm),
            f"{experiment}_rho_air_kg_m3": _finite_mean(rho_air),
            f"{experiment}_temperature_k": _finite_mean(temp),
            f"{experiment}_pressure_pa": _finite_mean(pressure),
            f"{experiment}_method": method,
            f"{experiment}_valid_grid_points": int(np.isfinite(rain_number_m3).sum()),
            f"{experiment}_rain_positive_grid_points": int(
                (np.isfinite(rain) & (rain > 0.0)).sum()
            ),
        }
        return row, np.asarray(rain_number_m3, dtype=np.float32)
    finally:
        resource.close()


def _read_masked_domain_task(
    task: tuple[str, str, int, dict[str, str], tuple[str, ...], int, bool],
) -> tuple[str, dict[str, Any], dict[str, np.ndarray], list[str]]:
    record_key, valid_time, lead, paths, experiments, level, return_grids = task
    row: dict[str, Any] = {}
    grids: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for exp in experiments:
        try:
            exp_row, grid = _read_experiment_domain(Path(paths[exp]), exp, level)
            row.update(exp_row)
            if return_grids:
                grids[exp] = grid
        except Exception as exc:
            warnings.append(
                f"WARNING {exp} {valid_time} lead +{lead:04d}: "
                f"failed to read masked domain: {exc}"
            )
            row.update(
                {
                    f"{exp}_rain_number_m3": np.nan,
                    f"{exp}_rain_number_kg": np.nan,
                    f"{exp}_rain_mixing_ratio_kgkg": np.nan,
                    f"{exp}_dmean_mm": np.nan,
                    f"{exp}_rho_air_kg_m3": np.nan,
                    f"{exp}_temperature_k": np.nan,
                    f"{exp}_pressure_pa": np.nan,
                    f"{exp}_method": "failed",
                    f"{exp}_valid_grid_points": 0,
                    f"{exp}_rain_positive_grid_points": 0,
                }
            )
            if return_grids and _WORKER_MASK is not None:
                grids[exp] = np.full(_WORKER_MASK.shape, np.nan, dtype=np.float32)
    return record_key, row, grids, warnings


def datetime64_key(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")


def datetime64_hours_since_epoch(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="datetime64[s]")
    return (values - np.datetime64("1970-01-01T00:00:00", "s")) / np.timedelta64(1, "h")


def init_grid_netcdf(
    path: Path,
    records: list[tuple[str, str, int]],
    domain_mask: DomainMask,
    experiments: tuple[str, ...],
    obs_by_valid_time: dict[str, ObservationWindow],
    level: int,
) -> Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset(path, "w", format="NETCDF4")
    n_time = len(records)
    ny, nx = domain_mask.mask.shape
    ds.createDimension("time", n_time)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    valid = np.asarray([np.datetime64(record[0]) for record in records], dtype="datetime64[s]")
    init = np.asarray([np.datetime64(record[1]) for record in records], dtype="datetime64[s]")
    leads = np.asarray([record[2] for record in records], dtype=np.int16)
    obs = np.asarray([obs_by_valid_time[record[0]].rain_number_m3 for record in records], dtype=np.float32)
    obs_precip = np.asarray([obs_by_valid_time[record[0]].precip_rate_mm_h for record in records], dtype=np.float32)

    tvar = ds.createVariable("valid_time", "f8", ("time",))
    tvar.units = "hours since 1970-01-01 00:00:00"
    tvar.calendar = "proleptic_gregorian"
    tvar[:] = datetime64_hours_since_epoch(valid)

    ivar = ds.createVariable("init_time", "f8", ("time",))
    ivar.units = "hours since 1970-01-01 00:00:00"
    ivar.calendar = "proleptic_gregorian"
    ivar[:] = datetime64_hours_since_epoch(init)

    lvar = ds.createVariable("lead_hours", "i2", ("time",))
    lvar.units = "h"
    lvar[:] = leads

    lat_var = ds.createVariable("lat", "f4", ("y", "x"), zlib=True, complevel=1)
    lat_var.units = "degrees_north"
    lat_var[:] = domain_mask.lat.astype(np.float32)

    lon_var = ds.createVariable("lon", "f4", ("y", "x"), zlib=True, complevel=1)
    lon_var.units = "degrees_east"
    lon_var[:] = domain_mask.lon.astype(np.float32)

    mask_var = ds.createVariable("radar_mask", "i1", ("y", "x"), zlib=True, complevel=1)
    mask_var.long_name = "Radar mask remapped to the ALARO model grid"
    mask_var[:] = domain_mask.mask.astype(np.int8)

    ovar = ds.createVariable("obs_rain_number_m3", "f4", ("time",), fill_value=np.nan)
    ovar.units = "m-3"
    ovar[:] = obs

    pvar = ds.createVariable("obs_precip_rate_mm_h", "f4", ("time",), fill_value=np.nan)
    pvar.units = "mm h-1"
    pvar[:] = obs_precip

    for exp in experiments:
        grid = ds.createVariable(
            f"{exp}_rain_number_m3",
            "f4",
            ("time", "y", "x"),
            fill_value=np.nan,
            zlib=True,
            complevel=1,
            chunksizes=(1, ny, nx),
        )
        grid.units = "m-3"
        grid.long_name = (
            f"{EXPERIMENT_LABELS.get(exp, exp)} lowest-layer rain number "
            "concentration over the masked radar domain"
        )
        mean = ds.createVariable(f"{exp}_rain_number_m3_mean", "f4", ("time",), fill_value=np.nan)
        mean.units = "m-3"
        mean.long_name = "Masked-domain mean of lowest-layer rain number concentration"

    ds.model_level_fid = f"S{level:03d}"
    ds.mask_file = str(MASK_FILE)
    ds.mask_variable = domain_mask.selected_var
    ds.description = (
        "Lowest ALARO model layer extracted from raw FA files with epygram, "
        "cropped to the remapped radar mask, and paired with ARM GOAmazon "
        "Manacapuru disdrometer observations."
    )
    return ds


def save_timeseries_csv(path: Path, rows: list[dict[str, Any]], experiments: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "valid_time_utc",
        "init_time_utc",
        "lead_hours",
        "obs_rain_number_m3",
        "obs_precip_rate_mm_h",
        "obs_median_volume_diameter_mm",
        "obs_sample_count",
    ]
    exp_fields: list[str] = []
    for exp in experiments:
        exp_fields.extend(
            [
                f"{exp}_rain_number_m3",
                f"{exp}_rain_number_kg",
                f"{exp}_rain_mixing_ratio_kgkg",
                f"{exp}_dmean_mm",
                f"{exp}_rho_air_kg_m3",
                f"{exp}_temperature_k",
                f"{exp}_pressure_pa",
                f"{exp}_method",
                f"{exp}_valid_grid_points",
                f"{exp}_rain_positive_grid_points",
            ]
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields + exp_fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_datetime(values: np.ndarray) -> list[Any]:
    return [v.astype("datetime64[s]").astype(object) for v in values]


def plot_timeseries(
    out: Path,
    valid_times: np.ndarray,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    dates = _as_datetime(valid_times)
    ax.plot(dates, np.where(obs > 0.0, obs, np.nan), color="#111111", lw=1.6, label="Obs")
    for exp, values in model.items():
        ax.plot(
            dates,
            np.where(values > 0.0, values, np.nan),
            color=EXPERIMENT_COLORS[exp],
            lw=1.2,
            alpha=0.85,
            label=EXPERIMENT_LABELS[exp],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_title(f"Disdrometer vs masked-domain lowest-layer rain number, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="upper right", frameon=False, ncol=4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_scatter(
    out: Path,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(model), figsize=(14, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    positives = [obs[obs > 0.0], *[v[v > 0.0] for v in model.values()]]
    combined = np.concatenate([v[np.isfinite(v)] for v in positives if v.size])
    if combined.size:
        lo = max(1.0e-3, float(np.nanpercentile(combined, 1)) * 0.5)
        hi = float(np.nanpercentile(combined, 99)) * 2.0
    else:
        lo, hi = 1.0e-2, 1.0e4

    for ax, (exp, values) in zip(axes, model.items(), strict=True):
        mask = np.isfinite(obs) & np.isfinite(values) & (obs > 0.0) & (values > 0.0)
        ax.scatter(
            obs[mask],
            values[mask],
            s=14,
            alpha=0.55,
            color=EXPERIMENT_COLORS[exp],
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(EXPERIMENT_LABELS[exp])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Observed (m$^{-3}$)")
    axes[0].set_ylabel("Masked-domain model mean (m$^{-3}$)")
    fig.suptitle(f"Observed vs masked-domain lowest model layer, {lead_text}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_distribution(
    out: Path,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    obs_valid = np.isfinite(obs) & (obs > 0.0)
    series = {"Obs": obs[obs_valid]}
    for exp, values in model.items():
        values = np.asarray(values)
        matched = obs_valid & np.isfinite(values) & (values > 0.0)
        series[EXPERIMENT_LABELS[exp]] = values[matched]
    finite = np.concatenate(
        [np.asarray(v)[np.isfinite(v) & (np.asarray(v) > 0.0)] for v in series.values()]
    )
    if finite.size:
        bins = np.logspace(
            math.log10(max(1.0e-3, float(np.nanpercentile(finite, 1)) * 0.5)),
            math.log10(float(np.nanpercentile(finite, 99)) * 2.0),
            34,
        )
    else:
        bins = np.logspace(-2, 5, 34)
    for label, values in series.items():
        values = np.asarray(values)
        clean = values[np.isfinite(values) & (values > 0.0)]
        if clean.size == 0:
            continue
        if label == "Obs":
            color = "#111111"
        else:
            exp = next(exp for exp in model if EXPERIMENT_LABELS[exp] == label)
            color = EXPERIMENT_COLORS[exp]
        ax.hist(clean, bins=bins, histtype="step", density=True, lw=1.7, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_ylabel("Density")
    ax.set_title(f"Matched observed and masked-domain distributions, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> dict[str, Path]:
    experiments = tuple(args.experiments)
    workers = max(1, int(args.workers))
    leads = parse_lead_selection(args.lead)
    output_tag = f"masked_domain_{lead_label(leads)}"
    lead_text = title_lead_text(leads)

    print(f"reading observations from {args.obs_zip}", flush=True)
    obs = read_observations(
        args.obs_zip,
        args.cache_dir,
        min_precip_rate_mm_h=args.obs_min_precip,
    )
    print(
        f"obs samples: {len(obs.times):,}; rainy/QC samples: "
        f"{int(np.isfinite(obs.rain_number_m3).sum()):,}",
        flush=True,
    )

    model_records = {
        exp: discover_model_records(exp, leads, args.model_root, args.max_days)
        for exp in experiments
    }
    common_records = sorted(set.intersection(*(set(v) for v in model_records.values())))
    if not common_records:
        raise RuntimeError("No common model files found for requested experiments/leads")

    sample_record = common_records[0]
    sample_fa = model_records[experiments[0]][sample_record]
    print(f"building mask from {args.mask_file}", flush=True)
    domain_mask = build_domain_mask(
        sample_fa,
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
        reference_fid=f"S{args.level:03d}RAIN",
    )
    print(
        "masked crop: "
        f"y={domain_mask.y_start}:{domain_mask.y_stop}, "
        f"x={domain_mask.x_start}:{domain_mask.x_stop}, "
        f"kept cells={int(domain_mask.mask.sum())}",
        flush=True,
    )

    unique_valid_times = sorted({record[0] for record in common_records})
    obs_windows: dict[str, ObservationWindow] = {}
    for valid_time_key in unique_valid_times:
        obs_windows[valid_time_key] = average_observation_window(
            obs,
            np.datetime64(valid_time_key),
            args.obs_window_minutes,
        )

    print(f"lead selection: {lead_text}", flush=True)
    print(f"common model records: {len(common_records):,}", flush=True)
    print(f"unique valid times: {len(unique_valid_times):,}", flush=True)
    print(f"workers: {workers}", flush=True)

    tasks = [
        (
            f"{init_time_key}+{lead:04d}{PF_KEY_SEP}{valid_time_key}",
            valid_time_key,
            lead,
            {exp: str(model_records[exp][record]) for exp in experiments},
            experiments,
            args.level,
            not args.no_grid_netcdf,
        )
        for record in common_records
        for valid_time_key, init_time_key, lead in [record]
    ]
    record_index = {
        f"{init_time_key}+{lead:04d}{PF_KEY_SEP}{valid_time_key}": idx
        for idx, (valid_time_key, init_time_key, lead) in enumerate(common_records)
    }

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.analytics_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.masked_netcdf_dir.mkdir(parents=True, exist_ok=True)

    grid_nc_path = args.masked_netcdf_dir / f"lowest_layer_rain_number_{output_tag}.nc"
    grid_ds: Dataset | None = None
    if not args.no_grid_netcdf:
        grid_ds = init_grid_netcdf(
            grid_nc_path,
            common_records,
            domain_mask,
            experiments,
            obs_windows,
            args.level,
        )

    model_rows: dict[str, dict[str, Any]] = {}
    try:
        if workers == 1:
            _init_masked_worker(domain_mask.mask, domain_mask.crop)
            for idx, task in enumerate(tasks, 1):
                record_key, model_row, grids, warnings = _read_masked_domain_task(task)
                for warning in warnings:
                    print(warning, flush=True)
                model_rows[record_key] = model_row
                if grid_ds is not None:
                    nc_idx = record_index[record_key]
                    for exp, grid in grids.items():
                        grid_ds.variables[f"{exp}_rain_number_m3"][nc_idx, :, :] = grid
                        grid_ds.variables[f"{exp}_rain_number_m3_mean"][nc_idx] = model_row[
                            f"{exp}_rain_number_m3"
                        ]
                if idx % args.progress_every == 0 or idx == len(tasks):
                    print(f"processed {idx}/{len(tasks)} masked domains", flush=True)
        else:
            with get_context("fork").Pool(
                processes=workers,
                initializer=_init_masked_worker,
                initargs=(domain_mask.mask, domain_mask.crop),
                maxtasksperchild=args.tasks_per_child,
            ) as pool:
                for idx, (record_key, model_row, grids, warnings) in enumerate(
                    pool.imap_unordered(_read_masked_domain_task, tasks),
                    1,
                ):
                    for warning in warnings:
                        print(warning, flush=True)
                    model_rows[record_key] = model_row
                    if grid_ds is not None:
                        nc_idx = record_index[record_key]
                        for exp, grid in grids.items():
                            grid_ds.variables[f"{exp}_rain_number_m3"][nc_idx, :, :] = grid
                            grid_ds.variables[f"{exp}_rain_number_m3_mean"][nc_idx] = model_row[
                                f"{exp}_rain_number_m3"
                            ]
                    if idx % args.progress_every == 0 or idx == len(tasks):
                        print(f"processed {idx}/{len(tasks)} masked domains", flush=True)
    finally:
        if grid_ds is not None:
            grid_ds.close()

    rows: list[dict[str, Any]] = []
    obs_values: list[float] = []
    valid_times_out: list[np.datetime64] = []
    init_times_out: list[np.datetime64] = []
    leads_out: list[int] = []
    model_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}
    model_qr_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}

    for valid_time_key, init_time_key, lead in common_records:
        record_key = f"{init_time_key}+{lead:04d}{PF_KEY_SEP}{valid_time_key}"
        window = obs_windows[valid_time_key]
        row: dict[str, Any] = {
            "valid_time_utc": valid_time_key.replace("T", " "),
            "init_time_utc": init_time_key.replace("T", " "),
            "lead_hours": lead,
            "obs_rain_number_m3": window.rain_number_m3,
            "obs_precip_rate_mm_h": window.precip_rate_mm_h,
            "obs_median_volume_diameter_mm": window.median_volume_diameter_mm,
            "obs_sample_count": window.sample_count,
        }
        row.update(model_rows[record_key])
        rows.append(row)
        obs_values.append(window.rain_number_m3)
        valid_times_out.append(np.datetime64(valid_time_key))
        init_times_out.append(np.datetime64(init_time_key))
        leads_out.append(lead)
        for exp in experiments:
            model_arrays[exp].append(row[f"{exp}_rain_number_m3"])
            model_qr_arrays[exp].append(row[f"{exp}_rain_mixing_ratio_kgkg"])

    valid_times_arr = np.asarray(valid_times_out, dtype="datetime64[s]")
    init_times_arr = np.asarray(init_times_out, dtype="datetime64[s]")
    leads_arr = np.asarray(leads_out, dtype=np.int16)
    obs_arr = np.asarray(obs_values, dtype=np.float64)
    model_arrs = {exp: np.asarray(values, dtype=np.float64) for exp, values in model_arrays.items()}
    model_qr_arrs = {
        exp: np.asarray(values, dtype=np.float64) for exp, values in model_qr_arrays.items()
    }

    npz_path = args.processed_dir / f"disdrometer_rain_number_comparison_{output_tag}.npz"
    np.savez(
        npz_path,
        valid_time=valid_times_arr.astype("datetime64[s]").astype(str),
        init_time=init_times_arr.astype("datetime64[s]").astype(str),
        lead_hours=leads_arr,
        obs_rain_number_m3=obs_arr,
        obs_precip_rate_mm_h=np.asarray([row["obs_precip_rate_mm_h"] for row in rows], dtype=float),
        obs_sample_count=np.asarray([row["obs_sample_count"] for row in rows], dtype=int),
        mask_lat=domain_mask.lat.astype(np.float32),
        mask_lon=domain_mask.lon.astype(np.float32),
        radar_mask=domain_mask.mask.astype(np.int8),
        **{f"{exp}_rain_number_m3": values for exp, values in model_arrs.items()},
        **{f"{exp}_rain_mixing_ratio_kgkg": values for exp, values in model_qr_arrs.items()},
    )

    timeseries_csv = args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_timeseries.csv"
    save_timeseries_csv(timeseries_csv, rows, experiments)

    summaries = [
        summarize_experiment(exp, obs_arr, model_arrs[exp], model_qr_arrs[exp])
        for exp in experiments
    ]
    summary_csv = args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_summary.csv"
    save_summary_csv(summary_csv, summaries)

    timeseries_png = args.figure_dir / f"disdrometer_rain_number_timeseries_{output_tag}.png"
    scatter_png = args.figure_dir / f"disdrometer_rain_number_scatter_{output_tag}.png"
    distribution_png = args.figure_dir / f"disdrometer_rain_number_distribution_{output_tag}.png"
    if not args.no_plots:
        plot_timeseries(timeseries_png, valid_times_arr, obs_arr, model_arrs, lead_text)
        plot_scatter(scatter_png, obs_arr, model_arrs, lead_text)
        plot_distribution(distribution_png, obs_arr, model_arrs, lead_text)

    outputs = {
        "masked_grid_netcdf": grid_nc_path,
        "npz": npz_path,
        "timeseries_csv": timeseries_csv,
        "summary_csv": summary_csv,
        "timeseries_png": timeseries_png,
        "scatter_png": scatter_png,
        "distribution_png": distribution_png,
    }
    for key, path in outputs.items():
        if path.exists():
            print(f"{key}: {path}", flush=True)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract bottom-layer rain number from raw ALARO FA files over the "
            "radar masked domain and compare masked-domain means with the "
            "ARM GOAmazon disdrometer."
        )
    )
    parser.add_argument(
        "--lead",
        default="all",
        help="Forecast lead selection: all, a single hour like 0024, or a range like 0000-0024.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Experiments to compare.",
    )
    parser.add_argument("--level", type=int, default=BOTTOM_LEVEL, help="Bottom model level fid number.")
    parser.add_argument("--obs-window-minutes", type=int, default=30)
    parser.add_argument("--obs-min-precip", type=float, default=0.1)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=32)
    parser.add_argument("--no-grid-netcdf", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--analytics-dir", type=Path, default=ANALYTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--masked-netcdf-dir", type=Path, default=MASKED_NETCDF_DIR)
    parser.set_defaults(station_lat=DEFAULT_LAT, station_lon=DEFAULT_LON)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_comparison(args)


if __name__ == "__main__":
    main()
