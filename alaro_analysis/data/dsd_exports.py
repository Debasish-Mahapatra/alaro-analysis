"""Dsd exports. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.masks import DomainMask


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
    valid_times = np.asarray([np.datetime64(r["valid_time"]) for r in rows], dtype="datetime64[s]")
    init_times = np.asarray([np.datetime64(r["init_time"]) for r in rows], dtype="datetime64[s]")
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


def save_samples_npz(path: Path, datasets: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, samples in datasets.items():
        for key, values in samples.items():
            payload[f"{name}__{key}"] = values
    save_npz_compressed(path, **payload)
