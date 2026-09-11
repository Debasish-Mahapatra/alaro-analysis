"""Model points. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.analysis.disdrometer import (
    RDMEANREQ_M,
    rain_mean_volume_diameter_mm,
    single_moment_rain_number_per_kg,
)
from alaro_analysis.common.constants import RD
from alaro_analysis.common.observation_settings import BOTTOM_LEVEL
from alaro_analysis.data.observation_models import ModelPoint


def _init_epygram_worker() -> None:
    import epygram

    epygram.init_env()


def _read_model_points_task(
    task: tuple[str, str, int, dict[str, str], tuple[str, ...], float, float, int],
) -> tuple[str, dict[str, Any], list[str]]:
    record_key, valid_time, lead, paths, experiments, lat, lon, level = task
    row: dict[str, Any] = {}
    warnings: list[str] = []
    for exp in experiments:
        try:
            point = read_model_point(
                Path(paths[exp]),
                exp,
                lat=lat,
                lon=lon,
                level=level,
            )
        except Exception as exc:
            warnings.append(
                f"WARNING {exp} {valid_time} lead +{lead:04d}: failed to read model point: {exc}"
            )
            point = ModelPoint(
                rain_number_m3=np.nan,
                rain_number_kg=np.nan,
                rain_mixing_ratio_kgkg=np.nan,
                dmean_mm=np.nan,
                rho_air_kg_m3=np.nan,
                temperature_k=np.nan,
                pressure_pa=np.nan,
                method="failed",
            )

        row[f"{exp}_rain_number_m3"] = point.rain_number_m3
        row[f"{exp}_rain_number_kg"] = point.rain_number_kg
        row[f"{exp}_rain_mixing_ratio_kgkg"] = point.rain_mixing_ratio_kgkg
        row[f"{exp}_dmean_mm"] = point.dmean_mm
        row[f"{exp}_rho_air_kg_m3"] = point.rho_air_kg_m3
        row[f"{exp}_temperature_k"] = point.temperature_k
        row[f"{exp}_pressure_pa"] = point.pressure_pa
        row[f"{exp}_method"] = point.method
    return record_key, row, warnings


def _fa_point(resource: Any, fid: str, lon: float, lat: float) -> float:
    field = resource.readfield(fid)
    if getattr(field, "spectral", False):
        field.sp2gp()
    return float(np.asarray(field.getvalue_ll(lon, lat)).squeeze())


def read_model_point(
    path: Path,
    experiment: str,
    lat: float,
    lon: float,
    level: int = BOTTOM_LEVEL,
) -> ModelPoint:
    import epygram

    rain_fid = f"S{level:03d}RAIN"
    temp_fid = f"S{level:03d}TEMPERATURE"
    pressure_fid = f"S{level:03d}PRESSURE"
    pnr_fid = f"S{level:03d}PNR"
    dmean_fid = f"S{level:03d}DMEANR"

    resource = epygram.formats.resource(str(path), "r")
    try:
        rain_qr = max(_fa_point(resource, rain_fid, lon, lat), 0.0)
        temperature_k = _fa_point(resource, temp_fid, lon, lat)
        pressure_pa = _fa_point(resource, pressure_fid, lon, lat)
        rho_air = pressure_pa / (RD * temperature_k)

        method = "derived_equilibrium_diameter"
        if experiment == "2mom":
            try:
                rain_number_kg = max(_fa_point(resource, pnr_fid, lon, lat), 0.0)
                method = "prognostic_pnr"
            except Exception:
                rain_number_kg = float(single_moment_rain_number_per_kg(rain_qr))
                method = "derived_equilibrium_diameter"
        else:
            rain_number_kg = float(single_moment_rain_number_per_kg(rain_qr))

        try:
            dmean_mm = _fa_point(resource, dmean_fid, lon, lat) * 1000.0
            if not np.isfinite(dmean_mm) or dmean_mm <= 0.0:
                raise ValueError("invalid DMEANR")
        except Exception:
            if rain_qr > 0.0 and rain_number_kg > 0.0:
                dmean_mm = float(rain_mean_volume_diameter_mm(rain_qr, rain_number_kg))
            elif experiment != "2mom" and rain_qr > 0.0:
                dmean_mm = RDMEANREQ_M * 1000.0
            else:
                dmean_mm = np.nan

        return ModelPoint(
            rain_number_m3=rain_number_kg * rho_air,
            rain_number_kg=rain_number_kg,
            rain_mixing_ratio_kgkg=rain_qr,
            dmean_mm=dmean_mm,
            rho_air_kg_m3=rho_air,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            method=method,
        )
    finally:
        resource.close()
