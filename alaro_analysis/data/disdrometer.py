"""Disdrometer. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.observation_settings import CACHE_DIR
from alaro_analysis.data.observation_models import ObservationSeries, ObservationWindow

STRICT_OBS_MEMBER = "MANAUS_DISDROMETER_DATA/Merged/maoldS10.b1.Manaus_disdrometer_QC_Strict.nc"


def integrate_drop_number(
    number_density_drops: np.ndarray,
    class_size_width_mm: np.ndarray,
) -> np.ndarray:
    """Integrate Parsivel drop number density over diameter bins.

    ``number_density_drops`` is in 1/(m^3 mm), and class widths are in mm,
    so the result is total drop number concentration in 1/m^3.
    """
    density = np.asarray(number_density_drops, dtype=float)
    widths = np.asarray(class_size_width_mm, dtype=float)
    density = np.where(np.isfinite(density) & (density >= 0.0), density, np.nan)
    widths = np.where(np.isfinite(widths) & (widths >= 0.0), widths, np.nan)

    if density.ndim != 2:
        raise ValueError("number_density_drops must be 2-D: time x particle_size")
    if widths.ndim == 1:
        if widths.shape[0] != density.shape[1]:
            raise ValueError("class_size_width length does not match particle bins")
        return np.nansum(density * widths[None, :], axis=1)
    if widths.shape == density.shape:
        return np.nansum(density * widths, axis=1)
    if widths.T.shape == density.shape:
        return np.nansum(density * widths.T, axis=1)
    raise ValueError(
        "class_size_width must be particle_size, time x particle_size, or particle_size x time"
    )


def datetime64_from_seconds_since(values: np.ndarray, units: str) -> np.ndarray:
    match = re.match(r"seconds since (\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units)
    if not match:
        raise ValueError(f"Unsupported time units: {units!r}")
    base = match.group(1)
    if match.group(2):
        base = f"{base}T{match.group(2)}"
    base_time = np.datetime64(base, "s")
    seconds = np.asarray(values, dtype="float64")
    return base_time + np.rint(seconds).astype("timedelta64[s]")


def extract_strict_obs(obs_zip: Path, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / Path(STRICT_OBS_MEMBER).name
    if target.exists() and target.stat().st_size > 0:
        return target

    with zipfile.ZipFile(obs_zip) as archive:
        with archive.open(STRICT_OBS_MEMBER) as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return target


def read_observations(
    obs_zip: Path,
    cache_dir: Path = CACHE_DIR,
    min_precip_rate_mm_h: float = 0.1,
) -> ObservationSeries:
    obs_path = extract_strict_obs(obs_zip, cache_dir)
    with Dataset(obs_path) as ds:
        times = datetime64_from_seconds_since(ds["time"][:], ds["time"].units)
        rain_number = integrate_drop_number(
            np.asarray(ds["number_density_drops"][:], dtype=float),
            np.asarray(ds["class_size_width"][:], dtype=float),
        )
        precip = np.asarray(ds["precip_rate"][:], dtype=float)
        dmedian = np.asarray(ds["median_volume_diameter"][:], dtype=float)

    precip = np.where(np.isfinite(precip) & (precip >= min_precip_rate_mm_h), precip, np.nan)
    rain_number = np.where(np.isfinite(precip), rain_number, np.nan)
    dmedian = np.where(np.isfinite(precip), dmedian, np.nan)
    return ObservationSeries(
        times=times,
        rain_number_m3=rain_number,
        precip_rate_mm_h=precip,
        median_volume_diameter_mm=dmedian,
    )


def average_observation_window(
    obs: ObservationSeries,
    valid_time: np.datetime64,
    half_window_minutes: int,
) -> ObservationWindow:
    half = np.timedelta64(int(half_window_minutes), "m")
    start = valid_time - half
    end = valid_time + half
    left = int(np.searchsorted(obs.times, start, side="left"))
    right = int(np.searchsorted(obs.times, end, side="right"))
    if right <= left:
        return ObservationWindow(np.nan, np.nan, np.nan, 0)

    rain_number = obs.rain_number_m3[left:right]
    precip = obs.precip_rate_mm_h[left:right]
    dmedian = obs.median_volume_diameter_mm[left:right]
    count = int(np.isfinite(rain_number).sum())
    return ObservationWindow(
        rain_number_m3=float(np.nanmean(rain_number)) if count else np.nan,
        precip_rate_mm_h=float(np.nanmean(precip)) if np.isfinite(precip).any() else np.nan,
        median_volume_diameter_mm=(
            float(np.nanmean(dmedian)) if np.isfinite(dmedian).any() else np.nan
        ),
        sample_count=count,
    )
