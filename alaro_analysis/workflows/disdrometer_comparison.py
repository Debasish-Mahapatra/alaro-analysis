from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    RD,
)


from alaro_analysis.common.constants import RUNS_ROOT
MODEL_ROOT = RUNS_ROOT / "ALARO"
OBS_ZIP = RUNS_ROOT / "Obs" / "MANAUS_DISDROMETER_DATA.zip"
STRICT_OBS_MEMBER = (
    "MANAUS_DISDROMETER_DATA/Merged/"
    "maoldS10.b1.Manaus_disdrometer_QC_Strict.nc"
)
PROCESSED_DIR = RUNS_ROOT / "processed-data" / "disdrometer_comparison"
ANALYTICS_DIR = PROCESSED_DIR / "analytics"
FIGURE_DIR = RUNS_ROOT / "figures" / "disdrometer_comparison"
CACHE_DIR = Path("cache") / "disdrometer"

BOTTOM_LEVEL = 87
DEFAULT_LAT = -3.21297
DEFAULT_LON = -60.5981
DEFAULT_ALT_M = 50.0
WATER_DENSITY = 1000.0
RDMEANREQ_M = 9.0e-4
SOURCE_RAIN_NUMBER_COEFFICIENT = 0.0019 / RDMEANREQ_M**3
PF_DAY_RE = re.compile(r"^pf(\d{8})$")
PF_OUTPUT_RE = re.compile(r"^pfABOFABOF\+(\d{4})$")


@dataclass(frozen=True)
class ObservationSeries:
    times: np.ndarray
    rain_number_m3: np.ndarray
    precip_rate_mm_h: np.ndarray
    median_volume_diameter_mm: np.ndarray


@dataclass(frozen=True)
class ObservationWindow:
    rain_number_m3: float
    precip_rate_mm_h: float
    median_volume_diameter_mm: float
    sample_count: int


@dataclass(frozen=True)
class ModelPoint:
    rain_number_m3: float
    rain_number_kg: float
    rain_mixing_ratio_kgkg: float
    dmean_mm: float
    rho_air_kg_m3: float
    temperature_k: float
    pressure_pa: float
    method: str


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
                f"WARNING {exp} {valid_time} lead +{lead:04d}: "
                f"failed to read model point: {exc}"
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


def single_moment_rain_number_per_kg(
    rain_mixing_ratio_kgkg: np.ndarray | float,
    dmean_m: float = RDMEANREQ_M,
) -> np.ndarray:
    """Rain number concentration per kg from the model's fixed diameter logic."""
    qr = np.maximum(np.asarray(rain_mixing_ratio_kgkg, dtype=float), 0.0)
    return (0.0019 / dmean_m**3) * qr


def rain_mean_volume_diameter_mm(
    rain_mixing_ratio_kgkg: np.ndarray | float,
    rain_number_kg: np.ndarray | float,
) -> np.ndarray:
    """Return mean volume diameter implied by q_r and N_r."""
    qr = np.asarray(rain_mixing_ratio_kgkg, dtype=float)
    nr = np.asarray(rain_number_kg, dtype=float)
    out = np.full(np.broadcast(qr, nr).shape, np.nan, dtype=float)
    mask = np.isfinite(qr) & np.isfinite(nr) & (qr > 0.0) & (nr > 0.0)
    out[mask] = (
        6.0 * qr[mask] / (WATER_DENSITY * math.pi * nr[mask])
    ) ** (1.0 / 3.0) * 1000.0
    return out


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
        "class_size_width must be particle_size, time x particle_size, "
        "or particle_size x time"
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


def parse_pf_day(path: Path) -> datetime:
    match = PF_DAY_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse model day from {path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def discover_model_files(
    experiment: str,
    lead: str,
    model_root: Path = MODEL_ROOT,
    max_days: int | None = None,
) -> dict[np.datetime64, Path]:
    lead_token = f"{int(lead):04d}"
    base = model_root / experiment / "untar-output"
    day_dirs = sorted(
        d for d in base.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name)
    )
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    files: dict[np.datetime64, Path] = {}
    for day_dir in day_dirs:
        path = day_dir / f"pfABOFABOF+{lead_token}"
        if not path.exists():
            continue
        init_time = parse_pf_day(day_dir)
        valid_time = init_time + timedelta(hours=int(lead_token))
        files[np.datetime64(valid_time, "s")] = path
    return files


def parse_lead_selection(value: str) -> tuple[int, ...] | None:
    """Return requested leads, or None for all available leads."""
    token = value.strip().lower()
    if token in {"all", "*"}:
        return None
    leads: list[int] = []
    for piece in token.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = (int(part) for part in piece.split("-", 1))
            leads.extend(range(start, end + 1))
        else:
            leads.append(int(piece))
    return tuple(sorted(set(leads)))


def lead_label(leads: tuple[int, ...] | None) -> str:
    if leads is None:
        return "all_leads"
    if len(leads) == 1:
        return f"lead{leads[0]:04d}"
    return f"leads{leads[0]:04d}-{leads[-1]:04d}"


def title_lead_text(leads: tuple[int, ...] | None) -> str:
    if leads is None:
        return "all forecast hours"
    if len(leads) == 1:
        return f"lead +{leads[0]:04d}"
    return f"leads +{leads[0]:04d}-+{leads[-1]:04d}"


def discover_model_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    model_root: Path = MODEL_ROOT,
    max_days: int | None = None,
) -> dict[tuple[str, str, int], Path]:
    """Return model files keyed by (valid_time, init_time, lead_hour)."""
    base = model_root / experiment / "untar-output"
    day_dirs = sorted(
        d for d in base.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name)
    )
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    lead_set = set(leads) if leads is not None else None
    files: dict[tuple[str, str, int], Path] = {}
    for day_dir in day_dirs:
        init_time = np.datetime64(parse_pf_day(day_dir), "s")
        for path in sorted(day_dir.iterdir()):
            match = PF_OUTPUT_RE.match(path.name)
            if not match:
                continue
            lead = int(match.group(1))
            if lead_set is not None and lead not in lead_set:
                continue
            valid_time = init_time + np.timedelta64(lead, "h")
            key = (datetime64_key(valid_time), datetime64_key(init_time), lead)
            files[key] = path
    return files


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


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def log10_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(mask.sum()) < 2:
        return np.nan
    return float(np.corrcoef(np.log10(x[mask]), np.log10(y[mask]))[0, 1])


def summarize_experiment(
    experiment: str,
    obs_values: np.ndarray,
    model_values: np.ndarray,
    model_qr: np.ndarray,
) -> dict[str, Any]:
    matched = np.isfinite(obs_values) & np.isfinite(model_values)
    positive = matched & (obs_values > 0.0) & (model_values > 0.0)
    bias = model_values[matched] - obs_values[matched]
    ratio = model_values[positive] / obs_values[positive]
    return {
        "experiment": experiment,
        "label": EXPERIMENT_LABELS.get(experiment, experiment),
        "n_matched": int(matched.sum()),
        "n_positive_pair": int(positive.sum()),
        "obs_mean_m3": finite_mean(obs_values[matched]),
        "obs_median_m3": finite_median(obs_values[matched]),
        "model_mean_m3": finite_mean(model_values[matched]),
        "model_median_m3": finite_median(model_values[matched]),
        "bias_mean_m3": finite_mean(bias),
        "bias_median_m3": finite_median(bias),
        "median_model_obs_ratio": finite_median(ratio),
        "log10_corr": log10_correlation(obs_values, model_values),
        "model_rain_qr_positive_count": int(np.sum(np.isfinite(model_qr) & (model_qr > 0.0))),
    }


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
            ]
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields + exp_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def _as_datetime(values: np.ndarray) -> list[datetime]:
    return [v.astype("datetime64[s]").astype(datetime) for v in values]


def datetime64_key(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")


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
            lw=1.4,
            label=EXPERIMENT_LABELS[exp],
            alpha=0.9,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_title(f"Near-surface rain number concentration, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="upper right", frameon=False, ncol=4)
    fig.savefig(out, dpi=450)
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
    positives = [obs[obs > 0.0]]
    positives.extend(v[v > 0.0] for v in model.values())
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
            s=18,
            alpha=0.65,
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
    axes[0].set_ylabel("Model (m$^{-3}$)")
    fig.suptitle(f"Disdrometer vs lowest model level rain number, {lead_text}")
    fig.tight_layout()
    fig.savefig(out, dpi=450)
    plt.close(fig)


def plot_distribution(
    out: Path,
    obs: np.ndarray,
    model: dict[str, np.ndarray],
    lead_text: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    series = {"Obs": obs, **{EXPERIMENT_LABELS[e]: v for e, v in model.items()}}
    finite = np.concatenate(
        [v[np.isfinite(v) & (v > 0.0)] for v in series.values() if np.asarray(v).size]
    )
    if finite.size:
        bins = np.logspace(
            math.log10(max(1.0e-3, float(np.nanpercentile(finite, 1)) * 0.5)),
            math.log10(float(np.nanpercentile(finite, 99)) * 2.0),
            34,
        )
    else:
        bins = np.logspace(-2, 4, 34)
    for label, values in series.items():
        values = np.asarray(values)
        clean = values[np.isfinite(values) & (values > 0.0)]
        if clean.size == 0:
            continue
        color = "#111111" if label == "Obs" else EXPERIMENT_COLORS[
            next(exp for exp in model if EXPERIMENT_LABELS[exp] == label)
        ]
        ax.hist(
            clean,
            bins=bins,
            histtype="step",
            density=True,
            lw=1.7,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rain drop number concentration (m$^{-3}$)")
    ax.set_ylabel("Density")
    ax.set_title(f"Matched rain number distributions, {lead_text}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out, dpi=450)
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> dict[str, Path]:
    experiments = tuple(args.experiments)
    workers = max(1, min(int(args.workers), 32))
    leads = parse_lead_selection(args.lead)
    output_tag = lead_label(leads)
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

    import epygram

    epygram.init_env()
    model_records = {
        exp: discover_model_records(exp, leads, args.model_root, args.max_days)
        for exp in experiments
    }
    common_records = sorted(set.intersection(*(set(v) for v in model_records.values())))
    if not common_records:
        raise RuntimeError("No common model files found for requested experiments/leads")
    unique_valid_times = sorted({record[0] for record in common_records})
    print(f"lead selection: {lead_text}", flush=True)
    print(f"common model records: {len(common_records):,}", flush=True)
    print(f"unique valid times: {len(unique_valid_times):,}", flush=True)
    print(f"workers: {workers}", flush=True)

    obs_windows: dict[str, ObservationWindow] = {}
    for valid_time_key in unique_valid_times:
        window = average_observation_window(
            obs,
            np.datetime64(valid_time_key),
            args.obs_window_minutes,
        )
        obs_windows[valid_time_key] = window

    tasks = [
        (
            f"{init_time_key}+{lead:04d}->{valid_time_key}",
            valid_time_key,
            lead,
            {exp: str(model_records[exp][record]) for exp in experiments},
            experiments,
            args.station_lat,
            args.station_lon,
            args.level,
        )
        for record in common_records
        for valid_time_key, init_time_key, lead in [record]
    ]

    model_rows: dict[str, dict[str, Any]] = {}
    if workers == 1:
        import epygram

        epygram.init_env()
        for idx, task in enumerate(tasks, 1):
            valid_time_key, model_row, warnings = _read_model_points_task(task)
            for warning in warnings:
                print(warning, flush=True)
            model_rows[valid_time_key] = model_row
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"processed {idx}/{len(tasks)} valid times", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_epygram_worker,
            maxtasksperchild=args.tasks_per_child,
        ) as pool:
            for idx, (valid_time_key, model_row, warnings) in enumerate(
                pool.imap_unordered(_read_model_points_task, tasks),
                1,
            ):
                for warning in warnings:
                    print(warning, flush=True)
                model_rows[valid_time_key] = model_row
                if idx % args.progress_every == 0 or idx == len(tasks):
                    print(f"processed {idx}/{len(tasks)} valid times", flush=True)

    rows: list[dict[str, Any]] = []
    model_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}
    model_qr_arrays: dict[str, list[float]] = {exp: [] for exp in experiments}
    obs_values: list[float] = []
    valid_times_out: list[np.datetime64] = []
    init_times_out: list[np.datetime64] = []
    leads_out: list[int] = []

    for record in common_records:
        valid_time_key, init_time_key, lead = record
        record_key = f"{init_time_key}+{lead:04d}->{valid_time_key}"
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
        obs_values.append(window.rain_number_m3)
        valid_times_out.append(np.datetime64(valid_time_key))
        init_times_out.append(np.datetime64(init_time_key))
        leads_out.append(lead)

        row.update(model_rows[record_key])
        for exp in experiments:
            model_arrays[exp].append(row[f"{exp}_rain_number_m3"])
            model_qr_arrays[exp].append(row[f"{exp}_rain_mixing_ratio_kgkg"])

        rows.append(row)

    valid_times_arr = np.asarray(valid_times_out, dtype="datetime64[s]")
    init_times_arr = np.asarray(init_times_out, dtype="datetime64[s]")
    leads_arr = np.asarray(leads_out, dtype=int)
    obs_arr = np.asarray(obs_values, dtype=float)
    model_arrs = {
        exp: np.asarray(values, dtype=float) for exp, values in model_arrays.items()
    }
    model_qr_arrs = {
        exp: np.asarray(values, dtype=float) for exp, values in model_qr_arrays.items()
    }

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.analytics_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.processed_dir / f"disdrometer_rain_number_comparison_{output_tag}.npz"
    np.savez(
        npz_path,
        valid_time=valid_times_arr.astype("datetime64[s]").astype(str),
        init_time=init_times_arr.astype("datetime64[s]").astype(str),
        lead_hours=leads_arr,
        obs_rain_number_m3=obs_arr,
        obs_precip_rate_mm_h=np.asarray(
            [row["obs_precip_rate_mm_h"] for row in rows], dtype=float
        ),
        obs_median_volume_diameter_mm=np.asarray(
            [row["obs_median_volume_diameter_mm"] for row in rows], dtype=float
        ),
        obs_sample_count=np.asarray([row["obs_sample_count"] for row in rows], dtype=int),
        **{f"{exp}_rain_number_m3": values for exp, values in model_arrs.items()},
        **{f"{exp}_rain_mixing_ratio_kgkg": values for exp, values in model_qr_arrs.items()},
    )

    timeseries_csv = (
        args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_timeseries.csv"
    )
    save_timeseries_csv(timeseries_csv, rows, experiments)

    summaries = [
        summarize_experiment(exp, obs_arr, model_arrs[exp], model_qr_arrs[exp])
        for exp in experiments
    ]
    summary_csv = (
        args.analytics_dir / f"disdrometer_rain_number_comparison_{output_tag}_summary.csv"
    )
    save_summary_csv(summary_csv, summaries)

    timeseries_png = (
        args.figure_dir / f"disdrometer_rain_number_timeseries_{output_tag}.png"
    )
    scatter_png = args.figure_dir / f"disdrometer_rain_number_scatter_{output_tag}.png"
    distribution_png = (
        args.figure_dir / f"disdrometer_rain_number_distribution_{output_tag}.png"
    )
    if not args.no_plots:
        plot_timeseries(timeseries_png, valid_times_arr, obs_arr, model_arrs, lead_text)
        plot_scatter(scatter_png, obs_arr, model_arrs, lead_text)
        plot_distribution(distribution_png, obs_arr, model_arrs, lead_text)

    outputs = {
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
            "Compare ARM GOAmazon Manacapuru disdrometer drop number "
            "concentration with lowest-level ALARO rain number."
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
    parser.add_argument("--level", type=int, default=BOTTOM_LEVEL, help="Model level to read.")
    parser.add_argument("--station-lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--station-lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--station-alt-m", type=float, default=DEFAULT_ALT_M)
    parser.add_argument("--obs-window-minutes", type=int, default=30)
    parser.add_argument("--obs-min-precip", type=float, default=0.1)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tasks-per-child", type=int, default=32)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--obs-zip", type=Path, default=OBS_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--analytics-dir", type=Path, default=ANALYTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_comparison(args)


if __name__ == "__main__":
    main()
