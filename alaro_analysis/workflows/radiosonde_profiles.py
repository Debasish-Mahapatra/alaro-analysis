"""Extract ALARO model profiles matched to ARM radiosonde launch times."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from alaro_analysis.common.constants import EPS, EXPERIMENT_LABELS, G


DEFAULT_SONDE_ROOT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ARM/radiosond_data/maosondewnpnM1.b1"
)
DEFAULT_ALARO_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
DEFAULT_OUTPUT_SUBDIR = "radiosonde-matched-profiles"
MODEL_FILE_TEMPLATE = "pfABOFABOF+{hour:04d}"
SONDE_RE = re.compile(r"^maosondewnpnM1\.b1\.(\d{8})\.(\d{6})\.cdf$")

EXPERIMENT_ALIASES = {
    "c1m": "control",
    "control": "control",
    "g1m": "graupel",
    "graupel": "graupel",
    "g2m": "2mom",
    "2mom": "2mom",
}


@dataclass(frozen=True)
class SondeLaunch:
    launch_id: str
    launch_time_utc: datetime
    model_valid_time_utc: datetime
    date_token: str
    model_hour: int
    sonde_file: Path
    launch_lat: float
    launch_lon: float


@dataclass(frozen=True)
class GridPoint:
    y: int
    x: int
    lat: float
    lon: float
    distance_deg: float


@dataclass(frozen=True)
class ProfileResult:
    launch: SondeLaunch
    source_file: Path
    tdry_c: np.ndarray
    dewpoint_c: np.ndarray
    rh_percent: np.ndarray
    pressure_pa: np.ndarray
    specific_humidity: np.ndarray
    temperature_k: np.ndarray
    model_height_m: np.ndarray
    model_levels: np.ndarray
    grid_point: GridPoint


@dataclass(frozen=True)
class MissingProfile:
    launch: SondeLaunch
    source_file: Path
    reason: str


def _require_faxarray():
    try:
        import faxarray as fx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faxarray is required for --source fa. Use the epygram environment, "
            "for example: /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/envs/epygram/bin/python "
            "-m alaro_analysis.workflows.radiosonde_profiles ..."
        ) from exc
    return fx


def _parse_yyyymmdd(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={value!r}; expected YYYYMMDD") from exc


def _parse_sonde_time(path: Path) -> datetime:
    match = SONDE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Not an ARM MAO sonde file name: {path.name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _first_finite_launch_coord(path: Path) -> tuple[float, float]:
    with xr.open_dataset(path, decode_times=False) as ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lon >= -180.0)
        & (lon <= 180.0)
    )
    if not np.any(valid):
        raise ValueError(f"No finite launch latitude/longitude in {path}")
    first = int(np.flatnonzero(valid)[0])
    return float(lat[first]), float(lon[first])


def discover_sonde_launches(
    sonde_root: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> list[SondeLaunch]:
    """Read ARM sonde launches and map each launch to its UTC model hour."""
    paths = sorted(sonde_root.glob("maosondewnpnM1.b1.*.*.cdf"))
    launches: list[SondeLaunch] = []

    for path in paths:
        launch_time = _parse_sonde_time(path)
        launch_date = launch_time.date()
        if start_date is not None and launch_date < start_date:
            continue
        if end_date is not None and launch_date > end_date:
            continue

        lat, lon = _first_finite_launch_coord(path)
        model_valid_time = launch_time.replace(minute=0, second=0, microsecond=0)
        date_token = model_valid_time.strftime("%Y%m%d")
        model_hour = int(model_valid_time.strftime("%H"))
        launch_id = launch_time.strftime("%Y%m%dT%H%M%SZ")
        launches.append(
            SondeLaunch(
                launch_id=launch_id,
                launch_time_utc=launch_time,
                model_valid_time_utc=model_valid_time,
                date_token=date_token,
                model_hour=model_hour,
                sonde_file=path,
                launch_lat=lat,
                launch_lon=lon,
            )
        )

        if limit is not None and len(launches) >= limit:
            break

    return launches


def median_launch_site(launches: Sequence[SondeLaunch]) -> tuple[float, float]:
    if not launches:
        raise ValueError("No radiosonde launches found.")
    lat = np.asarray([launch.launch_lat for launch in launches], dtype=np.float64)
    lon = np.asarray([launch.launch_lon for launch in launches], dtype=np.float64)
    return float(np.nanmedian(lat)), float(np.nanmedian(lon))


def _nearest_grid_point(lat: xr.DataArray, lon: xr.DataArray, site_lat: float, site_lon: float) -> GridPoint:
    lat_values = np.asarray(lat.values, dtype=np.float64)
    lon_values = np.asarray(lon.values, dtype=np.float64)
    distance = np.hypot(lat_values - site_lat, lon_values - site_lon)
    if not np.isfinite(distance).any():
        raise ValueError("Model grid has no finite latitude/longitude values.")
    y, x = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    return GridPoint(
        y=int(y),
        x=int(x),
        lat=float(lat_values[y, x]),
        lon=float(lon_values[y, x]),
        distance_deg=float(distance[y, x]),
    )


def _specific_humidity_to_vapor_pressure_pa(
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    q = np.asarray(specific_humidity, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    q = np.where(q > 0.0, q, np.nan)
    p = np.where(p > 0.0, p, np.nan)
    return q * p / (EPS + (1.0 - EPS) * q)


def dewpoint_c_from_specific_humidity(
    specific_humidity: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute dewpoint temperature in degC from q and pressure."""
    e = _specific_humidity_to_vapor_pressure_pa(specific_humidity, pressure_pa)
    log_term = np.log(e / 611.2)
    return (243.5 * log_term) / (17.67 - log_term)


def relative_humidity_percent(
    specific_humidity: np.ndarray,
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Compute RH in percent from q, T, and pressure."""
    e = _specific_humidity_to_vapor_pressure_pa(specific_humidity, pressure_pa)
    t = np.asarray(temperature_k, dtype=np.float64)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def _data_var_name(ds: xr.Dataset, requested: str) -> str:
    candidates = (requested, requested.replace(".", "_"))
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Variable {requested!r} not found. Available: {list(ds.data_vars)}")


def _model_levels_from_da(da: xr.DataArray) -> np.ndarray:
    level_values = da.attrs.get("level_values")
    if level_values is not None:
        return np.asarray(level_values, dtype=np.int32)
    if "level" in da.coords:
        return np.asarray(da["level"].values, dtype=np.int32)
    return np.arange(int(da.sizes["level"]), dtype=np.int32)


def _height_from_geopotentiel(da: xr.DataArray, point: GridPoint) -> np.ndarray:
    values = np.asarray(da.isel(y=point.y, x=point.x).values, dtype=np.float64).squeeze()
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"m", "meter", "meters", "metre", "metres"}:
        return values
    return values / G


def _source_file_for_launch(
    alaro_root: Path,
    experiment: str,
    launch: SondeLaunch,
    source: str,
) -> Path:
    if source == "fa":
        return (
            alaro_root
            / experiment
            / "untar-output"
            / f"pf{launch.date_token}"
            / MODEL_FILE_TEMPLATE.format(hour=launch.model_hour)
        )
    if source == "netcdf":
        return (
            alaro_root
            / experiment
            / "masked-netcdf"
            / "TEMPERATURE"
            / f"pf{launch.date_token}"
            / f"{MODEL_FILE_TEMPLATE.format(hour=launch.model_hour)}.nc"
        )
    raise ValueError(f"Unsupported source: {source}")


def _first_existing_source_file(
    alaro_root: Path,
    experiment: str,
    launches: Sequence[SondeLaunch],
    source: str,
) -> Path:
    for launch in launches:
        path = _source_file_for_launch(alaro_root, experiment, launch, source)
        if path.exists():
            return path
    raise FileNotFoundError(f"No {source} source files found for {experiment}")


def _grid_point_from_source_file(
    source_file: Path,
    *,
    alaro_root: Path,
    experiment: str,
    launch: SondeLaunch,
    site_lat: float,
    site_lon: float,
    source: str,
) -> GridPoint:
    if source == "fa":
        fx = _require_faxarray()
        ds = fx.open_dataset(str(source_file), variables=["PRESSURE"], stack_levels=True)
        try:
            return _nearest_grid_point(ds["lat"], ds["lon"], site_lat, site_lon)
        finally:
            ds.close()

    if source == "netcdf":
        with xr.open_dataset(source_file) as ds:
            return _nearest_grid_point(ds["lat"], ds["lon"], site_lat, site_lon)

    raise ValueError(f"Unsupported source: {source}")


def _read_fa_profile(
    source_file: Path,
    site_lat: float,
    site_lon: float,
    grid_point: GridPoint | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, GridPoint]:
    fx = _require_faxarray()
    ds = fx.open_dataset(
        str(source_file),
        variables=["TEMPERATURE", "HUMI.SPECIFI", "PRESSURE", "GEOPOTENTIEL"],
        stack_levels=True,
    )
    try:
        point = grid_point or _nearest_grid_point(ds["lat"], ds["lon"], site_lat, site_lon)
        temp_name = _data_var_name(ds, "TEMPERATURE")
        q_name = _data_var_name(ds, "HUMI.SPECIFI")
        pressure_name = _data_var_name(ds, "PRESSURE")
        height_name = _data_var_name(ds, "GEOPOTENTIEL")
        values = {
            "temperature_k": np.asarray(ds[temp_name].isel(y=point.y, x=point.x).values).squeeze(),
            "specific_humidity": np.asarray(ds[q_name].isel(y=point.y, x=point.x).values).squeeze(),
            "pressure_pa": np.asarray(ds[pressure_name].isel(y=point.y, x=point.x).values).squeeze(),
            "model_height_m": _height_from_geopotentiel(ds[height_name], point),
        }
        model_levels = _model_levels_from_da(ds[temp_name])
        return values, model_levels, point
    finally:
        ds.close()


def _extract_one_profile_worker(
    index: int,
    alaro_root: str,
    experiment: str,
    launch: SondeLaunch,
    site_lat: float,
    site_lon: float,
    source: str,
    grid_point: GridPoint | None,
) -> tuple[int, ProfileResult | None, MissingProfile | None]:
    root = Path(alaro_root)
    source_file = _source_file_for_launch(root, experiment, launch, source)
    if not source_file.exists():
        return (
            index,
            None,
            MissingProfile(launch=launch, source_file=source_file, reason="missing source file"),
        )

    try:
        if source == "fa":
            values, model_levels, point = _read_fa_profile(
                source_file, site_lat, site_lon, grid_point
            )
        else:
            values, model_levels, point = _read_netcdf_profile(
                root, experiment, launch, site_lat, site_lon, grid_point
            )
        return (
            index,
            _profile_from_values(launch, source_file, values, model_levels, point),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            index,
            None,
            MissingProfile(launch=launch, source_file=source_file, reason=str(exc)),
        )


def _read_netcdf_profile(
    alaro_root: Path,
    experiment: str,
    launch: SondeLaunch,
    site_lat: float,
    site_lon: float,
    grid_point: GridPoint | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, GridPoint]:
    values: dict[str, np.ndarray] = {}
    model_levels: np.ndarray | None = None
    point = grid_point

    files = {
        "temperature_k": ("TEMPERATURE", "TEMPERATURE"),
        "specific_humidity": ("HUMI.SPECIFI", "HUMI.SPECIFI"),
        "pressure_pa": ("PRESSURE", "PRESSURE"),
        "model_height_m": ("GEOPOTENTIEL", "GEOPOTENTIEL"),
    }
    for output_name, (folder_name, var_name) in files.items():
        path = (
            alaro_root
            / experiment
            / "masked-netcdf"
            / folder_name
            / f"pf{launch.date_token}"
            / f"{MODEL_FILE_TEMPLATE.format(hour=launch.model_hour)}.nc"
        )
        with xr.open_dataset(path) as ds:
            if point is None:
                point = _nearest_grid_point(ds["lat"], ds["lon"], site_lat, site_lon)
            ds_name = _data_var_name(ds, var_name)
            da = ds[ds_name]
            if output_name == "model_height_m":
                values[output_name] = _height_from_geopotentiel(da, point)
            else:
                values[output_name] = np.asarray(da.isel(y=point.y, x=point.x).values).squeeze()
            if model_levels is None:
                model_levels = _model_levels_from_da(da)

    if point is None or model_levels is None:
        raise ValueError("No NetCDF profile values were read.")
    return values, model_levels, point


def _profile_from_values(
    launch: SondeLaunch,
    source_file: Path,
    values: dict[str, np.ndarray],
    model_levels: np.ndarray,
    grid_point: GridPoint,
) -> ProfileResult:
    temperature_k = np.asarray(values["temperature_k"], dtype=np.float64)
    specific_humidity = np.asarray(values["specific_humidity"], dtype=np.float64)
    pressure_pa = np.asarray(values["pressure_pa"], dtype=np.float64)
    model_height_m = np.asarray(values["model_height_m"], dtype=np.float64)
    return ProfileResult(
        launch=launch,
        source_file=source_file,
        tdry_c=temperature_k - 273.15,
        dewpoint_c=dewpoint_c_from_specific_humidity(specific_humidity, pressure_pa),
        rh_percent=relative_humidity_percent(specific_humidity, temperature_k, pressure_pa),
        pressure_pa=pressure_pa,
        specific_humidity=specific_humidity,
        temperature_k=temperature_k,
        model_height_m=model_height_m,
        model_levels=np.asarray(model_levels, dtype=np.int32),
        grid_point=grid_point,
    )


def extract_experiment_profiles(
    *,
    alaro_root: Path,
    experiment: str,
    launches: Sequence[SondeLaunch],
    site_lat: float,
    site_lon: float,
    source: str,
    progress_every: int = 25,
    workers: int = 1,
) -> tuple[list[ProfileResult], list[MissingProfile]]:
    workers = max(1, int(workers))
    indexed_results: list[tuple[int, ProfileResult]] = []
    indexed_missing: list[tuple[int, MissingProfile]] = []
    grid_point: GridPoint | None = None

    if launches:
        sample_file = _first_existing_source_file(alaro_root, experiment, launches, source)
        sample_launch = next(
            launch
            for launch in launches
            if _source_file_for_launch(alaro_root, experiment, launch, source) == sample_file
        )
        grid_point = _grid_point_from_source_file(
            sample_file,
            alaro_root=alaro_root,
            experiment=experiment,
            launch=sample_launch,
            site_lat=site_lat,
            site_lon=site_lon,
            source=source,
        )
        print(
            f"[{experiment}] nearest grid point y={grid_point.y} x={grid_point.x} "
            f"lat={grid_point.lat:.6f} lon={grid_point.lon:.6f}",
            flush=True,
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _extract_one_profile_worker,
                    index,
                    str(alaro_root),
                    experiment,
                    launch,
                    site_lat,
                    site_lon,
                    source,
                    grid_point,
                )
                for index, launch in enumerate(launches, start=1)
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                index, result, miss = future.result()
                if result is not None:
                    indexed_results.append((index, result))
                if miss is not None:
                    indexed_missing.append((index, miss))
                if progress_every > 0 and (done % progress_every == 0 or done == len(futures)):
                    print(
                        f"[{experiment}] {done}/{len(launches)} launches checked; "
                        f"profiles={len(indexed_results)} missing={len(indexed_missing)}",
                        flush=True,
                    )
    else:
        for index, launch in enumerate(launches, start=1):
            _, result, miss = _extract_one_profile_worker(
                index,
                str(alaro_root),
                experiment,
                launch,
                site_lat,
                site_lon,
                source,
                grid_point,
            )
            if result is not None:
                indexed_results.append((index, result))
            if miss is not None:
                indexed_missing.append((index, miss))

            if progress_every > 0 and (index % progress_every == 0 or index == len(launches)):
                print(
                    f"[{experiment}] {index}/{len(launches)} launches checked; "
                    f"profiles={len(indexed_results)} missing={len(indexed_missing)}",
                    flush=True,
                )

    indexed_results.sort(key=lambda item: item[0])
    indexed_missing.sort(key=lambda item: item[0])
    return [item[1] for item in indexed_results], [item[1] for item in indexed_missing]


def _datetime64_seconds(values: Iterable[datetime]) -> np.ndarray:
    return np.asarray(
        [np.datetime64(v.replace(tzinfo=None), "s") for v in values],
        dtype="datetime64[s]",
    )


def _stack_profiles(results: Sequence[ProfileResult], attr: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(result, attr), dtype=np.float32) for result in results])


def _build_output_dataset(
    experiment: str,
    results: Sequence[ProfileResult],
    *,
    site_lat: float,
    site_lon: float,
    source: str,
) -> xr.Dataset:
    if not results:
        raise ValueError(f"No profiles extracted for {experiment}")

    model_levels = results[0].model_levels
    launch_index = np.arange(len(results), dtype=np.int32)
    launch_times = _datetime64_seconds(result.launch.launch_time_utc for result in results)
    valid_times = _datetime64_seconds(result.launch.model_valid_time_utc for result in results)
    point = results[0].grid_point

    ds = xr.Dataset(
        data_vars={
            "tdry": (
                ("launch", "level"),
                _stack_profiles(results, "tdry_c"),
                {"long_name": "Dry bulb air temperature", "units": "degC"},
            ),
            "dewpoint": (
                ("launch", "level"),
                _stack_profiles(results, "dewpoint_c"),
                {
                    "long_name": "Dewpoint temperature",
                    "units": "degC",
                    "comment": "Derived from model specific humidity and pressure.",
                },
            ),
            "rh": (
                ("launch", "level"),
                _stack_profiles(results, "rh_percent"),
                {
                    "long_name": "Relative humidity",
                    "units": "%",
                    "comment": "Derived from model specific humidity, dry-bulb temperature, and pressure.",
                },
            ),
            "pressure": (
                ("launch", "level"),
                _stack_profiles(results, "pressure_pa"),
                {"long_name": "Model pressure", "units": "Pa"},
            ),
            "specific_humidity": (
                ("launch", "level"),
                _stack_profiles(results, "specific_humidity"),
                {"long_name": "Model specific humidity", "units": "kg kg-1"},
            ),
            "temperature_k": (
                ("launch", "level"),
                _stack_profiles(results, "temperature_k"),
                {"long_name": "Model dry bulb air temperature", "units": "K"},
            ),
            "model_height": (
                ("launch", "level"),
                _stack_profiles(results, "model_height_m"),
                {
                    "long_name": "Model height above mean sea level",
                    "units": "m",
                    "comment": "Derived from ALARO GEOPOTENTIEL divided by standard gravity unless source was already in meters.",
                },
            ),
            "sonde_file": (
                ("launch",),
                np.asarray([str(result.launch.sonde_file) for result in results], dtype=object),
            ),
            "model_source_file": (
                ("launch",),
                np.asarray([str(result.source_file) for result in results], dtype=object),
            ),
            "launch_lat": (
                ("launch",),
                np.asarray([result.launch.launch_lat for result in results], dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "launch_lon": (
                ("launch",),
                np.asarray([result.launch.launch_lon for result in results], dtype=np.float32),
                {"units": "degrees_east"},
            ),
            "model_hour_utc": (
                ("launch",),
                np.asarray([result.launch.model_hour for result in results], dtype=np.int16),
                {"units": "hour"},
            ),
        },
        coords={
            "launch": launch_index,
            "level": np.arange(len(model_levels), dtype=np.int32),
            "model_level": ("level", model_levels, {"long_name": "ALARO model level"}),
            "launch_time_utc": ("launch", launch_times),
            "model_valid_time_utc": ("launch", valid_times),
        },
        attrs={
            "experiment": experiment,
            "experiment_label": EXPERIMENT_LABELS.get(experiment, experiment),
            "source": source,
            "site_lat": site_lat,
            "site_lon": site_lon,
            "model_grid_y": point.y,
            "model_grid_x": point.x,
            "model_grid_lat": point.lat,
            "model_grid_lon": point.lon,
            "model_grid_distance_deg": point.distance_deg,
            "time_convention": "ARM radiosonde launch filenames and model valid times are matched in UTC.",
            "hour_matching": "The radiosonde launch timestamp is floored to the containing UTC hour.",
            "vertical_coordinate": "model_height is height above mean sea level in meters.",
        },
    )
    return ds


def _write_missing_log(path: Path, missing: Sequence[MissingProfile]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["launch_id", "source_file", "reason", "sonde_file"])
        for item in missing:
            writer.writerow(
                [
                    item.launch.launch_id,
                    str(item.source_file),
                    item.reason,
                    str(item.launch.sonde_file),
                ]
            )


def _write_manifest(path: Path, results: Sequence[ProfileResult]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "launch_id",
                "launch_time_utc",
                "model_valid_time_utc",
                "model_hour_utc",
                "launch_lat",
                "launch_lon",
                "model_grid_y",
                "model_grid_x",
                "model_grid_lat",
                "model_grid_lon",
                "model_source_file",
                "sonde_file",
            ]
        )
        for result in results:
            launch = result.launch
            point = result.grid_point
            writer.writerow(
                [
                    launch.launch_id,
                    launch.launch_time_utc.isoformat(),
                    launch.model_valid_time_utc.isoformat(),
                    launch.model_hour,
                    f"{launch.launch_lat:.6f}",
                    f"{launch.launch_lon:.6f}",
                    point.y,
                    point.x,
                    f"{point.lat:.6f}",
                    f"{point.lon:.6f}",
                    str(result.source_file),
                    str(launch.sonde_file),
                ]
            )


def _resolve_experiments(tokens: Sequence[str]) -> list[str]:
    experiments: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in EXPERIMENT_ALIASES:
            raise ValueError(
                f"Unknown experiment {token!r}; expected one of "
                f"{', '.join(sorted(EXPERIMENT_ALIASES))}"
            )
        experiment = EXPERIMENT_ALIASES[key]
        if experiment not in experiments:
            experiments.append(experiment)
    return experiments


def run(args: argparse.Namespace) -> dict[str, object]:
    alaro_root = Path(args.alaro_root)
    sonde_root = Path(args.sonde_root)
    start_date = _parse_yyyymmdd(args.start_date, "start-date")
    end_date = _parse_yyyymmdd(args.end_date, "end-date")
    experiments = _resolve_experiments(args.experiments)

    launches = discover_sonde_launches(
        sonde_root,
        start_date=start_date,
        end_date=end_date,
        limit=args.max_launches,
    )
    if not launches:
        raise ValueError(f"No radiosonde launches found in {sonde_root}")

    if args.site_lat is None or args.site_lon is None:
        site_lat, site_lon = median_launch_site(launches)
    else:
        site_lat = float(args.site_lat)
        site_lon = float(args.site_lon)

    print(
        f"Using ARM radiosonde launches in UTC: {len(launches)} launches, "
        f"site=({site_lat:.6f}, {site_lon:.6f}), source={args.source}, "
        f"workers={args.workers}",
        flush=True,
    )

    summary: dict[str, object] = {
        "alaro_root": str(alaro_root),
        "sonde_root": str(sonde_root),
        "source": args.source,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "launches": len(launches),
        "workers": int(args.workers),
        "experiments": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for experiment in experiments:
        out_dir = alaro_root / experiment / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{experiment}_radiosonde_matched_profiles.nc"
        if out_file.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists and --no-overwrite was set: {out_file}")

        print(f"[{experiment}] extracting to {out_file}", flush=True)
        results, missing = extract_experiment_profiles(
            alaro_root=alaro_root,
            experiment=experiment,
            launches=launches,
            site_lat=site_lat,
            site_lon=site_lon,
            source=args.source,
            progress_every=args.progress_every,
            workers=args.workers,
        )

        manifest_file = out_dir / f"{experiment}_radiosonde_matched_manifest.csv"
        missing_file = out_dir / f"{experiment}_radiosonde_matched_missing.csv"
        summary_file = out_dir / f"{experiment}_radiosonde_matched_summary.json"

        if results:
            ds = _build_output_dataset(
                experiment,
                results,
                site_lat=site_lat,
                site_lon=site_lon,
                source=args.source,
            )
            encoding = {
                name: {"zlib": True, "complevel": int(args.compression_level)}
                for name in (
                    "tdry",
                    "dewpoint",
                    "rh",
                    "pressure",
                    "specific_humidity",
                    "temperature_k",
                    "model_height",
                )
            }
            ds.to_netcdf(out_file, mode="w", encoding=encoding)
            ds.close()
        _write_manifest(manifest_file, results)
        _write_missing_log(missing_file, missing)

        exp_summary = {
            "output_file": str(out_file) if results else None,
            "manifest_file": str(manifest_file),
            "missing_file": str(missing_file),
            "profiles_written": len(results),
            "missing_profiles": len(missing),
            "output_subdir": str(out_dir),
            "source": args.source,
            "workers": int(args.workers),
        }
        summary["experiments"][experiment] = exp_summary
        summary_file.write_text(json.dumps(exp_summary, indent=2, sort_keys=True) + "\n")
        print(
            f"[{experiment}] done: profiles={len(results)} missing={len(missing)}",
            flush=True,
        )

    if args.plot_output_dir:
        from alaro_analysis.workflows.plot_radiosonde_profiles import plot_profiles

        plot_args = argparse.Namespace(
            alaro_root=str(alaro_root),
            input_subdir=args.output_subdir,
            experiments=args.experiments,
            stat=args.plot_stat,
            max_height_m=args.plot_max_height_m,
            height_step_m=args.plot_height_step_m,
            min_samples=args.plot_min_samples,
            dpi=args.plot_dpi,
            output_dir=args.plot_output_dir,
        )
        plot_paths = plot_profiles(plot_args)
        summary["plot_outputs"] = [str(path) for path in plot_paths]
        for plot_path in plot_paths:
            print(f"Wrote plot: {plot_path}", flush=True)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dry-bulb temperature, dewpoint, and RH model profiles at "
            "the ARM MAO radiosonde launch site and launch UTC hours."
        )
    )
    parser.add_argument("--sonde-root", default=str(DEFAULT_SONDE_ROOT))
    parser.add_argument("--alaro-root", default=str(DEFAULT_ALARO_ROOT))
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["control", "graupel", "2mom"],
        help="Experiments or aliases to process: c1m/control g1m/graupel g2m/2mom.",
    )
    parser.add_argument(
        "--source",
        choices=["fa", "netcdf"],
        default="fa",
        help="Read raw FA untar-output files or the existing masked-netcdf cache.",
    )
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--site-lat", type=float, default=None)
    parser.add_argument("--site-lon", type=float, default=None)
    parser.add_argument("--start-date", metavar="YYYYMMDD", default=None)
    parser.add_argument("--end-date", metavar="YYYYMMDD", default=None)
    parser.add_argument(
        "--max-launches",
        type=int,
        default=None,
        help="Limit launches for smoke tests.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes per experiment. Use with care for raw FA reads.",
    )
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        help="Optional directory for separate tdry/dewpoint/RH plots after extraction.",
    )
    parser.add_argument("--plot-stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--plot-max-height-m", type=float, default=20000.0)
    parser.add_argument("--plot-height-step-m", type=float, default=250.0)
    parser.add_argument("--plot-min-samples", type=int, default=10)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
