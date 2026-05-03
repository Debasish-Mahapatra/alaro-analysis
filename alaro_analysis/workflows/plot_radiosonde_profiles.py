"""Plot ARM radiosonde and matched ALARO thermodynamic profiles."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS, G
from alaro_analysis.workflows.radiosonde_profiles import (
    DEFAULT_ALARO_ROOT,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_SONDE_ROOT,
    EXPERIMENT_ALIASES,
    MODEL_FILE_TEMPLATE,
    SondeLaunch,
    _data_var_name,
    dewpoint_c_from_specific_humidity,
    discover_sonde_launches,
    relative_humidity_percent,
)


DEFAULT_FIGURES_ROOT = DEFAULT_ALARO_ROOT.parent / "figures"
DEFAULT_OUTPUT_DIR = DEFAULT_FIGURES_ROOT / "radiosonde_profiles"

PANEL_VARS = {
    "tdry": {
        "model_name": "tdry",
        "obs_name": "tdry",
        "label": "Dry-bulb temperature (degC)",
        "title": "Dry-bulb temperature",
        "filename": "radiosonde_tdry_profile.png",
        "xlim": None,
    },
    "dewpoint": {
        "model_name": "dewpoint",
        "obs_name": "dp",
        "label": "Dewpoint temperature (degC)",
        "title": "Dewpoint temperature",
        "filename": "radiosonde_dewpoint_profile.png",
        "xlim": None,
    },
    "rh": {
        "model_name": "rh",
        "obs_name": "rh",
        "label": "Relative humidity (%)",
        "title": "Relative humidity",
        "filename": "radiosonde_rh_profile.png",
        "xlim": (0.0, 105.0),
    },
}


def _resolve_experiments(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in EXPERIMENT_ALIASES:
            raise ValueError(f"Unknown experiment {token!r}")
        exp = EXPERIMENT_ALIASES[key]
        if exp not in out:
            out.append(exp)
    return out


def _resolve_panel_names(tokens: Sequence[str] | None) -> list[str]:
    if not tokens:
        return list(PANEL_VARS)
    out: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in PANEL_VARS:
            raise ValueError(f"Unknown plot variable {token!r}; expected one of {', '.join(PANEL_VARS)}")
        if key not in out:
            out.append(key)
    return out


def _clean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).squeeze()
    return np.where((arr <= -9000.0) | ~np.isfinite(arr), np.nan, arr)


def _interp_profile(height_m: np.ndarray, values: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    z = _clean(height_m)
    v = _clean(values)
    mask = np.isfinite(z) & np.isfinite(v)
    if np.count_nonzero(mask) < 2:
        return np.full_like(target_m, np.nan, dtype=np.float64)
    z = z[mask]
    v = v[mask]
    order = np.argsort(z)
    z = z[order]
    v = v[order]
    z_unique, unique_idx = np.unique(z, return_index=True)
    v_unique = v[unique_idx]
    if z_unique.size < 2:
        return np.full_like(target_m, np.nan, dtype=np.float64)
    out = np.interp(target_m, z_unique, v_unique)
    out[(target_m < z_unique[0]) | (target_m > z_unique[-1])] = np.nan
    return out


def _spatial_nanmean(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=(-2, -1))


def _aggregate(profiles: list[np.ndarray], stat: str, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if not profiles:
        raise ValueError("No profiles to aggregate.")
    arr = np.stack(profiles)
    counts = np.count_nonzero(np.isfinite(arr), axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "median":
            center = np.nanmedian(arr, axis=0)
        else:
            center = np.nanmean(arr, axis=0)
    center[counts < min_samples] = np.nan
    return center, counts


def _masked_netcdf_path(
    alaro_root: Path,
    experiment: str,
    masked_subdir: str,
    folder_name: str,
    launch: SondeLaunch,
) -> Path:
    return (
        alaro_root
        / experiment
        / masked_subdir
        / folder_name
        / f"pf{launch.date_token}"
        / f"{MODEL_FILE_TEMPLATE.format(hour=launch.model_hour)}.nc"
    )


def _level_yx_values(da: xr.DataArray) -> np.ndarray:
    values = np.asarray(da.values, dtype=np.float64).squeeze()
    if values.ndim != 3:
        raise ValueError(
            f"Expected a level/y/x field after squeezing {da.name!r}; got shape {values.shape}"
        )
    return values


def _read_masked_area_profile(
    alaro_root: Path,
    experiment: str,
    launch: SondeLaunch,
    masked_subdir: str,
) -> dict[str, np.ndarray]:
    files = {
        "temperature_k": ("TEMPERATURE", "TEMPERATURE"),
        "specific_humidity": ("HUMI.SPECIFI", "HUMI.SPECIFI"),
        "pressure_pa": ("PRESSURE", "PRESSURE"),
        "height": ("GEOPOTENTIEL", "GEOPOTENTIEL"),
    }
    for folder_name, _ in files.values():
        path = _masked_netcdf_path(alaro_root, experiment, masked_subdir, folder_name, launch)
        if not path.exists():
            raise FileNotFoundError(path)

    fields: dict[str, np.ndarray] = {}
    for output_name, (folder_name, var_name) in files.items():
        path = _masked_netcdf_path(alaro_root, experiment, masked_subdir, folder_name, launch)
        with xr.open_dataset(path) as ds:
            da = ds[_data_var_name(ds, var_name)]
            values = _level_yx_values(da)
            if output_name == "height":
                units = str(da.attrs.get("units", "")).strip().lower()
                if units not in {"m", "meter", "meters", "metre", "metres"}:
                    values = values / G
            fields[output_name] = values

    temperature_k = fields["temperature_k"]
    pressure_pa = fields["pressure_pa"]
    specific_humidity = fields["specific_humidity"]
    return {
        "height": _spatial_nanmean(fields["height"]),
        "tdry": _spatial_nanmean(temperature_k - 273.15),
        "dewpoint": _spatial_nanmean(
            dewpoint_c_from_specific_humidity(specific_humidity, pressure_pa)
        ),
        "rh": _spatial_nanmean(
            relative_humidity_percent(specific_humidity, temperature_k, pressure_pa)
        ),
    }


def _area_profile_worker(
    index: int,
    alaro_root: str,
    experiment: str,
    launch: SondeLaunch,
    masked_subdir: str,
    height_grid_m: np.ndarray,
) -> tuple[int, dict[str, np.ndarray] | None, str | None]:
    try:
        raw = _read_masked_area_profile(Path(alaro_root), experiment, launch, masked_subdir)
        profiles = {
            panel_name: _interp_profile(raw["height"], raw[panel_name], height_grid_m)
            for panel_name in PANEL_VARS
        }
        return index, profiles, None
    except Exception as exc:  # noqa: BLE001
        return index, None, f"{launch.launch_id}: {exc}"


def _read_area_model_profiles(
    alaro_root: Path,
    experiment: str,
    launches: Sequence[SondeLaunch],
    height_grid_m: np.ndarray,
    *,
    masked_subdir: str,
    workers: int,
    progress_every: int,
) -> dict[str, list[np.ndarray]]:
    workers = max(1, int(workers))
    indexed: list[tuple[int, dict[str, np.ndarray]]] = []
    missing: list[tuple[int, str]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _area_profile_worker,
                    index,
                    str(alaro_root),
                    experiment,
                    launch,
                    masked_subdir,
                    height_grid_m,
                )
                for index, launch in enumerate(launches, start=1)
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                index, profiles, reason = future.result()
                if profiles is not None:
                    indexed.append((index, profiles))
                if reason is not None:
                    missing.append((index, reason))
                if progress_every > 0 and (done % progress_every == 0 or done == len(futures)):
                    print(
                        f"[{experiment}] area profiles {done}/{len(launches)}; "
                        f"read={len(indexed)} missing={len(missing)}",
                        flush=True,
                    )
    else:
        for index, launch in enumerate(launches, start=1):
            _, profiles, reason = _area_profile_worker(
                index,
                str(alaro_root),
                experiment,
                launch,
                masked_subdir,
                height_grid_m,
            )
            if profiles is not None:
                indexed.append((index, profiles))
            if reason is not None:
                missing.append((index, reason))
            if progress_every > 0 and (index % progress_every == 0 or index == len(launches)):
                print(
                    f"[{experiment}] area profiles {index}/{len(launches)}; "
                    f"read={len(indexed)} missing={len(missing)}",
                    flush=True,
                )

    if missing:
        print(f"[{experiment}] skipped {len(missing)} masked-area profiles", flush=True)
        for _, reason in missing[:5]:
            print(f"[{experiment}] missing sample: {reason}", flush=True)

    if not indexed:
        raise ValueError(f"No radar-masked area profiles were read for {experiment}")

    indexed.sort(key=lambda item: item[0])
    profiles_by_name = {name: [] for name in PANEL_VARS}
    for _, profiles in indexed:
        for panel_name in PANEL_VARS:
            profiles_by_name[panel_name].append(profiles[panel_name])
    return profiles_by_name


def _first_zero_crossing_height_km(values: np.ndarray, height_km: np.ndarray) -> float:
    v = _clean(values)
    z = _clean(height_km)
    mask = np.isfinite(v) & np.isfinite(z)
    v = v[mask]
    z = z[mask]
    if v.size < 2:
        return np.nan

    order = np.argsort(z)
    v = v[order]
    z = z[order]

    exact = np.where(v == 0.0)[0]
    if exact.size:
        return float(z[exact[0]])

    crossings = np.where((v[:-1] > 0.0) & (v[1:] < 0.0))[0]
    if crossings.size == 0:
        crossings = np.where((v[:-1] < 0.0) & (v[1:] > 0.0))[0]
    if crossings.size == 0:
        return np.nan

    i = int(crossings[0])
    v0, v1 = v[i], v[i + 1]
    z0, z1 = z[i], z[i + 1]
    if v1 == v0:
        return float(z0)
    return float(z0 + (0.0 - v0) * (z1 - z0) / (v1 - v0))


def _draw_freezing_level(
    ax: plt.Axes,
    freezing_km: float,
    *,
    color: str,
    label: str,
    marker: str | None = None,
    mark_at_zero_c: bool = False,
) -> None:
    if not np.isfinite(freezing_km):
        return
    ax.axhline(
        freezing_km,
        color=color,
        lw=1.1,
        ls=":",
        alpha=0.9,
        label=f"{label} 0 degC height",
        zorder=2,
    )
    if mark_at_zero_c and marker is not None:
        ax.plot(
            0.0,
            freezing_km,
            marker=marker,
            ms=6,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
            zorder=5,
        )


def _fmt_txt(value: object) -> str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.10g}"


def _write_profile_txt(
    output_path: Path,
    *,
    panel_name: str,
    spec: dict[str, object],
    y_km: np.ndarray,
    obs_data: dict[str, tuple[np.ndarray, np.ndarray]],
    model_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    experiments: Sequence[str],
    obs_freezing_km: float,
    model_freezing_km: dict[str, float],
    stat: str,
    model_source: str,
) -> Path:
    txt_dir = output_path.parent / "data_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"{output_path.stem}.txt"

    with txt_path.open("w", encoding="utf-8") as fh:
        title = f"Radiosonde {spec['title']} {stat} Height Profile Plot Data"
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Source plot: {output_path}\n")
        fh.write(f"Model source: {model_source}\n")
        fh.write(f"Variable: {panel_name}\n")
        fh.write(f"Statistic: {stat}\n")
        fh.write("Vertical coordinate: height above mean sea level (km)\n")
        fh.write(f"X variable: {spec['label']}\n\n")

        fh.write("Freezing-level data\n")
        fh.write("-------------------\n")
        fh.write("series,z_freeze_km\n")
        fh.write(f"ARM sonde,{_fmt_txt(obs_freezing_km)}\n")
        for exp in experiments:
            label = EXPERIMENT_LABELS.get(exp, exp)
            fh.write(f"{label},{_fmt_txt(model_freezing_km[exp])}\n")
        fh.write("\n")

        columns = ["height_km", "ARM_sonde", "ARM_sonde_count"]
        for exp in experiments:
            label = EXPERIMENT_LABELS.get(exp, exp)
            columns.extend([label, f"{label}_count"])
        fh.write("Profile data\n")
        fh.write("------------\n")
        fh.write(",".join(columns) + "\n")

        obs_center, obs_count = obs_data[panel_name]
        for idx, height_km in enumerate(y_km):
            row: list[object] = [height_km, obs_center[idx], obs_count[idx]]
            for exp in experiments:
                center, count = model_data[exp][panel_name]
                row.extend([center[idx], count[idx]])
            fh.write(",".join(_fmt_txt(value) for value in row) + "\n")

    return txt_path


def _read_obs_profiles(model_ds: xr.Dataset, height_grid_m: np.ndarray) -> dict[str, list[np.ndarray]]:
    sonde_files = [str(item) for item in model_ds["sonde_file"].values]
    return _read_obs_profiles_from_files(sonde_files, height_grid_m)


def _read_obs_profiles_from_files(
    sonde_files: Sequence[str | Path],
    height_grid_m: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    profiles = {name: [] for name in PANEL_VARS}
    for sonde_file in sonde_files:
        with xr.open_dataset(sonde_file, engine="scipy", decode_times=False) as ds:
            height = _clean(ds["alt"].values)
            for panel_name, spec in PANEL_VARS.items():
                profiles[panel_name].append(
                    _interp_profile(height, ds[spec["obs_name"]].values, height_grid_m)
                )
    return profiles


def _read_model_profiles(
    ds: xr.Dataset,
    height_grid_m: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    height = np.asarray(ds["model_height"].values, dtype=np.float64)
    profiles = {name: [] for name in PANEL_VARS}
    for launch_idx in range(ds.sizes["launch"]):
        z = height[launch_idx, :]
        for panel_name, spec in PANEL_VARS.items():
            values = np.asarray(ds[spec["model_name"]].isel(launch=launch_idx).values)
            profiles[panel_name].append(_interp_profile(z, values, height_grid_m))
    return profiles


def plot_profiles(args: argparse.Namespace) -> list[Path]:
    alaro_root = Path(args.alaro_root)
    experiments = _resolve_experiments(args.experiments)
    panel_names = _resolve_panel_names(getattr(args, "variables", None))
    height_grid_m = np.arange(0.0, args.max_height_m + args.height_step_m, args.height_step_m)
    output_dir = Path(args.output_dir)
    model_source = getattr(args, "model_source", "point")

    model_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    obs_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    if model_source == "masked-area":
        sonde_root = Path(getattr(args, "sonde_root", DEFAULT_SONDE_ROOT))
        launches = discover_sonde_launches(sonde_root)
        if not launches:
            raise ValueError(f"No radiosonde launches found in {sonde_root}")

        obs_profiles = _read_obs_profiles_from_files(
            [launch.sonde_file for launch in launches], height_grid_m
        )
        obs_data = {
            name: _aggregate(values, args.stat, args.min_samples)
            for name, values in obs_profiles.items()
        }

        for exp in experiments:
            model_profiles = _read_area_model_profiles(
                alaro_root,
                exp,
                launches,
                height_grid_m,
                masked_subdir=getattr(args, "masked_netcdf_subdir", "masked-netcdf"),
                workers=getattr(args, "workers", 1),
                progress_every=getattr(args, "progress_every", 100),
            )
            model_data[exp] = {
                name: _aggregate(values, args.stat, args.min_samples)
                for name, values in model_profiles.items()
            }
    elif model_source == "point":
        for exp in experiments:
            model_file = (
                alaro_root
                / exp
                / args.input_subdir
                / f"{exp}_radiosonde_matched_profiles.nc"
            )
            with xr.open_dataset(model_file) as ds:
                if "model_height" not in ds:
                    raise ValueError(
                        f"{model_file} has no model_height variable. "
                        "Rerun alaro-radiosonde-profiles after the height extraction update."
                    )
                model_profiles = _read_model_profiles(ds, height_grid_m)
                model_data[exp] = {
                    name: _aggregate(values, args.stat, args.min_samples)
                    for name, values in model_profiles.items()
                }
                if obs_data is None:
                    obs_profiles = _read_obs_profiles(ds, height_grid_m)
                    obs_data = {
                        name: _aggregate(values, args.stat, args.min_samples)
                        for name, values in obs_profiles.items()
                    }
    else:
        raise ValueError(f"Unsupported model_source={model_source!r}")

    if obs_data is None:
        raise ValueError("No observation profiles were read.")

    y_km = height_grid_m / 1000.0
    obs_freezing_km = _first_zero_crossing_height_km(obs_data["tdry"][0], y_km)
    model_freezing_km = {
        exp: _first_zero_crossing_height_km(model_data[exp]["tdry"][0], y_km)
        for exp in experiments
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for panel_name in panel_names:
        spec = PANEL_VARS[panel_name]
        fig, ax = plt.subplots(figsize=(5.2, 6.5), constrained_layout=True)
        obs_center, _ = obs_data[panel_name]
        ax.plot(obs_center, y_km, color="black", lw=2.4, label="ARM sonde")
        _draw_freezing_level(
            ax,
            obs_freezing_km,
            color="black",
            label="ARM",
            marker="o",
            mark_at_zero_c=panel_name == "tdry",
        )
        if panel_name == "tdry":
            ax.axvline(0.0, color="0.25", lw=1.2, ls="--", label="0 degC")

        for exp in experiments:
            center, _ = model_data[exp][panel_name]
            color = EXPERIMENT_COLORS.get(exp, "0.5")
            ax.plot(
                center,
                y_km,
                color=color,
                lw=2.0,
                label=EXPERIMENT_LABELS.get(exp, exp),
            )
            _draw_freezing_level(
                ax,
                model_freezing_km[exp],
                color=color,
                label=EXPERIMENT_LABELS.get(exp, exp),
                marker="s",
                mark_at_zero_c=panel_name == "tdry",
            )

        if not getattr(args, "no_title", False):
            title_suffix = " (radar-mask area mean)" if model_source == "masked-area" else ""
            ax.set_title(f"{args.stat.title()} {spec['title']} profile{title_suffix}", fontsize=13)
        ax.set_xlabel(spec["label"], fontsize=11)
        ax.set_ylabel("Height above mean sea level (km)", fontsize=11)
        ax.set_ylim(0.0, args.max_height_m / 1000.0)
        ax.grid(True, color="0.88", linewidth=0.8)
        if spec["xlim"] is not None:
            ax.set_xlim(*spec["xlim"])
        ax.legend(loc="best", frameon=False)

        stem = Path(str(spec["filename"])).stem
        if model_source == "masked-area":
            stem = stem.replace("radiosonde_", "radiosonde_area_mean_", 1)
        output = output_dir / f"{stem}_{args.stat}_height.png"
        fig.savefig(output, dpi=args.dpi)
        plt.close(fig)
        outputs.append(output)
        if getattr(args, "write_txt", False):
            txt_path = _write_profile_txt(
                output,
                panel_name=panel_name,
                spec=spec,
                y_km=y_km,
                obs_data=obs_data,
                model_data=model_data,
                experiments=experiments,
                obs_freezing_km=obs_freezing_km,
                model_freezing_km=model_freezing_km,
                stat=args.stat,
                model_source=model_source,
            )
            print(f"Wrote {txt_path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot matched ARM sonde and ALARO profiles for tdry, dewpoint, and RH."
    )
    parser.add_argument("--alaro-root", default=str(DEFAULT_ALARO_ROOT))
    parser.add_argument("--input-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--model-source",
        choices=["point", "masked-area"],
        default="point",
        help="Use nearest-neighbor extracted profiles or radar-masked domain area averages.",
    )
    parser.add_argument(
        "--sonde-root",
        default=str(DEFAULT_SONDE_ROOT),
        help="ARM sonde directory used by --model-source masked-area.",
    )
    parser.add_argument(
        "--masked-netcdf-subdir",
        default="masked-netcdf",
        help="Experiment subdirectory containing radar-masked NetCDF files.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["control", "graupel", "2mom"],
        help="Experiments or aliases to plot: c1m/control g1m/graupel g2m/2mom.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=list(PANEL_VARS),
        default=None,
        help="Subset of variables to plot: tdry dewpoint rh.",
    )
    parser.add_argument("--stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--max-height-m", type=float, default=20000.0)
    parser.add_argument("--height-step-m", type=float, default=250.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel readers for --model-source masked-area.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress print interval for --model-source masked-area.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw a title on the plot.",
    )
    parser.add_argument(
        "--write-txt",
        action="store_true",
        help="Write the plotted profile data to output-dir/data_txt with matching stems.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for separate variable figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = plot_profiles(args)
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
