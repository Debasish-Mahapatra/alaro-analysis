#!/usr/bin/env python3
"""Sounding profiles against the ARM MAO radiosondes for the GoAmazon IOPs.

Model temperature, dewpoint and relative humidity at the grid point nearest the
median launch site, matched to every ARM radiosonde launched in the campaign
IOPs; each launch time is floored to its UTC hour, as in the paper's sounding
figure. Sonde and model profiles go onto one 250 m height grid, and the sonde's
time mask is applied to every model: at each height a launch counts only where
the sonde and all four models have a value, so every mean uses the same launches.

C1M comes from the continuous control run (FA); G1M, G2M and G2M-XCU from the IOP
warm-restart runs (masked NetCDF, which equals FA at the site). The figure goes
to figure 06 of the IOP series
(``iop-run-figures/06_sounding_profiles/sounding_profiles_<period>_450dpi.jpg``) for the
combined IOPs, IOP1 and IOP2.

A second figure per period splits the launches into day and night by their local
launch time (Amazon local time, UTC-4; day = 06:00-18:00 LT) and stacks the two, day
on top, with shared x-limits per column
(``sounding_profiles_day_night_<period>_450dpi.jpg``). ARM MAO launched five times a
day, near 01:30, 07:30, 10:30, 13:30 and 19:30 LT, so day holds the 07:30, 10:30 and
13:30 launches and night the 19:30 and 01:30 launches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.figio import DEFAULT_DPI, convert_to_jpg
from alaro_analysis.common.iop_settings import (
    CACHE_DIRS,
    CONTROL_RUN,
    FIGURES,
    GOAMAZON_IOPS,
    IOP_PERIODS,
    IOP_RUNS,
    figure_files,
    write_figure_index,
)
from alaro_analysis.common.parallel import resolve_workers
from alaro_analysis.common.radiosonde_settings import DEFAULT_SONDE_ROOT
from alaro_analysis.data.radiosondes import (
    MissingProfile,
    ProfileResult,
    SondeLaunch,
    _build_output_dataset,
    _extract_one_profile_worker,
    _write_missing_log,
    discover_sonde_launches,
    median_launch_site,
)
from alaro_analysis.plotting.radiosondes import (
    COMBINED_PANEL_XLABELS,
    _first_zero_crossing_height_km,
    _fmt_txt,
    _read_model_profiles,
    _read_obs_profiles_from_files,
    draw_combined_profiles,
    draw_delta_profiles,
    draw_profile_rows,
)

DEFAULT_WORK_DIR = CACHE_DIRS["soundings"]
FIGURE_NUMBER = 6
FIGURE_STEM = "sounding_profiles"
# Experiment -> IOP -> (run root, experiment directory, source read by data.radiosondes)
MODEL_RUNS = {
    "control": {iop: (CONTROL_RUN.parent, CONTROL_RUN.name, "fa") for iop in GOAMAZON_IOPS},
    "graupel": {iop: (IOP_RUNS / f"iop{iop}", "graupel", "netcdf") for iop in GOAMAZON_IOPS},
    "2mom": {iop: (IOP_RUNS / f"iop{iop}", "2mom", "netcdf") for iop in GOAMAZON_IOPS},
    "2mom-xcu": {iop: (IOP_RUNS / f"iop{iop}", "NO3M", "netcdf") for iop in GOAMAZON_IOPS},
}
PANELS = ("tdry", "dewpoint", "rh")
# As in the paper figure: model minus sonde up to 20 km in 250 m steps, >= 10 launches a height.
MAX_HEIGHT_M = 20000.0
HEIGHT_STEP_M = 250.0
MIN_SAMPLES = 10
# Day/night split on the local launch time (Amazon local time).
UTC_OFFSET_HOURS = -4
DAY_START_LT, DAY_END_LT = 6, 18
DAY_NIGHT_STEM = "sounding_profiles_day_night"
DELTA_STEM = "sounding_profiles_day_minus_night"
# Runs with the 3MT convection scheme; the shaded band is for their mean.
BAND_RUNS = ("control", "graupel", "2mom")
N_BOOT = 4000


def iop_launches(sonde_root: Path) -> dict[int, list[SondeLaunch]]:
    return {
        iop: discover_sonde_launches(sonde_root, start_date=start, end_date=end)
        for iop, (start, end) in GOAMAZON_IOPS.items()
    }


def cache_path(work: Path, iop: int, key: str) -> Path:
    return work / f"iop{iop}" / f"{EXPERIMENT_LABELS[key]}_sounding_profiles.nc"


def launch_key(launch: SondeLaunch) -> np.datetime64:
    return np.datetime64(launch.launch_time_utc.replace(tzinfo=None), "s")


def local_hour(launch: SondeLaunch) -> float:
    t = launch.launch_time_utc
    return (t.hour + t.minute / 60.0 + UTC_OFFSET_HOURS) % 24.0


def time_of_day(launch: SondeLaunch) -> str:
    """'day' when launched between DAY_START_LT and DAY_END_LT local time, else 'night'."""
    return "day" if DAY_START_LT <= local_hour(launch) < DAY_END_LT else "night"


def launch_times_lt(launches: list[SondeLaunch]) -> str:
    """Local launch hours of a subset in time order from 18 LT, e.g. '07, 10, 13'."""
    hours = [int(local_hour(x)) for x in launches]
    counts = {h: hours.count(h) for h in set(hours)}
    # Leave out the odd off-schedule launch (a handful per IOP).
    keep = [h for h, n in counts.items() if n >= 0.1 * max(counts.values())]
    return ", ".join(f"{h:02d}" for h in sorted(keep, key=lambda h: (h - 18) % 24))


def extract_profiles(
    run: tuple[Path, str, str],
    launches: list[SondeLaunch],
    site: tuple[float, float],
    workers: int,
    attempts: int,
) -> tuple[list[ProfileResult], list[MissingProfile]]:
    """Model profiles of one run at the launch hours, in launch order.

    The FA reader occasionally aborts a worker (glibc heap corruption). The broken
    pool fails its remaining launches, which the next attempt redoes.
    """
    root, experiment, source = run
    found: dict[str, ProfileResult] = {}
    missing: dict[str, MissingProfile] = {}
    pending = list(launches)
    for _ in range(attempts):
        crashed = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _extract_one_profile_worker,
                    index,
                    str(root),
                    experiment,
                    launch,
                    site[0],
                    site[1],
                    source,
                    None,
                ): launch
                for index, launch in enumerate(pending)
            }
            for future in as_completed(futures):
                launch = futures[future]
                try:
                    _, result, miss = future.result()
                except Exception:  # noqa: BLE001
                    crashed.append(launch)
                    continue
                if result is not None:
                    found[launch.launch_id] = result
                else:
                    missing[launch.launch_id] = miss
        if not crashed:
            break
        pending = crashed
    else:
        raise RuntimeError(
            f"{experiment}: {len(crashed)} launches failed after {attempts} attempts"
        )
    ordered = [found[x.launch_id] for x in launches if x.launch_id in found]
    return ordered, [missing[x.launch_id] for x in launches if x.launch_id in missing]


def build_cache(work, iop, key, launches, site, workers, attempts) -> Path:
    out = cache_path(work, iop, key)
    run = MODEL_RUNS[key][iop]
    results, missing = extract_profiles(run, launches, site, workers, attempts)
    if not results:
        raise RuntimeError(f"No {key} profiles for IOP{iop} under {run[0] / run[1]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = _build_output_dataset(key, results, site_lat=site[0], site_lon=site[1], source=run[2])
    tmp = out.with_name(out.name + ".tmp")
    ds.to_netcdf(tmp, engine="netcdf4")
    ds.close()
    tmp.replace(out)
    _write_missing_log(out.with_name(f"{out.stem}_missing.csv"), missing)
    label = EXPERIMENT_LABELS[key]
    print(f"[IOP{iop} {label}] {len(results)} profiles, {len(missing)} missing", flush=True)
    return out


def matched_profiles(work: Path, iop: int, launches: list[SondeLaunch], heights_m) -> dict:
    """Sonde and model profiles on the height grid, at the launches every model has."""
    models: dict[str, dict[np.datetime64, dict[str, np.ndarray]]] = {}
    grid = {}
    for key in MODEL_RUNS:
        with xr.open_dataset(cache_path(work, iop, key)) as ds:
            times = np.asarray(ds["launch_time_utc"].values).astype("datetime64[s]")
            profiles = _read_model_profiles(ds, heights_m)
            grid = grid or dict(ds.attrs)
        models[key] = {t: {p: profiles[p][i] for p in PANELS} for i, t in enumerate(times)}
    common = [x for x in launches if all(launch_key(x) in models[k] for k in MODEL_RUNS)]
    obs = _read_obs_profiles_from_files([x.sonde_file for x in common], heights_m)
    return {
        "launches": len(launches),
        "common": len(common),
        "dates": [x.launch_time_utc.date() for x in common],
        "grid": grid,
        "obs": {p: np.stack(obs[p]) for p in PANELS},
        "models": {
            k: {p: np.stack([models[k][launch_key(x)][p] for x in common]) for p in PANELS}
            for k in MODEL_RUNS
        },
    }


def masked_means(obs: np.ndarray, models: dict[str, np.ndarray], min_samples: int):
    """Mean (launch, height) profiles over the samples where the sonde and every model exist.

    Returns ``(sonde mean, counts)`` and ``{model: (mean, counts)}``; every series
    shares the counts, and heights with fewer than ``min_samples`` launches are NaN.
    """
    valid = np.isfinite(obs)
    for values in models.values():
        valid &= np.isfinite(values)
    counts = valid.sum(axis=0)

    def mean(values: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(valid, values, 0.0).sum(axis=0) / counts
        out[counts < min_samples] = np.nan
        return out

    return (mean(obs), counts), {key: (mean(values), counts) for key, values in models.items()}


def period_data(parts: list[dict], heights_m: np.ndarray) -> dict:
    """Profile means of one period in the layout ``draw_combined_profiles`` reads."""
    obs_data, model_data = {}, {key: {} for key in MODEL_RUNS}
    for panel in PANELS:
        obs = np.concatenate([part["obs"][panel] for part in parts])
        models = {
            key: np.concatenate([part["models"][key][panel] for part in parts])
            for key in MODEL_RUNS
        }
        obs_data[panel], means = masked_means(obs, models, MIN_SAMPLES)
        for key in MODEL_RUNS:
            model_data[key][panel] = means[key]
    y_km = heights_m / 1000.0
    return {
        "experiments": list(MODEL_RUNS),
        "obs_data": obs_data,
        "model_data": model_data,
        "y_km": y_km,
        "obs_freezing_km": _first_zero_crossing_height_km(obs_data["tdry"][0], y_km),
        "model_freezing_km": {
            key: _first_zero_crossing_height_km(model_data[key]["tdry"][0], y_km)
            for key in MODEL_RUNS
        },
        "model_source": "point",
    }


def header_lines(title: str, parts: dict[int, dict], subset: str = "") -> list[str]:
    iops = tuple(parts)
    grid = parts[iops[0]]["grid"]
    names = " and ".join(f"IOP{iop}" for iop in iops)
    combined = " combined" if len(iops) > 1 else ""
    lines = [
        title,
        "=" * len(title),
        f"Period: GoAmazon2014/5 {names}{combined} (Martin et al. 2016, ACP 16, 4785).",
    ]
    for iop in iops:
        start, end = GOAMAZON_IOPS[iop]
        lines.append(
            f"IOP{iop}: {start} to {end} inclusive, {parts[iop]['launches']} ARM MAO radiosonde "
            f"launches{subset}, {parts[iop]['common']} of them with all four models."
        )
    lines += [
        "Time mask: the models are sampled only at the radiosonde launch times, each floored to "
        "its UTC hour. At each height a launch counts only where the sonde and all four models "
        "have a value, so every mean uses the same launches (n_launches); heights with fewer "
        f"than {MIN_SAMPLES} launches are blank.",
        f"Site: median launch position ({grid['site_lat']:.4f}, {grid['site_lon']:.4f}); "
        f"nearest model grid point ({grid['model_grid_lat']:.4f}, {grid['model_grid_lon']:.4f}).",
        "C1M: continuous control run (FA). G1M, G2M, G2M-XCU: IOP warm-restart runs "
        "(masked NetCDF, identical to FA at the site grid point).",
        "Statistic: mean. Height above mean sea level (km). Plotted: model minus ARM sonde.",
    ]
    return lines


def profile_lines(data: dict) -> list[str]:
    labels = [EXPERIMENT_LABELS[key] for key in MODEL_RUNS]
    lines = [
        "",
        "0 degC height (km)",
        "------------------",
        "series,z_freeze_km",
        f"ARM sonde,{_fmt_txt(data['obs_freezing_km'])}",
    ]
    lines += [f"{EXPERIMENT_LABELS[k]},{_fmt_txt(data['model_freezing_km'][k])}" for k in MODEL_RUNS]
    for panel in PANELS:
        heading = COMBINED_PANEL_XLABELS[panel]
        columns = ["height_km", "n_launches", "ARM_sonde", *labels, *[f"{x}-ARM" for x in labels]]
        lines += ["", heading, "-" * len(heading), ",".join(columns)]
        obs, counts = data["obs_data"][panel]
        for i, height in enumerate(data["y_km"]):
            means = [data["model_data"][key][panel][0][i] for key in MODEL_RUNS]
            row = [height, counts[i], obs[i], *means, *[m - obs[i] for m in means]]
            lines.append(",".join(_fmt_txt(value) for value in row))
    return lines


def day_minus_night(day_parts: list[dict], night_parts: list[dict], heights_m, seed: int = 0):
    """Day-minus-night change of each model's bias per height, with a bootstrap band.

    For each half, the bias of a launch counts at a height only where the sonde and
    all four models have a value (the figures' joint mask), and heights with fewer
    than MIN_SAMPLES launches in either half are blank. The band is the 95 % range
    of the change of the mean bias of BAND_RUNS when whole days are resampled, so
    same-day launches stay together.
    """
    rng = np.random.default_rng(seed)
    dates = sorted({d for part in day_parts + night_parts for d in part["dates"]})
    index = {d: i for i, d in enumerate(dates)}
    out = {"models": {k: {} for k in MODEL_RUNS}, "band": {}, "counts": {}}
    for panel in PANELS:
        halves = {}
        for half, parts in (("day", day_parts), ("night", night_parts)):
            obs = np.concatenate([part["obs"][panel] for part in parts])
            models = {
                k: np.concatenate([part["models"][k][panel] for part in parts]) for k in MODEL_RUNS
            }
            valid = np.isfinite(obs)
            for values in models.values():
                valid &= np.isfinite(values)
            bias = {k: np.where(valid, models[k] - obs, np.nan) for k in MODEL_RUNS}
            day_of = np.array([index[d] for part in parts for d in part["dates"]])
            halves[half] = (bias, valid, day_of)
        counts = {half: halves[half][1].sum(axis=0) for half in halves}
        enough = (counts["day"] >= MIN_SAMPLES) & (counts["night"] >= MIN_SAMPLES)
        out["counts"][panel] = counts
        with np.errstate(invalid="ignore"):
            for k in MODEL_RUNS:
                change = np.nanmean(halves["day"][0][k], axis=0) - np.nanmean(
                    halves["night"][0][k], axis=0
                )
                out["models"][k][panel] = np.where(enough, change, np.nan)

        # Per-day sums of the BAND_RUNS mean bias, so a resample is one matrix product.
        sums, numbers = {}, {}
        for half, (bias, valid, day_of) in halves.items():
            mean_bias = np.mean([bias[k] for k in BAND_RUNS], axis=0)
            sums[half] = np.zeros((len(dates), mean_bias.shape[1]))
            numbers[half] = np.zeros((len(dates), mean_bias.shape[1]))
            np.add.at(sums[half], day_of, np.where(valid, mean_bias, 0.0))
            np.add.at(numbers[half], day_of, valid.astype(float))
        weights = np.stack(
            [np.bincount(rng.integers(0, len(dates), len(dates)), minlength=len(dates))
             for _ in range(N_BOOT)]
        ).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (weights @ sums["day"]) / (weights @ numbers["day"]) - (
                weights @ sums["night"]
            ) / (weights @ numbers["night"])
            low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        out["band"][panel] = (np.where(enough, low, np.nan), np.where(enough, high, np.nan))
    return out


def write_delta_txt(txt: Path, delta: dict, parts: dict[int, dict], y_km) -> None:
    labels = [EXPERIMENT_LABELS[key] for key in MODEL_RUNS]
    band = "+".join(EXPERIMENT_LABELS[k] for k in BAND_RUNS)
    lines = header_lines("Sounding bias, day minus night", parts)
    lines += [
        f"Launches split on their local launch time (UTC{UTC_OFFSET_HOURS:+d}): day = "
        f"{DAY_START_LT:02d}:00-{DAY_END_LT:02d}:00 LT, night = the rest.",
        "Plotted: (model - ARM sonde) by day minus (model - ARM sonde) at night, per height =",
        "the model's day-night contrast minus the sonde's. 0 = the day-night change is right.",
        f"Band: 95 % range of the same quantity for the mean of {band}, from {N_BOOT} resamples",
        "of whole days (launches of one day stay together). Where the band excludes 0 the",
        "change is not launch-to-launch noise.",
    ]
    for panel in PANELS:
        heading = f"Day - night of {COMBINED_PANEL_XLABELS[panel]}"
        columns = ["height_km", "n_day", "n_night", *labels, f"mean_{band}_low95", f"mean_{band}_high95"]
        lines += ["", heading, "-" * len(heading), ",".join(columns)]
        low, high = delta["band"][panel]
        counts = delta["counts"][panel]
        for i, height in enumerate(y_km):
            row = [height, counts["day"][i], counts["night"][i],
                   *[delta["models"][k][panel][i] for k in MODEL_RUNS], low[i], high[i]]
            lines.append(",".join(_fmt_txt(value) for value in row))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_txt(txt: Path, data: dict, parts: dict[int, dict]) -> None:
    lines = header_lines("Sounding profiles", parts) + profile_lines(data)
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_day_night_txt(txt: Path, rows: dict[str, tuple[dict, dict[int, dict]]]) -> None:
    """Both halves of the day/night figure in one file, day first."""
    lines = [
        "Sounding profiles by time of day",
        "================================",
        f"Launches split on their local launch time (UTC{UTC_OFFSET_HOURS:+d}): day = "
        f"{DAY_START_LT:02d}:00-{DAY_END_LT:02d}:00 LT, night = the rest.",
    ]
    for label, (data, parts) in rows.items():
        heading = f"{label.upper()}"
        lines += ["", "#" * 60, *header_lines(heading, parts, subset=f" in the {label}")]
        lines += profile_lines(data)
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_outputs(root: Path, period: str) -> tuple[Path, Path, Path]:
    """Temporary PNG render, final JPG, and data text of the figure."""
    return figure_files(root, FIGURE_NUMBER, FIGURE_STEM, period)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--output-dir", type=Path, default=FIGURES)
    p.add_argument("--sonde-root", type=Path, default=DEFAULT_SONDE_ROOT)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--recompute", action="store_true", help="Re-extract cached model profiles.")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parse_configured_args(p, "make_iop_sounding_profiles", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    launches = iop_launches(args.sonde_root)
    site = median_launch_site([x for group in launches.values() for x in group])
    workers = resolve_workers(args.workers)
    for iop, group in launches.items():
        print(f"IOP{iop}: {len(group)} radiosonde launches", flush=True)
        for key in MODEL_RUNS:
            if args.recompute or not cache_path(args.work_dir, iop, key).exists():
                build_cache(args.work_dir, iop, key, group, site, workers, args.attempts)

    heights_m = np.arange(0.0, MAX_HEIGHT_M + HEIGHT_STEP_M, HEIGHT_STEP_M)
    parts = {
        iop: matched_profiles(args.work_dir, iop, group, heights_m)
        for iop, group in launches.items()
    }
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = period_data([parts[iop] for iop in iops], heights_m)
        png, jpg, txt = figure_outputs(args.output_dir, key)
        draw_combined_profiles(
            data,
            png,
            panel_names=PANELS,
            diff_vars=PANELS,
            max_height_m=MAX_HEIGHT_M,
            dpi=args.dpi,
        )
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_txt(txt, data, {iop: parts[iop] for iop in iops})
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)

    # Day and night, stacked.
    split_parts = {
        half: {
            iop: matched_profiles(
                args.work_dir, iop, [x for x in group if time_of_day(x) == half], heights_m
            )
            for iop, group in launches.items()
        }
        for half in ("day", "night")
    }
    for key in args.periods:
        iops = IOP_PERIODS[key]
        rows, row_data = {}, {}
        for half in ("day", "night"):
            subset = [x for iop in iops for x in launches[iop] if time_of_day(x) == half]
            data = period_data([split_parts[half][iop] for iop in iops], heights_m)
            n = sum(split_parts[half][iop]["common"] for iop in iops)
            rows[f"{half.capitalize()}: launches at {launch_times_lt(subset)} LT, n = {n}"] = data
            row_data[half] = (data, {iop: split_parts[half][iop] for iop in iops})
        png, jpg, txt = figure_files(args.output_dir, FIGURE_NUMBER, DAY_NIGHT_STEM, key)
        draw_profile_rows(
            rows,
            png,
            panel_names=PANELS,
            diff_vars=PANELS,
            max_height_m=MAX_HEIGHT_M,
            dpi=args.dpi,
        )
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_day_night_txt(txt, row_data)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)

        # Delta of the delta: day minus night of each model's bias.
        delta = day_minus_night(
            [split_parts["day"][iop] for iop in iops],
            [split_parts["night"][iop] for iop in iops],
            heights_m,
        )
        all_launches = period_data([parts[iop] for iop in iops], heights_m)
        delta.update(
            y_km=heights_m / 1000.0,
            experiments=list(MODEL_RUNS),
            band_label="95 % range, mean of "
            + ", ".join(EXPERIMENT_LABELS[k] for k in BAND_RUNS),
            freezing_km=all_launches["obs_freezing_km"],
        )
        png, jpg, txt = figure_files(args.output_dir, FIGURE_NUMBER, DELTA_STEM, key)
        draw_delta_profiles(delta, png, panel_names=PANELS, max_height_m=MAX_HEIGHT_M, dpi=args.dpi)
        convert_to_jpg(png, jpg, dpi=args.dpi)
        png.unlink()
        write_delta_txt(txt, delta, {iop: parts[iop] for iop in iops}, heights_m / 1000.0)
        print(f"[saved] {jpg}\n[saved] {txt}", flush=True)
    write_figure_index(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
