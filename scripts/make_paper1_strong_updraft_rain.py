#!/usr/bin/env python3
"""Paper1 hydrometeor profile sampled only inside strong updrafts.

This is configured as the rain-only version by default.  A thin wrapper can
override the module-level profile constants to render graupel with the same
strong-updraft and isotherm logic.  The independent hourly files +0000 through
+0023 are used by default; +0024 is excluded because it duplicates the next
day's +0000 valid time for instantaneous fields.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS


DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")
H_BINS_KM = np.linspace(0.0, 20.0, 101)
TEMP_VAR = "TEMPERATURE"
PROFILE_VAR = "RAIN"
PROFILE_LABEL = "Rain"
PROFILE_SLUG = "rain"
PROFILE_FOLDER_PREFIX = "06"
PROFILE_COLOR = "#d62728"
PROFILE_GENERIC_ID = "qr, specific humidity of rain (kg/kg)"
PROFILE_XMIN_KG_KG = 1.0e-9
PROFILE_XMAX_KG_KG = 5.0e-4
STEP_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)*\.nc$")
ISOTHERMS_C = (0.0, -10.0, -20.0)
ISOTHERM_STYLE = {
    0.0: {"color": "#111111", "linestyle": "--", "linewidth": 1.1},
    -10.0: {"color": "#555555", "linestyle": "-.", "linewidth": 1.0},
    -20.0: {"color": "#888888", "linestyle": ":", "linewidth": 1.2},
}


def default_output_dir() -> Path:
    return PAPER_ROOT / f"{PROFILE_FOLDER_PREFIX}_strong_updraft_{PROFILE_SLUG}_profile"


def default_cache_dir() -> Path:
    return Path(
        "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/"
        f"paper1_strong_updraft_{PROFILE_SLUG}_profile"
    )


def figure_name() -> str:
    return f"strong_updraft_{PROFILE_SLUG}_vertical_profiles_450dpi.png"


def text_name() -> str:
    return f"strong_updraft_{PROFILE_SLUG}_vertical_profiles_data.txt"


@dataclass(frozen=True)
class RainProfile:
    experiment: str
    h_bins_km: np.ndarray
    rain_kg_kg: np.ndarray
    rain_count: np.ndarray
    temperature_k: np.ndarray
    temperature_count: np.ndarray
    strong_sample_count: np.ndarray
    isotherm_heights_km: dict[float, float]
    n_days_seen: int
    n_files_seen: int
    n_files_used: int
    n_files_with_strong_updraft: int
    missing_files: int
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Build {PROFILE_SLUG}-only strong-updraft profiles with isotherm overlays."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--min-lead-hour", type=int, default=0)
    parser.add_argument(
        "--max-lead-hour",
        type=int,
        default=23,
        help="Maximum forecast lead hour to include. Default 23 avoids duplicate +0024.",
    )
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-updraft-mesh-frac", type=float, default=0.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-height-km", type=float, default=20.0)
    parser.add_argument(
        "--xmin-kg-kg",
        "--rain-xmin-kg-kg",
        dest="xmin_kg_kg",
        type=float,
        default=PROFILE_XMIN_KG_KG,
    )
    parser.add_argument(
        "--xmax-kg-kg",
        "--rain-xmax-kg-kg",
        dest="xmax_kg_kg",
        type=float,
        default=PROFILE_XMAX_KG_KG,
    )
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cache_path(args: argparse.Namespace, experiment: str) -> Path:
    day_tag = "all-days" if args.max_days is None else f"{args.max_days}days"
    lead_tag = f"lead{args.min_lead_hour}_to{args.max_lead_hour}"
    return (
        args.cache_dir
        / f"{experiment}_strong_updraft_{PROFILE_SLUG}_omega{slug_float(args.min_updraft_pa_s)}"
        f"_mesh{slug_float(args.min_updraft_mesh_frac)}_{lead_tag}_{day_tag}.npz"
    )


def list_days(exp_dir: Path) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / "UD_OMEGA"
    if not var_dir.exists():
        return []
    return sorted(p for p in var_dir.iterdir() if p.is_dir() and p.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int, max_lead_hour: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(day_dir.glob("*.nc")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        lead = int(match.group(1))
        if min_lead_hour <= lead <= max_lead_hour:
            out.append(path)
    return out


def read_field(path: Path, variable: str) -> np.ndarray:
    with Dataset(path) as ds:
        var_name = variable if variable in ds.variables else None
        if var_name is None:
            data_vars = [
                name
                for name, var in ds.variables.items()
                if var.dimensions and name not in {"time", "level", "x", "y", "lat", "lon"}
            ]
            if not data_vars:
                raise KeyError(f"No data variable found in {path}")
            var_name = data_vars[0]
        raw = ds.variables[var_name][:]

    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected (level, y, x) after squeeze for {path}, got {arr.shape}")
    return arr


def _empty_accumulators() -> dict[str, Any]:
    n_height = len(H_BINS_KM) - 1
    return {
        "rain_sum": np.zeros(n_height, dtype=np.float64),
        "rain_count": np.zeros(n_height, dtype=np.int64),
        "temperature_sum": np.zeros(n_height, dtype=np.float64),
        "temperature_count": np.zeros(n_height, dtype=np.int64),
        "strong_sample_count": np.zeros(n_height, dtype=np.int64),
        "n_files_seen": 0,
        "n_files_used": 0,
        "n_files_with_strong_updraft": 0,
        "missing_files": 0,
    }


def process_day(task: tuple[Path, Path, int, int, float, float]) -> dict[str, Any]:
    exp_dir, day_dir, min_lead_hour, max_lead_hour, min_updraft_pa_s, min_mesh_frac = task
    data_dir = exp_dir / "masked-netcdf"
    out = _empty_accumulators()

    for omega_path in list_steps(day_dir, min_lead_hour, max_lead_hour):
        day_name = omega_path.parent.name
        file_name = omega_path.name
        out["n_files_seen"] += 1

        mesh_path = data_dir / "UD_MESH_FRAC" / day_name / file_name
        height_path = data_dir / "GEOPOTENTIEL" / day_name / file_name
        rain_path = data_dir / PROFILE_VAR / day_name / file_name
        temp_path = data_dir / TEMP_VAR / day_name / file_name
        if not all(path.exists() for path in (mesh_path, height_path, rain_path, temp_path)):
            out["missing_files"] += 1
            continue

        try:
            omega = read_field(omega_path, "UD_OMEGA")
            mesh = read_field(mesh_path, "UD_MESH_FRAC")
            height_km = read_field(height_path, "GEOPOTENTIEL") / 1000.0
            rain = read_field(rain_path, PROFILE_VAR)
            temperature = read_field(temp_path, TEMP_VAR)
        except Exception:
            out["missing_files"] += 1
            continue

        strong_mask = (
            np.isfinite(omega)
            & np.isfinite(mesh)
            & np.isfinite(height_km)
            & (mesh > min_mesh_frac)
            & ((-omega) >= min_updraft_pa_s)
        )
        out["n_files_used"] += 1
        if not np.any(strong_mask):
            continue

        height_flat = height_km[strong_mask]
        h_idx = np.digitize(height_flat, H_BINS_KM) - 1
        valid_height = (h_idx >= 0) & (h_idx < len(H_BINS_KM) - 1)
        h_idx = h_idx[valid_height]
        if h_idx.size == 0:
            continue

        out["n_files_with_strong_updraft"] += 1
        np.add.at(out["strong_sample_count"], h_idx, 1)

        rain_values = rain[strong_mask][valid_height]
        rain_finite = np.isfinite(rain_values)
        if np.any(rain_finite):
            rain_values = np.maximum(rain_values[rain_finite], 0.0)
            np.add.at(out["rain_sum"], h_idx[rain_finite], rain_values)
            np.add.at(out["rain_count"], h_idx[rain_finite], 1)

        temp_values = temperature[strong_mask][valid_height]
        temp_finite = np.isfinite(temp_values)
        if np.any(temp_finite):
            np.add.at(out["temperature_sum"], h_idx[temp_finite], temp_values[temp_finite])
            np.add.at(out["temperature_count"], h_idx[temp_finite], 1)

    return out


def combine_day_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = _empty_accumulators()
    for result in results:
        for key in (
            "rain_sum",
            "rain_count",
            "temperature_sum",
            "temperature_count",
            "strong_sample_count",
        ):
            out[key] += result[key]
        for key in (
            "n_files_seen",
            "n_files_used",
            "n_files_with_strong_updraft",
            "missing_files",
        ):
            out[key] += result[key]
    return out


def mean_from_sum_count(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(sums.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def isotherm_height_km(
    height_km: np.ndarray,
    temperature_k: np.ndarray,
    target_c: float,
) -> float:
    temperature_c = temperature_k - 273.15
    finite = np.isfinite(height_km) & np.isfinite(temperature_c)
    z = height_km[finite]
    t = temperature_c[finite]
    if z.size < 2:
        return float("nan")
    order = np.argsort(z)
    z = z[order]
    t = t[order]
    diff = t - target_c
    for i in range(diff.size - 1):
        if diff[i] == 0:
            return float(z[i])
        if diff[i] * diff[i + 1] <= 0 and t[i] != t[i + 1]:
            frac = (target_c - t[i]) / (t[i + 1] - t[i])
            return float(z[i] + frac * (z[i + 1] - z[i]))
    return float("nan")


def build_profile(args: argparse.Namespace, experiment: str) -> RainProfile:
    exp_dir = args.data_root / experiment
    days = list_days(exp_dir)
    if args.max_days is not None:
        days = days[: args.max_days]
    if not days:
        raise RuntimeError(f"No day directories found for {experiment}: {exp_dir}")

    print(
        f"[{experiment}] processing {len(days)} days with {args.workers} workers; "
        f"lead {args.min_lead_hour}..{args.max_lead_hour}; "
        f"-UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s",
        flush=True,
    )
    tasks = [
        (
            exp_dir,
            day,
            args.min_lead_hour,
            args.max_lead_hour,
            args.min_updraft_pa_s,
            args.min_updraft_mesh_frac,
        )
        for day in days
    ]

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            results.append(process_day(task))
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"[{experiment}] {idx}/{len(tasks)} days", flush=True)
    else:
        with get_context("fork").Pool(processes=args.workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(process_day, tasks), start=1):
                results.append(result)
                if idx % args.progress_every == 0 or idx == len(tasks):
                    used = sum(r["n_files_used"] for r in results)
                    strong = sum(r["n_files_with_strong_updraft"] for r in results)
                    print(
                        f"[{experiment}] {idx}/{len(tasks)} days; "
                        f"{used} files read; {strong} with strong updrafts",
                        flush=True,
                    )

    combined = combine_day_results(results)
    rain = mean_from_sum_count(combined["rain_sum"], combined["rain_count"])
    temperature = mean_from_sum_count(
        combined["temperature_sum"], combined["temperature_count"]
    )
    centers = 0.5 * (H_BINS_KM[:-1] + H_BINS_KM[1:])
    isotherms = {
        target: isotherm_height_km(centers, temperature, target) for target in ISOTHERMS_C
    }
    return RainProfile(
        experiment=experiment,
        h_bins_km=H_BINS_KM.copy(),
        rain_kg_kg=rain,
        rain_count=combined["rain_count"],
        temperature_k=temperature,
        temperature_count=combined["temperature_count"],
        strong_sample_count=combined["strong_sample_count"],
        isotherm_heights_km=isotherms,
        n_days_seen=len(days),
        n_files_seen=combined["n_files_seen"],
        n_files_used=combined["n_files_used"],
        n_files_with_strong_updraft=combined["n_files_with_strong_updraft"],
        missing_files=combined["missing_files"],
        source=str(exp_dir / "masked-netcdf"),
    )


def save_cache(path: Path, profile: RainProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(list(profile.isotherm_heights_km), dtype=np.float64)
    heights = np.asarray([profile.isotherm_heights_km[t] for t in targets], dtype=np.float64)
    np.savez_compressed(
        path,
        experiment=np.asarray(profile.experiment),
        h_bins_km=profile.h_bins_km,
        rain_kg_kg=profile.rain_kg_kg,
        rain_count=profile.rain_count,
        temperature_k=profile.temperature_k,
        temperature_count=profile.temperature_count,
        strong_sample_count=profile.strong_sample_count,
        isotherm_targets_c=targets,
        isotherm_heights_km=heights,
        n_days_seen=np.asarray(profile.n_days_seen, dtype=np.int64),
        n_files_seen=np.asarray(profile.n_files_seen, dtype=np.int64),
        n_files_used=np.asarray(profile.n_files_used, dtype=np.int64),
        n_files_with_strong_updraft=np.asarray(
            profile.n_files_with_strong_updraft, dtype=np.int64
        ),
        missing_files=np.asarray(profile.missing_files, dtype=np.int64),
        source=np.asarray(profile.source),
    )


def load_cache(path: Path) -> RainProfile:
    with np.load(path, allow_pickle=False) as data:
        targets = np.asarray(data["isotherm_targets_c"], dtype=np.float64)
        heights = np.asarray(data["isotherm_heights_km"], dtype=np.float64)
        return RainProfile(
            experiment=str(data["experiment"]),
            h_bins_km=np.asarray(data["h_bins_km"], dtype=np.float64),
            rain_kg_kg=np.asarray(data["rain_kg_kg"], dtype=np.float64),
            rain_count=np.asarray(data["rain_count"], dtype=np.int64),
            temperature_k=np.asarray(data["temperature_k"], dtype=np.float64),
            temperature_count=np.asarray(data["temperature_count"], dtype=np.int64),
            strong_sample_count=np.asarray(data["strong_sample_count"], dtype=np.int64),
            isotherm_heights_km={
                float(target): float(height) for target, height in zip(targets, heights)
            },
            n_days_seen=int(data["n_days_seen"]),
            n_files_seen=int(data["n_files_seen"]),
            n_files_used=int(data["n_files_used"]),
            n_files_with_strong_updraft=int(data["n_files_with_strong_updraft"]),
            missing_files=int(data["missing_files"]),
            source=str(data["source"]),
        )


def get_profile(args: argparse.Namespace, experiment: str) -> RainProfile:
    path = cache_path(args, experiment)
    if path.exists() and not args.recompute:
        print(f"[{experiment}] using cache {path}", flush=True)
        return load_cache(path)
    profile = build_profile(args, experiment)
    save_cache(path, profile)
    print(f"[{experiment}] saved cache {path}", flush=True)
    return profile


def plot_profiles(
    profiles: dict[str, RainProfile],
    experiments: list[str],
    output_path: Path,
    *,
    max_height_km: float,
    rain_xmin_kg_kg: float,
    rain_xmax_kg_kg: float,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, len(experiments), figsize=(9.8, 4.4), sharey=True)
    if len(experiments) == 1:
        axes = np.asarray([axes])

    legend_handles = []
    legend_labels = []
    for ax, experiment in zip(axes, experiments):
        result = profiles[experiment]
        centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
        height_ok = centers <= max_height_km
        rain_kg_kg = result.rain_kg_kg
        ok = (
            height_ok
            & np.isfinite(rain_kg_kg)
            & (rain_kg_kg > 0.0)
            & (result.rain_count > 0)
        )
        rain_line = ax.plot(
            rain_kg_kg[ok],
            centers[ok],
            color=PROFILE_COLOR,
            linewidth=2.0,
            label=f"{EXPERIMENT_LABELS[experiment]} {PROFILE_SLUG}",
        )[0]
        if experiment == experiments[0]:
            legend_handles.append(rain_line)
            legend_labels.append(PROFILE_LABEL)

        for target_c in ISOTHERMS_C:
            y = result.isotherm_heights_km.get(target_c, float("nan"))
            if not np.isfinite(y) or y < 0 or y > max_height_km:
                continue
            line = ax.axhline(y, **ISOTHERM_STYLE[target_c])
            label = f"{target_c:g} C"
            ax.text(
                rain_xmax_kg_kg * 0.93,
                y + 0.05,
                label,
                ha="right",
                va="bottom",
                fontsize=7,
                color=ISOTHERM_STYLE[target_c]["color"],
            )
            if experiment == experiments[0]:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(EXPERIMENT_LABELS[experiment], fontweight="normal", pad=6)
        ax.set_xlabel(f"{PROFILE_LABEL} mixing ratio (kg kg$^{{-1}}$)")
        ax.set_xscale("log")
        ax.set_xlim(rain_xmin_kg_kg, rain_xmax_kg_kg)
        ax.set_ylim(0.0, max_height_km)
        ax.grid(True, which="major", color="#d0d0d0", linewidth=0.55, alpha=0.7)
        ax.grid(True, which="minor", color="#e6e6e6", linewidth=0.35, alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(
            0.04,
            0.96,
            f"{result.n_files_with_strong_updraft:,}/{result.n_files_used:,} timesteps",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#333333",
        )

    axes[0].set_ylabel("Height (km)")
    fig.suptitle(
        f"Strong-updraft {PROFILE_SLUG} profiles",
        fontsize=11,
        fontweight="normal",
        y=0.99,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text_data(
    profiles: dict[str, RainProfile],
    experiments: list[str],
    output_path: Path,
    *,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Paper1 strong-updraft {PROFILE_SLUG} vertical profiles\n")
        f.write("#\n")
        f.write("# Strong updraft definition:\n")
        f.write(f"#   UD_MESH_FRAC > {args.min_updraft_mesh_frac:g}\n")
        f.write(f"#   -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s\n")
        f.write(f"#   forecast lead hours {args.min_lead_hour}..{args.max_lead_hour}\n")
        f.write("#   +0024 is excluded by default to avoid duplicate valid times.\n")
        f.write("#\n")
        f.write("# Units:\n")
        f.write(f"#   Raw FA {PROFILE_VAR} fields have no plain text unit string, but their\n")
        f.write(f"#   generic identifier maps to {PROFILE_GENERIC_ID}.\n")
        f.write("#   Converted NetCDF metadata stores this as units=1, i.e. dimensionless\n")
        f.write("#   kg/kg mixing ratio. The plot x-axis uses kg/kg directly.\n")
        f.write(
            f"#   Plot x-axis is logarithmic from {args.xmin_kg_kg:g} "
            f"to {args.xmax_kg_kg:g} kg/kg.\n"
        )
        f.write("#\n")
        f.write("# Isotherms:\n")
        f.write("#   Isotherm heights are interpolated from the strong-updraft-mean\n")
        f.write("#   TEMPERATURE profile in each experiment.\n")
        for experiment in experiments:
            result = profiles[experiment]
            parts = [
                f"{target:g}C={result.isotherm_heights_km.get(target, float('nan')):.3f} km"
                for target in ISOTHERMS_C
            ]
            f.write(
                f"#   {experiment} ({EXPERIMENT_LABELS[experiment]}): "
                + ", ".join(parts)
                + "\n"
            )
        f.write("#\n")
        f.write("# Summary by experiment:\n")
        for experiment in experiments:
            result = profiles[experiment]
            f.write(
                "#   "
                f"{experiment} ({EXPERIMENT_LABELS[experiment]}): "
                f"days={result.n_days_seen}, files_seen={result.n_files_seen}, "
                f"files_used={result.n_files_used}, "
                f"files_with_strong_updraft={result.n_files_with_strong_updraft}, "
                f"missing_or_failed_file_groups={result.missing_files}, "
                f"strong_gridpoint_samples={int(np.sum(result.strong_sample_count))}\n"
            )
        f.write("#\n")
        header = [
            "experiment",
            "experiment_label",
            "height_bin_left_km",
            "height_bin_right_km",
            "height_center_km",
            "strong_gridpoint_samples",
            f"mean_{PROFILE_VAR}_kg_kg-1",
            f"mean_{PROFILE_VAR}_g_kg-1",
            f"count_{PROFILE_VAR}",
            "mean_TEMPERATURE_K",
            "mean_TEMPERATURE_C",
            "count_TEMPERATURE",
        ]
        f.write("\t".join(header) + "\n")
        for experiment in experiments:
            result = profiles[experiment]
            centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
            for idx, center in enumerate(centers):
                rain_kg_kg = result.rain_kg_kg[idx]
                rain_g_kg = rain_kg_kg * 1000.0 if np.isfinite(rain_kg_kg) else np.nan
                temp_k = result.temperature_k[idx]
                temp_c = temp_k - 273.15 if np.isfinite(temp_k) else np.nan
                row = [
                    experiment,
                    EXPERIMENT_LABELS[experiment],
                    f"{result.h_bins_km[idx]:.3f}",
                    f"{result.h_bins_km[idx + 1]:.3f}",
                    f"{center:.3f}",
                    str(int(result.strong_sample_count[idx])),
                    f"{rain_kg_kg:.10e}" if np.isfinite(rain_kg_kg) else "nan",
                    f"{rain_g_kg:.10e}" if np.isfinite(rain_g_kg) else "nan",
                    str(int(result.rain_count[idx])),
                    f"{temp_k:.6f}" if np.isfinite(temp_k) else "nan",
                    f"{temp_c:.6f}" if np.isfinite(temp_c) else "nan",
                    str(int(result.temperature_count[idx])),
                ]
                f.write("\t".join(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    profiles = {experiment: get_profile(args, experiment) for experiment in args.experiments}
    fig_path = args.output_dir / figure_name()
    txt_path = args.output_dir / text_name()
    plot_profiles(
        profiles,
        args.experiments,
        fig_path,
        max_height_km=args.max_height_km,
        rain_xmin_kg_kg=args.xmin_kg_kg,
        rain_xmax_kg_kg=args.xmax_kg_kg,
        dpi=args.dpi,
    )
    write_text_data(profiles, args.experiments, txt_path, args=args)
    print(f"[done] figure: {fig_path}", flush=True)
    print(f"[done] data:   {txt_path}", flush=True)


if __name__ == "__main__":
    main()
