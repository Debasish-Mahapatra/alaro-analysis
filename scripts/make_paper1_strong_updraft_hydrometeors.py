#!/usr/bin/env python3
"""Paper1 hydrometeor profiles sampled only inside strong updrafts.

The updraft filter is intentionally explicit here:

    strong updraft = finite UD_OMEGA/UD_MESH_FRAC, UD_MESH_FRAC > 0,
                     and -UD_OMEGA >= 10 Pa/s

The existing updraft-hydrometeor cache bins intensity as abs(UD_OMEGA), which is
useful for that diagnostic but not strict enough for a "strong updraft only"
paper plot.  This script therefore recomputes the 1-D height profiles directly
from the masked NetCDF files, then caches the compact profile products.
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
from matplotlib.ticker import ScalarFormatter
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS


DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
PAPER_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/paper1")
OUTPUT_DIR = PAPER_ROOT / "06_strong_updraft_hydrometeor_profiles"
CACHE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/"
    "paper1_strong_updraft_hydrometeor_profiles"
)

FIGURE_NAME = "strong_updraft_hydrometeor_vertical_profiles_450dpi.png"
TEXT_NAME = "strong_updraft_hydrometeor_vertical_profiles_data.txt"

HYDROMETEORS = ("LIQUID_WATER", "SOLID_WATER", "GRAUPEL", "SNOW", "RAIN")
HYDRO_LABELS = {
    "LIQUID_WATER": "Liquid water",
    "SOLID_WATER": "Cloud ice",
    "GRAUPEL": "Graupel",
    "SNOW": "Snow",
    "RAIN": "Rain",
}
HYDRO_COLORS = {
    "LIQUID_WATER": "#1f77b4",
    "SOLID_WATER": "#7b3294",
    "GRAUPEL": "#e66101",
    "SNOW": "#008c8c",
    "RAIN": "#d62728",
}
H_BINS_KM = np.linspace(0.0, 20.0, 101)
STEP_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)*\.nc$")


@dataclass(frozen=True)
class ExperimentProfile:
    experiment: str
    h_bins_km: np.ndarray
    profiles: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    strong_sample_count: np.ndarray
    n_days_seen: int
    n_files_seen: int
    n_files_used: int
    n_files_with_strong_updraft: int
    missing_files: int
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper1 vertical hydrometeor profiles conditioned on strong updrafts."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--min-lead-hour",
        type=int,
        default=0,
        help="Minimum forecast lead hour to include. Default 0 uses the entire dataset.",
    )
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-updraft-mesh-frac", type=float, default=0.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-height-km", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cache_path(args: argparse.Namespace, experiment: str) -> Path:
    day_tag = "all-days" if args.max_days is None else f"{args.max_days}days"
    return (
        args.cache_dir
        / f"{experiment}_strong_updraft_omega{slug_float(args.min_updraft_pa_s)}"
        f"_mesh{slug_float(args.min_updraft_mesh_frac)}"
        f"_lead{args.min_lead_hour}_{day_tag}.npz"
    )


def list_days(exp_dir: Path) -> list[Path]:
    var_dir = exp_dir / "masked-netcdf" / "UD_OMEGA"
    if not var_dir.exists():
        return []
    return sorted(p for p in var_dir.iterdir() if p.is_dir() and p.name.startswith("pf"))


def list_steps(day_dir: Path, min_lead_hour: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(day_dir.glob("*.nc")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        if int(match.group(1)) >= min_lead_hour:
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
        "sums": {h: np.zeros(n_height, dtype=np.float64) for h in HYDROMETEORS},
        "counts": {h: np.zeros(n_height, dtype=np.int64) for h in HYDROMETEORS},
        "strong_sample_count": np.zeros(n_height, dtype=np.int64),
        "n_files_seen": 0,
        "n_files_used": 0,
        "n_files_with_strong_updraft": 0,
        "missing_files": 0,
    }


def process_day(task: tuple[Path, Path, int, float, float]) -> dict[str, Any]:
    exp_dir, day_dir, min_lead_hour, min_updraft_pa_s, min_mesh_frac = task
    data_dir = exp_dir / "masked-netcdf"
    out = _empty_accumulators()

    for omega_path in list_steps(day_dir, min_lead_hour):
        day_name = omega_path.parent.name
        file_name = omega_path.name
        out["n_files_seen"] += 1

        mesh_path = data_dir / "UD_MESH_FRAC" / day_name / file_name
        height_path = data_dir / "GEOPOTENTIEL" / day_name / file_name
        needed = [mesh_path, height_path] + [
            data_dir / hydro / day_name / file_name for hydro in HYDROMETEORS
        ]
        if not all(path.exists() for path in needed):
            out["missing_files"] += 1
            continue

        try:
            omega = read_field(omega_path, "UD_OMEGA")
            mesh = read_field(mesh_path, "UD_MESH_FRAC")
            height_km = read_field(height_path, "GEOPOTENTIEL") / 1000.0
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
        if not np.any(strong_mask):
            out["n_files_used"] += 1
            continue

        height_flat = height_km[strong_mask]
        h_idx = np.digitize(height_flat, H_BINS_KM) - 1
        valid_height = (h_idx >= 0) & (h_idx < len(H_BINS_KM) - 1)
        h_idx = h_idx[valid_height]
        if h_idx.size == 0:
            out["n_files_used"] += 1
            continue

        np.add.at(out["strong_sample_count"], h_idx, 1)
        out["n_files_used"] += 1
        out["n_files_with_strong_updraft"] += 1

        for hydro in HYDROMETEORS:
            hydro_path = data_dir / hydro / day_name / file_name
            try:
                values = read_field(hydro_path, hydro)[strong_mask][valid_height]
            except Exception:
                out["missing_files"] += 1
                continue
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            values = np.maximum(values[finite], 0.0)
            np.add.at(out["sums"][hydro], h_idx[finite], values)
            np.add.at(out["counts"][hydro], h_idx[finite], 1)

    return out


def combine_day_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = _empty_accumulators()
    for result in results:
        for hydro in HYDROMETEORS:
            out["sums"][hydro] += result["sums"][hydro]
            out["counts"][hydro] += result["counts"][hydro]
        out["strong_sample_count"] += result["strong_sample_count"]
        for key in (
            "n_files_seen",
            "n_files_used",
            "n_files_with_strong_updraft",
            "missing_files",
        ):
            out[key] += result[key]
    return out


def compute_experiment(args: argparse.Namespace, experiment: str) -> ExperimentProfile:
    exp_dir = args.data_root / experiment
    days = list_days(exp_dir)
    if args.max_days is not None:
        days = days[: args.max_days]
    if not days:
        raise RuntimeError(f"No day directories found for {experiment}: {exp_dir}")

    print(
        f"[{experiment}] processing {len(days)} days with {args.workers} workers; "
        f"lead >= {args.min_lead_hour} h; -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s",
        flush=True,
    )
    tasks = [
        (exp_dir, day, args.min_lead_hour, args.min_updraft_pa_s, args.min_updraft_mesh_frac)
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
    profiles: dict[str, np.ndarray] = {}
    for hydro in HYDROMETEORS:
        counts = combined["counts"][hydro]
        sums = combined["sums"][hydro]
        profile = np.full(counts.shape, np.nan, dtype=np.float64)
        ok = counts > 0
        profile[ok] = sums[ok] / counts[ok]
        profiles[hydro] = profile

    return ExperimentProfile(
        experiment=experiment,
        h_bins_km=H_BINS_KM.copy(),
        profiles=profiles,
        counts=combined["counts"],
        strong_sample_count=combined["strong_sample_count"],
        n_days_seen=len(days),
        n_files_seen=combined["n_files_seen"],
        n_files_used=combined["n_files_used"],
        n_files_with_strong_updraft=combined["n_files_with_strong_updraft"],
        missing_files=combined["missing_files"],
        source=str(exp_dir / "masked-netcdf"),
    )


def save_cache(path: Path, profile: ExperimentProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "experiment": np.asarray(profile.experiment),
        "h_bins_km": profile.h_bins_km,
        "strong_sample_count": profile.strong_sample_count,
        "n_days_seen": np.asarray(profile.n_days_seen, dtype=np.int64),
        "n_files_seen": np.asarray(profile.n_files_seen, dtype=np.int64),
        "n_files_used": np.asarray(profile.n_files_used, dtype=np.int64),
        "n_files_with_strong_updraft": np.asarray(
            profile.n_files_with_strong_updraft, dtype=np.int64
        ),
        "missing_files": np.asarray(profile.missing_files, dtype=np.int64),
        "source": np.asarray(profile.source),
    }
    for hydro in HYDROMETEORS:
        save_dict[f"profile_{hydro}"] = profile.profiles[hydro]
        save_dict[f"count_{hydro}"] = profile.counts[hydro]
    np.savez_compressed(path, **save_dict)


def load_cache(path: Path) -> ExperimentProfile:
    with np.load(path, allow_pickle=False) as data:
        experiment = str(data["experiment"])
        profiles = {h: np.asarray(data[f"profile_{h}"], dtype=np.float64) for h in HYDROMETEORS}
        counts = {h: np.asarray(data[f"count_{h}"], dtype=np.int64) for h in HYDROMETEORS}
        return ExperimentProfile(
            experiment=experiment,
            h_bins_km=np.asarray(data["h_bins_km"], dtype=np.float64),
            profiles=profiles,
            counts=counts,
            strong_sample_count=np.asarray(data["strong_sample_count"], dtype=np.int64),
            n_days_seen=int(data["n_days_seen"]),
            n_files_seen=int(data["n_files_seen"]),
            n_files_used=int(data["n_files_used"]),
            n_files_with_strong_updraft=int(data["n_files_with_strong_updraft"]),
            missing_files=int(data["missing_files"]),
            source=str(data["source"]),
        )


def get_profile(args: argparse.Namespace, experiment: str) -> ExperimentProfile:
    path = cache_path(args, experiment)
    if path.exists() and not args.recompute:
        print(f"[{experiment}] using cache {path}", flush=True)
        return load_cache(path)
    profile = compute_experiment(args, experiment)
    save_cache(path, profile)
    print(f"[{experiment}] saved cache {path}", flush=True)
    return profile


def global_xmax(profiles: dict[str, ExperimentProfile], max_height_km: float) -> float:
    xmax = 0.0
    for result in profiles.values():
        centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
        height_ok = centers <= max_height_km
        for hydro in HYDROMETEORS:
            values = result.profiles[hydro][height_ok]
            if np.any(np.isfinite(values)):
                xmax = max(xmax, float(np.nanmax(values)))
    return max(xmax * 1.12, 1.0e-8)


def plot_profiles(
    profiles: dict[str, ExperimentProfile],
    experiments: list[str],
    output_path: Path,
    *,
    max_height_km: float,
    min_updraft_pa_s: float,
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
    xmax = global_xmax(profiles, max_height_km)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-4, -4))

    legend_handles = []
    legend_labels = []
    for ax, experiment in zip(axes, experiments):
        result = profiles[experiment]
        centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
        height_ok = centers <= max_height_km
        for hydro in HYDROMETEORS:
            values = result.profiles[hydro]
            ok = height_ok & np.isfinite(values) & (result.counts[hydro] > 0)
            line = ax.plot(
                values[ok],
                centers[ok],
                color=HYDRO_COLORS[hydro],
                linewidth=1.8,
                label=HYDRO_LABELS[hydro],
            )[0]
            if experiment == experiments[0]:
                legend_handles.append(line)
                legend_labels.append(HYDRO_LABELS[hydro])
        ax.set_title(EXPERIMENT_LABELS[experiment], fontweight="normal", pad=6)
        ax.set_xlabel("Mixing ratio (kg kg$^{-1}$)")
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(0.0, max_height_km)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, color="#d0d0d0", linewidth=0.55, alpha=0.7)
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
        f"Hydrometeors in strong updrafts (-UD_OMEGA >= {min_updraft_pa_s:g} Pa/s)",
        fontsize=11,
        fontweight="normal",
        y=0.99,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(HYDROMETEORS),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_text_data(
    profiles: dict[str, ExperimentProfile],
    experiments: list[str],
    output_path: Path,
    *,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Paper1 strong-updraft hydrometeor vertical profiles\n")
        f.write("#\n")
        f.write("# Strong updraft definition:\n")
        f.write(f"#   UD_MESH_FRAC > {args.min_updraft_mesh_frac:g}\n")
        f.write(f"#   -UD_OMEGA >= {args.min_updraft_pa_s:g} Pa/s\n")
        f.write(f"#   forecast lead hour >= {args.min_lead_hour}\n")
        f.write("#\n")
        f.write("# Data source:\n")
        f.write(f"#   {args.data_root}/<experiment>/masked-netcdf\n")
        f.write("#   Existing masked-NetCDF ROI is used; NO3M/G2M-XCU is not included.\n")
        f.write("#   Hydrometeor values are clipped at zero before averaging.\n")
        f.write("#   Height is GEOPOTENTIEL / 1000 and binned into 0.2 km bins.\n")
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
        ]
        header.extend(f"mean_{hydro}_kg_kg-1" for hydro in HYDROMETEORS)
        header.extend(f"count_{hydro}" for hydro in HYDROMETEORS)
        f.write("\t".join(header) + "\n")

        for experiment in experiments:
            result = profiles[experiment]
            centers = 0.5 * (result.h_bins_km[:-1] + result.h_bins_km[1:])
            for idx, center in enumerate(centers):
                row: list[str] = [
                    experiment,
                    EXPERIMENT_LABELS[experiment],
                    f"{result.h_bins_km[idx]:.3f}",
                    f"{result.h_bins_km[idx + 1]:.3f}",
                    f"{center:.3f}",
                    str(int(result.strong_sample_count[idx])),
                ]
                for hydro in HYDROMETEORS:
                    value = result.profiles[hydro][idx]
                    row.append(f"{value:.10e}" if np.isfinite(value) else "nan")
                for hydro in HYDROMETEORS:
                    row.append(str(int(result.counts[hydro][idx])))
                f.write("\t".join(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    profiles = {experiment: get_profile(args, experiment) for experiment in args.experiments}
    fig_path = args.output_dir / FIGURE_NAME
    txt_path = args.output_dir / TEXT_NAME
    plot_profiles(
        profiles,
        args.experiments,
        fig_path,
        max_height_km=args.max_height_km,
        min_updraft_pa_s=args.min_updraft_pa_s,
        dpi=args.dpi,
    )
    write_text_data(profiles, args.experiments, txt_path, args=args)
    print(f"[done] figure: {fig_path}", flush=True)
    print(f"[done] data:   {txt_path}", flush=True)


if __name__ == "__main__":
    main()
