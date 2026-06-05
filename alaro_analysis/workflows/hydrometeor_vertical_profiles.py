#!/usr/bin/env python3
"""Hydrometeor vertical profile comparison for C1M, G1M, and G2M.

This workflow remakes the old ``Hydrometeor_Vertical_Profiles_C1M_G1M_G2M``
notebook figure from the current ALARO masked-NetCDF data.  It uses cached
full-domain two-year profiles when available and can recompute directly from
the masked NetCDF files with ``--recompute``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS
from alaro_analysis.common.naming import safe_name
from alaro_analysis.data.dataset_io import read_vertical_profile
from alaro_analysis.data.discovery import collect_file_records


DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/hydrometeor_vertical_profiles"
)
DATA_TXT_DIR = OUTPUT_DIR / "data_txt"
CACHE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/hydrometeor_vertical_profiles"
)
LEGACY_CACHE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data/data.old.2"
)

FIGURE_NAME = "Hydrometeor_Vertical_Profiles_C1M_G1M_G2M.png"
TEXT_NAME = "Hydrometeor_Vertical_Profiles_C1M_G1M_G2M.txt"

SPECIES = ("LIQUID_WATER", "SOLID_WATER", "GRAUPEL", "SNOW", "RAIN")
SPECIES_LABEL = {
    "LIQUID_WATER": "Liquid water",
    "SOLID_WATER": "Solid water",
    "GRAUPEL": "Graupel",
    "SNOW": "Snow",
    "RAIN": "Rain",
}
SPECIES_COLOR = {
    "LIQUID_WATER": "blue",
    "SOLID_WATER": "purple",
    "GRAUPEL": "orange",
    "SNOW": "darkcyan",
    "RAIN": "red",
}
PANEL_LABELS = {"control": "(a)", "graupel": "(b)", "2mom": "(c)"}


@dataclass(frozen=True)
class ProfileData:
    profile: np.ndarray
    counts: np.ndarray
    n_files: int
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hydrometeor vertical profiles for C1M, G1M, and G2M."
    )
    add_config_argument(parser)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-txt-dir", type=Path, default=DATA_TXT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--legacy-cache-dir", type=Path, default=LEGACY_CACHE_DIR)
    parser.add_argument(
        "--spatial-tag",
        default="full-domain",
        help="Cache tag for the spatial averaging domain.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=-4,
        help="UTC to local-hour offset used only when recomputing cache files.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute profiles from masked NetCDF instead of using existing caches.",
    )
    parser.add_argument("--max-height-km", type=float, default=20.0)
    parser.add_argument("--xmin", type=float, default=-0.5e-5)
    parser.add_argument("--xmax", type=float, default=5.0e-5)
    parser.add_argument("--dpi", type=int, default=450)
    return parse_configured_args(parser, "hydrometeor_vertical_profiles", argv=argv)


def profile_cache_path(
    cache_dir: Path,
    variable: str,
    experiment: str,
    spatial_tag: str,
) -> Path:
    return (
        cache_dir
        / safe_name(variable)
        / "2years"
        / f"{experiment}_{spatial_tag}_profile.npz"
    )


def legacy_profile_cache_path(
    cache_dir: Path,
    variable: str,
    experiment: str,
    spatial_tag: str,
) -> Path:
    return (
        cache_dir
        / safe_name(variable)
        / "2years"
        / f"{experiment}_{spatial_tag}_diurnal_profile.npz"
    )


def height_cache_path(cache_dir: Path, experiment: str, spatial_tag: str) -> Path:
    return cache_dir / "height_axis" / "2years" / f"{experiment}_{spatial_tag}.npz"


def profile_from_diurnal(mean: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a (level, hour) diurnal profile into one weighted profile."""
    mean_arr = np.asarray(mean, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.float64)
    if mean_arr.shape != count_arr.shape:
        raise ValueError(f"mean/count shape mismatch: {mean_arr.shape} vs {count_arr.shape}")
    if mean_arr.ndim != 2:
        raise ValueError(f"Expected a 2-D diurnal profile, got {mean_arr.shape}")

    valid = np.isfinite(mean_arr) & (count_arr > 0)
    weighted = np.where(valid, mean_arr, 0.0) * np.where(valid, count_arr, 0.0)
    level_counts = np.sum(np.where(valid, count_arr, 0.0), axis=1)
    profile = np.full(mean_arr.shape[0], np.nan, dtype=np.float64)
    ok = level_counts > 0
    profile[ok] = np.sum(weighted, axis=1)[ok] / level_counts[ok]
    return profile, level_counts.astype(np.int64)


def load_cached_profile(path: Path) -> ProfileData:
    with np.load(path, allow_pickle=False) as data:
        if "profile" in data.files:
            profile = np.asarray(data["profile"], dtype=np.float64)
            if "counts" in data.files:
                counts = np.asarray(data["counts"], dtype=np.int64)
            else:
                counts = np.zeros(profile.shape, dtype=np.int64)
        elif "mean" in data.files and "counts" in data.files:
            profile, counts = profile_from_diurnal(data["mean"], data["counts"])
        else:
            raise KeyError(f"{path} must contain either profile/counts or mean/counts")

        n_files_raw = data["n_files"] if "n_files" in data.files else np.asarray([0])
        n_files = int(np.ravel(n_files_raw)[0]) if np.size(n_files_raw) else 0
    return ProfileData(profile=profile, counts=counts, n_files=n_files, source=str(path))


def save_profile_cache(path: Path, profile_data: ProfileData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        profile=profile_data.profile,
        counts=profile_data.counts,
        n_files=np.asarray([profile_data.n_files], dtype=np.int64),
        source=np.asarray([profile_data.source]),
    )


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def cached_profile_candidates(
    cache_dir: Path,
    legacy_cache_dir: Path,
    variable: str,
    experiment: str,
    spatial_tag: str,
) -> list[Path]:
    old_data_dir = legacy_cache_dir.parent / "old.data"
    return [
        profile_cache_path(cache_dir, variable, experiment, spatial_tag),
        legacy_profile_cache_path(legacy_cache_dir, variable, experiment, spatial_tag),
        legacy_profile_cache_path(old_data_dir, variable, experiment, spatial_tag),
    ]


def compute_profile_from_raw(
    experiment_dir: Path,
    variable: str,
    *,
    max_days: int | None,
    utc_offset_hours: int,
) -> ProfileData:
    variable_dir = experiment_dir / "masked-netcdf" / variable
    records = collect_file_records(
        variable_dir=variable_dir,
        max_days=max_days,
        allowed_months=None,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No NetCDF files found in {variable_dir}")

    first_profile, _ = read_vertical_profile(records[0][1], variable)
    sums = np.zeros(first_profile.shape, dtype=np.float64)
    counts = np.zeros(first_profile.shape, dtype=np.int64)

    for idx, (_, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(file_path, variable)
        if profile.shape != first_profile.shape:
            raise ValueError(
                f"Inconsistent profile shape in {file_path}: "
                f"{profile.shape} vs {first_profile.shape}"
            )
        valid = np.isfinite(profile)
        sums[valid] += profile[valid]
        counts[valid] += 1
        if idx % 2000 == 0 or idx == len(records):
            print(f"[{experiment_dir.name}/{variable}] {idx}/{len(records)} files", flush=True)

    out = np.full(first_profile.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return ProfileData(profile=out, counts=counts, n_files=len(records), source=str(variable_dir))


def get_profile(
    *,
    data_root: Path,
    cache_dir: Path,
    legacy_cache_dir: Path,
    experiment: str,
    variable: str,
    spatial_tag: str,
    max_days: int | None,
    utc_offset_hours: int,
    recompute: bool,
) -> ProfileData:
    cache_path = profile_cache_path(cache_dir, variable, experiment, spatial_tag)
    if not recompute:
        hit = first_existing(
            cached_profile_candidates(
                cache_dir, legacy_cache_dir, variable, experiment, spatial_tag
            )
        )
        if hit is not None:
            return load_cached_profile(hit)

    profile_data = compute_profile_from_raw(
        data_root / experiment,
        variable,
        max_days=max_days,
        utc_offset_hours=utc_offset_hours,
    )
    save_profile_cache(cache_path, profile_data)
    return profile_data


def load_height_cache(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "height_km" in data.files:
            height_km = np.asarray(data["height_km"], dtype=np.float64)
        elif "height_m" in data.files:
            height_km = np.asarray(data["height_m"], dtype=np.float64) / 1000.0
        else:
            raise KeyError(f"{path} must contain height_km or height_m")
    return height_km


def compute_height_from_raw(
    experiment_dir: Path,
    *,
    max_days: int | None,
    utc_offset_hours: int,
) -> np.ndarray:
    profile_data = compute_profile_from_raw(
        experiment_dir,
        "GEOPOTENTIEL",
        max_days=max_days,
        utc_offset_hours=utc_offset_hours,
    )
    # The current masked NetCDF GEOPOTENTIEL cache stores height in meters.
    return profile_data.profile / 1000.0


def get_height_km(
    *,
    data_root: Path,
    cache_dir: Path,
    legacy_cache_dir: Path,
    experiment: str,
    spatial_tag: str,
    max_days: int | None,
    utc_offset_hours: int,
    recompute: bool,
) -> tuple[np.ndarray, str]:
    cache_path = height_cache_path(cache_dir, experiment, spatial_tag)
    if not recompute:
        candidates = [
            height_cache_path(cache_dir, experiment, spatial_tag),
            height_cache_path(legacy_cache_dir, experiment, spatial_tag),
            height_cache_path(legacy_cache_dir.parent / "old.data", experiment, spatial_tag),
        ]
        hit = first_existing(candidates)
        if hit is not None:
            return load_height_cache(hit), str(hit)

    height_km = compute_height_from_raw(
        data_root / experiment,
        max_days=max_days,
        utc_offset_hours=utc_offset_hours,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, height_km=height_km)
    return height_km, str(cache_path)


def assert_consistent_lengths(
    experiment: str,
    height_km: np.ndarray,
    profiles: dict[str, ProfileData],
) -> None:
    for variable, profile_data in profiles.items():
        if profile_data.profile.shape != height_km.shape:
            raise ValueError(
                f"{experiment}/{variable} profile length {profile_data.profile.shape} "
                f"does not match height length {height_km.shape}"
            )


def write_data_txt(
    path: Path,
    profiles: dict[str, dict[str, ProfileData]],
    heights_km: dict[str, np.ndarray],
    height_sources: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    species_columns = ",".join(SPECIES)
    count_columns = ",".join(f"count_{name}" for name in SPECIES)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Hydrometeor vertical profiles: C1M, G1M, G2M\n")
        fh.write("=" * 72 + "\n")
        fh.write("Data used to plot Hydrometeor_Vertical_Profiles_C1M_G1M_G2M.png\n")
        fh.write("Mixing-ratio units: kg kg-1 (dimensionless model mixing ratio)\n")
        fh.write("Height units: km, from GEOPOTENTIEL height cache/raw masked NetCDF\n")
        fh.write("Averaging: spatial mean, then all available times in the two-year period\n")
        fh.write("\nSources\n")
        for experiment in EXPERIMENTS:
            label = EXPERIMENT_LABELS[experiment]
            fh.write(f"{label} height_source: {height_sources[experiment]}\n")
            for variable in SPECIES:
                source = profiles[experiment][variable].source
                n_files = profiles[experiment][variable].n_files
                fh.write(f"{label} {variable} source: {source} (n_files={n_files})\n")
        fh.write("\n")
        fh.write(
            "experiment,experiment_label,level_index,height_km,"
            f"{species_columns},{count_columns}\n"
        )
        for experiment in EXPERIMENTS:
            label = EXPERIMENT_LABELS[experiment]
            height = heights_km[experiment]
            n_levels = height.size
            for lev in range(n_levels):
                values = [profiles[experiment][var].profile[lev] for var in SPECIES]
                counts = [profiles[experiment][var].counts[lev] for var in SPECIES]
                value_text = ",".join(f"{val:.10e}" for val in values)
                count_text = ",".join(str(int(count)) for count in counts)
                fh.write(
                    f"{experiment},{label},{lev},{height[lev]:.10e},"
                    f"{value_text},{count_text}\n"
                )


def plot_profiles(
    output_path: Path,
    profiles: dict[str, dict[str, ProfileData]],
    heights_km: dict[str, np.ndarray],
    *,
    max_height_km: float,
    xmin: float,
    xmax: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 12), sharey=True)

    for idx, experiment in enumerate(EXPERIMENTS):
        ax = axes[idx]
        height = heights_km[experiment]
        for variable in SPECIES:
            ax.plot(
                profiles[experiment][variable].profile,
                height,
                color=SPECIES_COLOR[variable],
                linewidth=3,
                label=SPECIES_LABEL[variable],
            )

        ax.set_title(
            EXPERIMENT_LABELS[experiment],
            fontsize=16,
            fontweight="bold",
            color="black",
            pad=12,
        )
        ax.set_xlabel("Mixing Ratio (kg kg$^{-1}$)", fontsize=15, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Height (km)", fontsize=15, fontweight="bold")
            ax.legend(loc="upper right", fontsize=13, framealpha=0.95, edgecolor="black")

        ax.text(
            0.96,
            0.04,
            PANEL_LABELS[experiment],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 3.0},
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.0, max_height_km)
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.grid(False)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_profiles(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, ProfileData]],
    dict[str, np.ndarray],
    dict[str, str],
]:
    profiles: dict[str, dict[str, ProfileData]] = {}
    heights_km: dict[str, np.ndarray] = {}
    height_sources: dict[str, str] = {}

    for experiment in EXPERIMENTS:
        print(f"Processing {EXPERIMENT_LABELS[experiment]}...", flush=True)
        height, height_source = get_height_km(
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            legacy_cache_dir=args.legacy_cache_dir,
            experiment=experiment,
            spatial_tag=args.spatial_tag,
            max_days=args.max_days,
            utc_offset_hours=args.utc_offset_hours,
            recompute=args.recompute,
        )
        heights_km[experiment] = height
        height_sources[experiment] = height_source

        profiles[experiment] = {}
        for variable in SPECIES:
            profiles[experiment][variable] = get_profile(
                data_root=args.data_root,
                cache_dir=args.cache_dir,
                legacy_cache_dir=args.legacy_cache_dir,
                experiment=experiment,
                variable=variable,
                spatial_tag=args.spatial_tag,
                max_days=args.max_days,
                utc_offset_hours=args.utc_offset_hours,
                recompute=args.recompute,
            )
        assert_consistent_lengths(experiment, heights_km[experiment], profiles[experiment])

    return profiles, heights_km, height_sources


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profiles, heights_km, height_sources = build_profiles(args)

    output_path = args.output_dir / FIGURE_NAME
    text_path = args.data_txt_dir / TEXT_NAME
    plot_profiles(
        output_path,
        profiles,
        heights_km,
        max_height_km=args.max_height_km,
        xmin=args.xmin,
        xmax=args.xmax,
        dpi=args.dpi,
    )
    write_data_txt(text_path, profiles, heights_km, height_sources)
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


if __name__ == "__main__":
    main()
