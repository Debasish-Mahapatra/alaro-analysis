from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.common.models import AxisSpec
from alaro_analysis.common.naming import safe_name


def build_cache_file(
    intermediate_dir: Path,
    analysis_name: str,
    period_subdir: Path,
    experiment: str,
    spatial_tag: str,
) -> Path:
    return (
        intermediate_dir
        / safe_name(analysis_name)
        / period_subdir
        / f"{experiment}_{spatial_tag}.npz"
    )


def build_diurnal_cache_file(
    intermediate_dir: Path,
    variable: str,
    period_subdir: Path,
    experiment: str,
    spatial_tag: str | None = None,
) -> Path:
    suffix = f"{experiment}_diurnal_profile.npz"
    if spatial_tag is not None:
        suffix = f"{experiment}_{spatial_tag}_diurnal_profile.npz"
    return intermediate_dir / safe_name(variable) / period_subdir / suffix


def build_height_cache_file(
    intermediate_dir: Path,
    period_subdir: Path,
    experiment: str,
    aggregate: str,
    spatial_tag: str | None = None,
) -> Path:
    suffix = f"{experiment}_height_profile_{aggregate}.npz"
    if spatial_tag is not None:
        suffix = f"{experiment}_{spatial_tag}_height_profile_{aggregate}.npz"
    return intermediate_dir / "geopotential" / period_subdir / suffix


def save_cache(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **payload)


def load_cache(cache_file: Path, *, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    with np.load(cache_file, allow_pickle=allow_pickle) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_diurnal_profile_cache(
    cache_file: Path,
) -> tuple[np.ndarray, np.ndarray | None, int | None, Path | None]:
    payload = load_cache(cache_file, allow_pickle=True)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    counts = None
    if "counts" in payload:
        counts = np.asarray(payload["counts"], dtype=np.int64)
    n_files = None
    if "n_files" in payload:
        n_files = int(np.asarray(payload["n_files"]).ravel()[0])
    sample_file = None
    if "sample_file" in payload:
        sample_file = Path(str(np.asarray(payload["sample_file"]).ravel()[0]))
    return mean, counts, n_files, sample_file


def save_diurnal_profile_cache(
    cache_file: Path,
    *,
    mean: np.ndarray,
    counts: np.ndarray,
    n_files: int,
    sample_file: Path,
) -> None:
    save_cache(
        cache_file,
        {
            "mean": np.asarray(mean, dtype=np.float64),
            "counts": np.asarray(counts, dtype=np.int64),
            "n_files": np.array([n_files], dtype=np.int64),
            "sample_file": np.array([str(sample_file)]),
        },
    )


def load_height_profile_cache(cache_file: Path) -> np.ndarray:
    payload = load_cache(cache_file)
    return np.asarray(payload["height_m"], dtype=np.float64)


def save_height_profile_cache(
    cache_file: Path,
    *,
    height_m: np.ndarray,
    n_files: int,
) -> None:
    save_cache(
        cache_file,
        {
            "height_m": np.asarray(height_m, dtype=np.float64),
            "n_files": np.array([n_files], dtype=np.int64),
        },
    )


def find_cache_file(intermediate_roots: list[Path], relpath: Path) -> Path | None:
    for root in intermediate_roots:
        candidate = root / relpath
        if candidate.exists():
            return candidate
    return None


def find_existing_cache(intermediate_roots: list[Path], relpaths: list[Path]) -> Path | None:
    for root in intermediate_roots:
        for relpath in relpaths:
            candidate = root / relpath
            if candidate.exists():
                return candidate
    return None


def cache_relpath(
    variable: str,
    period_subdir: Path | str,
    experiment: str,
    spatial_tag: str | None = None,
) -> Path:
    return build_diurnal_cache_file(
        intermediate_dir=Path("."),
        variable=variable,
        period_subdir=Path(period_subdir),
        experiment=experiment,
        spatial_tag=spatial_tag,
    ).relative_to(".")


def height_relpaths(
    period_subdir: Path | str,
    experiment: str,
    aggregate: str,
    spatial_tag: str | None = None,
    *,
    include_geopotentiel_fallback: bool = False,
) -> list[Path]:
    relpaths = [
        build_height_cache_file(
            intermediate_dir=Path("."),
            period_subdir=Path(period_subdir),
            experiment=experiment,
            aggregate=aggregate,
            spatial_tag=spatial_tag,
        ).relative_to(".")
    ]
    if include_geopotentiel_fallback:
        suffix = f"{experiment}_height_profile_{aggregate}.npz"
        if spatial_tag is not None:
            suffix = f"{experiment}_{spatial_tag}_height_profile_{aggregate}.npz"
        relpaths.append(Path("geopotentiel") / Path(period_subdir) / suffix)
    return relpaths


def load_diurnal_mean(
    intermediate_roots: list[Path],
    variable: str,
    period_subdir: Path | str,
    experiment: str,
    spatial_tag: str | None = None,
) -> np.ndarray:
    relpath = cache_relpath(variable, period_subdir, experiment, spatial_tag)
    path = find_cache_file(intermediate_roots, relpath)
    if path is None:
        raise FileNotFoundError(
            f"Cache not found for {variable} / {experiment}: {relpath}\n"
            f"Searched roots: {intermediate_roots}"
        )
    mean, _, _, _ = load_diurnal_profile_cache(path)
    return mean


def load_height_axis(
    intermediate_roots: list[Path],
    period_subdir: Path | str,
    experiment: str,
    aggregate: str = "first",
    *,
    spatial_tag: str | None = None,
    fallback_levels: int | None = None,
    allow_geopotentiel_fallback: bool = False,
) -> AxisSpec:
    relpaths = height_relpaths(
        period_subdir,
        experiment,
        aggregate,
        spatial_tag,
        include_geopotentiel_fallback=allow_geopotentiel_fallback,
    )
    found = find_existing_cache(intermediate_roots, relpaths)
    if found is None:
        if fallback_levels is None:
            raise FileNotFoundError(
                f"Height cache not found for {experiment}: {relpaths}\n"
                f"Searched roots: {intermediate_roots}"
            )
        return AxisSpec(
            values=np.arange(fallback_levels, dtype=np.float64),
            label="Model level",
            is_height_km=False,
        )

    height_m = load_height_profile_cache(found)
    return AxisSpec(values=height_m / 1000.0, label="Height (km)", is_height_km=True)


def load_temperature_profile(
    intermediate_roots: list[Path],
    period_subdir: Path | str,
    experiment: str,
    spatial_tag: str | None = None,
) -> np.ndarray | None:
    relpath = cache_relpath("TEMPERATURE", period_subdir, experiment, spatial_tag)
    path = find_cache_file(intermediate_roots, relpath)
    if path is None:
        return None
    try:
        mean, _, _, _ = load_diurnal_profile_cache(path)
        return mean
    except Exception:
        return None
