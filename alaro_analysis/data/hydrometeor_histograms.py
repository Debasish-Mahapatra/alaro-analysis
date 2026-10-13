"""Hydrometeor histograms. Shared implementation; command-line orchestration lives in recipes.

Per-species log10(mass fraction) histograms over the SIPAM radar domain on the
native 4 km grid (``data.radar_domain``), read from raw FA, plus the mean
temperature profile each figure needs for its 0 degC line. No normalisation is applied here or at plot time: a
cell is the plain number of grid-point-hours in that bin at that model level.

The day workers live here rather than in a recipe because a recipe module runs as
``__main__`` under ``alaro-recipe``, and a worker function defined there cannot be
pickled to a process pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.data.convection_sections import N_LEVELS, read_fa_stack

# Paper row order: graupel, snow, cloud ice, cloud water, rain.
SPECIES = ("GRAUPEL", "SNOW", "SOLID_WATER", "LIQUID_WATER", "RAIN")
# The 2-ice control has no prognostic graupel; its FA field is identically zero.
ABSENT_SPECIES = {"control": ("GRAUPEL",)}
NBINS = 60
LOG_EDGES = np.linspace(-11.0, -1.0, NBINS + 1)  # log10(q), 1e-11 .. 1e-1
DBIN = (LOG_EDGES[-1] - LOG_EDGES[0]) / NBINS


def species_for(experiment: str) -> tuple[str, ...]:
    absent = ABSENT_SPECIES.get(experiment, ())
    return tuple(name for name in SPECIES if name not in absent)


def histogram_levels(values: np.ndarray) -> np.ndarray:
    """counts[N_LEVELS, NBINS] of log10(q) for a (level, ...) array.

    Only strictly positive values are counted; zeros, NaNs (cells outside the
    radar domain) and values outside the bin range are dropped.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    counts = np.zeros((N_LEVELS, NBINS), dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(flat > 0.0, np.log10(flat), np.nan)
    bins = np.floor((logs - LOG_EDGES[0]) / DBIN)
    for level in range(min(flat.shape[0], N_LEVELS)):
        b = bins[level]
        b = b[np.isfinite(b)]
        b = b[(b >= 0) & (b < NBINS)].astype(np.int64)
        if b.size:
            counts[level] += np.bincount(b, minlength=NBINS)
    return counts


def _empty(species: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.zeros((N_LEVELS, NBINS), dtype=np.int64) for name in species}


def fa_day(task: tuple[str, str, tuple[str, ...]]):
    """Histogram one day of raw FA over the radar domain; FA is top first, so flip."""
    import epygram

    root, day_token, species = task
    counts = _empty(species)
    temp_sum = np.zeros(N_LEVELS)
    temp_count = np.zeros(N_LEVELS, dtype=np.int64)
    problems: list[str] = []
    n_files = 0
    for hour in range(24):
        path = Path(root) / f"pf{day_token}" / f"pfABOFABOF+{hour:04d}"
        if not path.exists():
            problems.append(f"{day_token} +{hour:04d}: missing")
            continue
        try:
            resource = epygram.formats.resource(str(path), "r")
            try:
                stacks = {name: read_fa_stack(resource, name) for name in species}
                temperature = read_fa_stack(resource, "TEMPERATURE")
            finally:
                resource.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} +{hour:04d}: {exc}")
            continue
        for name, values in stacks.items():
            counts[name] += histogram_levels(values)[::-1]
        profile = temperature.mean(axis=1)[::-1]
        finite = np.isfinite(profile)
        temp_sum[finite] += profile[finite]
        temp_count[finite] += 1
        n_files += 1
    return counts, temp_sum, temp_count, n_files, tuple(problems)
