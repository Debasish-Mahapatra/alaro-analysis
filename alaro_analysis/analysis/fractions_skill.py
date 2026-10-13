"""Fractions skill score on a masked verification domain.

Roberts and Lean (2008): threshold both fields, take the fraction of exceeding
cells in each n x n neighbourhood, and score

    FSS = 1 - sum (O - M)^2 / sum (O^2 + M^2)

summed over every scored cell and every time. This aggregated form stays defined
when single hours are dry, and summing its two components over sub-periods gives
the combined-period score exactly.

The verification domain here is a radar footprint inside a rectangular box, so
fractions are taken over the valid cells of each neighbourhood only. A plain
rolling mean over the box would count the out-of-footprint cells as agreed
no-rain and inflate the score near the edge (nwpeval did this before its
September 2026 fix). ``nwpeval.metrics.fss`` now applies the same valid-neighbour
rule and agrees with this module exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# Below this a window's valid-cell share is numerical noise from the filter.
_MIN_VALID_SHARE = 1e-9


def neighbourhood_fractions(binary: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Fraction of exceeding cells among the valid cells of each size x size window.

    ``binary`` is (..., y, x) of 0/1 and ``valid`` is (y, x) bool. Cells outside
    ``valid`` never count, however much rain they hold. Cells with no valid
    neighbour come back NaN.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"neighbourhood size must be a positive odd integer, got {size}")
    valid_f = np.asarray(valid, dtype=float)
    field = np.asarray(binary, dtype=float) * valid_f
    window = (1,) * (field.ndim - 2) + (size, size)
    exceeding = uniform_filter(field, size=window, mode="constant", cval=0.0)
    share = uniform_filter(valid_f, size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(share > _MIN_VALID_SHARE, exceeding / share, np.nan)


def fss_components(
    obs_binary: np.ndarray,
    model_binary: np.ndarray,
    valid: np.ndarray,
    size: int,
    score: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sums of (O - M)^2 and of O^2 + M^2 over the scored cells and all times.

    ``score`` limits which cells are scored and defaults to ``valid``.
    """
    score = np.asarray(valid if score is None else score, dtype=bool)
    o = neighbourhood_fractions(obs_binary, valid, size)[..., score]
    m = neighbourhood_fractions(model_binary, valid, size)[..., score]
    return float(np.nansum((o - m) ** 2)), float(np.nansum(o**2 + m**2))


def fss_from_components(mse: float, mse_reference: float) -> float:
    return 1.0 - mse / mse_reference if mse_reference > 0 else float("nan")


def useful_skill(base_rate: float) -> float:
    """FSS of a uniform forecast at the observed base rate, 0.5 + f0 / 2.

    Roberts and Lean (2008) take a score at or above this as useful skill.
    """
    return 0.5 + base_rate / 2.0
