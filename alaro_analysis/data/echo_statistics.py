"""Day workers for the radar-echo statistics (see ``analysis.radar_echo``).

They live here, not in the recipe, because a recipe runs as ``__main__`` under
``alaro-recipe`` and a function defined there cannot be pickled to a process pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.analysis.radar_echo import (
    ICE_MAX_T_C,
    UTC_OFFSET,
    accumulate,
    empty_stats,
    interpolate_to_heights,
    iwc_from_dbz,
    linear_to_heights,
)

_DOMAINS: tuple[np.ndarray, np.ndarray] | None = None
_T_PROFILE: np.ndarray | None = None  # degC at the radar heights, mean of the IOP
_Z_IWC_T = None


def init_worker(domains, t_profile_c=None, z_iwc_t=None) -> None:
    global _DOMAINS, _T_PROFILE, _Z_IWC_T
    _DOMAINS = domains
    _T_PROFILE = None if t_profile_c is None else np.asarray(t_profile_c, dtype=float)
    _Z_IWC_T = z_iwc_t


def _ice_levels():
    return None if _T_PROFILE is None else _T_PROFILE <= ICE_MAX_T_C


def _local_hour(t) -> int:
    return int((np.datetime64(t, "h").astype(int) + UTC_OFFSET) % 24)


def sipam_day(path: str) -> dict:
    import xarray as xr

    classify, stats_domain = _DOMAINS
    stats = empty_stats()
    with xr.open_dataset(path) as ds:
        dbz = ds["dbz"].values.astype(np.float64)
        times = ds["time"].values
    ice = _ice_levels()
    for i, t in enumerate(times):
        iwc = None
        if ice is not None and _Z_IWC_T is not None:
            iwc = iwc_from_dbz(dbz[i], _T_PROFILE[:, None, None], _Z_IWC_T)
        accumulate(stats, dbz[i], classify, stats_domain, _local_hour(t), iwc_h=iwc, ice_levels=ice)
    return stats


def model_day(task: tuple[str, frozenset]) -> dict:
    """Statistics of one converted model day, only at radar-valid hours (YYYYMMDDHH)."""
    import xarray as xr

    path, valid_hours = task
    classify, stats_domain = _DOMAINS
    stats = empty_stats()
    with xr.open_dataset(path) as ds:
        times = ds["time"].values
        keep = [i for i, t in enumerate(times)
                if str(np.datetime64(t, "h")).replace("-", "").replace("T", "")[:10] in valid_hours]
        if not keep:
            return stats
        dbz = ds["dbz"].values[keep]
        height = ds["height"].values[keep]
        # the model's own instantaneous rain rate (mm/h): precipitation flux at the lowest level at the
        # snapshot, the same instant as the reflectivity (the hourly accumulations lag the classification)
        rain = 3600.0 * sum(np.maximum(ds[v].values[keep][:, -1], 0.0) for v in ("cv_prec_flux", "st_prec_flux"))
        ice_mass = sum(np.maximum(ds[v].values[keep], 0.0) for v in ("qs", "qg", "qi"))
        rho = ds["p"].values[keep] / (287.06 * ds["t"].values[keep])
    ice = _ice_levels()
    for j, i in enumerate(keep):
        dbz_h = interpolate_to_heights(dbz[j], height[j])
        iwc = linear_to_heights(1000.0 * rho[j] * ice_mass[j], height[j]) if ice is not None else None
        accumulate(stats, dbz_h, classify, stats_domain, _local_hour(times[i]), surface_rain=rain[j],
                   iwc_h=iwc, ice_levels=ice)
    return stats


def add(total: dict, part: dict) -> dict:
    for key, value in part.items():
        total[key] = total[key] + value
    return total
