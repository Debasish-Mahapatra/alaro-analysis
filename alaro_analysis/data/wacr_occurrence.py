"""Hydrometeor occurrence above the ARM site T3 (Manacapuru): the W-band cloud radar against the runs.

The local ARM archive copy holds the WACR Doppler-spectra files ``maowacrspeccmaskcopolM1.a0``
for 31 Aug - 30 Sep 2014, inside IOP2. Spectra are stored only where the radar's cloud mask
found a hydrometeor, so ``locator_mask`` (time, range; missing where nothing was detected) is a
hydrometeor mask every 2 s and 30 m. The mask cannot tell insects from cloud: in the boundary
layer it flags 50-80 % of the profiles (10 Sep 2014: 81 % at 0.4 km, 47 % at 1 km), so the
comparison uses heights from ``MIN_HEIGHT_KM`` up only.

The runs have no W-band operator. A model level counts as a hydrometeor when its total
condensate (cloud water, cloud ice, rain, snow, graupel) reaches a mass threshold;
``Q_THRESHOLDS`` bracket the radar's sensitivity.

Profiles are split by whether there is rain above the site: SIPAM 20 dBZ at 2 km in the
3x3 cells around T3 during the 12-min scan (observations), the model's own 20 dBZ at 2 km in
the 3x3 cells around the column (runs). In rain the W-band beam is attenuated, so the observed occurrence above
rain is a lower bound and the comparison is cleanest without rain. Only scans of the
radar-valid hours are used, so the observations and the runs share the hours.

The workers live here so they can be pickled to a process pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.data.microphysics_columns import PROFILE_HEIGHTS_KM

WACR_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ftp.archive.arm.gov/mahapatrad1/243912")
WACR_GLOB = "maowacrspeccmaskcopolM1.a0.*.nc"
T3_LAT, T3_LON = -3.2133, -60.5987  # ARM MAO M1
MIN_HEIGHT_KM = 4.0
Q_THRESHOLDS = (1e-6, 1e-5, 1e-4)  # kg/kg total condensate
BOX = 2  # model cells either side of the T3 cell (5x5 cells, 20 km)
RAIN_DBZ_2KM = 20.0
SCAN_S = 720  # SIPAM volume every 12 min
BIN_KM = 0.25

_SLOT_KEYS: np.ndarray | None = None  # sorted 12-min slot indices (epoch s // 720) with a SIPAM scan
_SLOT_RAIN: np.ndarray | None = None
_T3: tuple[int, int] | None = None
_HOURS: frozenset | None = None  # epoch hours used for the runs


def t3_cell(lon: np.ndarray, lat: np.ndarray) -> tuple[int, int]:
    d = np.hypot((lon - T3_LON) * np.cos(np.radians(T3_LAT)), lat - T3_LAT)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return int(j), int(i)


def init_wacr(slot_keys: np.ndarray, slot_rain: np.ndarray) -> None:
    global _SLOT_KEYS, _SLOT_RAIN
    order = np.argsort(slot_keys)
    _SLOT_KEYS, _SLOT_RAIN = np.asarray(slot_keys)[order], np.asarray(slot_rain)[order]


def _gate_bins(range_m: np.ndarray, antenna_m: float) -> np.ndarray:
    """Index of the profile height bin (+-125 m) of every gate, -1 outside."""
    z_km = (range_m + antenna_m) / 1000.0
    k = np.rint(z_km / BIN_KM).astype(int) - 1
    return np.where((k >= 0) & (k < PROFILE_HEIGHTS_KM.size), k, -1)


def wacr_file(path: str) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    """{(epoch hour, rain flag): (detection fraction summed over profiles per height bin, n profiles)}."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        mask = np.ma.filled(ds["locator_mask"][:].astype(float), np.nan)
        t = float(ds["base_time"][:]) + np.asarray(ds["time_offset"][:], dtype=float)
        bins = _gate_bins(np.asarray(ds["range"][:], dtype=float), float(str(ds.antenna_altitude).split()[0]))
    detected = np.isfinite(mask)
    nh = PROFILE_HEIGHTS_KM.size
    ok = bins >= 0
    per_bin = np.bincount(bins[ok], minlength=nh).astype(float)
    frac = np.zeros((detected.shape[0], nh))
    for k in np.unique(bins[ok]):
        frac[:, k] = detected[:, bins == k].mean(axis=1)
    frac[:, per_bin == 0] = np.nan
    slot = (t // SCAN_S).astype(np.int64)
    pos = np.clip(np.searchsorted(_SLOT_KEYS, slot), 0, _SLOT_KEYS.size - 1)
    scanned = _SLOT_KEYS[pos] == slot
    rain = _SLOT_RAIN[pos].astype(int)
    hour = np.rint(t / 3600.0).astype(np.int64)  # nearest hour, like the instantaneous model snapshot
    out: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for key in set(zip(hour[scanned].tolist(), rain[scanned].tolist())):
        sel = scanned & (hour == key[0]) & (rain == key[1])
        out[key] = (np.nansum(frac[sel], axis=0), int(sel.sum()))
    return out


def init_model(t3: tuple[int, int], hours: frozenset) -> None:
    global _T3, _HOURS
    _T3, _HOURS = t3, hours


def empty_model_stats() -> dict[str, np.ndarray]:
    nh = PROFILE_HEIGHTS_KM.size
    return {
        "occ": np.zeros((2, len(Q_THRESHOLDS), nh)),  # rows: no rain at 2 km, rain
        "occ_ice": np.zeros((2, nh)),  # cloud ice + snow + graupel >= the middle threshold
        "occ_liquid": np.zeros((2, nh)),  # cloud water + rain >= the middle threshold
        "n": np.zeros(2),
    }


def model_day(path: str) -> dict[str, np.ndarray]:
    import xarray as xr

    from alaro_analysis.analysis.radar_echo import interpolate_to_heights
    from alaro_analysis.data.microphysics_columns import _to_heights

    j, i = _T3
    m = BOX + 1  # one more cell around the box for the 3x3 rain neighbourhood
    box = dict(y=slice(j - m, j + m + 1), x=slice(i - m, i + m + 1))
    stats = empty_model_stats()
    with xr.open_dataset(path) as ds:
        hours = ds["time"].values.astype("datetime64[h]").astype(np.int64)
        keep = np.array([h in _HOURS for h in hours.tolist()])
        if not keep.any():
            return stats
        ds = ds.isel(time=np.flatnonzero(keep), **box)
        get = {v: ds[v].values.astype(float) for v in ("height", "dbz", "qc", "qi", "qr", "qs", "qg")}
    mid = Q_THRESHOLDS[len(Q_THRESHOLDS) // 2]
    nl, side = get["height"].shape[1], 2 * BOX + 1
    for n in range(get["height"].shape[0]):
        dbz2 = interpolate_to_heights(get["dbz"][n].reshape(nl, -1), get["height"][n].reshape(nl, -1),
                                      target_km=[2.0])[0].reshape(2 * m + 1, 2 * m + 1)
        wet = np.nan_to_num(dbz2, nan=-99.0) >= RAIN_DBZ_2KM
        rain = np.zeros((side, side), bool)
        for dy in range(3):  # like the observations: 20 dBZ anywhere in the 3x3 cells around the column
            for dx in range(3):
                rain |= wet[dy:dy + side, dx:dx + side]
        rain = rain.ravel().astype(int)
        flat = {v: a[n][:, 1:-1, 1:-1].reshape(nl, -1) for v, a in get.items()}
        height = flat["height"]
        ice = _to_heights(np.maximum(flat["qi"] + flat["qs"] + flat["qg"], 0.0), height)
        liquid = _to_heights(np.maximum(flat["qc"] + flat["qr"], 0.0), height)
        total = ice + liquid
        for r in (0, 1):
            sel = rain == r
            stats["n"][r] += sel.sum()
            for k, q in enumerate(Q_THRESHOLDS):
                stats["occ"][r, k] += np.nansum(total[:, sel] >= q, axis=1)
            stats["occ_ice"][r] += np.nansum(ice[:, sel] >= mid, axis=1)
            stats["occ_liquid"][r] += np.nansum(liquid[:, sel] >= mid, axis=1)
    return stats
