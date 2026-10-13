"""Column microphysics statistics of one converted model day (``data.microphysics_fields``).

One pass over a day file accumulates, over the cells of the 4 km SIPAM radar domain
and every hourly snapshot:

* ``dsd``: surface rain drop size distribution, from the lowest model level with each
  scheme's own law (1-moment Abel & Boutle 2012, 2-moment gamma with the ALARO
  variable shape), truncated at the disdrometer's 0.312 mm like the paper's DSD
  figure (``dsd.truncated_moments``): joint histogram of
  Dm and log10 Nw, and Dm per rain-rate bin;
* ``profiles``: mean hydrometeor mass fractions, precipitation fluxes (3MT convection
  scheme and resolved microphysics) and mass occurrence at fixed heights;
* ``coldpool``: 2 m temperature and humidity departures from the hour's median over
  non-raining cells, binned by the rain of the hour ending at the snapshot; by local hour,
  the domain-mean 2 m temperature and the departure of the cells with at least 1 mm.

The worker lives here (not in a recipe) so it can be pickled to a process pool.
"""

from __future__ import annotations

import numpy as np

PROFILE_HEIGHTS_KM = np.arange(0.25, 16.01, 0.25)
DM_EDGES = np.linspace(0.3, 4.0, 75)
LOGNW_EDGES = np.linspace(1.0, 6.0, 101)
RAIN_EDGES = np.array([0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0])  # mm/h
COLD_RAIN_EDGES = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 300.0])  # mm in the hour
SPECIES = ("qc", "qi", "qr", "qs", "qg")
DMIN_M = 0.312e-3
MIN_RR = 0.1
MAX_DBZ = 55.0
RD = 287.06
UTC_OFFSET = -4

_INSIDE: np.ndarray | None = None


def init_worker(inside: np.ndarray) -> None:
    global _INSIDE
    _INSIDE = inside


def empty_stats() -> dict[str, np.ndarray]:
    nh = PROFILE_HEIGHTS_KM.size
    return {
        "dsd_hist": np.zeros((DM_EDGES.size - 1, LOGNW_EDGES.size - 1)),
        "dsd_dm_by_rr": np.zeros((RAIN_EDGES.size - 1, DM_EDGES.size - 1)),
        "dsd_mu_sum": np.zeros(1),
        "dsd_n": np.zeros(1),
        "species_sum": np.zeros((len(SPECIES), nh)),
        "flux_sum": np.zeros((2, nh)),  # 3MT (CV_PREC_FLUX), resolved (ST_PREC_FLUX) precipitation flux (kg m-2 s-1)
        "mass_occurrence": np.zeros((3, nh)),  # qs+qg >= 1e-5, 1e-4; qr >= 1e-4 kg/kg
        "n_profile": np.zeros(1),
        "coldpool_t": np.zeros(COLD_RAIN_EDGES.size - 1),
        "coldpool_q": np.zeros(COLD_RAIN_EDGES.size - 1),
        "coldpool_n": np.zeros(COLD_RAIN_EDGES.size - 1),
        "rain_hour": np.zeros((2, 24)),  # mean rain (mm per hour) summed over cells, number of cells
        # by local hour: domain-mean 2 m temperature (K) summed over snapshots, and the 2 m temperature of
        # cells with >= 1 mm in the hour minus the hour's non-raining median, summed over those cells
        "t2m_hour": np.zeros((2, 24)),
        "coldpool_hour": np.zeros((2, 24)),
        # 2-moment runs: in-cloud droplet number (cm-3, where qc > 1e-5) and in-rain drop number (m-3,
        # where qr > 1e-6) at the profile heights: sums and counts
        "nc_incloud": np.zeros((2, nh)),
        "nr_inrain": np.zeros((2, nh)),
    }


def _to_heights(values: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    """(level, n) on top-first model levels -> (n_heights, n), linear in height."""
    z = height_m[::-1]
    v = values[::-1]
    out = np.empty((PROFILE_HEIGHTS_KM.size, values.shape[1]))
    for j in range(values.shape[1]):
        out[:, j] = np.interp(PROFILE_HEIGHTS_KM * 1000.0, z[:, j], v[:, j], left=np.nan, right=np.nan)
    return out


def _dsd(stats: dict, qr: np.ndarray, nr: np.ndarray | None, rho: np.ndarray) -> None:
    from alaro_analysis.dsd.truncated_moments import moments_abel_boutle, moments_two_moment

    rain = qr > 1e-7
    if not rain.any():
        return
    if nr is None:
        s = moments_abel_boutle(qr[rain], rho[rain], DMIN_M, MIN_RR, MAX_DBZ)
    else:
        ok = rain & (nr > 0)
        s = moments_two_moment(qr[ok], nr[ok], rho[ok], DMIN_M, MIN_RR, MAX_DBZ)
    if s["dm_mm"].size == 0:
        return
    stats["dsd_hist"] += np.histogram2d(s["dm_mm"], s["log_nw"], [DM_EDGES, LOGNW_EDGES])[0]
    # rain rate of the truncated spectrum, recomputed from LWC and Dm with the same fall law
    rr = _rain_rate(s["lwc_g_m3"].astype(float), s["dm_mm"].astype(float))
    stats["dsd_dm_by_rr"] += np.histogram2d(rr, s["dm_mm"], [RAIN_EDGES, DM_EDGES])[0]
    stats["dsd_mu_sum"] += float(np.sum(s["mu"]))
    stats["dsd_n"] += s["dm_mm"].size


def _rain_rate(lwc_g_m3: np.ndarray, dm_mm: np.ndarray) -> np.ndarray:
    """mm/h, LWC times a mass-weighted fall speed 9.65 - 10.3 exp(-0.6 Dm) (Atlas et al. 1973)."""
    v = np.maximum(9.65 - 10.3 * np.exp(-0.6 * dm_mm), 0.1)
    return 3.6 * lwc_g_m3 * v


def model_day(task: tuple[str, bool]) -> dict:
    import xarray as xr

    path, two_moment = task
    stats = empty_stats()
    inside = _INSIDE
    with xr.open_dataset(path) as ds:
        get = {v: ds[v].values for v in ("height", "t", "p", "qc", "qi", "qr", "qs", "qg",
                                          "cv_prec_flux", "st_prec_flux", "rain_con", "rain_gec",
                                          "t2m", "q2m")}
        nr_all = ds["nr"].values if (two_moment and "nr" in ds) else None
        nc_all = ds["nc"].values if (two_moment and "nc" in ds) else None
        times = ds["time"].values
    for i, t in enumerate(times):
        sel = inside
        height = get["height"][i][:, sel]
        tt, pp = get["t"][i][:, sel], get["p"][i][:, sel]
        rho = pp / (RD * tt)
        # surface DSD (lowest level)
        nr = None if nr_all is None else nr_all[i][-1, sel].astype(float)
        _dsd(stats, get["qr"][i][-1, sel].astype(float), nr, rho[-1].astype(float))
        # profiles at fixed heights
        for k, sp in enumerate(SPECIES):
            stats["species_sum"][k] += np.nansum(_to_heights(np.maximum(get[sp][i][:, sel], 0.0), height), axis=1)
        flux = np.stack([get["cv_prec_flux"][i][:, sel], get["st_prec_flux"][i][:, sel]]).astype(float)
        for k in range(2):
            stats["flux_sum"][k] += np.nansum(_to_heights(np.maximum(flux[k], 0.0), height), axis=1)
        ice = _to_heights(np.maximum(get["qs"][i][:, sel] + get["qg"][i][:, sel], 0.0), height)
        qrz = _to_heights(np.maximum(get["qr"][i][:, sel], 0.0), height)
        stats["mass_occurrence"][0] += np.nansum(ice >= 1e-5, axis=1)
        stats["mass_occurrence"][1] += np.nansum(ice >= 1e-4, axis=1)
        stats["mass_occurrence"][2] += np.nansum(qrz >= 1e-4, axis=1)
        stats["n_profile"] += sel.sum()
        if nc_all is not None:
            qcz = _to_heights(get["qc"][i][:, sel], height)
            ncz = _to_heights(nc_all[i][:, sel] * rho, height) / 1e6  # cm-3
            cloudy = np.isfinite(ncz) & (qcz > 1e-5)
            stats["nc_incloud"][0] += np.where(cloudy, ncz, 0.0).sum(axis=1)
            stats["nc_incloud"][1] += cloudy.sum(axis=1)
        if nr_all is not None:
            nrz = _to_heights(nr_all[i][:, sel] * rho, height)  # m-3
            rainy = np.isfinite(nrz) & (qrz > 1e-6)
            stats["nr_inrain"][0] += np.where(rainy, nrz, 0.0).sum(axis=1)
            stats["nr_inrain"][1] += rainy.sum(axis=1)
        # cold pools: departures from the hour's median over non-raining cells
        rain = (get["rain_con"][i] + get["rain_gec"][i])[sel]
        t2, q2 = get["t2m"][i][sel], get["q2m"][i][sel]
        dry = rain < 0.01
        if dry.sum() > 50:
            dt, dq = t2 - np.median(t2[dry]), q2 - np.median(q2[dry])
            b = np.digitize(rain, COLD_RAIN_EDGES) - 1
            ok = (b >= 0) & (b < COLD_RAIN_EDGES.size - 1)
            stats["coldpool_t"] += np.bincount(b[ok], weights=dt[ok], minlength=COLD_RAIN_EDGES.size - 1)
            stats["coldpool_q"] += np.bincount(b[ok], weights=dq[ok], minlength=COLD_RAIN_EDGES.size - 1)
            stats["coldpool_n"] += np.bincount(b[ok], minlength=COLD_RAIN_EDGES.size - 1)
        lh = int((np.datetime64(t, "h").astype(int) + UTC_OFFSET) % 24)
        stats["rain_hour"][0, lh] += rain.sum()
        stats["rain_hour"][1, lh] += rain.size
        stats["t2m_hour"][0, lh] += float(np.mean(t2))
        stats["t2m_hour"][1, lh] += 1
        if dry.sum() > 50 and (rain >= 1.0).any():
            stats["coldpool_hour"][0, lh] += float(np.sum(dt[rain >= 1.0]))
            stats["coldpool_hour"][1, lh] += int(np.sum(rain >= 1.0))
    return stats
