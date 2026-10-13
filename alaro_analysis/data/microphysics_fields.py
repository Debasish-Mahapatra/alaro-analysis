"""Microphysics fields of the GoAmazon IOP runs, converted from raw FA to NetCDF.

Only the fields the microphysics analyses use are read, on the window of the native
4 km SIPAM radar domain (``data.radar_domain``), at the instantaneous hourly snapshots
(leads +0001..+0024). Precipitation is accumulated in FA, so the surface fields are
differenced between leads and stored as the amount of the hour ending at the snapshot.

The day worker lives here, not in the recipe, because a recipe runs as ``__main__``
under ``alaro-recipe`` and a function defined there cannot be pickled to a process pool.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path

import numpy as np

from alaro_analysis.common.constants import G

N_LEVELS = 87
FA_FILE = "pfABOFABOF+{lead:04d}"
# FA name -> (NetCDF name, units, long name). Model levels S001 (top) .. S087 (surface).
LEVEL_FIELDS = {
    "REFLEC_DBZ": ("dbz", "dBZ", "simulated radar reflectivity (GPPRS0D, model's own operator)"),
    "GEOPOTENTIEL": ("height", "m", "height above mean sea level (geopotential / g)"),
    "TEMPERATURE": ("t", "K", "temperature"),
    "PRESSURE": ("p", "Pa", "pressure"),
    "HUMI.SPECIFI": ("q", "kg kg-1", "specific humidity"),
    "LIQUID_WATER": ("qc", "kg kg-1", "cloud liquid water mass fraction"),
    "SOLID_WATER": ("qi", "kg kg-1", "cloud ice mass fraction"),
    "RAIN": ("qr", "kg kg-1", "rain mass fraction"),
    "SNOW": ("qs", "kg kg-1", "snow mass fraction"),
    "GRAUPEL": ("qg", "kg kg-1", "graupel mass fraction"),
    "PNR": ("nr", "kg-1", "rain number per kg of air (2-moment runs only)"),
    "NC_LIQUID_WA": ("nc", "kg-1", "cloud droplet number per kg of air (2-moment runs only)"),
    "CV_PREC_FLUX": ("cv_prec_flux", "kg m-2 s-1", "convective (3MT) precipitation flux"),
    "ST_PREC_FLUX": ("st_prec_flux", "kg m-2 s-1", "stratiform (resolved) precipitation flux"),
    "VERT.VELOCIT": ("w", "m s-1", "resolved vertical velocity, positive up"),
}
# Accumulated surface precipitation, stored as the amount of the hour ending at the snapshot.
ACCUMULATED_FIELDS = {
    "SURFPREC.EAU.CON": ("rain_con", "kg m-2", "convective rain in the hour ending at the snapshot"),
    "SURFPREC.EAU.GEC": ("rain_gec", "kg m-2", "resolved rain in the hour ending at the snapshot"),
    "SURFPREC.NEI.CON": ("snow_con", "kg m-2", "convective snow in the hour ending at the snapshot"),
    "SURFPREC.NEI.GEC": ("snow_gec", "kg m-2", "resolved snow in the hour ending at the snapshot"),
    "SURFPREC.GRA.GEC": ("graupel_gec", "kg m-2", "resolved graupel in the hour ending at the snapshot"),
}
SURFACE_FIELDS = {
    "CLSTEMPERATURE": ("t2m", "K", "2 m temperature"),
    "CLSHUMI.SPECIFIQ": ("q2m", "kg kg-1", "2 m specific humidity"),
    "SURFCAPE.POS.F00": ("cape", "J kg-1", "CAPE"),
    "SURFCIEN.POS.F00": ("cin", "J kg-1", "CIN"),
}

_WINDOW: tuple[int, int, int, int, np.ndarray] | None = None


def init_worker(window) -> None:
    global _WINDOW
    import epygram

    epygram.init_env()
    _WINDOW = window


def _crop(field) -> np.ndarray:
    y0, y1, x0, x1, _ = _WINDOW
    if field.spectral:
        field.sp2gp()
    # single precision straight away: a day of snapshots must fit beside ecTrans (cgroup 96 GB)
    return np.asarray(field.getdata()[y0:y1, x0:x1], dtype=np.float32)


def read_snapshot(resource, names: set[str]) -> dict[str, np.ndarray]:
    """Level and surface fields of one FA file; level fields come back (87, y, x), S001 first."""
    out = {}
    for name in LEVEL_FIELDS:
        if name not in names:
            continue
        out[name] = np.stack([_crop(resource.readfield(f"S{k:03d}{name}")) for k in range(1, N_LEVELS + 1)])
    for name in (*ACCUMULATED_FIELDS, *SURFACE_FIELDS):
        if name in names:
            out[name] = _crop(resource.readfield(name))
    out["GEOPOTENTIEL"] = (out["GEOPOTENTIEL"] / G).astype(np.float32)
    return out


def convert_day(task: tuple[str, str, str]) -> tuple[str, int, tuple[str, ...]]:
    """Write one day (leads +0001..+0024) to ``out``; task = (fa_root, day, out)."""
    import epygram

    root, day, out = task
    out = Path(out)
    if out.exists():
        return day, -1, ()
    problems: list[str] = []
    snapshots, times = [], []
    previous = None
    for lead in range(25):
        path = Path(root) / f"pf{day}" / FA_FILE.format(lead=lead)
        try:
            resource = epygram.formats.resource(str(path), "r")
            try:
                names = {n[4:] if n[:1] == "S" and n[1:4].isdigit() else n for n in resource.listfields()}
                if lead == 0:
                    snap = {n: _crop(resource.readfield(n)) for n in ACCUMULATED_FIELDS if n in names}
                else:
                    snap = read_snapshot(resource, names)
            finally:
                resource.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day} +{lead:04d}: {exc!r}")
            return day, 0, tuple(problems)
        accumulated = {n: snap[n] for n in ACCUMULATED_FIELDS if n in snap}
        if lead > 0:
            for name in ACCUMULATED_FIELDS:
                if name in snap:
                    snap[name] = np.maximum(snap[name] - previous.get(name, 0.0), 0.0)
            snapshots.append(snap)
            times.append(np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}T00") + np.timedelta64(lead, "h"))
        previous = accumulated
    write_day(out, snapshots, np.array(times, dtype="datetime64[ns]"))
    return day, len(snapshots), tuple(problems)


def write_day(out: Path, snapshots: list[dict], times: np.ndarray) -> None:
    import xarray as xr

    y0, y1, x0, x1, mask = _WINDOW
    data_vars, encoding = {}, {}
    for table, dims in ((LEVEL_FIELDS, ("time", "level", "y", "x")),
                        (ACCUMULATED_FIELDS, ("time", "y", "x")),
                        (SURFACE_FIELDS, ("time", "y", "x"))):
        for fa_name, (name, units, long_name) in table.items():
            if fa_name not in snapshots[0]:
                continue
            values = np.stack([s[fa_name] for s in snapshots]).astype(np.float32)
            data_vars[name] = (dims, values, {"units": units, "long_name": long_name, "fa_name": fa_name})
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "level": np.arange(1, N_LEVELS + 1, dtype=np.int16),
            "inside": (("y", "x"), mask.astype(np.int8),
                       {"long_name": "cell inside the native 4 km SIPAM radar domain"}),
        },
        attrs={
            "window_fa_index": f"y {y0}:{y1}, x {x0}:{x1} of the 480 x 480 FA grid",
            "levels": "FA model levels S001 (top) .. S087 (lowest)",
            "time": "instantaneous snapshot at the valid time (UTC); surface precipitation = hour ending then",
        },
    )
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, out)
