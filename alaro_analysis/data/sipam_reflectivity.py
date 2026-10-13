"""SIPAM S-band radar reflectivity (raw CAPPI volumes) on the 4 km model grid.

The raw files (``ARM/downloaded_data/Manaus_Radar_MM_YYYY/cappi_YYYYMMDDHHMM.``) hold
15 constant-altitude levels (2..16 km, 1 km apart) of 500 x 500 float32 dBZ on a
~1 km grid centred on the radar (-3.15, -59.99): first index latitude increasing
northward, second longitude increasing eastward (checked against IMERG: spatial
correlation 0.35 as read, <= 0.09 for any flip). -99 marks both "no echo" and "no
data". The calibration offsets per period are those of the original rainfall
processing (``ARM/cappi-radar-rainfall.py``).

Each scan is averaged onto the cells of the model window in linear Z, with no-echo
pixels counting as zero, so a 4 km cell holds the reflectivity the radar would see if
it had the model's resolution. Cells whose mean is below MIN_DBZ are "no echo" (NaN).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

RAW_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ARM/downloaded_data")
RADAR_LAT, RADAR_LON = -3.15, -59.99
N_PIX = 500
HEIGHTS_KM = np.arange(2, 17, dtype=float)
MIN_DBZ = 0.0  # below this a coarse cell counts as no echo
CALIBRATION = (  # (first day, last day, offset dB) - from ARM/cappi-radar-rainfall.py
    (datetime(2014, 1, 3), datetime(2014, 2, 5), 1.0),
    (datetime(2014, 2, 6), datetime(2014, 8, 19), 3.0),
    (datetime(2014, 8, 20), datetime(2014, 10, 16), -2.5),
    (datetime(2014, 10, 17), datetime(2015, 3, 6), -5.5),
    (datetime(2015, 3, 7), datetime(2015, 7, 5), -4.0),
    (datetime(2015, 7, 6), datetime(2015, 10, 28), -1.5),
    (datetime(2015, 10, 29), datetime(2015, 12, 31), 1.0),
)

_MAP: dict | None = None


def pixel_lonlat() -> tuple[np.ndarray, np.ndarray]:
    dlat = 1 / 111
    dlon = 1 / (40075 * np.cos(np.radians(RADAR_LAT)) / 360)
    lat = np.linspace(RADAR_LAT - dlat * 250, RADAR_LAT + dlat * 250, N_PIX)
    lon = np.linspace(RADAR_LON - dlon * 250, RADAR_LON + dlon * 250, N_PIX)
    return np.meshgrid(lon, lat)


def calibration_offset(t: datetime) -> float:
    day = datetime(t.year, t.month, t.day)
    for first, last, offset in CALIBRATION:
        if first <= day <= last:
            return offset
    return 0.0


def build_map(model_lon: np.ndarray, model_lat: np.ndarray, inside: np.ndarray) -> dict:
    """Pixel -> model-cell index (-1 outside), from nearest model cell centre in km."""
    from scipy.spatial import cKDTree

    plon, plat = pixel_lonlat()
    k = np.cos(np.radians(RADAR_LAT)) * 111.0
    tree = cKDTree(np.column_stack([model_lon.ravel() * k, model_lat.ravel() * 111.0]))
    dist, idx = tree.query(np.column_stack([plon.ravel() * k, plat.ravel() * 111.0]))
    idx = np.where((dist <= 2.9) & inside.ravel()[idx], idx, -1)  # half a 4 km diagonal
    n_cells = model_lon.size
    counts = np.bincount(idx[idx >= 0], minlength=n_cells)
    return {"idx": idx, "counts": counts, "shape": model_lon.shape}


def init_worker(mapping: dict) -> None:
    global _MAP
    _MAP = mapping


def read_scan(path: Path) -> np.ndarray | None:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != 15 * N_PIX * N_PIX:
        return None
    dbz = raw.reshape(15, N_PIX * N_PIX).astype(np.float64)
    t = datetime.strptime(path.name[6:18], "%Y%m%d%H%M")
    echo = dbz > -98
    dbz[echo] += calibration_offset(t)
    return np.where(echo, 10.0 ** (dbz / 10.0), 0.0)


def coarse_scan(zlin: np.ndarray) -> np.ndarray:
    """(15, pixels) linear Z -> (15, y, x) dBZ on the model grid; NaN = no echo / outside."""
    idx, counts, shape = _MAP["idx"], _MAP["counts"], _MAP["shape"]
    use = idx >= 0
    out = np.full((zlin.shape[0], counts.size), np.nan)
    have = counts > 0
    for k in range(zlin.shape[0]):
        sums = np.bincount(idx[use], weights=zlin[k, use], minlength=counts.size)
        mean = np.where(have, sums / np.maximum(counts, 1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k] = np.where(mean > 10 ** (MIN_DBZ / 10), 10 * np.log10(mean), np.nan)
    return out.reshape(zlin.shape[0], *shape)


def process_day(task: tuple[str, list[str], str]) -> tuple[str, int]:
    """Coarse-grain every scan of one day whose hour is radar-valid; task = (day, hours, out)."""
    import xarray as xr

    day, valid_hours, out = task
    out = Path(out)
    if out.exists():
        return day, -1
    folder = RAW_ROOT / f"Manaus_Radar_{day[4:6]}_{day[:4]}"
    valid = set(valid_hours)
    scans, times = [], []
    for path in sorted(folder.glob(f"cappi_{day}*.")):
        t = datetime.strptime(path.name[6:18], "%Y%m%d%H%M")
        if t.strftime("%Y%m%d%H") not in valid:
            continue
        zlin = read_scan(path)
        if zlin is None:
            continue
        scans.append(coarse_scan(zlin).astype(np.float32))
        times.append(np.datetime64(t, "ns"))
    if not scans:
        return day, 0
    ds = xr.Dataset(
        {"dbz": (("time", "height", "y", "x"), np.stack(scans),
                 {"units": "dBZ", "long_name": "SIPAM reflectivity, linear-Z mean on the 4 km model cells",
                  "no_echo": f"NaN = no echo (mean below {MIN_DBZ} dBZ) or outside the radar domain"})},
        coords={"time": np.array(times), "height": HEIGHTS_KM * 1000.0},
    )
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(tmp, encoding={"dbz": {"zlib": True, "complevel": 4}})
    tmp.replace(out)
    return day, len(scans)
