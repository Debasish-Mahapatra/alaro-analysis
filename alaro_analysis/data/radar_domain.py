"""SIPAM radar domain on the native 4 km model grid.

Built from the radar's own ~1 km coverage, not from the 0.1 deg radar mask: a model
cell is inside when at least half of the ~1 km radar pixels that fall in it lie
inside the radar coverage. The 0.1 deg mask (``mask/Radar_mask_latlon.nc``) belongs
to the rain-bias figures only.

The mask is stored once on the full 480 x 480 FA grid (``MASK_FILE``); building it
needs one raw FA file for the model longitudes and latitudes and the epygram
environment, loading it needs neither.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from alaro_analysis.common.paths import runs_path

NATIVE_RADAR = runs_path(
    "old-archive/rainfall-regridded-to-imerge/sipam/Manaus_Radar_Rainfall_2014-15-hourly-masked.nc"
)
MASK_FILE = runs_path("mask/Radar_mask_4km_native.nc")
COVERAGE_THRESHOLD = 0.5
# Physical (C) zone of the 480 x 480 FA grid; the extension zone holds no model state.
C_ZONE = 434
# A 4 km model cell holds about 16 radar pixels; fewer than this means the radar
# grid only grazes it.
MIN_PIXELS = 8


def native_footprint(path: Path = NATIVE_RADAR, samples: int = 40):
    """The radar's fixed coverage on its own ~1 km grid, as (footprint, lat, lon).

    The masked radar file carries one fixed footprint; it is read at ``samples``
    valid hours spread over the record, and they must all agree.
    """
    with xr.open_dataset(path) as ds:
        valid = np.flatnonzero(ds["valid_time_mask"].values > 0)
        picks = valid[np.linspace(0, valid.size - 1, samples).astype(int)]
        patterns = [np.isfinite(ds["rainfall_rate"].isel(time=int(i)).values) for i in picks]
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    if not all(np.array_equal(p, patterns[0]) for p in patterns):
        raise RuntimeError("the radar footprint changes between valid hours")
    return patterns[0], lat, lon


def coverage_on_model_grid(footprint, lat, lon, model_lon, model_lat):
    """Share of each model cell's radar pixels that lie in the footprint, and their count.

    Every radar pixel is assigned to its nearest model cell (distances in a local
    equirectangular frame).
    """
    from scipy.spatial import cKDTree

    coslat = np.cos(np.deg2rad(np.mean(lat)))
    tree = cKDTree(np.c_[model_lon.ravel() * coslat, model_lat.ravel()])
    lon2, lat2 = np.meshgrid(lon, lat)
    _, nearest = tree.query(np.c_[lon2.ravel() * coslat, lat2.ravel()])
    total = np.bincount(nearest, minlength=model_lon.size).reshape(model_lon.shape)
    inside = np.bincount(
        nearest, weights=footprint.ravel().astype(float), minlength=model_lon.size
    ).reshape(model_lon.shape)
    share = np.divide(inside, total, out=np.zeros(model_lon.shape), where=total > 0)
    return share, total


def build_mask(fa_sample: Path, out: Path = MASK_FILE) -> Path:
    """Write the 4 km radar-domain mask for the FA grid of ``fa_sample``."""
    import epygram

    epygram.init_env()
    resource = epygram.formats.resource(str(fa_sample), "r")
    try:
        geometry = resource.readfield("S001UD_MESH_FRAC").geometry
        lon, lat = (np.ma.getdata(a).astype(np.float64) for a in geometry.get_lonlat_grid())
    finally:
        resource.close()

    footprint, radar_lat, radar_lon = native_footprint()
    c = slice(0, C_ZONE)
    share = np.zeros(lon.shape)
    total = np.zeros(lon.shape, dtype=np.int64)
    share[c, c], total[c, c] = coverage_on_model_grid(
        footprint, radar_lat, radar_lon, lon[c, c], lat[c, c]
    )
    mask = (total >= MIN_PIXELS) & (share >= COVERAGE_THRESHOLD)

    ds = xr.Dataset(
        {
            "radar_mask": (("y", "x"), mask.astype(np.int8)),
            "coverage_fraction": (("y", "x"), share.astype(np.float32)),
            "radar_pixels": (("y", "x"), total.astype(np.int32)),
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        },
        attrs={
            "title": "SIPAM radar domain on the native 4 km ALARO grid",
            "rule": f"cell inside when >= {COVERAGE_THRESHOLD:.0%} of its ~1 km radar pixels are in "
            f"the radar coverage (and it holds >= {MIN_PIXELS} radar pixels)",
            "radar_source": str(NATIVE_RADAR),
            "grid_source": str(fa_sample),
            "cells_inside": int(mask.sum()),
            "note": "the 0.1 deg mask Radar_mask_latlon.nc is for the rain-bias figures only",
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def load_mask(path: Path = MASK_FILE) -> np.ndarray:
    """(480, 480) boolean radar-domain mask on the FA grid."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing; build it with alaro_analysis.data.radar_domain.build_mask"
        )
    with xr.open_dataset(path) as ds:
        return ds["radar_mask"].values.astype(bool)


def mask_window(path: Path = MASK_FILE) -> tuple[int, int, int, int, np.ndarray]:
    """Crop (y0, y1, x0, x1) of the FA grid around the domain, and the mask inside it."""
    mask = load_mask(path)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1, mask[y0:y1, x0:x1]
