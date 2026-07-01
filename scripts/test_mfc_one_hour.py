#!/usr/bin/env python3
"""One-hour test of the moisture flux convergence pipeline.

Reads a SINGLE raw-FA state (control, one forecast hour), forms the column
vapour flux and its divergence with the exact same package helpers used by the
2-year driver, and plots the instantaneous MFC raw vs flux-smoothed so the
operator's behaviour on a single state can be eyeballed. Must run under the
`epygram` conda env (needs faxarray to read raw FA).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from alaro_analysis.analysis.derived import (
    column_integrated_vapor_flux,
    compute_dp_pa,
    horizontal_divergence_spherical,
)

FLUX_VARS = ("WIND.U.PHYS", "WIND.V.PHYS", "HUMI.SPECIFI", "PRESSURE")
SECONDS_PER_DAY = 86400.0
MANAUS_LON, MANAUS_LAT = -60.0217, -3.1190
CROP = (-61.4, -58.6, -4.6, -1.7)  # radar-rectangle, same as the paper figure

STATE = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO/control/untar-output/pf20140101/pfABOFABOF+0020"
OUT_DIR = "/mnt/HDS_CLIMATE/CLIMATE/deba/microphysics-paper/9. moisture flux convergence"
OUT_PNG = f"{OUT_DIR}/9. moisture flux convergence_1hour-test_450dpi.png"


def open_fa(path):
    import faxarray as fx

    return fx.open_dataset(path, variables=list(FLUX_VARS), stack_levels=True)


def data_var(ds, requested):
    if requested in ds.data_vars:
        return requested
    token = requested.replace(".", "").replace("_", "").upper()
    for name in ds.data_vars:
        if name.replace("_", "").replace(".", "").upper() == token:
            return name
    raise KeyError(f"{requested!r} not in {list(ds.data_vars)}")


def read_field(ds, requested):
    return np.asarray(ds[data_var(ds, requested)].isel(time=0).values, dtype=np.float32)


def box_smooth(f, k):
    """Separable (2k+1) box mean, NaN-safe, no scipy dependency."""
    if k <= 0:
        return f
    g = np.where(np.isfinite(f), f, 0.0)
    w = np.isfinite(f).astype(float)
    n = 2 * k + 1
    ker = np.ones(n)
    def conv1d(a, axis):
        return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), axis, a)
    num = conv1d(conv1d(g, 0), 1)
    den = conv1d(conv1d(w, 0), 1)
    out = np.full_like(f, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def crop(arr, lon, lat, bbox):
    lo0, lo1, la0, la1 = bbox
    inside = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    ys, xs = np.where(inside)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return arr[sl], sl


def add_context(ax, ccrs, cfeature, lon, lat):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    for adder in (
        lambda: ax.coastlines(resolution="10m", linewidth=0.45),
        lambda: ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.7),
        lambda: ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4),
    ):
        try:
            adder()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] map feature skipped: {exc}")
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.plot(MANAUS_LON, MANAUS_LAT, marker="*", markersize=8, markerfacecolor="white",
            markeredgecolor="black", transform=ccrs.PlateCarree(), zorder=10)


def main():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print(f"[read] {STATE}")
    with open_fa(STATE) as ds:
        u = read_field(ds, "WIND.U.PHYS")
        v = read_field(ds, "WIND.V.PHYS")
        q = read_field(ds, "HUMI.SPECIFI")
        pressure = read_field(ds, "PRESSURE")
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)

    dp = compute_dp_pa(pressure.astype(np.float64)[None, ...])[0]
    qx, qy = column_integrated_vapor_flux(q, u, v, dp)

    qxc, sl = crop(qx, lon, lat, CROP)
    qyc = qy[sl]; lonc = lon[sl]; latc = lat[sl]
    print(f"[crop] {qxc.shape}  (radar rectangle)")

    mfc_raw = -horizontal_divergence_spherical(qxc, qyc, lonc, latc) * SECONDS_PER_DAY
    mfc_sm = -horizontal_divergence_spherical(
        box_smooth(qxc, 2), box_smooth(qyc, 2), lonc, latc) * SECONDS_PER_DAY

    for tag, m in (("raw", mfc_raw), ("smoothed-2cell", mfc_sm)):
        print(f"[{tag}] mean={np.nanmean(m):+.2f}  std={np.nanstd(m):.2f}  "
              f"min={np.nanmin(m):+.1f}  max={np.nanmax(m):+.1f}  mm/day")

    scale = float(np.nanpercentile(np.abs(mfc_raw[np.isfinite(mfc_raw)]), 98))
    norm = mcolors.TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4),
                             subplot_kw={"projection": proj}, constrained_layout=True)
    for ax, field, title in (
        (axes[0], mfc_raw, "Raw centred-difference divergence"),
        (axes[1], mfc_sm, "Flux smoothed 2 cells, then divergence"),
    ):
        im = ax.pcolormesh(lonc, latc, np.ma.masked_invalid(field), transform=proj,
                           shading="auto", cmap="BrBG", norm=norm)
        ax.set_title(title, fontsize=11, fontweight="bold")
        add_context(ax, ccrs, cfeature, lonc, latc)
        ax.text(0.02, 0.98, f"mean={np.nanmean(field):+.2f}  std={np.nanstd(field):.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, aspect=40)
    cbar.set_label("Moisture flux convergence (mm day$^{-1}$):  + import,  − export", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))

    fig.suptitle("Vertically integrated moisture flux convergence — single state "
                 "(C1M, 2014-01-01 20 UTC)", fontsize=13, fontweight="bold")
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
