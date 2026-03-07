#!/usr/bin/env python3
"""
Plot South America map with:
1) Raw FA model domain boundary.
2) Converted masked NetCDF overlay (4 km masked footprint or values).

Output is saved at high resolution (default: 450 dpi).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "cartopy is required for this script (borders/rivers map layers)."
    ) from exc

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import faxarray as fx

from alaro_analysis.data.dataset_io import select_data_var_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw South America map with FA domain boundary and converted "
            "masked NetCDF overlay."
        )
    )
    parser.add_argument("--fa-file", type=Path, required=True, help="Path to raw FA file.")
    parser.add_argument(
        "--masked-nc-file",
        type=Path,
        required=True,
        help="Path to converted masked NetCDF file (4 km output).",
    )
    parser.add_argument(
        "--fa-reference-var",
        default="PRESSURE",
        help="FA variable used only to open lon/lat grid (default: PRESSURE).",
    )
    parser.add_argument(
        "--masked-var",
        default=None,
        help="Variable to plot from masked NetCDF. Default: first data variable.",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=("footprint", "values"),
        default="footprint",
        help="Overlay masked file as finite-data footprint or values (default: footprint).",
    )
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help=(
            "Do not draw filled overlay data; only draw analysis-domain boundary "
            "from finite points in masked NetCDF."
        ),
    )
    parser.add_argument(
        "--fill-analysis-mask",
        action="store_true",
        help=(
            "Draw a semi-transparent binary analysis mask (1 for finite cells) "
            "without saving any intermediate mask file."
        ),
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.30,
        help="Alpha for binary analysis-mask fill (default: 0.30).",
    )
    parser.add_argument(
        "--hide-fa-boundary",
        action="store_true",
        help="Hide raw FA domain boundary line.",
    )
    parser.add_argument(
        "--analysis-boundary-color",
        default="#ff7f0e",
        help="Line color for analysis-domain boundary (default: #ff7f0e).",
    )
    parser.add_argument(
        "--analysis-boundary-width",
        type=float,
        default=1.8,
        help="Line width for analysis-domain boundary (default: 1.8).",
    )
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=(-90.0, -30.0, -60.0, 20.0),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Map extent (default: South America).",
    )
    parser.add_argument(
        "--main-extent-mode",
        choices=("auto", "manual"),
        default="auto",
        help="Main-map extent mode: auto-fit around domain boundaries or use --extent.",
    )
    parser.add_argument(
        "--main-pad-deg",
        type=float,
        default=2.0,
        help="Padding in degrees for auto main extent (default: 2.0).",
    )
    parser.add_argument(
        "--inset",
        action="store_true",
        help="Add zoom-out inset map (South America context).",
    )
    parser.add_argument(
        "--inset-extent",
        nargs=4,
        type=float,
        default=(-90.0, -30.0, -60.0, 20.0),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Inset extent (default: South America).",
    )
    parser.add_argument(
        "--inset-position",
        nargs=4,
        type=float,
        default=(0.74, 0.56, 0.24, 0.34),
        metavar=("X0", "Y0", "W", "H"),
        help="Inset position in figure fraction (default: 0.74 0.56 0.24 0.34).",
    )
    parser.add_argument(
        "--inset-title",
        default="",
        help="Optional inset title (default: empty).",
    )
    parser.add_argument(
        "--legend-location",
        choices=("outside", "inside", "none"),
        default="outside",
        help="Legend placement (default: outside).",
    )
    parser.add_argument(
        "--no-gridlines",
        action="store_true",
        help="Disable map gridlines/labels for a cleaner figure.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.0, 8.0),
        metavar=("W", "H"),
        help="Figure size inches (default: 10 8).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG file path.",
    )
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI (default: 450).")
    return parser.parse_args()


def _to_2d_field(da: xr.DataArray) -> xr.DataArray:
    out = da
    for dim in list(out.dims):
        if dim.lower() in {"y", "x", "lat", "lon", "latitude", "longitude"}:
            continue
        out = out.isel({dim: 0})
    if out.ndim != 2:
        raise ValueError(
            f"Expected a 2D field after selecting first non-spatial indices, got dims={out.dims}"
        )
    return out


def _load_fa_lon_lat(fa_file: Path, fa_reference_var: str) -> tuple[np.ndarray, np.ndarray]:
    ds = None
    try:
        ds = fx.open_dataset(
            str(fa_file),
            variables=[fa_reference_var],
            stack_levels=True,
        )
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        # Fallback: auto-pick first FA variable.
        fa = fx.FADataset(str(fa_file))
        try:
            variables = list(fa.variables)
        finally:
            fa.close()
        if not variables:
            raise RuntimeError(f"No variables found in FA file: {fa_file}") from exc
        picked = variables[0]
        ds = fx.open_dataset(str(fa_file), variables=[picked], stack_levels=True)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    finally:
        if ds is not None:
            ds.close()
    return lon, lat


def _grid_boundary(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_lon = lon[0, :]
    top_lat = lat[0, :]
    right_lon = lon[1:, -1]
    right_lat = lat[1:, -1]
    bot_lon = lon[-1, -2::-1]
    bot_lat = lat[-1, -2::-1]
    left_lon = lon[-2:0:-1, 0]
    left_lat = lat[-2:0:-1, 0]
    boundary_lon = np.concatenate([top_lon, right_lon, bot_lon, left_lon, top_lon[:1]])
    boundary_lat = np.concatenate([top_lat, right_lat, bot_lat, left_lat, top_lat[:1]])
    return boundary_lon, boundary_lat


def _load_masked_overlay(masked_nc_file: Path, masked_var: Optional[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with xr.open_dataset(masked_nc_file, decode_times=False) as ds:
        var_name = select_data_var_name(ds, masked_var)
        field = _to_2d_field(ds[var_name])

        lon = None
        lat = None
        if "lon" in ds:
            lon = ds["lon"]
        elif "lon" in ds.coords:
            lon = ds.coords["lon"]
        if "lat" in ds:
            lat = ds["lat"]
        elif "lat" in ds.coords:
            lat = ds.coords["lat"]

        if lon is None or lat is None:
            raise ValueError(
                "Masked NetCDF must provide lon/lat in variables or coords."
            )

        lon_arr = np.asarray(lon.values, dtype=np.float64)
        lat_arr = np.asarray(lat.values, dtype=np.float64)
        data = np.asarray(field.values, dtype=np.float64)

        if lon_arr.ndim == 1 and lat_arr.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_arr, lat_arr)
        elif lon_arr.ndim == 2 and lat_arr.ndim == 2:
            lon2d, lat2d = lon_arr, lat_arr
        else:
            raise ValueError("Unsupported lon/lat layout in masked NetCDF.")

        if data.shape != lon2d.shape:
            raise ValueError(
                f"Overlay field shape mismatch: data={data.shape}, lon/lat={lon2d.shape}."
            )
        return lon2d, lat2d, data, var_name


def _bounds_from_points(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float] | None:
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(finite):
        return None
    lon_f = lon[finite]
    lat_f = lat[finite]
    return (
        float(np.nanmin(lon_f)),
        float(np.nanmax(lon_f)),
        float(np.nanmin(lat_f)),
        float(np.nanmax(lat_f)),
    )


def _merge_bounds(bounds_list: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    lon_min = min(b[0] for b in bounds_list)
    lon_max = max(b[1] for b in bounds_list)
    lat_min = min(b[2] for b in bounds_list)
    lat_max = max(b[3] for b in bounds_list)
    return lon_min, lon_max, lat_min, lat_max


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    pad_deg: float,
) -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = bounds
    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg
    return lon_min, lon_max, lat_min, lat_max


def _draw_base(ax) -> None:
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfefff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f8f5ee", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="#444444", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7, edgecolor="#444444", zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"), linewidth=0.6, edgecolor="#4f81bd", alpha=0.8, zorder=1)


def _draw_analysis_boundary(
    ax,
    overlay_lon: np.ndarray,
    overlay_lat: np.ndarray,
    analysis_footprint: np.ndarray,
    color: str,
    linewidth: float,
) -> None:
    ax.contour(
        overlay_lon,
        overlay_lat,
        np.where(np.isfinite(analysis_footprint), 1.0, 0.0),
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )


def _draw_analysis_fill(
    ax,
    overlay_lon: np.ndarray,
    overlay_lat: np.ndarray,
    analysis_footprint: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    cmap = ListedColormap([color])
    ax.pcolormesh(
        overlay_lon,
        overlay_lat,
        analysis_footprint,
        cmap=cmap,
        shading="auto",
        alpha=float(alpha),
        transform=ccrs.PlateCarree(),
        zorder=5,
    )


def main() -> None:
    args = parse_args()

    if not args.fa_file.exists():
        raise FileNotFoundError(f"FA file not found: {args.fa_file}")
    if not args.masked_nc_file.exists():
        raise FileNotFoundError(f"Masked NetCDF file not found: {args.masked_nc_file}")

    fa_lon, fa_lat = _load_fa_lon_lat(args.fa_file, args.fa_reference_var)
    domain_lon, domain_lat = _grid_boundary(fa_lon, fa_lat)

    overlay_lon, overlay_lat, overlay_data, overlay_var = _load_masked_overlay(
        masked_nc_file=args.masked_nc_file,
        masked_var=args.masked_var,
    )
    analysis_footprint = np.where(np.isfinite(overlay_data), 1.0, np.nan)
    has_analysis_boundary = np.isfinite(analysis_footprint).any()

    fig = plt.figure(figsize=tuple(args.figsize))
    if args.inset:
        # Reserve room for inset at right to keep title and map clean.
        ax = fig.add_axes([0.06, 0.11, 0.72, 0.81], projection=ccrs.PlateCarree())
    else:
        ax = fig.add_axes([0.07, 0.09, 0.88, 0.84], projection=ccrs.PlateCarree())
    _draw_base(ax)

    # Main extent
    if args.main_extent_mode == "auto":
        bounds = []
        fa_bounds = _bounds_from_points(domain_lon, domain_lat)
        if fa_bounds is not None:
            bounds.append(fa_bounds)
        if has_analysis_boundary:
            analysis_bounds = _bounds_from_points(
                overlay_lon[np.isfinite(analysis_footprint)],
                overlay_lat[np.isfinite(analysis_footprint)],
            )
            if analysis_bounds is not None:
                bounds.append(analysis_bounds)
        if bounds:
            extent_main = _expand_bounds(_merge_bounds(bounds), pad_deg=float(args.main_pad_deg))
        else:
            extent_main = tuple(args.extent)
    else:
        extent_main = tuple(args.extent)
    ax.set_extent(extent_main, crs=ccrs.PlateCarree())

    if args.boundary_only:
        if args.fill_analysis_mask and has_analysis_boundary:
            _draw_analysis_fill(
                ax=ax,
                overlay_lon=overlay_lon,
                overlay_lat=overlay_lat,
                analysis_footprint=analysis_footprint,
                color=args.analysis_boundary_color,
                alpha=args.mask_alpha,
            )
        if has_analysis_boundary:
            _draw_analysis_boundary(
                ax=ax,
                overlay_lon=overlay_lon,
                overlay_lat=overlay_lat,
                analysis_footprint=analysis_footprint,
                color=args.analysis_boundary_color,
                linewidth=args.analysis_boundary_width,
            )
    else:
        # Converted masked data overlay.
        if args.overlay_mode == "footprint":
            _draw_analysis_fill(
                ax=ax,
                overlay_lon=overlay_lon,
                overlay_lat=overlay_lat,
                analysis_footprint=analysis_footprint,
                color=args.analysis_boundary_color,
                alpha=args.mask_alpha,
            )
            if has_analysis_boundary:
                _draw_analysis_boundary(
                    ax=ax,
                    overlay_lon=overlay_lon,
                    overlay_lat=overlay_lat,
                    analysis_footprint=analysis_footprint,
                    color=args.analysis_boundary_color,
                    linewidth=max(1.0, args.analysis_boundary_width),
                )
        else:
            mesh = ax.pcolormesh(
                overlay_lon,
                overlay_lat,
                overlay_data,
                cmap="viridis",
                shading="auto",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.86, pad=0.02)
            cbar.set_label(f"{overlay_var} (masked NetCDF values)")
            if has_analysis_boundary:
                _draw_analysis_boundary(
                    ax=ax,
                    overlay_lon=overlay_lon,
                    overlay_lat=overlay_lat,
                    analysis_footprint=analysis_footprint,
                    color=args.analysis_boundary_color,
                    linewidth=max(1.0, args.analysis_boundary_width),
                )

    # FA domain boundary.
    if not args.hide_fa_boundary:
        ax.plot(
            domain_lon,
            domain_lat,
            color="black",
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            zorder=7,
            label="Raw FA domain boundary",
        )

    # Legend helpers.
    if has_analysis_boundary:
        ax.plot(
            [],
            [],
            color=args.analysis_boundary_color,
            linewidth=args.analysis_boundary_width,
            label="Analysis-domain boundary (masked NetCDF footprint)",
        )
    if (not args.boundary_only) and args.overlay_mode == "footprint":
        ax.plot([], [], color=args.analysis_boundary_color, linewidth=4.0, alpha=0.6, label="4 km masked-data footprint")

    if not args.no_gridlines:
        grid = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        grid.top_labels = False
        grid.right_labels = False

    # Inset: zoom-out context
    if args.inset:
        inset_pos = tuple(float(v) for v in args.inset_position)
        inset_ax = fig.add_axes(inset_pos, projection=ccrs.PlateCarree())
        _draw_base(inset_ax)
        inset_ax.patch.set_facecolor("white")
        inset_ax.patch.set_alpha(0.96)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#222222")
        inset_ax.set_extent(tuple(args.inset_extent), crs=ccrs.PlateCarree())

        if not args.hide_fa_boundary:
            inset_ax.plot(
                domain_lon,
                domain_lat,
                color="black",
                linewidth=1.3,
                transform=ccrs.PlateCarree(),
                zorder=7,
            )
        if has_analysis_boundary:
            _draw_analysis_boundary(
                ax=inset_ax,
                overlay_lon=overlay_lon,
                overlay_lat=overlay_lat,
                analysis_footprint=analysis_footprint,
                color=args.analysis_boundary_color,
                linewidth=max(1.0, args.analysis_boundary_width * 0.9),
            )

        # Rectangle showing main-map extent on inset.
        rect = Rectangle(
            (extent_main[0], extent_main[2]),
            extent_main[1] - extent_main[0],
            extent_main[3] - extent_main[2],
            fill=False,
            edgecolor="#2a2a2a",
            linewidth=1.1,
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=8,
        )
        inset_ax.add_patch(rect)
        if args.inset_title:
            inset_ax.set_title(args.inset_title, fontsize=8, pad=2)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    title = args.title
    if title is None:
        if args.boundary_only:
            title = "Domain Boundaries: FA Domain + Analysis Domain"
        else:
            title = f"South America Domain Map: Raw FA Boundary + Masked NetCDF Overlay ({overlay_var})"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    if args.legend_location == "outside" and handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            framealpha=0.95,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif args.legend_location == "inside" and handles:
        ax.legend(loc="lower left", framealpha=0.95, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[saved] {args.output}")
    print(f"FA file: {args.fa_file}")
    print(f"Masked NetCDF file: {args.masked_nc_file} (var={overlay_var})")


if __name__ == "__main__":
    main()
