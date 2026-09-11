#!/usr/bin/env python3
"""Column profile of the two raw-FA vertical velocity fields at one location.

VERT.VELOCIT is w = Dz/Dt in m/s (positive upward); VITESSE_VERT is
omega = DPi/Dt in Pa/s (negative upward).  They are the same motion, linked by
omega = -rho*g*w with rho = p / (Rd * T).  This script converts omega back to a
velocity,

    w_omega = -omega / (rho * g),

and plots w, w_omega and their difference over all 87 model levels for a single
hour, area-averaged over an N x N box of grid points around a target longitude
and latitude.

The conversion is applied per grid point before averaging, so the box mean is a
mean of velocities rather than a velocity built from mean p and T.

Requires the `epygram` conda environment:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import epygram


DATA_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/ALARO")
OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/vertical_velocity_delta"
)

RD = 287.06
G = 9.80665
DPI = 450
N_LEVELS = 87

# Manaus, central Amazon
DEFAULT_LON = -60.0217
DEFAULT_LAT = -3.1190


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile VERT.VELOCIT against VITESSE_VERT converted to m/s, "
            "area-averaged over a box of grid points."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment", default="2mom")
    parser.add_argument("--date", default="20140101")
    parser.add_argument("--lead", type=int, default=15, help="lead hour, 0..23")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument(
        "--box", type=int, default=4, help="box side in grid points; 1 = single point"
    )
    parser.add_argument(
        "--top-km",
        type=float,
        default=20.0,
        help="cut the plot off above this height; 0 keeps the whole column",
    )
    return parser.parse_args(argv)


def fa_path(data_root: Path, experiment: str, date: str, lead: int) -> Path:
    return data_root / experiment / "untar-output" / f"pf{date}" / f"pfABOFABOF+{lead:04d}"


def read_gridpoint(resource, name: str) -> np.ndarray:
    field = resource.readfield(name)
    if field.spectral:
        field.sp2gp()
    return np.asarray(field.getdata(), dtype=np.float64)


def box_slice(iy: int, ix: int, side: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Box of `side` grid points centred as closely as possible on (iy, ix)."""
    half = side // 2
    y0 = min(max(iy - half, 0), shape[0] - side)
    x0 = min(max(ix - half, 0), shape[1] - side)
    return slice(y0, y0 + side), slice(x0, x0 + side)


def strip_zeros(axis) -> None:
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    path = fa_path(args.data_root, args.experiment, args.date, args.lead)
    if not path.exists():
        raise SystemExit(f"FA file not found: {path}")

    epygram.init_env()
    resource = epygram.formats.resource(str(path), "r")
    try:
        geometry = resource.readfield("S001VERT.VELOCIT").geometry
        lon2d, lat2d = (np.asarray(a, dtype=np.float64) for a in geometry.get_lonlat_grid())
        dims = geometry.dimensions
        ny_c, nx_c = int(dims["Y_Czone"]), int(dims["X_Czone"])

        # nearest grid point inside the C zone
        dist = (lon2d[:ny_c, :nx_c] - args.lon) ** 2 + (lat2d[:ny_c, :nx_c] - args.lat) ** 2
        iy, ix = np.unravel_index(int(np.argmin(dist)), dist.shape)
        rows, cols = box_slice(iy, ix, args.box, (ny_c, nx_c))

        w = np.empty(N_LEVELS)
        w_omega = np.empty(N_LEVELS)
        omega = np.empty(N_LEVELS)
        pressure = np.empty(N_LEVELS)
        height = np.empty(N_LEVELS)
        for k in range(N_LEVELS):
            prefix = f"S{k + 1:03d}"
            w_k = read_gridpoint(resource, f"{prefix}VERT.VELOCIT")[rows, cols]
            om_k = read_gridpoint(resource, f"{prefix}VITESSE_VERT")[rows, cols]
            p_k = read_gridpoint(resource, f"{prefix}PRESSURE")[rows, cols]
            t_k = read_gridpoint(resource, f"{prefix}TEMPERATURE")[rows, cols]
            phi_k = read_gridpoint(resource, f"{prefix}GEOPOTENTIEL")[rows, cols]
            rho_k = p_k / (RD * t_k)
            w[k] = w_k.mean()
            omega[k] = om_k.mean()
            w_omega[k] = (-om_k / (rho_k * G)).mean()
            pressure[k] = p_k.mean()
            # geopotential height above mean sea level
            height[k] = (phi_k / G).mean()
    finally:
        resource.close()

    box_lon = lon2d[rows, cols]
    box_lat = lat2d[rows, cols]
    p_hpa = pressure / 100.0
    z_km = height / 1000.0
    delta = w - w_omega
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * delta / np.abs(w)
    rel[np.abs(w) < 0.01 * np.nanmax(np.abs(w))] = np.nan

    keep = z_km <= args.top_km if args.top_km > 0 else np.ones_like(z_km, bool)

    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(w[keep], z_km[keep], color="#1f77b4", lw=1.8, label="VERT.VELOCIT  ($w$)")
    ax.plot(
        w_omega[keep], z_km[keep], color="#d62728", lw=1.4, ls="--",
        label="VITESSE_VERT  ($-\\omega/\\rho g$)",
    )
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("Vertical velocity  (m s$^{-1}$)")
    ax.set_ylabel("Height  (km)")
    ax.set_title("Both fields as velocity", fontsize=11)
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    ax.plot(delta[keep], z_km[keep], color="#6a3d9a", lw=1.6)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("$w - (-\\omega/\\rho g)$  (m s$^{-1}$)")
    ax.set_title("Absolute difference", fontsize=11)

    ax = axes[2]
    ax.plot(rel[keep], z_km[keep], color="#ff7f0e", lw=1.6)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("Difference  (% of $|w|$)")
    ax.set_title("Relative difference", fontsize=11)

    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
        strip_zeros(ax.xaxis)
        strip_zeros(ax.yaxis)

    # pressure equivalent of the height ticks, for reference
    z_sorted = np.argsort(z_km)
    pressure_axis = axes[2].secondary_yaxis(
        "right",
        functions=(
            lambda z: np.interp(z, z_km[z_sorted], p_hpa[z_sorted]),
            lambda p: np.interp(p, p_hpa[z_sorted][::-1], z_km[z_sorted][::-1]),
        ),
    )
    pressure_axis.set_ylabel("Pressure  (hPa)", fontsize=10)
    strip_zeros(pressure_axis.yaxis)

    box_desc = (
        "single grid point"
        if args.box == 1
        else f"{args.box}x{args.box} grid-point mean"
    )
    fig.suptitle(
        f"ALARO {args.experiment} — geometric vs pressure vertical velocity over the "
        f"column\n{box_desc} centred on {box_lon.mean():.3f}°E, {box_lat.mean():.3f}°N",
        fontsize=13,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"vertical_velocity_profile_delta_{args.experiment}_{args.date}"
        f"_+{args.lead:04d}_{args.box}x{args.box}"
        f"_{abs(box_lon.mean()):.2f}W_{abs(box_lat.mean()):.2f}S"
    )
    figure_path = args.output_dir / f"{stem}.png"
    fig.savefig(figure_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"file        : {path}",
        f"box         : {box_desc}",
        f"lon range   : {box_lon.min():.4f} .. {box_lon.max():.4f}",
        f"lat range   : {box_lat.min():.4f} .. {box_lat.max():.4f}",
        "",
        f"{'lev':>4} {'z_km':>8} {'p_hPa':>9} {'w':>12} {'-om/(rho g)':>13} {'delta':>12} {'omega':>11} {'rel_%':>8}",
    ]
    for k in range(N_LEVELS):
        rel_text = f"{rel[k]:>8.2f}" if np.isfinite(rel[k]) else f"{'--':>8}"
        lines.append(
            f"{k + 1:>4} {z_km[k]:>8.3f} {p_hpa[k]:>9.2f} {w[k]:>12.6f} "
            f"{w_omega[k]:>13.6f} {delta[k]:>12.6f} {omega[k]:>11.5f} {rel_text}"
        )
    finite = np.isfinite(delta)
    lines += [
        "",
        f"max |delta|            : {np.nanmax(np.abs(delta)):.6f} m/s "
        f"at {z_km[np.nanargmax(np.abs(delta))]:.2f} km "
        f"({p_hpa[np.nanargmax(np.abs(delta))]:.1f} hPa)",
        f"rms delta              : {np.sqrt(np.nanmean(delta[finite] ** 2)):.6f} m/s",
        f"max |w|                : {np.nanmax(np.abs(w)):.6f} m/s",
        f"mean |delta| / mean |w|: {np.nanmean(np.abs(delta)) / np.nanmean(np.abs(w)):.4f}",
        f"corr(w, -omega/(rho g)): {np.corrcoef(w, w_omega)[0, 1]:.6f}",
    ]
    text_path = args.output_dir / f"{stem}.txt"
    text_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[-6:]))
    print(f"\nwrote {figure_path}")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()
