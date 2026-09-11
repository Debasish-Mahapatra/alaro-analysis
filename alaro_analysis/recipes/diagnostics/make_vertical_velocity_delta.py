#!/usr/bin/env python3
"""Delta between the two raw-FA vertical velocity fields at one time and level.

VERT.VELOCIT is the geometric vertical velocity w = Dz/Dt in m/s (positive
upward).  VITESSE_VERT is the vertical pressure velocity omega = DPi/Dt in Pa/s
(negative upward).  They describe the same motion, linked by omega = -rho*g*w
with rho = p / (Rd * T), so a raw subtraction is meaningless.  This script puts
both on a common footing by re-expressing omega as a velocity,

    w_omega = -omega / (rho * g),

and maps the residual delta_w = w - w_omega.

Both fields are spectral in the FA and are transformed to gridpoint space here.
The biperiodicization extension zone carries large non-physical values -- for
VITESSE_VERT its standard deviation is several times the physical one -- so the
domain is cropped to the C zone before plotting and before any statistic.

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

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.paths import runs_path

DATA_ROOT = runs_path("ALARO")
OUTPUT_DIR = runs_path("figures/vertical_velocity_delta")

RD = 287.06
G = 9.80665
DPI = 450


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map VERT.VELOCIT, VITESSE_VERT and their delta after unit "
            "conversion, at a single time and model level."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment", default="2mom")
    parser.add_argument("--date", default="20140101", help="YYYYMMDD of the pf day directory")
    parser.add_argument("--lead", type=int, default=15, help="lead hour, 0..23")
    parser.add_argument(
        "--level",
        type=int,
        default=51,
        help="FA model level S### (1 = model top, 87 = surface)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="keep the extension zone instead of cropping to the C zone",
    )
    return parse_configured_args(parser, "make_vertical_velocity_delta", argv=argv)


def fa_path(data_root: Path, experiment: str, date: str, lead: int) -> Path:
    return data_root / experiment / "untar-output" / f"pf{date}" / f"pfABOFABOF+{lead:04d}"


def read_gridpoint(resource, name: str) -> np.ndarray:
    """Read one FA field, transforming to gridpoint space when spectral."""
    field = resource.readfield(name)
    if field.spectral:
        field.sp2gp()
    return np.asarray(field.getdata(), dtype=np.float64)


def c_zone_slice(resource, name: str) -> tuple[slice, slice]:
    """Slices selecting the C zone, excluding the I and E zones."""
    dims = resource.readfield(name).geometry.dimensions
    y0 = int(dims["Y_CIoffset"])
    x0 = int(dims["X_CIoffset"])
    return (
        slice(y0, y0 + int(dims["Y_Czone"])),
        slice(x0, x0 + int(dims["X_Czone"])),
    )


def strip_zeros(cbar) -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    cbar.ax.xaxis.set_major_formatter(fmt)
    cbar.ax.yaxis.set_major_formatter(fmt)


def sym_limit(values: np.ndarray, pct: float = 99.5) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(finite), pct)), 1e-12)


def main(argv: list[str] | None = None) -> None:
    import epygram

    args = parse_args(argv)

    path = fa_path(args.data_root, args.experiment, args.date, args.lead)
    if not path.exists():
        raise SystemExit(f"FA file not found: {path}")

    epygram.init_env()
    resource = epygram.formats.resource(str(path), "r")
    try:
        prefix = f"S{args.level:03d}"
        w = read_gridpoint(resource, f"{prefix}VERT.VELOCIT")
        omega = read_gridpoint(resource, f"{prefix}VITESSE_VERT")
        pressure = read_gridpoint(resource, f"{prefix}PRESSURE")
        temperature = read_gridpoint(resource, f"{prefix}TEMPERATURE")
        lon, lat = resource.readfield(f"{prefix}VERT.VELOCIT").geometry.get_lonlat_grid()
        rows, cols = c_zone_slice(resource, f"{prefix}VERT.VELOCIT")
    finally:
        resource.close()

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    if not args.no_crop:
        w, omega = w[rows, cols], omega[rows, cols]
        pressure, temperature = pressure[rows, cols], temperature[rows, cols]
        lon, lat = lon[rows, cols], lat[rows, cols]

    rho = pressure / (RD * temperature)
    w_omega = -omega / (rho * G)
    delta_w = w - w_omega
    delta_omega = omega - (-rho * G * w)

    p_hpa = float(np.nanmean(pressure)) / 100.0
    lim_w = sym_limit(np.concatenate([w.ravel(), w_omega.ravel()]))
    lim_om = sym_limit(omega)
    lim_d = sym_limit(delta_w)

    panels = [
        (
            "VERT.VELOCIT  (w = Dz/Dt)",
            w,
            "RdBu_r",
            -lim_w,
            lim_w,
            "m s$^{-1}$  (positive = upward)",
        ),
        (
            "VITESSE_VERT  ($\\omega$ = D$\\Pi$/Dt)",
            omega,
            "RdBu",
            -lim_om,
            lim_om,
            "Pa s$^{-1}$  (negative = upward)",
        ),
        (
            "VITESSE_VERT as velocity  ($-\\omega/\\rho g$)",
            w_omega,
            "RdBu_r",
            -lim_w,
            lim_w,
            "m s$^{-1}$  (positive = upward)",
        ),
        (
            "Delta  (w $-$ $(-\\omega/\\rho g)$)",
            delta_w,
            "PuOr_r",
            -lim_d,
            lim_d,
            "m s$^{-1}$",
        ),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (title, data, cmap, vmin, vmax, cbar_label) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label(cbar_label, fontsize=9)
        strip_zeros(cbar)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")

    p_lo = float(np.nanmin(pressure)) / 100.0
    p_hi = float(np.nanmax(pressure)) / 100.0
    fig.suptitle(
        f"ALARO {args.experiment} — vertical velocity in geometric and pressure "
        f"coordinates on model level {args.level} "
        f"({p_lo:.0f}–{p_hi:.0f} hPa, mean {p_hpa:.0f} hPa)",
        fontsize=13,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"vertical_velocity_delta_{args.experiment}_{args.date}_+{args.lead:04d}_S{args.level:03d}"
    )
    figure_path = args.output_dir / f"{stem}.png"
    fig.savefig(figure_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    finite = np.isfinite(delta_w)
    rel = np.abs(delta_w[finite]).sum() / max(np.abs(w[finite]).sum(), 1e-30)
    lines = [
        f"file            : {path}",
        f"model level     : S{args.level:03d}   mean pressure {p_hpa:.1f} hPa",
        f"domain          : {'full grid' if args.no_crop else 'C zone'}  shape {w.shape}",
        "",
        f"w              min/max/std : {w.min():+.4f} {w.max():+.4f} {w.std():.4f}  m/s",
        f"omega          min/max/std : {omega.min():+.4f} {omega.max():+.4f} {omega.std():.4f}  Pa/s",
        f"-omega/(rho g) min/max/std : {w_omega.min():+.4f} {w_omega.max():+.4f} {w_omega.std():.4f}  m/s",
        "",
        f"delta_w        min/max/rms : {delta_w.min():+.4f} {delta_w.max():+.4f} {np.sqrt((delta_w**2).mean()):.4f}  m/s",
        f"delta_omega    min/max/rms : {delta_omega.min():+.4f} {delta_omega.max():+.4f} {np.sqrt((delta_omega**2).mean()):.4f}  Pa/s",
        f"mean |delta_w| / mean |w|  : {rel:.4f}",
        f"corr(w, -omega/(rho g))    : {np.corrcoef(w.ravel(), w_omega.ravel())[0, 1]:.6f}",
    ]
    text_path = args.output_dir / f"{stem}.txt"
    text_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {figure_path}")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()
