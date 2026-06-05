#!/usr/bin/env python3
"""Vertically integrated moisture flux convergence: G1M versus C1M.

For every hourly raw-FA state file this reads model-level specific humidity
(HUMI.SPECIFI), zonal/meridional wind (WIND.U.PHYS, WIND.V.PHYS) and full
pressure (PRESSURE), then forms the mass-weighted column water-vapour flux

    Qx = (1/g) * sum_level(q * u * dp)        [kg m^-1 s^-1]
    Qy = (1/g) * sum_level(q * v * dp)

and time-averages Qx, Qy over the full two-year period at every grid point.
The vertically integrated moisture flux convergence is then

    MFC = -( d(Qx)/dx + d(Qy)/dy )            [kg m^-2 s^-1  ==  mm s^-1]

mapped here in mm/day so it compares directly with rainfall.

Sign convention
---------------
POSITIVE  = convergence, i.e. net moisture IMPORT into the column.
NEGATIVE  = divergence,  i.e. net moisture EXPORT out of the column.

If the G1M - C1M difference is negative over the region of interest, G1M
imports less moisture (the column is being starved), consistent with the
sustained multi-year drying.

The flux is integrated to the column first and the horizontal divergence is
taken on the time-mean column flux.  Because finite differencing is linear,
this equals the time mean of the column-flux divergence; it is the standard,
mass-consistent vertically integrated moisture flux divergence.

Wind, full pressure and full-column humidity exist only in the raw FA files
(they are not in the masked-NetCDF), so this MUST run under the `epygram`
conda env (faxarray).  See examples/run_moisture_flux_convergence.sh.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.analysis.derived import (
    column_integrated_vapor_flux,
    compute_dp_pa,
    horizontal_divergence_spherical,
)
from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.plotting.style import resolve_workers


from alaro_analysis.common.constants import RUNS_ROOT
DEFAULT_DATA_ROOT = RUNS_ROOT / "ALARO"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "moisture_flux_convergence" / "2years"
DEFAULT_PROCESSED_DIR = (
    RUNS_ROOT / "processed-data" / "moisture_flux_convergence" / "2years"
)

FLUX_VARS = ("WIND.U.PHYS", "WIND.V.PHYS", "HUMI.SPECIFI", "PRESSURE")
HOUR_RE = re.compile(r"\+(\d{4})(?:\.[^.]+)?$")
SECONDS_PER_DAY = 86400.0

FIGURE_NAME = "moisture_flux_convergence_c1m_g1m_diff_450dpi.png"
TEXT_NAME = "moisture_flux_convergence_data.txt"
FIELDS_NPZ = "moisture_flux_convergence_fields.npz"

MANAUS_LON = -60.0217
MANAUS_LAT = -3.1190

# Generous central-Amazon box, well inside the LAM relaxation/extension zone and
# away from the Andes. Colour scales are derived here so the basin signal stays
# visible instead of being saturated by extreme near-boundary / terrain values.
SCALE_LON_MIN, SCALE_LON_MAX = -66.0, -54.0
SCALE_LAT_MIN, SCALE_LAT_MAX = -9.0, 3.0

DEFAULT_EXPERIMENTS = ("control", "graupel")
DIFF_PAIR = ("graupel", "control")  # G1M - C1M


@dataclass(frozen=True)
class FluxMap:
    experiment: str
    lon: np.ndarray
    lat: np.ndarray
    mean_qx: np.ndarray
    mean_qy: np.ndarray
    counts: np.ndarray
    n_days: int
    n_files_seen: int
    n_files_used: int
    missing_files: int
    source: str


# ---------------------------------------------------------------------------
# Raw-FA reading
# ---------------------------------------------------------------------------


def open_fa(path: Path):
    import faxarray as fx

    return fx.open_dataset(str(path), variables=list(FLUX_VARS), stack_levels=True)


def data_var(ds, requested: str) -> str:
    if requested in ds.data_vars:
        return requested
    compact = requested.replace(".", "_")
    if compact in ds.data_vars:
        return compact
    token = requested.replace(".", "").replace("_", "").upper()
    for name in ds.data_vars:
        if name.replace("_", "").replace(".", "").upper() == token:
            return name
    raise KeyError(f"{requested!r} not found; available={list(ds.data_vars)}")


def read_field(ds, requested: str) -> np.ndarray:
    arr = ds[data_var(ds, requested)].isel(time=0).values
    return np.asarray(arr, dtype=np.float32)


def read_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open_fa(path) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    return lon, lat


def list_day_dirs(exp_root: Path, max_days: int | None) -> list[Path]:
    out = sorted(p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("pf"))
    if max_days is not None:
        out = out[:max_days]
    return out


def list_step_files(day_dir: Path, min_lead: int, max_lead: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(day_dir.iterdir()):
        if not path.is_file():
            continue
        match = HOUR_RE.search(path.name)
        if not match:
            continue
        lead = int(match.group(1))
        if min_lead <= lead <= max_lead:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Per-day worker
# ---------------------------------------------------------------------------


def process_day(task: tuple[str, str, int, int]) -> dict[str, Any]:
    experiment, day_dir_raw, min_lead, max_lead = task
    day_dir = Path(day_dir_raw)
    qx_sum: np.ndarray | None = None
    qy_sum: np.ndarray | None = None
    counts: np.ndarray | None = None
    n_files_seen = 0
    n_files_used = 0
    missing_files = 0
    warnings: list[str] = []

    for path in list_step_files(day_dir, min_lead, max_lead):
        n_files_seen += 1
        try:
            with open_fa(path) as ds:
                u = read_field(ds, "WIND.U.PHYS")
                v = read_field(ds, "WIND.V.PHYS")
                q = read_field(ds, "HUMI.SPECIFI")
                pressure = read_field(ds, "PRESSURE")
        except Exception as exc:  # noqa: BLE001
            missing_files += 1
            warnings.append(f"{experiment} {path}: read failed: {exc}")
            continue

        try:
            dp = compute_dp_pa(pressure.astype(np.float64)[None, ...])[0]
            qx, qy = column_integrated_vapor_flux(q, u, v, dp)
        except Exception as exc:  # noqa: BLE001
            missing_files += 1
            warnings.append(f"{experiment} {path}: compute failed: {exc}")
            continue

        valid = np.isfinite(qx) & np.isfinite(qy)
        if qx_sum is None:
            qx_sum = np.zeros(qx.shape, dtype=np.float64)
            qy_sum = np.zeros(qy.shape, dtype=np.float64)
            counts = np.zeros(qx.shape, dtype=np.int64)
        qx_sum[valid] += qx[valid]
        qy_sum[valid] += qy[valid]
        counts[valid] += 1
        n_files_used += 1

    return {
        "qx_sum": qx_sum,
        "qy_sum": qy_sum,
        "counts": counts,
        "n_files_seen": n_files_seen,
        "n_files_used": n_files_used,
        "missing_files": missing_files,
        "warnings": tuple(warnings),
    }


def checkpoint_path(args: argparse.Namespace, experiment: str) -> Path:
    p = cache_path(args, experiment)
    return p.with_name(f"{p.stem}.checkpoint.npz")


def save_checkpoint(path: Path, state: dict[str, Any], completed: set[str]) -> None:
    """Atomically persist partial accumulators + the set of finished days."""
    if state["qx"] is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            qx_sum=state["qx"],
            qy_sum=state["qy"],
            counts=state["counts"],
            completed_days=np.asarray(sorted(completed)),
            n_files_seen=np.asarray([state["seen"]], dtype=np.int64),
            n_files_used=np.asarray([state["used"]], dtype=np.int64),
            missing_files=np.asarray([state["missing"]], dtype=np.int64),
        )
    os.replace(tmp, path)  # atomic: a kill mid-write leaves the old checkpoint intact


def load_checkpoint(path: Path) -> tuple[dict[str, Any], set[str]]:
    with np.load(path, allow_pickle=False) as data:
        state = {
            "qx": np.asarray(data["qx_sum"], dtype=np.float64),
            "qy": np.asarray(data["qy_sum"], dtype=np.float64),
            "counts": np.asarray(data["counts"], dtype=np.int64),
            "seen": int(data["n_files_seen"][0]),
            "used": int(data["n_files_used"][0]),
            "missing": int(data["missing_files"][0]),
        }
        completed = {str(name) for name in data["completed_days"].tolist()}
    return state, completed


def build_flux_map(args: argparse.Namespace, experiment: str) -> tuple[FluxMap, list[str]]:
    exp_root = args.data_root / experiment / "untar-output"
    days = list_day_dirs(exp_root, args.max_days)
    if not days:
        raise RuntimeError(f"No raw-FA day directories under {exp_root}")

    sample_steps = list_step_files(days[0], args.min_lead_hour, args.max_lead_hour)
    if not sample_steps:
        raise RuntimeError(f"No step files in {days[0]}")
    lon, lat = read_lon_lat(sample_steps[0])

    state: dict[str, Any] = {
        "qx": None, "qy": None, "counts": None, "seen": 0, "used": 0, "missing": 0,
    }
    completed: set[str] = set()
    warnings: list[str] = []

    ckpt = checkpoint_path(args, experiment)
    if ckpt.exists() and not args.recompute:
        state, completed = load_checkpoint(ckpt)
        print(f"[{experiment}] resuming from checkpoint: {len(completed)} days done", flush=True)

    todo = [day for day in days if day.name not in completed]
    workers = resolve_workers(args.workers)
    print(
        f"[{experiment}] {len(days)} days total, {len(todo)} to do, "
        f"leads {args.min_lead_hour}..{args.max_lead_hour}, {workers} workers",
        flush=True,
    )

    def accumulate(day_name: str, part: dict[str, Any]) -> None:
        if part["qx_sum"] is not None:
            if state["qx"] is None:
                state["qx"] = np.zeros_like(part["qx_sum"])
                state["qy"] = np.zeros_like(part["qy_sum"])
                state["counts"] = np.zeros_like(part["counts"])
            state["qx"] += part["qx_sum"]
            state["qy"] += part["qy_sum"]
            state["counts"] += part["counts"]
        state["seen"] += int(part["n_files_seen"])
        state["used"] += int(part["n_files_used"])
        state["missing"] += int(part["missing_files"])
        completed.add(day_name)
        warnings.extend(part["warnings"])

    if todo:
        done = 0
        if workers <= 1:
            for day in todo:
                accumulate(day.name, process_day(
                    (experiment, str(day), args.min_lead_hour, args.max_lead_hour)))
                done += 1
                if done % args.checkpoint_every == 0:
                    save_checkpoint(ckpt, state, completed)
                if done % args.progress_every == 0 or done == len(todo):
                    print(f"[{experiment}] {len(completed)}/{len(days)} days; {state['used']} files", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                fut_to_day = {
                    pool.submit(
                        process_day,
                        (experiment, str(day), args.min_lead_hour, args.max_lead_hour),
                    ): day.name
                    for day in todo
                }
                for future in as_completed(fut_to_day):
                    accumulate(fut_to_day[future], future.result())
                    done += 1
                    if done % args.checkpoint_every == 0:
                        save_checkpoint(ckpt, state, completed)
                        print(f"[{experiment}] checkpoint @ {len(completed)} days", flush=True)
                    if done % args.progress_every == 0 or done == len(todo):
                        print(f"[{experiment}] {len(completed)}/{len(days)} days; {state['used']} files", flush=True)
        save_checkpoint(ckpt, state, completed)

    if state["qx"] is None:
        raise RuntimeError("No valid files were processed.")

    counts = state["counts"]
    mean_qx = np.full(counts.shape, np.nan, dtype=np.float64)
    mean_qy = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    mean_qx[ok] = state["qx"][ok] / counts[ok]
    mean_qy[ok] = state["qy"][ok] / counts[ok]

    flux_map = FluxMap(
        experiment=experiment,
        lon=lon,
        lat=lat,
        mean_qx=mean_qx,
        mean_qy=mean_qy,
        counts=counts,
        n_days=len(days),
        n_files_seen=state["seen"],
        n_files_used=state["used"],
        missing_files=state["missing"],
        source=str(exp_root),
    )
    return flux_map, warnings


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_path(args: argparse.Namespace, experiment: str) -> Path:
    day_tag = "all-days" if args.max_days is None else f"{args.max_days}days"
    return (
        args.processed_dir
        / f"{experiment}_column_vapor_flux_lead{args.min_lead_hour}"
        f"_to{args.max_lead_hour}_{day_tag}.npz"
    )


def save_cache(path: Path, fm: FluxMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment=np.asarray(fm.experiment),
        lon=fm.lon,
        lat=fm.lat,
        mean_qx=fm.mean_qx,
        mean_qy=fm.mean_qy,
        counts=fm.counts,
        n_days=np.asarray(fm.n_days, dtype=np.int64),
        n_files_seen=np.asarray(fm.n_files_seen, dtype=np.int64),
        n_files_used=np.asarray(fm.n_files_used, dtype=np.int64),
        missing_files=np.asarray(fm.missing_files, dtype=np.int64),
        source=np.asarray(fm.source),
    )


def load_cache(path: Path) -> FluxMap:
    with np.load(path, allow_pickle=False) as data:
        return FluxMap(
            experiment=str(data["experiment"]),
            lon=np.asarray(data["lon"], dtype=np.float64),
            lat=np.asarray(data["lat"], dtype=np.float64),
            mean_qx=np.asarray(data["mean_qx"], dtype=np.float64),
            mean_qy=np.asarray(data["mean_qy"], dtype=np.float64),
            counts=np.asarray(data["counts"], dtype=np.int64),
            n_days=int(data["n_days"]),
            n_files_seen=int(data["n_files_seen"]),
            n_files_used=int(data["n_files_used"]),
            missing_files=int(data["missing_files"]),
            source=str(data["source"]),
        )


def get_flux_map(args: argparse.Namespace, experiment: str) -> FluxMap:
    path = cache_path(args, experiment)
    if path.exists() and not args.recompute:
        print(f"[{experiment}] using cache {path}", flush=True)
        return load_cache(path)
    fm, warnings = build_flux_map(args, experiment)
    save_cache(path, fm)
    ckpt = checkpoint_path(args, experiment)
    if ckpt.exists():
        ckpt.unlink()  # final cache supersedes the checkpoint
    print(
        f"[{experiment}] saved cache {path} "
        f"(files used {fm.n_files_used}/{fm.n_files_seen}, missing {fm.missing_files})",
        flush=True,
    )
    if warnings:
        warn_path = path.with_name(f"{experiment}_warnings.txt")
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[{experiment}] {len(warnings)} warnings -> {warn_path}", flush=True)
    return fm


# ---------------------------------------------------------------------------
# Derived convergence + region statistics
# ---------------------------------------------------------------------------


def crop_fluxmap(fm: FluxMap, bbox: tuple[float, float, float, float]) -> FluxMap:
    """Crop a FluxMap to a lon/lat bounding box (the physical model domain).

    The lateral-boundary relaxation/coupling ("init") zone of the LAM is not
    free model dynamics, so the analysis must exclude it. Because the column
    flux is a per-column (local) quantity, cropping the flux and then taking the
    divergence is identical to cropping the inputs before the vertical integral.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    inside = (
        (fm.lon >= lon_min)
        & (fm.lon <= lon_max)
        & (fm.lat >= lat_min)
        & (fm.lat <= lat_max)
    )
    if not inside.any():
        raise ValueError(f"Crop bbox {bbox} selects no grid points.")
    ys, xs = np.where(inside)
    sl = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    return FluxMap(
        experiment=fm.experiment,
        lon=fm.lon[sl],
        lat=fm.lat[sl],
        mean_qx=fm.mean_qx[sl],
        mean_qy=fm.mean_qy[sl],
        counts=fm.counts[sl],
        n_days=fm.n_days,
        n_files_seen=fm.n_files_seen,
        n_files_used=fm.n_files_used,
        missing_files=fm.missing_files,
        source=fm.source,
    )


def mfc_mm_day(fm: FluxMap) -> np.ndarray:
    div = horizontal_divergence_spherical(fm.mean_qx, fm.mean_qy, fm.lon, fm.lat)
    return -div * SECONDS_PER_DAY


def trim_border(field: np.ndarray, n: int) -> np.ndarray:
    """Mask the outer ``n`` grid cells (LAM relaxation/extension zone) to NaN."""
    if n <= 0:
        return field
    out = np.array(field, dtype=np.float64, copy=True)
    out[:n, :] = np.nan
    out[-n:, :] = np.nan
    out[:, :n] = np.nan
    out[:, -n:] = np.nan
    return out


def region_mask(lon: np.ndarray, lat: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bounds
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


def masked_mean(field: np.ndarray, mask: np.ndarray) -> float:
    sel = field[mask & np.isfinite(field)]
    return float(np.mean(sel)) if sel.size else float("nan")


def symmetric_scale(arrays: Sequence[np.ndarray], percentile: float) -> float:
    parts = [np.abs(np.asarray(a, dtype=np.float64)).ravel() for a in arrays]
    finite = np.concatenate(parts)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    scale = float(np.nanpercentile(finite, percentile))
    return scale if scale > 0.0 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def add_map_context(ax, ccrs, cfeature, lon: np.ndarray, lat: np.ndarray) -> None:
    ax.set_extent(
        [
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="10m", linewidth=0.45)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.plot(
        MANAUS_LON,
        MANAUS_LAT,
        marker="*",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


def plot_maps(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    mfc_c1m: np.ndarray,
    mfc_g1m: np.ndarray,
    mfc_diff: np.ndarray,
    abs_scale: float,
    diff_scale: float,
    domain_means: dict[str, float],
    output_path: Path,
    dpi: int,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    projection = ccrs.PlateCarree()
    abs_norm = mcolors.TwoSlopeNorm(vmin=-abs_scale, vcenter=0.0, vmax=abs_scale)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_scale, vcenter=0.0, vmax=diff_scale)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.2),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    panels = [
        ("C1M", mfc_c1m, abs_norm),
        ("G1M", mfc_g1m, abs_norm),
        ("G1M − C1M", mfc_diff, diff_norm),
    ]
    abs_image = None
    diff_image = None
    for idx, (title, field, norm) in enumerate(panels):
        ax = axes[idx]
        image = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_invalid(field),
            transform=projection,
            shading="auto",
            cmap="BrBG",
            norm=norm,
        )
        if idx < 2:
            abs_image = image
        else:
            diff_image = image
        ax.set_title(title, fontsize=12, fontweight="bold")
        add_map_context(ax, ccrs, cfeature, lon, lat)
        ax.text(
            0.02,
            0.98,
            f"mean = {domain_means[title]:+.2f} mm day$^{{-1}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    cbar_abs = fig.colorbar(
        abs_image, ax=axes[:2], orientation="horizontal", fraction=0.05, pad=0.06, aspect=40
    )
    cbar_abs.set_label(
        "Moisture flux convergence (mm day$^{-1}$):  + import,  − export", fontsize=10
    )
    cbar_abs.ax.tick_params(labelsize=8)
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    cbar_diff = fig.colorbar(
        diff_image, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.06, aspect=22
    )
    cbar_diff.set_label("G1M − C1M (mm day$^{-1}$)", fontsize=10)
    cbar_diff.ax.tick_params(labelsize=8)
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    fig.suptitle(
        "Vertically integrated moisture flux convergence", fontsize=14, fontweight="bold"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text + field outputs
# ---------------------------------------------------------------------------


def write_outputs(
    *,
    output_dir: Path,
    figure_path: Path,
    flux_maps: dict[str, FluxMap],
    mfc: dict[str, np.ndarray],
    mfc_diff: np.ndarray,
    abs_scale: float,
    diff_scale: float,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    graupel, control = DIFF_PAIR
    lon = flux_maps[control].lon
    lat = flux_maps[control].lat

    # Region budget rows.
    interior = np.isfinite(mfc_diff)  # finite where centred differences exist
    regions: list[tuple[str, np.ndarray]] = [("radar-rectangle domain (interior)", interior)]

    text_path = output_dir / TEXT_NAME
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("w", encoding="utf-8") as fh:
        fh.write("Vertically integrated moisture flux convergence (G1M vs C1M)\n")
        fh.write("============================================================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Gridded fields (NetCDF-free): {output_dir / FIELDS_NPZ}\n\n")
        fh.write("Method\n------\n")
        fh.write("Qx = (1/g) sum_level(q * u * dp); Qy = (1/g) sum_level(q * v * dp).\n")
        fh.write("dp from PRESSURE via midpoint half-levels; q clipped at 0.\n")
        fh.write("Qx, Qy time-averaged over all hourly states, then\n")
        fh.write("MFC = -(dQx/dx + dQy/dy) on the spherical metric, in mm/day.\n")
        fh.write("Inputs: WIND.U.PHYS, WIND.V.PHYS, HUMI.SPECIFI, PRESSURE (raw FA).\n")
        fh.write(
            f"Forecast lead hours used: {args.min_lead_hour}..{args.max_lead_hour}.\n"
        )
        fh.write(
            "Analysis crop (rectangle bounding the radar mask, inside the LAM physical "
            f"domain): lon[{args.crop_bbox[0]:g}, {args.crop_bbox[1]:g}], "
            f"lat[{args.crop_bbox[2]:g}, {args.crop_bbox[3]:g}].\n"
        )
        fh.write("Sign: POSITIVE = convergence (import); NEGATIVE = divergence (export).\n\n")
        fh.write("Sources\n-------\n")
        for exp, fm in flux_maps.items():
            fh.write(
                f"{EXPERIMENT_LABELS[exp]} ({exp}): days={fm.n_days}, "
                f"files_used={fm.n_files_used}/{fm.n_files_seen}, "
                f"missing={fm.missing_files}, source={fm.source}\n"
            )
        fh.write(f"\nColour limits: abs +/-{abs_scale:.6g}, diff +/-{diff_scale:.6g} mm/day.\n\n")
        fh.write("Region-mean moisture flux convergence (mm/day)\n")
        fh.write("----------------------------------------------\n")
        fh.write("region,c1m,g1m,g1m_minus_c1m\n")
        for name, mask in regions:
            c1 = masked_mean(mfc[control], mask)
            g1 = masked_mean(mfc[graupel], mask)
            fh.write(f"{name},{c1:.6g},{g1:.6g},{g1 - c1:.6g}\n")

    fields_path = output_dir / FIELDS_NPZ
    np.savez_compressed(
        fields_path,
        lon=lon,
        lat=lat,
        mfc_c1m=mfc[control],
        mfc_g1m=mfc[graupel],
        mfc_g1m_minus_c1m=mfc_diff,
        mean_qx_c1m=flux_maps[control].mean_qx,
        mean_qy_c1m=flux_maps[control].mean_qy,
        mean_qx_g1m=flux_maps[graupel].mean_qx,
        mean_qy_g1m=flux_maps[graupel].mean_qy,
        counts_c1m=flux_maps[control].counts,
        counts_g1m=flux_maps[graupel].counts,
    )
    return text_path, fields_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        help="Experiments to process; must include 'control' and 'graupel'.",
    )
    parser.add_argument("--min-lead-hour", type=int, default=0)
    parser.add_argument(
        "--max-lead-hour",
        type=int,
        default=23,
        help="Max forecast lead hour; 23 avoids duplicate +0024 states.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Persist partial accumulators every N finished days so a killed run "
        "resumes instead of restarting.",
    )
    parser.add_argument(
        "--crop-bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=[-61.4, -58.6, -4.6, -1.7],
        help="Rectangular domain crop applied to the cached flux before the divergence. "
        "Default is the rectangle that bounds the radar mask (every radar-mask point lies "
        "inside it, plus a small margin for the divergence stencil); it also sits well "
        "inside the LAM physical domain. Re-plot cheaply from cache to change it "
        "(e.g. the wider central-Amazon ROI: -67 -53 -10 4).",
    )
    parser.add_argument(
        "--trim-border",
        type=int,
        default=0,
        help="Extra ring of grid cells to mask inside the crop (usually 0; the crop "
        "already removes the relaxation/extension zone).",
    )
    parser.add_argument("--abs-percentile", type=float, default=98.0)
    parser.add_argument("--diff-percentile", type=float, default=98.0)
    parser.add_argument("--abs-scale", type=float, default=None)
    parser.add_argument("--diff-scale", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.processed_dir = args.processed_dir.resolve()
    args.data_root = args.data_root.resolve()

    experiments = list(dict.fromkeys(args.experiments))
    for required in DIFF_PAIR:
        if required not in experiments:
            experiments.append(required)

    # Flux caches are full-domain (the per-column integral is local); crop to the
    # physical model domain BEFORE the divergence so the LAM relaxation/coupling
    # (init) edges never enter the analysis, scaling, region means or the map.
    flux_maps_full = {exp: get_flux_map(args, exp) for exp in experiments}
    crop = tuple(args.crop_bbox)
    flux_maps = {exp: crop_fluxmap(fm, crop) for exp, fm in flux_maps_full.items()}
    mfc = {exp: trim_border(mfc_mm_day(fm), args.trim_border) for exp, fm in flux_maps.items()}

    graupel, control = DIFF_PAIR
    mfc_diff = mfc[graupel] - mfc[control]

    lon = flux_maps[control].lon
    lat = flux_maps[control].lat

    # Derive colour scales over the central-Amazon box so the basin signal is
    # visible; extreme near-boundary / Andes values saturate rather than dominate.
    # (For the default radar-rectangle crop this box already contains the whole
    # domain, so the scale comes from the full crop.)
    scale_box = region_mask(lon, lat, (SCALE_LON_MIN, SCALE_LON_MAX, SCALE_LAT_MIN, SCALE_LAT_MAX))

    def boxed(field: np.ndarray) -> np.ndarray:
        return np.where(scale_box, field, np.nan)

    abs_scale = args.abs_scale or symmetric_scale(
        [boxed(mfc[control]), boxed(mfc[graupel])], args.abs_percentile
    )
    diff_scale = args.diff_scale or symmetric_scale([boxed(mfc_diff)], args.diff_percentile)

    # Mean over the whole radar-rectangle crop (interior, where the divergence is
    # defined). This is the headline number; no sub-box.
    domain_means = {
        "C1M": float(np.nanmean(mfc[control])),
        "G1M": float(np.nanmean(mfc[graupel])),
        "G1M − C1M": float(np.nanmean(mfc_diff)),
    }

    figure_path = args.output_dir / FIGURE_NAME
    plot_maps(
        lon=lon,
        lat=lat,
        mfc_c1m=mfc[control],
        mfc_g1m=mfc[graupel],
        mfc_diff=mfc_diff,
        abs_scale=abs_scale,
        diff_scale=diff_scale,
        domain_means=domain_means,
        output_path=figure_path,
        dpi=args.dpi,
    )
    text_path, fields_path = write_outputs(
        output_dir=args.output_dir,
        figure_path=figure_path,
        flux_maps=flux_maps,
        mfc=mfc,
        mfc_diff=mfc_diff,
        abs_scale=abs_scale,
        diff_scale=diff_scale,
        args=args,
    )

    print(f"[saved] {figure_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)
    print(f"[saved] {fields_path}", flush=True)
    print(
        f"[result] Radar-rectangle MFC: C1M={domain_means['C1M']:+.3f}, "
        f"G1M={domain_means['G1M']:+.3f}, "
        f"G1M-C1M={domain_means['G1M − C1M']:+.3f} mm/day",
        flush=True,
    )
    verdict = (
        "G1M imports LESS moisture (drying-consistent)"
        if domain_means["G1M − C1M"] < 0
        else "G1M imports MORE moisture"
    )
    print(f"[result] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
