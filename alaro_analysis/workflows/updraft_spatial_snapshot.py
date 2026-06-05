"""Figure-6.12-style spatial snapshot of condensate transport and drafts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from alaro_analysis.common.figio import strip_cbar_zeros
import numpy as np
import xarray as xr

from alaro_analysis.common.constants import G


from alaro_analysis.common.constants import RUNS_ROOT
DEFAULT_ALARO_ROOT = RUNS_ROOT / "ALARO"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "updraft_spatial_snapshot_fig612_like"
EXPERIMENT_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
REQUIRED_VARIABLES = (
    "UD_OMEGA",
    "UD_MESH_FRAC",
    "DD_OMEGA",
    "DD_MESH_FRAC",
    "LIQUID_WATER",
    "SOLID_WATER",
    "GEOPOTENTIEL",
)


@dataclass(frozen=True)
class SnapshotSelection:
    day: str
    filename: str
    level_index: int
    level_value: int | None
    score: float
    updraft_max: float = np.nan
    downdraft_max: float = np.nan


@dataclass(frozen=True)
class SnapshotFields:
    lon: np.ndarray
    lat: np.ndarray
    height_m: np.ndarray
    cloud_transport_mg_m2_s: np.ndarray
    updraft_flux_signed: np.ndarray
    downdraft_flux_signed: np.ndarray
    level_value: int | None


def variable_path(alaro_root: Path, experiment: str, variable: str, day: str, filename: str) -> Path:
    return alaro_root / experiment / "masked-netcdf" / variable / day / filename


def first_data_var(ds: xr.Dataset) -> str:
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables")
    return next(iter(ds.data_vars))


def read_var(path: Path, variable: str, *, level_index: int | None = None) -> tuple[np.ndarray, xr.Dataset]:
    with xr.open_dataset(path, decode_times=False) as ds:
        var_name = variable if variable in ds.data_vars else first_data_var(ds)
        da = ds[var_name]
        if level_index is None:
            arr = np.asarray(da.values, dtype=np.float64)
        else:
            arr = np.asarray(da.isel(time=0, level=level_index).values, dtype=np.float64)
        return arr, ds.load()


def read_2d(path: Path, variable: str, level_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    arr, ds = read_var(path, variable, level_index=level_index)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    da = ds[variable] if variable in ds.data_vars else ds[first_data_var(ds)]
    level_values = da.attrs.get("level_values")
    level_value = None
    if level_values is not None and len(level_values) > level_index:
        level_value = int(level_values[level_index])
    return arr, lon, lat, level_value


def signed_updraft_flux(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    return np.where(mesh > 0, omega * mesh / G, 0.0)


def signed_downdraft_flux(omega: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    return np.where(mesh > 0, omega * mesh / G, 0.0)


def cloud_condensate_transport(
    liquid: np.ndarray,
    solid: np.ndarray,
    updraft_flux_signed_value: np.ndarray,
    downdraft_flux_signed_value: np.ndarray,
) -> np.ndarray:
    condensate = np.maximum(liquid, 0.0) + np.maximum(solid, 0.0)
    # Mass flux (kg m-2 s-1) times mixing ratio (kg kg-1) -> kg m-2 s-1.
    # Convert kg to mg to match the scale/style of the reference figure.
    return condensate * (updraft_flux_signed_value + downdraft_flux_signed_value) * 1.0e6


def available_step_files(alaro_root: Path, experiment: str, max_days: int) -> list[Path]:
    root = alaro_root / experiment / "masked-netcdf" / "UD_OMEGA"
    days = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("pf"))
    if max_days > 0:
        days = days[:max_days]
    files: list[Path] = []
    for day in days:
        files.extend(sorted(p for p in day.iterdir() if p.suffix == ".nc"))
    return files


def draft_activity_score(updraft_level_max: np.ndarray, downdraft_level_max: np.ndarray) -> np.ndarray:
    """Score levels for a Figure-6.12-like snapshot where both drafts are visible."""
    updraft_visible = np.minimum(updraft_level_max / 0.5, 1.0)
    downdraft_visible = np.minimum(downdraft_level_max / 0.05, 1.0)
    return updraft_visible + downdraft_visible + 0.05 * updraft_level_max + 2.0 * downdraft_level_max


def select_active_snapshot(
    alaro_root: Path,
    experiment: str,
    *,
    max_days: int,
) -> SnapshotSelection:
    best: SnapshotSelection | None = None
    for ud_path in available_step_files(alaro_root, experiment, max_days=max_days):
        day = ud_path.parent.name
        filename = ud_path.name
        paths = {
            "UD_MESH_FRAC": variable_path(alaro_root, experiment, "UD_MESH_FRAC", day, filename),
            "DD_OMEGA": variable_path(alaro_root, experiment, "DD_OMEGA", day, filename),
            "DD_MESH_FRAC": variable_path(alaro_root, experiment, "DD_MESH_FRAC", day, filename),
        }
        if not all(path.exists() for path in paths.values()):
            continue
        try:
            omega, _ = read_var(ud_path, "UD_OMEGA")
            mesh, ds = read_var(paths["UD_MESH_FRAC"], "UD_MESH_FRAC")
            dd_omega, _ = read_var(paths["DD_OMEGA"], "DD_OMEGA")
            dd_mesh, _ = read_var(paths["DD_MESH_FRAC"], "DD_MESH_FRAC")
        except Exception:
            continue
        omega = np.asarray(omega[0], dtype=np.float64)
        mesh = np.asarray(mesh[0], dtype=np.float64)
        dd_omega = np.asarray(dd_omega[0], dtype=np.float64)
        dd_mesh = np.asarray(dd_mesh[0], dtype=np.float64)
        updraft_flux = np.abs(signed_updraft_flux(omega, mesh))
        downdraft_flux = np.abs(signed_downdraft_flux(dd_omega, dd_mesh))
        if not np.isfinite(updraft_flux).any() or not np.isfinite(downdraft_flux).any():
            continue
        updraft_level_max = np.nanmax(updraft_flux, axis=(1, 2))
        downdraft_level_max = np.nanmax(downdraft_flux, axis=(1, 2))
        level_scores = draft_activity_score(updraft_level_max, downdraft_level_max)
        level_index = int(np.nanargmax(level_scores))
        score = float(level_scores[level_index])
        da = ds["UD_MESH_FRAC"] if "UD_MESH_FRAC" in ds.data_vars else ds[first_data_var(ds)]
        level_values = da.attrs.get("level_values")
        level_value = None
        if level_values is not None and len(level_values) > level_index:
            level_value = int(level_values[level_index])
        candidate = SnapshotSelection(
            day=day,
            filename=filename,
            level_index=level_index,
            level_value=level_value,
            score=score,
            updraft_max=float(updraft_level_max[level_index]),
            downdraft_max=float(downdraft_level_max[level_index]),
        )
        if best is None or candidate.score > best.score:
            best = candidate
    if best is None:
        raise RuntimeError(f"No active updraft snapshot found for {experiment}")
    return best


def load_snapshot_fields(
    alaro_root: Path,
    experiment: str,
    selection: SnapshotSelection,
) -> SnapshotFields:
    day = selection.day
    filename = selection.filename
    level_index = selection.level_index
    values: dict[str, np.ndarray] = {}
    lon = lat = None
    level_value = selection.level_value
    for variable in REQUIRED_VARIABLES:
        path = variable_path(alaro_root, experiment, variable, day, filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing {variable}: {path}")
        arr, lon_candidate, lat_candidate, level_candidate = read_2d(path, variable, level_index)
        values[variable] = arr
        if lon is None:
            lon = lon_candidate
            lat = lat_candidate
        if variable == "UD_OMEGA" and level_candidate is not None:
            level_value = level_candidate

    updraft = signed_updraft_flux(values["UD_OMEGA"], values["UD_MESH_FRAC"])
    downdraft = signed_downdraft_flux(values["DD_OMEGA"], values["DD_MESH_FRAC"])
    transport = cloud_condensate_transport(
        values["LIQUID_WATER"],
        values["SOLID_WATER"],
        updraft,
        downdraft,
    )
    return SnapshotFields(
        lon=np.asarray(lon),
        lat=np.asarray(lat),
        height_m=values["GEOPOTENTIEL"],
        cloud_transport_mg_m2_s=transport,
        updraft_flux_signed=updraft,
        downdraft_flux_signed=downdraft,
        level_value=level_value,
    )


def transport_difference(
    alaro_root: Path,
    experiment: str,
    reference_experiment: str | None,
    selection: SnapshotSelection,
) -> tuple[SnapshotFields, np.ndarray, str]:
    candidate = load_snapshot_fields(alaro_root, experiment, selection)
    if reference_experiment is None:
        return candidate, candidate.cloud_transport_mg_m2_s, "Cloud condensate transport proxy"

    reference = load_snapshot_fields(alaro_root, reference_experiment, selection)
    diff = candidate.cloud_transport_mg_m2_s - reference.cloud_transport_mg_m2_s
    label = (
        "Cloud condensate transport flux difference "
        f"({EXPERIMENT_LABELS.get(experiment, experiment)} - "
        f"{EXPERIMENT_LABELS.get(reference_experiment, reference_experiment)})"
    )
    return candidate, diff, label


def activity_crop(
    panels: Sequence[np.ndarray],
    *,
    pad: int,
    transport_threshold: float,
    updraft_threshold: float,
    downdraft_threshold: float,
) -> tuple[slice, slice]:
    transport, updraft, downdraft = panels
    active = (
        (np.isfinite(transport) & (np.abs(transport) >= transport_threshold))
        | (np.isfinite(updraft) & (np.abs(updraft) >= updraft_threshold))
        | (np.isfinite(downdraft) & (np.abs(downdraft) >= downdraft_threshold))
    )
    if not active.any():
        return slice(None), slice(None)
    ys, xs = np.where(active)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, active.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, active.shape[1])
    return slice(y0, y1), slice(x0, x1)


def crop2d(arr: np.ndarray, y_slice: slice, x_slice: slice) -> np.ndarray:
    return np.asarray(arr[y_slice, x_slice])


def make_transport_norm() -> tuple[mcolors.BoundaryNorm, mcolors.Colormap]:
    bounds = np.asarray([-550, -250, -100, -10, -5, -1, 1, 5, 10, 100, 250, 550], dtype=float)
    cmap = plt.get_cmap("RdBu_r", len(bounds) - 1)
    return mcolors.BoundaryNorm(bounds, cmap.N, clip=True), cmap


def make_updraft_norm() -> tuple[mcolors.BoundaryNorm, mcolors.Colormap]:
    bounds = np.asarray([-11, -5, -2, -0.5, -0.05, 0], dtype=float)
    cmap = mcolors.ListedColormap(["#5c0a0a", "#b2182b", "#d6604d", "#e6a385", "#fbf7f2"])
    return mcolors.BoundaryNorm(bounds, cmap.N, clip=True), cmap


def make_downdraft_norm() -> tuple[mcolors.BoundaryNorm, mcolors.Colormap]:
    bounds = np.asarray([0, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0], dtype=float)
    cmap = mcolors.ListedColormap(["#e8ffff", "#bdeef0", "#77bfd1", "#3f8fc5", "#2859a5", "#07133d"])
    return mcolors.BoundaryNorm(bounds, cmap.N, clip=True), cmap


def _panel_label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 2.0},
    )


def plot_snapshot(
    *,
    output_path: Path,
    fields: SnapshotFields,
    transport_field: np.ndarray,
    transport_title: str,
    selection: SnapshotSelection,
    experiment_label: str,
    crop_pad: int,
    dpi: int,
) -> tuple[slice, slice]:
    y_slice, x_slice = activity_crop(
        (transport_field, fields.updraft_flux_signed, fields.downdraft_flux_signed),
        pad=crop_pad,
        transport_threshold=1.0,
        updraft_threshold=0.05,
        downdraft_threshold=0.01,
    )
    lon = crop2d(fields.lon, y_slice, x_slice)
    lat = crop2d(fields.lat, y_slice, x_slice)
    panels = [
        crop2d(transport_field, y_slice, x_slice),
        crop2d(fields.updraft_flux_signed, y_slice, x_slice),
        crop2d(fields.downdraft_flux_signed, y_slice, x_slice),
    ]
    titles = ["cloud condensates transport error", "updraft", "downdraft"]
    units = [
        r"mg m$^{-2}$ s$^{-1}$",
        r"kg m$^{-2}$ s$^{-1}$",
        r"kg m$^{-2}$ s$^{-1}$",
    ]
    norms_cmaps = [make_transport_norm(), make_updraft_norm(), make_downdraft_norm()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    labels = ("(a)", "(b)", "(c)")
    for ax, panel, title, unit, (norm, cmap), label in zip(
        axes, panels, titles, units, norms_cmaps, labels, strict=True
    ):
        mesh = ax.pcolormesh(lon, lat, panel, cmap=cmap, norm=norm, shading="auto")
        ax.set_title(title, fontsize=12, fontweight="bold", color="black", pad=10)
        _panel_label(ax, label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        strip_cbar_zeros(cbar)
        cbar.ax.set_title(unit, fontsize=9, pad=6)
        cbar.ax.tick_params(labelsize=9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return y_slice, x_slice


def write_txt(
    txt_path: Path,
    *,
    figure_path: Path,
    alaro_root: Path,
    experiment: str,
    reference_experiment: str | None,
    selection: SnapshotSelection,
    fields: SnapshotFields,
    transport_field: np.ndarray,
    transport_title: str,
    y_slice: slice,
    x_slice: slice,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lon = crop2d(fields.lon, y_slice, x_slice)
    lat = crop2d(fields.lat, y_slice, x_slice)
    height = crop2d(fields.height_m, y_slice, x_slice)
    transport = crop2d(transport_field, y_slice, x_slice)
    updraft = crop2d(fields.updraft_flux_signed, y_slice, x_slice)
    downdraft = crop2d(fields.downdraft_flux_signed, y_slice, x_slice)
    title = f"Figure 6.12-style updraft spatial snapshot data: {figure_path.stem}"

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"ALARO root: {alaro_root}\n")
        fh.write(f"Experiment: {experiment} ({EXPERIMENT_LABELS.get(experiment, experiment)})\n")
        fh.write(f"Reference experiment: {reference_experiment or 'none'}\n")
        fh.write(f"Day: {selection.day}\n")
        fh.write(f"File: {selection.filename}\n")
        fh.write(f"Level index: {selection.level_index}\n")
        fh.write(f"Model level value: {selection.level_value}\n")
        fh.write(f"Mean selected height: {float(np.nanmean(height)):.10g} m\n")
        fh.write(f"Auto-selection score: {selection.score:.10g} dimensionless combined draft-activity score\n")
        fh.write(f"Selected max updraft flux magnitude: {selection.updraft_max:.10g} kg m^-2 s^-1\n")
        fh.write(f"Selected max downdraft flux magnitude: {selection.downdraft_max:.10g} kg m^-2 s^-1\n")
        fh.write(f"Panel a: {transport_title}\n")
        fh.write("Panel a method: (liquid_water + solid_water) * (updraft_flux + downdraft_flux) * 1e6.\n")
        fh.write("If a reference experiment is supplied, panel a is candidate minus reference.\n")
        fh.write("Updraft/downdraft flux convention here follows the reference-style signed plot: negative is upward, positive is downward.\n")
        fh.write("Updraft flux: UD_OMEGA * UD_MESH_FRAC / g.\n")
        fh.write("Downdraft flux: DD_OMEGA * DD_MESH_FRAC / g.\n")
        fh.write(f"Crop y slice: {y_slice.start}:{y_slice.stop}; x slice: {x_slice.start}:{x_slice.stop}\n\n")

        fh.write("Summary\n")
        fh.write("-------\n")
        fh.write("field,min,max,mean,finite_count\n")
        for name, arr in (
            ("panel_a_mg_m2_s", transport),
            ("updraft_signed_kg_m2_s", updraft),
            ("downdraft_signed_kg_m2_s", downdraft),
            ("height_m", height),
        ):
            finite = arr[np.isfinite(arr)]
            fh.write(
                f"{name},{float(np.nanmin(arr)):.12g},{float(np.nanmax(arr)):.12g},"
                f"{float(np.nanmean(arr)):.12g},{int(finite.size)}\n"
            )

        fh.write("\nGridpoint data\n")
        fh.write("--------------\n")
        fh.write("y_index,x_index,lat,lon,height_m,panel_a_mg_m2_s,updraft_signed_kg_m2_s,downdraft_signed_kg_m2_s\n")
        y0 = 0 if y_slice.start is None else int(y_slice.start)
        x0 = 0 if x_slice.start is None else int(x_slice.start)
        for yi in range(transport.shape[0]):
            for xi in range(transport.shape[1]):
                fh.write(
                    f"{y0 + yi},{x0 + xi},"
                    f"{float(lat[yi, xi]):.12g},{float(lon[yi, xi]):.12g},"
                    f"{float(height[yi, xi]):.12g},{float(transport[yi, xi]):.12g},"
                    f"{float(updraft[yi, xi]):.12g},{float(downdraft[yi, xi]):.12g}\n"
                )


def make_snapshot_plot(
    *,
    alaro_root: Path = DEFAULT_ALARO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiment: str = "2mom",
    reference_experiment: str | None = "control",
    day: str | None = None,
    filename: str | None = None,
    level_index: int | None = None,
    auto_max_days: int = 31,
    crop_pad: int = 8,
    dpi: int = 400,
) -> dict[str, Path]:
    if day is None or filename is None or level_index is None:
        selection = select_active_snapshot(alaro_root, experiment, max_days=auto_max_days)
        if day is not None:
            selection = SnapshotSelection(
                day,
                selection.filename,
                selection.level_index,
                selection.level_value,
                selection.score,
                selection.updraft_max,
                selection.downdraft_max,
            )
        if filename is not None:
            selection = SnapshotSelection(
                selection.day,
                filename,
                selection.level_index,
                selection.level_value,
                selection.score,
                selection.updraft_max,
                selection.downdraft_max,
            )
        if level_index is not None:
            selection = SnapshotSelection(
                selection.day,
                selection.filename,
                level_index,
                None,
                selection.score,
                selection.updraft_max,
                selection.downdraft_max,
            )
    else:
        selection = SnapshotSelection(day, filename, level_index, None, np.nan)

    fields, transport_field, transport_title = transport_difference(
        alaro_root,
        experiment,
        reference_experiment,
        selection,
    )
    exp_label = EXPERIMENT_LABELS.get(experiment, experiment)
    ref_label = EXPERIMENT_LABELS.get(reference_experiment, reference_experiment) if reference_experiment else None
    stem = f"fig612_like_{exp_label}"
    if ref_label:
        stem += f"_minus_{ref_label}"
    output_path = output_dir / f"{stem}.png"
    y_slice, x_slice = plot_snapshot(
        output_path=output_path,
        fields=fields,
        transport_field=transport_field,
        transport_title=transport_title,
        selection=selection,
        experiment_label=exp_label,
        crop_pad=crop_pad,
        dpi=dpi,
    )
    txt_path = output_dir / "data_txt" / f"{stem}.txt"
    write_txt(
        txt_path,
        figure_path=output_path,
        alaro_root=alaro_root,
        experiment=experiment,
        reference_experiment=reference_experiment,
        selection=selection,
        fields=fields,
        transport_field=transport_field,
        transport_title=transport_title,
        y_slice=y_slice,
        x_slice=x_slice,
    )
    return {"plot": output_path, "txt": txt_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a Figure-6.12-style spatial snapshot of condensate transport, updraft and downdraft."
    )
    parser.add_argument("--alaro-root", type=Path, default=DEFAULT_ALARO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_LABELS), default="2mom")
    parser.add_argument("--reference-experiment", choices=tuple(EXPERIMENT_LABELS), default="control")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--day", default=None, help="Day folder, e.g. pf20140101. Default: auto-select.")
    parser.add_argument("--filename", default=None, help="Step file, e.g. pfABOFABOF+0012.nc. Default: auto-select.")
    parser.add_argument("--level-index", type=int, default=None, help="0-based model level index. Default: auto-select.")
    parser.add_argument("--auto-max-days", type=int, default=31)
    parser.add_argument("--crop-pad", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=450)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = None if args.no_reference else args.reference_experiment
    outputs = make_snapshot_plot(
        alaro_root=args.alaro_root,
        output_dir=args.output_dir,
        experiment=args.experiment,
        reference_experiment=reference,
        day=args.day,
        filename=args.filename,
        level_index=args.level_index,
        auto_max_days=args.auto_max_days,
        crop_pad=args.crop_pad,
        dpi=args.dpi,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
