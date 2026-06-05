"""CFAD-style distributions of DSD diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alaro_analysis.common.figio import strip_cbar_zeros
import numpy as np

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS, RD
from alaro_analysis.common.dsd import (
    MP_FIXED_N0_PER_M3_MM,
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.common.vertical import centers_to_edges
from alaro_analysis.data.cache import signature as cache_signature
from alaro_analysis.workflows.disdrometer_comparison import (
    RUNS_ROOT,
    lead_label,
    parse_lead_selection,
)
from alaro_analysis.workflows.disdrometer_dsd import (
    MASK_FILE,
    NETCDF_ROOT,
    build_domain_mask_from_netcdf,
)
from alaro_analysis.workflows.dsd_vertical_profiles import (
    discover_vertical_records,
    freezing_level_km,
    read_masked_level_mean,
)


OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_profiles"
DATA_TXT_DIR = OUTPUT_DIR / "data_txt"
CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_profiles"
FIGURE_NAME = "dsd_cfad_D0_logNw.png"
TEXT_NAME = "dsd_cfad_D0_logNw.txt"

DSD_FIELDS = ("d0_mm", "log_nw")
PANEL_FIELDS = DSD_FIELDS
FIELD_LABELS = {
    "d0_mm": "D$_0$ (mm)",
    "log_nw": "log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)",
}
ROW_LABELS = {
    "d0_mm": "D$_0$",
    "log_nw": "log$_{10}$ N$_w$",
}
PANEL_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
PANEL_LABEL_FONTSIZE = 16
COLORBAR_LABEL_FONTSIZE = 16
COLORBAR_TICK_FONTSIZE = 14


@dataclass(frozen=True)
class CfadGrid:
    height_km: np.ndarray
    x_edges: np.ndarray
    hist: np.ndarray
    counts: np.ndarray
    n_profiles: int
    source: str


@dataclass(frozen=True)
class ExperimentCfad:
    grids: dict[str, CfadGrid]
    temperature_k: np.ndarray
    freezing_level_km: float


_WORKER_MASK: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build CFAD-style height-value distributions for D0/log10(Nw) "
            "from all model timesteps."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-txt-dir", type=Path, default=DATA_TXT_DIR)
    parser.add_argument(
        "--lead",
        default="all",
        help="Forecast leads used for D0/log10(Nw). Use 'all' for all model timesteps.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tasks-per-child", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--min-qr", type=float, default=1.0e-7)
    parser.add_argument(
        "--onemom-closure",
        choices=("abel_boutle", "fixed_n0"),
        default="abel_boutle",
    )
    parser.add_argument(
        "--twomom-closure",
        choices=("gamma_mu1", "marshall_palmer"),
        default="gamma_mu1",
    )
    parser.add_argument("--n0-fixed", type=float, default=MP_FIXED_N0_PER_M3_MM)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--d0-range", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--lognw-range", type=float, nargs=2, default=(0.0, 9.0))
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_profiles", argv=argv)


def x_edges_for_args(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {
        "d0_mm": np.linspace(args.d0_range[0], args.d0_range[1], args.bins + 1),
        "log_nw": np.linspace(args.lognw_range[0], args.lognw_range[1], args.bins + 1),
    }


def add_values_to_hist(hist: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    levels = np.arange(vals.size)
    idx = np.searchsorted(edges, vals, side="right") - 1
    valid = np.isfinite(vals) & (idx >= 0) & (idx < edges.size - 1)
    if np.any(valid):
        np.add.at(hist, (levels[valid], idx[valid]), 1)
    return hist


def frequency_percent(hist: np.ndarray) -> np.ndarray:
    row_total = np.sum(hist, axis=1, keepdims=True)
    out = np.full(hist.shape, np.nan, dtype=np.float64)
    np.divide(100.0 * hist, row_total, out=out, where=row_total > 0)
    return out


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _process_dsd_task(
    task: tuple[str, dict[str, str], float, str, str, float],
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    experiment, paths, min_qr, onemom_closure, twomom_closure, n0_fixed = task
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    mask = _WORKER_MASK

    try:
        qr, _ = read_masked_level_mean(Path(paths["RAIN"]), "RAIN", mask)
        temp, _ = read_masked_level_mean(Path(paths["TEMPERATURE"]), "TEMPERATURE", mask)
        pres, _ = read_masked_level_mean(Path(paths["PRESSURE"]), "PRESSURE", mask)
        height_m, _ = read_masked_level_mean(Path(paths["GEOPOTENTIEL"]), "GEOPOTENTIEL", mask)
        if experiment == "2mom":
            pnr, _ = read_masked_level_mean(Path(paths["PNR"]), "PNR", mask)
        else:
            pnr = np.full(qr.shape, np.nan, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive worker logging
        return None, [f"WARNING {experiment} {paths.get('RAIN', '<missing>')}: {exc}"]

    rho = np.full(qr.shape, np.nan, dtype=np.float64)
    ok_rho = np.isfinite(pres) & np.isfinite(temp) & (temp > 0.0)
    rho[ok_rho] = pres[ok_rho] / (RD * temp[ok_rho])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if experiment == "2mom":
            if twomom_closure == "gamma_mu1":
                diag = gamma_dsd_from_q_n_per_kg(qr, pnr, rho, mu=1.0)
            else:
                diag = mp_from_q_n_per_kg(qr, pnr, rho)
        elif onemom_closure == "abel_boutle":
            diag = mp_from_q_abel_boutle(qr, rho)
        else:
            diag = mp_from_q_fixed_n0(qr, rho, n0_per_m3_mm=n0_fixed)

    rainy = np.isfinite(qr) & (qr >= min_qr)
    out = {
        "height_km": height_m / 1000.0,
        "temperature_k": temp,
    }
    for field in DSD_FIELDS:
        values = np.asarray(diag[field], dtype=np.float64)
        out[field] = np.where(rainy & np.isfinite(values), values, np.nan)
    return out, []


def _empty_dsd_accumulator(n_levels: int, x_edges: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "height_sum": np.zeros(n_levels, dtype=np.float64),
        "height_count": np.zeros(n_levels, dtype=np.int64),
        "temperature_sum": np.zeros(n_levels, dtype=np.float64),
        "temperature_count": np.zeros(n_levels, dtype=np.int64),
        "hist": {
            field: np.zeros((n_levels, x_edges[field].size - 1), dtype=np.int64)
            for field in DSD_FIELDS
        },
        "n_profiles": 0,
    }


def add_dsd_profile_to_accumulator(
    acc: dict[str, Any],
    profile: dict[str, np.ndarray],
    x_edges: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not acc:
        acc.update(_empty_dsd_accumulator(profile["height_km"].size, x_edges))

    height = np.asarray(profile["height_km"], dtype=np.float64)
    ok_height = np.isfinite(height)
    acc["height_sum"][ok_height] += height[ok_height]
    acc["height_count"][ok_height] += 1

    temp = np.asarray(profile["temperature_k"], dtype=np.float64)
    ok_temp = np.isfinite(temp)
    acc["temperature_sum"][ok_temp] += temp[ok_temp]
    acc["temperature_count"][ok_temp] += 1

    for field in DSD_FIELDS:
        add_values_to_hist(acc["hist"][field], profile[field], x_edges[field])
    acc["n_profiles"] += 1
    return acc


def grids_from_dsd_accumulator(
    acc: dict[str, Any],
    x_edges: dict[str, np.ndarray],
    *,
    source: str,
) -> tuple[dict[str, CfadGrid], np.ndarray, float]:
    height = np.full(acc["height_sum"].shape, np.nan, dtype=np.float64)
    ok_height = acc["height_count"] > 0
    height[ok_height] = acc["height_sum"][ok_height] / acc["height_count"][ok_height]

    temp = np.full(acc["temperature_sum"].shape, np.nan, dtype=np.float64)
    ok_temp = acc["temperature_count"] > 0
    temp[ok_temp] = acc["temperature_sum"][ok_temp] / acc["temperature_count"][ok_temp]
    freeze = freezing_level_km(height, temp)

    grids = {
        field: CfadGrid(
            height_km=height,
            x_edges=x_edges[field],
            hist=acc["hist"][field],
            counts=np.sum(acc["hist"][field], axis=1),
            n_profiles=int(acc["n_profiles"]),
            source=source,
        )
        for field in DSD_FIELDS
    }
    return grids, temp, freeze


def compute_dsd_cfads(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    mask: np.ndarray,
    x_edges: dict[str, np.ndarray],
    *,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> tuple[dict[str, CfadGrid], np.ndarray, float]:
    if not records:
        raise RuntimeError(f"No masked-NetCDF records found for {experiment}")
    tasks = [
        (
            experiment,
            {name: str(path) for name, path in rec[3].items()},
            min_qr,
            onemom_closure,
            twomom_closure,
            n0_fixed,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} DSD profiles", flush=True)
    acc: dict[str, Any] = {}
    if workers <= 1:
        _init_worker(mask)
        iterator = (_process_dsd_task(task) for task in tasks)
        for idx, (profile, warnings) in enumerate(iterator, start=1):
            if profile is not None:
                add_dsd_profile_to_accumulator(acc, profile, x_edges)
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(mask,),
            maxtasksperchild=tasks_per_child,
        ) as pool:
            for idx, (profile, warnings) in enumerate(
                pool.imap_unordered(_process_dsd_task, tasks),
                start=1,
            ):
                if profile is not None:
                    add_dsd_profile_to_accumulator(acc, profile, x_edges)
                for warning in warnings:
                    print(warning, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)

    if not acc:
        raise RuntimeError(f"No readable DSD profiles for {experiment}")
    return grids_from_dsd_accumulator(
        acc,
        x_edges,
        source=str(records[0][3]["RAIN"].parent.parent),
    )


def cache_path(cache_dir: Path, experiment: str, tag: str) -> Path:
    return cache_dir / f"{experiment}_{tag}.npz"


def legacy_cache_path(cache_dir: Path, experiment: str, tag: str) -> Path:
    return cache_dir / f"{experiment}_{tag}_ddh0024.npz"


def save_experiment_cfad(path: Path, cfad: ExperimentCfad, sig: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "temperature_k": cfad.temperature_k,
        "freezing_level_km": np.asarray([cfad.freezing_level_km], dtype=np.float64),
        "__signature__": np.asarray([sig]),
    }
    for field, grid in cfad.grids.items():
        payload[f"height__{field}"] = grid.height_km
        payload[f"x_edges__{field}"] = grid.x_edges
        payload[f"hist__{field}"] = grid.hist
        payload[f"counts__{field}"] = grid.counts
        payload[f"n_profiles__{field}"] = np.asarray([grid.n_profiles], dtype=np.int64)
        payload[f"source__{field}"] = np.asarray([grid.source])
    np.savez_compressed(path, **payload)


def load_experiment_cfad(path: Path) -> ExperimentCfad:
    with np.load(path, allow_pickle=False) as data:
        grids: dict[str, CfadGrid] = {}
        for field in PANEL_FIELDS:
            grids[field] = CfadGrid(
                height_km=np.asarray(data[f"height__{field}"], dtype=np.float64),
                x_edges=np.asarray(data[f"x_edges__{field}"], dtype=np.float64),
                hist=np.asarray(data[f"hist__{field}"], dtype=np.int64),
                counts=np.asarray(data[f"counts__{field}"], dtype=np.int64),
                n_profiles=int(np.ravel(data[f"n_profiles__{field}"])[0]),
                source=str(data[f"source__{field}"][0]),
            )
        return ExperimentCfad(
            grids=grids,
            temperature_k=np.asarray(data["temperature_k"], dtype=np.float64),
            freezing_level_km=float(np.ravel(data["freezing_level_km"])[0]),
        )


def experiment_cache_signature(
    args: argparse.Namespace,
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    mask: np.ndarray,
    x_edges: dict[str, np.ndarray],
) -> str:
    """Signature of every parameter that affects an experiment's CFAD grids."""
    edges = np.concatenate([np.asarray(x_edges[f], dtype=np.float64) for f in PANEL_FIELDS])
    return cache_signature(
        {
            "cache_version": 1,
            "experiment": experiment,
            "min_qr": args.min_qr,
            "onemom_closure": args.onemom_closure,
            "twomom_closure": args.twomom_closure,
            "n0_fixed": args.n0_fixed,
            "x_edges": edges,
            "mask": np.asarray(mask),
            "n_records": len(records),
            "source": str(records[0][3]["RAIN"].parent.parent) if records else "",
        }
    )


def read_cache_signature(path: Path) -> str | None:
    """Return the stored ``__signature__`` of a cache file, or None if absent."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if "__signature__" in data:
                return str(np.ravel(data["__signature__"])[0])
    except Exception:
        return None
    return None


def get_experiment_cfad(
    args: argparse.Namespace,
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    mask: np.ndarray,
    x_edges: dict[str, np.ndarray],
    tag: str,
) -> ExperimentCfad:
    sig = experiment_cache_signature(args, experiment, records, mask, x_edges)
    path = cache_path(args.cache_dir, experiment, tag)
    legacy_path = legacy_cache_path(args.cache_dir, experiment, tag)
    if not args.recompute:
        for candidate in (path, legacy_path):
            if not candidate.exists():
                continue
            if read_cache_signature(candidate) == sig:
                cfad = load_experiment_cfad(candidate)
                if candidate != path:
                    save_experiment_cfad(path, cfad, sig)
                return cfad
            print(
                f"  [{experiment}] cache {candidate.name} has changed/absent "
                "parameters; recomputing.",
                flush=True,
            )

    dsd_grids, temperature, freeze = compute_dsd_cfads(
        experiment,
        records,
        mask,
        x_edges,
        min_qr=args.min_qr,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        n0_fixed=args.n0_fixed,
        workers=max(1, int(args.workers)),
        progress_every=max(1, int(args.progress_every)),
        tasks_per_child=max(1, int(args.tasks_per_child)),
    )
    cfad = ExperimentCfad(
        grids=dsd_grids,
        temperature_k=temperature,
        freezing_level_km=freeze,
    )
    save_experiment_cfad(path, cfad, sig)
    return cfad


def row_vmax(cfads: dict[str, ExperimentCfad], experiments: list[str], field: str) -> float:
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    return max(1.0, float(np.nanpercentile(merged, 99.0)))


def cfad_colormap():
    try:
        import cmaps
    except ImportError as exc:  # pragma: no cover - depends on optional runtime env
        raise RuntimeError(
            "The cmaps package is required for this plot; install the full extra "
            "or run in an environment that provides cmaps."
        ) from exc
    return cmaps.WhViBlGrYeOrRe


def plot_cfads(
    output_path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: list[str],
    *,
    max_height_km: float,
    dpi: int,
    row_vmax_values: Mapping[str, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = cfad_colormap()
    fig, axes = plt.subplots(
        len(PANEL_FIELDS),
        len(experiments),
        figsize=(5.4 * len(experiments), 4.0 * len(PANEL_FIELDS)),
        sharey=True,
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for row, field in enumerate(PANEL_FIELDS):
        vmax = (
            float(row_vmax_values[field])
            if row_vmax_values is not None and field in row_vmax_values
            else row_vmax(cfads, experiments, field)
        )
        last_im = None
        for col, experiment in enumerate(experiments):
            ax = axes[row, col]
            panel_idx = row * len(experiments) + col
            grid = cfads[experiment].grids[field]
            freq = frequency_percent(grid.hist)
            y_edges = centers_to_edges(grid.height_km)
            last_im = ax.pcolormesh(
                grid.x_edges,
                y_edges,
                freq,
                cmap=cmap,
                shading="auto",
                vmin=0.0,
                vmax=vmax,
            )
            freeze = cfads[experiment].freezing_level_km
            if np.isfinite(freeze):
                ax.axhline(freeze, color="white", linestyle="--", linewidth=1.4, alpha=0.95)
                ax.axhline(freeze, color="black", linestyle="--", linewidth=0.7, alpha=0.75)

            if row == 0:
                ax.set_title(
                    EXPERIMENT_LABELS.get(experiment, experiment),
                    fontsize=TITLE_FONTSIZE,
                    pad=10,
                )
            ax.text(
                0.03,
                0.96,
                f"({PANEL_LABELS[panel_idx]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
            if col == 0:
                ax.set_ylabel(
                    f"{ROW_LABELS[field]}\nHeight (km)",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            ax.set_xlabel(FIELD_LABELS[field], fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylim(0.0, max_height_km)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(False)

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes[row, :], shrink=0.84, pad=0.012)
            strip_cbar_zeros(cbar)
            cbar.set_label("Frequency per level (%)", fontsize=COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_data_txt(
    path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: list[str],
    *,
    lead_tag: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("CFAD-style DSD diagnostic distributions\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"DSD lead selection: {lead_tag}\n")
        fh.write("Colormap: cmaps.WhViBlGrYeOrRe\n")
        fh.write(f"bins: {args.bins}\n")
        fh.write(f"d0_range: {tuple(args.d0_range)}\n")
        fh.write(f"lognw_range: {tuple(args.lognw_range)}\n")
        fh.write("\n")
        fh.write("experiment,field,n_profiles,total_samples,freezing_level_km,source\n")
        for experiment in experiments:
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            freeze = cfads[experiment].freezing_level_km
            for field in PANEL_FIELDS:
                grid = cfads[experiment].grids[field]
                fh.write(
                    f"{experiment},{label},{field},{grid.n_profiles},"
                    f"{int(np.sum(grid.hist))},{freeze:.10e},{grid.source}\n"
                )


def build_cfads(args: argparse.Namespace) -> tuple[dict[str, ExperimentCfad], str]:
    leads = parse_lead_selection(args.lead)
    lead_tag = lead_label(leads)
    tag = f"{lead_tag}_{args.onemom_closure}_{args.twomom_closure}_bins{args.bins}"
    if args.max_days is not None:
        tag += f"_first{args.max_days}days"
    x_edges = x_edges_for_args(args)

    records_by_experiment = {
        experiment: discover_vertical_records(
            experiment,
            leads,
            args.netcdf_root,
            args.max_days,
        )
        for experiment in args.experiments
    }
    sample_records = next((records for records in records_by_experiment.values() if records), None)
    if not sample_records:
        raise RuntimeError("No masked-NetCDF records found for any requested experiment")
    domain_mask = build_domain_mask_from_netcdf(
        sample_records[0][3]["RAIN"],
        args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
    )
    print(
        f"Radar mask: {domain_mask.n_cells} / {domain_mask.mask.size} cells "
        f"({domain_mask.selected_var})",
        flush=True,
    )

    cfads: dict[str, ExperimentCfad] = {}
    for experiment in args.experiments:
        print(f"Processing {EXPERIMENT_LABELS.get(experiment, experiment)}", flush=True)
        cfads[experiment] = get_experiment_cfad(
            args,
            experiment,
            records_by_experiment[experiment],
            domain_mask.mask,
            x_edges,
            tag,
        )
    return cfads, lead_tag


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfads, lead_tag = build_cfads(args)
    output_path = args.output_dir / FIGURE_NAME
    text_path = args.data_txt_dir / TEXT_NAME
    plot_cfads(
        output_path,
        cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
    )
    write_data_txt(
        text_path,
        cfads,
        args.experiments,
        lead_tag=lead_tag,
        args=args,
    )
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


if __name__ == "__main__":
    main()
