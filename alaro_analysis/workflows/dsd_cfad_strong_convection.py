"""CFAD-style DSD distributions sampled only in strong updraft regions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS, G, RD
from alaro_analysis.common.dsd import (
    MP_FIXED_N0_PER_M3_MM,
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.workflows.disdrometer_comparison import (
    RUNS_ROOT,
    lead_label,
    parse_lead_selection,
)
from alaro_analysis.workflows.disdrometer_dsd import (
    MASK_FILE,
    NETCDF_ROOT,
    DomainMask,
    build_domain_mask_from_netcdf,
)
from alaro_analysis.workflows.dsd_cfad_profiles import (
    DSD_FIELDS,
    FIGURE_NAME as BASE_FIGURE_NAME,
    TEXT_NAME as BASE_TEXT_NAME,
    CfadGrid,
    ExperimentCfad,
    add_values_to_hist,
    plot_cfads,
    x_edges_for_args,
)
from alaro_analysis.workflows.dsd_vertical_profiles import (
    _as_float_array,
    _to_level_yx,
    discover_vertical_records,
    freezing_level_km,
)


OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_strong_convection"
DATA_TXT_DIR = OUTPUT_DIR / "data_txt"
CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_strong_convection"
FIGURE_NAME = BASE_FIGURE_NAME.replace("dsd_cfad", "dsd_cfad_strong_convection")
TEXT_NAME = BASE_TEXT_NAME.replace("dsd_cfad", "dsd_cfad_strong_convection")
UPDRAFT_VARIABLE = "UD_OMEGA"
UPDRAFT_MESH_VARIABLE = "UD_MESH_FRAC"


@dataclass(frozen=True)
class StrongConvectionSettings:
    metric: str
    min_updraft_pa_s: float
    min_updraft_flux: float
    min_updraft_mesh_frac: float


_WORKER_MASK: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build D0/log10(Nw) CFADs using only strong updraft regions "
            "identified from UD_OMEGA and UD_MESH_FRAC."
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
    parser.add_argument(
        "--convection-metric",
        choices=("omega", "flux"),
        default="omega",
        help=(
            "Strong-convection mask. 'omega' uses -UD_OMEGA in Pa/s; "
            "'flux' uses (-UD_OMEGA * UD_MESH_FRAC) / g in kg m-2 s-1."
        ),
    )
    parser.add_argument(
        "--min-updraft-pa-s",
        type=float,
        default=10.0,
        help="Minimum upward pressure velocity for --convection-metric=omega.",
    )
    parser.add_argument(
        "--min-updraft-flux",
        type=float,
        default=0.01,
        help="Minimum updraft mass flux for --convection-metric=flux.",
    )
    parser.add_argument(
        "--min-updraft-mesh-frac",
        type=float,
        default=0.0,
        help="Minimum active updraft mesh fraction required for either metric.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--d0-range", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--lognw-range", type=float, nargs=2, default=(0.0, 9.0))
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_strong_convection", argv=argv)


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def slug_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def read_level_yx(path: Path, variable: str, mask_shape: tuple[int, int]) -> np.ndarray:
    with Dataset(path) as ds:
        arr = _as_float_array(ds.variables[variable][:])
    return _to_level_yx(arr, mask_shape, path, variable)


def masked_level_mean(field: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(field) & mask[np.newaxis, :, :]
    counts = np.sum(valid, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(valid, field, 0.0), axis=(1, 2))
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out, counts


def strong_convection_mask(
    omega: np.ndarray,
    mesh: np.ndarray,
    settings: StrongConvectionSettings,
) -> np.ndarray:
    active = np.isfinite(omega) & np.isfinite(mesh) & (mesh > settings.min_updraft_mesh_frac)
    upward_pa_s = -omega
    if settings.metric == "omega":
        return active & np.isfinite(upward_pa_s) & (upward_pa_s >= settings.min_updraft_pa_s)
    flux = upward_pa_s * mesh / G
    return active & np.isfinite(flux) & (flux >= settings.min_updraft_flux)


def level_mean_in_strong_convection(
    values: np.ndarray,
    base_mask: np.ndarray,
    convection_mask: np.ndarray,
    rainy_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask3d = base_mask[np.newaxis, :, :] & convection_mask & rainy_mask & np.isfinite(values)
    counts = np.sum(mask3d, axis=(1, 2)).astype(np.int64)
    sums = np.sum(np.where(mask3d, values, 0.0), axis=(1, 2))
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out, counts


def discover_strong_convection_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path,
    max_days: int | None,
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    records = discover_vertical_records(experiment, leads, netcdf_root, max_days)
    out: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for valid_dt, init_dt, lead, paths in records:
        base = paths["RAIN"].parents[2]
        day_name = paths["RAIN"].parent.name
        file_name = paths["RAIN"].name
        updraft_path = base / UPDRAFT_VARIABLE / day_name / file_name
        mesh_path = base / UPDRAFT_MESH_VARIABLE / day_name / file_name
        if updraft_path.exists() and mesh_path.exists():
            merged = dict(paths)
            merged[UPDRAFT_VARIABLE] = updraft_path
            merged[UPDRAFT_MESH_VARIABLE] = mesh_path
            out.append((valid_dt, init_dt, lead, merged))
    return out


def _empty_accumulator(n_levels: int, x_edges: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "height_sum": np.zeros(n_levels, dtype=np.float64),
        "height_count": np.zeros(n_levels, dtype=np.int64),
        "temperature_sum": np.zeros(n_levels, dtype=np.float64),
        "temperature_count": np.zeros(n_levels, dtype=np.int64),
        "hist": {
            field: np.zeros((n_levels, x_edges[field].size - 1), dtype=np.int64)
            for field in DSD_FIELDS
        },
        "selected_counts": {field: np.zeros(n_levels, dtype=np.int64) for field in DSD_FIELDS},
        "n_profiles": 0,
        "n_profiles_with_convection": 0,
    }


def add_profile_to_accumulator(
    acc: dict[str, Any],
    profile: dict[str, np.ndarray],
    x_edges: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not acc:
        acc.update(_empty_accumulator(profile["height_km"].size, x_edges))

    height = np.asarray(profile["height_km"], dtype=np.float64)
    ok_height = np.isfinite(height)
    acc["height_sum"][ok_height] += height[ok_height]
    acc["height_count"][ok_height] += 1

    temp = np.asarray(profile["temperature_k"], dtype=np.float64)
    ok_temp = np.isfinite(temp)
    acc["temperature_sum"][ok_temp] += temp[ok_temp]
    acc["temperature_count"][ok_temp] += 1

    any_convection = False
    for field in DSD_FIELDS:
        values = np.asarray(profile[field], dtype=np.float64)
        add_values_to_hist(acc["hist"][field], values, x_edges[field])
        counts = np.asarray(profile[f"{field}_selected_counts"], dtype=np.int64)
        acc["selected_counts"][field] += counts
        any_convection = any_convection or bool(np.any(counts > 0))
    acc["n_profiles"] += 1
    acc["n_profiles_with_convection"] += int(any_convection)
    return acc


def grids_from_accumulator(
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
            counts=acc["selected_counts"][field],
            n_profiles=int(acc["n_profiles"]),
            source=source,
        )
        for field in DSD_FIELDS
    }
    return grids, temp, freeze


def _process_task(
    task: tuple[str, dict[str, str], float, str, str, float, StrongConvectionSettings],
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    experiment, paths, min_qr, onemom_closure, twomom_closure, n0_fixed, settings = task
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    mask = _WORKER_MASK

    try:
        qr = read_level_yx(Path(paths["RAIN"]), "RAIN", mask.shape)
        temp = read_level_yx(Path(paths["TEMPERATURE"]), "TEMPERATURE", mask.shape)
        pres = read_level_yx(Path(paths["PRESSURE"]), "PRESSURE", mask.shape)
        height_m = read_level_yx(Path(paths["GEOPOTENTIEL"]), "GEOPOTENTIEL", mask.shape)
        omega = read_level_yx(Path(paths[UPDRAFT_VARIABLE]), UPDRAFT_VARIABLE, mask.shape)
        mesh = read_level_yx(Path(paths[UPDRAFT_MESH_VARIABLE]), UPDRAFT_MESH_VARIABLE, mask.shape)
        if experiment == "2mom":
            pnr = read_level_yx(Path(paths["PNR"]), "PNR", mask.shape)
        else:
            pnr = np.full(qr.shape, np.nan, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive worker logging
        return None, [f"WARNING {experiment} {paths.get('RAIN', '<missing>')}: {exc}"]

    n_levels = min(
        qr.shape[0],
        temp.shape[0],
        pres.shape[0],
        height_m.shape[0],
        omega.shape[0],
        mesh.shape[0],
        pnr.shape[0],
    )
    qr = qr[:n_levels]
    temp = temp[:n_levels]
    pres = pres[:n_levels]
    height_m = height_m[:n_levels]
    omega = omega[:n_levels]
    mesh = mesh[:n_levels]
    pnr = pnr[:n_levels]

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

    convection = strong_convection_mask(omega, mesh, settings)
    rainy = np.isfinite(qr) & (qr >= min_qr)
    height_km, _ = masked_level_mean(height_m / 1000.0, mask)
    temperature_k, _ = masked_level_mean(temp, mask)

    out = {
        "height_km": height_km,
        "temperature_k": temperature_k,
    }
    for field in DSD_FIELDS:
        values = np.asarray(diag[field], dtype=np.float64)
        means, counts = level_mean_in_strong_convection(values, mask, convection, rainy)
        out[field] = means
        out[f"{field}_selected_counts"] = counts
    return out, []


def compute_experiment_cfad(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    x_edges: dict[str, np.ndarray],
    *,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    settings: StrongConvectionSettings,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> ExperimentCfad:
    if not records:
        raise RuntimeError(f"No strong-convection-capable records found for {experiment}")
    tasks = [
        (
            experiment,
            {name: str(path) for name, path in rec[3].items()},
            min_qr,
            onemom_closure,
            twomom_closure,
            n0_fixed,
            settings,
        )
        for rec in records
    ]
    print(f"  [{experiment}] processing {len(tasks):,} strong-convection profiles", flush=True)

    acc: dict[str, Any] = {}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_task(task) for task in tasks)
        for idx, (profile, warnings) in enumerate(iterator, start=1):
            if profile is not None:
                add_profile_to_accumulator(acc, profile, x_edges)
            for warning in warnings:
                print(warning, flush=True)
            if idx % progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)
    else:
        with get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=tasks_per_child,
        ) as pool:
            for idx, (profile, warnings) in enumerate(
                pool.imap_unordered(_process_task, tasks),
                start=1,
            ):
                if profile is not None:
                    add_profile_to_accumulator(acc, profile, x_edges)
                for warning in warnings:
                    print(warning, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)

    if not acc:
        raise RuntimeError(f"No readable strong-convection profiles for {experiment}")
    grids, temp, freeze = grids_from_accumulator(
        acc,
        x_edges,
        source=str(records[0][3]["RAIN"].parent.parent),
    )
    return ExperimentCfad(grids=grids, temperature_k=temp, freezing_level_km=freeze)


def cache_path(cache_dir: Path, experiment: str, tag: str) -> Path:
    return cache_dir / f"{experiment}_{tag}.npz"


def save_experiment_cfad(path: Path, cfad: ExperimentCfad) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "temperature_k": cfad.temperature_k,
        "freezing_level_km": np.asarray([cfad.freezing_level_km], dtype=np.float64),
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
        for field in DSD_FIELDS:
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


def get_experiment_cfad(
    args: argparse.Namespace,
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    x_edges: dict[str, np.ndarray],
    tag: str,
    settings: StrongConvectionSettings,
) -> ExperimentCfad:
    path = cache_path(args.cache_dir, experiment, tag)
    if path.exists() and not args.recompute:
        return load_experiment_cfad(path)

    cfad = compute_experiment_cfad(
        experiment,
        records,
        domain_mask,
        x_edges,
        min_qr=args.min_qr,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        n0_fixed=args.n0_fixed,
        settings=settings,
        workers=max(1, int(args.workers)),
        progress_every=max(1, int(args.progress_every)),
        tasks_per_child=max(1, int(args.tasks_per_child)),
    )
    save_experiment_cfad(path, cfad)
    return cfad


def settings_from_args(args: argparse.Namespace) -> StrongConvectionSettings:
    return StrongConvectionSettings(
        metric=args.convection_metric,
        min_updraft_pa_s=float(args.min_updraft_pa_s),
        min_updraft_flux=float(args.min_updraft_flux),
        min_updraft_mesh_frac=float(args.min_updraft_mesh_frac),
    )


def cache_tag(args: argparse.Namespace, lead_tag: str, settings: StrongConvectionSettings) -> str:
    metric_threshold = (
        f"omega{slug_float(settings.min_updraft_pa_s)}pa_s"
        if settings.metric == "omega"
        else f"flux{slug_float(settings.min_updraft_flux)}kg_m2_s"
    )
    tag = (
        f"{lead_tag}_{args.onemom_closure}_{args.twomom_closure}_"
        f"{settings.metric}_{metric_threshold}_mesh{slug_float(settings.min_updraft_mesh_frac)}_"
        f"bins{args.bins}"
    )
    if args.max_days is not None:
        tag += f"_first{args.max_days}days"
    return tag


def write_data_txt(
    path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: list[str],
    *,
    lead_tag: str,
    args: argparse.Namespace,
    settings: StrongConvectionSettings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Strong-convection CFAD-style DSD diagnostic distributions\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"DSD lead selection: {lead_tag}\n")
        fh.write("Colormap: cmaps.WhViBlGrYeOrRe\n")
        fh.write(f"Strong convection metric: {settings.metric}\n")
        fh.write(
            "Strong convection mask: "
            f"{UPDRAFT_MESH_VARIABLE} > {settings.min_updraft_mesh_frac:g}; "
        )
        if settings.metric == "omega":
            fh.write(f"-{UPDRAFT_VARIABLE} >= {settings.min_updraft_pa_s:g} Pa s-1\n")
        else:
            fh.write(
                f"(-{UPDRAFT_VARIABLE} * {UPDRAFT_MESH_VARIABLE}) / g >= "
                f"{settings.min_updraft_flux:g} kg m-2 s-1\n"
            )
        fh.write("DSD values are level means over radar-domain cells satisfying the mask.\n")
        fh.write("Height and freezing level use radar-domain level means.\n")
        fh.write(f"min_qr: {args.min_qr:g} kg kg-1\n")
        fh.write(f"bins: {args.bins}\n")
        fh.write(f"d0_range: {tuple(args.d0_range)}\n")
        fh.write(f"lognw_range: {tuple(args.lognw_range)}\n")
        fh.write("\n")
        fh.write(
            "experiment,field,n_profiles,selected_gridpoint_samples,"
            "histogram_profile_samples,freezing_level_km,source\n"
        )
        for experiment in experiments:
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            freeze = cfads[experiment].freezing_level_km
            for field in DSD_FIELDS:
                grid = cfads[experiment].grids[field]
                fh.write(
                    f"{experiment},{label},{field},{grid.n_profiles},"
                    f"{int(np.sum(grid.counts))},{int(np.sum(grid.hist))},"
                    f"{freeze:.10e},{grid.source}\n"
                )


def build_cfads(args: argparse.Namespace) -> tuple[dict[str, ExperimentCfad], str, StrongConvectionSettings]:
    leads = parse_lead_selection(args.lead)
    lead_tag = lead_label(leads)
    settings = settings_from_args(args)
    tag = cache_tag(args, lead_tag, settings)
    x_edges = x_edges_for_args(args)

    records_by_experiment = {
        experiment: discover_strong_convection_records(
            experiment,
            leads,
            args.netcdf_root,
            args.max_days,
        )
        for experiment in args.experiments
    }
    sample_records = next((records for records in records_by_experiment.values() if records), None)
    if not sample_records:
        raise RuntimeError("No masked-NetCDF records with updraft fields found")
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
    print(
        "Strong convection: "
        f"{settings.metric}, min omega={settings.min_updraft_pa_s:g} Pa/s, "
        f"min flux={settings.min_updraft_flux:g} kg m-2 s-1, "
        f"min mesh={settings.min_updraft_mesh_frac:g}",
        flush=True,
    )

    cfads: dict[str, ExperimentCfad] = {}
    for experiment in args.experiments:
        print(f"Processing {EXPERIMENT_LABELS.get(experiment, experiment)}", flush=True)
        cfads[experiment] = get_experiment_cfad(
            args,
            experiment,
            records_by_experiment[experiment],
            domain_mask,
            x_edges,
            tag,
            settings,
        )
    return cfads, lead_tag, settings


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfads, lead_tag, settings = build_cfads(args)
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
        settings=settings,
    )
    print(f"Saved figure: {output_path}", flush=True)
    print(f"Saved data:   {text_path}", flush=True)


if __name__ == "__main__":
    main()
