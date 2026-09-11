"""Strong cfad. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.analysis.cfad import add_values_to_hist
from alaro_analysis.analysis.convection_sampling import (
    level_mean_in_strong_convection,
    masked_level_mean,
    strong_convection_mask,
)
from alaro_analysis.common.constants import EXPERIMENT_LABELS, RD, RUNS_ROOT
from alaro_analysis.common.dsd import (
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.common.dsd_models import (
    CfadGrid,
    ExperimentCfad,
    StrongConvectionSettings,
)
from alaro_analysis.data.cache import read_cache_signature, source_fingerprint
from alaro_analysis.data.cache import signature as cache_signature
from alaro_analysis.data.cfad_cache import load_experiment_cfad, save_experiment_cfad
from alaro_analysis.data.dsd_profiles import discover_vertical_records, read_level_yx
from alaro_analysis.data.masks import DomainMask, build_domain_mask_from_netcdf
from alaro_analysis.data.model_records import lead_label, parse_lead_selection
from alaro_analysis.dsd.profile_cfad import DSD_FIELDS, x_edges_for_args
from alaro_analysis.dsd.profile_cfad import FIGURE_NAME as BASE_FIGURE_NAME
from alaro_analysis.dsd.profile_cfad import TEXT_NAME as BASE_TEXT_NAME
from alaro_analysis.dsd.vertical_profiles import freezing_level_km

OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_strong_convection"


DATA_TXT_DIR = OUTPUT_DIR / "data_txt"


CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_strong_convection"


FIGURE_NAME = BASE_FIGURE_NAME.replace("dsd_cfad", "dsd_cfad_strong_convection")


TEXT_NAME = BASE_TEXT_NAME.replace("dsd_cfad", "dsd_cfad_strong_convection")


UPDRAFT_VARIABLE = "UD_OMEGA"


UPDRAFT_MESH_VARIABLE = "UD_MESH_FRAC"


_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def slug_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


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


def experiment_cache_signature(
    args: argparse.Namespace,
    experiment: str,
    records,
    domain_mask: DomainMask,
    x_edges: dict[str, np.ndarray],
    settings: StrongConvectionSettings,
) -> str:
    edges = np.concatenate([np.asarray(x_edges[f], dtype=np.float64) for f in sorted(x_edges)])
    return cache_signature(
        {
            "cache_version": 1,
            "experiment": experiment,
            "min_qr": args.min_qr,
            "onemom_closure": args.onemom_closure,
            "twomom_closure": args.twomom_closure,
            "n0_fixed": args.n0_fixed,
            "settings": repr(settings),
            "x_edges": edges,
            "mask": np.asarray(domain_mask.mask),
            "n_records": len(records),
            "source_inventory": source_fingerprint(records),
            "source": str(records[0][3]["RAIN"].parent.parent) if records else "",
        }
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
    sig = experiment_cache_signature(args, experiment, records, domain_mask, x_edges, settings)
    path = cache_path(args.cache_dir, experiment, tag)
    if path.exists() and not args.recompute:
        if read_cache_signature(path) == sig:
            return load_experiment_cfad(path)
        print(
            f"  [{experiment}] cache {path.name} parameters changed/absent; recomputing.",
            flush=True,
        )

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
    save_experiment_cfad(path, cfad, sig)
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


def build_cfads(
    args: argparse.Namespace,
) -> tuple[dict[str, ExperimentCfad], str, StrongConvectionSettings]:
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
