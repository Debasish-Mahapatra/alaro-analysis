"""Gridcell cfad. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alaro_analysis.analysis.convection_sampling import strong_convection_mask
from alaro_analysis.common.constants import EXPERIMENT_LABELS, RD, RUNS_ROOT
from alaro_analysis.common.dsd import (
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.common.dsd_models import (
    CfadGrid,
    DomainPair,
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
from alaro_analysis.dsd.strong_cfad import (
    UPDRAFT_MESH_VARIABLE,
    UPDRAFT_VARIABLE,
    settings_from_args,
    slug_float,
)
from alaro_analysis.dsd.vertical_profiles import freezing_level_km

OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_gridcell"


DATA_TXT_DIR = OUTPUT_DIR / "data_txt"


CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_gridcell"


FULL_FIGURE_NAME = "dsd_cfad_gridcell_full_D0_logNw_matched_colorbar.png"


STRONG_FIGURE_NAME = "dsd_cfad_gridcell_strong_convection_D0_logNw_matched_colorbar.png"


FULL_TEXT_NAME = "dsd_cfad_gridcell_full_D0_logNw_matched_colorbar.txt"


STRONG_TEXT_NAME = "dsd_cfad_gridcell_strong_convection_D0_logNw_matched_colorbar.txt"


DOMAIN_FULL = "full"


DOMAIN_STRONG = "strong"


_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def discover_gridcell_records(
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


def group_records_by_day(
    records: Sequence[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
) -> list[list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]]:
    groups: dict[str, list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]] = {}
    for record in records:
        day_name = record[3]["RAIN"].parent.name
        groups.setdefault(day_name, []).append(record)
    return [groups[key] for key in sorted(groups)]


def empty_accumulator(n_levels: int, x_edges: dict[str, np.ndarray]) -> dict[str, Any]:
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


def ensure_accumulator(
    acc: dict[str, Any],
    n_levels: int,
    x_edges: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not acc:
        acc.update(empty_accumulator(n_levels, x_edges))
    return acc


def add_level_sums(
    acc: dict[str, Any], height_m: np.ndarray, temp_k: np.ndarray, mask: np.ndarray
) -> None:
    valid_height = np.isfinite(height_m) & mask[np.newaxis, :, :]
    acc["height_sum"] += np.sum(np.where(valid_height, height_m / 1000.0, 0.0), axis=(1, 2))
    acc["height_count"] += np.sum(valid_height, axis=(1, 2)).astype(np.int64)

    valid_temp = np.isfinite(temp_k) & mask[np.newaxis, :, :]
    acc["temperature_sum"] += np.sum(np.where(valid_temp, temp_k, 0.0), axis=(1, 2))
    acc["temperature_count"] += np.sum(valid_temp, axis=(1, 2)).astype(np.int64)


def add_grid_values_to_hist(
    hist: np.ndarray,
    values: np.ndarray,
    sample_mask: np.ndarray,
    edges: np.ndarray,
) -> None:
    vals = np.asarray(values, dtype=np.float64)
    n_levels = min(hist.shape[0], vals.shape[0], sample_mask.shape[0])
    vals = vals[:n_levels]
    mask = sample_mask[:n_levels] & np.isfinite(vals)
    idx = np.searchsorted(edges, vals, side="right") - 1
    n_bins = edges.size - 1
    valid = mask & (idx >= 0) & (idx < n_bins)
    if not np.any(valid):
        return
    levels = np.broadcast_to(np.arange(n_levels)[:, np.newaxis, np.newaxis], vals.shape)
    linear = levels[valid] * n_bins + idx[valid]
    hist[:n_levels] += np.bincount(linear, minlength=n_levels * n_bins).reshape(n_levels, n_bins)


def add_partial(
    total: dict[str, Any], part: dict[str, Any], x_edges: dict[str, np.ndarray]
) -> dict[str, Any]:
    if not part:
        return total
    ensure_accumulator(total, part["height_sum"].size, x_edges)
    total["height_sum"] += part["height_sum"]
    total["height_count"] += part["height_count"]
    total["temperature_sum"] += part["temperature_sum"]
    total["temperature_count"] += part["temperature_count"]
    total["n_profiles"] += int(part["n_profiles"])
    for field in DSD_FIELDS:
        total["hist"][field] += part["hist"][field]
    return total


def cfad_from_accumulator(
    acc: dict[str, Any],
    x_edges: dict[str, np.ndarray],
    *,
    source: str,
) -> ExperimentCfad:
    height = np.full(acc["height_sum"].shape, np.nan, dtype=np.float64)
    ok_height = acc["height_count"] > 0
    height[ok_height] = acc["height_sum"][ok_height] / acc["height_count"][ok_height]

    temp = np.full(acc["temperature_sum"].shape, np.nan, dtype=np.float64)
    ok_temp = acc["temperature_count"] > 0
    temp[ok_temp] = acc["temperature_sum"][ok_temp] / acc["temperature_count"][ok_temp]

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
    return ExperimentCfad(
        grids=grids,
        temperature_k=temp,
        freezing_level_km=freezing_level_km(height, temp),
    )


def _process_day_task(
    task: tuple[
        str,
        list[dict[str, str]],
        float,
        str,
        str,
        float,
        StrongConvectionSettings,
        dict[str, np.ndarray],
    ],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    experiment, path_rows, min_qr, onemom_closure, twomom_closure, n0_fixed, settings, x_edges = (
        task
    )
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    mask2d = _WORKER_MASK
    partials: dict[str, dict[str, Any]] = {DOMAIN_FULL: {}, DOMAIN_STRONG: {}}
    warnings: list[str] = []

    for paths in path_rows:
        try:
            qr = read_level_yx(Path(paths["RAIN"]), "RAIN", mask2d.shape)
            temp = read_level_yx(Path(paths["TEMPERATURE"]), "TEMPERATURE", mask2d.shape)
            pres = read_level_yx(Path(paths["PRESSURE"]), "PRESSURE", mask2d.shape)
            height_m = read_level_yx(Path(paths["GEOPOTENTIEL"]), "GEOPOTENTIEL", mask2d.shape)
            omega = read_level_yx(Path(paths[UPDRAFT_VARIABLE]), UPDRAFT_VARIABLE, mask2d.shape)
            mesh = read_level_yx(
                Path(paths[UPDRAFT_MESH_VARIABLE]), UPDRAFT_MESH_VARIABLE, mask2d.shape
            )
            if experiment == "2mom":
                pnr = read_level_yx(Path(paths["PNR"]), "PNR", mask2d.shape)
            else:
                pnr = np.full(qr.shape, np.nan, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive worker logging
            warnings.append(f"WARNING {experiment} {paths.get('RAIN', '<missing>')}: {exc}")
            continue

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

        for domain in (DOMAIN_FULL, DOMAIN_STRONG):
            ensure_accumulator(partials[domain], n_levels, x_edges)
            add_level_sums(partials[domain], height_m, temp, mask2d)
            partials[domain]["n_profiles"] += 1

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

        base_mask = mask2d[np.newaxis, :, :] & np.isfinite(qr) & (qr >= min_qr)
        strong_mask = base_mask & strong_convection_mask(omega, mesh, settings)
        for field in DSD_FIELDS:
            values = np.asarray(diag[field], dtype=np.float64)
            add_grid_values_to_hist(
                partials[DOMAIN_FULL]["hist"][field],
                values,
                base_mask,
                x_edges[field],
            )
            add_grid_values_to_hist(
                partials[DOMAIN_STRONG]["hist"][field],
                values,
                strong_mask,
                x_edges[field],
            )

    return partials, warnings


def cache_path(cache_dir: Path, domain: str, experiment: str, tag: str) -> Path:
    return cache_dir / f"{domain}_{experiment}_{tag}.npz"


def pair_cache_signature(
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


def get_experiment_pair(
    args: argparse.Namespace,
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    x_edges: dict[str, np.ndarray],
    tag: str,
    settings: StrongConvectionSettings,
) -> DomainPair:
    sig = pair_cache_signature(args, experiment, records, domain_mask, x_edges, settings)
    full_path = cache_path(args.cache_dir, DOMAIN_FULL, experiment, tag)
    strong_path = cache_path(args.cache_dir, DOMAIN_STRONG, experiment, tag)
    if (
        not args.recompute
        and full_path.exists()
        and strong_path.exists()
        and read_cache_signature(full_path) == sig
        and read_cache_signature(strong_path) == sig
    ):
        return DomainPair(
            full=load_experiment_cfad(full_path),
            strong=load_experiment_cfad(strong_path),
        )

    if not records:
        raise RuntimeError(f"No grid-cell records found for {experiment}")
    grouped = group_records_by_day(records)
    tasks = [
        (
            experiment,
            [{name: str(path) for name, path in rec[3].items()} for rec in group],
            args.min_qr,
            args.onemom_closure,
            args.twomom_closure,
            args.n0_fixed,
            settings,
            x_edges,
        )
        for group in grouped
    ]
    print(
        f"  [{experiment}] processing {len(records):,} files in {len(tasks):,} day tasks",
        flush=True,
    )

    totals: dict[str, dict[str, Any]] = {DOMAIN_FULL: {}, DOMAIN_STRONG: {}}
    if args.workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_day_task(task) for task in tasks)
        for idx, (partials, warnings) in enumerate(iterator, start=1):
            for domain in (DOMAIN_FULL, DOMAIN_STRONG):
                add_partial(totals[domain], partials[domain], x_edges)
            for warning in warnings:
                print(warning, flush=True)
            if idx % args.progress_every == 0 or idx == len(tasks):
                print(f"  [{experiment}] processed {idx}/{len(tasks)} day tasks", flush=True)
    else:
        with get_context("fork").Pool(
            processes=max(1, int(args.workers)),
            initializer=_init_worker,
            initargs=(domain_mask.mask,),
            maxtasksperchild=max(1, int(args.tasks_per_child)),
        ) as pool:
            for idx, (partials, warnings) in enumerate(
                pool.imap_unordered(_process_day_task, tasks),
                start=1,
            ):
                for domain in (DOMAIN_FULL, DOMAIN_STRONG):
                    add_partial(totals[domain], partials[domain], x_edges)
                for warning in warnings:
                    print(warning, flush=True)
                if idx % args.progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)} day tasks", flush=True)

    source = str(records[0][3]["RAIN"].parent.parent)
    pair = DomainPair(
        full=cfad_from_accumulator(totals[DOMAIN_FULL], x_edges, source=source),
        strong=cfad_from_accumulator(totals[DOMAIN_STRONG], x_edges, source=source),
    )
    save_experiment_cfad(full_path, pair.full, sig)
    save_experiment_cfad(strong_path, pair.strong, sig)
    return pair


def tag_from_args(
    args: argparse.Namespace, lead_tag: str, settings: StrongConvectionSettings
) -> str:
    metric_threshold = (
        f"omega{slug_float(settings.min_updraft_pa_s)}pa_s"
        if settings.metric == "omega"
        else f"flux{slug_float(settings.min_updraft_flux)}kg_m2_s"
    )
    tag = (
        f"{lead_tag}_{args.onemom_closure}_{args.twomom_closure}_"
        f"{settings.metric}_{metric_threshold}_mesh{slug_float(settings.min_updraft_mesh_frac)}_"
        f"gridcell_bins{args.bins}"
    )
    if args.max_days is not None:
        tag += f"_first{args.max_days}days"
    return tag


def build_pairs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, ExperimentCfad],
    dict[str, ExperimentCfad],
    str,
    StrongConvectionSettings,
]:
    leads = parse_lead_selection(args.lead)
    lead_tag = lead_label(leads)
    settings = settings_from_args(args)
    tag = tag_from_args(args, lead_tag, settings)
    x_edges = x_edges_for_args(args)

    records_by_experiment = {
        experiment: discover_gridcell_records(
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

    full_cfads: dict[str, ExperimentCfad] = {}
    strong_cfads: dict[str, ExperimentCfad] = {}
    for experiment in args.experiments:
        print(f"Processing {EXPERIMENT_LABELS.get(experiment, experiment)}", flush=True)
        pair = get_experiment_pair(
            args,
            experiment,
            records_by_experiment[experiment],
            domain_mask,
            x_edges,
            tag,
            settings,
        )
        full_cfads[experiment] = pair.full
        strong_cfads[experiment] = pair.strong
    return full_cfads, strong_cfads, lead_tag, settings
