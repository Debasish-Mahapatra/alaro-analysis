"""Profile cfad. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.analysis.cfad import add_values_to_hist
from alaro_analysis.common.constants import EXPERIMENT_LABELS, RD, RUNS_ROOT
from alaro_analysis.common.dsd import (
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.common.dsd_models import CfadGrid, ExperimentCfad
from alaro_analysis.data.cache import read_cache_signature, source_fingerprint
from alaro_analysis.data.cache import signature as cache_signature
from alaro_analysis.data.cfad_cache import load_experiment_cfad, save_experiment_cfad
from alaro_analysis.data.dsd_profiles import (
    discover_vertical_records,
    read_masked_level_mean,
)
from alaro_analysis.data.masks import build_domain_mask_from_netcdf
from alaro_analysis.data.model_records import lead_label, parse_lead_selection
from alaro_analysis.dsd.vertical_profiles import freezing_level_km

OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_profiles"


DATA_TXT_DIR = OUTPUT_DIR / "data_txt"


CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_profiles"


FIGURE_NAME = "dsd_cfad_D0_logNw.png"


TEXT_NAME = "dsd_cfad_D0_logNw.txt"


DSD_FIELDS = ("d0_mm", "log_nw")


PANEL_FIELDS = DSD_FIELDS


_WORKER_MASK: np.ndarray | None = None


def x_edges_for_args(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {
        "d0_mm": np.linspace(args.d0_range[0], args.d0_range[1], args.bins + 1),
        "log_nw": np.linspace(args.lognw_range[0], args.lognw_range[1], args.bins + 1),
    }


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
            "source_inventory": source_fingerprint(records),
            "source": str(records[0][3]["RAIN"].parent.parent) if records else "",
        }
    )


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
