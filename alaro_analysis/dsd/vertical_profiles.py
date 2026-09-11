"""Vertical profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS, FREEZING_K, RD, RUNS_ROOT
from alaro_analysis.common.dsd import (
    gamma_dsd_from_q_n_per_kg,
    mp_from_q_abel_boutle,
    mp_from_q_fixed_n0,
    mp_from_q_n_per_kg,
)
from alaro_analysis.common.dsd_models import (
    DSD_FIELDS,
    DdhQvProfile,
    DsdVerticalProfile,
)
from alaro_analysis.data.cache import read_cache_signature, source_fingerprint
from alaro_analysis.data.cache import signature as cache_signature
from alaro_analysis.data.dsd_profile_cache import (
    load_ddh_qv_profile,
    load_dsd_profile,
    save_dsd_profile,
)
from alaro_analysis.data.dsd_profiles import (
    discover_vertical_records,
    read_masked_level_mean,
)
from alaro_analysis.data.masks import DomainMask, build_domain_mask_from_netcdf
from alaro_analysis.data.model_records import lead_label, parse_lead_selection

OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_vertical_profiles"


DATA_TXT_DIR = OUTPUT_DIR / "data_txt"


CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_vertical_profiles"


FIGURE_NAME = "dsd_vertical_profiles_D0_logNw_DDH_QV.png"


TEXT_NAME = "dsd_vertical_profiles_D0_logNw_DDH_QV.txt"


_WORKER_MASK: np.ndarray | None = None


def _lead_text(value: str) -> str:
    return f"{int(value):04d}"


def ddh_lead_text(leads: tuple[int, ...] | None, ddh_lead: str | None) -> str:
    if ddh_lead is not None:
        return _lead_text(ddh_lead)
    return "0024"


def cache_tag(
    leads: tuple[int, ...] | None,
    *,
    onemom_closure: str,
    twomom_closure: str,
    max_days: int | None,
) -> str:
    tag = f"{lead_label(leads)}_{onemom_closure}_{twomom_closure}"
    if max_days is not None:
        tag += f"_first{max_days}days"
    return tag


def dsd_cache_path(cache_dir: Path, experiment: str, tag: str) -> Path:
    return cache_dir / f"{experiment}_{tag}.npz"


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def empty_partial(n_levels: int) -> dict[str, Any]:
    return {
        "sums": {field: np.zeros(n_levels, dtype=np.float64) for field in DSD_FIELDS},
        "counts": {field: np.zeros(n_levels, dtype=np.int64) for field in DSD_FIELDS},
        "height_sum": np.zeros(n_levels, dtype=np.float64),
        "height_count": np.zeros(n_levels, dtype=np.int64),
        "temperature_sum": np.zeros(n_levels, dtype=np.float64),
        "temperature_count": np.zeros(n_levels, dtype=np.int64),
        "n_files": 0,
    }


def add_partial(total: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    if not total:
        total.update(empty_partial(part["height_sum"].size))
    for field in DSD_FIELDS:
        total["sums"][field] += part["sums"][field]
        total["counts"][field] += part["counts"][field]
    total["height_sum"] += part["height_sum"]
    total["height_count"] += part["height_count"]
    total["temperature_sum"] += part["temperature_sum"]
    total["temperature_count"] += part["temperature_count"]
    total["n_files"] += int(part["n_files"])
    return total


def _process_profile_task(
    task: tuple[str, dict[str, str], float, str, str, float],
) -> tuple[dict[str, Any] | None, list[str]]:
    experiment, paths, min_qr, onemom_closure, twomom_closure, n0_fixed = task
    if _WORKER_MASK is None:
        raise RuntimeError("Worker mask not initialised")
    mask = _WORKER_MASK
    warnings: list[str] = []

    try:
        qr, _ = read_masked_level_mean(Path(paths["RAIN"]), "RAIN", mask)
        temp, _ = read_masked_level_mean(Path(paths["TEMPERATURE"]), "TEMPERATURE", mask)
        pres, _ = read_masked_level_mean(Path(paths["PRESSURE"]), "PRESSURE", mask)
        height_m, height_count = read_masked_level_mean(
            Path(paths["GEOPOTENTIEL"]), "GEOPOTENTIEL", mask
        )
        if experiment == "2mom":
            pnr, _ = read_masked_level_mean(Path(paths["PNR"]), "PNR", mask)
        else:
            pnr = np.full(qr.shape, np.nan, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive worker logging
        label = paths.get("RAIN", "<missing RAIN>")
        return None, [f"WARNING {experiment} {label}: {exc}"]

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

    part = empty_partial(qr.size)
    rainy = np.isfinite(qr) & (qr >= min_qr)
    for field in DSD_FIELDS:
        values = np.asarray(diag[field], dtype=np.float64)
        valid = rainy & np.isfinite(values)
        part["sums"][field][valid] = values[valid]
        part["counts"][field][valid] = 1

    height_valid = np.isfinite(height_m)
    part["height_sum"][height_valid] = height_m[height_valid] / 1000.0
    part["height_count"][height_valid] = height_count[height_valid] > 0
    temp_valid = np.isfinite(temp)
    part["temperature_sum"][temp_valid] = temp[temp_valid]
    part["temperature_count"][temp_valid] = 1
    part["n_files"] = 1
    return part, warnings


def profile_from_accumulator(acc: dict[str, Any], source: str) -> DsdVerticalProfile:
    height = np.full(acc["height_sum"].shape, np.nan, dtype=np.float64)
    height_ok = acc["height_count"] > 0
    height[height_ok] = acc["height_sum"][height_ok] / acc["height_count"][height_ok]
    temperature = np.full(acc["temperature_sum"].shape, np.nan, dtype=np.float64)
    temp_ok = acc["temperature_count"] > 0
    temperature[temp_ok] = acc["temperature_sum"][temp_ok] / acc["temperature_count"][temp_ok]

    values: dict[str, np.ndarray] = {}
    for field in DSD_FIELDS:
        counts = acc["counts"][field]
        vals = np.full(counts.shape, np.nan, dtype=np.float64)
        ok = counts > 0
        vals[ok] = acc["sums"][field][ok] / counts[ok]
        values[field] = vals

    return DsdVerticalProfile(
        height_km=height,
        temperature_k=temperature,
        temperature_count=acc["temperature_count"].copy(),
        values=values,
        counts={field: acc["counts"][field].copy() for field in DSD_FIELDS},
        n_files=int(acc["n_files"]),
        source=source,
    )


def compute_dsd_profile(
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    *,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
) -> DsdVerticalProfile:
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
    print(f"  [{experiment}] processing {len(tasks):,} vertical profile files", flush=True)

    acc: dict[str, Any] = {}
    if workers <= 1:
        _init_worker(domain_mask.mask)
        iterator = (_process_profile_task(task) for task in tasks)
        for idx, (part, warnings) in enumerate(iterator, start=1):
            if part is not None:
                add_partial(acc, part)
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
            for idx, (part, warnings) in enumerate(
                pool.imap_unordered(_process_profile_task, tasks),
                start=1,
            ):
                if part is not None:
                    add_partial(acc, part)
                for warning in warnings:
                    print(warning, flush=True)
                if idx % progress_every == 0 or idx == len(tasks):
                    print(f"  [{experiment}] processed {idx}/{len(tasks)}", flush=True)

    if not acc:
        raise RuntimeError(f"No readable profile data for {experiment}")
    return profile_from_accumulator(acc, source=str(records[0][3]["RAIN"].parent.parent))


def get_dsd_profile(
    *,
    cache_dir: Path,
    experiment: str,
    tag: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    min_qr: float,
    onemom_closure: str,
    twomom_closure: str,
    n0_fixed: float,
    workers: int,
    progress_every: int,
    tasks_per_child: int,
    recompute: bool,
) -> DsdVerticalProfile:
    sig = cache_signature(
        {
            "cache_version": 1,
            "experiment": experiment,
            "min_qr": min_qr,
            "onemom_closure": onemom_closure,
            "twomom_closure": twomom_closure,
            "n0_fixed": n0_fixed,
            "mask": np.asarray(domain_mask.mask),
            "n_records": len(records),
            "source_inventory": source_fingerprint(records),
            "source": str(records[0][3]["RAIN"].parent.parent) if records else "",
        }
    )
    path = dsd_cache_path(cache_dir, experiment, tag)
    if path.exists() and not recompute:
        if read_cache_signature(path) == sig:
            return load_dsd_profile(path)
        print(
            f"  [{experiment}] cache {path.name} parameters changed/absent; recomputing.",
            flush=True,
        )
    profile = compute_dsd_profile(
        experiment,
        records,
        domain_mask,
        min_qr=min_qr,
        onemom_closure=onemom_closure,
        twomom_closure=twomom_closure,
        n0_fixed=n0_fixed,
        workers=workers,
        progress_every=progress_every,
        tasks_per_child=tasks_per_child,
    )
    save_dsd_profile(path, profile, sig)
    return profile


def freezing_level_km(height_km: np.ndarray, temperature_k: np.ndarray) -> float:
    z = np.asarray(height_km, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(t)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    z = z[ok]
    t = t[ok]
    order = np.argsort(z)
    z = z[order]
    t = t[order]
    diff = t - FREEZING_K
    crossings = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
    if crossings.size == 0:
        return float("nan")
    idx = int(crossings[0])
    denom = diff[idx] - diff[idx + 1]
    if denom == 0.0:
        return float(z[idx])
    weight = diff[idx] / denom
    return float(z[idx] + weight * (z[idx + 1] - z[idx]))


def freezing_levels_by_experiment(
    dsd_profiles: dict[str, DsdVerticalProfile],
    experiments: list[str],
) -> dict[str, float]:
    return {
        experiment: freezing_level_km(
            dsd_profiles[experiment].height_km,
            dsd_profiles[experiment].temperature_k,
        )
        for experiment in experiments
    }


def build_profiles(
    args: argparse.Namespace,
) -> tuple[
    dict[str, DsdVerticalProfile],
    dict[str, DdhQvProfile],
    str,
    str,
]:
    leads = parse_lead_selection(args.lead)
    lead_tag = lead_label(leads)
    ddh_lead = ddh_lead_text(leads, args.ddh_lead)
    tag = cache_tag(
        leads,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        max_days=args.max_days,
    )

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

    dsd_profiles: dict[str, DsdVerticalProfile] = {}
    qv_profiles: dict[str, DdhQvProfile] = {}
    for experiment in args.experiments:
        print(f"Processing {EXPERIMENT_LABELS.get(experiment, experiment)}", flush=True)
        dsd_profiles[experiment] = get_dsd_profile(
            cache_dir=args.cache_dir,
            experiment=experiment,
            tag=tag,
            records=records_by_experiment[experiment],
            domain_mask=domain_mask,
            min_qr=args.min_qr,
            onemom_closure=args.onemom_closure,
            twomom_closure=args.twomom_closure,
            n0_fixed=args.n0_fixed,
            workers=max(1, int(args.workers)),
            progress_every=max(1, int(args.progress_every)),
            tasks_per_child=max(1, int(args.tasks_per_child)),
            recompute=args.recompute,
        )
        qv_profiles[experiment] = load_ddh_qv_profile(
            args.ddh_agg_dir,
            experiment,
            lead_text_value=ddh_lead,
            block=args.qv_block,
        )

    return dsd_profiles, qv_profiles, lead_tag, ddh_lead
