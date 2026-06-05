"""Grid-cell DSD CFADs for full-domain and strong-updraft samples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

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
    CfadGrid,
    ExperimentCfad,
    frequency_percent,
    plot_cfads,
    save_experiment_cfad,
    load_experiment_cfad,
    x_edges_for_args,
)
from alaro_analysis.workflows.dsd_cfad_strong_convection import (
    UPDRAFT_MESH_VARIABLE,
    UPDRAFT_VARIABLE,
    StrongConvectionSettings,
    read_level_yx,
    settings_from_args,
    slug_float,
    strong_convection_mask,
)
from alaro_analysis.workflows.dsd_vertical_profiles import (
    discover_vertical_records,
    freezing_level_km,
)


OUTPUT_DIR = RUNS_ROOT / "figures" / "dsd_cfad_gridcell"
DATA_TXT_DIR = OUTPUT_DIR / "data_txt"
CACHE_DIR = RUNS_ROOT / "processed-data" / "dsd_cfad_gridcell"
FULL_FIGURE_NAME = "dsd_cfad_gridcell_full_D0_logNw_matched_colorbar.png"
STRONG_FIGURE_NAME = "dsd_cfad_gridcell_strong_convection_D0_logNw_matched_colorbar.png"
FULL_TEXT_NAME = "dsd_cfad_gridcell_full_D0_logNw_matched_colorbar.txt"
STRONG_TEXT_NAME = "dsd_cfad_gridcell_strong_convection_D0_logNw_matched_colorbar.txt"
DOMAIN_FULL = "full"
DOMAIN_STRONG = "strong"


@dataclass(frozen=True)
class DomainPair:
    full: ExperimentCfad
    strong: ExperimentCfad


_WORKER_MASK: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched full-domain and strong-updraft CFADs from grid-cell "
            "D0/log10(Nw), not from domain-mean DSD profiles."
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
    parser.add_argument("--tasks-per-child", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=50)
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
    )
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-updraft-flux", type=float, default=0.01)
    parser.add_argument("--min-updraft-mesh-frac", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--d0-range", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--lognw-range", type=float, nargs=2, default=(0.0, 9.0))
    parser.add_argument("--max-height-km", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument(
        "--frequency-percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile of positive frequency values used for the matched "
            "row colorbar scale. Use 100 for the literal maximum."
        ),
    )
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_gridcell_matched", argv=argv)


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


def add_level_sums(acc: dict[str, Any], height_m: np.ndarray, temp_k: np.ndarray, mask: np.ndarray) -> None:
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


def add_partial(total: dict[str, Any], part: dict[str, Any], x_edges: dict[str, np.ndarray]) -> dict[str, Any]:
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
    task: tuple[str, list[dict[str, str]], float, str, str, float, StrongConvectionSettings, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    experiment, path_rows, min_qr, onemom_closure, twomom_closure, n0_fixed, settings, x_edges = task
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
            mesh = read_level_yx(Path(paths[UPDRAFT_MESH_VARIABLE]), UPDRAFT_MESH_VARIABLE, mask2d.shape)
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


def get_experiment_pair(
    args: argparse.Namespace,
    experiment: str,
    records: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]],
    domain_mask: DomainMask,
    x_edges: dict[str, np.ndarray],
    tag: str,
    settings: StrongConvectionSettings,
) -> DomainPair:
    full_path = cache_path(args.cache_dir, DOMAIN_FULL, experiment, tag)
    strong_path = cache_path(args.cache_dir, DOMAIN_STRONG, experiment, tag)
    if full_path.exists() and strong_path.exists() and not args.recompute:
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
    save_experiment_cfad(full_path, pair.full)
    save_experiment_cfad(strong_path, pair.strong)
    return pair


def percentile_vmax(
    cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    field: str,
    percentile: float,
) -> float:
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    if percentile >= 100.0:
        return max(1.0, float(np.nanmax(merged)))
    return max(1.0, float(np.nanpercentile(merged, percentile)))


def matched_vmax_by_field(
    full_cfads: dict[str, ExperimentCfad],
    strong_cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    percentile: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in DSD_FIELDS:
        out[field] = max(
            percentile_vmax(full_cfads, experiments, field, percentile),
            percentile_vmax(strong_cfads, experiments, field, percentile),
        )
    return out


def tag_from_args(args: argparse.Namespace, lead_tag: str, settings: StrongConvectionSettings) -> str:
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


def write_data_txt(
    path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    *,
    title: str,
    lead_tag: str,
    args: argparse.Namespace,
    settings: StrongConvectionSettings,
    row_vmax_values: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(title + "\n")
        fh.write("=" * 72 + "\n")
        fh.write("DSD values: grid-cell D0/log10(Nw) samples binned by model level.\n")
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
        fh.write(f"min_qr: {args.min_qr:g} kg kg-1\n")
        fh.write(f"bins: {args.bins}\n")
        fh.write(f"d0_range: {tuple(args.d0_range)}\n")
        fh.write(f"lognw_range: {tuple(args.lognw_range)}\n")
        fh.write(f"matched_frequency_percentile: {args.frequency_percentile:g}\n")
        for field, vmax in row_vmax_values.items():
            fh.write(f"{field}_frequency_vmax_percent: {vmax:.10g}\n")
        fh.write("\n")
        fh.write("experiment,field,n_profiles,gridcell_samples,freezing_level_km,source\n")
        for experiment in experiments:
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            freeze = cfads[experiment].freezing_level_km
            for field in DSD_FIELDS:
                grid = cfads[experiment].grids[field]
                fh.write(
                    f"{experiment},{label},{field},{grid.n_profiles},"
                    f"{int(np.sum(grid.hist))},{freeze:.10e},{grid.source}\n"
                )


def build_pairs(args: argparse.Namespace) -> tuple[
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    full_cfads, strong_cfads, lead_tag, settings = build_pairs(args)
    row_vmax_values = matched_vmax_by_field(
        full_cfads,
        strong_cfads,
        args.experiments,
        float(args.frequency_percentile),
    )
    print(
        "Matched frequency vmax (%): "
        + ", ".join(f"{field}={value:g}" for field, value in row_vmax_values.items()),
        flush=True,
    )

    full_output = args.output_dir / FULL_FIGURE_NAME
    strong_output = args.output_dir / STRONG_FIGURE_NAME
    full_text = args.data_txt_dir / FULL_TEXT_NAME
    strong_text = args.data_txt_dir / STRONG_TEXT_NAME
    plot_cfads(
        full_output,
        full_cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
        row_vmax_values=row_vmax_values,
    )
    plot_cfads(
        strong_output,
        strong_cfads,
        args.experiments,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
        row_vmax_values=row_vmax_values,
    )
    write_data_txt(
        full_text,
        full_cfads,
        args.experiments,
        title="Grid-cell full-domain CFAD-style DSD diagnostic distributions",
        lead_tag=lead_tag,
        args=args,
        settings=settings,
        row_vmax_values=row_vmax_values,
    )
    write_data_txt(
        strong_text,
        strong_cfads,
        args.experiments,
        title="Grid-cell strong-convection CFAD-style DSD diagnostic distributions",
        lead_tag=lead_tag,
        args=args,
        settings=settings,
        row_vmax_values=row_vmax_values,
    )
    print(f"Saved full-domain figure:        {full_output}", flush=True)
    print(f"Saved strong-convection figure: {strong_output}", flush=True)
    print(f"Saved full-domain data:          {full_text}", flush=True)
    print(f"Saved strong-convection data:   {strong_text}", flush=True)


if __name__ == "__main__":
    main()
