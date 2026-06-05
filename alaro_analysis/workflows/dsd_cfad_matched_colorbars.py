"""Render full-domain and strong-convection DSD CFADs with matched colorbars."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS
from alaro_analysis.workflows.disdrometer_dsd import MASK_FILE, NETCDF_ROOT
from alaro_analysis.workflows.dsd_cfad_profiles import (
    CACHE_DIR as FULL_CACHE_DIR,
    DATA_TXT_DIR as FULL_DATA_TXT_DIR,
    DSD_FIELDS,
    FIGURE_NAME as FULL_FIGURE_NAME,
    OUTPUT_DIR as FULL_OUTPUT_DIR,
    TEXT_NAME as FULL_TEXT_NAME,
    ExperimentCfad,
    build_cfads as build_full_cfads,
    frequency_percent,
    plot_cfads,
    row_vmax,
    write_data_txt as write_full_data_txt,
)
from alaro_analysis.workflows.dsd_cfad_strong_convection import (
    CACHE_DIR as STRONG_CACHE_DIR,
    DATA_TXT_DIR as STRONG_DATA_TXT_DIR,
    FIGURE_NAME as STRONG_FIGURE_NAME,
    OUTPUT_DIR as STRONG_OUTPUT_DIR,
    TEXT_NAME as STRONG_TEXT_NAME,
    build_cfads as build_strong_cfads,
    write_data_txt as write_strong_data_txt,
)


MATCHED_SUFFIX = "_matched_colorbar"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the full-domain and strong-updraft DSD CFAD plots with "
            "matched frequency colorbar limits."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--netcdf-root", type=Path, default=NETCDF_ROOT)
    parser.add_argument("--mask-file", type=Path, default=MASK_FILE)
    parser.add_argument("--mask-var", default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--full-cache-dir", type=Path, default=FULL_CACHE_DIR)
    parser.add_argument("--strong-cache-dir", type=Path, default=STRONG_CACHE_DIR)
    parser.add_argument("--full-output-dir", type=Path, default=FULL_OUTPUT_DIR)
    parser.add_argument("--strong-output-dir", type=Path, default=STRONG_OUTPUT_DIR)
    parser.add_argument("--full-data-txt-dir", type=Path, default=FULL_DATA_TXT_DIR)
    parser.add_argument("--strong-data-txt-dir", type=Path, default=STRONG_DATA_TXT_DIR)
    parser.add_argument("--lead", default="all")
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
    parser.add_argument("--n0-fixed", type=float, default=None)
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
        "--frequency-scale",
        choices=("row-robust-max", "exact-max"),
        default="row-robust-max",
        help=(
            "row-robust-max matches the maximum row-wise CFAD scale from the "
            "two plots; exact-max uses the literal maximum frequency percentage."
        ),
    )
    parser.add_argument("--frequency-percentile", type=float, default=99.0)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, "dsd_cfad_matched_colorbars", argv=argv)


def matched_name(name: str) -> str:
    path = Path(name)
    return f"{path.stem}{MATCHED_SUFFIX}{path.suffix}"


def namespace_for_full(args: argparse.Namespace) -> SimpleNamespace:
    n0_fixed = args.n0_fixed
    if n0_fixed is None:
        from alaro_analysis.common.dsd import MP_FIXED_N0_PER_M3_MM

        n0_fixed = MP_FIXED_N0_PER_M3_MM
    return SimpleNamespace(
        netcdf_root=args.netcdf_root,
        mask_file=args.mask_file,
        mask_var=args.mask_var,
        mask_threshold=args.mask_threshold,
        cache_dir=args.full_cache_dir,
        output_dir=args.full_output_dir,
        data_txt_dir=args.full_data_txt_dir,
        lead=args.lead,
        experiments=args.experiments,
        workers=args.workers,
        tasks_per_child=args.tasks_per_child,
        progress_every=args.progress_every,
        max_days=args.max_days,
        min_qr=args.min_qr,
        onemom_closure=args.onemom_closure,
        twomom_closure=args.twomom_closure,
        n0_fixed=n0_fixed,
        bins=args.bins,
        d0_range=args.d0_range,
        lognw_range=args.lognw_range,
        max_height_km=args.max_height_km,
        dpi=args.dpi,
        recompute=args.recompute,
    )


def namespace_for_strong(args: argparse.Namespace) -> SimpleNamespace:
    ns = namespace_for_full(args)
    ns.cache_dir = args.strong_cache_dir
    ns.output_dir = args.strong_output_dir
    ns.data_txt_dir = args.strong_data_txt_dir
    ns.convection_metric = args.convection_metric
    ns.min_updraft_pa_s = args.min_updraft_pa_s
    ns.min_updraft_flux = args.min_updraft_flux
    ns.min_updraft_mesh_frac = args.min_updraft_mesh_frac
    return ns


def exact_vmax(
    cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    field: str,
) -> float:
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    return max(1.0, float(np.nanmax(np.concatenate(values))))


def percentile_vmax(
    cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    field: str,
    percentile: float,
) -> float:
    if percentile == 99.0:
        return row_vmax(cfads, list(experiments), field)
    values: list[np.ndarray] = []
    for experiment in experiments:
        freq = frequency_percent(cfads[experiment].grids[field].hist)
        finite = freq[np.isfinite(freq) & (freq > 0.0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    return max(1.0, float(np.nanpercentile(np.concatenate(values), percentile)))


def matched_vmax_by_field(
    full_cfads: dict[str, ExperimentCfad],
    strong_cfads: dict[str, ExperimentCfad],
    experiments: Sequence[str],
    *,
    frequency_scale: str,
    frequency_percentile: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in DSD_FIELDS:
        if frequency_scale == "exact-max":
            full_vmax = exact_vmax(full_cfads, experiments, field)
            strong_vmax = exact_vmax(strong_cfads, experiments, field)
        else:
            full_vmax = percentile_vmax(
                full_cfads,
                experiments,
                field,
                frequency_percentile,
            )
            strong_vmax = percentile_vmax(
                strong_cfads,
                experiments,
                field,
                frequency_percentile,
            )
        out[field] = max(full_vmax, strong_vmax)
    return out


def append_colorbar_metadata(
    path: Path,
    *,
    frequency_scale: str,
    frequency_percentile: float,
    row_vmax_values: dict[str, float],
) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\nMatched colorbar settings\n")
        fh.write("-" * 72 + "\n")
        fh.write(f"frequency_scale: {frequency_scale}\n")
        if frequency_scale == "row-robust-max":
            fh.write(f"frequency_percentile: {frequency_percentile:g}\n")
        for field, value in row_vmax_values.items():
            fh.write(f"{field}_frequency_vmax_percent: {value:.10g}\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    full_args = namespace_for_full(args)
    strong_args = namespace_for_strong(args)

    full_cfads, full_lead_tag = build_full_cfads(full_args)
    strong_cfads, strong_lead_tag, strong_settings = build_strong_cfads(strong_args)
    row_vmax_values = matched_vmax_by_field(
        full_cfads,
        strong_cfads,
        args.experiments,
        frequency_scale=args.frequency_scale,
        frequency_percentile=float(args.frequency_percentile),
    )
    print(
        "Matched frequency vmax (%): "
        + ", ".join(f"{field}={value:g}" for field, value in row_vmax_values.items()),
        flush=True,
    )

    full_output = args.full_output_dir / matched_name(FULL_FIGURE_NAME)
    strong_output = args.strong_output_dir / matched_name(STRONG_FIGURE_NAME)
    full_text = args.full_data_txt_dir / matched_name(FULL_TEXT_NAME)
    strong_text = args.strong_data_txt_dir / matched_name(STRONG_TEXT_NAME)

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
    write_full_data_txt(full_text, full_cfads, args.experiments, lead_tag=full_lead_tag, args=full_args)
    append_colorbar_metadata(
        full_text,
        frequency_scale=args.frequency_scale,
        frequency_percentile=args.frequency_percentile,
        row_vmax_values=row_vmax_values,
    )
    write_strong_data_txt(
        strong_text,
        strong_cfads,
        args.experiments,
        lead_tag=strong_lead_tag,
        args=strong_args,
        settings=strong_settings,
    )
    append_colorbar_metadata(
        strong_text,
        frequency_scale=args.frequency_scale,
        frequency_percentile=args.frequency_percentile,
        row_vmax_values=row_vmax_values,
    )
    print(f"Saved full-domain figure:        {full_output}", flush=True)
    print(f"Saved strong-convection figure: {strong_output}", flush=True)
    print(f"Saved full-domain data:          {full_text}", flush=True)
    print(f"Saved strong-convection data:   {strong_text}", flush=True)


if __name__ == "__main__":
    main()
