"""
Compare ALARO radiation terms against SURFEX net radiation.

The primary check is:
    ALARO_RN = SURFFLU.RAY.SOLA + SURFFLU.RAY.THER

and whether that roughly matches SURFEX SFX.RN over the masked domain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.analysis.radiation_compare import (
    collect_alaro_day_steps,
    compute_alaro_lines,
    compute_single_variable_line,
)
from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.radiation_compare_settings import (
    ALARO_RADIATION_CACHE_VERSION,
)
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.spatial import build_spatial_window, spatial_window_tag
from alaro_analysis.data.cache import build_cache_file
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.data.discovery import (
    collect_file_records,
    discover_variable_maps,
    resolve_var_name,
)
from alaro_analysis.plotting.radiation_compare import (
    plot_alaro_components,
    plot_net_comparison,
)

DEFAULT_ALARO_CONTROL_DIR = runs_path("control/masked-netcdf")
DEFAULT_ALARO_GRAUPEL_DIR = runs_path("graupel/masked-netcdf")
DEFAULT_ALARO_2MOM_DIR = runs_path("2mom/masked-netcdf")

DEFAULT_SURFEX_CONTROL_DIR = runs_path("SURFEX/control/masked-netcdf")
DEFAULT_SURFEX_GRAUPEL_DIR = runs_path("SURFEX/graupel/masked-netcdf")
DEFAULT_SURFEX_2MOM_DIR = runs_path("SURFEX/2mom/masked-netcdf")

DEFAULT_OUTPUT_DIR = runs_path("figures/radiation_compare")
DEFAULT_INTERMEDIATE_DIR = runs_path("processed-data/radiation_compare")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare de-accumulated ALARO radiation-derived net radiation against SURFEX SFX.RN."
    )
    add_config_argument(parser)
    parser.add_argument("--alaro-control-dir", type=Path, default=DEFAULT_ALARO_CONTROL_DIR)
    parser.add_argument("--alaro-graupel-dir", type=Path, default=DEFAULT_ALARO_GRAUPEL_DIR)
    parser.add_argument("--alaro-twomom-dir", type=Path, default=DEFAULT_ALARO_2MOM_DIR)

    parser.add_argument("--surfex-control-dir", type=Path, default=DEFAULT_SURFEX_CONTROL_DIR)
    parser.add_argument("--surfex-graupel-dir", type=Path, default=DEFAULT_SURFEX_GRAUPEL_DIR)
    parser.add_argument("--surfex-twomom-dir", type=Path, default=DEFAULT_SURFEX_2MOM_DIR)

    parser.add_argument("--shortwave-down-var", default="SURFRF.SHORT.DO")
    parser.add_argument("--longwave-down-var", default="SURFRF.LONG.DO")
    parser.add_argument("--solar-net-var", default="SURFFLU.RAY.SOLA")
    parser.add_argument("--thermal-net-var", default="SURFFLU.RAY.THER")
    parser.add_argument("--surfex-net-var", default="SFX.RN")

    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Subset of seasons (wet dry ...). Use 'all' for all seasons.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--overwrite-intermediate",
        action="store_true",
        help="Overwrite existing intermediate cache files.",
    )
    parser.add_argument(
        "--recompute",
        dest="overwrite_intermediate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=-4,
        help="Local timezone offset from UTC (Amazon is -4).",
    )
    parser.add_argument(
        "--y-slice",
        default=None,
        help="Optional Python-style Y slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--x-slice",
        default=None,
        help="Optional Python-style X slice start:end for spatial averaging.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List discovered variables per experiment/root and exit.",
    )
    return parse_configured_args(parser, "radiation_compare", argv=argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spatial_window = build_spatial_window(args.y_slice, args.x_slice)
    spatial_tag = spatial_window_tag(spatial_window)

    alaro_dirs = {
        "control": args.alaro_control_dir.resolve(),
        "graupel": args.alaro_graupel_dir.resolve(),
        "2mom": args.alaro_twomom_dir.resolve(),
    }
    surfex_dirs = {
        "control": args.surfex_control_dir.resolve(),
        "graupel": args.surfex_graupel_dir.resolve(),
        "2mom": args.surfex_twomom_dir.resolve(),
    }
    for label, roots in (("ALARO", alaro_dirs), ("SURFEX", surfex_dirs)):
        for exp, d in roots.items():
            if not d.exists():
                raise FileNotFoundError(f"{label} {exp} data dir not found: {d}")

    alaro_variable_maps = discover_variable_maps(alaro_dirs)
    surfex_variable_maps = discover_variable_maps(surfex_dirs)

    if args.list_variables:
        for label, variable_maps in (
            ("ALARO", alaro_variable_maps),
            ("SURFEX", surfex_variable_maps),
        ):
            print(f"\n{label} variables:", flush=True)
            for exp in EXPERIMENTS:
                names = sorted(variable_maps[exp].values())
                print(f"- {exp}: {', '.join(names) if names else '(none)'}", flush=True)
        return

    selected_seasons = resolve_seasons(args.seasons)
    periods = build_period_specs(set(args.analysis_modes), selected_seasons)
    if not periods:
        raise RuntimeError("No analysis periods selected.")

    resolved_alaro = {
        exp: {
            "SW_DOWN": resolve_var_name(alaro_variable_maps, exp, (args.shortwave_down_var,)),
            "LW_DOWN": resolve_var_name(alaro_variable_maps, exp, (args.longwave_down_var,)),
            "SW_NET": resolve_var_name(alaro_variable_maps, exp, (args.solar_net_var,)),
            "LW_NET": resolve_var_name(alaro_variable_maps, exp, (args.thermal_net_var,)),
        }
        for exp in EXPERIMENTS
    }
    resolved_surfex = {
        exp: resolve_var_name(surfex_variable_maps, exp, (args.surfex_net_var,))
        for exp in EXPERIMENTS
    }

    print("\nALARO directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {alaro_dirs[exp]}", flush=True)
    print("\nSURFEX directories:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {surfex_dirs[exp]}", flush=True)
    source_roots = [
        alaro_dirs[exp] / name
        for exp in EXPERIMENTS
        for name in resolved_alaro[exp].values()
        if name
    ]
    source_roots += [
        surfex_dirs[exp] / resolved_surfex[exp] for exp in EXPERIMENTS if resolved_surfex[exp]
    ]
    cache = CacheContext.from_netcdf("radiation_compare", vars(args), source_roots)

    print("\nResolved ALARO variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved_alaro[exp]}", flush=True)
    print("\nResolved SURFEX variable names:", flush=True)
    for exp in EXPERIMENTS:
        print(f"- {exp}: {resolved_surfex[exp]}", flush=True)
    print("\nOutput directory:", args.output_dir.resolve(), flush=True)
    print("Intermediate directory:", args.intermediate_dir.resolve(), flush=True)
    print("Periods:", [p.key for p in periods], flush=True)
    print("Spatial averaging tag:", spatial_tag, flush=True)
    print(
        "Spatial Y slice:",
        f"{spatial_window.y_start}:{spatial_window.y_end}",
        "| X slice:",
        f"{spatial_window.x_start}:{spatial_window.x_end}",
        flush=True,
    )
    print(
        "ALARO radiation terms are treated as accumulated forecast-step totals and de-accumulated to W m-2; "
        "+0024 is required and used.",
        flush=True,
    )
    print(
        "SURFEX SFX.RN remains a direct masked spatial mean; SURFEX +0024 is still ignored by design.",
        flush=True,
    )

    output_dir = args.output_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    required_alaro_keys = ("SW_DOWN", "LW_DOWN", "SW_NET", "LW_NET", "SW_UP", "LW_UP", "ALARO_RN")

    for period in periods:
        print(f"\n===== Comparing radiation for {period.label} ({period.key}) =====", flush=True)
        alaro_payloads: dict[str, dict[str, np.ndarray]] = {}
        surfex_payloads: dict[str, dict[str, np.ndarray]] = {}

        for exp in EXPERIMENTS:
            if resolved_surfex[exp] is None:
                print(
                    f"[warn] {period.key}/{exp}: SURFEX variable '{args.surfex_net_var}' not found.",
                    flush=True,
                )
                continue
            if resolved_alaro[exp]["SW_NET"] is None or resolved_alaro[exp]["LW_NET"] is None:
                print(
                    f"[warn] {period.key}/{exp}: missing required ALARO variables "
                    f"{args.solar_net_var}/{args.thermal_net_var}.",
                    flush=True,
                )
                continue

            alaro_ref_dir = alaro_dirs[exp] / resolved_alaro[exp]["SW_NET"]
            surfex_ref_dir = surfex_dirs[exp] / resolved_surfex[exp]

            alaro_day_steps = collect_alaro_day_steps(
                variable_dir=alaro_ref_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
            )
            surfex_records = collect_file_records(
                variable_dir=surfex_ref_dir,
                max_days=args.max_days,
                allowed_months=period.allowed_months,
                utc_offset_hours=args.utc_offset_hours,
            )
            if not alaro_day_steps:
                print(
                    f"[warn] {period.key}/{exp}: no ALARO day sequences found in {alaro_ref_dir}",
                    flush=True,
                )
                continue
            if not surfex_records:
                print(
                    f"[warn] {period.key}/{exp}: no SURFEX records found in {surfex_ref_dir}",
                    flush=True,
                )
                continue

            cache_file = build_cache_file(
                intermediate_dir=intermediate_dir,
                analysis_name=f"radiation_compare_v{ALARO_RADIATION_CACHE_VERSION}",
                period_subdir=period.output_subdir,
                experiment=exp,
                spatial_tag=spatial_tag,
            )
            use_cache = cache.valid(cache_file) and not args.overwrite_intermediate
            if use_cache:
                payload = cache.load(cache_file)
                if all(key in payload for key in required_alaro_keys) and "SURFEX_RN" in payload:
                    alaro_payloads[exp] = {
                        key: np.asarray(payload[key], dtype=np.float64)
                        for key in required_alaro_keys
                    }
                    surfex_payloads[exp] = {
                        "SURFEX_RN": np.asarray(payload["SURFEX_RN"], dtype=np.float64)
                    }
                    continue
                print(
                    f"[{period.key}/{exp}] cache missing new radiation keys; recomputing.",
                    flush=True,
                )

            if exp not in alaro_payloads or exp not in surfex_payloads:
                alaro_payload = compute_alaro_lines(
                    experiment=exp,
                    experiment_dir=alaro_dirs[exp],
                    day_steps=alaro_day_steps,
                    names=resolved_alaro[exp],  # type: ignore[arg-type]
                    spatial_window=spatial_window,
                    utc_offset_hours=args.utc_offset_hours,
                )
                surfex_payload = compute_single_variable_line(
                    experiment=exp,
                    variable_dir=surfex_ref_dir,
                    variable_name=resolved_surfex[exp],  # type: ignore[arg-type]
                    records=surfex_records,
                    spatial_window=spatial_window,
                )
                alaro_payloads[exp] = alaro_payload
                surfex_payloads[exp] = surfex_payload
                if args.max_days is None:
                    cache.save(
                        cache_file,
                        {
                            "cache_version": np.array(
                                [ALARO_RADIATION_CACHE_VERSION], dtype=np.int64
                            ),
                            **{key: value for key, value in alaro_payload.items()},
                            "SURFEX_RN": surfex_payload["SURFEX_RN"],
                        },
                    )

        if not all(exp in alaro_payloads and exp in surfex_payloads for exp in EXPERIMENTS):
            print(f"[warn] {period.key}: incomplete experiment set; skipping plots.", flush=True)
            continue

        suffix = f"_{spatial_tag}" if spatial_tag != "full-domain" else ""
        net_out = (
            output_dir
            / "net_radiation_compare"
            / period.output_subdir
            / f"net_radiation_compare{suffix}.png"
        )
        comp_out = (
            output_dir
            / "alaro_components"
            / period.output_subdir
            / f"alaro_radiation_components{suffix}.png"
        )

        plot_net_comparison(
            period_label=period.label,
            output_file=net_out,
            alaro_lines=alaro_payloads,
            surfex_lines=surfex_payloads,
            utc_offset_hours=args.utc_offset_hours,
        )
        plot_alaro_components(
            period_label=period.label,
            output_file=comp_out,
            alaro_lines=alaro_payloads,
            utc_offset_hours=args.utc_offset_hours,
        )

    print("\nCompleted ALARO/SURFEX radiation comparison.", flush=True)


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
