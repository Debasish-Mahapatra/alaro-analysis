"""
Cache-only plotting workflow for normal vs RAD liquid/solid water pairs.

Outputs in a separate tree:
1) Individual variable panels (C1M absolute, G1M-C1M, G2M-G1M)
2) Pair comparison panels per experiment
   - normal absolute
   - RAD absolute
   - RAD - normal
   - normal - RAD

This script does not process raw netCDF files. It reads existing intermediate
.npz files produced by the main hydrometeor script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.analysis.pair_analysis import (
    compute_pair_comparison_scales,
    compute_variable_scales,
)
from alaro_analysis.common.cli_config import add_config_argument, parse_configured_args
from alaro_analysis.common.constants import EXPERIMENTS, SEASONS
from alaro_analysis.common.models import VerticalAxis
from alaro_analysis.common.naming import safe_name
from alaro_analysis.common.paths import runs_path
from alaro_analysis.common.seasons import build_period_specs, resolve_seasons
from alaro_analysis.common.vertical import compute_freezing_line_km
from alaro_analysis.data.cache import load_diurnal_mean, load_height_axis
from alaro_analysis.plotting.pair_analysis import (
    plot_individual_panels,
    plot_pair_across_experiments,
    plot_pair_panel,
    plot_sum_diff_grid,
)

try:
    import cmaps  # type: ignore
except Exception:
    cmaps = None

PAIR_PRESETS = {
    "liquid": ("LIQUID_WATER", "RAD_LIQUID_W"),
    "solid": ("SOLID_WATER", "RAD_SOLID_WA"),
}

DEFAULT_INTERMEDIATE_DIRS = (
    runs_path("processed-data/hydrometeors"),
    runs_path("processed-data"),
)
DEFAULT_OUTPUT_DIR = runs_path("figures/hydrometeor_pair_analysis")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair workflow plots for LIQUID/SOLID water vs RAD variables."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--intermediate-dirs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INTERMEDIATE_DIRS),
        help="One or more intermediate-cache roots to search (first match wins).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for this dedicated workflow.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=("liquid", "solid", "all"),
        default=("all",),
        help="Which pair presets to include.",
    )
    parser.add_argument(
        "--analysis-modes",
        nargs="+",
        default=("full", "seasonal"),
        choices=("full", "seasonal"),
        help="Run full 2-year analysis, seasonal analysis, or both.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=("individual", "pair", "sum_diff"),
        default=("individual", "pair", "sum_diff"),
        help="Which outputs to generate: individual panels, pair panels, sum-diff grids, or all.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(SEASONS.keys()),
        help="Season subset for seasonal mode; use 'all' for all seasons.",
    )
    parser.add_argument(
        "--temperature-variable",
        default="TEMPERATURE",
        help="Temperature cache variable for freezing-level overlay.",
    )
    parser.add_argument(
        "--skip-freezing-level",
        action="store_true",
        help="Disable freezing-level overlay.",
    )
    parser.add_argument(
        "--height-aggregate",
        choices=("first", "mean-all"),
        default="first",
        help="Geopotential height cache style to use.",
    )
    parser.add_argument(
        "--max-height-km",
        type=float,
        default=20.0,
        help="Upper y-limit when using height axis.",
    )
    parser.add_argument(
        "--absolute-low-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for pair-shared absolute log scale.",
    )
    parser.add_argument(
        "--absolute-high-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for pair-shared absolute log scale.",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=98.0,
        help="Percentile for symmetric anomaly scales.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Figure dpi (default 450).",
    )
    parser.add_argument(
        "--scale-reference",
        choices=("full_2yr", "all_periods"),
        default="full_2yr",
        help=(
            "Reference set for colorbar limits. "
            "'full_2yr' uses only full-period caches; 'all_periods' uses all selected periods."
        ),
    )
    parser.add_argument(
        "--cross-anomaly-mode",
        choices=("both", "normal-minus-rad", "rad-minus-normal"),
        default="both",
        help=("Direction(s) for pair anomalies. 'normal-minus-rad' means NORMAL - RAD only."),
    )
    parser.add_argument(
        "--pair-layout",
        choices=("panel", "single", "experiments"),
        default="panel",
        help=(
            "Layout for pair output. 'panel' keeps multi-subplot figure, "
            "'single' saves one selected cross-anomaly map per experiment, "
            "'experiments' saves one figure with C1M/G1M/G2M side-by-side."
        ),
    )
    return parse_configured_args(parser, "pair_analysis", argv=argv)


def resolve_selected_pairs(pair_args: tuple[str, ...] | list[str]) -> list[tuple[str, str]]:
    tokens = [p.strip().lower() for p in pair_args if p.strip()]
    if not tokens or "all" in tokens:
        return [PAIR_PRESETS["liquid"], PAIR_PRESETS["solid"]]
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        pair = PAIR_PRESETS[token]
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)
    return selected


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.absolute_low_percentile < 100.0):
        raise ValueError("--absolute-low-percentile must be in [0, 100).")
    if not (0.0 < args.absolute_high_percentile <= 100.0):
        raise ValueError("--absolute-high-percentile must be in (0, 100].")
    if args.absolute_low_percentile >= args.absolute_high_percentile:
        raise ValueError("--absolute-low-percentile must be lower than --absolute-high-percentile.")
    if not (0.0 < args.anomaly_percentile <= 100.0):
        raise ValueError("--anomaly-percentile must be in (0, 100].")
    if args.pair_layout in ("single", "experiments") and args.cross_anomaly_mode == "both":
        raise ValueError(
            "--pair-layout single|experiments requires "
            "--cross-anomaly-mode normal-minus-rad or rad-minus-normal."
        )

    intermediate_roots: list[Path] = []
    for root in args.intermediate_dirs:
        resolved = root.resolve()
        if resolved not in intermediate_roots:
            intermediate_roots.append(resolved)
    for root in intermediate_roots:
        if not root.exists():
            print(f"[warn] Intermediate root not found: {root}")
    if not any(root.exists() for root in intermediate_roots):
        raise FileNotFoundError("None of the intermediate roots exist.")

    mode_set = set(args.analysis_modes)
    periods = build_period_specs(mode_set, resolve_seasons(args.seasons))
    if not periods:
        raise RuntimeError("No periods selected.")

    selected_pairs = resolve_selected_pairs(args.pairs)
    variables = sorted({v for pair in selected_pairs for v in pair})
    period_map = {p.key: p for p in periods}
    if args.scale_reference == "full_2yr" and "full_2yr" in period_map:
        scale_periods = [period_map["full_2yr"]]
    else:
        scale_periods = periods

    print("Intermediate roots (search order):")
    for root in intermediate_roots:
        print(f"- {root}")
    print("Selected pairs:", selected_pairs)
    print("Outputs:", list(args.outputs))
    print("Pair layout:", args.pair_layout)
    print("Periods:", [p.key for p in periods])
    print("Scale reference periods:", [p.key for p in scale_periods])
    print(
        f"Scales: abs p{args.absolute_low_percentile:g}-p{args.absolute_high_percentile:g}, "
        f"anomaly p{args.anomaly_percentile:g}"
    )
    print("Freezing-level overlay:", not args.skip_freezing_level)

    profiles: dict[tuple[str, str, str], np.ndarray] = {}
    for period in periods:
        for variable in variables:
            for exp in EXPERIMENTS:
                profiles[(period.key, variable, exp)] = load_diurnal_mean(
                    intermediate_roots, variable, period.output_subdir, exp
                )

    axes_by_period: dict[str, VerticalAxis] = {}
    for period in periods:
        fallback_levels = profiles[(period.key, variables[0], "control")].shape[0]
        axes_by_period[period.key] = load_height_axis(
            intermediate_roots=intermediate_roots,
            period_subdir=period.output_subdir,
            experiment="control",
            aggregate=args.height_aggregate,
            fallback_levels=fallback_levels,
            allow_geopotentiel_fallback=True,
        )

    freezing_by_period: dict[str, np.ndarray | None] = {p.key: None for p in periods}
    if not args.skip_freezing_level:
        missing_temp = False
        temp_profiles: dict[tuple[str, str], np.ndarray] = {}
        for period in periods:
            for exp in EXPERIMENTS:
                try:
                    temp_profiles[(period.key, exp)] = load_diurnal_mean(
                        intermediate_roots,
                        args.temperature_variable,
                        period.output_subdir,
                        exp,
                    )
                except FileNotFoundError:
                    missing_temp = True
                    break
            if missing_temp:
                break
        if missing_temp:
            print(
                f"[warn] Temperature cache '{args.temperature_variable}' missing in at least one period/experiment. "
                "Freezing-level overlay disabled."
            )
        else:
            for period in periods:
                freezing_by_period[period.key] = compute_freezing_line_km(
                    axes_by_period[period.key],
                    [temp_profiles[(period.key, exp)] for exp in EXPERIMENTS],
                )

    variable_scales: dict[str, tuple[tuple[float, float], float]] = {}
    for variable in variables:
        variable_scales[variable] = compute_variable_scales(
            variable=variable,
            scale_periods=scale_periods,
            profiles=profiles,
            absolute_low_percentile=args.absolute_low_percentile,
            absolute_high_percentile=args.absolute_high_percentile,
            anomaly_percentile=args.anomaly_percentile,
        )
        abs_limits_var, model_anom_scale_var = variable_scales[variable]
        print(
            f"{safe_name(variable)}: abs=[{abs_limits_var[0]:.3e}, {abs_limits_var[1]:.3e}], "
            f"model_anom=±{model_anom_scale_var:.3e}"
        )

    pair_scales: dict[str, tuple[tuple[float, float], float]] = {}
    for pair in selected_pairs:
        pair_id = f"{safe_name(pair[0])}_vs_{safe_name(pair[1])}"
        pair_scales[pair_id] = compute_pair_comparison_scales(
            pair=pair,
            scale_periods=scale_periods,
            profiles=profiles,
            absolute_low_percentile=args.absolute_low_percentile,
            absolute_high_percentile=args.absolute_high_percentile,
            anomaly_percentile=args.anomaly_percentile,
        )
        abs_limits, cross_anom_scale = pair_scales[pair_id]
        print(
            f"{pair_id}: shared_abs=[{abs_limits[0]:.3e}, {abs_limits[1]:.3e}], "
            f"cross_anom=±{cross_anom_scale:.3e}"
        )

    output_root = args.output_dir.resolve()
    saved = 0
    output_set = set(args.outputs)

    for pair in selected_pairs:
        normal_var, rad_var = pair
        pair_id = f"{safe_name(normal_var)}_vs_{safe_name(rad_var)}"
        pair_abs_limits, cross_anom_scale = pair_scales[pair_id]

        if "individual" in output_set:
            for variable in (normal_var, rad_var):
                var_abs_limits, var_anom_scale = variable_scales[variable]
                for period in periods:
                    output_file = (
                        output_root
                        / "individual"
                        / safe_name(variable)
                        / period.output_subdir
                        / f"{safe_name(variable)}_panel_c1m_g1m-c1m_g2m-g1m.png"
                    )
                    plot_individual_panels(
                        variable=variable,
                        control=profiles[(period.key, variable, "control")],
                        graupel=profiles[(period.key, variable, "graupel")],
                        twomom=profiles[(period.key, variable, "2mom")],
                        axis=axes_by_period[period.key],
                        max_height_km=args.max_height_km,
                        period_label=period.label,
                        freezing_line_km=freezing_by_period[period.key],
                        abs_limits=var_abs_limits,
                        anom_scale=var_anom_scale,
                        output_file=output_file,
                        dpi=args.dpi,
                    )
                    saved += 1

        if "pair" in output_set:
            if args.cross_anomaly_mode == "both":
                suffix = "panel_abs_rad-minus-normal_normal-minus-rad"
            elif args.cross_anomaly_mode == "normal-minus-rad":
                suffix = "normal-minus-rad"
            else:
                suffix = "rad-minus-normal"
            if args.pair_layout == "panel":
                suffix = f"panel_abs_{suffix}" if "panel_abs" not in suffix else suffix
            elif args.pair_layout == "single":
                suffix = f"single_{suffix}"
                for period in periods:
                    for exp in EXPERIMENTS:
                        output_file = (
                            output_root
                            / "pair_comparison"
                            / pair_id
                            / exp
                            / period.output_subdir
                            / f"{pair_id}_{exp}_{suffix}.png"
                        )
                        plot_pair_panel(
                            pair=pair,
                            experiment=exp,
                            normal=profiles[(period.key, normal_var, exp)],
                            rad=profiles[(period.key, rad_var, exp)],
                            axis=axes_by_period[period.key],
                            max_height_km=args.max_height_km,
                            period_label=period.label,
                            freezing_line_km=freezing_by_period[period.key],
                            abs_limits=pair_abs_limits,
                            cross_anom_scale=cross_anom_scale,
                            cross_anomaly_mode=args.cross_anomaly_mode,
                            pair_layout=args.pair_layout,
                            output_file=output_file,
                            dpi=args.dpi,
                        )
                        saved += 1
            else:
                suffix = f"experiments_{suffix}"
                for period in periods:
                    output_file = (
                        output_root
                        / "pair_comparison"
                        / pair_id
                        / period.output_subdir
                        / f"{pair_id}_{suffix}.png"
                    )
                    per_exp = {
                        exp: (
                            profiles[(period.key, normal_var, exp)],
                            profiles[(period.key, rad_var, exp)],
                        )
                        for exp in EXPERIMENTS
                    }
                    plot_pair_across_experiments(
                        pair=pair,
                        period_label=period.label,
                        axis=axes_by_period[period.key],
                        max_height_km=args.max_height_km,
                        freezing_line_km=freezing_by_period[period.key],
                        cross_anom_scale=cross_anom_scale,
                        cross_anomaly_mode=args.cross_anomaly_mode,
                        per_experiment_arrays=per_exp,
                        output_file=output_file,
                        dpi=args.dpi,
                    )
                    saved += 1

        if "sum_diff" in output_set:
            for period in periods:
                output_file = (
                    output_root
                    / "sum_diff"
                    / pair_id
                    / period.output_subdir
                    / f"{pair_id}_sum_diff_grid.png"
                )
                plot_sum_diff_grid(
                    pair=pair,
                    profiles=profiles,
                    period=period,
                    axis=axes_by_period[period.key],
                    max_height_km=args.max_height_km,
                    freezing_line_km=freezing_by_period[period.key],
                    output_file=output_file,
                    dpi=args.dpi,
                )
                saved += 1

    print(f"\nCompleted. Total figures generated: {saved}")


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
