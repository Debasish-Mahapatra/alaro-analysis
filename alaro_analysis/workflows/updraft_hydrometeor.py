"""
Joint updraft-intensity vs height distributions of hydrometeor mixing ratios.

Inspired by Figure 9 of Van Weverberg et al. (2024, QJRMS):
  - x-axis: updraft vertical velocity w (m/s)
  - y-axis: height (km)
  - colour shading: mean hydrometeor mixing ratio in each (w, height) bin
  - contours: absolute frequency (number of grid-point samples)
  - marginal panels: vertical profile (right) and velocity distribution (bottom)

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.updraft_hydrometeor
"""

from __future__ import annotations

from pathlib import Path

import alaro_analysis.common.updraft_hydrometeor_settings as science_settings
from alaro_analysis.analysis.updraft_hydrometeor import (
    accumulate_histograms,
    load_cache,
    save_cache,
)
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.updraft_hydrometeor_settings import (
    CACHE_DIR,
    DATA_ROOT,
    EXPERIMENTS,
    HYDROMETEORS,
    MAX_DAYS,
    OUTPUT_DIR,
    STRATIFY_CONFIGS,
)
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.plotting.updraft_hydrometeor import (
    plot_binned_means,
    plot_figure,
    plot_regression_slopes,
)


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Updraft-hydrometeor joint distributions")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument("--hydrometeors", nargs="+", default=HYDROMETEORS)
    parser.add_argument("--max-days", type=int, default=MAX_DAYS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--stratify",
        nargs="+",
        default=list(STRATIFY_CONFIGS.keys()),
        choices=list(STRATIFY_CONFIGS.keys()),
        help="Stratification variable(s): flux, extent, intensity",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parse_configured_args(parser, "updraft_hydrometeor", argv=argv)
    contexts = {}
    for exp in args.experiments:
        settings = {
            **vars(args),
            **{
                name: getattr(science_settings, name)
                for name in [
                    "HYDROMETEORS",
                    "H_BINS",
                    "STRATIFY_CONFIGS",
                    "FLUX_BINS",
                    "MAX_DAYS",
                    "MIN_LEAD_HOUR",
                    "G",
                    "HYDRO_SHORT",
                ]
            },
        }
        contexts[exp] = CacheContext.from_netcdf(
            "updraft_hydrometeor", settings, [args.data_root / exp / "masked-netcdf"]
        )

    stratifications = args.stratify

    for strat in stratifications:
        print(f"\n{'=' * 60}")
        print(f"  Stratification: {strat}")
        print(f"{'=' * 60}")

        results = {}
        for exp in args.experiments:
            print(f"Processing {exp} [{strat}]...")
            cached = (
                None
                if args.no_cache
                else load_cache(
                    exp, stratify=strat, cache_dir=args.cache_dir, context=contexts[exp]
                )
            )
            if cached is not None:
                print(f"  Loaded from cache")
                results[exp] = cached
            else:
                r = accumulate_histograms(
                    exp,
                    args.hydrometeors,
                    max_days=args.max_days,
                    stratify=strat,
                    workers=args.workers,
                    data_root=args.data_root,
                )
                save_cache(exp, r, stratify=strat, cache_dir=args.cache_dir, context=contexts[exp])
                results[exp] = r

        outpath = (
            Path(args.output)
            if args.output
            else args.output_dir / f"updraft_hydrometeor_{strat}.png"
        )

        # Joint distribution heatmap (one per stratification).
        plot_figure(results, args.hydrometeors, args.experiments, outpath, stratify=strat)

        # Analysis 1: Binned means
        binned_path = args.output_dir / f"updraft_hydrometeor_{strat}_binned_means.png"
        plot_binned_means(results, args.hydrometeors, args.experiments, binned_path, stratify=strat)

        # Analysis 2: Regression slopes
        regr_path = args.output_dir / f"updraft_hydrometeor_{strat}_regression_slopes.png"
        plot_regression_slopes(
            results, args.hydrometeors, args.experiments, regr_path, stratify=strat
        )


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
