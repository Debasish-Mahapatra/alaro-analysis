"""
Convective feedback diagnostics: loading, downdrafts, and moisture recycling.

Three hypotheses for how microphysics feeds back on convective dynamics:

  H1 — Condensate loading:  heavy hydrometeors weigh down updrafts.
        Test: updraft intensity conditioned on total condensate mass.

  H2 — Downdraft feedback:  graupel evaporation drives stronger downdrafts
        that suppress subsequent convection.
        Test: downdraft flux / intensity / extent comparison across schemes.

  H3 — Cumulative moisture misplacement:  graupel evaporates above the BL,
        depositing moisture that doesn't recycle, leaving the next day drier.
        Test: pre-convection (06–09 LT) thermodynamic profiles.

Usage:
    source /mnt/HDS_CLIMATE/CLIMATE/deba/miniconda3/etc/profile.d/conda.sh
    conda activate epygram
    python -m alaro_analysis.workflows.convective_feedback
"""

from __future__ import annotations

from pathlib import Path

import alaro_analysis.common.convective_feedback_settings as science_settings
from alaro_analysis.analysis.convective_feedback import (
    accumulate_h1,
    accumulate_h2,
    accumulate_h3,
    accumulate_h4,
    load_h1,
    load_h2,
    load_h3,
    load_h4,
    save_h1,
    save_h2,
    save_h3,
    save_h4,
)
from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.convective_feedback_settings import (
    CACHE_DIR,
    DATA_ROOT,
    EXPERIMENTS,
    OUTPUT_DIR,
)
from alaro_analysis.data.cache_context import CacheContext
from alaro_analysis.plotting.convective_feedback import (
    plot_h1,
    plot_h2,
    plot_h3,
    plot_h4,
)


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Convective feedback diagnostics")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    parser.add_argument(
        "--hypotheses",
        nargs="+",
        default=["h1", "h2", "h3", "h4"],
        choices=["h1", "h2", "h3", "h4"],
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parse_configured_args(parser, "convective_feedback", argv=argv)
    contexts = {}
    for exp in args.experiments:
        settings = {
            **vars(args),
            **{
                name: getattr(science_settings, name)
                for name in [
                    "HYDROMETEORS",
                    "G",
                    "MIN_LEAD_HOUR",
                    "LOADING_BINS",
                    "H_BINS",
                    "DD_H_BINS",
                    "THERMO_VARS",
                    "PRECONV_LEADS",
                    "H4_H_BINS",
                ]
            },
        }
        contexts[exp] = CacheContext.from_netcdf(
            "convective_feedback", settings, [args.data_root / exp / "masked-netcdf"]
        )

    if "h1" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H1: Condensate loading")
        print("=" * 60)
        h1_results = {}
        for exp in args.experiments:
            cached = (
                None
                if args.no_cache
                else load_h1(exp, cache_dir=args.cache_dir, context=contexts[exp])
            )
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h1_results[exp] = cached
            else:
                r = accumulate_h1(
                    exp, max_days=args.max_days, workers=args.workers, data_root=args.data_root
                )
                save_h1(exp, r, cache_dir=args.cache_dir, context=contexts[exp])
                h1_results[exp] = r
        plot_h1(
            h1_results, args.experiments, args.output_dir / "convective_feedback_h1_loading.png"
        )

    if "h2" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H2: Downdraft feedback")
        print("=" * 60)
        h2_results = {}
        for exp in args.experiments:
            cached = (
                None
                if args.no_cache
                else load_h2(exp, cache_dir=args.cache_dir, context=contexts[exp])
            )
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h2_results[exp] = cached
            else:
                r = accumulate_h2(
                    exp, max_days=args.max_days, workers=args.workers, data_root=args.data_root
                )
                save_h2(exp, r, cache_dir=args.cache_dir, context=contexts[exp])
                h2_results[exp] = r
        plot_h2(
            h2_results, args.experiments, args.output_dir / "convective_feedback_h2_downdrafts.png"
        )

    if "h3" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H3: Pre-convection moisture")
        print("=" * 60)
        h3_results = {}
        for exp in args.experiments:
            cached = (
                None
                if args.no_cache
                else load_h3(exp, cache_dir=args.cache_dir, context=contexts[exp])
            )
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h3_results[exp] = cached
            else:
                r = accumulate_h3(
                    exp, max_days=args.max_days, workers=args.workers, data_root=args.data_root
                )
                save_h3(exp, r, cache_dir=args.cache_dir, context=contexts[exp])
                h3_results[exp] = r
        plot_h3(
            h3_results, args.experiments, args.output_dir / "convective_feedback_h3_moisture.png"
        )

    if "h4" in args.hypotheses:
        print("\n" + "=" * 60)
        print("  H4: Moisture flux divergence")
        print("=" * 60)
        h4_results = {}
        for exp in args.experiments:
            cached = (
                None
                if args.no_cache
                else load_h4(exp, cache_dir=args.cache_dir, context=contexts[exp])
            )
            if cached is not None:
                print(f"  {exp}: loaded from cache")
                h4_results[exp] = cached
            else:
                r = accumulate_h4(
                    exp, max_days=args.max_days, workers=args.workers, data_root=args.data_root
                )
                save_h4(exp, r, cache_dir=args.cache_dir, context=contexts[exp])
                h4_results[exp] = r
        plot_h4(
            h4_results,
            args.experiments,
            args.output_dir / "convective_feedback_h4_moist_divergence.png",
        )


def __getattr__(name):
    """Compatibility for moved functions; new callers use the owning module."""
    from alaro_analysis.common.legacy_exports import resolve_export

    return resolve_export(__name__, name)


if __name__ == "__main__":
    main()
