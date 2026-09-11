#!/usr/bin/env python3
"""Paper1 hydrometeor profile sampled only inside strong updrafts.

This is configured as the rain-only version by default.  An explicit ProfileSpec selects rain or graupel with the same
strong-updraft and isotherm logic.  The independent hourly files +0000 through
+0023 are used by default; +0024 is excluded because it duplicates the next
day's +0000 valid time for instantaneous fields.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.constants import EXPERIMENT_LABELS, EXPERIMENTS
from alaro_analysis.common.paths import data_path, runs_path
from alaro_analysis.common.strong_profile_settings import (
    DATA_ROOT,
    H_BINS_KM,
    ISOTHERM_STYLE,
    ISOTHERMS_C,
    PAPER_ROOT,
    STEP_RE,
    TEMP_VAR,
)
from alaro_analysis.common.study_specs import RAIN_PROFILE, ProfileSpec
from alaro_analysis.data.cache import save_npz_compressed
from alaro_analysis.data.strong_updraft_profiles import (
    RainProfile,
    _empty_accumulators,
    build_profile,
    cache_path,
    combine_day_results,
    get_profile,
    isotherm_height_km,
    list_days,
    list_steps,
    load_cache,
    mean_from_sum_count,
    process_day,
    read_field,
    save_cache,
    slug_float,
)
from alaro_analysis.plotting.strong_updraft_profiles import plot_profiles, write_text_data


def default_output_dir(*, spec: ProfileSpec = RAIN_PROFILE) -> Path:
    return PAPER_ROOT / f"{spec.folder_prefix}_strong_updraft_{spec.slug}_profile"


def default_cache_dir(*, spec: ProfileSpec = RAIN_PROFILE) -> Path:
    return runs_path("processed-data") / f"paper1_strong_updraft_{spec.slug}_profile"


def figure_name(*, spec: ProfileSpec = RAIN_PROFILE) -> str:
    return f"strong_updraft_{spec.slug}_vertical_profiles_450dpi.png"


def text_name(*, spec: ProfileSpec = RAIN_PROFILE) -> str:
    return f"strong_updraft_{spec.slug}_vertical_profiles_data.txt"


def parse_args(
    argv: list[str] | None = None, *, spec: ProfileSpec = RAIN_PROFILE
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build {spec.slug}-only strong-updraft profiles with isotherm overlays."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(spec=spec))
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir(spec=spec))
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS)
    )
    parser.add_argument("--min-lead-hour", type=int, default=0)
    parser.add_argument(
        "--max-lead-hour",
        type=int,
        default=23,
        help="Maximum forecast lead hour to include. Default 23 avoids duplicate +0024.",
    )
    parser.add_argument("--min-updraft-pa-s", type=float, default=10.0)
    parser.add_argument("--min-updraft-mesh-frac", type=float, default=0.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-height-km", type=float, default=20.0)
    parser.add_argument(
        "--xmin-kg-kg", "--rain-xmin-kg-kg", dest="xmin_kg_kg", type=float, default=spec.xmin_kg_kg
    )
    parser.add_argument(
        "--xmax-kg-kg", "--rain-xmax-kg-kg", dest="xmax_kg_kg", type=float, default=spec.xmax_kg_kg
    )
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--recompute", action="store_true")
    return parse_configured_args(parser, f"make_paper1_strong_updraft_{spec.slug}", argv=argv)


def main(argv: list[str] | None = None, *, spec: ProfileSpec = RAIN_PROFILE) -> None:
    args = parse_args(argv, spec=spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        experiment: get_profile(args, experiment, spec=spec) for experiment in args.experiments
    }
    fig_path = args.output_dir / figure_name(spec=spec)
    txt_path = args.output_dir / text_name(spec=spec)
    plot_profiles(
        profiles,
        args.experiments,
        fig_path,
        max_height_km=args.max_height_km,
        rain_xmin_kg_kg=args.xmin_kg_kg,
        rain_xmax_kg_kg=args.xmax_kg_kg,
        dpi=args.dpi,
        spec=spec,
    )
    write_text_data(profiles, args.experiments, txt_path, args=args, spec=spec)
    print(f"[done] figure: {fig_path}", flush=True)
    print(f"[done] data:   {txt_path}", flush=True)


if __name__ == "__main__":
    main()
