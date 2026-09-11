"""Cfad exports. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.dsd_models import ExperimentCfad
from alaro_analysis.dsd.profile_cfad import PANEL_FIELDS


def write_data_txt(
    path: Path,
    cfads: dict[str, ExperimentCfad],
    experiments: list[str],
    *,
    lead_tag: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("CFAD-style DSD diagnostic distributions\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"DSD lead selection: {lead_tag}\n")
        fh.write("Colormap: cmaps.WhViBlGrYeOrRe\n")
        fh.write(f"bins: {args.bins}\n")
        fh.write(f"d0_range: {tuple(args.d0_range)}\n")
        fh.write(f"lognw_range: {tuple(args.lognw_range)}\n")
        fh.write("\n")
        fh.write("experiment,field,n_profiles,total_samples,freezing_level_km,source\n")
        for experiment in experiments:
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            freeze = cfads[experiment].freezing_level_km
            for field in PANEL_FIELDS:
                grid = cfads[experiment].grids[field]
                fh.write(
                    f"{experiment},{label},{field},{grid.n_profiles},"
                    f"{int(np.sum(grid.hist))},{freeze:.10e},{grid.source}\n"
                )
