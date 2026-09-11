"""Gridcell cfad exports. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.dsd_models import ExperimentCfad, StrongConvectionSettings
from alaro_analysis.dsd.profile_cfad import DSD_FIELDS
from alaro_analysis.dsd.strong_cfad import UPDRAFT_MESH_VARIABLE, UPDRAFT_VARIABLE


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
