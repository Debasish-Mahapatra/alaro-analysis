"""Dsd profile exports. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.common.dsd_models import DdhQvProfile, DsdVerticalProfile
from alaro_analysis.dsd.vertical_profiles import freezing_levels_by_experiment


def write_data_txt(
    path: Path,
    dsd_profiles: dict[str, DsdVerticalProfile],
    qv_profiles: dict[str, DdhQvProfile],
    experiments: list[str],
    *,
    lead_tag: str,
    ddh_lead: str,
    onemom_closure: str,
    twomom_closure: str,
    qv_block: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Vertical DSD diagnostics and DDH QV profiles\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"DSD lead selection: {lead_tag}\n")
        fh.write(f"DDH QV lead: lead{ddh_lead}_VZ\n")
        fh.write(f"1-moment closure: {onemom_closure}\n")
        fh.write(f"2-moment closure: {twomom_closure}\n")
        fh.write(f"DDH QV block: {qv_block}\n\n")

        fh.write("DSD profiles\n")
        fh.write(
            "experiment,experiment_label,level_index,height_km,"
            "temperature_k,d0_mm,log_nw,"
            "count_temperature,count_d0,count_log_nw,n_files,source\n"
        )
        for experiment in experiments:
            profile = dsd_profiles[experiment]
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for idx, height in enumerate(profile.height_km):
                fh.write(
                    f"{experiment},{label},{idx},{height:.10e},"
                    f"{profile.temperature_k[idx]:.10e},"
                    f"{profile.values['d0_mm'][idx]:.10e},"
                    f"{profile.values['log_nw'][idx]:.10e},"
                    f"{int(profile.temperature_count[idx])},"
                    f"{int(profile.counts['d0_mm'][idx])},"
                    f"{int(profile.counts['log_nw'][idx])},"
                    f"{profile.n_files},{profile.source}\n"
                )

        fh.write("\nFreezing levels\n")
        fh.write("experiment,experiment_label,freezing_level_km\n")
        for experiment, freeze_z in freezing_levels_by_experiment(
            dsd_profiles, experiments
        ).items():
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            fh.write(f"{experiment},{label},{freeze_z:.10e}\n")

        fh.write("\nDDH QV profile\n")
        fh.write("experiment,experiment_label,level_index,height_km,qv,n_days,source\n")
        for experiment in experiments:
            profile = qv_profiles[experiment]
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            for idx, height in enumerate(profile.height_km):
                fh.write(
                    f"{experiment},{label},{idx},{height:.10e},"
                    f"{profile.values[idx]:.10e},{profile.n_days},{profile.source}\n"
                )
