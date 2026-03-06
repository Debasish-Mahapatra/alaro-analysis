from __future__ import annotations

from pathlib import Path

from alaro_analysis.common.timeparse import has_pf_subdirs


def discover_variables(experiment_dirs: dict[str, Path]) -> dict[str, set[str]]:
    available: dict[str, set[str]] = {}
    for exp, exp_dir in experiment_dirs.items():
        vars_for_exp = {
            p.name
            for p in exp_dir.iterdir()
            if p.is_dir() and has_pf_subdirs(p) and not p.name.startswith(".")
        }
        available[exp] = vars_for_exp
    return available
