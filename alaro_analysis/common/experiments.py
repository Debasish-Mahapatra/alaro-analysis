"""Canonical experiment identities and their historical filesystem aliases."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Controls the IOP figures show (environment ALARO_IOP_CONTROLS). "both": the control with and
# without the diagnostic graupel, as C1M_diaggrp and C1M_nodiaggrp. "nodiag": only the run
# without it, named C1M in the control's red - the paper's four runs.
DIAG_CONTROL_SHOWN = os.environ.get("ALARO_IOP_CONTROLS", "both") != "nodiag"
# Every name the run with the diagnostic graupel goes by (catalog and rain keys, label, display).
DIAG_CONTROL_NAMES = frozenset({"control", "C1M", "C1M_diaggrp"})


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    color: str
    aliases: tuple[str, ...]
    unavailable_variables: tuple[str, ...] = ()
    display: str = ""  # name in figures when it differs from ``label``, which keys the cache files


EXPERIMENT_CATALOG = {
    # C1M without the diagnostic graupel (LLPSGRP off in aplmphys), IOP warm-restart runs.
    "control-nodiag": Experiment("control-nodiag", "C1M_nodiaggrp",
                                 "#ff7f0e" if DIAG_CONTROL_SHOWN else "#d62728",
                                 ("C1M_nodiaggrp", "control_nodiaggrp"), ("GRAUPEL", "QG", "PNR"),
                                 "" if DIAG_CONTROL_SHOWN else "C1M"),
    "control": Experiment("control", "C1M", "#d62728", ("C1M",), ("GRAUPEL", "QG", "PNR"), "C1M_diaggrp"),
    "graupel": Experiment("graupel", "G1M", "#1f77b4", ("G1M",), ("PNR",)),
    "2mom": Experiment("2mom", "G2M", "#2ca02c", ("G2M",)),
    # G2M-XCU purple, as in every IOP rain, drop-size and microphysics figure.
    "2mom-xcu": Experiment("2mom-xcu", "G2M-XCU", "#9467bd", ("NO3M", "no3m", "G2M-XCU")),
}


NODIAG_NAMES = frozenset({"control-nodiag", "nodiag", "C1M_nodiaggrp", "control_nodiaggrp"})


def shown_runs(names):
    """``names`` without the run with the diagnostic graupel when only the other control is shown."""
    return tuple(n for n in names if DIAG_CONTROL_SHOWN or n not in DIAG_CONTROL_NAMES)


def run_display(name: str) -> str:
    """Figure name of a run given by any of its names: C1M_nodiaggrp is C1M in the four-run set."""
    if name in NODIAG_NAMES:
        return EXPERIMENT_CATALOG["control-nodiag"].display or "C1M_nodiaggrp"
    if name in DIAG_CONTROL_NAMES:
        return "C1M_diaggrp"
    return name


def run_color(name: str) -> str:
    """Line colour of a run given by any of its names."""
    if name in NODIAG_NAMES:
        return EXPERIMENT_CATALOG["control-nodiag"].color
    return EXPERIMENT_CATALOG[canonical_experiment(name)].color


def canonical_experiment(value: str) -> str:
    """Resolve a known alias without renaming existing files or directories."""
    for key, experiment in EXPERIMENT_CATALOG.items():
        if value.casefold() in {name.casefold() for name in (key, *experiment.aliases)}:
            return key
    raise ValueError(
        f"Unknown experiment {value!r}; expected one of {', '.join(EXPERIMENT_CATALOG)}"
    )
