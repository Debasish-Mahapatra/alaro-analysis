"""Canonical experiment identities and their historical filesystem aliases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    color: str
    aliases: tuple[str, ...]
    unavailable_variables: tuple[str, ...] = ()


EXPERIMENT_CATALOG = {
    "control": Experiment("control", "C1M", "#d62728", ("C1M",), ("GRAUPEL", "QG", "PNR")),
    "graupel": Experiment("graupel", "G1M", "#1f77b4", ("G1M",), ("PNR",)),
    "2mom": Experiment("2mom", "G2M", "#2ca02c", ("G2M",)),
    "2mom-xcu": Experiment("2mom-xcu", "G2M-XCU", "#ff7f0e", ("NO3M", "no3m", "G2M-XCU")),
}


def canonical_experiment(value: str) -> str:
    """Resolve a known alias without renaming existing files or directories."""
    for key, experiment in EXPERIMENT_CATALOG.items():
        if value.casefold() in {name.casefold() for name in (key, *experiment.aliases)}:
            return key
    raise ValueError(
        f"Unknown experiment {value!r}; expected one of {', '.join(EXPERIMENT_CATALOG)}"
    )
