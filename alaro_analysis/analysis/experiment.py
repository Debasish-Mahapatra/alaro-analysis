"""
ExperimentSet: a convenience dataclass for multi-experiment orchestration.

Encapsulates the repeated pattern of control/graupel/2mom directories
and provides common operations (validation, variable discovery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS
from alaro_analysis.common.naming import normalize_var_token
from alaro_analysis.common.timeparse import has_pf_subdirs
from alaro_analysis.data.discovery import discover_variables


@dataclass
class ExperimentSet:
    """A collection of experiment data directories.

    Parameters
    ----------
    experiment_dirs : dict[str, Path]
        Mapping from experiment name (``"control"``, ``"graupel"``,
        ``"2mom"``) to the masked-netcdf root directory.
    geopotential_dirs : dict[str, Path] | None
        Optional mapping from experiment name to the geopotential
        variable directory (for height-axis construction).
    """

    experiment_dirs: dict[str, Path]
    geopotential_dirs: dict[str, Path] | None = None

    # ----- convenience constructors -----

    @classmethod
    def from_three_dirs(
        cls,
        control: Path | str,
        graupel: Path | str,
        twomom: Path | str,
        *,
        control_geo: Path | str | None = None,
        graupel_geo: Path | str | None = None,
        twomom_geo: Path | str | None = None,
    ) -> ExperimentSet:
        """Build an ExperimentSet from explicit directory paths."""
        experiment_dirs = {
            "control": Path(control),
            "graupel": Path(graupel),
            "2mom": Path(twomom),
        }
        geo_dirs = None
        if any(d is not None for d in (control_geo, graupel_geo, twomom_geo)):
            geo_dirs = {}
            if control_geo is not None:
                geo_dirs["control"] = Path(control_geo)
            if graupel_geo is not None:
                geo_dirs["graupel"] = Path(graupel_geo)
            if twomom_geo is not None:
                geo_dirs["2mom"] = Path(twomom_geo)
        return cls(experiment_dirs=experiment_dirs, geopotential_dirs=geo_dirs)

    # ----- validation -----

    def validate(self) -> None:
        """Raise FileNotFoundError if any experiment directory is missing."""
        for exp, d in self.experiment_dirs.items():
            if not d.exists():
                raise FileNotFoundError(f"{exp} data dir not found: {d}")
        if self.geopotential_dirs:
            for exp, d in self.geopotential_dirs.items():
                if not d.exists():
                    raise FileNotFoundError(
                        f"{exp} geopotential dir not found: {d}"
                    )

    # ----- variable discovery -----

    def discover_variables(self) -> dict[str, set[str]]:
        """Discover available variable directories for each experiment."""
        return discover_variables(self.experiment_dirs)

    def common_variables(
        self,
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Return sorted list of variables common to all experiments."""
        available = self.discover_variables()
        common = set.intersection(*(s for s in available.values()))
        if exclude:
            common -= exclude
        return sorted(common)

    def discover_variable_maps(self) -> dict[str, dict[str, str]]:
        """Build normalised-token -> directory-name maps per experiment.

        Useful for fuzzy variable name resolution (e.g. matching
        user-supplied ``"SFX.RN"`` to the actual directory name).
        """
        maps: dict[str, dict[str, str]] = {}
        for exp, exp_dir in self.experiment_dirs.items():
            token_map: dict[str, str] = {}
            for p in sorted(exp_dir.iterdir()):
                if not p.is_dir() or p.name.startswith(".") or not has_pf_subdirs(p):
                    continue
                token = normalize_var_token(p.name)
                if token and token not in token_map:
                    token_map[token] = p.name
            maps[exp] = token_map
        return maps

    def resolve_var_name(
        self,
        experiment: str,
        candidates: tuple[str, ...] | list[str],
        *,
        variable_maps: dict[str, dict[str, str]] | None = None,
    ) -> str | None:
        """Resolve a user-facing variable name to the actual directory name."""
        if variable_maps is None:
            variable_maps = self.discover_variable_maps()
        token_map = variable_maps.get(experiment, {})
        for cand in candidates:
            token = normalize_var_token(cand)
            if token in token_map:
                return token_map[token]
        return None

    # ----- labels -----

    @staticmethod
    def label(experiment: str) -> str:
        """Return the short label for an experiment (C1M, G1M, G2M)."""
        return EXPERIMENT_LABELS.get(experiment, experiment)

    @property
    def experiments(self) -> tuple[str, ...]:
        """Ordered tuple of experiment names."""
        return EXPERIMENTS
