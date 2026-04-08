"""
ExperimentSet: the main entry point for ALARO analysis.

Provides a simple, high-level API for common tasks::

    from alaro_analysis import ExperimentSet

    exps = ExperimentSet.from_three_dirs(control, graupel, twomom)
    exps.plot_surface_diurnal("CLPMHAUT.MOD.XFU", "output.png",
                              label="BL height", unit="m")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
        Mapping from experiment name to the masked-netcdf root directory.
    fa_input_dirs : dict[str, Path] | None
        Optional mapping from experiment name to the raw FA input
        directory (for conversion). Only needed if you call ``.convert()``.
    geopotential_dirs : dict[str, Path] | None
        Optional mapping from experiment name to the geopotential
        variable directory (for height-axis construction).
    """

    experiment_dirs: dict[str, Path]
    fa_input_dirs: dict[str, Path] | None = None
    geopotential_dirs: dict[str, Path] | None = None

    # ----- convenience constructors -----

    @classmethod
    def from_three_dirs(
        cls,
        control: Path | str,
        graupel: Path | str,
        twomom: Path | str,
        *,
        fa_control: Path | str | None = None,
        fa_graupel: Path | str | None = None,
        fa_twomom: Path | str | None = None,
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
        fa_dirs = None
        if any(d is not None for d in (fa_control, fa_graupel, fa_twomom)):
            fa_dirs = {}
            if fa_control is not None:
                fa_dirs["control"] = Path(fa_control)
            if fa_graupel is not None:
                fa_dirs["graupel"] = Path(fa_graupel)
            if fa_twomom is not None:
                fa_dirs["2mom"] = Path(fa_twomom)
        geo_dirs = None
        if any(d is not None for d in (control_geo, graupel_geo, twomom_geo)):
            geo_dirs = {}
            if control_geo is not None:
                geo_dirs["control"] = Path(control_geo)
            if graupel_geo is not None:
                geo_dirs["graupel"] = Path(graupel_geo)
            if twomom_geo is not None:
                geo_dirs["2mom"] = Path(twomom_geo)
        return cls(
            experiment_dirs=experiment_dirs,
            fa_input_dirs=fa_dirs,
            geopotential_dirs=geo_dirs,
        )

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
        """Build normalised-token -> directory-name maps per experiment."""
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
        """Return the short label for an experiment (C1M, G1M, G2M, ...)."""
        return EXPERIMENT_LABELS.get(experiment, experiment)

    @property
    def experiments(self) -> tuple[str, ...]:
        """Ordered tuple of experiment names present in this set."""
        ordered = [e for e in EXPERIMENTS if e in self.experiment_dirs]
        # Include any extra experiments not in the default list
        for e in self.experiment_dirs:
            if e not in ordered:
                ordered.append(e)
        return tuple(ordered)

    # =====================================================================
    # High-level API: compute, plot, convert
    # =====================================================================

    def compute_surface_diurnal(
        self,
        variable: str,
        *,
        allowed_months: tuple[int, ...] | None = None,
        utc_offset_hours: int = -4,
        max_days: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the mean 24-hour surface diurnal cycle for all experiments.

        Returns a dict mapping experiment name to a 24-element array.

        Example::

            data = exps.compute_surface_diurnal("CLPMHAUT.MOD.XFU")
            # data["control"] -> ndarray of shape (24,)
        """
        from alaro_analysis.analysis.profiles import compute_surface_diurnal_cycle
        from alaro_analysis.data.discovery import collect_file_records

        var_maps = self.discover_variable_maps()
        line_data: dict[str, np.ndarray] = {}

        for exp in self.experiments:
            var_name = self.resolve_var_name(exp, [variable], variable_maps=var_maps)
            if var_name is None:
                print(f"[warn] {exp}: variable '{variable}' not found, skipping.", flush=True)
                continue

            var_dir = self.experiment_dirs[exp] / var_name
            if not var_dir.exists():
                print(f"[warn] {exp}: directory {var_dir} not found, skipping.", flush=True)
                continue

            records = collect_file_records(
                var_dir,
                max_days=max_days,
                allowed_months=allowed_months,
                utc_offset_hours=utc_offset_hours,
            )
            if not records:
                print(f"[warn] {exp}: no records found in {var_dir}", flush=True)
                continue

            mean, _, _ = compute_surface_diurnal_cycle(
                records, var_name, token_normalizer=normalize_var_token,
            )
            line_data[exp] = mean

        return line_data

    def plot_surface_diurnal(
        self,
        variable: str,
        output_file: Path | str,
        *,
        label: str = "",
        unit: str = "",
        period_label: str = "Full 2-year (all months)",
        allowed_months: tuple[int, ...] | None = None,
        utc_offset_hours: int = -4,
        max_days: int | None = None,
        zoom_inset: bool = False,
        dpi: int = 450,
    ) -> None:
        """Compute and plot the surface diurnal cycle in one call.

        Example::

            exps.plot_surface_diurnal(
                "CLPMHAUT.MOD.XFU", "pblh.png",
                label="Boundary layer height", unit="m",
            )
        """
        from alaro_analysis.plotting.panels import plot_surface_diurnal_cycle

        line_data = self.compute_surface_diurnal(
            variable,
            allowed_months=allowed_months,
            utc_offset_hours=utc_offset_hours,
            max_days=max_days,
        )

        if not line_data:
            print(f"[warn] No data for '{variable}', skipping plot.", flush=True)
            return

        plot_surface_diurnal_cycle(
            line_data,
            Path(output_file),
            variable_label=label or variable,
            variable_unit=unit,
            period_label=period_label,
            utc_offset_hours=utc_offset_hours,
            zoom_inset=zoom_inset,
            dpi=dpi,
        )

    def convert(
        self,
        variables: str | list[str],
        *,
        mask_file: str | Path | None = None,
        workers: int = 16,
        overwrite: bool = False,
    ) -> None:
        """Convert FA files to masked NetCDF for all experiments.

        Requires ``fa_input_dirs`` to be set (via constructor or
        ``from_three_dirs(fa_control=..., ...)``).

        Example::

            exps.convert("CLPMHAUT.MOD.XFU",
                         mask_file="/path/to/Radar_mask_latlon.nc")
        """
        from alaro_analysis.converter.pipeline import run_conversion
        from alaro_analysis.converter.models import RunConfig

        if self.fa_input_dirs is None:
            raise ValueError(
                "fa_input_dirs not set. Pass fa_control/fa_graupel/fa_twomom "
                "to from_three_dirs(), or set fa_input_dirs directly."
            )

        if isinstance(variables, str):
            variables = [variables]

        for exp in self.experiments:
            if exp not in self.fa_input_dirs:
                print(f"[warn] {exp}: no FA input dir, skipping conversion.", flush=True)
                continue

            input_root = self.fa_input_dirs[exp]
            output_root = self.experiment_dirs[exp]

            print(f"\n===== Converting {', '.join(variables)} for {exp} =====", flush=True)

            cfg = RunConfig(
                input_root=str(input_root),
                output_root=str(output_root),
                workers=workers,
                bbox_west=-67.0,
                bbox_east=-53.0,
                bbox_south=-10.0,
                bbox_north=4.0,
                include_init=True,
                include_hour24=False,
                compress="zlib",
                compress_level=1,
                overwrite=overwrite,
                skip_incomplete_days=True,
                start_date=None,
                end_date=None,
                mask_file=str(mask_file) if mask_file else None,
                mask_var=None,
                mask_lat_name=None,
                mask_lon_name=None,
                mask_threshold=0.5,
                quiet=False,
            )
            run_conversion(cfg, variables)
