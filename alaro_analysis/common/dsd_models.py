"""Dsd models. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DSD_FIELDS = ("d0_mm", "log_nw")


@dataclass(frozen=True)
class DsdVerticalProfile:
    height_km: np.ndarray
    temperature_k: np.ndarray
    temperature_count: np.ndarray
    values: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    n_files: int
    source: str


@dataclass(frozen=True)
class DdhQvProfile:
    height_km: np.ndarray
    values: np.ndarray
    n_days: int
    source: str
    block: str


@dataclass(frozen=True)
class CfadGrid:
    height_km: np.ndarray
    x_edges: np.ndarray
    hist: np.ndarray
    counts: np.ndarray
    n_profiles: int
    source: str


@dataclass(frozen=True)
class ExperimentCfad:
    grids: dict[str, CfadGrid]
    temperature_k: np.ndarray
    freezing_level_km: float


@dataclass(frozen=True)
class StrongConvectionSettings:
    metric: str
    min_updraft_pa_s: float
    min_updraft_flux: float
    min_updraft_mesh_frac: float


@dataclass(frozen=True)
class DomainPair:
    full: ExperimentCfad
    strong: ExperimentCfad
