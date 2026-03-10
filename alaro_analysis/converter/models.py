from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariablePlan:
    output_vars: tuple[str, ...]
    read_vars: tuple[str, ...]
    missing_requested: tuple[str, ...]
    derive_model_rh: bool
    rh_q_var: str | None
    rh_t_var: str | None
    rh_p_var: str | None


@dataclass(frozen=True)
class CropWindow:
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    source_y: int
    source_x: int


@dataclass(frozen=True)
class FileTask:
    day_name: str
    hour: int
    source_file: str


@dataclass(frozen=True)
class RunConfig:
    input_root: str
    output_root: str
    workers: int
    bbox_west: float
    bbox_east: float
    bbox_south: float
    bbox_north: float
    include_init: bool
    include_hour24: bool
    compress: str
    compress_level: int
    overwrite: bool
    skip_incomplete_days: bool
    start_date: str | None
    end_date: str | None
    mask_file: str | None
    mask_var: str | None
    mask_lat_name: str | None
    mask_lon_name: str | None
    mask_threshold: float
    quiet: bool


__all__ = ["CropWindow", "FileTask", "RunConfig", "VariablePlan"]
