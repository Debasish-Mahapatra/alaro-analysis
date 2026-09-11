"""Vertical integration and warm-layer budget metrics, independent of figure layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_LABELS
from alaro_analysis.ddh.io import EXPERIMENTS, freezing_level_km, load_temperature

EXPERIMENT_ORDER = tuple(EXPERIMENTS.keys())


GRAUPEL_EXPERIMENTS = ("graupel", "2mom")


LAYER_ORDER = ("0-3 km", "3 km-freezing level", "0-freezing level")


@dataclass(frozen=True)
class LayerDef:
    label: str
    bottom_km: float
    top_kind: str
    fixed_top_km: float | None = None


@dataclass(frozen=True)
class LayerMetrics:
    experiment: str
    label: str
    layer: str
    bottom_km: float
    top_km: float
    qv_condcv_sink: float
    qv_condrs_sink: float
    qv_total_cond_sink: float
    ql_amount: float
    qr_amount: float
    warm_liquid_rain_amount: float


@dataclass(frozen=True)
class GraupelMetrics:
    experiment: str
    label: str
    layer: str
    bottom_km: float
    top_km: float
    qg_auto_rs: float
    qg_evap_rs: float
    qg_prec_rs: float
    qg_amount: float


LAYERS = (
    LayerDef("0-3 km", 0.0, "fixed", 3.0),
    LayerDef("3 km-freezing level", 3.0, "freezing", None),
    LayerDef("0-freezing level", 0.0, "freezing", None),
)


def _load_npz(agg_dir: Path, experiment: str, species: str, lead: str) -> dict[str, np.ndarray]:
    path = agg_dir / f"lead{lead}_VZ" / f"{experiment}_{species}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing DDH aggregate: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _block(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    key = f"block__{name}"
    if key not in data:
        raise KeyError(f"Missing DDH block {key}")
    return np.asarray(data[key], dtype=np.float64)


def integrate_layer(
    altitude_km: np.ndarray,
    profile: np.ndarray,
    *,
    bottom_km: float,
    top_km: float,
) -> float:
    z = np.asarray(altitude_km, dtype=np.float64)
    values = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    z = z[finite]
    values = values[finite]
    order = np.argsort(z)
    z = z[order]
    values = values[order]

    bottom = max(float(bottom_km), float(z[0]))
    top = min(float(top_km), float(z[-1]))
    if top <= bottom:
        return float("nan")
    layer_z = np.concatenate(([bottom], z[(z > bottom) & (z < top)], [top]))
    layer_values = np.interp(layer_z, z, values)
    return float(np.trapezoid(layer_values, layer_z))


def compute_layer_metrics(
    *,
    agg_dir: Path,
    lead: str,
    experiments: Iterable[str] = EXPERIMENT_ORDER,
) -> list[LayerMetrics]:
    rows: list[LayerMetrics] = []
    for experiment in experiments:
        qv = _load_npz(agg_dir, experiment, "QV", lead)
        ql = _load_npz(agg_dir, experiment, "QL", lead)
        qr = _load_npz(agg_dir, experiment, "QR", lead)
        for layer in LAYERS:
            top = layer_top(experiment, layer)
            condcv = -integrate_layer(
                qv["altitude_km"],
                _block(qv, "condcv"),
                bottom_km=layer.bottom_km,
                top_km=top,
            )
            condrs = -integrate_layer(
                qv["altitude_km"],
                _block(qv, "condrs"),
                bottom_km=layer.bottom_km,
                top_km=top,
            )
            ql_amount = integrate_layer(
                ql["altitude_km"],
                _block(ql, "VQLM"),
                bottom_km=layer.bottom_km,
                top_km=top,
            )
            qr_amount = integrate_layer(
                qr["altitude_km"],
                _block(qr, "VQRM"),
                bottom_km=layer.bottom_km,
                top_km=top,
            )
            rows.append(
                LayerMetrics(
                    experiment=experiment,
                    label=EXPERIMENT_LABELS.get(experiment, experiment),
                    layer=layer.label,
                    bottom_km=layer.bottom_km,
                    top_km=top,
                    qv_condcv_sink=condcv,
                    qv_condrs_sink=condrs,
                    qv_total_cond_sink=condcv + condrs,
                    ql_amount=ql_amount,
                    qr_amount=qr_amount,
                    warm_liquid_rain_amount=ql_amount + qr_amount,
                )
            )
    return rows


def compute_graupel_metrics(
    *,
    agg_dir: Path,
    lead: str,
    experiments: Iterable[str] = GRAUPEL_EXPERIMENTS,
) -> list[GraupelMetrics]:
    rows: list[GraupelMetrics] = []
    layer = next(item for item in LAYERS if item.label == "3 km-freezing level")
    for experiment in experiments:
        qg = _load_npz(agg_dir, experiment, "QG", lead)
        top = layer_top(experiment, layer)
        rows.append(
            GraupelMetrics(
                experiment=experiment,
                label=EXPERIMENT_LABELS.get(experiment, experiment),
                layer=layer.label,
                bottom_km=layer.bottom_km,
                top_km=top,
                qg_auto_rs=integrate_layer(
                    qg["altitude_km"],
                    _block(qg, "auto-rs"),
                    bottom_km=layer.bottom_km,
                    top_km=top,
                ),
                qg_evap_rs=integrate_layer(
                    qg["altitude_km"],
                    _block(qg, "evap-rs"),
                    bottom_km=layer.bottom_km,
                    top_km=top,
                ),
                qg_prec_rs=integrate_layer(
                    qg["altitude_km"],
                    _block(qg, "prec-rs"),
                    bottom_km=layer.bottom_km,
                    top_km=top,
                ),
                qg_amount=integrate_layer(
                    qg["altitude_km"],
                    _block(qg, "VQGM"),
                    bottom_km=layer.bottom_km,
                    top_km=top,
                ),
            )
        )
    return rows


def layer_top(experiment: str, layer: LayerDef) -> float:
    if layer.top_kind == "fixed":
        if layer.fixed_top_km is None:
            raise ValueError(f"Layer {layer.label} has no fixed top")
        return float(layer.fixed_top_km)
    if layer.top_kind == "freezing":
        z_freeze = freezing_level_km(load_temperature(experiment))
        if not np.isfinite(z_freeze):
            raise ValueError(f"Could not find freezing level for {experiment}")
        return float(z_freeze)
    raise ValueError(f"Unsupported layer top kind: {layer.top_kind}")
