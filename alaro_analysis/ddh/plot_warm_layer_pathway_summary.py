"""Warm-layer DDH pathway summary for the graupel mechanism question."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alaro_analysis.common.constants import EXPERIMENT_COLORS, EXPERIMENT_LABELS
from alaro_analysis.ddh.io import AGG_DIR, EXPERIMENTS, freezing_level_km, load_temperature
from alaro_analysis.ddh.plot_style import (
    CONVECTION_COLOR,
    PANEL_GRID_ALPHA,
    RESOLVED_COLOR,
)


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "DDH-warm_layer_pathway_summary"
)
DATA_TXT_SUBDIR = "data_txt"
FIGURE_NAME = "warm_layer_condensation_pathway_summary.png"
TEXT_NAME = "warm_layer_condensation_pathway_summary.txt"
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


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


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


def _positions() -> tuple[np.ndarray, list[str], list[float]]:
    positions = []
    labels = []
    centers = []
    n_exp = len(EXPERIMENT_ORDER)
    gap = 1.05
    for layer_idx, layer in enumerate(LAYER_ORDER):
        start = layer_idx * (n_exp + gap)
        group_positions = []
        for exp_idx, experiment in enumerate(EXPERIMENT_ORDER):
            x = start + exp_idx
            positions.append(x)
            labels.append(EXPERIMENT_LABELS.get(experiment, experiment))
            group_positions.append(x)
        centers.append(float(np.mean(group_positions)))
    return np.asarray(positions, dtype=np.float64), labels, centers


def _metric_map(metrics: list[LayerMetrics]) -> dict[tuple[str, str], LayerMetrics]:
    return {(row.layer, row.experiment): row for row in metrics}


def _annotate_layer_groups(ax, centers: list[float]) -> None:
    ymin, ymax = ax.get_ylim()
    for center, label in zip(centers, LAYER_ORDER):
        ax.text(
            center,
            -0.16,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )
    for split in (
        0.5 * (centers[0] + centers[1]),
        0.5 * (centers[1] + centers[2]),
    ):
        ax.axvline(split, color="0.75", lw=0.8, ls=":")
    ax.set_ylim(ymin, ymax)


def plot_summary(
    metrics: list[LayerMetrics],
    graupel: list[GraupelMetrics],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_by_key = _metric_map(metrics)
    x, labels, centers = _positions()
    bar_width = 0.72

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.2, 13.5),
        constrained_layout=True,
    )

    ax = axes[0]
    conv_values = []
    resolved_values = []
    total_values = []
    for layer in LAYER_ORDER:
        for experiment in EXPERIMENT_ORDER:
            row = metric_by_key[(layer, experiment)]
            conv_values.append(row.qv_condcv_sink)
            resolved_values.append(row.qv_condrs_sink)
            total_values.append(row.qv_total_cond_sink)
    conv_arr = np.asarray(conv_values)
    resolved_arr = np.asarray(resolved_values)
    total_arr = np.asarray(total_values)
    ax.bar(
        x,
        conv_arr,
        width=bar_width,
        color=CONVECTION_COLOR,
        label="Convection-scheme condensation",
    )
    ax.bar(
        x,
        resolved_arr,
        width=bar_width,
        bottom=conv_arr,
        color=RESOLVED_COLOR,
        label="Resolved-microphysics condensation",
    )
    ax.plot(x, total_arr, "ko", ms=4.2, label="Total")
    ax.set_title("(a) Vapour condensation routing", loc="left", fontweight="bold")
    ax.set_ylabel("Condensation sink\n(g kg$^{-1}$ day$^{-1}$ km)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=PANEL_GRID_ALPHA)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    _annotate_layer_groups(ax, centers)

    ax = axes[1]
    reservoir_values = []
    colors = []
    for layer in LAYER_ORDER:
        for experiment in EXPERIMENT_ORDER:
            row = metric_by_key[(layer, experiment)]
            reservoir_values.append(max(row.warm_liquid_rain_amount, 1.0e-12))
            colors.append(EXPERIMENT_COLORS.get(experiment, "0.5"))
    ax.bar(x, reservoir_values, width=bar_width, color=colors)
    ax.set_yscale("log")
    ax.set_title("(b) Warm liquid plus rain reservoir", loc="left", fontweight="bold")
    ax.set_ylabel("QL + QR amount\n(g kg$^{-1}$ km)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=PANEL_GRID_ALPHA, which="both")
    for layer_idx, layer in enumerate(LAYER_ORDER):
        c1m = metric_by_key[(layer, "control")].warm_liquid_rain_amount
        g1m = metric_by_key[(layer, "graupel")].warm_liquid_rain_amount
        g2m = metric_by_key[(layer, "2mom")].warm_liquid_rain_amount
        xpos = centers[layer_idx]
        ax.text(
            xpos,
            0.94,
            f"G1M/C1M={g1m / c1m:.2f}\nG2M/C1M={g2m / c1m:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    _annotate_layer_groups(ax, centers)

    ax = axes[2]
    gx = np.arange(len(graupel), dtype=np.float64)
    width = 0.22
    auto = np.asarray([row.qg_auto_rs for row in graupel])
    evap = np.asarray([row.qg_evap_rs for row in graupel])
    prec = np.asarray([row.qg_prec_rs for row in graupel])
    stored = np.asarray([row.qg_amount for row in graupel])
    ax.bar(gx - width, auto, width=width, color="#E8891D", label="QG auto-rs")
    ax.bar(gx, evap, width=width, color="#9B59B6", label="QG evap-rs")
    ax.bar(gx + width, prec, width=width, color="#FF6B6B", label="QG prec-rs")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_title(
        "(c) Graupel pathway above 3 km",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylabel("QG budget term\n(g kg$^{-1}$ day$^{-1}$ km)")
    ax.set_xticks(gx, [row.label for row in graupel])
    ax.grid(axis="y", alpha=PANEL_GRID_ALPHA)
    ax.legend(loc="upper left", ncols=3, fontsize=9)

    fig.suptitle(
        "Warm-layer pathway summary from DDH +24 budgets",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_txt(
    metrics: list[LayerMetrics],
    graupel: list[GraupelMetrics],
    path: Path,
    figure_path: Path,
    *,
    agg_dir: Path,
    lead: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        title = "Warm-layer DDH Pathway Summary Plot Data"
        fh.write(f"{title}\n")
        fh.write(f"{'=' * len(title)}\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Aggregated DDH directory: {agg_dir}\n")
        fh.write(f"Lead: {lead}\n")
        fh.write("Condensation values are positive QV sinks: -QV.condcv and -QV.condrs.\n")
        fh.write("Layer tops labelled freezing level use each experiment's mean freezing level.\n")
        fh.write("Panel C uses raw QG budget-term signs.\n\n")

        fh.write("Condensation routing and warm reservoir data\n")
        fh.write("--------------------------------------------\n")
        columns = (
            "experiment",
            "label",
            "layer",
            "bottom_km",
            "top_km",
            "qv_condcv_sink_gkgday_km",
            "qv_condrs_sink_gkgday_km",
            "qv_total_cond_sink_gkgday_km",
            "ql_amount_gkg_km",
            "qr_amount_gkg_km",
            "warm_liquid_rain_amount_gkg_km",
        )
        fh.write(",".join(columns) + "\n")
        for row in metrics:
            values = (
                row.experiment,
                row.label,
                row.layer,
                row.bottom_km,
                row.top_km,
                row.qv_condcv_sink,
                row.qv_condrs_sink,
                row.qv_total_cond_sink,
                row.ql_amount,
                row.qr_amount,
                row.warm_liquid_rain_amount,
            )
            fh.write(",".join(_fmt(value) for value in values) + "\n")

        fh.write("\nGraupel pathway data\n")
        fh.write("--------------------\n")
        columns = (
            "experiment",
            "label",
            "layer",
            "bottom_km",
            "top_km",
            "qg_auto_rs_gkgday_km",
            "qg_evap_rs_gkgday_km",
            "qg_prec_rs_gkgday_km",
            "qg_amount_gkg_km",
        )
        fh.write(",".join(columns) + "\n")
        for row in graupel:
            values = (
                row.experiment,
                row.label,
                row.layer,
                row.bottom_km,
                row.top_km,
                row.qg_auto_rs,
                row.qg_evap_rs,
                row.qg_prec_rs,
                row.qg_amount,
            )
            fh.write(",".join(_fmt(value) for value in values) + "\n")


def run(
    *,
    agg_dir: Path = AGG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    lead: str = "0024",
) -> dict[str, Path]:
    metrics = compute_layer_metrics(agg_dir=agg_dir, lead=lead)
    graupel = compute_graupel_metrics(agg_dir=agg_dir, lead=lead)
    figure_path = output_dir / FIGURE_NAME
    txt_path = output_dir / DATA_TXT_SUBDIR / TEXT_NAME
    plot_summary(metrics, graupel, figure_path)
    write_txt(metrics, graupel, txt_path, figure_path, agg_dir=agg_dir, lead=lead)
    outputs = {"figure": figure_path, "txt": txt_path}
    for key, path in outputs.items():
        print(f"{key}: {path}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agg-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lead", default="0024")
    args = parser.parse_args()
    run(agg_dir=args.agg_dir, output_dir=args.output_dir, lead=args.lead)


if __name__ == "__main__":
    main()
