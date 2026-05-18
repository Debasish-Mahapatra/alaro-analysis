"""G1M-C1M process fingerprint for the graupel mechanism question."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alaro_analysis.ddh.io import AGG_DIR
from alaro_analysis.ddh.plot_warm_layer_pathway_summary import compute_layer_metrics


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "DDH-g1m_c1m_process_fingerprint"
)
DATA_TXT_SUBDIR = "data_txt"
FIGURE_NAME = "g1m_c1m_process_fingerprint.png"
TEXT_NAME = "g1m_c1m_process_fingerprint.txt"
UPDRAFT_TEXT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "DDH-updraft_condensation_check/data_txt/"
    "lower_layer_condensation_updraft_consistency.txt"
)
RAINFALL_TEXT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/Analysis/figures/"
    "rainfall_spatial_bias_maps/data_txt/spatial_mean_rainfall_maps.txt"
)


@dataclass(frozen=True)
class FingerprintRow:
    panel: str
    group: str
    label: str
    unit: str
    c1m_value: float
    g1m_value: float

    @property
    def percent_change(self) -> float:
        return (self.g1m_value / self.c1m_value - 1.0) * 100.0


def parse_csv_section(path: Path, heading: str) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise ValueError(f"Could not find section {heading!r} in {path}") from exc

    header_index = None
    for index in range(start + 1, len(lines)):
        line = lines[index].strip()
        if not line or set(line) == {"-"}:
            continue
        header_index = index
        break
    if header_index is None:
        raise ValueError(f"Could not find CSV header for section {heading!r} in {path}")

    data_lines = []
    for line in lines[header_index:]:
        if not line.strip():
            break
        data_lines.append(line)

    return list(csv.DictReader(data_lines))


def metric_value(
    rows: list[dict[str, str]],
    *,
    metric: str,
    label: str,
    hour_window: str = "all_hours",
) -> float:
    for row in rows:
        if (
            row.get("metric") == metric
            and row.get("label") == label
            and row.get("hour_window") == hour_window
        ):
            return float(row["mean_value"])
    raise KeyError(f"Missing {metric} {label} {hour_window}")


def rainfall_mean(rows: list[dict[str, str]], dataset: str) -> float:
    for row in rows:
        if row.get("dataset") == dataset:
            return float(row["domain_mean_mm_h"])
    raise KeyError(f"Missing rainfall dataset {dataset}")


def layer_lookup(rows, experiment: str, layer: str):
    for row in rows:
        if row.experiment == experiment and row.layer == layer:
            return row
    raise KeyError(f"Missing DDH layer {experiment} {layer}")


def build_rows(*, agg_dir: Path, lead: str) -> list[FingerprintRow]:
    ddh_rows = compute_layer_metrics(agg_dir=agg_dir, lead=lead)
    updraft_rows = parse_csv_section(UPDRAFT_TEXT, "0-3 km updraft diagnostics")
    rainfall_rows = parse_csv_section(RAINFALL_TEXT, "Panel summary")

    c_03 = layer_lookup(ddh_rows, "control", "0-3 km")
    g_03 = layer_lookup(ddh_rows, "graupel", "0-3 km")
    c_fl = layer_lookup(ddh_rows, "control", "0-freezing level")
    g_fl = layer_lookup(ddh_rows, "graupel", "0-freezing level")

    rows = [
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft area fraction",
            "fraction",
            metric_value(updraft_rows, metric="updraft_extent", label="C1M"),
            metric_value(updraft_rows, metric="updraft_extent", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft mass flux",
            "kg m-2 s-1",
            metric_value(updraft_rows, metric="updraft_flux", label="C1M"),
            metric_value(updraft_rows, metric="updraft_flux", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Updraft diagnostics",
            "Updraft intensity",
            "Pa s-1",
            metric_value(updraft_rows, metric="updraft_intensity", label="C1M"),
            metric_value(updraft_rows, metric="updraft_intensity", label="G1M"),
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Convection-scheme condensation",
            "g kg-1 day-1 km",
            c_03.qv_condcv_sink,
            g_03.qv_condcv_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Resolved-microphysics condensation",
            "g kg-1 day-1 km",
            c_03.qv_condrs_sink,
            g_03.qv_condrs_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Condensation route",
            "Total condensation",
            "g kg-1 day-1 km",
            c_03.qv_total_cond_sink,
            g_03.qv_total_cond_sink,
        ),
        FingerprintRow(
            "0-3 km",
            "Warm-water reservoir",
            "Cloud liquid plus rain amount",
            "g kg-1 km",
            c_03.warm_liquid_rain_amount,
            g_03.warm_liquid_rain_amount,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Convection-scheme condensation",
            "g kg-1 day-1 km",
            c_fl.qv_condcv_sink,
            g_fl.qv_condcv_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Resolved-microphysics condensation",
            "g kg-1 day-1 km",
            c_fl.qv_condrs_sink,
            g_fl.qv_condrs_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Condensation route",
            "Total condensation",
            "g kg-1 day-1 km",
            c_fl.qv_total_cond_sink,
            g_fl.qv_total_cond_sink,
        ),
        FingerprintRow(
            "0-freezing level",
            "Warm-water reservoir",
            "Cloud liquid plus rain amount",
            "g kg-1 km",
            c_fl.warm_liquid_rain_amount,
            g_fl.warm_liquid_rain_amount,
        ),
        FingerprintRow(
            "Rainfall",
            "Surface rainfall",
            "Domain-mean rainfall",
            "mm h-1",
            rainfall_mean(rainfall_rows, "C1M"),
            rainfall_mean(rainfall_rows, "G1M"),
        ),
    ]
    return rows


def write_text(path: Path, *, figure_path: Path, rows: list[FingerprintRow], lead: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("G1M-C1M Process Fingerprint Plot Data\n")
        fh.write("=====================================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"DDH aggregated directory: {AGG_DIR}\n")
        fh.write(f"DDH lead: {lead}\n")
        fh.write(f"Updraft source text: {UPDRAFT_TEXT}\n")
        fh.write(f"Rainfall source text: {RAINFALL_TEXT}\n")
        fh.write("Percent change formula: (G1M / C1M - 1) * 100.\n")
        fh.write(
            "Updraft diagnostics use UD_OMEGA and UD_MESH_FRAC only; "
            "VITESSE_VERT is not used.\n\n"
        )
        fh.write("Plotted rows\n")
        fh.write("------------\n")
        fh.write(
            "panel,group,quantity,unit,c1m_value,g1m_value,"
            "g1m_minus_c1m_percent\n"
        )
        for row in rows:
            fh.write(
                f"{row.panel},{row.group},{row.label},{row.unit},"
                f"{row.c1m_value:.12g},{row.g1m_value:.12g},"
                f"{row.percent_change:.12g}\n"
            )


def color_for(row: FingerprintRow) -> str:
    if row.label.startswith("Updraft "):
        return "#5a5a5a"
    if row.label == "Convection-scheme condensation":
        return "#d55e00"
    if row.label == "Resolved-microphysics condensation":
        return "#0072b2"
    if row.label == "Total condensation":
        return "#222222"
    if row.group == "Warm-water reservoir":
        return "#009e73"
    return "#7b3294"


def display_label(label: str) -> str:
    replacements = {
        "Convection-scheme condensation": "Convection-scheme\ncondensation",
        "Resolved-microphysics condensation": "Resolved-microphysics\ncondensation",
        "Cloud liquid plus rain amount": "Cloud liquid + rain\namount",
        "Domain-mean rainfall": "Domain-mean\nrainfall",
    }
    return replacements.get(label, fill(label, width=26))


def draw_panel(
    ax,
    rows: list[FingerprintRow],
    title: str,
    *,
    xlim: tuple[float, float],
    show_xlabel: bool,
) -> None:
    rows = list(rows)
    y = list(range(len(rows)))
    values = [row.percent_change for row in rows]
    colors = [color_for(row) for row in rows]

    ax.barh(y, values, color=colors, height=0.68)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlim(*xlim)
    ax.set_yticks(y, [display_label(row.label) for row in rows])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=6)
    if show_xlabel:
        ax.set_xlabel("G1M relative to C1M (%)", fontsize=9)
    else:
        ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=8)

    span = xlim[1] - xlim[0]
    for yy, value in zip(y, values):
        if abs(value) >= 14:
            xpos = value * 0.55
            ha = "center"
            color = "white"
        elif value >= 0:
            xpos = min(value + 0.025 * span, xlim[1] - 0.05 * span)
            ha = "left"
            color = "#222222"
        else:
            xpos = max(value - 0.025 * span, xlim[0] + 0.05 * span)
            ha = "right"
            color = "#222222"
        ax.text(
            xpos,
            yy,
            f"{value:+.0f}%",
            va="center",
            ha=ha,
            fontsize=8,
            fontweight="bold",
            color=color,
        )


def make_plot(rows: list[FingerprintRow], figure_path: Path) -> None:
    top_rows = [row for row in rows if row.panel == "0-3 km"]
    bottom_rows = [
        row
        for row in rows
        if row.panel == "0-freezing level" or row.panel == "Rainfall"
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.6, 6.8),
        constrained_layout=False,
        gridspec_kw={"height_ratios": [7, 5]},
    )
    draw_panel(
        axes[0],
        top_rows,
        "0-3 km: updrafts weaken, but convection-scheme condensation rises",
        xlim=(-110, 75),
        show_xlabel=False,
    )
    draw_panel(
        axes[1],
        bottom_rows,
        "0-freezing level: total condensation, warm water, and rainfall decrease",
        xlim=(-110, 20),
        show_xlabel=True,
    )

    fig.suptitle(
        "G1M minus C1M after adding graupel",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.01,
        0.01,
        "DDH condensation uses lead 0024. Updraft diagnostics use UD_OMEGA and UD_MESH_FRAC only.",
        fontsize=6.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.91, bottom=0.10, hspace=0.28)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a clean G1M-C1M process-fingerprint plot."
    )
    parser.add_argument("--agg-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--lead", default="0024")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_path = args.output_dir / FIGURE_NAME
    text_path = args.output_dir / DATA_TXT_SUBDIR / TEXT_NAME
    rows = build_rows(agg_dir=args.agg_dir, lead=args.lead)
    make_plot(rows, figure_path)
    write_text(text_path, figure_path=figure_path, rows=rows, lead=args.lead)
    print(f"Wrote {figure_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()
