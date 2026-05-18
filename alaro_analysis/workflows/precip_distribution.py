"""Corrected precipitation PDF plots for common-valid rainfall data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


RUNS_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS")
DEFAULT_DATA_DIR = (
    RUNS_ROOT
    / "rainfall-regridded-to-imerge"
    / "masked-production-final-hourly-imerg"
    / "common-valid-time-production"
)
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "figures" / "precip_distribution_corrected"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    filename: str
    variable: str
    color: str
    linestyle: str = "-"


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig("radar", "Radar", "Radar_common_valid.nc", "rainfall_rate", "black"),
    DatasetConfig("imerg", "IMERG(GPM)", "IMERG_common_valid.nc", "precipitation", "dimgray", ":"),
    DatasetConfig("control", "C1M", "Control_common_valid.nc", "total_rain", "#d62728"),
    DatasetConfig("graupel", "G1M", "Graupel_common_valid.nc", "total_rain", "#1f77b4"),
    DatasetConfig("2mom", "G2M", "2-Moment_common_valid.nc", "total_rain", "#2ca02c"),
)


@dataclass(frozen=True)
class SampleSet:
    config: DatasetConfig
    values: np.ndarray

    @property
    def n_valid(self) -> int:
        return int(self.values.size)

    @property
    def n_positive(self) -> int:
        return int(np.count_nonzero(self.values > 0.0))

    def n_ge(self, threshold: float) -> int:
        return int(np.count_nonzero(self.values >= threshold))

    def n_gt(self, threshold: float) -> int:
        return int(np.count_nonzero(self.values > threshold))


def read_samples(data_dir: Path) -> list[SampleSet]:
    samples: list[SampleSet] = []
    for cfg in DATASETS:
        path = data_dir / cfg.filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {cfg.label} file: {path}")
        with xr.open_dataset(path) as ds:
            if cfg.variable not in ds:
                raise KeyError(f"{cfg.variable!r} not found in {path}")
            values = np.asarray(ds[cfg.variable].values, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite values found for {cfg.label}: {path}")
        samples.append(SampleSet(cfg, values))
    return samples


def common_log_bins(
    samples: list[SampleSet],
    *,
    lower: float,
    n_bins: int,
    upper: float | None = None,
) -> np.ndarray:
    if lower <= 0.0:
        raise ValueError("PDF lower edge must be positive for log-spaced bins.")
    if upper is None:
        maxima = [float(np.nanmax(s.values)) for s in samples if s.values.size]
        upper = max(maxima)
    if upper <= lower:
        raise ValueError(f"Global maximum {upper:g} must be greater than lower edge {lower:g}.")
    return np.logspace(np.log10(lower), np.log10(upper), n_bins + 1)


def compute_unconditional_pdf(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bin counts and PDF density normalized by all finite samples."""
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts.astype(np.float64) / (values.size * widths)
    return counts.astype(np.int64), density


def compute_ccdf(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return P(X >= threshold) using all finite samples in the denominator."""
    sorted_values = np.sort(values)
    idx = np.searchsorted(sorted_values, thresholds, side="left")
    return (sorted_values.size - idx).astype(np.float64) / sorted_values.size


def write_pdf_txt(
    path: Path,
    *,
    figure_path: Path,
    data_dir: Path,
    samples: list[SampleSet],
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    min_threshold: float,
    x_axis: str,
    y_axis: str,
    tail_break: float | None = None,
    tail_marker: float | None = None,
    tail_counts: dict[str, int] | None = None,
    tail_densities: dict[str, float] | None = None,
    fraction_ranges: tuple[tuple[float, float], ...] = (),
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation PDF data\n")
        fh.write("======================\n")
        fh.write(f"Figure: {figure_path}\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write(f"X axis scale: {x_axis}\n")
        fh.write(f"Y axis scale: {y_axis}\n")
        fh.write("Method: common log-spaced bins for every dataset; no Radar clipping.\n")
        fh.write(
            "Density: bin_count / (all_finite_sample_count * bin_width), "
            "so the plotted PDF is not renormalized after dry/light values are removed.\n"
        )
        if tail_break is not None and tail_marker is not None:
            fh.write(
                f"Tail accounting: values > {tail_break:g} mm h^-1 are excluded "
                "from the linear-x PDF curve and listed below as a compressed tail "
                f"bin using x={tail_marker:g} mm h^-1 for bookkeeping only.\n"
            )
        fh.write(f"First plotted bin edge: {min_threshold:g} mm h^-1\n")
        fh.write("\nDataset summary\n")
        fh.write("dataset,n_valid,n_positive,n_ge_first_edge,n_gt_100,max_mm_h\n")
        for sample in samples:
            vals = sample.values
            fh.write(
                f"{sample.config.label},{sample.n_valid},{sample.n_positive},"
                f"{sample.n_ge(min_threshold)},{sample.n_gt(100.0)},"
                f"{float(np.nanmax(vals)):.10g}\n"
            )
        if fraction_ranges:
            fh.write("\nFraction of all gridpoints by rainfall range\n")
            header = ["dataset"]
            for left, right in fraction_ranges:
                tag = f"{left:g}_to_{right:g}_mm_h"
                header.append(f"count_{tag}")
                header.append(f"fraction_{tag}")
            fh.write(",".join(header) + "\n")
            for sample in samples:
                row = [sample.config.label]
                vals = sample.values
                for left, right in fraction_ranges:
                    count = int(np.count_nonzero((vals >= left) & (vals < right)))
                    row.append(str(count))
                    row.append(f"{count / sample.n_valid:.12g}")
                fh.write(",".join(row) + "\n")
        fh.write("\nPlotted data\n")
        header = ["bin_left_mm_h", "bin_right_mm_h", "bin_center_mm_h"]
        for sample in samples:
            header.append(f"{sample.config.label}_density")
            header.append(f"{sample.config.label}_count")
        fh.write(",".join(header) + "\n")
        for i, center in enumerate(centers):
            row = [f"{edges[i]:.10g}", f"{edges[i + 1]:.10g}", f"{center:.10g}"]
            for sample in samples:
                row.append(f"{densities[sample.config.key][i]:.12g}")
                row.append(str(int(counts[sample.config.key][i])))
            fh.write(",".join(row) + "\n")
        if tail_break is not None and tail_marker is not None and tail_counts is not None:
            fh.write("\nCompressed tail bin\n")
            fh.write("-------------------\n")
            fh.write("dataset,tail_left_mm_h,tail_bookkeeping_x_mm_h,tail_count,tail_density_bookkeeping\n")
            for sample in samples:
                key = sample.config.key
                density = float("nan") if tail_densities is None else tail_densities[key]
                fh.write(
                    f"{sample.config.label},{tail_break:.10g},{tail_marker:.10g},"
                    f"{int(tail_counts[key])},{density:.12g}\n"
                )


def plot_pdf(
    samples: list[SampleSet],
    *,
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    output_path: Path,
    xscale_log: bool,
    yscale_log: bool,
    tail_marker: float | None = None,
    tail_densities: dict[str, float] | None = None,
    x_max: float | None = None,
    show_tail_marker: bool = True,
    dpi: int,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10, 6))
    positive_y: list[float] = []
    for sample in samples:
        cfg = sample.config
        density = densities[cfg.key]
        mask = density > 0.0
        positive_y.extend(density[mask].tolist())
        ax.plot(
            centers[mask],
            density[mask],
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=2.0,
        )
        if show_tail_marker and tail_marker is not None and tail_densities is not None:
            tail_density = tail_densities[cfg.key]
            if tail_density > 0.0:
                positive_y.append(tail_density)
                ax.plot(
                    [tail_marker],
                    [tail_density],
                    marker="o",
                    markersize=5,
                    color=cfg.color,
                    linestyle="None",
                )
    if xscale_log:
        ax.set_xscale("log")
    if yscale_log:
        ax.set_yscale("log")
        if positive_y:
            ax.set_ylim(bottom=max(min(positive_y) * 0.6, 1.0e-10))
    ax.set_xlabel(r"Precipitation intensity (mm h$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"PDF density (mm$^{-1}$ h)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.25)
    ax.legend(loc="upper right", fontsize=14)
    if x_max is not None and not xscale_log:
        ax.set_xlim(0.0, x_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pdf_min_threshold: float = 0.1,
    pdf_bins: int = 99,
    pdf_tail_break: float | None = None,
    pdf_tail_marker: float | None = None,
    dpi: int = 400,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    samples = read_samples(data_dir)
    if pdf_tail_break is not None and pdf_tail_marker is None:
        pdf_tail_marker = pdf_tail_break + 5.0

    pdf_edges = common_log_bins(
        samples, lower=pdf_min_threshold, n_bins=pdf_bins, upper=pdf_tail_break
    )
    pdf_edges_full = common_log_bins(samples, lower=pdf_min_threshold, n_bins=pdf_bins)
    pdf_counts: dict[str, np.ndarray] = {}
    pdf_densities: dict[str, np.ndarray] = {}
    pdf_counts_full: dict[str, np.ndarray] = {}
    pdf_densities_full: dict[str, np.ndarray] = {}
    tail_counts: dict[str, int] | None = None
    tail_densities: dict[str, float] | None = None
    if pdf_tail_break is not None:
        if pdf_tail_marker is None:
            raise ValueError("pdf_tail_marker is required when pdf_tail_break is set.")
        tail_width = pdf_tail_marker - pdf_tail_break
        if tail_width <= 0.0:
            raise ValueError("pdf_tail_marker must be greater than pdf_tail_break for PDF tail accounting.")
        tail_counts = {}
        tail_densities = {}
    for sample in samples:
        counts, density = compute_unconditional_pdf(sample.values, pdf_edges)
        pdf_counts[sample.config.key] = counts
        pdf_densities[sample.config.key] = density
        counts_full, density_full = compute_unconditional_pdf(sample.values, pdf_edges_full)
        pdf_counts_full[sample.config.key] = counts_full
        pdf_densities_full[sample.config.key] = density_full
        if pdf_tail_break is not None and tail_counts is not None and tail_densities is not None:
            n_tail = sample.n_gt(pdf_tail_break)
            tail_counts[sample.config.key] = n_tail
            tail_densities[sample.config.key] = n_tail / (sample.n_valid * tail_width)

    pdf_ylog_path = output_dir / "precip_pdf.png"
    pdf_loglog_path = output_dir / "precip_pdf_loglog_allgrid.png"
    pdf_ylog_txt = txt_dir / "precip_pdf.txt"
    pdf_loglog_txt = txt_dir / "precip_pdf_loglog_allgrid.txt"

    plot_pdf(
        samples,
        edges=pdf_edges,
        densities=pdf_densities,
        output_path=pdf_ylog_path,
        xscale_log=False,
        yscale_log=True,
        tail_marker=pdf_tail_marker,
        tail_densities=tail_densities,
        x_max=pdf_tail_break,
        show_tail_marker=False,
        dpi=dpi,
    )
    plot_pdf(
        samples,
        edges=pdf_edges_full,
        densities=pdf_densities_full,
        output_path=pdf_loglog_path,
        xscale_log=True,
        yscale_log=True,
        dpi=dpi,
    )
    write_pdf_txt(
        pdf_ylog_txt,
        figure_path=pdf_ylog_path,
        data_dir=data_dir,
        samples=samples,
        edges=pdf_edges,
        densities=pdf_densities,
        counts=pdf_counts,
        min_threshold=pdf_min_threshold,
        x_axis="linear",
        y_axis="log",
        tail_break=pdf_tail_break,
        tail_marker=pdf_tail_marker,
        tail_counts=tail_counts,
        tail_densities=tail_densities,
    )
    write_pdf_txt(
        pdf_loglog_txt,
        figure_path=pdf_loglog_path,
        data_dir=data_dir,
        samples=samples,
        edges=pdf_edges_full,
        densities=pdf_densities_full,
        counts=pdf_counts_full,
        min_threshold=pdf_min_threshold,
        x_axis="log",
        y_axis="log",
    )
    return {
        "pdf_ylog": pdf_ylog_path,
        "pdf_loglog": pdf_loglog_path,
        "pdf_ylog_txt": pdf_ylog_txt,
        "pdf_loglog_txt": pdf_loglog_txt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot corrected precipitation PDF versions from common-valid rainfall data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-min-threshold", type=float, default=0.1)
    parser.add_argument("--pdf-bins", type=int, default=99)
    parser.add_argument(
        "--pdf-tail-break",
        type=float,
        default=None,
        help="Optional cap for the linear PDF; by default the full common-valid tail is plotted.",
    )
    parser.add_argument(
        "--pdf-tail-marker",
        type=float,
        default=None,
        help="Bookkeeping x-position used in data_txt for values beyond --pdf-tail-break.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = make_plots(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pdf_min_threshold=args.pdf_min_threshold,
        pdf_bins=args.pdf_bins,
        pdf_tail_break=args.pdf_tail_break,
        pdf_tail_marker=args.pdf_tail_marker,
        dpi=args.dpi,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
