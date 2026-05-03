"""Corrected precipitation PDF and CCDF plots for common-valid rainfall data."""

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
    / "masked-production-final"
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
) -> np.ndarray:
    if lower <= 0.0:
        raise ValueError("PDF lower edge must be positive for log-spaced bins.")
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
    data_dir: Path,
    samples: list[SampleSet],
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    min_threshold: float,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation PDF data\n")
        fh.write("======================\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: common log-spaced bins for every dataset; no Radar clipping.\n")
        fh.write(
            "Density: bin_count / (all_finite_sample_count * bin_width), "
            "so the plotted PDF is not renormalized after dry/light values are removed.\n"
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


def write_ccdf_txt(
    path: Path,
    *,
    data_dir: Path,
    samples: list[SampleSet],
    thresholds: np.ndarray,
    ccdfs: dict[str, np.ndarray],
    min_threshold: float,
    x_break: float,
    x_marker: float,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Precipitation CCDF data\n")
        fh.write("=======================\n")
        fh.write(f"Source data directory: {data_dir}\n")
        fh.write("Method: P(X >= x) with all finite valid samples in the denominator.\n")
        fh.write(f"First plotted threshold: {min_threshold:g} mm h^-1\n")
        fh.write(
            f"Tail handling: values > {x_break:g} mm h^-1 are also shown as one "
            f"final marker at x={x_marker:g} mm h^-1.\n"
        )
        fh.write("\nDataset summary\n")
        fh.write("dataset,n_valid,n_positive,n_ge_0p1,n_gt_100,p_gt_100,max_mm_h\n")
        for sample in samples:
            p_gt = sample.n_gt(x_break) / sample.n_valid
            fh.write(
                f"{sample.config.label},{sample.n_valid},{sample.n_positive},"
                f"{sample.n_ge(0.1)},{sample.n_gt(x_break)},"
                f"{p_gt:.12g},{float(np.nanmax(sample.values)):.10g}\n"
            )
        fh.write("\nPlotted CCDF curve data\n")
        header = ["threshold_mm_h"] + [f"{sample.config.label}_ccdf" for sample in samples]
        fh.write(",".join(header) + "\n")
        for i, threshold in enumerate(thresholds):
            row = [f"{threshold:.10g}"]
            for sample in samples:
                row.append(f"{ccdfs[sample.config.key][i]:.12g}")
            fh.write(",".join(row) + "\n")
        fh.write("\nFinal tail markers\n")
        fh.write("dataset,x_marker_mm_h,p_gt_100\n")
        for sample in samples:
            fh.write(
                f"{sample.config.label},{x_marker:.10g},"
                f"{sample.n_gt(x_break) / sample.n_valid:.12g}\n"
            )


def plot_pdf(
    samples: list[SampleSet],
    *,
    edges: np.ndarray,
    densities: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10, 6))
    for sample in samples:
        cfg = sample.config
        density = densities[cfg.key]
        mask = density > 0.0
        ax.plot(
            centers[mask],
            density[mask],
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=2.0,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Precipitation intensity (mm h$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"PDF (mm$^{-1}$ h)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.25)
    ax.legend(loc="upper right", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ccdf(
    samples: list[SampleSet],
    *,
    thresholds: np.ndarray,
    ccdfs: dict[str, np.ndarray],
    output_path: Path,
    x_break: float,
    x_marker: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    positive_y: list[float] = []
    for sample in samples:
        cfg = sample.config
        y = ccdfs[cfg.key]
        mask = y > 0.0
        positive_y.extend(y[mask].tolist())
        ax.semilogy(
            thresholds[mask],
            y[mask],
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            linewidth=1.7,
        )
        p_gt = sample.n_gt(x_break) / sample.n_valid
        if p_gt > 0.0:
            positive_y.append(p_gt)
            ax.semilogy(
                [x_marker],
                [p_gt],
                marker="o",
                markersize=5,
                color=cfg.color,
                linestyle="None",
            )
            y_end = y[-1]
            if y_end > 0.0:
                ax.semilogy(
                    [x_break, x_marker],
                    [y_end, p_gt],
                    color=cfg.color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                )
    bottom = max(min(positive_y) * 0.6, 1.0e-8) if positive_y else 1.0e-8
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(bottom=bottom, top=1.1)
    ax.set_xlabel(r"Precipitation (mm h$^{-1}$)", fontsize=11)
    ax.set_ylabel(r"CCDF [P(X >= x)]", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.25)
    ax.text(
        x_marker,
        bottom * 1.25,
        r">=100 mm h$^{-1}$",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pdf_min_threshold: float = 0.1,
    pdf_bins: int = 99,
    ccdf_min_threshold: float = 0.1,
    ccdf_thresholds: int = 1000,
    x_break: float = 100.0,
    x_marker: float = 105.0,
    dpi: int = 400,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "data_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    samples = read_samples(data_dir)

    pdf_edges = common_log_bins(samples, lower=pdf_min_threshold, n_bins=pdf_bins)
    pdf_counts: dict[str, np.ndarray] = {}
    pdf_densities: dict[str, np.ndarray] = {}
    for sample in samples:
        counts, density = compute_unconditional_pdf(sample.values, pdf_edges)
        pdf_counts[sample.config.key] = counts
        pdf_densities[sample.config.key] = density

    thresholds = np.linspace(ccdf_min_threshold, x_break, ccdf_thresholds)
    ccdfs = {
        sample.config.key: compute_ccdf(sample.values, thresholds)
        for sample in samples
    }

    pdf_path = output_dir / "precip_pdf.png"
    ccdf_path = output_dir / "precip_ccdf_linearlog_finalbin_ge100.png"
    pdf_txt = txt_dir / "precip_pdf.txt"
    ccdf_txt = txt_dir / "precip_ccdf_linearlog_finalbin_ge100.txt"

    plot_pdf(samples, edges=pdf_edges, densities=pdf_densities, output_path=pdf_path, dpi=dpi)
    plot_ccdf(
        samples,
        thresholds=thresholds,
        ccdfs=ccdfs,
        output_path=ccdf_path,
        x_break=x_break,
        x_marker=x_marker,
        dpi=dpi,
    )
    write_pdf_txt(
        pdf_txt,
        data_dir=data_dir,
        samples=samples,
        edges=pdf_edges,
        densities=pdf_densities,
        counts=pdf_counts,
        min_threshold=pdf_min_threshold,
    )
    write_ccdf_txt(
        ccdf_txt,
        data_dir=data_dir,
        samples=samples,
        thresholds=thresholds,
        ccdfs=ccdfs,
        min_threshold=ccdf_min_threshold,
        x_break=x_break,
        x_marker=x_marker,
    )
    return {
        "pdf": pdf_path,
        "ccdf": ccdf_path,
        "pdf_txt": pdf_txt,
        "ccdf_txt": ccdf_txt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot corrected precipitation PDF and CCDF from common-valid rainfall data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-min-threshold", type=float, default=0.1)
    parser.add_argument("--pdf-bins", type=int, default=99)
    parser.add_argument("--ccdf-min-threshold", type=float, default=0.1)
    parser.add_argument("--ccdf-thresholds", type=int, default=1000)
    parser.add_argument("--x-break", type=float, default=100.0)
    parser.add_argument("--x-marker", type=float, default=105.0)
    parser.add_argument("--dpi", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = make_plots(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pdf_min_threshold=args.pdf_min_threshold,
        pdf_bins=args.pdf_bins,
        ccdf_min_threshold=args.ccdf_min_threshold,
        ccdf_thresholds=args.ccdf_thresholds,
        x_break=args.x_break,
        x_marker=args.x_marker,
        dpi=args.dpi,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
