"""Export text data tables for the DSD ``new plots`` figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alaro_analysis.workflows.disdrometer_dsd import (
    FIGURE_DIR,
    PANEL_ORDER,
    PROCESSED_DIR,
    _compute_obs_contour,
)


def _load_samples(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path) as npz:
        return {
            name: {
                key.split("__", 1)[1]: np.asarray(npz[key])
                for key in npz.files
                if key.startswith(f"{name}__")
            }
            for name in PANEL_ORDER
        }


def _axis_edges(
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = samples[name][x_field]
        sy = samples[name]["log_nw"]
        if sx.size:
            x_los.append(float(np.nanpercentile(sx, 0.5)))
            x_his.append(float(np.nanpercentile(sx, 99.5)))
        if sy.size:
            y_los.append(float(np.nanpercentile(sy, 0.5)))
            y_his.append(float(np.nanpercentile(sy, 99.5)))
    if not x_los or not y_los:
        raise RuntimeError("No samples available for DSD text export")
    x_lo = max(0.2, min(x_los))
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    return (
        np.linspace(x_lo - x_pad, x_hi + x_pad, bins + 1),
        np.linspace(y_lo - y_pad, y_hi + y_pad, bins + 1),
    )


def _write_plot_txt(
    path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    *,
    x_field: str,
    x_label: str,
    plot_png: Path,
    bins: int,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_edges, y_edges = _axis_edges(samples, x_field, bins=bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    contour = _compute_obs_contour(samples["obs"][x_field], samples["obs"]["log_nw"], x_edges, y_edges)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write("# Text data for the corresponding PNG plot.\n")
        f.write(f"# PNG: {plot_png}\n")
        f.write(f"# x_field: {x_field}\n")
        f.write(f"# x_label: {x_label}\n")
        f.write("# y_field: log_nw\n")
        f.write("# y_label: log10 Nw (m^-3 mm^-1)\n")
        f.write(f"# bins: {bins}\n")
        f.write("# frequency_fraction is the panel histogram divided by that panel's total samples.\n")
        f.write("# log10_frequency is log10(frequency_fraction); NaN means the bin is empty.\n")
        f.write("#\n")
        f.write("# dataset_counts\n")
        f.write("dataset,n_samples,mu_p05,mu_p50,mu_p95,dm_p50,lognw_p50\n")
        for name in PANEL_ORDER:
            sample = samples[name]
            mu = sample.get("mu", np.full(sample[x_field].shape, np.nan))
            mu_pct = np.nanpercentile(mu, [5, 50, 95]) if mu.size else [np.nan, np.nan, np.nan]
            dm50 = np.nanmedian(sample["dm_mm"]) if sample["dm_mm"].size else np.nan
            lognw50 = np.nanmedian(sample["log_nw"]) if sample["log_nw"].size else np.nan
            f.write(
                f"{name},{sample[x_field].size},{mu_pct[0]:.9g},{mu_pct[1]:.9g},"
                f"{mu_pct[2]:.9g},{dm50:.9g},{lognw50:.9g}\n"
            )

        f.write("\n# x_edges\n")
        f.write(",".join(f"{v:.9g}" for v in x_edges) + "\n")
        f.write("\n# y_edges\n")
        f.write(",".join(f"{v:.9g}" for v in y_edges) + "\n")

        f.write("\n# joint_histograms\n")
        f.write("dataset,x_bin_left,x_bin_right,y_bin_bottom,y_bin_top,frequency_fraction,log10_frequency\n")
        for name in PANEL_ORDER:
            sx = samples[name][x_field]
            sy = samples[name]["log_nw"]
            h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
            freq = h / max(1.0, h.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                log_freq = np.log10(np.where(freq > 0.0, freq, np.nan))
            for ix in range(freq.shape[0]):
                for iy in range(freq.shape[1]):
                    f.write(
                        f"{name},{x_edges[ix]:.9g},{x_edges[ix + 1]:.9g},"
                        f"{y_edges[iy]:.9g},{y_edges[iy + 1]:.9g},"
                        f"{freq[ix, iy]:.9g},{log_freq[ix, iy]:.9g}\n"
                    )

        f.write("\n# x_marginal_pdf\n")
        f.write("dataset,x_center,pdf\n")
        for name in PANEL_ORDER:
            sx = samples[name][x_field]
            x_pdf = np.histogram(sx, bins=x_edges, density=True)[0] if sx.size else np.zeros(bins)
            for center, value in zip(x_centers, x_pdf):
                f.write(f"{name},{center:.9g},{value:.9g}\n")

        f.write("\n# y_marginal_pdf\n")
        f.write("dataset,y_center,pdf\n")
        for name in PANEL_ORDER:
            sy = samples[name]["log_nw"]
            y_pdf = np.histogram(sy, bins=y_edges, density=True)[0] if sy.size else np.zeros(bins)
            for center, value in zip(y_centers, y_pdf):
                f.write(f"{name},{center:.9g},{value:.9g}\n")

        if contour is not None:
            xc, yc, field, levels = contour
            f.write("\n# obs_contour_smoothed_frequency_percent\n")
            f.write("# contour_levels_percent: " + ",".join(f"{v:.9g}" for v in levels) + "\n")
            f.write("x_center,y_center,smoothed_frequency_percent\n")
            for ix, x_val in enumerate(xc):
                for iy, y_val in enumerate(yc):
                    f.write(f"{x_val:.9g},{y_val:.9g},{field[iy, ix]:.9g}\n")


def _export_set(
    npz_path: Path,
    figure_dir: Path,
    *,
    prefix: str,
    title_note: str,
    output_tag: str,
    bins: int,
) -> None:
    samples = _load_samples(npz_path)
    data_dir = figure_dir / "data_txt"
    for x_field, x_label, suffix in (
        ("d0_mm", "D0 (mm)", "logNw_D0"),
        ("dm_mm", "Dm (mm)", "logNw_Dm"),
    ):
        png = figure_dir / f"{prefix}_{suffix}_{output_tag}.png"
        txt = data_dir / f"{png.stem}.txt"
        _write_plot_txt(
            txt,
            samples,
            x_field=x_field,
            x_label=x_label,
            plot_png=png,
            bins=bins,
            title=f"{title_note}; {x_label} vs log10 Nw",
        )
        print(f"wrote {txt}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export text data for DSD new plots.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--plot-root", type=Path, default=FIGURE_DIR / "new plots")
    parser.add_argument("--output-tag", default="all_leads")
    parser.add_argument("--bins", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _export_set(
        args.processed_dir / f"disdrometer_dsd_new_plots_mu1_raw_{args.output_tag}.npz",
        args.plot_root / "mu = 1",
        prefix="dsd_percell_mu1_raw",
        title_note="mu=1 normalized gamma, raw per-cell fields",
        output_tag=args.output_tag,
        bins=args.bins,
    )
    _export_set(
        args.processed_dir / f"disdrometer_dsd_new_plots_native_fitted_mu_raw_{args.output_tag}.npz",
        args.plot_root / "old way of doing mu",
        prefix="dsd_percell_native_fitted_mu_raw",
        title_note="native/fitted-mu normalized gamma, raw per-cell fields",
        output_tag=args.output_tag,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
