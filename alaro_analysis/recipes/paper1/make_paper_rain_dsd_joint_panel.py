#!/usr/bin/env python3
"""Paper figure 10, rebuilt on the GoAmazon IOPs: log10 Nw against Dm, five datasets.

One panel per dataset (Manacapuru disdrometer, C1M, G1M, G2M, G2M-XCU). Each panel
shows the joint distribution shaded by log10 relative frequency on a shared colour
scale, with the one-dimensional distributions of log10 Nw (left) and Dm (bottom)
overlaid for every dataset and the panel's own drawn in bold, and the observed
distribution outlined in black on every panel. The sample count is in each title.

Model drop size distributions come from the scheme-native moments, truncated at the
disdrometer's minimum detectable diameter Dmin = 0.312 mm
(``data.microphysics_columns``), cached per IOP by ``build_iop_microphysics_columns``;
the observed ones from ``build_iop_disdrometer_seasons``.

Two changes from the published version:

1. G2M-XCU is included as a fifth dataset. It shares G2M's two-moment rain, so its
   spread comes from the same variable-shape distribution, under a different
   convective forcing.
2. The period is the GoAmazon IOPs, not the continuous 2-year run: only C1M still has
   2 years of raw FA on this machine.

**The ARM laser disdrometer only starts on 25 September 2014.** IOP2 therefore uses
the days it does cover, 25 Sep - 15 Oct 2014, which are the same days as the run.
IOP1 (Feb - Mar 2014) has no observations at all, so its panel borrows the same
calendar season of 2015: read it as the season's climatology of drop spectra, not as
a paired sample. The combined figure mixes the two, so the per-IOP figures are the
honest ones.

Written beside the published figure under a different name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.figio import DEFAULT_DPI, LEGEND_FONTSIZE
from alaro_analysis.common.iop_settings import CACHE_DIRS, GOAMAZON_IOPS, IOP_PERIODS
from alaro_analysis.common.paths import data_path

OUTPUT_DIR = data_path("microphysics-paper/10. raindrop size distribution")
FIGURE_STEM = "10. raindrop size distribution_iops"
PERIOD_NAMES = {"combined": "iop1+iop2", "iop1": "iop1", "iop2": "iop2"}
# Shown under the legend, in the free cell of the 2x3 grid: (name, span)
PERIOD_TITLES = {
    "combined": ("GoAmazon IOP1 and IOP2", "combined"),
    "iop1": ("GoAmazon IOP1", "wet season, 1 Feb - 31 Mar 2014"),
    "iop2": ("GoAmazon IOP2", "dry season, 15 Aug - 15 Oct 2014"),
}
# Disdrometer season cached per IOP by build_iop_disdrometer_seasons.
# IOP2 uses the day-matched overlap; the instrument only started on 25 Sep 2014, so
# IOP1 has no observations of its own and borrows the same season of 2015.
OBS_SEASON = {1: "wet", 2: "iop2_2014"}
# Observed source: the ARM Parsivel (no IOP1 coverage) or the CHUVA micro rain radar
# at Manacapuru UEA, which ran through both IOPs (build_iop_mrr_dsd).
OBS_SOURCES = ("parsivel", "parsivel2015", "mrr")
OBS_LABEL = {"parsivel": "Disdrometer", "parsivel2015": "Disdrometer", "mrr": "MRR"}
# (source, IOP) -> cache file, and whether those observations are the IOP's own days
OBS_CACHE = {
    ("parsivel", 1): ("disdrometer/disdrometer_wet.npz", False),
    ("parsivel", 2): ("disdrometer/disdrometer_iop2_2014.npz", True),
    ("parsivel2015", 1): ("disdrometer/disdrometer_wet.npz", False),
    ("parsivel2015", 2): ("disdrometer/disdrometer_dry_2015.npz", False),
    ("mrr", 1): ("mrr/mrr_iop1.npz", True),
    ("mrr", 2): ("mrr/mrr_iop2.npz", True),
}
OBS_WHEN = {
    ("parsivel", 1): "1 Feb - 31 Mar 2015",
    ("parsivel", 2): "25 Sep - 15 Oct 2014",
    ("parsivel2015", 1): "1 Feb - 31 Mar 2015",
    ("parsivel2015", 2): "15 Aug - 15 Oct 2015",
    ("mrr", 1): "1 Feb - 27 Mar 2014",
    ("mrr", 2): "26 Aug - 20 Sep 2014",
}
MRR_SPANS = {
    1: "1 Feb - 27 Mar 2014, the same days as the run",
    2: "26 Aug - 20 Sep 2014, the same days as the run",
}
OBS_SPANS = {
    "wet": "1 Feb - 31 Mar 2015 (same season, different year: the disdrometer "
           "record starts 25 Sep 2014, so IOP1 itself is not covered)",
    "dry": "15 Aug - 15 Oct of 2014 and 2015",
    "iop2_2014": "25 Sep - 15 Oct 2014, the part of IOP2 the disdrometer covers",
}
DATASETS = (
    ("obs", "Disdrometer", "#111111"),
    ("C1M", "C1M", "#d62728"),
    ("G1M", "G1M", "#1f77b4"),
    ("G2M", "G2M", "#2ca02c"),
    ("G2M-XCU", "G2M-XCU", "#9467bd"),
)
# Fraction of observed samples the black contour encloses (highest-density region).
CONTOUR_FRACTION = 0.90
SMOOTH_BINS = 1.5
FREQ_FLOOR = 1e-6  # log10 colour scale floor of the relative frequency


PERIOD_IOPS = {"combined": (1, 2), "iop1": (1,), "iop2": (2,)}


def obs_note(source: str, iops: tuple[int, ...]) -> str:
    """Say, per IOP, whether the observations are that IOP's own days or a stand-in."""
    lines = []
    for iop in iops:
        when = OBS_WHEN[(source, iop)]
        own = OBS_CACHE[(source, iop)][1]
        tail = "the IOP days themselves" if own else "a stand-in from the following year"
        lines.append(f"IOP{iop} observations: {when}, {tail}")
    return "\n".join(lines)


def load_histograms(cache: Path, iops: tuple[int, ...], source: str) -> dict:
    """{key: (hist, n)} plus the shared bin edges, summed over the IOPs."""
    out: dict[str, np.ndarray] = {}
    edges: dict[str, np.ndarray] = {}
    for key, _, _ in DATASETS:
        total = None
        for iop in iops:
            if key == "obs":
                path = cache / OBS_CACHE[(source, iop)][0]
            else:
                path = cache / "columns" / f"{key}_iop{iop}.npz"
            with np.load(path, allow_pickle=True) as data:
                hist = np.asarray(data["dsd_hist"], dtype=np.float64)
                edges.setdefault("dm", np.asarray(data["dm_edges"], dtype=np.float64))
                edges.setdefault("lognw", np.asarray(data["lognw_edges"], dtype=np.float64))
            total = hist if total is None else total + hist
        out[key] = total
    return {"hist": out, **edges}


def relative(hist: np.ndarray) -> np.ndarray:
    total = hist.sum()
    return hist / total if total > 0 else hist


def contour_level(field: np.ndarray, fraction: float) -> float:
    """Density level whose contour encloses ``fraction`` of the distribution."""
    flat = np.sort(field[field > 0].ravel())[::-1]
    if flat.size == 0:
        return float("nan")
    cumulative = np.cumsum(flat)
    index = int(np.searchsorted(cumulative, fraction * cumulative[-1]))
    return float(flat[min(index, flat.size - 1)])


def observed_contour(hist: np.ndarray, dm_centers, lognw_centers):
    """Smoothed observed frequency field and the single level to draw."""
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(relative(hist), sigma=SMOOTH_BINS)
    level = contour_level(field, CONTOUR_FRACTION)
    if not np.isfinite(level) or level <= 0:
        return None
    return dm_centers, lognw_centers, field.T, [level]


def marginals(hist: np.ndarray, dm_edges, lognw_edges):
    """Probability densities of Dm and log10 Nw from the joint histogram."""
    dm_width = np.diff(dm_edges)
    lognw_width = np.diff(lognw_edges)
    total = hist.sum()
    if total <= 0:
        return np.zeros(dm_width.size), np.zeros(lognw_width.size)
    return hist.sum(axis=1) / total / dm_width, hist.sum(axis=0) / total / lognw_width


def draw(data: dict, png: Path, dpi: int, source: str, period: str) -> None:
    dm_edges, lognw_edges = data["dm"], data["lognw"]
    dm_centers = 0.5 * (dm_edges[:-1] + dm_edges[1:])
    lognw_centers = 0.5 * (lognw_edges[:-1] + lognw_edges[1:])
    hist = data["hist"]
    fractions = {key: relative(hist[key]) for key, _, _ in DATASETS}
    pdfs = {key: marginals(hist[key], dm_edges, lognw_edges) for key, _, _ in DATASETS}
    contour = observed_contour(hist["obs"], dm_centers, lognw_centers)

    vmax = max(float(np.nanmax(f)) for f in fractions.values())
    norm = mcolors.LogNorm(vmin=FREQ_FLOOR, vmax=vmax)
    dm_max = max(pdf[0].max() for pdf in pdfs.values()) * 1.1
    lognw_max = max(pdf[1].max() for pdf in pdfs.values()) * 1.1

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17})
    fig = plt.figure(figsize=(22.0, 13.5))
    outer = fig.add_gridspec(2, 3, wspace=0.26, hspace=0.30,
                             left=0.045, right=0.925, top=0.93, bottom=0.07)

    mesh = None
    for index, (key, label, color) in enumerate(DATASETS):
        label = OBS_LABEL[source] if key == "obs" else label
        cell = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[divmod(index, 3)],
            width_ratios=[0.26, 1.0], height_ratios=[1.0, 0.26], wspace=0.05, hspace=0.05,
        )
        ax = fig.add_subplot(cell[0, 1])
        ax_left = fig.add_subplot(cell[0, 0], sharey=ax)
        ax_below = fig.add_subplot(cell[1, 1], sharex=ax)

        shown = np.ma.masked_where(fractions[key] <= 0, fractions[key])
        mesh = ax.pcolormesh(dm_edges, lognw_edges, shown.T, cmap="inferno", norm=norm,
                             shading="auto")
        if contour is not None:
            ax.contour(contour[0], contour[1], contour[2], levels=contour[3],
                       colors="black", linewidths=1.3)
        ax.set_title(f"{label} (n={int(hist[key].sum()):,})", pad=10)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.grid(alpha=0.2, lw=0.5)

        for other, _, other_color in DATASETS:
            own = other == key
            dm_pdf, lognw_pdf = pdfs[other]
            ax_below.plot(dm_centers, dm_pdf, color=other_color,
                          lw=3.0 if own else 1.0, alpha=1.0 if own else 0.55,
                          zorder=3 if own else 2)
            ax_left.plot(lognw_pdf, lognw_centers, color=other_color,
                         lw=3.0 if own else 1.0, alpha=1.0 if own else 0.55,
                         zorder=3 if own else 2)
        ax_left.set_xlim(lognw_max, 0.0)
        ax_left.set_ylim(lognw_edges[0], lognw_edges[-1])
        ax_left.set_ylabel(r"log$_{10}$ N$_w$ (m$^{-3}$ mm$^{-1}$)")
        ax_left.set_xlabel("PDF")
        ax_left.grid(alpha=0.2, lw=0.5)
        ax_below.set_ylim(0.0, dm_max)
        ax_below.set_xlim(dm_edges[0], dm_edges[-1])
        ax_below.set_xlabel(r"D$_m$ = M$_4$/M$_3$ (mm)")
        ax_below.set_ylabel("PDF")
        ax_below.grid(alpha=0.2, lw=0.5)
        for axis in (ax_left, ax_below):
            axis.tick_params(labelsize=11)
            axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))

    cax = fig.add_axes([0.945, 0.30, 0.013, 0.42])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("relative frequency")
    bar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    handles = [
        Line2D([0], [0], color=c, lw=2.6, label=OBS_LABEL[source] if k == "obs" else l)
        for k, l, c in DATASETS
    ]
    handles.append(Line2D([0], [0], color="black", lw=1.3,
                          label=f"observed {int(CONTOUR_FRACTION * 100)} % contour"))
    # The sixth cell of the 2x3 grid is free: the legend goes there rather than
    # under the panels, where it would sit on the bottom marginals.
    ax_legend = fig.add_subplot(outer[1, 2])
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, loc="center", ncol=1, fontsize=LEGEND_FONTSIZE + 3,
                     frameon=False, handlelength=2.6, labelspacing=1.1)
    name, span = PERIOD_TITLES[period]
    note = obs_note(source, PERIOD_IOPS[period])
    ax_legend.text(0.5, 0.11, name, transform=ax_legend.transAxes, ha="center", va="bottom",
                   fontsize=LEGEND_FONTSIZE + 5, fontweight="bold")
    ax_legend.text(0.5, 0.045, span, transform=ax_legend.transAxes, ha="center", va="bottom",
                   fontsize=LEGEND_FONTSIZE + 2)
    ax_legend.text(0.5, -0.09, note, transform=ax_legend.transAxes, ha="center", va="top",
                   fontsize=LEGEND_FONTSIZE, style="italic", color="#333333")
    fig.suptitle(
        r"log$_{10}$ N$_w$ against D$_m$, scheme moments truncated at D$_{min}$ = 0.312 mm",
        fontsize=19,
    )
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_txt(txt: Path, data: dict, iops: tuple[int, ...], source: str) -> None:
    dm_edges, lognw_edges = data["dm"], data["lognw"]
    dm_centers = 0.5 * (dm_edges[:-1] + dm_edges[1:])
    lognw_centers = 0.5 * (lognw_edges[:-1] + lognw_edges[1:])
    hist = data["hist"]
    seasons = "; ".join(
        f"IOP{i}: {OBS_WHEN[(source, i)]}, "
        + ("the IOP days themselves" if OBS_CACHE[(source, i)][1] else
           "a stand-in from the same season of the following year")
        for i in iops
    )
    lines = [
        "Joint distribution of log10 Nw and Dm, disdrometer and the four runs.",
        f"Model period: GoAmazon2014/5 {' and '.join(f'IOP{i}' for i in iops)}; "
        + ", ".join(f"IOP{i}: {GOAMAZON_IOPS[i][0]} to {GOAMAZON_IOPS[i][1]}" for i in iops),
        f"Observed period: {seasons}.",
        "The ARM laser disdrometer record starts on 25 Sep 2014: IOP2 is matched day for "
        "day, IOP1 has no observations of its own and borrows the same calendar season of "
        "2015. The MRR version instead observes both IOPs on the days of the runs, at one "
        "range gate about 800 m above Manacapuru UEA rather than at the surface; the "
        "spectra, QC and DSD parameters are otherwise identical (build_iop_mrr_dsd)."
        if source == "mrr" else
        "The ARM laser disdrometer record starts on 25 Sep 2014: IOP2 is matched day for "
        "day, IOP1 has no observations of its own and borrows the same calendar season of "
        "2015.",
        "Model samples: rainy cells of the bottom model level over the 4 km radar domain, "
        "hourly. Dm = M4/M3 and Nw from LWC and Dm, built from the scheme-native moments "
        "(1-moment Abel & Boutle 2012 with mu = 0 for C1M/G1M; ALARO variable-shape gamma "
        "for G2M and G2M-XCU) and truncated at Dmin = 0.312 mm with upper-incomplete-gamma "
        "factors, so the model sees the part of the spectrum the Parsivel sees.",
        f"Bins: {dm_centers.size} Dm x {lognw_centers.size} log10 Nw; the figure shades the "
        "relative frequency of each dataset on a shared log scale.",
        f"The black contour encloses {int(CONTOUR_FRACTION * 100)} % of the observed samples "
        f"(Gaussian-smoothed, sigma = {SMOOTH_BINS} bins).",
        "",
        "sample counts",
        "dataset,n_samples",
    ]
    for key, label, _ in DATASETS:
        lines.append(f"{label},{int(hist[key].sum())}")
    for key, label, _ in DATASETS:
        dm_pdf, lognw_pdf = marginals(hist[key], dm_edges, lognw_edges)
        lines += ["", f"{label} marginal of Dm", "dm_mm,pdf_per_mm"]
        lines += [f"{z:.4f},{v:.6g}" for z, v in zip(dm_centers, dm_pdf)]
        lines += ["", f"{label} marginal of log10 Nw", "log10_nw,pdf_per_decade"]
        lines += [f"{z:.4f},{v:.6g}" for z, v in zip(lognw_centers, lognw_pdf)]
    txt.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIRS["microphysics"])
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--periods", nargs="+", default=list(IOP_PERIODS), choices=list(IOP_PERIODS))
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--observations", choices=OBS_SOURCES, default="parsivel",
                   help="observed dataset in the first panel")
    return parse_configured_args(p, "make_paper_rain_dsd_joint_panel", argv=argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in args.periods:
        iops = IOP_PERIODS[key]
        data = load_histograms(args.cache_dir, iops, args.observations)
        suffix = "" if args.observations == "parsivel" else f"_{args.observations}"
        name = f"{FIGURE_STEM}{suffix}_{PERIOD_NAMES[key]}"
        png = args.output_dir / f"{name}_450dpi.png"
        txt = args.output_dir / f"{name}_data.txt"
        draw(data, png, args.dpi, args.observations, key)
        write_txt(txt, data, iops, args.observations)
        print(f"[saved] {png}\n[saved] {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
