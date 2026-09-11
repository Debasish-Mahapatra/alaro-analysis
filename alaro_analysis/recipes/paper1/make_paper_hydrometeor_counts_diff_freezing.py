#!/usr/bin/env python3
"""Hydrometeor mass-fraction histograms, RAW COUNTS -- no normalisation at all.

Identical to make_paper_hydrometeor_cfad_diff_freezing.py in every respect
except one: the per-level "divide by the level total and multiply by 100"
step is removed.  Each cell is the plain number of grid-point-hours whose mass
fraction fell in that bin at that model level, summed over all 17544 hourly
files and every grid point of the radar domain.

That makes this NOT a CFAD.  A CFAD normalises each height to 100 %, which
shows the shape of the distribution but hides how much and how often; here the
colour is the raw occurrence count, so heights and species are compared on one
absolute footing.  All three experiments sample exactly the same 17544 files,
so the counts and their differences are directly comparable.

Columns: C1M (absolute counts), G1M - C1M (difference), G2M - G1M (difference).
Rows (top -> bottom): graupel, snow, cloud ice, cloud water, rain.

Absolute panels use the WhiteBlueGreenYellowRed log colour scale; difference
panels use a diverging RdBu_r scale symmetric about zero at the 98th percentile
of |difference|.  A per-level median mass-fraction line is drawn on the absolute
panels -- that line is a property of the distribution and is unchanged by
dropping the normalisation.  Empty bins are left blank rather than drawn as
zero.  Each column carries the 0 degC (freezing)
level of its model -- C1M, G1M, G2M respectively -- as a horizontal dashed line
labelled with its height in km, computed from the 2-year mean temperature
profile (same caches the diurnal plots use).

Graupel is not a prognostic species in the 2-ice control, so its C1M panel is
"no data" and its middle column shows the G1M absolute CFAD instead of C1M-G1M.

CFAD counts come from the (from-raw) histogram caches built by
build_hydrometeor_cfad_from_netcdf.py; freezing levels from the 2-year
temperature/geopotential profile caches.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from alaro_analysis.common.cli_config import parse_configured_args
from alaro_analysis.common.paths import data_path, runs_path
from alaro_analysis.common.vertical import centers_to_edges

CACHE = runs_path("processed-data/paper6_hydrometeor_cfad")
TEMP = runs_path("processed-data/temperature/2years")
GEO = runs_path("processed-data/geopotential/2years")
OUTPUT_DIR = data_path("microphysics-paper/6. hydrometeor mixing-ratio cfad")
FIGURE_NAME = "6. hydrometeor mass-fraction counts_c1m_g1m-c1m_g2m-g1m_freezing_450dpi.png"
TEXT_NAME = "6. hydrometeor mass-fraction counts_c1m_g1m-c1m_g2m-g1m_freezing_data.txt"

MAX_HEIGHT_KM = 20.0
# Raw-count colour scale.  Counts run 1..4.9e6 over the two years; 10 is the
# floor below which a bin is a handful of grid-point-hours and means nothing.
COUNT_VMIN, COUNT_VMAX = 1e1, 5e6
DIFF_PERCENTILE = 98.0  # difference scale = this percentile of |diff|

# --percent rescales every cell by ONE global constant: the number of samples
# behind each model level (files x radar-mask grid points).  That is a genuine
# frequency in %, but unlike a CFAD it uses the same denominator everywhere, so
# it does not rescale each height to 100 % -- the picture is identical to the
# raw counts and only the numbers on the colourbar change.
N_FILES = 17544
N_POINTS = 2822
SAMPLES_PER_LEVEL = N_FILES * N_POINTS
FREEZING_K = 273.15
# Minimum mass fraction counted as "hydrometeor present". Values below this are
# numerical residue (advection/sedimentation remnants of 1e-11..1e-9 kg/kg that
# occur in <0.1% of points near the surface); with the per-level 100%
# normalisation they would otherwise paint saturated rows where no physically
# meaningful condensate exists. Bins below the threshold are dropped and each
# level is renormalised over the remaining bins.
MIN_MASS_FRACTION = 1e-8

# Rows: (symbol, counts-species, full name) top -> bottom
ROWS = [
    (r"$q_\mathrm{g}$", "GRAUPEL", "Graupel"),
    (r"$q_\mathrm{s}$", "SNOW", "Snow"),
    (r"$q_\mathrm{i}$", "SOLID_WATER", "Cloud ice"),
    (r"$q_\mathrm{c}$", "LIQUID_WATER", "Cloud water"),
    (r"$q_\mathrm{r}$", "RAIN", "Rain"),
]
EXP_LABEL = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
# Column -> model whose freezing (0 degC) level is drawn on it: C1M, G1M, G2M.
COL_FREEZE_EXP = ("control", "graupel", "2mom")

# Fonts.  Everything is bold: rcParams below make regular text AND mathtext
# (the 10^n tick labels, the q symbols) render bold, since fontweight="bold"
# alone is ignored inside $...$.
TICK_FS, AXIS_FS, TITLE_FS = 17, 20, 23
CBAR_FS, CBAR_TICK_FS = 20, 15
SYM_FS, NAME_FS, LEG_FS = 30, 19, 15
LETTER_FS = 28  # (a)-(o) panel tags, bottom-right of each panel
FREEZE_COLOR = "black"

plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.default": "bf",
    }
)


def sci_tick(v, _pos=None) -> str:
    """Tick label as m x 10^n, e.g. 7.5x10^5.  Small numbers stay plain."""
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e4:  # readable as-is (percent mode)
        return f"{v:g}"
    exp = int(np.floor(np.log10(a)))
    mant = v / 10.0**exp
    m = f"{mant:.1f}".rstrip("0").rstrip(".")
    if m == "1":
        return rf"$10^{{{exp}}}$"
    if m == "-1":
        return rf"$-10^{{{exp}}}$"
    return rf"${m}{{\times}}10^{{{exp}}}$"


def load_exp(exp: str) -> dict:
    return dict(np.load(CACHE / f"{exp}_cfad.npz"))


def counts_and_median(counts: np.ndarray, log_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw per-level counts (NO normalisation), and per-level median mass fraction.

    The CFAD version divided each row by its own total and multiplied by 100.
    That step is deliberately absent: the array is returned exactly as counted.
    The median is a percentile of the distribution, so it is identical either
    way -- normalising a row cannot move its 50th percentile.
    """
    counts = counts.astype(np.float64)
    centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    cum = np.cumsum(counts, axis=1)
    med = np.full(counts.shape[0], np.nan)
    for lev in range(counts.shape[0]):
        if cum[lev, -1] <= 0:
            continue
        frac = cum[lev] / cum[lev, -1]
        med[lev] = centers[min(int(np.searchsorted(frac, 0.5)), centers.size - 1)]
    return counts, med


def freezing_level_km(exp: str) -> float:
    """Height (km) of the 0 degC level from the 2-year mean temperature profile."""
    t = dict(np.load(TEMP / f"{exp}_full-domain_diurnal_profile.npz", allow_pickle=True))
    g = dict(np.load(GEO / f"{exp}_full-domain_height_profile_first.npz", allow_pickle=True))
    mean = np.asarray(t["mean"], dtype=np.float64)  # (L, 24) Kelvin
    cnt = np.asarray(t["counts"], dtype=np.float64)  # (L, 24)
    w = cnt.sum(axis=1)
    tprof = np.where(w > 0, (mean * cnt).sum(axis=1) / np.maximum(w, 1.0), np.nan)
    h_km = np.asarray(g["height_m"], dtype=np.float64) / 1000.0
    order = np.argsort(h_km)
    z, tt = h_km[order], tprof[order]
    fin = np.isfinite(z) & np.isfinite(tt)
    z, tt = z[fin], tt[fin]
    # First crossing of 273.15 K scanning upward from the surface.
    for i in range(z.size - 1):
        a, b = tt[i] - FREEZING_K, tt[i + 1] - FREEZING_K
        if a == 0.0:
            return float(z[i])
        if a * b < 0.0:
            return float(z[i] + (z[i + 1] - z[i]) * (FREEZING_K - tt[i]) / (tt[i + 1] - tt[i]))
    return float("nan")


def main(argv: list[str] | None = None) -> None:
    import cmaps

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--percent",
        action="store_true",
        help="divide every cell by the same global sample count and show %% of "
        "sampled grid-point-hours instead of raw counts (still no per-level "
        "normalisation)",
    )
    args = parse_configured_args(ap, "make_paper_hydrometeor_counts_diff_freezing", argv=argv)
    scale = 100.0 / SAMPLES_PER_LEVEL if args.percent else 1.0
    unit = "Frequency (% of samples)" if args.percent else "Count"
    figure_name, text_name = FIGURE_NAME, TEXT_NAME
    # Raw counts are the primary figure 6; the % variant is a companion in backup/.
    out_dir = OUTPUT_DIR
    if args.percent:
        figure_name = figure_name.replace("counts_", "percent-of-samples_")
        text_name = text_name.replace("counts_", "percent-of-samples_")
        out_dir = OUTPUT_DIR / "backup"
        print(
            f"[units] % of {SAMPLES_PER_LEVEL:,} samples per level "
            f"({N_FILES:,} files x {N_POINTS:,} points)"
        )

    data = {e: load_exp(e) for e in ("control", "graupel", "2mom")}
    log_edges_full = np.asarray(data["control"]["log_edges"], dtype=np.float64)
    dbin = log_edges_full[1] - log_edges_full[0]
    i0 = int(round((np.log10(MIN_MASS_FRACTION) - log_edges_full[0]) / dbin))
    assert abs(log_edges_full[i0] - np.log10(MIN_MASS_FRACTION)) < 1e-6, (
        "MIN_MASS_FRACTION must fall on a histogram bin edge"
    )
    log_edges = log_edges_full[i0:]
    q_edges = 10.0**log_edges

    # Common vertical grid from control geopotential; ascending, cropped to 20 km.
    h = np.asarray(data["control"]["height_km"], dtype=np.float64)
    order = np.argsort(h)
    h_sorted = h[order]
    keep = np.isfinite(h_sorted) & (h_sorted >= 0.0) & (h_sorted <= MAX_HEIGHT_KM)
    y = h_sorted[keep]
    y_edges = centers_to_edges(y)

    cnt, med = {}, {}
    for e in ("control", "graupel", "2mom"):
        for _, sp, _ in ROWS:
            key = f"counts_{sp}"
            if key in data[e]:
                c, m = counts_and_median(data[e][key][:, i0:], log_edges)
                c = c * scale
                cnt[(e, sp)] = c
                med[(e, sp)] = m

    frz = {e: freezing_level_km(e) for e in ("control", "graupel", "2mom")}

    def crop(a):
        return a[order][keep]

    freq_cmap = cmaps.WhiteBlueGreenYellowRed
    freq_norm = mcolors.LogNorm(vmin=COUNT_VMIN * scale, vmax=COUNT_VMAX * scale)

    # Symmetric difference scale from the differences actually plotted.
    pool = []
    for _, sp, _ in ROWS:
        if (("control", sp) in cnt) and (("graupel", sp) in cnt):
            pool.append(crop(cnt[("graupel", sp)]) - crop(cnt[("control", sp)]))
        if (("graupel", sp) in cnt) and (("2mom", sp) in cnt):
            pool.append(crop(cnt[("2mom", sp)]) - crop(cnt[("graupel", sp)]))
    flat = np.abs(np.concatenate([d.ravel() for d in pool]))
    diff_scale = float(np.percentile(flat[np.isfinite(flat)], DIFF_PERCENTILE)) or 1.0
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_scale, vcenter=0.0, vmax=diff_scale)
    unit_word = "%" if args.percent else "counts"
    scale_txt = f"{diff_scale:.4g}" if args.percent else f"{diff_scale:,.0f}"
    print(f"[diff scale] +/-{scale_txt} {unit_word} (p{DIFF_PERCENTILE:g} of |difference|)")

    # Only rows 0 and 1 carry titles, so only the q_g -> q_s gap needs to be
    # wide.  A zero-height spacer row gives that one gap a double hspace while
    # q_s/q_i/q_c/q_r stay tight, and the reclaimed height goes into the panels.
    fig = plt.figure(figsize=(23.0, 30.0))
    gs = fig.add_gridspec(
        len(ROWS) + 1,
        3,
        height_ratios=[1.0, 0.001] + [1.0] * (len(ROWS) - 1),
        left=0.115,
        right=0.980,
        top=0.975,
        bottom=0.038,
        hspace=0.16,
        wspace=0.44,
    )
    grid_row = [0] + list(range(2, len(ROWS) + 1))
    axes = np.empty((len(ROWS), 3), dtype=object)
    for _r in range(len(ROWS)):
        for _c in range(3):
            axes[_r, _c] = fig.add_subplot(gs[grid_row[_r], _c])
    # Explicit margins: a wide LEFT margin holds the species symbol AND the
    # "Height (km)" label with clear space between them.

    letters = iter("abcdefghijklmnopqrstuvwxyz")

    for r, (sym, sp, name) in enumerate(ROWS):
        if sp == "GRAUPEL":
            plan = [
                ("nodata", None, "C1M", None),
                ("abs", ("graupel", sp), "G1M", "graupel"),
                ("diff", ("2mom", "graupel", sp), "G2M − G1M", "2mom"),
            ]
        else:
            plan = [
                ("abs", ("control", sp), "C1M", "control"),
                ("diff", ("graupel", "control", sp), "G1M − C1M", "graupel"),
                ("diff", ("2mom", "graupel", sp), "G2M − G1M", "2mom"),
            ]

        for c, (kind, payload, title, med_exp) in enumerate(plan):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xlim(q_edges[0], q_edges[-1])
            ax.set_ylim(0.0, MAX_HEIGHT_KM)
            # Taller panels make matplotlib choose 2.5 km steps; 5 km is cleaner.
            ax.set_yticks(np.arange(0.0, MAX_HEIGHT_KM + 0.1, 5.0))
            ax.tick_params(axis="both", labelsize=TICK_FS)
            # Row 1 (graupel) and row 2 (snow) set the two title patterns; rows
            # 3-5 repeat row 2, so titling them again is redundant.
            if r <= 1:
                ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"Mass fraction (kg kg$^{-1}$)", fontsize=AXIS_FS, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Height (km)", fontsize=AXIS_FS, fontweight="bold")

            # Panel tag, bottom-right, on a white patch so it stays readable
            # where the data reaches the corner.  The "no data" panel is tagged
            # too, so the grid reads a,b,c / d,e,f / ... without gaps.
            ax.text(
                0.975,
                0.03,
                f"({next(letters)})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=LETTER_FS,
                fontweight="bold",
                zorder=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0},
            )

            handles = []
            if kind == "nodata":
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="0.5",
                )
                continue
            elif kind == "abs":
                e, s = payload
                # Zero is a real value but LogNorm cannot draw it; leave it blank.
                field = np.ma.masked_less_equal(crop(cnt[(e, s)]), 0.0)
                pcm = ax.pcolormesh(
                    q_edges, y_edges, field, cmap=freq_cmap, norm=freq_norm, shading="flat"
                )
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(pcm, cax=cax)
                cb.set_label(unit, fontsize=CBAR_FS, fontweight="bold")
                cb.ax.tick_params(labelsize=CBAR_TICK_FS)
                cb.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            else:
                a, b, s = payload
                A, B = crop(cnt[(a, s)]), crop(cnt[(b, s)])
                # A zero count is information (the species is absent there), so
                # it must enter the subtraction; blank only where BOTH are empty.
                diff = np.ma.masked_where((A <= 0) & (B <= 0), A - B)
                pcm = ax.pcolormesh(
                    q_edges, y_edges, diff, cmap="RdBu_r", norm=diff_norm, shading="flat"
                )
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(pcm, cax=cax)
                cb.set_label(rf"$\Delta$ {unit}", fontsize=CBAR_FS, fontweight="bold")
                cb.ax.tick_params(labelsize=CBAR_TICK_FS)
                cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_tick))

            if med_exp is not None:
                m = crop(med[(med_exp, sp)])
                (line,) = ax.plot(m, y, color="black", lw=2.4, label="Median", zorder=6)
                handles.append(line)

            fexp = COL_FREEZE_EXP[c]
            zf = frz[fexp]
            if np.isfinite(zf):
                fl = ax.axhline(
                    zf, color=FREEZE_COLOR, lw=2.4, ls="--", zorder=7, label="0 °C level"
                )
                handles.append(fl)

            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    framealpha=0.9,
                    prop={"size": LEG_FS, "weight": "bold"},
                )

    # Species symbol only, in the wide left margin, well clear of "Height (km)".
    for r, (sym, _sp, _name) in enumerate(ROWS):
        pos = axes[r, 0].get_position()
        yc = 0.5 * (pos.y0 + pos.y1)
        fig.text(
            0.038,
            yc,
            sym,
            rotation=90,
            va="center",
            ha="center",
            fontsize=SYM_FS,
            fontweight="bold",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / figure_name
    fig.savefig(out, dpi=450, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    txt = out_dir / text_name
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("Hydrometeor mass-fraction histograms, RAW COUNTS (no normalisation)\n")
        fh.write("=" * 66 + "\n")
        fh.write(f"Figure: {out}\n\n")
        fh.write("NO NORMALISATION IS APPLIED. Each cell is the plain number of\n")
        fh.write("grid-point-hours whose mass fraction fell in that bin at that model\n")
        fh.write("level, summed over every hourly file and every radar-domain grid point.\n")
        fh.write("This is therefore NOT a CFAD: heights are NOT rescaled to 100 %, so a\n")
        fh.write("level that is rarely occupied stays faint instead of being stretched to\n")
        fh.write("full scale. Counts are comparable across heights, species and models\n")
        fh.write("because all three experiments sample the same number of files.\n\n")
        fh.write("Columns: C1M (absolute counts), G1M - C1M, G2M - G1M.\n")
        fh.write("Rows: graupel, snow, cloud ice, cloud water, rain.\n")
        fh.write("Top row only: C1M (no data), G1M (absolute), G2M - G1M.\n")
        if args.percent:
            fh.write(f"Units: % of the {SAMPLES_PER_LEVEL:,} samples behind each level\n")
            fh.write(f"  ({N_FILES:,} hourly files x {N_POINTS:,} radar-mask grid points).\n")
            fh.write("  ONE global constant, identical for every level, species and model,\n")
            fh.write("  so this is a frequency WITHOUT the per-level rescaling of a CFAD.\n")
        fh.write(
            "Absolute: WhiteBlueGreenYellowRed LogNorm "
            f"{COUNT_VMIN * scale:g}..{COUNT_VMAX * scale:g} {'%' if args.percent else 'counts'}.\n"
        )
        fh.write(
            f"Difference: RdBu_r +/-{scale_txt} {unit_word} "
            f"(p{DIFF_PERCENTILE:g} of |difference|).\n"
        )
        fh.write("Empty bins are left blank; a zero on one side of a difference is kept.\n")
        fh.write(f"Threshold: only mass fractions >= {MIN_MASS_FRACTION:g} kg/kg are counted\n")
        fh.write("  (bins below dropped; sub-threshold values are numerical residue).\n")
        fh.write("Black SOLID line = per-level median of that column's RAW mass fraction\n")
        fh.write("  (C1M / G1M / G2M for columns 1 / 2 / 3) -- NOT a median of the differences.\n")
        fh.write("Black DASHED line = 0 degC (freezing) level of that column's model.\n\n")
        fh.write("Freezing (0 degC) level from the 2-year mean temperature profile (km):\n")
        for e in ("control", "graupel", "2mom"):
            fh.write(f"  {EXP_LABEL[e]} ({e}): {frz[e]:.2f} km\n")
        fh.write(f"\nn_files per experiment: { {e: int(data[e]['n_files']) for e in data} }\n")
    print(f"[saved] {txt}")
    print("[freezing km]", {EXP_LABEL[e]: round(frz[e], 2) for e in frz})


if __name__ == "__main__":
    main()
