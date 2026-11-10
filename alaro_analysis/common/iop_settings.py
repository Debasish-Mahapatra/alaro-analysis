"""Iop settings."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from alaro_analysis.common.experiments import DIAG_CONTROL_SHOWN
from alaro_analysis.common.paths import data_path, runs_path

ROOT = runs_path("")
CONT = ROOT / "ALARO"
IOP = ROOT / "iop-runs"
OUT = IOP / "figures"
CACHE = IOP / "analysis" / "cache"
LOG = IOP / "analysis" / "logs"
EXPS = ("NO3M", "graupel", "2mom")
LABEL = {"NO3M": "G2M-XCU", "graupel": "G1M", "2mom": "G2M"}
VARS = (
    "SURFPREC.EAU.GEC",
    "SURFPREC.EAU.CON",
    "SURFPREC.NEI.GEC",
    "SURFPREC.NEI.CON",
    "SURFPREC.GRA.GEC",
)
PERIODS = {1: (date(2014, 1, 15), date(2014, 2, 28)), 2: (date(2014, 7, 29), date(2014, 8, 31))}
# Converted IOP runs, the continuous control run and the IOP figure tree.
IOP_RUNS = ROOT / "IOP-runs"
CONTROL_RUN = ROOT / "amazon-control-run" / "control"
# Five runs with both controls in iop-run-figures-nodiaggrp; the paper's four runs, C1M without
# the diagnostic graupel (ALARO_IOP_CONTROLS=nodiag), in iop-run-figures-paper. The older
# four-run series with the diagnostic-graupel control stays in iop-run-figures.
FIGURES = data_path("iop-run-figures-nodiaggrp" if DIAG_CONTROL_SHOWN else "iop-run-figures-paper")
# GoAmazon2014/5 intensive operating periods, inclusive (Martin et al. 2016, ACP 16, 4785).
GOAMAZON_IOPS = {
    1: (date(2014, 2, 1), date(2014, 3, 31)),
    2: (date(2014, 8, 15), date(2014, 10, 15)),
}

# ------------------------------------------------------------ IOP analysis layout
# Caches, one folder per topic; the README in ANALYSIS maps them to figures.
ANALYSIS = IOP_RUNS / "analysis"
CACHE_DIRS = {
    "rainfall": ANALYSIS / "rainfall",
    "soundings": ANALYSIS / "soundings",
    "convection": ANALYSIS / "convection",
    "hydrometeors": ANALYSIS / "hydrometeors",
    "closure_budget": ANALYSIS / "closure_budget",
    "microphysics": ANALYSIS / "microphysics",
}
# Figure number -> (folder name, what it shows, cache topic or None, recipes).
# Recipes ask for their folder by NAME (``figure_files(root, "<name>", ...)``), so the
# order of the series lives here alone: after renumbering, ``sync_figure_folders()``
# renames the folders already on disk to match.
FIGURE_SERIES = {
    1: ("domain", "Model domain, SIPAM radar mask and observation sites (the paper's figure 1)", None,
        "none: image supplied by the user, no code in this repository"),
    2: ("spatial_bias_vs_radar", "Rain bias maps against the SIPAM radar", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    3: ("mean_diurnal_cycle", "Mean diurnal cycle of rain", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    4: ("rain_pdf", "Rain-rate distribution against the radar and IMERG, log-log (a) and log-linear (b)",
        "rainfall", "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    5: ("sounding_profiles", "Temperature, dewpoint and RH against the ARM MAO radiosondes",
        "soundings", "make_iop_sounding_profiles"),
    6: ("updraft_and_cloud_cover",
        "Diurnal height sections of updraft extent, convective upward mass flux (column split, "
        "convective 20 km boxes) and cloud cover, all five runs",
        "convection", "build_iop_updraft_partition, build_iop_updraft_cloud_partition, "
        "make_iop_draft_sections"),
    7: ("downdraft_and_vertical_velocity",
        "Diurnal height sections of downdraft extent, convective downward mass flux (column split, "
        "convective 20 km boxes) and resolved downward velocity, all five runs", "convection",
        "build_iop_convection_diurnal, build_iop_updraft_partition, make_iop_draft_sections"),
    8: ("hydrometeor_mass_fraction", "Hydrometeor mass-fraction histograms", "hydrometeors",
        "build_iop_hydrometeor_histograms, make_iop_hydrometeor_figures"),
    9: ("closure_budget_compression",
        "Closure budget compression from DDH, with the convective upward mass flux (column split, "
        "convective 20 km boxes), all five runs; day and night as well", "closure_budget",
        "build_iop_closure_budgets, build_iop_updraft_partition, make_iop_closure_budget_figures"),
    10: ("convective_mass_flux_vs_profiler",
         "Convective mass flux against the ARM radar wind profiler at T3 (Giangrande et al. 2016): "
         "upward and downward, the model as simulated (with the profiler's error in the data text)",
         "convection", "build_iop_rwp_mass_flux, build_iop_rwp_velocity, build_iop_rwp_model_sample, "
         "make_iop_rwp_mass_flux_comparison"),
    11: ("vertical_velocity_vs_profiler",
         "Vertical velocity against the ARM radar wind profiler at T3, one 2x3 figure per period: the model "
         "as the profiler sees it (strong up/downdrafts, tails), the time-averaged profiler against resolved w, "
         "and the scale plot below and above the melting layer",
         "convection", "build_iop_rwp_velocity, build_iop_rwp_model_sample, make_iop_rwp_velocity_figures"),
    12: ("raindrop_size_distribution",
         "Joint distribution of log10 Nw and Dm: disdrometer and micro rain radar against the five runs",
         "microphysics", "build_iop_disdrometer_seasons, build_iop_mrr_dsd, build_iop_microphysics_columns, "
         "make_paper_rain_dsd_joint_panel"),
    13: ("rain_evaporation",
         "Offline rain evaporation rate of the one- and two-moment schemes (ACEVMEL formulas): against "
         "the water-vapour content (a) and its sensitivity to the G2M rain number (b)", None,
         "make_iop_rain_evaporation"),
}
# Supplementary figures: S-number -> (folder name, what it shows, cache topic, recipes)
SUPPLEMENTARY_SERIES = {
    1: ("spatial_bias_vs_imerg", "Rain bias maps against IMERG", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    2: ("rmse_vs_radar", "Hourly rain RMSE maps against the SIPAM radar", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    3: ("rmse_vs_imerg", "Hourly rain RMSE maps against IMERG", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    4: ("diurnal_cycle_errors",
        "Errors of the diurnal cycle of rain against the radar and IMERG: hourly differences, MSE split "
        "into offset, amplitude and timing, 24-h harmonic, and the shift of the peak in every cell",
        "rainfall", "build_iop_rainfall_common_valid, make_iop_diurnal_cycle_errors"),
    5: ("rain_accumulation_roi", "Total rain per IOP over the whole ROI, IMERG and the five runs",
        "rainfall", "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    6: ("rain_bias_vs_imerg_roi", "Relative rain bias against IMERG per IOP over the whole ROI",
        "rainfall", "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    7: ("rain_pdf_roi_vs_imerg",
        "Rain-rate distribution over the whole ROI against IMERG, log-log (a) and log-linear (b)",
        "rainfall", "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    8: ("sounding_profiles_day_night",
        "Temperature, dewpoint and RH against the ARM MAO radiosondes, day and night launches apart",
        "soundings", "make_iop_sounding_profiles"),
    9: ("sounding_profiles_day_minus_night",
        "Day minus night of the model minus sonde differences, with a 95 % range from resampling days",
        "soundings", "make_iop_sounding_profiles"),
    10: ("updraft_downdraft_profiles", "Updraft and downdraft profiles", "convection",
         "build_iop_convection_diurnal, make_iop_convection_figures"),
    11: ("convective_flux_sampling",
         "G2M-XCU against G2M with the convective columns picked alike: share of boxes with convection, "
         "upward mass flux per convective box and over the whole domain, IOP1 and IOP2",
         "convection", "build_iop_updraft_partition (also --rain-only), make_iop_convective_flux_sampling"),
    12: ("closure_budget_updraft_partition",
        "Closure budget, IOP1 vs IOP2, with the convective upward mass flux (profiler sampling)",
        "closure_budget", "build_iop_closure_budgets, build_iop_updraft_partition, "
        "make_iop_updraft_partition_figures"),
    13: ("closure_budget_compression_day_night",
         "Closure budget compression from DDH by day (06-18 LT) and night (18-06 LT), with the "
         "convective upward mass flux of the same hours", "closure_budget",
         "build_iop_closure_budgets, build_iop_updraft_partition, make_iop_closure_budget_figures"),
    14: ("convective_mass_flux_vs_profiler_60km",
         "Convective mass flux against the ARM radar wind profiler at the supervisor's second scale: "
         "60 km boxes over three hourly snapshots against 3 h profiler windows", "convection",
         "build_iop_rwp_mass_flux, build_iop_rwp_velocity, build_iop_rwp_model_sample --box-km 60, "
         "make_iop_rwp_mass_flux_comparison"),
    15: ("fractions_skill_score", "Fractions skill score against the radar", "rainfall",
         "make_iop_fss"),
}
# Top of every height axis in the IOP figure series (0-15 km, user's choice 2026-10-21).
# Only the axis: data crops, integrals and the data text files keep their own ranges.
PLOT_TOP_KM = 15.0
# Period key -> IOPs it covers; the key goes in every figure file name.
IOP_PERIODS = {"combined": (1, 2), "iop1": (1,), "iop2": (2,)}


def figure_folder(root, figure, numbered=None) -> Path:
    """Folder of a figure by its folder name, whether it sits in the main or the
    supplementary series, or by its number in ``numbered`` (a series dict)."""
    name = numbered[figure][0] if isinstance(figure, int) else figure
    for number, (entry, *_) in FIGURE_SERIES.items():
        if entry == name:
            return Path(root) / f"{number:02d}_{name}"
    for number, (entry, *_) in SUPPLEMENTARY_SERIES.items():
        if entry == name:
            return Path(root) / "supplementary" / f"S{number:02d}_{name}"
    raise KeyError(f"no figure named {name!r} in either series")


def figure_files(root, figure, stem: str, period: str):
    """(temporary PNG, JPG, data text) of one figure in the IOP series.

    ``figure`` is the folder name (main or supplementary, wherever the series put it)
    or its number in ``FIGURE_SERIES``. Layout:
    ``<root>/<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the plotted data in
    ``<root>/<NN>_<name>/data/<stem>_<period>.txt``.
    """
    folder = figure_folder(root, figure, FIGURE_SERIES)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{period}"
    return (
        folder / f".{name}.render.png",
        folder / f"{name}_450dpi.jpg",
        folder / "data" / f"{name}.txt",
    )


def supplementary_files(root, figure, stem: str, period: str):
    """(temporary PNG, JPG, data text) of one supplementary figure.

    ``figure`` is the folder name (main or supplementary, wherever the series put it)
    or its number in ``SUPPLEMENTARY_SERIES``. Layout:
    ``<root>/supplementary/S<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the plotted
    data in ``.../data/<stem>_<period>.txt``.
    """
    folder = figure_folder(root, figure, SUPPLEMENTARY_SERIES)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{period}"
    return (
        folder / f".{name}.render.png",
        folder / f"{name}_450dpi.jpg",
        folder / "data" / f"{name}.txt",
    )


def sync_figure_folders(root=FIGURES) -> list[tuple[Path, Path]]:
    """Rename figure folders on disk to the numbers ``FIGURE_SERIES`` and
    ``SUPPLEMENTARY_SERIES`` give them, also between the main and supplementary series.

    A folder is matched by the name after its number. Returns the (old, new) moves.
    """
    root = Path(root)
    wanted = {name: root / f"{n:02d}_{name}" for n, (name, *_) in FIGURE_SERIES.items()}
    wanted.update({name: root / "supplementary" / f"S{n:02d}_{name}"
                   for n, (name, *_) in SUPPLEMENTARY_SERIES.items()})
    pattern = re.compile(r"^S?\d\d_(.+)$")
    found = []
    for parent in (root, root / "supplementary"):
        for folder in sorted(parent.iterdir()) if parent.exists() else []:
            match = pattern.match(folder.name)
            if folder.is_dir() and match and match.group(1) in wanted and folder != wanted[match.group(1)]:
                found.append((folder, wanted[match.group(1)]))
    staged = []
    for folder, target in found:  # two steps, so a shift of the whole series cannot collide
        temporary = folder.with_name(f".renumbering_{folder.name}")
        folder.rename(temporary)
        staged.append((folder, temporary, target))
    for folder, temporary, target in staged:
        if target.exists():
            raise FileExistsError(f"{target} exists; {folder} left at {temporary}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(target)
    return [(folder, target) for folder, _, target in staged]


def write_figure_index(root=FIGURES):
    """Write ``README.txt`` in ``root``: every figure folder, its files and its cache."""
    lines = [
        "GoAmazon IOP figure series",
        "==========================",
        "One folder per figure: <NN>_<name>/<stem>_<period>_450dpi.jpg, plotted data in",
        "<NN>_<name>/data/<stem>_<period>.txt. Periods: combined = both IOPs, iop1 =",
        "1 Feb-31 Mar 2014, iop2 = 15 Aug-15 Oct 2014. Model-grid figures use the native 4 km",
        "radar domain (ALARO-RUNS/mask/Radar_mask_4km_native.nc); only the rain figures against",
        "the radar and IMERG (bias and RMSE maps, diurnal cycle, PDFs) and the fractions skill",
        "score use the 0.1 deg radar mask; the *_roi supplementary figures cover the whole ROI",
        "(-67..-53 E, -10..4 N) against IMERG with no radar mask.",
        f"Caches: {ANALYSIS}/<topic>/ (see the README there).",
        "",
    ]
    for number, (name, what, topic, recipes) in FIGURE_SERIES.items():
        folder = root / f"{number:02d}_{name}"
        files = sorted(p.name for p in folder.glob("*_450dpi.jpg")) if folder.exists() else []
        lines.append(f"{number:02d}_{name}  {what}")
        cache = f"analysis/{CACHE_DIRS[topic].name}/" if topic else "none"
        lines.append(f"    cache: {cache}   recipes: {recipes}")
        lines.extend(f"    {f}" for f in files) if files else lines.append("    (not built)")
        lines.append("")
    lines += ["Supplementary figures: supplementary/S<NN>_<name>/", ""]
    for number, (name, what, topic, recipes) in SUPPLEMENTARY_SERIES.items():
        folder = root / "supplementary" / f"S{number:02d}_{name}"
        files = sorted(p.name for p in folder.glob("*_450dpi.jpg")) if folder.exists() else []
        lines.append(f"S{number:02d}_{name}  {what}")
        lines.append(f"    cache: analysis/{CACHE_DIRS[topic].name}/   recipes: {recipes}")
        lines.extend(f"    {f}" for f in files) if files else lines.append("    (not built)")
        lines.append("")
    out = root / "README.txt"
    out.write_text("\n".join(lines))
    return out


G = 9.80665
OFFSET = -4
NLEV = 87
WORKERS = 32
FH = re.compile("\\+(\\d{4})$")
