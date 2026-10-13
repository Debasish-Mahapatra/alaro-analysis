"""Iop settings."""

from __future__ import annotations

import re
from datetime import date

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
FIGURES = data_path("iop-run-figures")
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
# Figure number -> (folder name, what it shows, cache topic, recipes)
FIGURE_SERIES = {
    1: ("spatial_bias_vs_radar", "Rain bias maps against the SIPAM radar", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    2: ("spatial_bias_vs_imerg", "Rain bias maps against IMERG", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    3: ("mean_diurnal_cycle", "Mean diurnal cycle of rain", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    4: ("rain_pdf_loglog", "Rain-rate distribution, log-log", "rainfall",
        "build_iop_rainfall_common_valid, make_iop_rainfall_verification"),
    5: ("rain_pdf_log_linear", "Rain-rate distribution, log-linear (radar box; whole ROI vs IMERG)",
        "rainfall", "build_iop_rainfall_common_valid, make_iop_rainfall_verification; "
        "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    6: ("sounding_profiles", "Temperature, dewpoint and RH against the ARM MAO radiosondes",
        "soundings", "make_iop_sounding_profiles"),
    7: ("updraft_downdraft_profiles", "Updraft and downdraft profiles", "convection",
        "build_iop_convection_diurnal, make_iop_convection_figures"),
    8: ("updraft_and_cloud_cover", "Diurnal height sections of updraft and cloud cover",
        "convection", "build_iop_convection_diurnal, make_iop_convection_figures"),
    9: ("downdraft_and_vertical_velocity",
        "Diurnal height sections of downdraft and resolved vertical velocity", "convection",
        "build_iop_convection_diurnal, make_iop_convection_figures"),
    10: ("hydrometeor_mass_fraction", "Hydrometeor mass-fraction histograms", "hydrometeors",
         "build_iop_hydrometeor_histograms, make_iop_hydrometeor_figures"),
    11: ("closure_budget_compression", "Closure budget compression from DDH", "closure_budget",
         "build_iop_closure_budgets, make_iop_closure_budget_figures"),
    12: ("fractions_skill_score", "Fractions skill score against the radar", "rainfall",
         "make_iop_fss"),
}
# Supplementary figures: S-number -> (folder name, what it shows, cache topic, recipes)
SUPPLEMENTARY_SERIES = {
    1: ("rain_accumulation_roi", "Total rain per IOP over the whole ROI, IMERG and the four runs",
        "rainfall", "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    2: ("rain_bias_vs_imerg_roi", "Relative rain bias against IMERG per IOP over the whole ROI",
        "rainfall", "build_iop_rainfall_roi, make_iop_rain_pdf_roi"),
    3: ("closure_budget_updraft_partition",
        "Closure budget, IOP1 vs IOP2, with the convective upward mass flux (profiler sampling)",
        "closure_budget", "build_iop_closure_budgets, build_iop_updraft_partition, "
        "make_iop_updraft_partition_figures"),
    4: ("microphysics_story",
        "Microphysics against SIPAM reflectivity, the disdrometer and the ARM W-band cloud radar: vertical "
        "structure, convective-stratiform split, ice aloft, drop sizes and the precipitation pathway (STORY.md)",
        "microphysics", "build_iop_microphysics_fields, build_iop_sipam_reflectivity, "
        "build_iop_echo_statistics, build_iop_microphysics_columns, build_iop_disdrometer_seasons, "
        "build_iop_wacr_occurrence, build_iop_water_budget, make_iop_microphysics_story"),
}
# Period key -> IOPs it covers; the key goes in every figure file name.
IOP_PERIODS = {"combined": (1, 2), "iop1": (1,), "iop2": (2,)}


def figure_files(root, number: int, stem: str, period: str):
    """(temporary PNG, JPG, data text) of one figure in the IOP series.

    Layout: ``<root>/<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the plotted data
    in ``<root>/<NN>_<name>/data/<stem>_<period>.txt``.
    """
    folder = root / f"{number:02d}_{FIGURE_SERIES[number][0]}"
    (folder / "data").mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{period}"
    return (
        folder / f".{name}.render.png",
        folder / f"{name}_450dpi.jpg",
        folder / "data" / f"{name}.txt",
    )


def supplementary_files(root, number: int, stem: str, period: str):
    """(temporary PNG, JPG, data text) of one supplementary figure.

    Layout: ``<root>/supplementary/S<NN>_<name>/<stem>_<period>_450dpi.jpg`` with the
    plotted data in ``.../data/<stem>_<period>.txt``.
    """
    folder = root / "supplementary" / f"S{number:02d}_{SUPPLEMENTARY_SERIES[number][0]}"
    (folder / "data").mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{period}"
    return (
        folder / f".{name}.render.png",
        folder / f"{name}_450dpi.jpg",
        folder / "data" / f"{name}.txt",
    )


def write_figure_index(root=FIGURES):
    """Write ``README.txt`` in ``root``: every figure folder, its files and its cache."""
    lines = [
        "GoAmazon IOP figure series",
        "==========================",
        "One folder per figure: <NN>_<name>/<stem>_<period>_450dpi.jpg, plotted data in",
        "<NN>_<name>/data/<stem>_<period>.txt. Periods: combined = both IOPs, iop1 =",
        "1 Feb-31 Mar 2014, iop2 = 15 Aug-15 Oct 2014. Model-grid figures (6-11) use the",
        "native 4 km radar domain (ALARO-RUNS/mask/Radar_mask_4km_native.nc); only the rain",
        "figures (1-5, 12) use the 0.1 deg radar mask, except the *_roi_vs_imerg variant in 05,",
        "which covers the whole ROI (-67..-53 E, -10..4 N) against IMERG with no radar mask.",
        f"Caches: {ANALYSIS}/<topic>/ (see the README there).",
        "",
    ]
    for number, (name, what, topic, recipes) in FIGURE_SERIES.items():
        folder = root / f"{number:02d}_{name}"
        files = sorted(p.name for p in folder.glob("*_450dpi.jpg")) if folder.exists() else []
        lines.append(f"{number:02d}_{name}  {what}")
        lines.append(f"    cache: analysis/{CACHE_DIRS[topic].name}/   recipes: {recipes}")
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
