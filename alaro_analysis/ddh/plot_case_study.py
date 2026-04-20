"""Per-day case study: find a day where the negative-correction term is
prominent in the +0024 budget, and plot the +0024 minus +0012 difference for
each budget component of QV, QL and QI.

Strategy
--------
1. For every day in every experiment, measure the prominence of the 'neg'
   component in the +0024 budget as max over altitude of |rate| summed across
   QV + QL + QI.  This picks days where the negative correction is large and
   therefore visible.
2. Rank days by the cross-experiment minimum of that metric so we pick a day
   that is prominent in all three experiments (not just one).  That makes the
   three-panel comparison meaningful.
3. For that day, read +0024 and +0012 .dta files directly and plot the
   difference (+0024 minus +0012) for every budget component.

Inputs:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead{0024,0012}_VZ/
      {exp}/DDH20YYMMDD/{VAR}/*.dta
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/_aggregated/
      temperature_{exp}.npz

Outputs:
  /mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/DDH-figures/neg/
      case_YYYYMMDD_diff_0024_minus_0012.png
      case_YYYYMMDD_report.txt        ranking and chosen day
"""
from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROCESSED_BASE = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed"
)
AGG_DIR = PROCESSED_BASE / "_aggregated"
FIG_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures/DDH-figures/neg"
)

EXPERIMENTS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
EXP_COLORS = {"control": "#d62728", "graupel": "#1f77b4", "2mom": "#2ca02c"}
SPECIES_DEFAULT = ("QV", "QL", "QI")
SPECIES = SPECIES_DEFAULT
T_FREEZE_K = 273.15
Z_MAX_KM = 20.0
N_WORKERS = 32

BLOCK_COLORS = {
    "cond-cv": "#1f77b4", "cond-rs": "#4e79b9",
    "evap-cv": "#d62728", "evap-rs": "#d4a6a7",
    "auto-cv": "#2ca02c", "auto-rs": "#83d083",
    "prec-cv": "#9467bd", "prec-rs": "#bfa3d2",
    "turdiff": "#17becf", "turconv": "#6ec6ce",
    "dynam":   "#ff7f0e",
    "neg":     "#000000",
    "TQVRESIDUAL": "#bcbd22", "TQLRESIDUAL": "#bcbd22",
    "TQIRESIDUAL": "#bcbd22", "TQNRESIDUAL": "#bcbd22",
    "TQRRESIDUAL": "#bcbd22", "TQSRESIDUAL": "#bcbd22",
    "TQGRESIDUAL": "#bcbd22",
    "TQLCOMPSUM": "#888888", "TQICOMPSUM": "#888888",
    "TQNCOMPSUM": "#888888", "TQRCOMPSUM": "#888888",
    "TQSCOMPSUM": "#888888", "TQGCOMPSUM": "#888888",
}

BLOCK_LABELS = {
    "cond-cv":   "condensation (convective)",
    "cond-rs":   "condensation (resolved)",
    "evap-cv":   "evaporation (convective)",
    "evap-rs":   "evaporation (resolved)",
    "auto-cv":   "autoconversion (convective)",
    "auto-rs":   "autoconversion (resolved)",
    "prec-cv":   "precipitation flux (convective)",
    "prec-rs":   "precipitation flux (resolved)",
    "turdiff":   "turbulent diffusion",
    "turconv":   "turbulent convection",
    "dynam":     "dynamics",
    "neg":       "negative correction",
}

def _pretty_label(block: str) -> str:
    if block in BLOCK_LABELS:
        return BLOCK_LABELS[block]
    if block.endswith("RESIDUAL"):
        return "residual"
    if block.endswith("COMPSUM"):
        return "sum of components"
    return block

_DTA_RE = re.compile(r"^([A-Z]+)\.DHFDLABOF\+\d+\.([^.]+)\.dta$")


def load_day(exp: str, day: str, lead: str) -> dict[str, dict[str, np.ndarray]]:
    """Return species -> block -> profile for one experiment, one day.

    Each profile is a 2-column array; column 0 altitude_km, column 1 value.
    """
    root = PROCESSED_BASE / f"lead{lead}_VZ" / exp / day
    out: dict[str, dict[str, np.ndarray]] = {}
    if not root.exists():
        return out
    for var_dir in root.iterdir():
        if not var_dir.is_dir():
            continue
        var = var_dir.name
        if var not in SPECIES:
            continue
        blocks: dict[str, np.ndarray] = {}
        alt: np.ndarray | None = None
        for f in var_dir.iterdir():
            m = _DTA_RE.match(f.name)
            if not m:
                continue
            block = m.group(2)
            try:
                arr = np.loadtxt(f)
            except Exception:
                continue
            if alt is None:
                alt = arr[:, 0]
            blocks[block] = arr[:, 1]
        if blocks and alt is not None:
            blocks["__altitude_km__"] = alt
            out[var] = blocks
    return out


def _prominence_score(args):
    exp, day = args
    day_data = load_day(exp, day, "0024")
    score = 0.0
    for var in SPECIES:
        v = day_data.get(var)
        if v is None or "neg" not in v:
            continue
        alt = v["__altitude_km__"]
        mask = (alt >= 0) & (alt <= Z_MAX_KM)
        score += float(np.nanmax(np.abs(v["neg"][mask])))
    return (exp, day, score)


def select_case(days_per_exp: dict[str, list[str]]) -> tuple[str, dict[str, float]]:
    """Rank days by the minimum across experiments of |neg| prominence and
    return the best common day."""
    tasks = []
    for exp, days in days_per_exp.items():
        tasks.extend((exp, d) for d in days)
    scores: dict[str, dict[str, float]] = {d: {} for d in days_per_exp["control"]}
    # Multiprocess the prominence calculation.
    with Pool(N_WORKERS) as pool:
        for exp, day, s in pool.imap_unordered(_prominence_score, tasks):
            scores.setdefault(day, {})[exp] = s
    common_days = [d for d, m in scores.items()
                   if all(e in m for e in EXPERIMENTS)]
    if not common_days:
        raise SystemExit("No day has all three experiments with neg data")
    ranked = sorted(common_days,
                    key=lambda d: min(scores[d][e] for e in EXPERIMENTS),
                    reverse=True)
    best = ranked[0]
    return best, {d: min(scores[d][e] for e in EXPERIMENTS) for d in ranked[:20]}


def load_temperature(exp: str):
    path = AGG_DIR / f"temperature_{exp}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {"altitude_km": d["altitude_km"],
            "temperature_k": d["temperature_k"]}


def freezing_level_km(temp) -> float:
    if temp is None:
        return float("nan")
    z = np.asarray(temp["altitude_km"], dtype=np.float64)
    t = np.asarray(temp["temperature_k"], dtype=np.float64)
    m = np.isfinite(z) & np.isfinite(t)
    z, t = z[m], t[m]
    if z.size < 2:
        return float("nan")
    o = np.argsort(z); z, t = z[o], t[o]
    diff = t - T_FREEZE_K
    sc = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if sc.size == 0:
        return float("nan")
    i = int(sc[0])
    w = diff[i] / (diff[i] - diff[i + 1])
    return float(z[i] + w * (z[i + 1] - z[i]))


def tick_formatter():
    return mticker.FuncFormatter(lambda v, _: f"{v:g}")


def plot_grid(case_data: dict[str, dict[str, dict[str, np.ndarray]]],
              temps, path: Path, title: str, is_diff: bool = False):
    """case_data: exp -> species -> block -> profile (includes __altitude_km__).

    Plots SPECIES rows x EXPERIMENTS columns.  Neg block drawn in black/thick.
    """
    exps = list(case_data.keys())
    fig, axes = plt.subplots(len(SPECIES), len(exps),
                             figsize=(4.5 * len(exps), 3.2 * len(SPECIES)),
                             sharey=True)
    if len(SPECIES) == 1:
        axes = np.array([axes])
    if len(exps) == 1:
        axes = axes[:, None]

    for i, var in enumerate(SPECIES):
        for j, exp in enumerate(exps):
            ax = axes[i, j]
            v = case_data.get(exp, {}).get(var)
            if v is None or "__altitude_km__" not in v:
                ax.set_visible(False)
                continue
            z = v["__altitude_km__"]
            for block, profile in v.items():
                if block == "__altitude_km__":
                    continue
                if block.startswith("V") and block.endswith("M"):
                    continue
                color = BLOCK_COLORS.get(block, "#999999")
                lw = 2.0 if block == "neg" else 1.0
                zorder = 6 if block == "neg" else 2
                ax.plot(profile, z, color=color, lw=lw,
                        label=_pretty_label(block), zorder=zorder)
            z0 = freezing_level_km(temps.get(exp))
            if np.isfinite(z0):
                ax.axhline(z0, color="k", lw=1.0, ls="--", alpha=0.8, zorder=1)
            ax.axvline(0, color="k", lw=0.6, alpha=0.6)
            ax.set_ylim(0, Z_MAX_KM)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(tick_formatter())
            if j == 0:
                ax.set_ylabel(f"{var}\nAltitude (km)")
            if i == 0:
                ax.set_title(EXPERIMENTS[exp])
            if i == len(SPECIES) - 1:
                unit = r"$\Delta$ rate" if is_diff else "rate"
                ax.set_xlabel(f"{unit} (g kg$^{{-1}}$ day$^{{-1}}$)")
            if i == 0 and j == len(exps) - 1:
                ax.legend(fontsize=7, loc="center left",
                          bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path}")


def diff_case(case_0024: dict, case_0012: dict) -> dict:
    out = {}
    for exp, per_var in case_0024.items():
        out[exp] = {}
        v12 = case_0012.get(exp, {})
        for var, blocks in per_var.items():
            if var not in v12:
                continue
            alt = blocks.get("__altitude_km__")
            alt12 = v12[var].get("__altitude_km__")
            if alt is None or alt12 is None or alt.shape != alt12.shape:
                continue
            d = {"__altitude_km__": alt}
            for block, prof in blocks.items():
                if block == "__altitude_km__":
                    continue
                if block in v12[var] and prof.shape == v12[var][block].shape:
                    d[block] = prof - v12[var][block]
            out[exp][var] = d
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default=None,
                        help="Force a specific day (e.g. DDH20140716); "
                             "default: auto-pick prominent-neg day.")
    parser.add_argument("--species", nargs="+", default=list(SPECIES_DEFAULT),
                        choices=("QV", "QL", "QI", "QR", "QS", "QG"),
                        help="Species to include in the figure (one row each).")
    parser.add_argument("--tag", default=None,
                        help="Output filename suffix, e.g. 'QL' -> "
                             "case_<day>_diff_0024_minus_0012_QL.png.")
    args = parser.parse_args()

    global SPECIES
    SPECIES = tuple(args.species)

    days_per_exp = {}
    for exp in EXPERIMENTS:
        root = PROCESSED_BASE / "lead0024_VZ" / exp
        days_per_exp[exp] = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name.startswith("DDH20")
        )
    # use the intersection (defensive)
    common = set(days_per_exp["control"])
    for e in ("graupel", "2mom"):
        common &= set(days_per_exp[e])
    for e in EXPERIMENTS:
        days_per_exp[e] = sorted(common)

    if args.day:
        best_day = args.day
        ranking = {}
    else:
        best_day, ranking = select_case(days_per_exp)

    print(f"Selected case day: {best_day}")

    case_0024 = {e: load_day(e, best_day, "0024") for e in EXPERIMENTS}
    case_0012 = {e: load_day(e, best_day, "0012") for e in EXPERIMENTS}
    case_diff = diff_case(case_0024, case_0012)
    temps = {e: load_temperature(e) for e in EXPERIMENTS}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    date_tag = best_day.replace("DDH", "")
    suffix = f"_{args.tag}" if args.tag else ""
    plot_grid(case_diff, temps,
              FIG_DIR / f"case_{date_tag}_diff_0024_minus_0012{suffix}.png",
              f"Budget change between +0024 and +0012 on {date_tag}",
              is_diff=True)

    report = FIG_DIR / f"case_{date_tag}_report.txt"
    with open(report, "w") as f:
        f.write(f"Selected case day: {best_day}\n")
        f.write("Metric: for each day, min over experiments of the max |neg| "
                "rate across species in 0-20 km (+0024 data).\n\n")
        f.write("Top-20 days by that metric:\n")
        for day, score in sorted(ranking.items(), key=lambda kv: -kv[1])[:20]:
            f.write(f"  {day}  {score:10.3f}\n")
    print(f"  report: {report}")


if __name__ == "__main__":
    main()
