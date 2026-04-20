"""High-level orchestration for the DDH microphysics-budget pipeline.

Four smaller modules do the actual work:

* :mod:`alaro_analysis.ddh.aggregate_budgets` - per-day .dta -> per-(exp, var) npz
* :mod:`alaro_analysis.ddh.extract_temperature` - annual-mean T(z) from DDH LFA
* :mod:`alaro_analysis.ddh.plot_budgets` - the seven main figures
* :mod:`alaro_analysis.ddh.plot_case_study` - +0024 minus +0012 case figure

``run_pipeline()`` chains them in the normal order so end users get
everything with a single call.  The upstream ddhb run (which produces the
.dta files on disk) is intentionally *not* part of this function - that
step is a long-running shell job (bash ``run_ddh_budgets.sh``) that should
be triggered separately.

Example
-------
::

    from alaro_analysis.ddh import run_pipeline
    run_pipeline()                              # everything, defaults
    run_pipeline(leads=("0024",), case_day=None) # skip +0012 / case study
"""
from __future__ import annotations

from pathlib import Path

from . import aggregate_budgets, extract_temperature, plot_budgets, plot_case_study


def aggregate_all(leads: tuple[str, ...] = ("0012", "0024"),
                  ycoor: str = "VZ") -> None:
    """Run ``aggregate_budgets`` for every forecast lead."""
    for lead in leads:
        print(f"--- aggregate_budgets  lead={lead}  ycoor={ycoor} ---")
        aggregate_budgets.run(lead=lead, ycoor=ycoor)


def extract_temperatures() -> Path:
    """Compute the annual-mean temperature profile used for 0 C isotherms."""
    print("--- extract_temperature ---")
    return extract_temperature.run()


def plot_all(lead: str = "0024",
             case_day: str | None = None,
             case_species: tuple[str, ...] = ("QV", "QL", "QI"),
             case_tag: str | None = None) -> dict[str, list[Path] | Path]:
    """Produce the seven main figures plus one case-study figure."""
    print(f"--- plot_budgets  lead={lead} ---")
    main_figs = plot_budgets.run(lead=lead)
    print("--- plot_case_study ---")
    case_fig  = plot_case_study.run(day=case_day,
                                    species=case_species,
                                    tag=case_tag)
    return {"main": main_figs, "case": case_fig}


def run_pipeline(leads: tuple[str, ...] = ("0012", "0024"),
                 ycoor: str = "VZ",
                 plot_lead: str = "0024",
                 case_day: str | None = None,
                 case_species: tuple[str, ...] = ("QV", "QL", "QI"),
                 case_tag: str | None = None,
                 skip_aggregate: bool = False,
                 skip_temperature: bool = False,
                 skip_plots: bool = False) -> dict[str, list[Path] | Path] | None:
    """End-to-end DDH analysis.

    The aggregation and temperature-extraction steps read the raw .dta files
    produced by ``run_ddh_budgets.sh``; that shell stage must have run first.
    The ``skip_*`` flags let you re-run only part of the pipeline during
    iterative work without recomputing everything.
    """
    if not skip_aggregate:
        aggregate_all(leads=leads, ycoor=ycoor)
    if not skip_temperature:
        extract_temperatures()
    if skip_plots:
        return None
    return plot_all(lead=plot_lead,
                    case_day=case_day,
                    case_species=case_species,
                    case_tag=case_tag)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leads", nargs="+", default=["0012", "0024"],
                        help="Forecast leads to aggregate.")
    parser.add_argument("--plot-lead", default="0024",
                        help="Lead used for the main seven figures.")
    parser.add_argument("--case-day", default=None,
                        help="Force a specific case-study day.")
    parser.add_argument("--case-species", nargs="+", default=["QV", "QL", "QI"])
    parser.add_argument("--case-tag", default=None)
    parser.add_argument("--skip-aggregate",   action="store_true")
    parser.add_argument("--skip-temperature", action="store_true")
    parser.add_argument("--skip-plots",       action="store_true")
    args = parser.parse_args()
    run_pipeline(
        leads=tuple(args.leads),
        plot_lead=args.plot_lead,
        case_day=args.case_day,
        case_species=tuple(args.case_species),
        case_tag=args.case_tag,
        skip_aggregate=args.skip_aggregate,
        skip_temperature=args.skip_temperature,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
