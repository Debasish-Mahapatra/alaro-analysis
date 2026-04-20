"""DDH microphysics-budget pipeline.

Tools for orchestrating ``ddhb`` (ddhtoolbox) over the full 730-day DDH archive
and plotting the resulting budgets.

Usage
-----
Shell orchestration (ddhb parallel driver):

    bash alaro_analysis/ddh/run_ddh_budgets.sh

Python entry points:

    python -m alaro_analysis.ddh.aggregate_budgets --lead 0024 --ycoor VZ
    python -m alaro_analysis.ddh.extract_temperature
    python -m alaro_analysis.ddh.plot_budgets
    python -m alaro_analysis.ddh.plot_case_study --species QL --tag QL

Shared helpers (paths, experiments, IO, plot primitives) live in
:mod:`alaro_analysis.ddh.io`.
"""

from . import io  # re-export so callers can do `from alaro_analysis.ddh import io`

__all__ = ["io"]
