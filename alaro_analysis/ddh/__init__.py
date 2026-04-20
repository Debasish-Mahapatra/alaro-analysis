"""DDH microphysics-budget pipeline.

Tools for orchestrating ``ddhb`` (ddhtoolbox) over the full 730-day DDH archive
and plotting the resulting budgets.

Typical use
-----------

After running the shell driver once to produce the .dta files::

    bash alaro_analysis/ddh/run_ddh_budgets.sh

the whole Python-side analysis reduces to a single call::

    from alaro_analysis.ddh import run_pipeline
    run_pipeline()

which aggregates per-day .dta files, extracts the annual-mean temperature
profile (used for the 0 C isotherm), draws the seven main figures and the
+0024 minus +0012 case study.

Submodules (each has a ``run(**kwargs)`` Python entry point in addition to
``python -m`` invocation)::

    python -m alaro_analysis.ddh.aggregate_budgets --lead 0024 --ycoor VZ
    python -m alaro_analysis.ddh.extract_temperature
    python -m alaro_analysis.ddh.plot_budgets
    python -m alaro_analysis.ddh.plot_case_study --species QL --tag QL
    python -m alaro_analysis.ddh.pipeline              # one-shot end-to-end

Shared helpers (paths, experiments, IO, plot primitives) live in
:mod:`alaro_analysis.ddh.io`.
"""

from . import io
from .pipeline import (
    aggregate_all,
    extract_temperatures,
    plot_all,
    run_pipeline,
)

__all__ = [
    "aggregate_all",
    "extract_temperatures",
    "io",
    "plot_all",
    "run_pipeline",
]
