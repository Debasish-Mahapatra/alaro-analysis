"""Reusable analysis building blocks for ALARO model outputs.

Primary API::

    from alaro_analysis import ExperimentSet

For advanced / low-level use, import directly from submodules::

    from alaro_analysis.analysis.profiles import compute_diurnal_profile
    from alaro_analysis.analysis.derived import compute_theta_e_field
"""

from .experiment import ExperimentSet

__all__ = ["ExperimentSet"]
