"""Plotting style helpers.

``resolve_workers`` now lives in :mod:`alaro_analysis.common.parallel`; it is
re-exported here for backward compatibility with existing imports.
"""

from __future__ import annotations

from alaro_analysis.common.parallel import resolve_workers

__all__ = ["resolve_workers"]
