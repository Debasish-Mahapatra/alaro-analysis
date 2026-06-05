"""Cell / feature tracking of precipitation using tobac.

A self-contained line of work, separate from the microphysics and
spatial-bias analysis. It consumes the converted NetCDF data products
(see :mod:`alaro_analysis.converter`) and may reuse the shared
``common``/``data`` helpers, but keeps its tracking-specific logic here.
"""
