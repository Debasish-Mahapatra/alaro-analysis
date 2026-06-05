#!/usr/bin/env python3
"""
Plot mean diurnal cycle of boundary-layer height (CLPMHAUT.MOD.XFU)
for control / graupel / 2mom experiments, with an LCL overlay when
the converted H00100TEMPERATUR / H00100HUMI.SPECI / H00100PRESSURE
inputs are available.

Usage:
    python plot_pblh_diurnal.py
    python plot_pblh_diurnal.py --analysis-modes full
"""

from alaro_analysis.workflows.surface import main

if __name__ == "__main__":
    main()
