#!/usr/bin/env python3
"""
Plot mean diurnal cycle of boundary-layer height (CLPMHAUT.MOD.XFU)
for control / graupel / 2mom experiments.

Usage:
    python plot_pblh_diurnal.py
    python plot_pblh_diurnal.py --seasons wet dry --zoom-inset
"""

from alaro_analysis.workflows.surface import main

if __name__ == "__main__":
    main()
