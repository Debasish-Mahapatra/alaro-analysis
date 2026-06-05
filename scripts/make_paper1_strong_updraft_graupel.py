#!/usr/bin/env python3
"""Paper1 graupel profile sampled only inside strong updrafts."""

from __future__ import annotations

import make_paper1_strong_updraft_rain as workflow


workflow.PROFILE_VAR = "GRAUPEL"
workflow.PROFILE_LABEL = "Graupel"
workflow.PROFILE_SLUG = "graupel"
workflow.PROFILE_FOLDER_PREFIX = "07"
workflow.PROFILE_COLOR = "#e66101"
workflow.PROFILE_GENERIC_ID = "qg, specific humidity of graupel (kg/kg)"
workflow.PROFILE_XMIN_KG_KG = 1.0e-9
workflow.PROFILE_XMAX_KG_KG = 5.0e-4


if __name__ == "__main__":
    workflow.main()
