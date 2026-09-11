"""Explicit variable choices for the strong-updraft paper recipes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSpec:
    variable: str
    label: str
    slug: str
    folder_prefix: str
    color: str
    generic_id: str
    xmin_kg_kg: float = 1.0e-9
    xmax_kg_kg: float = 5.0e-4


RAIN_PROFILE = ProfileSpec(
    "RAIN", "Rain", "rain", "06", "#d62728", "qr, specific humidity of rain (kg/kg)"
)
GRAUPEL_PROFILE = ProfileSpec(
    "GRAUPEL", "Graupel", "graupel", "07", "#e66101", "qg, specific humidity of graupel (kg/kg)"
)
