"""Scientific variable names, units and bounded-value conventions."""

from __future__ import annotations

LINEAR_ABSOLUTE_VARIABLE_KEYS = {
    "HUMI.RELATIVE",
    "CLOUD_FRACTI",
    "SURFNEBUL.BASSE",
    "SURFNEBUL.HAUTE",
    "SURFNEBUL.MOYENN",
    "SURFNEBUL.TOTALE",
    "UD_MESH_FRAC",
    "UPDRAFT_EXTENT",
    "UPDRAFT_INTENSITY",
    "UPDRAFT_FLUX",
    "TEMPERATURE",
    "HUMI.SPECIFI",
    "RAIN",
    "SNOW",
    "GRAUPEL",
    "LIQUID_WATER",
    "SOLID_WATER",
    "RAD_LIQUID_W",
    "RAD_SOLID_WA",
}


UNIT_INTERVAL_VARIABLE_KEYS = {
    "HUMI.RELATIVE",
    "CLOUD_FRACTI",
    "SURFNEBUL.BASSE",
    "SURFNEBUL.HAUTE",
    "SURFNEBUL.MOYENN",
    "SURFNEBUL.TOTALE",
    "UD_MESH_FRAC",
    "UPDRAFT_EXTENT",
}


def variable_label(name: str) -> str:
    key = name.strip().upper()
    if key == "UPDRAFT_FLUX":
        return "Updraft Flux"
    if key == "UPDRAFT_EXTENT":
        return "Updraft Extent"
    if key == "UPDRAFT_INTENSITY":
        return "Updraft Intensity"
    return name


def variable_key(name: str) -> str:
    return name.strip().upper()


def variable_unit(name: str) -> str:
    key = variable_key(name)
    if key == "UPDRAFT_FLUX":
        return "kg/(m^2 s)"
    if key == "UPDRAFT_INTENSITY":
        return "Pa/s"
    if key == "UPDRAFT_EXTENT":
        return "Fraction"
    return ""


def uses_linear_absolute_scale(name: str) -> bool:
    return variable_key(name) in LINEAR_ABSOLUTE_VARIABLE_KEYS


def is_unit_interval_variable(name: str) -> bool:
    return variable_key(name) in UNIT_INTERVAL_VARIABLE_KEYS
