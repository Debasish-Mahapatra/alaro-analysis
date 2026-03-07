from __future__ import annotations

import argparse
from pathlib import Path

from .aliases import normalize_var_token


def parse_var_block(text: str) -> list[str]:
    vars_out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            vars_out.append(line)
    return vars_out


def unique_preserve(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_plevels_csv(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        part = token.strip()
        if not part:
            continue
        value = int(part)
        if value == 0:
            value = 100000
        out.append(value)
    return out


def to_pressure_aliases(items: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item.startswith("P_"):
            out.append(item)
        else:
            out.append(f"P_{item}")
    return out


MODEL_LEVEL_VARS_TEXT = """
DD_OMEGA
DD_MESH_FRAC
CV_PREC_FLUX
ST_PREC_FLUX
PRESSURE
NC_LIQUID_WA
# DMEANR
KT273GRAUPEL
KT273RAIN
KT273SNOW
KT273DD_OMEGA
KT273DD_MESH_FRA
KT273HUMI.SPECIF
KT273TEMPERATUR

# WIND.U.PHYS
# WIND.V.PHYS
# PNR
# REFLEC_DBZ
# CLOUD_FRACTI
# HUMI.SPECIFI
# HUMI.RELATIVE
# TEMPERATURE
# GEOPOTENTIEL
# LIQUID_WATER
# SOLID_WATER
# GRAUPEL
# SNOW
# RAIN
# RAD_LIQUID_W
# RAD_SOLID_WA
# VERT.VELOC
# VITESSE_VE
# UD_OMEGA
# UD_MESH_FRAC
"""

PLEVELS_CSV = "00000,92500,85000,70000,50000,20000"

PLEVEL_VARS_TEXT = """
# TEMPERATUR
# HUMI.SPECI
# GEOPOTENTI
# WIND.U.PHY
# WIND.V.PHY
# REFLEC_DBZ
# CLOUD_FRAC
# LIQUID_WAT
# SOLID_WATE
# VERT.VELOC
# VITESSE_VE
# CV_PREC_FL
# ST_PREC_FL
"""

SURFACE_VARS_TEXT = """
# SURFNEBUL.TOTALE
# SURFNEBUL.HAUTE
# SURFNEBUL.MOYENN
# SURFNEBUL.BASSE
# SURFTEMPERATURE
# SOMMFLU.RAY.THER
# SOMMFLU.RAY.SOLA
# SURFCAPE.POS.F00
# SURFCIEN.POS.F00
# CLSHUMI.SPECIFIQ
# CLSVENT.ZONAL
# CLSVENT.MERIDIEN
# SURFGEOPOTEN
# SURFPRESSION
# SURFPREC.EAU.GEC
# SURFPREC.EAU.CON
"""

MODEL_LEVEL_VARS = parse_var_block(MODEL_LEVEL_VARS_TEXT)
PLEVELS_PA = parse_plevels_csv(PLEVELS_CSV)
PLEVEL_VARS = parse_var_block(PLEVEL_VARS_TEXT)
PRESSURE_LEVEL_VARS = to_pressure_aliases(PLEVEL_VARS)
SURFACE_VARS = parse_var_block(SURFACE_VARS_TEXT)
REQUESTED_VARS = unique_preserve(
    MODEL_LEVEL_VARS + PRESSURE_LEVEL_VARS + SURFACE_VARS
)
AUX_VAR = "GEOPOTENTIEL"
GRAVITY = 9.80665
DERIVED_MODEL_RH_VAR = "HUMI.RELATIVE"
MODEL_RH_Q_VAR = "HUMI.SPECIFI"
MODEL_RH_T_VAR = "TEMPERATURE"
MODEL_RH_PRESSURE_CANDIDATES = ("PRESSURE",)


def parse_vars_tokens(tokens: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        for part in token.split(","):
            item = part.strip()
            if item:
                out.append(item)
    return unique_preserve(out)


def load_vars_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Variable file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Variable file is not a regular file: {path}")
    lines = parse_var_block(path.read_text())
    return parse_vars_tokens(lines)


def resolve_requested_vars_from_cli(args: argparse.Namespace) -> tuple[list[str], str]:
    selected = list(REQUESTED_VARS)
    source = "built-in defaults"

    if args.vars_file is not None:
        selected = load_vars_file(Path(args.vars_file))
        source = f"vars-file:{Path(args.vars_file)}"

    if args.vars:
        selected = parse_vars_tokens(args.vars)
        source = "--vars"

    if args.append_vars:
        selected = unique_preserve(selected + parse_vars_tokens(args.append_vars))
        source = f"{source}+append"

    if args.drop_vars:
        drop_norm = {normalize_var_token(name) for name in parse_vars_tokens(args.drop_vars)}
        selected = [name for name in selected if normalize_var_token(name) not in drop_norm]
        source = f"{source}+drop"

    return selected, source


__all__ = [
    "AUX_VAR",
    "DERIVED_MODEL_RH_VAR",
    "GRAVITY",
    "MODEL_LEVEL_VARS",
    "MODEL_LEVEL_VARS_TEXT",
    "MODEL_RH_PRESSURE_CANDIDATES",
    "MODEL_RH_Q_VAR",
    "MODEL_RH_T_VAR",
    "PLEVELS_CSV",
    "PLEVELS_PA",
    "PLEVEL_VARS",
    "PLEVEL_VARS_TEXT",
    "PRESSURE_LEVEL_VARS",
    "REQUESTED_VARS",
    "SURFACE_VARS",
    "SURFACE_VARS_TEXT",
    "load_vars_file",
    "parse_plevels_csv",
    "parse_var_block",
    "parse_vars_tokens",
    "resolve_requested_vars_from_cli",
    "to_pressure_aliases",
    "unique_preserve",
]
