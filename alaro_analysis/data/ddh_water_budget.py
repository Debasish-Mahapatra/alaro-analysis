"""Water budget of the DDH domain from the raw DDH (LFA) files, for one run day.

The IOP runs write a DDH budget for one box of about 150 km x 150 km centred on the ARM
site T3 (DOCD001: 61.27-59.93 W, 3.88-2.54 S). Each ``DHFDLABOF+00HH`` file holds the
budget accumulated from the 00 UTC start, so ``+0024`` is the day. Its articles are
read with the toolbox's ``lfac`` (text dump of one article), not with epygram, whose
LFA reader aborts the process now and then.

Articles (kg m-2 accumulated over the window, half levels top to surface unless stated):

* precipitation fluxes through each half level, 3MT (``C``) and resolved (``S``):
  rain ``FQRPL*``, snow ``FQSPL*``, graupel ``FQGPL*``;
* process pseudo-fluxes, i.e. the process integrated from the model top down to the half
  level, so the surface value is the column total: evaporation (or sublimation) of
  falling rain / snow / graupel ``FQRPEV*`` / ``FQSPEV*`` / ``FQGPEV*``; net condensation to
  cloud liquid / ice ``FQLC*`` / ``FQNC*``; conversion of cloud liquid to rain ``FQLPFP*``,
  of cloud ice to snow ``FQNPFP*`` and to graupel ``FQGPFPS``; turbulent vapour flux
  ``FQVDIFT`` (its surface value is minus the surface evaporation); enthalpy fluxes (J m-2):
  turbulent ``FCTDIFT`` (surface value: minus the sensible heat flux), net solar ``FCTRSO``
  and net thermal ``FCTRTH`` radiation (top and surface values);
* full levels: layer mass ``VPP0/VPP1`` (dp/g, start and end), enthalpy ``VCT0/VCT1``
  (cp T dp/g) and vapour ``VQV0/VQV1`` (q dp/g); horizontal moisture-flux
  convergence ``TQVDIVFLUHOR`` (kg m-2, positive = moisture brought into the box).

A missing article (no graupel in the 2-ice C1M) is returned as zeros.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from alaro_analysis.common.paths import data_path

LFAC = data_path("ddhtoolbox") / "tools" / "lfa" / "lfac"
HALF_LEVEL = (
    "FQRPLS", "FQRPLC", "FQSPLS", "FQSPLC", "FQGPLS", "FQGPLC",
    "FQRPEVS", "FQRPEVC", "FQSPEVS", "FQSPEVC", "FQGPEVS", "FQGPEVC",
    "FQLCS", "FQLCC", "FQNCS", "FQNCC",
    "FQLPFPS", "FQLPFPC", "FQNPFPS", "FQNPFPC", "FQGPFPS",
    "FQVDIFT", "FQVDIFC",
    "FCTDIFT", "FCTRSO", "FCTRTH",
)
FULL_LEVEL = ("VPP0", "VPP1", "VCT0", "VCT1", "VQV0", "VQV1", "TQVDIVFLUHOR")
N_LEVELS = 87


def read_article(path: Path, name: str) -> np.ndarray | None:
    done = subprocess.run([str(LFAC), str(path), name], capture_output=True, text=True)
    if done.returncode or not done.stdout.strip():
        return None
    return np.array([float(v.replace("D", "E")) for v in done.stdout.split()], dtype=float)


def read_budget(path: Path, names: tuple[str, ...] = HALF_LEVEL + FULL_LEVEL) -> dict[str, np.ndarray] | None:
    """The articles ``names`` of one DDH file, or None when the file is unreadable."""
    out: dict[str, np.ndarray] = {}
    for name in names:
        values = read_article(path, name)
        size = N_LEVELS + 1 if name in HALF_LEVEL else N_LEVELS
        if values is None:
            if name in ("VPP0", "VCT0", "VQV0", "FQRPLS", "FQRPEVS"):
                return None
            values = np.zeros(size)
        if values.size != size:
            return None
        out[name] = values
    return out


# --------------------------------------------------------------------------- derived quantities
G = 9.80665
RD = 287.06
CP = 1004.709
PRECIPITATION = ("FQRPLS", "FQRPLC", "FQSPLS", "FQSPLC", "FQGPLS", "FQGPLC")
PRECIPITATION_3MT = ("FQRPLC", "FQSPLC", "FQGPLC")
CONDENSATION = ("FQLCS", "FQLCC", "FQNCS", "FQNCC")
FORMATION = ("FQLPFPS", "FQLPFPC", "FQNPFPS", "FQNPFPC", "FQGPFPS")
# Evaporation of the falling rain, snow and graupel. Always summed over the three: in the
# melting layer the files book snow and graupel losses against rain gains of the same
# layer, so only the sum is the water that goes back to vapour.
EVAPORATION = ("FQRPEVS", "FQRPEVC", "FQSPEVS", "FQSPEVC", "FQGPEVS", "FQGPEVC")


def full_level_heights(vpp: np.ndarray, vct: np.ndarray, vqv: np.ndarray) -> np.ndarray:
    """Height (m above the surface) of the full levels, top first, from the layer mass,
    enthalpy and vapour of the DDH (hydrostatic, virtual temperature)."""
    t = vct / vpp / CP
    tv = t * (1.0 + 0.608 * vqv / vpp)
    p_half = np.concatenate([[0.0], np.cumsum(vpp * G)])
    p_mid = 0.5 * (p_half[1:] + p_half[:-1])
    dz = RD * tv / G * np.diff(p_half) / p_mid
    z_half = np.concatenate([np.cumsum(dz[::-1])[::-1], [0.0]])
    return 0.5 * (z_half[1:] + z_half[:-1])


def column_total(data: dict, names: tuple[str, ...], prefix: str = "day_") -> np.ndarray:
    """Surface value of the pseudo-fluxes (or fluxes) ``names``, summed: the column total."""
    return sum(data[prefix + n][..., -1] for n in names)


def per_level(data: dict, names: tuple[str, ...], prefix: str = "day_") -> np.ndarray:
    """Amount of the processes ``names`` in each full level (kg m-2 over the window), top first."""
    return sum(np.diff(data[prefix + n], axis=-1) for n in names)
