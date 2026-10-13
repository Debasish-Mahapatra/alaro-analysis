"""Dominant hydrometeor species per model cell, for comparison with radar classification.

A polarimetric classifier labels each radar gate with the species that dominates its
echo. The model equivalent is the species holding most of the precipitating mass in a
cell, counted only where the model would be detectable: simulated reflectivity above
the same 0 dBZ floor used on the radar.

Cloud droplets (qc) are left out on purpose: they scatter far too little to set a radar
gate's classification, which is why the CSU scheme has no cloud category either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPECIES = ("qr", "qi", "qs", "qg")       # rain, cloud ice, snow, graupel
HEIGHT_EDGES_KM = np.arange(0.0, 16.5, 0.5)
DBZ_MIN = 0.0                            # the radar pass keeps gates above 0 dBZ
MASS_MIN = 1.0e-7                        # kg/kg, below this "dominant" is meaningless


def species_counts_day(task: tuple[str, bool]) -> tuple[np.ndarray, int, str]:
    """Counts per (height bin, species) for one day file; task = (path, radar mask only)."""
    from netCDF4 import Dataset

    path, inside_only = task
    counts = np.zeros((HEIGHT_EDGES_KM.size - 1, len(SPECIES)), dtype=np.int64)
    try:
        with Dataset(path) as data:
            dbz = np.asarray(data["dbz"][:], dtype=np.float32)
            height_km = np.asarray(data["height"][:], dtype=np.float32) / 1000.0
            mass = np.stack([np.asarray(data[name][:], dtype=np.float32) for name in SPECIES])
            inside = np.asarray(data["inside"][:], dtype=bool) if inside_only else None
    except Exception as exc:  # noqa: BLE001 - one bad day must not stop the pass
        return counts, 0, f"{Path(path).name}: {exc}"

    total = np.nansum(mass, axis=0)
    good = np.isfinite(dbz) & (dbz > DBZ_MIN) & (total > MASS_MIN)
    if inside is not None:
        good &= inside[None, None, :, :]
    if not good.any():
        return counts, 0, ""
    dominant = np.argmax(np.nan_to_num(mass), axis=0)
    rows = np.digitize(height_km[good], HEIGHT_EDGES_KM) - 1
    valid = (rows >= 0) & (rows < counts.shape[0])
    np.add.at(counts, (rows[valid], dominant[good][valid]), 1)
    return counts, int(good.sum()), ""
