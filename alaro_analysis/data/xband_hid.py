"""Hydrometeor identification from the CHUVA X-band dual-pol radar at T3.

The level-1 volumes are GAMIC HDF5: one group per sweep, moments stored as scaled
integers with their physical range in the dataset attributes. Only numpy arrays are
needed downstream, so the files are read directly rather than through a radar toolkit.

Classification is the CSU fuzzy-logic scheme (Dolan & Rutledge 2009, Thompson et al.
2014) with its X-band membership functions, driven by Z, ZDR, KDP, RhoHV and a
temperature profile taken from the ARM radiosondes at the same site, so the melting
level is observed rather than assumed.

Classes (CSU convention):
    1 drizzle, 2 rain, 3 ice crystals, 4 aggregates, 5 wet snow, 6 vertical ice,
    7 low-density graupel, 8 high-density graupel, 9 hail, 10 big drops
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MOMENTS = {"Z": "dz", "ZDR": "zdr", "KDP": "kdp", "RHOHV": "rho"}
RHOHV_MIN = 0.80          # below this the gate is clutter or noise, not hydrometeors
DZ_MIN = 0.0              # dBZ, the classifier is meaningless in clear air
MAX_RANGE_KM = 60.0       # beyond this the beam is too high and too broad to trust
HEIGHT_EDGES_KM = np.arange(0.0, 16.5, 0.5)
CLASS_NAMES = {
    1: "drizzle", 2: "rain", 3: "ice crystals", 4: "aggregates", 5: "wet snow",
    6: "vertical ice", 7: "low-density graupel", 8: "high-density graupel",
    9: "hail", 10: "big drops",
}
EARTH_RADIUS_KM = 6371.0


def read_volume(path: Path) -> dict[str, np.ndarray] | None:
    """Z, ZDR, KDP, RhoHV, gate height and range of one volume, sweeps concatenated."""
    import h5py

    fields: dict[str, list[np.ndarray]] = {name: [] for name in MOMENTS}
    heights: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        sweeps = sorted((k for k in handle if k.startswith("scan")),
                        key=lambda k: int(k[4:]))
        if not sweeps:
            return None
        for sweep in sweeps:
            group = handle[sweep]
            moments = {}
            for key in group:
                if not key.startswith("moment_"):
                    continue
                attrs = group[key].attrs
                name = attrs.get("moment")
                name = name.decode() if isinstance(name, bytes) else name
                if name not in MOMENTS:
                    continue
                raw = group[key][:]
                low = float(attrs["dyn_range_min"])
                high = float(attrs["dyn_range_max"])
                # GAMIC packing: value = min + (raw - 1) * gain, raw = 0 means no data
                gain = (high - low) / (np.iinfo(raw.dtype).max - 1)
                values = low + (raw.astype(np.float64) - 1.0) * gain
                moments[name] = np.where(raw == 0, np.nan, values)
            if len(moments) < len(MOMENTS):
                continue
            how = group["how"].attrs
            n_bins = int(how["bin_count"])
            start = float(how.get("range_start", 0.0))
            step = (float(how["range_step"]) if "range_step" in how
                    else float(how["range"]) / n_bins)
            gate_km = (start + (np.arange(n_bins) + 0.5) * step) / 1000.0
            elevation = np.radians(np.asarray(group["ray_header"]["elevation_start"], float))
            # beam height with the 4/3 earth radius approximation
            effective = 4.0 / 3.0 * EARTH_RADIUS_KM
            height = np.sqrt(
                gate_km[None, :] ** 2 + effective ** 2
                + 2.0 * gate_km[None, :] * effective * np.sin(elevation[:, None])
            ) - effective
            # sweeps can differ in gate count (500 vs 495), so flatten each one
            for name in MOMENTS:
                fields[name].append(moments[name].ravel())
            heights.append(height.ravel())
            ranges.append(np.broadcast_to(gate_km[None, :], height.shape).ravel())
    if not heights:
        return None
    out = {name: np.concatenate(values) for name, values in fields.items()}
    out["height_km"] = np.concatenate(heights)
    out["range_km"] = np.concatenate(ranges)
    return out


def classify(volume: dict[str, np.ndarray], temperature_km, temperature_c,
             zdr_offset: float = 0.0) -> np.ndarray:
    """CSU fuzzy-logic class per gate; 0 where the gate is rejected.

    ``zdr_offset`` is the radar's own differential bias, measured from the
    vertical-pointing scans where rain must have ZDR = 0 dB, and subtracted here.
    It matters: graupel has low intrinsic ZDR, so a positive bias pushes graupel
    gates into the higher-ZDR aggregate class.
    """
    from csu_radartools import csu_fhc

    good = (
        np.isfinite(volume["Z"]) & np.isfinite(volume["ZDR"])
        & np.isfinite(volume["KDP"]) & np.isfinite(volume["RHOHV"])
        & (volume["RHOHV"] > RHOHV_MIN) & (volume["Z"] > DZ_MIN)
        & (volume["range_km"] <= MAX_RANGE_KM)
        & (volume["height_km"] <= HEIGHT_EDGES_KM[-1])
    )
    classes = np.zeros(good.size, dtype=np.int8)
    if not good.any():
        return classes
    temperature = np.interp(volume["height_km"][good], temperature_km, temperature_c)
    scores = csu_fhc.csu_fhc_summer(
        dz=volume["Z"][good], zdr=volume["ZDR"][good] - zdr_offset,
        kdp=volume["KDP"][good],
        rho=volume["RHOHV"][good], T=temperature, band="X",
    )
    classes[good] = np.asarray(scores, dtype=np.int8)
    return classes


def histogram(classes: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    """Counts per (height bin, class 1..10)."""
    keep = classes > 0
    out = np.zeros((HEIGHT_EDGES_KM.size - 1, len(CLASS_NAMES)), dtype=np.int64)
    if not keep.any():
        return out
    rows = np.digitize(height_km[keep], HEIGHT_EDGES_KM) - 1
    valid = (rows >= 0) & (rows < out.shape[0])
    np.add.at(out, (rows[valid], classes[keep][valid] - 1), 1)
    return out


def process_volume(task) -> tuple[np.ndarray, int, str]:
    """Class histogram of one volume; task = (path, sonde km, sonde degC, ZDR offset)."""
    path, temperature_km, temperature_c, zdr_offset = task
    try:
        volume = read_volume(Path(path))
        if volume is None:
            return np.zeros((HEIGHT_EDGES_KM.size - 1, len(CLASS_NAMES)), np.int64), 0, "empty"
        classes = classify(volume, temperature_km, temperature_c, zdr_offset)
        return histogram(classes, volume["height_km"]), int((classes > 0).sum()), ""
    except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the pass
        return (np.zeros((HEIGHT_EDGES_KM.size - 1, len(CLASS_NAMES)), np.int64), 0,
                f"{Path(path).name}: {exc}")
