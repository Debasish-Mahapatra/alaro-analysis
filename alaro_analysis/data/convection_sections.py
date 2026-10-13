"""Convection sections. Shared implementation; command-line orchestration lives in recipes.

Per-hour domain-mean vertical profiles of the parameterized updraft and downdraft,
the resolved vertical velocity, cloud cover and temperature over the SIPAM radar
domain on the native 4 km grid (``data.radar_domain``), read from the raw FA output.
The masked NetCDF archive is not used: it is cut to the older 0.1 deg radar mask.

The day workers live here rather than in the recipe because a recipe module runs
as ``__main__`` under ``alaro-recipe``, and a worker function defined there cannot
be pickled to a process pool.
"""

from __future__ import annotations

import os

# One thread per worker: the ecTrans spectral transform would otherwise
# oversubscribe the machine once per worker.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings
from pathlib import Path

import numpy as np

from alaro_analysis.analysis.updrafts import downdraft_mass_flux, updraft_mass_flux
from alaro_analysis.common.constants import G
from alaro_analysis.data.radar_domain import mask_window

UTC_OFFSET = -4
N_LEVELS = 87
FIELDS = (
    "UD_MESH_FRAC",
    "UD_OMEGA",
    "DD_MESH_FRAC",
    "DD_OMEGA",
    "VERT.VELOCIT",
    "CLOUD_FRACTI",
    "TEMPERATURE",
)
QUANTITIES = ("ud_extent", "ud_flux", "dd_extent", "dd_flux", "w_up", "w_dn", "cloud", "temp")

_WINDOW: tuple[int, int, int, int, np.ndarray] | None = None


def nonnegative(values: np.ndarray) -> np.ndarray:
    """``max(values, 0)`` that keeps NaN (cells outside the domain) as NaN."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), np.maximum(values, 0.0), np.nan)


def profiles_from_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Domain-mean vertical profiles of one hour, from (level, ...) field arrays.

    Every cell of the domain counts, including those without an updraft or
    downdraft (they add 0). Cells outside the domain are NaN and stay out of every
    mean, so no mean is diluted by counting them as zero.

    The draft fields are cleaned the way the physics cleans them on entry. The
    model advects them semi-Lagrangian without the quasi-monotone limiter
    (``YUAL_NL%LQM=.F.``), so they overshoot and undershoot a little, and the
    3MT scheme discards that before use: mesh fractions below 0 become 0
    (``accvud``, ``acmodo``/``acnsdo``), updraft omega above 0 becomes 0
    (``PUDOM = MIN(0, PUDOM)``) and downdraft omega below 0 becomes 0
    (``PDDOM = MAX(0, PDDOM)``).
    """
    axis = tuple(range(1, fields["UD_MESH_FRAC"].ndim))
    ud_mesh = nonnegative(fields["UD_MESH_FRAC"])
    dd_mesh = nonnegative(fields["DD_MESH_FRAC"])
    ud_omega = -nonnegative(-np.asarray(fields["UD_OMEGA"], dtype=np.float64))
    dd_omega = nonnegative(fields["DD_OMEGA"])
    w = fields["VERT.VELOCIT"]
    finite_w = np.isfinite(w)
    parts = {
        "ud_extent": ud_mesh,
        "ud_flux": updraft_mass_flux(ud_omega, ud_mesh),
        "dd_extent": dd_mesh,
        "dd_flux": downdraft_mass_flux(dd_omega, dd_mesh),
        "w_up": np.where(finite_w, np.maximum(w, 0.0), np.nan),
        "w_dn": np.where(finite_w, np.maximum(-w, 0.0), np.nan),
        "cloud": fields["CLOUD_FRACTI"],
        "temp": fields["TEMPERATURE"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {key: np.nanmean(values, axis=axis) for key, values in parts.items()}


# Upward and downward mass flux of convective columns in 20 km boxes, for the
# comparison with the GoAmazon radar wind profiler (Giangrande et al. 2016).
DRAFT_FIELDS = ("UD_MESH_FRAC", "UD_OMEGA", "DD_MESH_FRAC", "DD_OMEGA", "VITESSE_VERT")
RAIN_FIELDS = ("SURFPREC.EAU.CON", "SURFPREC.EAU.GEC")
BOX_CELLS = 5  # 5 x 5 native cells = 20 km
BOX_MIN_INSIDE = 0.5  # boxes at least half inside the radar domain
RAIN_MIN_MM_H = 1.0  # a convective column rains at least this much in the hour
ACTIVE_MESH = 1.0e-3  # a 3MT updraft is present (smaller fractions are advection noise)
BOX_QUANTITIES = ("box_up", "box_up_nodd", "box_down", "box_par_up", "box_res_up", "box_frequency")


def column_fluxes(omega, sig_u, om_u_rel, sig_d, om_d):
    """Upward and downward mass flux (kg m-2 s-1, both >= 0) of columns split into
    updraft, downdraft and environment, as the 3MT scheme defines them.

    omega     grid-mean pressure velocity (VITESSE_VERT), Pa s-1, negative up
    sig_u     updraft mesh fraction; om_u_rel its velocity RELATIVE to its environment
              (ACCVUD: "updraught relative velocity"; Gerard et al. 2009, around eq. 4)
    sig_d     downdraft mesh fraction; om_d its ABSOLUTE velocity (ACMODO: "absolute DD
              velocity"), sitting inside the updraft's environment

    updraft environment  om_e  = omega - sig_u * om_u_rel
    updraft              om_up = omega + (1 - sig_u) * om_u_rel
    rest of the column   om_ed = ((1 - sig_u) * om_e - sig_d * om_d) / (1 - sig_u - sig_d)

    The three pieces add back to omega, so up - down = -omega / g. ``up_nodd`` is the
    same without the downdraft split (updraft + environment only).
    """
    g = G
    sig_u = np.minimum(sig_u, 0.999)  # fractions at or above 1 are advection overshoots
    om_e = omega - sig_u * om_u_rel
    om_up = omega + (1.0 - sig_u) * om_u_rel
    om_d = np.maximum(om_d, om_e)  # ACMODO: no downdraft slower than its environment
    sig_d = np.minimum(sig_d, np.maximum(1.0 - sig_u - 1.0e-3, 0.0))
    rest = 1.0 - sig_u - sig_d
    om_ed = ((1.0 - sig_u) * om_e - sig_d * om_d) / rest
    up = -(sig_u * np.minimum(om_up, 0) + sig_d * np.minimum(om_d, 0) + rest * np.minimum(om_ed, 0)) / g
    down = (sig_u * np.maximum(om_up, 0) + sig_d * np.maximum(om_d, 0) + rest * np.maximum(om_ed, 0)) / g
    return {
        "box_up": up,
        "box_up_nodd": -(sig_u * np.minimum(om_up, 0) + (1.0 - sig_u) * np.minimum(om_e, 0)) / g,
        "box_down": down,
        "box_par_up": -sig_u * np.minimum(om_up, 0) / g,
        "box_res_up": -np.minimum(omega, 0) / g,
    }


def box_layout(mask: np.ndarray, size: int = BOX_CELLS, min_inside: float = BOX_MIN_INSIDE):
    """(rows, cols, boxes kept) of the size x size tiling of the crop window."""
    ny, nx = (mask.shape[0] // size) * size, (mask.shape[1] // size) * size
    inside = mask[:ny, :nx].reshape(ny // size, size, nx // size, size).mean(axis=(1, 3))
    return ny, nx, inside >= min_inside


def box_means(values: np.ndarray, ny: int, nx: int, size: int = BOX_CELLS) -> np.ndarray:
    """Box sums divided by the full box area: (..., ny, nx) -> (..., ny/size, nx/size)."""
    lead = values.shape[:-2]
    tiles = values[..., :ny, :nx].reshape(*lead, ny // size, size, nx // size, size)
    return tiles.mean(axis=(-3, -1))


def convective_boxes(fields: dict[str, np.ndarray], rain: np.ndarray, mask: np.ndarray,
                     has_scheme: bool = True) -> tuple[dict[str, np.ndarray], int, int]:
    """Level profiles summed over the convective 20 km boxes of one snapshot.

    fields: (level, y, x) crops of DRAFT_FIELDS; rain: (y, x) mm in the hour ending at
    the snapshot. A column is convective if it rains >= RAIN_MIN_MM_H and, in runs with
    3MT, carries an updraft (mesh > ACTIVE_MESH at some level). Fluxes of the other
    columns count as 0, and every box is divided by its full area. A box counts when it
    holds at least one convective column. Returns (sum over those boxes per quantity,
    number of those boxes, number of boxes in the domain).
    """
    omega = np.asarray(fields["VITESSE_VERT"], dtype=np.float64)
    if has_scheme:
        sig_u = np.minimum(nonnegative(fields["UD_MESH_FRAC"]), 1.0)
        om_u_rel = -nonnegative(-np.asarray(fields["UD_OMEGA"], dtype=np.float64))
        sig_d = np.minimum(nonnegative(fields["DD_MESH_FRAC"]), 1.0)
        om_d = nonnegative(fields["DD_OMEGA"])
        active = (sig_u > ACTIVE_MESH).any(axis=0)
    else:  # no 3MT: the draft fields are not meaningful
        sig_u = om_u_rel = sig_d = om_d = np.zeros_like(omega)
        active = np.ones(omega.shape[1:], dtype=bool)
    convective = (rain >= RAIN_MIN_MM_H) & active
    fluxes = column_fluxes(omega, sig_u, om_u_rel, sig_d, om_d)
    ny, nx, kept = box_layout(mask)
    n_conv = box_means(convective.astype(np.float64), ny, nx)
    use = kept & (n_conv > 0)
    out = {key: box_means(np.where(convective, value, 0.0), ny, nx)[:, use].sum(axis=1)
           for key, value in fluxes.items()}
    return out, int(use.sum()), int(kept.sum())


def build_window() -> tuple[int, int, int, int, np.ndarray]:
    """Crop window and mask of the 4 km radar domain on the raw FA grid."""
    return mask_window()


def init_worker(window: tuple[int, int, int, int, np.ndarray]) -> None:
    global _WINDOW
    import epygram

    epygram.init_env()
    _WINDOW = window


def read_fa_stack(resource, name: str) -> np.ndarray:
    """(level, masked points) for one FA field, S001 first."""
    y0, y1, x0, x1, mask = _WINDOW
    out = np.empty((N_LEVELS, int(mask.sum())), dtype=np.float64)
    for k in range(N_LEVELS):
        field = resource.readfield(f"S{k + 1:03d}{name}")
        if field.spectral:
            field.sp2gp()
        out[k] = np.asarray(field.getdata(), dtype=np.float64)[y0:y1, x0:x1][mask]
    return out


def _accumulators() -> tuple[dict, dict]:
    sums = {q: np.zeros((N_LEVELS, 24)) for q in QUANTITIES}
    counts = {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in QUANTITIES}
    return sums, counts


def fa_day(task: tuple[str, str]) -> tuple[dict, dict, int, tuple[str, ...]]:
    """Reduce one day of raw FA; profiles come back surface first."""
    import epygram

    root, day_token = task
    sums, counts = _accumulators()
    problems: list[str] = []
    n_files = 0
    for hour in range(24):
        path = Path(root) / f"pf{day_token}" / f"pfABOFABOF+{hour:04d}"
        if not path.exists():
            problems.append(f"{day_token} +{hour:04d}: missing")
            continue
        try:
            resource = epygram.formats.resource(str(path), "r")
            try:
                fields = {name: read_fa_stack(resource, name) for name in FIELDS}
            finally:
                resource.close()
            profiles = profiles_from_fields(fields)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} +{hour:04d}: {exc}")
            continue
        n_files += 1
        local_hour = (hour + UTC_OFFSET) % 24
        for key, profile in profiles.items():
            profile = profile[::-1]  # FA is top first; store surface first
            finite = np.isfinite(profile)
            sums[key][finite, local_hour] += profile[finite]
            counts[key][finite, local_hour] += 1
    return sums, counts, n_files, tuple(problems)


def read_fa_crop(resource, name: str, levels: bool = True) -> np.ndarray:
    """(level, y, x) crop of one FA field on the radar-domain window, S001 first.

    The whole window is returned, inside and outside the mask: 20 km boxes on the
    domain edge use every one of their cells.
    """
    y0, y1, x0, x1, _ = _WINDOW
    names = [f"S{k + 1:03d}{name}" for k in range(N_LEVELS)] if levels else [name]
    out = np.empty((len(names), y1 - y0, x1 - x0), dtype=np.float64)
    for k, full in enumerate(names):
        field = resource.readfield(full)
        if field.spectral:
            field.sp2gp()
        out[k] = np.asarray(field.getdata(), dtype=np.float64)[y0:y1, x0:x1]
    return out if levels else out[0]


def fa_day_boxes(task: tuple[str, str, bool]) -> tuple[dict, dict, int, tuple[str, ...]]:
    """One day of convective-box profiles; task = (root, day, has_scheme).

    Snapshots are the instantaneous fields at leads +0001..+0024, each paired with the
    rain of the hour ending at it (accumulated rain differenced between leads). Sums
    and box counts are binned by Amazon local hour; ``box_frequency`` sums the number
    of convective boxes and counts the number of boxes, so its mean is the fraction of
    box-hours with convection.
    """
    import epygram

    root, day_token, has_scheme = task
    mask = _WINDOW[4]
    names = DRAFT_FIELDS if has_scheme else ("VITESSE_VERT",)
    sums = {q: np.zeros((N_LEVELS, 24)) for q in BOX_QUANTITIES}
    counts = {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in BOX_QUANTITIES}
    problems: list[str] = []
    n_files = 0
    previous = None
    for lead in range(25):
        path = Path(root) / f"pf{day_token}" / f"pfABOFABOF+{lead:04d}"
        if not path.exists():
            problems.append(f"{day_token} +{lead:04d}: missing")
            previous = None
            continue
        try:
            resource = epygram.formats.resource(str(path), "r")
            try:
                accumulated = sum(read_fa_crop(resource, name, levels=False) for name in RAIN_FIELDS)
                fields = {} if lead == 0 else {name: read_fa_crop(resource, name) for name in names}
            finally:
                resource.close()
            if lead == 0 or previous is None:
                previous = accumulated
                continue
            rain = np.maximum(accumulated - previous, 0.0)
            previous = accumulated
            box_sums, n_conv, n_boxes = convective_boxes(fields, rain, mask, has_scheme)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} +{lead:04d}: {exc}")
            previous = None
            continue
        n_files += 1
        local_hour = (lead + UTC_OFFSET) % 24
        for key, profile in box_sums.items():
            sums[key][:, local_hour] += profile[::-1]  # FA is top first; store surface first
            counts[key][:, local_hour] += n_conv
        sums["box_frequency"][:, local_hour] += n_conv
        counts["box_frequency"][:, local_hour] += n_boxes
    return sums, counts, n_files, tuple(problems)


def fa_height(root: Path, day_token: str) -> np.ndarray:
    """Height above mean sea level per model level, surface first, in metres."""
    import epygram

    path = Path(root) / f"pf{day_token}" / "pfABOFABOF+0000"
    resource = epygram.formats.resource(str(path), "r")
    try:
        return (read_fa_stack(resource, "GEOPOTENTIEL").mean(axis=1) / G)[::-1]
    finally:
        resource.close()


# Domain-mean diurnal sections with the mass flux of the 3-piece column split
# (``column_fluxes``), i.e. the supervisor's definition rather than the updraft
# piece on its own. Same domain and binning as ``fa_day``: every cell of the radar
# domain counts, no convective or box conditioning.
PARTITION_FIELDS = (
    "UD_MESH_FRAC",
    "UD_OMEGA",
    "DD_MESH_FRAC",
    "DD_OMEGA",
    "VITESSE_VERT",
    "CLOUD_FRACTI",
    "TEMPERATURE",
)
# Without 3MT the draft fields carry no meaning (G2M-XCU), so they are not read.
PARTITION_FIELDS_NO_SCHEME = ("VITESSE_VERT", "CLOUD_FRACTI", "TEMPERATURE")
PARTITION_QUANTITIES = ("ud_extent", "flux_up", "flux_par", "flux_res", "cloud", "temp")


def partition_profiles_from_fields(
    fields: dict[str, np.ndarray], has_scheme: bool = True
) -> dict[str, np.ndarray]:
    """Domain-mean vertical profiles of one hour, mass flux from the column split.

    ``flux_up`` is the total upward mass flux of the column: the rising part of the
    3MT updraft, of the downdraft and of the remaining environment, each times its
    area (``column_fluxes``). It is not the updraft piece plus the resolved ascent -
    the resolved grid-mean omega is the area mean of all three. ``flux_par`` is the
    updraft piece alone and ``flux_res`` the rising resolved motion alone, both kept
    for reference.

    Without 3MT the column is one piece, so ``flux_up`` equals ``flux_res`` and the
    parameterized quantities come back NaN rather than a misleading zero.

    The draft fields are cleaned the way the physics cleans them on entry, as in
    ``profiles_from_fields``.
    """
    omega = np.asarray(fields["VITESSE_VERT"], dtype=np.float64)
    if has_scheme:
        sig_u = np.minimum(nonnegative(fields["UD_MESH_FRAC"]), 1.0)
        om_u_rel = -nonnegative(-np.asarray(fields["UD_OMEGA"], dtype=np.float64))
        sig_d = np.minimum(nonnegative(fields["DD_MESH_FRAC"]), 1.0)
        om_d = nonnegative(fields["DD_OMEGA"])
    else:
        sig_u = om_u_rel = sig_d = om_d = np.zeros_like(omega)
    fluxes = column_fluxes(omega, sig_u, om_u_rel, sig_d, om_d)
    absent = np.full_like(omega, np.nan)
    parts = {
        "ud_extent": sig_u if has_scheme else absent,
        "flux_up": fluxes["box_up"],
        "flux_par": fluxes["box_par_up"] if has_scheme else absent,
        "flux_res": fluxes["box_res_up"],
        "cloud": fields["CLOUD_FRACTI"],
        "temp": fields["TEMPERATURE"],
    }
    axis = tuple(range(1, omega.ndim))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {key: np.nanmean(values, axis=axis) for key, values in parts.items()}


def fa_day_partition(task: tuple[str, str, bool]) -> tuple[dict, dict, int, tuple[str, ...]]:
    """One day of domain-mean partition profiles; task = (root, day, has_scheme)."""
    import epygram

    root, day_token, has_scheme = task
    names = PARTITION_FIELDS if has_scheme else PARTITION_FIELDS_NO_SCHEME
    sums = {q: np.zeros((N_LEVELS, 24)) for q in PARTITION_QUANTITIES}
    counts = {q: np.zeros((N_LEVELS, 24), dtype=np.int64) for q in PARTITION_QUANTITIES}
    problems: list[str] = []
    n_files = 0
    for hour in range(24):
        path = Path(root) / f"pf{day_token}" / f"pfABOFABOF+{hour:04d}"
        if not path.exists():
            problems.append(f"{day_token} +{hour:04d}: missing")
            continue
        try:
            resource = epygram.formats.resource(str(path), "r")
            try:
                fields = {name: read_fa_stack(resource, name) for name in names}
            finally:
                resource.close()
            profiles = partition_profiles_from_fields(fields, has_scheme)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} +{hour:04d}: {exc}")
            continue
        n_files += 1
        local_hour = (hour + UTC_OFFSET) % 24
        for key, profile in profiles.items():
            profile = profile[::-1]  # FA is top first; store surface first
            finite = np.isfinite(profile)
            sums[key][finite, local_hour] += profile[finite]
            counts[key][finite, local_hour] += 1
    return sums, counts, n_files, tuple(problems)
