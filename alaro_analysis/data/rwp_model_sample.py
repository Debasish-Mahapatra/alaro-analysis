"""The runs sampled the way the T3 radar wind profiler samples the atmosphere.

Model side of the comparisons with the profiler (supervisor's note, sections 1 and 2; the
observed side is ``data.rwp_profiler``). Each hourly snapshot of the 4 km radar domain is
cut into the 20 km boxes of ``convection_sections`` and each column into 3MT updraft,
downdraft and the rest (``column_pieces``), as for the box mass flux, with three changes
that bring the model closer to what the profiler sees:

* echo: a level counts only where the model's own reflectivity reaches 10 dBZ, because the
  profiler retrieves w only inside precipitation echo and sees nothing above echo top;
* the convective column is defined several ways (``VARIANTS``): rain of the hour of at
  least 1, 5 or 10 mm, or Steiner-convective at 3 km, always with an active 3MT updraft in
  the runs that have one; the former definition (rain >= 1 mm, every level) is kept to tie
  the result back to the earlier one;
* the w of every piece goes into a histogram with its area (and area x density) as weight,
  so the random retrieval noise can be added afterwards, analytically, at any level.

Fields: the draft fields and omega from raw FA, reflectivity, temperature, pressure,
humidity, height and the rain of the hour from the IOP microphysics NetCDF
(``build_iop_microphysics_fields``), which hold the same snapshots on the same window.
The day worker lives here, not in the recipe, so a process pool can pickle it.

The supervisor's second scale is here too (``fa_day_rwp_blocks``): 60 km boxes (15 x 15
cells) over three consecutive hourly snapshots, the counterpart of the profiler's 3 h
window. A box-window counts when convective columns cover at least five minutes' worth
of it (5/180, the profiler's rule), and its flux is the mean over the three snapshots,
each divided by the full box.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path

import numpy as np

from alaro_analysis.analysis.radar_echo import ECHO_DBZ, interpolate_to_heights, steiner_classes
from alaro_analysis.common.constants import G
from alaro_analysis.data import convection_sections
from alaro_analysis.data.convection_sections import (
    ACTIVE_MESH,
    BOX_CELLS,
    DRAFT_FIELDS,
    box_layout,
    box_means,
    column_pieces,
    nonnegative,
    read_fa_crop,
)
from alaro_analysis.data.rwp_profiler import CONVECTIVE_MINUTES, W_EDGES

R_DRY = 287.058
VIRTUAL = 0.6078  # T_v = T (1 + 0.6078 q)
# name -> (minimum rain of the hour in mm or None, Steiner-convective at 3 km, echo levels only)
VARIANTS = {
    "rain1_all": (1.0, False, False),
    "rain1": (1.0, False, True),
    "rain5": (5.0, False, True),
    "rain10": (10.0, False, True),
    "steiner": (None, True, True),
}
LEVEL_KEYS = ("up", "down", "up_area", "down_area", "ud_w1", "ud_w2")
HIST_KEYS = ("hist_area", "hist_mass", "hist_grid")
SCALAR_KEYS = ("boxes", "boxes_all", "conv_share")
BLOCK_SNAPSHOTS = 3  # snapshots per window at the 60 km scale (3 h at 5 m s-1)
BLOCK_CELLS = 15  # 60 km boxes
# A 60 km box-window counts when its convective columns, averaged over the three snapshots,
# cover at least the share of the window the profiler needs: five minutes of convective
# echo in three hours.
MIN_WINDOW_SHARE = CONVECTIVE_MINUTES / (BLOCK_SNAPSHOTS * 60.0)
NC_FIELDS = ("dbz", "height", "t", "p", "q", "rain_con", "rain_gec")


def air_density(t: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Moist air density (kg m-3) from temperature (K), pressure (Pa) and specific humidity."""
    return p / (R_DRY * t * (1.0 + VIRTUAL * q))


def velocity_bins(w: np.ndarray, edges: np.ndarray = W_EDGES) -> np.ndarray:
    """Histogram bin of each velocity; beyond the outer edges -> the end bins."""
    return np.clip(np.digitize(np.nan_to_num(w), edges) - 1, 0, edges.size - 2)


def box_shares(rain, dbz_3km, mask, drafts=None, size: int = BOX_CELLS) -> tuple[np.ndarray, dict]:
    """(boxes kept, {variant: share of each box's cells that are convective columns})."""
    active = (drafts[0] > ACTIVE_MESH).any(axis=0) if drafts is not None else np.ones(mask.shape, bool)
    steiner = steiner_classes(dbz_3km, np.ones_like(mask, dtype=bool)) == 2
    ny, nx, kept = box_layout(mask, size)
    yy, xx = np.indices(mask.shape)
    tiled = (yy < ny) & (xx < nx)
    shares = {}
    for name, (rain_min, use_steiner, _) in VARIANTS.items():
        base = steiner if use_steiner else rain >= rain_min
        shares[name] = box_means((base & active & tiled).astype(np.float64), ny, nx, size)
    return kept, shares


def snapshot_sample(omega, rho, dbz, rain, dbz_3km, mask, drafts=None, size: int = BOX_CELLS,
                    allowed: dict | None = None) -> dict[str, dict]:
    """Profiler-like statistics of one snapshot, per convective-column variant.

    omega, rho, dbz: (level, y, x); rain, dbz_3km, mask: (y, x); drafts: (sig_u, om_u_rel,
    sig_d, om_d) cleaned as the physics cleans them, or None without 3MT. Every quantity is
    a sum over the boxes (``size`` x ``size`` cells, 20 km by default) that hold a convective
    column, each column's pieces weighted by their share of the full box. Per variant:

    up, down            upward / downward mass flux (kg m-2 s-1, both >= 0)
    up_area, down_area  area of the rising / sinking pieces
    hist_area           (level, bin) piece area by piece w (m s-1)
    hist_mass           the same weighted by area x density, for mass fluxes with noise
    hist_grid           cell area by resolved (grid-mean) w
    ud_w1, ud_w2        sum of sig_u rho w_u and sig_u rho w_u^2 over rising 3MT updrafts
    boxes, boxes_all, conv_share  boxes used, boxes in the domain, and the summed share of
                        convective columns in the used boxes
    ``out["_box_conv"]`` holds, per variant, which boxes held a convective column.
    ``allowed`` ({variant: boxes}) restricts the boxes further (the 60 km windows).
    """
    n_levels = omega.shape[0]
    n_bins = W_EDGES.size - 1
    if drafts is not None:
        sig_u, om_u_rel, sig_d, om_d = drafts
        active = (sig_u > ACTIVE_MESH).any(axis=0)
    else:
        sig_u = om_u_rel = sig_d = om_d = np.zeros_like(omega)
        active = np.ones(omega.shape[1:], dtype=bool)
    areas, omegas, _ = column_pieces(omega, sig_u, om_u_rel, sig_d, om_d)
    areas = np.stack(areas)
    omegas = np.stack(omegas)
    w_pieces = -omegas / (rho * G)
    w_grid = -omega / (rho * G)
    level = np.arange(n_levels)[:, None, None]
    piece_bin = (level * n_bins + velocity_bins(w_pieces)).ravel()
    grid_bin = (level * n_bins + velocity_bins(w_grid)).ravel()
    updraft = (sig_u > ACTIVE_MESH) & (w_pieces[0] > 0)
    echo = np.nan_to_num(dbz, nan=-99.0) >= ECHO_DBZ
    steiner = steiner_classes(dbz_3km, np.ones_like(mask, dtype=bool)) == 2

    ny, nx, kept = box_layout(mask, size)
    yy, xx = np.indices(mask.shape)
    tiled = (yy < ny) & (xx < nx)
    box_y = np.minimum(yy // size, kept.shape[0] - 1)
    box_x = np.minimum(xx // size, kept.shape[1] - 1)
    n_cells = n_levels * n_bins
    out = {"_box_conv": {}}
    for name, (rain_min, use_steiner, echo_only) in VARIANTS.items():
        base = steiner if use_steiner else rain >= rain_min
        convective = base & active & tiled
        share = box_means(convective.astype(np.float64), ny, nx, size)
        use = kept & (share > 0)
        if allowed is not None:
            use = use & allowed[name]
        out["_box_conv"][name] = use
        cells = convective & use[box_y, box_x]
        weight = np.broadcast_to(np.where(cells, 1.0 / size**2, 0.0), omega.shape)
        if echo_only:
            weight = weight * echo
        piece_weight = areas * weight
        mass_weight = piece_weight * rho
        out[name] = {
            "up": -(piece_weight * np.minimum(omegas, 0.0)).sum(axis=(0, 2, 3)) / G,
            "down": (piece_weight * np.maximum(omegas, 0.0)).sum(axis=(0, 2, 3)) / G,
            "up_area": (piece_weight * (omegas < 0)).sum(axis=(0, 2, 3)),
            "down_area": (piece_weight * (omegas > 0)).sum(axis=(0, 2, 3)),
            "ud_w1": (mass_weight[0] * updraft * w_pieces[0]).sum(axis=(1, 2)),
            "ud_w2": (mass_weight[0] * updraft * w_pieces[0] ** 2).sum(axis=(1, 2)),
            "hist_area": np.bincount(piece_bin, piece_weight.ravel(), n_cells).reshape(n_levels, n_bins),
            "hist_mass": np.bincount(piece_bin, mass_weight.ravel(), n_cells).reshape(n_levels, n_bins),
            "hist_grid": np.bincount(grid_bin, np.ascontiguousarray(weight).ravel(), n_cells)
            .reshape(n_levels, n_bins),
            "boxes": float(use.sum()),
            "boxes_all": float(kept.sum()),
            "conv_share": float(share[use].sum()),
        }
    return out


def empty_sums(n_levels: int) -> dict:
    n_bins = W_EDGES.size - 1
    sums = {name: {**{k: np.zeros(n_levels) for k in LEVEL_KEYS},
                   **{k: np.zeros((n_levels, n_bins)) for k in HIST_KEYS},
                   **{k: 0.0 for k in SCALAR_KEYS}} for name in VARIANTS}
    sums["height_agl_sum"] = np.zeros(n_levels)
    sums["snapshots"] = 0
    return sums


def add_sums(total: dict, part: dict) -> dict:
    """In-place sum of two ``empty_sums`` structures; returns ``total``."""
    for name in VARIANTS:
        for key, value in part[name].items():
            total[name][key] = total[name][key] + value
    total["height_agl_sum"] = total["height_agl_sum"] + part["height_agl_sum"]
    total["snapshots"] += part["snapshots"]
    return total


def _snapshot(fa_root, day_token, lead, times, start, nc, names, has_scheme, orography, mask):
    """(omega, rho, dbz, rain, dbz_3km, drafts) of one lead, its mean AGL height, the orography."""
    import epygram

    index = np.flatnonzero(times == start + np.timedelta64(lead, "h"))
    path = Path(fa_root) / f"pf{day_token}" / f"pfABOFABOF+{lead:04d}"
    if index.size != 1 or not path.exists():
        raise FileNotFoundError("missing snapshot")
    i = int(index[0])
    resource = epygram.formats.resource(str(path), "r")
    try:
        fields = {name: read_fa_crop(resource, name) for name in names}
        if orography is None:
            orography = read_fa_crop(resource, "SURFGEOPOTEN", levels=False) / G
    finally:
        resource.close()
    level = {name: np.asarray(nc[name][i], dtype=np.float64) for name in ("dbz", "height", "t", "p", "q")}
    rho = air_density(level["t"], level["p"], level["q"])
    rain = np.asarray(nc["rain_con"][i] + nc["rain_gec"][i], dtype=np.float64)
    dbz_3km = interpolate_to_heights(level["dbz"], level["height"], [3.0])[0]
    drafts = None
    if has_scheme:
        drafts = (
            np.minimum(nonnegative(fields["UD_MESH_FRAC"]), 1.0),
            -nonnegative(-fields["UD_OMEGA"]),
            np.minimum(nonnegative(fields["DD_MESH_FRAC"]), 1.0),
            nonnegative(fields["DD_OMEGA"]),
        )
    agl = (level["height"] - orography[None])[:, mask].mean(axis=1)
    return (fields["VITESSE_VERT"], rho, level["dbz"], rain, dbz_3km, drafts), agl, orography


def _open_day(nc_path):
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        return ds["time"].values, {name: ds[name].values for name in NC_FIELDS}


def fa_day_rwp(task) -> tuple[dict, int, tuple[str, ...]]:
    """One day of profiler-like statistics; task = (fa_root, nc_path, day, has_scheme, leads).

    ``leads`` are the leads (1..24) whose valid time ends an hour the profiler observed.
    Returns (sums, snapshots, problems); levels stay in FA order (top first).
    """
    fa_root, nc_path, day_token, has_scheme, leads = task
    mask = convection_sections._WINDOW[4]
    sums = empty_sums(convection_sections.N_LEVELS)
    problems: list[str] = []
    times, nc = _open_day(nc_path)
    names = DRAFT_FIELDS if has_scheme else ("VITESSE_VERT",)
    orography = None
    start = np.datetime64(f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:]}T00")
    for lead in leads:
        try:
            inputs, agl, orography = _snapshot(fa_root, day_token, lead, times, start, nc, names,
                                               has_scheme, orography, mask)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} +{lead:04d}: {exc!r}")
            continue
        part = snapshot_sample(*inputs[:5], mask, inputs[5])
        for name in VARIANTS:
            for key, value in part[name].items():
                sums[name][key] = sums[name][key] + value
        sums["height_agl_sum"] += agl
        sums["snapshots"] += 1
    return sums, sums["snapshots"], tuple(problems)


def fa_day_rwp_blocks(task) -> tuple[dict, int, tuple[str, ...]]:
    """One day at the 60 km scale; task = (fa_root, nc_path, day, has_scheme, blocks).

    Block k is the 3 h window [3k, 3k + 3) UTC, sampled by the snapshots at leads 3k+1 ..
    3k+3 (each ends one of its hours). Per variant, a box-window counts when its convective
    columns, averaged over the three snapshots, cover at least ``MIN_WINDOW_SHARE`` of it
    (the profiler's five minutes in three hours); the sums run over the three snapshots and
    ``boxes`` counts box-snapshots of the counted windows, so sum / boxes is the window-mean
    flux of a 60 km box divided by its full area. A block with a missing snapshot is skipped.
    """
    fa_root, nc_path, day_token, has_scheme, blocks = task
    mask = convection_sections._WINDOW[4]
    sums = empty_sums(convection_sections.N_LEVELS)
    problems: list[str] = []
    times, nc = _open_day(nc_path)
    names = DRAFT_FIELDS if has_scheme else ("VITESSE_VERT",)
    orography = None
    start = np.datetime64(f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:]}T00")
    for k in blocks:
        block = empty_sums(convection_sections.N_LEVELS)
        try:
            snaps = [_snapshot(fa_root, day_token, lead, times, start, nc, names, has_scheme, orography, mask)
                     for lead in range(BLOCK_SNAPSHOTS * k + 1, BLOCK_SNAPSHOTS * (k + 1) + 1)]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{day_token} block {k}: {exc!r}")
            continue
        orography = snaps[-1][2]
        # the window's convective share per box, then only the boxes that reach the profiler's rule
        share = {name: 0.0 for name in VARIANTS}
        for inputs, _, _ in snaps:
            kept, shares = box_shares(inputs[3], inputs[4], mask, inputs[5], BLOCK_CELLS)
            for name in VARIANTS:
                share[name] = share[name] + shares[name] / BLOCK_SNAPSHOTS
        allowed = {name: kept & (share[name] >= MIN_WINDOW_SHARE) for name in VARIANTS}
        for inputs, agl, _ in snaps:
            part = snapshot_sample(*inputs[:5], mask, inputs[5], size=BLOCK_CELLS, allowed=allowed)
            for name in VARIANTS:
                for key, value in part[name].items():
                    block[name][key] = block[name][key] + value
            block["height_agl_sum"] += agl
            block["snapshots"] += 1
        for name in VARIANTS:
            block[name]["boxes"] = float(BLOCK_SNAPSHOTS * allowed[name].sum())
        add_sums(sums, block)
    return sums, sums["snapshots"], tuple(problems)


def noisy_fluxes(hist_mass: np.ndarray, hist_area: np.ndarray, noise, edges: np.ndarray = W_EDGES):
    """Upward / downward mass flux and rising / sinking area with Gaussian noise added to w.

    Every piece of velocity w carries independent noise e ~ N(0, noise), as each profiler
    sample does, so its expected rising flux is rho * area * E[max(w + e, 0)]
    = rho * area * (w Phi(w/s) + s phi(w/s)) and its rising area area * Phi(w/s). The net
    flux is unchanged. ``noise`` is a scalar or one value per level (m s-1); 0 gives the
    histograms' own split. Inputs (level, bin); returns four (level,) arrays.
    """
    from scipy.special import ndtr

    centres = 0.5 * (edges[:-1] + edges[1:])[None, :]
    s = np.broadcast_to(np.asarray(noise, dtype=np.float64), (hist_mass.shape[0],))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(s > 0, centres / np.where(s > 0, s, 1.0), np.sign(centres) * np.inf)
    up_cdf = ndtr(x)
    pdf = np.where(s > 0, np.exp(-0.5 * np.where(np.isfinite(x), x, 0.0) ** 2) / np.sqrt(2 * np.pi), 0.0)
    rising = centres * up_cdf + s * pdf
    sinking = -centres * (1.0 - up_cdf) + s * pdf
    return ((hist_mass * rising).sum(axis=1), (hist_mass * sinking).sum(axis=1),
            (hist_area * up_cdf).sum(axis=1), (hist_area * (1.0 - up_cdf)).sum(axis=1))


def add_noise(hist: np.ndarray, noise, edges: np.ndarray = W_EDGES) -> np.ndarray:
    """(level, bin) histograms convolved with Gaussian noise of ``noise`` m s-1 (scalar or per level)."""
    width = edges[1] - edges[0]
    s = np.broadcast_to(np.asarray(noise, dtype=np.float64), (hist.shape[0],))
    out = np.empty_like(hist, dtype=np.float64)
    offsets = np.arange(-hist.shape[1] + 1, hist.shape[1]) * width
    for k, (row, sk) in enumerate(zip(hist, s)):
        if sk <= 0:
            out[k] = row
            continue
        kernel = np.exp(-0.5 * (offsets / sk) ** 2)
        full = np.convolve(row, kernel / kernel.sum(), mode="full")
        # keep the original bins; mass pushed past the ends goes into the end bins
        centre = full[hist.shape[1] - 1: 2 * hist.shape[1] - 1].copy()
        centre[0] += full[: hist.shape[1] - 1].sum()
        centre[-1] += full[2 * hist.shape[1] - 1:].sum()
        out[k] = centre
    return out
