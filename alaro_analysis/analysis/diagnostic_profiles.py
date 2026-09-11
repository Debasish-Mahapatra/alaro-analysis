"""Diagnostic profiles. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from alaro_analysis.analysis.derived import (
    choose_bottom_level_index,
    compute_dp_pa,
    compute_theta_e_field,
    maybe_convert_pressure_to_pa,
)
from alaro_analysis.analysis.profiles import (
    finalize_line_means,
    finalize_profile_means,
    line_hour_accumulate,
    profile_hour_accumulate,
)
from alaro_analysis.analysis.updrafts import downdraft_mass_flux, updraft_mass_flux
from alaro_analysis.common.constants import CP_D, LV, G
from alaro_analysis.common.models import SpatialWindow, VerticalAxis
from alaro_analysis.common.naming import normalize_var_token
from alaro_analysis.common.spatial import apply_spatial_window_to_array
from alaro_analysis.data.dataset_io import (
    nanmean_with_count,
    read_time_level_yx,
    read_vertical_profile,
    resolve_data_var_name,
)
from alaro_analysis.data.discovery import collect_file_records


def align_tlyx_shapes(
    arrays: Sequence[np.ndarray], *, allow_truncate: bool = False
) -> list[np.ndarray]:
    """Validate aligned T/L/Y/X fields; historical truncation must be requested."""
    if not arrays:
        return []
    if any(arr.ndim != 4 for arr in arrays):
        raise ValueError("All fields must have (time, level, y, x) dimensions")
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        return list(arrays)
    if not allow_truncate:
        raise ValueError(f"Incompatible field shapes {shapes}; align coordinates before computing")
    import warnings

    warnings.warn(f"Legacy truncation of mismatched field shapes {shapes}", stacklevel=2)
    slices = tuple(slice(0, min(sizes)) for sizes in zip(*shapes))
    return [arr[slices] for arr in arrays]


def mean_profile(field_tlyx: np.ndarray) -> np.ndarray:
    mean, _ = nanmean_with_count(field_tlyx, axis=(0, 2, 3))
    return np.asarray(mean, dtype=np.float64)


def compute_geopotential_height_profile(
    geopotential_dir: Path,
    height_variable: str,
    max_days: int | None,
    allowed_months: tuple[int, ...] | None,
    utc_offset_hours: int,
    aggregate: str,
    spatial_window: SpatialWindow,
) -> tuple[np.ndarray, int]:
    records = collect_file_records(
        variable_dir=geopotential_dir,
        max_days=max_days,
        allowed_months=allowed_months,
        utc_offset_hours=utc_offset_hours,
    )
    if not records:
        raise RuntimeError(f"No valid geopotential files found in {geopotential_dir}")

    if aggregate == "first":
        profile, _ = read_vertical_profile(
            records[0][1],
            height_variable,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
            force_time_level_yx=True,
        )
        return profile, 1

    first, _ = read_vertical_profile(
        records[0][1],
        height_variable,
        spatial_window=spatial_window,
        token_normalizer=normalize_var_token,
        force_time_level_yx=True,
    )
    sums = np.zeros_like(first, dtype=np.float64)
    counts = np.zeros_like(first, dtype=np.int64)
    for idx, (_, file_path) in enumerate(records, start=1):
        profile, _ = read_vertical_profile(
            file_path,
            height_variable,
            spatial_window=spatial_window,
            token_normalizer=normalize_var_token,
            force_time_level_yx=True,
        )
        n = min(sums.size, profile.size)
        valid = np.isfinite(profile[:n])
        sums[:n][valid] += profile[:n][valid]
        counts[:n][valid] += 1
        if idx % 4000 == 0 or idx == len(records):
            print(
                f"[{geopotential_dir.parent.name}/GEOPOTENTIEL] {idx}/{len(records)} files",
                flush=True,
            )

    mean = np.full_like(sums, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    return mean, len(records)


def align_axis_and_profile(
    axis: VerticalAxis,
    profile: np.ndarray,
) -> tuple[VerticalAxis, np.ndarray]:
    n = min(axis.values.size, profile.shape[0])
    axis_new = VerticalAxis(
        values=axis.values[:n], label=axis.label, is_height_km=axis.is_height_km
    )
    return axis_new, profile[:n, :]


def get_peer_file(base_file: Path, experiment_dir: Path, var_name: str) -> Path:
    return experiment_dir / var_name / base_file.parent.name / base_file.name


def compute_downdraft_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        req = (
            get_peer_file(base_file, experiment_dir, names["UD_OMEGA"]),
            get_peer_file(base_file, experiment_dir, names["UD_MESH_FRAC"]),
            get_peer_file(base_file, experiment_dir, names["DD_OMEGA"]),
            get_peer_file(base_file, experiment_dir, names["DD_MESH_FRAC"]),
        )
        if not all(p.exists() for p in req):
            return None
        ud_omega, ud_mesh, dd_omega, dd_mesh = align_tlyx_shapes(
            [
                read_time_level_yx(
                    req[0],
                    names["UD_OMEGA"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    req[1],
                    names["UD_MESH_FRAC"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    req[2],
                    names["DD_OMEGA"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    req[3],
                    names["DD_MESH_FRAC"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
            ]
        )

        # Mass flux: M_d = sigma_d * omega_d / g
        # DD omega is positive (downward in pressure coords), so flux is positive.
        # We only care where mesh > 0 (active downdrafts).
        dd_flux = downdraft_mass_flux(dd_omega, dd_mesh)

        # UD mass flux: M_u = - sigma_u * omega_u / g
        # UD omega is negative (upward), so -omega gives positive flux.
        ud_flux = updraft_mass_flux(ud_omega, ud_mesh)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dd_flux / ud_flux
        ratio = np.where(np.abs(ud_flux) < 1e-10, np.nan, ratio)

        # DD intensity: |omega| only where downdraft is active
        dd_omega_abs = np.where(dd_mesh > 0, np.abs(dd_omega), np.nan)

        return {
            "DD_EXTENT": mean_profile(dd_mesh),
            "DD_FLUX": mean_profile(dd_flux),
            "DD_INTENSITY": mean_profile(dd_omega_abs),
            "DD_TO_UD_FLUX_RATIO": mean_profile(ratio),
            "NET_CONVECTIVE_FLUX": mean_profile(ud_flux - dd_flux),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/downdraft",
    )
    print(f"[{experiment}/downdraft] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_precip_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        cv_file = get_peer_file(base_file, experiment_dir, names["CV_PREC_FLUX"])
        st_file = get_peer_file(base_file, experiment_dir, names["ST_PREC_FLUX"])
        if not cv_file.exists() or not st_file.exists():
            return None
        cv, st = align_tlyx_shapes(
            [
                read_time_level_yx(
                    cv_file,
                    names["CV_PREC_FLUX"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    st_file,
                    names["ST_PREC_FLUX"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
            ]
        )
        total = cv + st
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = cv / total
        frac = np.where(np.abs(total) < 1e-12, np.nan, frac)
        return {
            "CV_PREC_FLUX": mean_profile(cv),
            "ST_PREC_FLUX": mean_profile(st),
            "TOTAL_PREC_FLUX": mean_profile(total),
            "CONVECTIVE_PRECIP_FRACTION": mean_profile(frac),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/precip",
    )
    print(f"[{experiment}/precip] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_thermo_profiles(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    z_m: np.ndarray,
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def reader(base_file: Path):
        t_file = get_peer_file(base_file, experiment_dir, names["TEMPERATURE"])
        q_file = get_peer_file(base_file, experiment_dir, names["HUMI.SPECIFI"])
        p_file = get_peer_file(base_file, experiment_dir, names["PRESSURE"])
        if not t_file.exists() or not q_file.exists() or not p_file.exists():
            return None
        t, q, p = align_tlyx_shapes(
            [
                read_time_level_yx(
                    t_file,
                    names["TEMPERATURE"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    q_file,
                    names["HUMI.SPECIFI"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
                read_time_level_yx(
                    p_file,
                    names["PRESSURE"],
                    spatial_window,
                    token_normalizer=normalize_var_token,
                ),
            ]
        )
        nlev = min(t.shape[1], z_m.size)
        t = t[:, :nlev, :, :]
        q = q[:, :nlev, :, :]
        p = p[:, :nlev, :, :]
        z = np.asarray(z_m[:nlev], dtype=np.float64)[None, :, None, None]

        theta_e = compute_theta_e_field(t, q, p)
        mse = CP_D * t + LV * q + G * z
        return {
            "THETA_E": mean_profile(theta_e),
            "MSE": mean_profile(mse),
        }

    sums, counts, used = profile_hour_accumulate(
        records=records,
        profile_reader=reader,
        progress_tag=f"{experiment}/thermo",
    )
    print(f"[{experiment}/thermo] used files: {used}/{len(records)}", flush=True)
    return finalize_profile_means(sums, counts)


def compute_kt273_lines(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def safe_scalar_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def mean_scalar(arr: np.ndarray) -> float:
        if arr.ndim >= 2:
            arr = apply_spatial_window_to_array(arr, spatial_window, Path("<in-memory>"))
        return safe_scalar_mean(arr)

    def reader(base_file: Path):
        required = (
            "KT273GRAUPEL",
            "KT273DD_OMEGA",
            "KT273DD_MESH_FRA",
            "KT273RAIN",
            "KT273HUMI.SPECIF",
        )
        files = {k: get_peer_file(base_file, experiment_dir, names[k]) for k in required}
        if not all(fp.exists() for fp in files.values()):
            return None

        vals: dict[str, float] = {}
        for key in required:
            with xr.open_dataset(files[key], decode_times=False) as ds:
                vn = resolve_data_var_name(
                    ds,
                    names[key],
                    token_normalizer=normalize_var_token,
                )
                vals[key] = mean_scalar(np.asarray(ds[vn].values, dtype=np.float64))

        # Downdraft mass flux: M_dd = -omega_dd * sigma_dd / g  [kg m-2 s-1]
        # omega_dd is downward-positive in pressure coords (ω > 0 = subsidence),
        # so the negative sign makes M_dd positive for active downdrafts.
        return {
            "KT273GRAUPEL": vals["KT273GRAUPEL"],
            "KT273DD_FLUX": -(vals["KT273DD_OMEGA"] * vals["KT273DD_MESH_FRA"]) / G,
            "KT273RAIN": vals["KT273RAIN"],
            "KT273HUMI_SPECIF": vals["KT273HUMI.SPECIF"],
        }

    sums, counts, used = line_hour_accumulate(
        records=records,
        line_reader=reader,
        progress_tag=f"{experiment}/kt273",
    )
    print(f"[{experiment}/kt273] used files: {used}/{len(records)}", flush=True)
    return finalize_line_means(sums, counts)


def compute_column_lines(
    experiment: str,
    experiment_dir: Path,
    records: list[tuple[int, Path]],
    names: dict[str, str],
    available_condensate_names: list[str],
    spatial_window: SpatialWindow,
) -> dict[str, np.ndarray]:
    def safe_scalar_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def reader(base_file: Path):
        p_file = get_peer_file(base_file, experiment_dir, names["PRESSURE"])
        cv_file = get_peer_file(base_file, experiment_dir, names["CV_PREC_FLUX"])
        st_file = get_peer_file(base_file, experiment_dir, names["ST_PREC_FLUX"])
        if not p_file.exists() or not cv_file.exists() or not st_file.exists():
            return None

        p = read_time_level_yx(
            p_file,
            names["PRESSURE"],
            spatial_window,
            token_normalizer=normalize_var_token,
        )
        p = maybe_convert_pressure_to_pa(p)
        cv = read_time_level_yx(
            cv_file,
            names["CV_PREC_FLUX"],
            spatial_window,
            token_normalizer=normalize_var_token,
        )
        st = read_time_level_yx(
            st_file,
            names["ST_PREC_FLUX"],
            spatial_window,
            token_normalizer=normalize_var_token,
        )
        p, cv, st = align_tlyx_shapes([p, cv, st])
        total_prec = cv + st

        q_fields: list[np.ndarray] = []
        for q_name in available_condensate_names:
            q_file = get_peer_file(base_file, experiment_dir, q_name)
            if not q_file.exists():
                continue
            q_arr = read_time_level_yx(
                q_file,
                q_name,
                spatial_window,
                token_normalizer=normalize_var_token,
            )
            q_fields.append(q_arr)
        if not q_fields:
            return None

        aligned = align_tlyx_shapes([p, total_prec, *q_fields])
        p = aligned[0]
        total_prec = aligned[1]
        q_fields = aligned[2:]

        q_sum = np.zeros_like(p, dtype=np.float64)
        for q_arr in q_fields:
            q_sum += q_arr

        dp = compute_dp_pa(p)
        column_cond = np.nansum(q_sum * dp / G, axis=1)  # (time, y, x)
        col_mean = safe_scalar_mean(column_cond)

        bottom_idx = choose_bottom_level_index(p)
        surface_flux = total_prec[:, bottom_idx, :, :]
        surface_mean = safe_scalar_mean(surface_flux)

        with np.errstate(divide="ignore", invalid="ignore"):
            residence = col_mean / surface_mean
        if not np.isfinite(residence):
            residence = np.nan

        return {
            "COLUMN_CONDENSATE": col_mean,
            "SURFACE_PRECIP_FLUX": surface_mean,
            "RESIDENCE_TIME": float(residence),
        }

    sums, counts, used = line_hour_accumulate(
        records=records,
        line_reader=reader,
        progress_tag=f"{experiment}/column",
    )
    print(f"[{experiment}/column] used files: {used}/{len(records)}", flush=True)
    return finalize_line_means(sums, counts)


def compute_vertical_derivative(profile: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    z = np.asarray(height_km, dtype=np.float64) * 1000.0  # m
    out = np.full(profile.shape, np.nan, dtype=np.float64)
    for hour in range(profile.shape[1]):
        col = profile[:, hour]
        finite = np.isfinite(z) & np.isfinite(col)
        if np.sum(finite) < 3:
            continue
        idx = np.where(finite)[0]
        zf = z[idx]
        vf = col[idx]
        order = np.argsort(zf)
        idx = idx[order]
        zf = zf[order]
        vf = vf[order]
        if np.any(np.diff(zf) <= 0):
            unique = np.concatenate(([True], np.diff(zf) > 0.0))
            idx = idx[unique]
            zf = zf[unique]
            vf = vf[unique]
        if zf.size < 3:
            continue
        grad = np.gradient(vf, zf)
        full = np.full(col.shape, np.nan, dtype=np.float64)
        full[idx] = grad
        out[:, hour] = full
    return out


def compute_bl_gradient_line(
    theta_e_profile: np.ndarray,
    height_km: np.ndarray,
    zmax_km: float = 2.0,
) -> np.ndarray:
    z = np.asarray(height_km, dtype=np.float64)
    line = np.full((theta_e_profile.shape[1],), np.nan, dtype=np.float64)
    for hour in range(theta_e_profile.shape[1]):
        v = theta_e_profile[:, hour]
        finite = np.isfinite(z) & np.isfinite(v) & (z >= 0.0) & (z <= zmax_km)
        if np.sum(finite) < 2:
            continue
        zz = z[finite]
        vv = v[finite]
        order = np.argsort(zz)
        zz = zz[order]
        vv = vv[order]
        coeff = np.polyfit(zz, vv, deg=1)
        line[hour] = coeff[0]  # K/km
    return line
