"""Dsd profile cache. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alaro_analysis.common.dsd_models import (
    DSD_FIELDS,
    DdhQvProfile,
    DsdVerticalProfile,
)
from alaro_analysis.data.cache import save_npz_compressed


def save_dsd_profile(path: Path, profile: DsdVerticalProfile, sig: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_compressed(
        path,
        height_km=profile.height_km,
        temperature_k=profile.temperature_k,
        temperature_count=profile.temperature_count,
        n_files=np.asarray([profile.n_files], dtype=np.int64),
        source=np.asarray([profile.source]),
        __signature__=np.asarray([sig]),
        **{field: profile.values[field] for field in DSD_FIELDS},
        **{f"count__{field}": profile.counts[field] for field in DSD_FIELDS},
    )


def load_dsd_profile(path: Path) -> DsdVerticalProfile:
    with np.load(path, allow_pickle=False) as data:
        source = str(data["source"][0]) if "source" in data.files else str(path)
        n_files_raw = data["n_files"] if "n_files" in data.files else np.asarray([0])
        n_files = int(np.ravel(n_files_raw)[0]) if np.size(n_files_raw) else 0
        height = np.asarray(data["height_km"], dtype=np.float64)
        if "temperature_k" in data.files:
            temperature = np.asarray(data["temperature_k"], dtype=np.float64)
        else:
            temperature = np.full(height.shape, np.nan, dtype=np.float64)
        if "temperature_count" in data.files:
            temperature_count = np.asarray(data["temperature_count"], dtype=np.int64)
        else:
            temperature_count = np.zeros(height.shape, dtype=np.int64)
        return DsdVerticalProfile(
            height_km=height,
            temperature_k=temperature,
            temperature_count=temperature_count,
            values={field: np.asarray(data[field], dtype=np.float64) for field in DSD_FIELDS},
            counts={
                field: np.asarray(data[f"count__{field}"], dtype=np.int64) for field in DSD_FIELDS
            },
            n_files=n_files,
            source=source,
        )


def load_ddh_qv_profile(
    agg_dir: Path,
    experiment: str,
    *,
    lead_text_value: str,
    block: str,
) -> DdhQvProfile:
    path = agg_dir / f"lead{lead_text_value}_VZ" / f"{experiment}_QV.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing DDH QV aggregate: {path}")

    key = f"block__{block}"
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            blocks = sorted(k[len("block__") :] for k in data.files if k.startswith("block__"))
            raise KeyError(f"{path} has no {key}. Available QV blocks: {blocks}")
        height = np.asarray(data["altitude_km"], dtype=np.float64)
        values = np.asarray(data[key], dtype=np.float64)
        n_days = int(data["days"].shape[0]) if "days" in data.files else 0

    order = np.argsort(height)
    return DdhQvProfile(
        height_km=height[order],
        values=values[order],
        n_days=n_days,
        source=str(path),
        block=block,
    )
