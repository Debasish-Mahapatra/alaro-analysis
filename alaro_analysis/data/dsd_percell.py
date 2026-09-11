"""Bottom-level grid-cell discovery and worker-local mask IO, shared by DSD methods."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PF_DAY_RE = re.compile(r"^pf(\d{8})$")


PF_FILE_RE = re.compile(r"^pfABOFABOF\+(\d{4})\.nc$")


def discover_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    netcdf_root: Path,
    max_days: int | None,
) -> list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]]:
    base = netcdf_root / experiment / "masked-netcdf"
    needed = ["RAIN", "TEMPERATURE", "PRESSURE"]
    if experiment == "2mom":
        needed.append("PNR")
    ref_dir = base / "RAIN"
    day_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]
    lead_set = set(leads) if leads is not None else None
    out: list[tuple[np.datetime64, np.datetime64, int, dict[str, Path]]] = []
    for day_dir in day_dirs:
        m = PF_DAY_RE.match(day_dir.name)
        if not m:
            continue
        init_dt = datetime.strptime(m.group(1), "%Y%m%d")
        for path in sorted(day_dir.iterdir()):
            fm = PF_FILE_RE.match(path.name)
            if not fm:
                continue
            lead = int(fm.group(1))
            if lead_set is not None and lead not in lead_set:
                continue
            paths: dict[str, Path] = {}
            ok = True
            for var in needed:
                cand = base / var / day_dir.name / path.name
                if not cand.exists():
                    ok = False
                    break
                paths[var] = cand
            if ok:
                valid_dt = init_dt + timedelta(hours=lead)
                out.append((np.datetime64(valid_dt, "s"), np.datetime64(init_dt, "s"), lead, paths))
    out.sort(key=lambda x: x[0])
    return out


_WORKER_MASK: np.ndarray | None = None


def _init_worker(mask: np.ndarray) -> None:
    global _WORKER_MASK
    _WORKER_MASK = np.asarray(mask, dtype=bool)


def _read_masked_field(path: Path, var: str) -> np.ndarray:
    if _WORKER_MASK is None:
        raise RuntimeError("worker mask not initialised")
    with Dataset(path) as ds:
        data = ds.variables[var]
        if data.ndim == 4:
            field = np.asarray(data[0, 0], dtype=np.float64)
        elif data.ndim == 3:
            field = np.asarray(data[0], dtype=np.float64)
        elif data.ndim == 2:
            field = np.asarray(data[:], dtype=np.float64)
        else:
            raise ValueError(f"Unexpected ndim for {var} in {path}: {data.ndim}")
    return field[_WORKER_MASK]


def _parse_leads(text: str) -> tuple[int, ...] | None:
    if text.lower() in {"all", "*"}:
        return None
    leads: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            leads.extend(range(int(a), int(b) + 1))
        else:
            leads.append(int(piece))
    return tuple(sorted(set(leads)))
