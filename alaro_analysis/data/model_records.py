"""Model records. Shared implementation; command-line orchestration lives in workflows."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from alaro_analysis.common.observation_settings import MODEL_ROOT
from alaro_analysis.data.disdrometer_exports import datetime64_key

PF_DAY_RE = re.compile(r"^pf(\d{8})$")


PF_OUTPUT_RE = re.compile(r"^pfABOFABOF\+(\d{4})$")


def parse_pf_day(path: Path) -> datetime:
    match = PF_DAY_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse model day from {path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def discover_model_files(
    experiment: str,
    lead: str,
    model_root: Path = MODEL_ROOT,
    max_days: int | None = None,
) -> dict[np.datetime64, Path]:
    lead_token = f"{int(lead):04d}"
    base = model_root / experiment / "untar-output"
    day_dirs = sorted(d for d in base.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    files: dict[np.datetime64, Path] = {}
    for day_dir in day_dirs:
        path = day_dir / f"pfABOFABOF+{lead_token}"
        if not path.exists():
            continue
        init_time = parse_pf_day(day_dir)
        valid_time = init_time + timedelta(hours=int(lead_token))
        files[np.datetime64(valid_time, "s")] = path
    return files


def parse_lead_selection(value: str) -> tuple[int, ...] | None:
    """Return requested leads, or None for all available leads."""
    token = value.strip().lower()
    if token in {"all", "*"}:
        return None
    leads: list[int] = []
    for piece in token.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = (int(part) for part in piece.split("-", 1))
            leads.extend(range(start, end + 1))
        else:
            leads.append(int(piece))
    return tuple(sorted(set(leads)))


def lead_label(leads: tuple[int, ...] | None) -> str:
    if leads is None:
        return "all_leads"
    if len(leads) == 1:
        return f"lead{leads[0]:04d}"
    return f"leads{leads[0]:04d}-{leads[-1]:04d}"


def title_lead_text(leads: tuple[int, ...] | None) -> str:
    if leads is None:
        return "all forecast hours"
    if len(leads) == 1:
        return f"lead +{leads[0]:04d}"
    return f"leads +{leads[0]:04d}-+{leads[-1]:04d}"


def discover_model_records(
    experiment: str,
    leads: tuple[int, ...] | None,
    model_root: Path = MODEL_ROOT,
    max_days: int | None = None,
) -> dict[tuple[str, str, int], Path]:
    """Return model files keyed by (valid_time, init_time, lead_hour)."""
    base = model_root / experiment / "untar-output"
    day_dirs = sorted(d for d in base.iterdir() if d.is_dir() and PF_DAY_RE.match(d.name))
    if max_days is not None:
        day_dirs = day_dirs[:max_days]

    lead_set = set(leads) if leads is not None else None
    files: dict[tuple[str, str, int], Path] = {}
    for day_dir in day_dirs:
        init_time = np.datetime64(parse_pf_day(day_dir), "s")
        for path in sorted(day_dir.iterdir()):
            match = PF_OUTPUT_RE.match(path.name)
            if not match:
                continue
            lead = int(match.group(1))
            if lead_set is not None and lead not in lead_set:
                continue
            valid_time = init_time + np.timedelta64(lead, "h")
            key = (datetime64_key(valid_time), datetime64_key(init_time), lead)
            files[key] = path
    return files
