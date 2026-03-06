from __future__ import annotations

from pathlib import Path

from .constants import DAY_RE, FILE_HOUR_RE


def parse_utc_hour_from_name(file_name: str) -> int | None:
    match = FILE_HOUR_RE.search(file_name)
    if match is None:
        return None
    utc_hour = int(match.group(1))
    if utc_hour == 24:
        return None
    if utc_hour < 0 or utc_hour > 23:
        return None
    return utc_hour


def parse_month_from_day_name(day_name: str) -> int | None:
    match = DAY_RE.match(day_name)
    if match is None:
        return None
    yyyymmdd = match.group(1)
    return int(yyyymmdd[4:6])


def has_pf_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("pf"):
            return True
    return False
