from __future__ import annotations


def resolve_workers(requested: int) -> int:
    return max(1, min(int(requested), 16))
