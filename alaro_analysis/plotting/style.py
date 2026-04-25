from __future__ import annotations


def resolve_workers(requested: int) -> int:
    # Cap at 32 (up from the previous 16).  32 is the default for
    # parallel jobs in this project (see MEMORY.md / feedback_32_workers.md).
    return max(1, min(int(requested), 32))
