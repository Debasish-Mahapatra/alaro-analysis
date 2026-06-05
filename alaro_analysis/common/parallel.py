"""Shared process-pool helper for the day/file parallel loops.

Most heavy workflows build a list of ``(experiment, day)`` tasks and reduce
per-task results. They all repeated the same
``get_context("fork").Pool(...).imap_unordered`` + progress-print boilerplate.
This wraps it once and routes the worker count through ``resolve_workers``
(the project-standard 32 cap).

The worker passed in must be a module-level (picklable) function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from multiprocessing import get_context


def resolve_workers(requested: int) -> int:
    """Clamp a requested worker count to ``[1, 32]`` (the project default cap)."""
    return max(1, min(int(requested), 32))


def imap_unordered_progress(
    worker: Callable,
    tasks: Iterable,
    *,
    workers: int = 32,
    desc: str = "tasks",
    every: int = 50,
    context: str = "fork",
) -> Iterator:
    """Run ``worker(task)`` over ``tasks`` in a process pool.

    Yields each result as it completes (unordered) and prints ``i/n`` progress
    every ``every`` items. Worker count is ``min(resolve_workers(workers), n)``.
    """
    tasks = list(tasks)
    n = len(tasks)
    if n == 0:
        return
    nworkers = max(1, min(resolve_workers(workers), n))
    with get_context(context).Pool(nworkers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, tasks), 1):
            yield result
            if i % every == 0 or i == n:
                print(f"  {i}/{n} {desc}", flush=True)
