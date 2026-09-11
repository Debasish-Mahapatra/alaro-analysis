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
    if nworkers == 1:
        for i, task in enumerate(tasks, 1):
            yield worker(task)
            if i % every == 0 or i == n:
                print(f"  {i}/{n} {desc}", flush=True)
        return
    with get_context(context).Pool(nworkers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, tasks), 1):
            yield result
            if i % every == 0 or i == n:
                print(f"  {i}/{n} {desc}", flush=True)


def execute_jobs(worker: Callable, jobs: Iterable, *, workers: int = 32) -> Iterator:
    """Yield (key, value, error) for keyed positional-argument jobs.

    Serial and process execution have the same result/error contract. A failed
    job does not hide the failures of other jobs; orchestration decides whether
    partial products are useful and must report errors to its caller.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = list(jobs)
    count = len(jobs)
    if count == 0:
        return
    nworkers = min(resolve_workers(workers), count)
    if nworkers == 1:
        for key, arguments in jobs:
            try:
                value = worker(*arguments)
            except Exception as exc:
                yield key, None, str(exc)
            else:
                yield key, value, None
        return
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        futures = {pool.submit(worker, *arguments): key for key, arguments in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                yield key, None, str(exc)
            else:
                yield key, value, None
