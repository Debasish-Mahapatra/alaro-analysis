from __future__ import annotations

from alaro_analysis.common.parallel import imap_unordered_progress, resolve_workers


def _square(x):
    return x * x


def test_resolve_workers_caps_at_32():
    assert resolve_workers(100) == 32
    assert resolve_workers(8) == 8
    assert resolve_workers(0) == 1
    assert resolve_workers(-5) == 1


def test_imap_unordered_progress_runs_all_tasks():
    results = list(
        imap_unordered_progress(_square, [1, 2, 3, 4, 5], workers=2, desc="n", every=2)
    )
    assert sorted(results) == [1, 4, 9, 16, 25]


def test_imap_unordered_progress_empty():
    assert list(imap_unordered_progress(_square, [], workers=4)) == []
