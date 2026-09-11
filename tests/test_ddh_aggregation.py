import json

import numpy as np
import pytest

from alaro_analysis.ddh.aggregate_budgets import _aggregate_one, parse_day_var


def day(root, date, text):
    path = root / "lead0024_VZ" / "control" / date / "QV"
    path.mkdir(parents=True)
    (path / "done.ok").touch()
    (path / "QV.DHFDLABOF+0024.condcv.dta").write_text(text)
    return path


def test_ddh_preserves_per_block_counts_and_reports_incomplete_days(tmp_path):
    root = tmp_path / "source"
    day(root, "DDH20140101", "1 2\n2 4\n")
    day(root, "DDH20140102", "1 4\n2 nan\n")
    (root / "lead0024_VZ" / "control" / "DDH20140103").mkdir()
    output = tmp_path / "output"
    _aggregate_one(("control", "QV", "0024", "VZ", root, output, False))
    with np.load(output / "lead0024_VZ" / "control_QV.npz") as data:
        np.testing.assert_allclose(data["block__condcv"], [3, 4])
        np.testing.assert_array_equal(data["count__condcv"], [2, 1])
        assert len(json.loads(str(data["__metadata__"]))["issues"]) == 1


def test_malformed_completed_block_requires_explicit_partial_mode(tmp_path):
    path = day(tmp_path, "DDH20140101", "bad input\n")
    with pytest.raises(ValueError, match="Unreadable DDH block"):
        parse_day_var(path, "VZ")
    issues = []
    assert parse_day_var(path, "VZ", allow_partial=True, issues=issues) is None
    assert issues
