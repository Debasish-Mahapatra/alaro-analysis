import pytest

from alaro_analysis.common.parallel import execute_jobs


def divide(numerator, denominator):
    return numerator / denominator


@pytest.mark.parametrize("workers", [1, 2])
def test_serial_and_process_jobs_keep_failures_attached_to_their_keys(workers):
    results = {
        key: (value, error)
        for key, value, error in execute_jobs(
            divide,
            [("good", (6, 2)), ("bad", (1, 0))],
            workers=workers,
        )
    }
    assert results["good"] == (3.0, None)
    assert results["bad"][0] is None
    assert "division by zero" in results["bad"][1]
