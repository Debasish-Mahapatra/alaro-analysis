from __future__ import annotations

import numpy as np

from alaro_analysis.workflows.precip_distribution import (
    compute_ccdf,
    compute_unconditional_pdf,
)


def test_ccdf_uses_all_samples_in_denominator():
    values = np.array([0.0, 0.0, 1.0, 2.0])
    thresholds = np.array([0.0, 0.5, 1.0, 2.0, 3.0])

    ccdf = compute_ccdf(values, thresholds)

    np.testing.assert_allclose(ccdf, [1.0, 0.5, 0.5, 0.25, 0.0])


def test_unconditional_pdf_is_not_renormalized_to_wet_samples():
    values = np.array([0.0, 0.5, 1.5])
    edges = np.array([0.1, 1.1, 2.1])

    counts, density = compute_unconditional_pdf(values, edges)

    np.testing.assert_array_equal(counts, [1, 1])
    np.testing.assert_allclose(density, [1.0 / 3.0, 1.0 / 3.0])
    assert np.isclose(np.sum(density * np.diff(edges)), 2.0 / 3.0)
