import numpy as np

from alaro_analysis.common.vertical import centers_to_edges


def test_centers_to_edges_single_level():
    edges = centers_to_edges(np.array([10.0]))
    np.testing.assert_allclose(edges, np.array([9.5, 10.5]))
