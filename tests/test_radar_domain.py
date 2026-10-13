"""4 km radar domain built from the radar's own ~1 km coverage."""

import numpy as np

from alaro_analysis.data.radar_domain import coverage_on_model_grid


def test_coverage_is_the_share_of_each_cells_radar_pixels_inside_the_footprint():
    # Model cells 0.04 deg wide; radar pixels 0.01 deg, so 16 pixels per cell.
    model_lon, model_lat = np.meshgrid(np.arange(3) * 0.04, np.arange(2) * 0.04)
    lon = np.arange(-0.015, 0.1, 0.01)[:12]
    lat = np.arange(-0.015, 0.06, 0.01)[:8]
    footprint = np.zeros((lat.size, lon.size), bool)
    footprint[:, :4] = True   # the first column of model cells is fully covered
    footprint[:2, 4:8] = True  # the middle column only in its lower half

    share, total = coverage_on_model_grid(footprint, lat, lon, model_lon, model_lat)

    assert (total == 16).all()
    np.testing.assert_allclose(share[:, 0], 1.0)
    np.testing.assert_allclose(share[:, 2], 0.0)
    np.testing.assert_allclose(share[0, 1], 0.5)
    np.testing.assert_allclose(share[1, 1], 0.0)
