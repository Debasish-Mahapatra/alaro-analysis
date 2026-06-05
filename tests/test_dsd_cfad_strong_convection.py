from __future__ import annotations

import numpy as np

from alaro_analysis.workflows import dsd_cfad_strong_convection as strong


def test_strong_convection_mask_uses_upward_pressure_velocity():
    settings = strong.StrongConvectionSettings(
        metric="omega",
        min_updraft_pa_s=10.0,
        min_updraft_flux=0.01,
        min_updraft_mesh_frac=0.0,
    )
    omega = np.array([[[-12.0, -8.0], [5.0, -20.0]]])
    mesh = np.array([[[0.2, 0.3], [0.4, 0.0]]])

    mask = strong.strong_convection_mask(omega, mesh, settings)

    np.testing.assert_array_equal(mask, [[[True, False], [False, False]]])


def test_strong_convection_mask_can_use_flux_threshold():
    settings = strong.StrongConvectionSettings(
        metric="flux",
        min_updraft_pa_s=10.0,
        min_updraft_flux=0.2,
        min_updraft_mesh_frac=0.0,
    )
    omega = np.array([[[-20.0, -5.0], [-20.0, -20.0]]])
    mesh = np.array([[[0.2, 0.5], [0.05, 0.0]]])

    mask = strong.strong_convection_mask(omega, mesh, settings)

    np.testing.assert_array_equal(mask, [[[True, True], [False, False]]])


def test_level_mean_in_strong_convection_respects_domain_rain_and_updraft_masks():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    domain = np.array([[True, True], [False, True]])
    convection = np.array(
        [
            [[True, False], [True, True]],
            [[False, True], [True, True]],
        ]
    )
    rainy = np.array(
        [
            [[True, True], [True, False]],
            [[True, True], [True, True]],
        ]
    )

    means, counts = strong.level_mean_in_strong_convection(
        values,
        domain,
        convection,
        rainy,
    )

    np.testing.assert_allclose(means, [1.0, 7.0])
    np.testing.assert_array_equal(counts, [1, 2])
