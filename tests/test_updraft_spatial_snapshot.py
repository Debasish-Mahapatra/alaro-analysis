from __future__ import annotations

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.workflows.updraft_spatial_snapshot import (
    activity_crop,
    cloud_condensate_transport,
    signed_downdraft_flux,
    signed_updraft_flux,
)


def test_signed_updraft_flux_keeps_reference_negative_upward_convention():
    omega = np.array([[-9.81, 2.0], [-4.905, -3.0]])
    mesh = np.array([[0.5, 0.5], [0.0, 0.25]])

    flux = signed_updraft_flux(omega, mesh)

    np.testing.assert_allclose(
        flux,
        np.array([[-9.81 * 0.5 / G, 2.0 * 0.5 / G], [0.0, -3.0 * 0.25 / G]]),
    )


def test_signed_downdraft_flux_keeps_positive_downward_convention():
    omega = np.array([[9.81, -2.0], [4.905, 3.0]])
    mesh = np.array([[0.5, 0.5], [0.0, 0.25]])

    flux = signed_downdraft_flux(omega, mesh)

    np.testing.assert_allclose(
        flux,
        np.array([[9.81 * 0.5 / G, -2.0 * 0.5 / G], [0.0, 3.0 * 0.25 / G]]),
    )


def test_cloud_condensate_transport_uses_total_condensate_and_mg_conversion():
    liquid = np.array([[1.0e-4, -1.0e-4]])
    solid = np.array([[2.0e-4, 3.0e-4]])
    updraft = np.array([[-0.5, -0.25]])
    downdraft = np.array([[0.1, 0.05]])

    transport = cloud_condensate_transport(liquid, solid, updraft, downdraft)

    expected = np.array([[(3.0e-4) * (-0.4) * 1.0e6, (3.0e-4) * (-0.2) * 1.0e6]])
    np.testing.assert_allclose(transport, expected)


def test_activity_crop_pads_around_any_active_panel():
    transport = np.zeros((8, 9))
    updraft = np.zeros((8, 9))
    downdraft = np.zeros((8, 9))
    transport[2, 3] = 2.0
    downdraft[5, 6] = 0.02

    y_slice, x_slice = activity_crop(
        (transport, updraft, downdraft),
        pad=1,
        transport_threshold=1.0,
        updraft_threshold=0.05,
        downdraft_threshold=0.01,
    )

    assert y_slice == slice(1, 7)
    assert x_slice == slice(2, 8)
