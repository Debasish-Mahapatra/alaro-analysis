"""IOP convection diurnal sections: the per-hour domain-mean reduction."""

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.data.convection_sections import profiles_from_fields


def _fields(**overrides):
    # Two levels, four cells; the last cell is outside the radar domain.
    base = {name: np.zeros((2, 4)) for name in
            ("UD_MESH_FRAC", "UD_OMEGA", "DD_MESH_FRAC", "DD_OMEGA", "VERT.VELOCIT",
             "CLOUD_FRACTI", "TEMPERATURE")}
    for values in base.values():
        values[:, 3] = np.nan
    base.update(overrides)
    return base


def test_masked_cells_never_enter_the_domain_mean():
    mesh = np.array([[0.2, 0.0, 0.4, np.nan], [0.0, 0.0, 0.0, np.nan]])
    omega = np.array([[-10.0, -5.0, -20.0, np.nan], [0.0, 0.0, 0.0, np.nan]])

    out = profiles_from_fields(_fields(UD_MESH_FRAC=mesh, UD_OMEGA=omega))

    # Means divide by the three valid cells, not by four.
    assert out["ud_extent"][0] == (0.2 + 0.0 + 0.4) / 3
    assert out["ud_flux"][0] == ((10.0 * 0.2 + 20.0 * 0.4) / G) / 3
    assert out["ud_flux"][1] == 0.0


def test_updraft_and_downdraft_fluxes_are_both_positive_magnitudes():
    # ALARO sign convention: UD_OMEGA < 0 rising, DD_OMEGA > 0 sinking.
    up = profiles_from_fields(
        _fields(
            UD_MESH_FRAC=np.full((2, 4), 0.5),
            UD_OMEGA=np.full((2, 4), -4.0),
            DD_MESH_FRAC=np.full((2, 4), 0.5),
            DD_OMEGA=np.full((2, 4), 4.0),
        )
    )
    assert up["ud_flux"][0] == 0.5 * 4.0 / G
    assert up["dd_flux"][0] == 0.5 * 4.0 / G


def test_merging_iops_weights_each_by_its_sampled_hours():
    from alaro_analysis.recipes.iop.make_iop_convection_figures import merge_sections

    iop1 = (np.array([[2.0, np.nan]]), np.array([[10, 0]]))
    iop2 = (np.array([[4.0, 5.0]]), np.array([[30, 20]]))

    merged = merge_sections([iop1, iop2])

    # 10 hours of 2 and 30 of 4 give 3.5, not the unweighted 3.
    assert merged[0, 0] == 3.5
    # A cell only one IOP sampled keeps that IOP's value.
    assert merged[0, 1] == 5.0


def test_relative_bias_masks_cells_where_the_reference_is_negligible():
    from alaro_analysis.recipes.iop.make_iop_convection_figures import relative_bias

    reference = np.array([[1.0, 100.0, 0.0]])
    model = np.array([[2.0, 150.0, 7.0]])

    rel = relative_bias(model - reference, reference)

    assert np.isnan(rel[0, 0])  # 1.0 is below 5 % of the p98 of the reference
    assert rel[0, 1] == 50.0
    assert np.isnan(rel[0, 2])


def test_resolved_velocity_splits_into_up_and_down_magnitudes():
    w = np.array([[2.0, -3.0, 0.0, np.nan], [0.0, 0.0, 0.0, np.nan]])

    out = profiles_from_fields(_fields(**{"VERT.VELOCIT": w}))

    assert out["w_up"][0] == 2.0 / 3
    assert out["w_dn"][0] == 3.0 / 3


def test_negative_mesh_fractions_count_as_zero():
    # Advection undershoots: the scheme itself never produces a fraction below 0.
    ud = np.array([[0.3, -0.05, -1e-5, np.nan], [0.0, 0.0, 0.0, np.nan]])
    dd = np.array([[-0.01, 0.02, 0.0, np.nan], [0.0, 0.0, 0.0, np.nan]])
    omega = np.array([[-6.0, -6.0, -6.0, np.nan], [0.0, 0.0, 0.0, np.nan]])

    out = profiles_from_fields(
        _fields(UD_MESH_FRAC=ud, UD_OMEGA=omega, DD_MESH_FRAC=dd, DD_OMEGA=-omega)
    )

    assert out["ud_extent"][0] == 0.3 / 3
    assert out["dd_extent"][0] == 0.02 / 3
    assert out["ud_flux"][0] == (6.0 * 0.3 / G) / 3


def test_draft_velocities_pointing_the_wrong_way_count_as_zero():
    # The physics resets a sinking updraft omega and a rising downdraft omega to 0.
    mesh = np.array([[0.2, 0.2, 0.2, np.nan], [0.0, 0.0, 0.0, np.nan]])
    ud_omega = np.array([[-5.0, 3.0, -5.0, np.nan], [0.0, 0.0, 0.0, np.nan]])
    dd_omega = np.array([[2.0, -1.0, 2.0, np.nan], [0.0, 0.0, 0.0, np.nan]])

    out = profiles_from_fields(
        _fields(UD_MESH_FRAC=mesh, UD_OMEGA=ud_omega, DD_MESH_FRAC=mesh, DD_OMEGA=dd_omega)
    )

    assert out["ud_flux"][0] == (2 * 5.0 * 0.2 / G) / 3
    assert out["dd_flux"][0] == (2 * 2.0 * 0.2 / G) / 3


def test_column_fluxes_reproduce_the_worked_example_and_never_double_count():
    from alaro_analysis.common.constants import G
    from alaro_analysis.data.convection_sections import column_fluxes

    # updraft 5 %, relative updraft omega -40 Pa/s, grid mean -5 Pa/s, no downdraft
    f = column_fluxes(np.array(-5.0), np.array(0.05), np.array(-40.0), np.array(0.0), np.array(0.0))
    # environment rises (-3 Pa/s), so the upward flux is the net ascent: 5 / g = 0.51
    assert np.isclose(f["box_up"], 5.0 / G) and np.isclose(round(float(f["box_up"]), 2), 0.51)
    assert np.isclose(f["box_up"], f["box_up_nodd"])
    assert np.isclose(f["box_par_up"], 0.05 * 43.0 / G)
    assert f["box_up"] < f["box_par_up"] + f["box_res_up"]

    # environment sinks: the upward flux is the updraft alone, the sinking is downward flux
    f = column_fluxes(np.array(-1.0), np.array(0.05), np.array(-40.0), np.array(0.0), np.array(0.0))
    assert np.isclose(f["box_up"], f["box_par_up"])
    assert np.isclose(f["box_up"] - f["box_down"], 1.0 / G)


def test_column_fluxes_with_a_downdraft_keep_the_net_flux():
    from alaro_analysis.common.constants import G
    from alaro_analysis.data.convection_sections import column_fluxes

    omega = np.array([-5.0, 2.0, 0.5])
    f = column_fluxes(omega, np.array([0.05, 0.02, 0.0]), np.array([-40.0, -10.0, 0.0]),
                      np.array([0.03, 0.04, 0.0]), np.array([8.0, 6.0, 0.0]))
    assert np.allclose(f["box_up"] - f["box_down"], -omega / G)
    assert (f["box_up"] >= 0).all() and (f["box_down"] >= 0).all()
    # without drafts the column is one piece
    assert np.isclose(f["box_up"][2], 0.0) and np.isclose(f["box_down"][2], 0.5 / G)


def test_convective_boxes_divide_by_the_full_box_area():
    from alaro_analysis.common.constants import G
    from alaro_analysis.data.convection_sections import convective_boxes

    ny = nx = 10  # four 5 x 5 boxes
    mask = np.ones((ny, nx), dtype=bool)
    mask[5:, 5:] = False  # this box is outside the domain
    omega = np.zeros((1, ny, nx))
    omega[0, 1, 1] = -9.0  # one convective column in box (0, 0)
    omega[0, 2, 7] = -9.0  # ascent without rain in box (0, 1): not convective
    rain = np.zeros((ny, nx))
    rain[1, 1] = 5.0
    fields = {"VITESSE_VERT": omega}
    sums, n_conv, n_boxes = convective_boxes(fields, rain, mask, has_scheme=False)
    assert (n_conv, n_boxes) == (1, 3)
    assert np.isclose(sums["box_up"][0], 9.0 / G / 25)
