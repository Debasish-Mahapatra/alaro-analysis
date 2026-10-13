"""Microphysics story: radar-echo statistics, SIPAM coarse-graining and column diagnostics."""

import numpy as np


def test_interpolate_to_heights_works_in_linear_z_and_handles_top_first_levels():
    from alaro_analysis.analysis.radar_echo import interpolate_to_heights

    height = np.array([3000.0, 2000.0, 1000.0])[:, None]  # top first, like FA
    dbz = np.array([10.0, 30.0, 20.0])[:, None]
    out = interpolate_to_heights(dbz, height, target_km=[1.5, 2.0, 2.5, 4.0])
    zlin = 0.5 * (10 ** 3.0 + 10 ** 2.0)
    assert np.isclose(out[0, 0], 10 * np.log10(zlin))  # halfway 1-2 km, in linear Z
    assert np.isclose(out[1, 0], 30.0)
    assert np.isnan(out[3, 0])  # above the column


def test_linear_to_heights_interpolates_in_value():
    from alaro_analysis.analysis.radar_echo import linear_to_heights

    height = np.array([2000.0, 1000.0])[:, None]
    out = linear_to_heights(np.array([4.0, 2.0])[:, None], height, target_km=[1.5])
    assert np.isclose(out[0, 0], 3.0)


def test_echo_tops_pick_the_highest_level_above_threshold():
    from alaro_analysis.analysis.radar_echo import echo_tops

    heights = np.array([2.0, 3.0, 4.0, 5.0])
    dbz = np.array([[45.0, 25.0], [42.0, 15.0], [30.0, np.nan], [41.0, 5.0]])
    tops = echo_tops(dbz, thresholds=(20.0, 40.0), heights_km=heights)
    assert tops[0, 0] == 5.0 and tops[1, 0] == 5.0
    assert tops[0, 1] == 2.0 and np.isnan(tops[1, 1])


def test_steiner_marks_a_peak_convective_and_the_flat_shield_stratiform():
    from alaro_analysis.analysis.radar_echo import steiner_classes

    dbz = np.full((15, 15), 25.0)
    dbz[7, 7] = 45.0  # >= 40 dBZ: convective centre
    dbz[2, 2] = 36.0  # peak 11 dB above a 25 dBZ background: convective (needs > 6.5 dB)
    dbz[12, 12] = np.nan  # no echo
    classes = steiner_classes(dbz, np.ones_like(dbz, bool))
    assert classes[7, 7] == 2 and classes[2, 2] == 2
    assert classes[0, 14] == 1 and classes[12, 12] == 0
    assert (classes == 2).sum() == 2  # background < 35 dBZ: no convective radius on a 4 km grid


def test_steiner_spreads_a_core_to_its_neighbours_in_a_strong_background():
    from alaro_analysis.analysis.radar_echo import steiner_classes

    dbz = np.full((15, 15), 38.0)
    dbz[7, 7] = 50.0
    classes = steiner_classes(dbz, np.ones_like(dbz, bool))
    assert classes[7, 7] == 2 and classes[6, 7] == 2 and classes[7, 8] == 2 and classes[6, 6] == 1


def test_precipitating_column_filter_drops_weak_clear_air_echo():
    from alaro_analysis.analysis.radar_echo import accumulate, empty_stats

    ny = nx = 9
    dbz = np.full((15, ny, nx), np.nan)
    dbz[0] = 15.0  # 15 dBZ at 2 km everywhere, never 20 dBZ: clear air
    dbz[:5, 4, 4] = 30.0  # one precipitating column
    stats = empty_stats()
    dom = np.ones((ny, nx), bool)
    accumulate(stats, dbz, dom, dom, local_hour=12)
    assert stats["occurrence"][0, 0] == 1  # only the precipitating column counts at 2 km
    assert stats["n_cells"][0] == ny * nx


def test_sipam_coarse_graining_averages_linear_z_with_no_echo_as_zero():
    from alaro_analysis.data import sipam_reflectivity as S

    S.init_worker({"idx": np.array([0, 0, 1, -1]), "counts": np.array([2, 1]), "shape": (1, 2)})
    zlin = np.array([[1000.0, 0.0, 10.0, 99.0]])  # 30 dBZ + no echo -> 27 dBZ; 10 dBZ alone
    out = S.coarse_scan(zlin)
    assert np.isclose(out[0, 0, 0], 10 * np.log10(500.0))
    assert np.isclose(out[0, 0, 1], 10.0)


def test_calibration_offsets_follow_the_original_periods():
    from datetime import datetime

    from alaro_analysis.data.sipam_reflectivity import calibration_offset

    assert calibration_offset(datetime(2014, 2, 15, 12)) == 3.0
    assert calibration_offset(datetime(2014, 9, 1)) == -2.5


def test_rain_rate_uses_the_atlas_fall_speed():
    from alaro_analysis.data.microphysics_columns import _rain_rate

    r = _rain_rate(np.array([1.0]), np.array([1.5]))
    assert np.isclose(r[0], 3.6 * (9.65 - 10.3 * np.exp(-0.9)))


def test_wacr_gates_fall_in_the_quarter_km_profile_bins():
    from alaro_analysis.data.wacr_occurrence import _gate_bins

    bins = _gate_bins(np.array([100.0, 240.0, 380.0, 3990.0, 17000.0]), antenna_m=4.0)
    # 0.104 km is below the first bin (0.125-0.375 km), 0.244 km in bin 0 (0.25 km), 0.384 km in bin 1
    # (0.5 km), 3.994 km in bin 15 (4 km), 17 km above the top (16 km)
    assert bins.tolist() == [-1, 0, 1, 15, -1]


def test_t3_cell_is_the_nearest_model_cell():
    from alaro_analysis.data.wacr_occurrence import T3_LAT, T3_LON, t3_cell

    lon, lat = np.meshgrid(np.arange(-61.0, -60.0, 0.04), np.arange(-3.5, -3.0, 0.04))
    j, i = t3_cell(lon, lat)
    assert abs(lat[j, i] - T3_LAT) <= 0.02 and abs(lon[j, i] - T3_LON) <= 0.02


def test_ice_water_of_non_precipitating_columns_counts_only_in_the_all_cells_occurrence():
    from alaro_analysis.analysis.radar_echo import IWC_THRESHOLDS, accumulate, empty_stats

    ny = nx = 5
    dbz = np.full((15, ny, nx), np.nan)
    dbz[6] = 12.0  # weak ice echo at 8 km, no column reaches 20 dBZ: an anvil
    iwc = np.full((15, ny, nx), np.nan)
    iwc[6] = 0.2
    ice = np.zeros(15, bool)
    ice[6] = True
    stats = empty_stats()
    dom = np.ones((ny, nx), bool)
    accumulate(stats, dbz, dom, dom, local_hour=3, iwc_h=iwc, ice_levels=ice)
    j = IWC_THRESHOLDS.index(0.1)
    assert stats["iwc_occurrence_all_cells"][j, 6] == ny * nx
    assert stats["iwc_occurrence"][0, j, 6] == 0


def test_ddh_heights_follow_the_hypsometric_equation():
    from alaro_analysis.data.ddh_water_budget import CP, G, RD, full_level_heights

    p_half = np.array([0.0, 50000.0, 100000.0])  # two layers of 500 hPa, top first
    vpp = np.diff(p_half) / G
    t = np.array([250.0, 290.0])
    z = full_level_heights(vpp, CP * t * vpp, np.zeros(2))
    lower = RD * 290.0 / G * 50000.0 / 75000.0  # thickness of the lower layer
    assert np.isclose(z[1], 0.5 * lower)
    assert z[0] > lower


def test_ddh_column_totals_and_levels_come_from_the_pseudo_fluxes():
    from alaro_analysis.data.ddh_water_budget import column_total, per_level

    data = {"day_A": np.array([[0.0, 1.0, 3.0]]), "day_B": np.array([[0.0, 0.5, 0.5]])}
    assert column_total(data, ("A", "B"))[0] == 3.5
    assert per_level(data, ("A", "B"))[0].tolist() == [1.5, 2.0]
