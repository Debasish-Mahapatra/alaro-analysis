"""Radar wind profiler comparison: the observed statistics and the profiler-like model sample."""

import numpy as np

from alaro_analysis.common.constants import G
from alaro_analysis.data.convection_sections import convective_boxes
from alaro_analysis.data.rwp_model_sample import add_noise, noisy_fluxes, snapshot_sample
from alaro_analysis.data.rwp_profiler import (
    W_EDGES,
    block_means,
    exceedance,
    flux_weighted_updraft,
    gate_histogram,
    histogram_quantiles,
    window_profiles,
)


def test_averaging_windows_skip_samples_without_echo():
    w = np.array([[2.0], [np.nan], [4.0], [np.nan]])
    valid = np.isfinite(w)

    means, kept = block_means(w, valid, 2)

    # every pair holds one valid sample: kept at the 50 % cover, averaged over it alone
    assert kept.all()
    assert np.allclose(means[:, 0], [2.0, 4.0, 4.0])
    means, kept = block_means(w, valid, 3)
    assert kept[:, 0].tolist() == [True, False] and means[0, 0] == 3.0


def test_profiler_echo_threshold_drops_weak_samples_but_not_windows():
    n = 600  # one hour of 6 s profiles, one gate
    day = {"convective": np.ones((n, 1), bool), "w": np.full((n, 1), 2.0), "rho": np.ones((n, 1)),
           "dbz": np.where(np.arange(n)[:, None] < 300, 20.0, 2.0), "n_times": n}
    every = window_profiles(day, 1)
    echo = window_profiles(day, 1, echo_dbz=5.0)
    assert np.isclose(every["flux_up"][0, 0], 2.0) and np.isclose(echo["flux_up"][0, 0], 1.0)
    assert echo["n_conv"][0, 0] == 300 and echo["coverage"][0, 0] == 1.0
    assert echo["present"][0] and every["present"][0]


def test_histograms_put_outliers_in_the_end_bins():
    values = np.array([[-50.0, 0.05], [50.0, 0.05]])
    hist = gate_histogram(values, np.ones_like(values, dtype=bool))
    assert hist[0, 0] == 1 and hist[0, -1] == 1
    assert hist[1, 300] == 2  # 0.05 sits in the bin [0, 0.1)


def test_histogram_statistics():
    hist = np.zeros(W_EDGES.size - 1)
    hist[300:310] = 1.0  # uniform on [0, 1)
    assert np.isclose(histogram_quantiles(hist, [0.5])[0], 0.5)
    up, down = exceedance(hist[None], 0.5)
    assert np.isclose(up[0], 0.5) and down[0] == 0.0
    centres = 0.5 * (W_EDGES[:-1] + W_EDGES[1:])[300:310]
    assert np.isclose(flux_weighted_updraft(hist), (centres**2).sum() / centres.sum())


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    shape = (3, 10, 10)
    omega = rng.normal(0.0, 3.0, shape)
    sig_u = np.where(rng.random(shape) < 0.3, rng.uniform(0, 0.2, shape), 0.0)
    om_u_rel = -rng.uniform(0, 40.0, shape)
    sig_d = np.where(rng.random(shape) < 0.3, rng.uniform(0, 0.1, shape), 0.0)
    om_d = rng.uniform(0, 10.0, shape)
    rain = rng.uniform(0, 3.0, shape[1:])
    return omega, (sig_u, om_u_rel, sig_d, om_d), rain


def test_unmasked_variant_reproduces_the_box_mass_flux():
    omega, drafts, rain = _scene()
    mask = np.ones(omega.shape[1:], dtype=bool)
    fields = dict(zip(("UD_MESH_FRAC", "UD_OMEGA", "DD_MESH_FRAC", "DD_OMEGA"), drafts),
                  VITESSE_VERT=omega)
    boxes, n_used, _ = convective_boxes(fields, rain, mask)
    rho = np.full(omega.shape, 1.1)
    out = snapshot_sample(omega, rho, np.full(omega.shape, 30.0), rain,
                          np.zeros(omega.shape[1:]), mask, drafts)["rain1_all"]

    assert out["boxes"] == n_used
    assert np.allclose(out["up"], boxes["box_up"])
    assert np.allclose(out["down"], boxes["box_down"])
    assert np.allclose(out["up_area"], boxes["box_up_area"])
    # the histograms carry the same mass flux, to within the 0.1 m/s bins
    centres = 0.5 * (W_EDGES[:-1] + W_EDGES[1:])
    net = (out["hist_mass"] * centres).sum(axis=1)
    assert np.allclose(net, out["up"] - out["down"], atol=0.1 * 1.1 * out["hist_area"].sum(axis=1).max())


def test_levels_without_echo_drop_out():
    omega, drafts, rain = _scene(1)
    mask = np.ones(omega.shape[1:], dtype=bool)
    dbz = np.full(omega.shape, 30.0)
    dbz[0] = 0.0  # no echo at the top level
    out = snapshot_sample(omega, np.ones(omega.shape), dbz, rain, np.zeros(omega.shape[1:]), mask, drafts)
    assert out["rain1"]["up"][0] == 0.0 and out["rain1"]["hist_area"][0].sum() == 0.0
    assert np.isclose(out["rain1"]["up"][1], out["rain1_all"]["up"][1])


def test_noise_keeps_the_net_flux_and_adds_to_both_directions():
    hist = np.zeros((2, W_EDGES.size - 1))
    hist[0, 300] = 1.0  # w ~ 0.05 m/s
    hist[1, 320] = 0.5  # w ~ 2.05 m/s
    up0, down0, _, _ = noisy_fluxes(hist, hist, 0.0)
    up, down, area_up, area_down = noisy_fluxes(hist, hist, 1.5)
    assert np.allclose(up - down, up0 - down0)
    assert (up > up0).all() and (down > down0).all()
    assert np.isclose(up[0], 0.05 * 0.5133 + 1.5 * 0.3989, rtol=1e-2)
    assert np.allclose(area_up + area_down, hist.sum(axis=1))
    smoothed = add_noise(hist, 1.5)
    assert np.allclose(smoothed.sum(axis=1), hist.sum(axis=1))
    centres = 0.5 * (W_EDGES[:-1] + W_EDGES[1:])
    assert np.isclose((smoothed[0] * centres).sum(), 0.05, atol=1e-6)
    assert np.isclose(np.sqrt((smoothed[0] * (centres - 0.05) ** 2).sum()), 1.5, rtol=1e-3)
