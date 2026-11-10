"""Diurnal-cycle error diagnostics."""

import numpy as np

from alaro_analysis.analysis.diurnal_errors import (
    CENTRES,
    first_harmonic,
    mean_cycle,
    mse_decomposition,
    wrap_hours,
)


def test_first_harmonic_recovers_amplitude_and_clock_time_of_the_peak():
    cycle = 1.0 + 0.4 * np.cos(2 * np.pi * (CENTRES - 15.0) / 24)
    amplitude, peak = first_harmonic(cycle)
    assert np.isclose(amplitude, 0.4) and np.isclose(peak, 15.0)
    amplitude, peak = first_harmonic(np.stack([cycle, np.roll(cycle, 3)], axis=1))
    assert np.allclose(peak, [15.0, 18.0])


def test_mse_decomposition_is_exact():
    rng = np.random.default_rng(0)
    obs, model = rng.random(24), rng.random(24) * 2 + 0.3
    parts = mse_decomposition(model, obs)
    assert np.isclose(parts["offset"] + parts["amplitude"] + parts["shape"], parts["mse"])
    shifted = mse_decomposition(obs + 0.5, obs)
    assert np.isclose(shifted["offset"], 0.25) and np.isclose(shifted["shape"], 0.0)


def test_mean_cycle_and_wrapping():
    hours = np.tile(np.arange(24), 3)
    values = np.repeat(np.arange(3.0), 24)
    assert np.allclose(mean_cycle(values, hours), 1.0)
    assert np.allclose(wrap_hours([13.0, -13.0, 1.5]), [-11.0, 11.0, 1.5])
