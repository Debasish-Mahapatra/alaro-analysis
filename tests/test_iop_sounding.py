"""IOP sounding profiles: the sonde time mask applies to every dataset."""

import numpy as np

from alaro_analysis.recipes.iop.make_iop_sounding_profiles import masked_means


def test_masked_means_use_the_same_launches_for_sonde_and_every_model():
    # Height 0: launch 2 lacks model a. Height 1: launch 1 lacks the sonde.
    obs = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    models = {"a": np.array([[2.0, 4.0], [4.0, 8.0], [np.nan, 10.0]]), "b": np.full((3, 2), 7.0)}

    (obs_mean, counts), model_means = masked_means(obs, models, min_samples=1)

    assert counts.tolist() == [2, 2]
    assert obs_mean.tolist() == [2.0, 4.0]
    assert model_means["a"][0].tolist() == [3.0, 7.0]
    assert model_means["b"][0].tolist() == [7.0, 7.0]
    assert all(np.array_equal(n, counts) for _, n in model_means.values())


def test_masked_means_blank_heights_with_too_few_launches():
    obs = np.ones((3, 1))
    models = {"a": np.array([[1.0], [np.nan], [np.nan]])}

    (obs_mean, counts), model_means = masked_means(obs, models, min_samples=2)

    assert counts.tolist() == [1]
    assert np.isnan(obs_mean[0]) and np.isnan(model_means["a"][0][0])


def test_launches_split_into_day_and_night_on_local_time():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from alaro_analysis.recipes.iop.make_iop_sounding_profiles import launch_times_lt, time_of_day

    def at(hour, minute):
        return SimpleNamespace(launch_time_utc=datetime(2014, 2, 3, hour, minute, tzinfo=timezone.utc))

    # ARM MAO schedule in UTC: 05:30, 11:30, 14:30, 17:15, 23:30 -> 01:30, 07:30, 10:30, 13:15, 19:30 LT
    day = [at(11, 30), at(14, 30), at(17, 15)]
    night = [at(23, 30), at(5, 30)]
    assert [time_of_day(x) for x in day] == ["day"] * 3
    assert [time_of_day(x) for x in night] == ["night"] * 2
    assert time_of_day(at(10, 0)) == "day" and time_of_day(at(9, 59)) == "night"  # 06:00 LT edge
    assert launch_times_lt(day * 5) == "07, 10, 13"
    assert launch_times_lt(night * 5) == "19, 01"


def test_day_minus_night_is_the_change_of_each_models_bias():
    from datetime import date, timedelta

    from alaro_analysis.recipes.iop.make_iop_sounding_profiles import (
        MODEL_RUNS,
        PANELS,
        day_minus_night,
    )

    rng = np.random.default_rng(1)
    n, nz = 40, 5
    days = [date(2014, 2, 1) + timedelta(days=i) for i in range(n)]

    def part(offset):
        obs = rng.normal(20.0, 1.0, (n, nz))
        noise = rng.normal(0.0, 0.1, (n, nz))
        return {
            "dates": days,
            "obs": {p: obs for p in PANELS},
            "models": {k: {p: obs + offset + noise for p in PANELS} for k in MODEL_RUNS},
        }

    day, night = part(1.0), part(0.0)
    night["obs"]["tdry"] = night["obs"]["tdry"].copy()
    night["obs"]["tdry"][:, -1] = np.nan  # a height no night launch reached

    out = day_minus_night([day], [night], np.arange(nz) * 250.0)

    change = out["models"]["graupel"]["tdry"]
    np.testing.assert_allclose(change[:-1], 1.0, atol=0.1)
    assert np.isnan(change[-1])
    low, high = out["band"]["tdry"]
    assert np.all(low[:-1] < 1.0) and np.all(high[:-1] > 1.0) and np.all(low[:-1] > 0.8)
