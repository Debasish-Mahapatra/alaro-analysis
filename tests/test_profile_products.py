import numpy as np
import xarray as xr

from alaro_analysis.common.constants import G
from alaro_analysis.data.profile_products import load_profile


def write_field(root, variable, values, hour=0):
    path = root / variable / "pf20140101" / f"pfABOFABOF+{hour:04d}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {variable: (("time", "level", "y", "x"), np.asarray(values).reshape(1, 1, 1, -1))}
    ).to_netcdf(path)
    return path


def test_raw_profile_cache_matches_direct_reduction_and_detects_new_inputs(tmp_path):
    root = tmp_path / "inputs"
    cache = tmp_path / "temperature.npz"
    write_field(root, "TEMPERATURE", [280.0, 284.0, np.nan])
    first = load_profile(cache, root, "TEMPERATURE")
    assert first.mean[0, 20] == 282.0
    assert first.counts[0, 20] == 1
    cached = load_profile(cache, root, "TEMPERATURE")
    np.testing.assert_array_equal(first.mean, cached.mean)
    write_field(root, "TEMPERATURE", [286.0, 288.0, np.nan], hour=1)
    changed = load_profile(cache, root, "TEMPERATURE")
    assert changed.n_files == 2
    assert changed.mean[0, 21] == 287.0


def test_updraft_product_preserves_domain_and_invalidates_old_cache(tmp_path):
    root = tmp_path / "inputs"
    cache = tmp_path / "flux.npz"
    write_field(root, "UD_OMEGA", [-G, np.nan])
    write_field(root, "UD_MESH_FRAC", [1.0, np.nan])
    np.savez(cache, mean=np.full((1, 24), 0.5), __signature__="old")
    product = load_profile(cache, root, "UPDRAFT_FLUX")
    assert product.mean[0, 20] == 1.0
    with np.load(cache) as payload:
        assert "calculation_version" in str(payload["__metadata__"])


def test_replacing_input_at_same_path_invalidates_cache(tmp_path):
    root = tmp_path / "inputs"
    cache = tmp_path / "temperature.npz"
    write_field(root, "TEMPERATURE", [280.0])
    assert load_profile(cache, root, "TEMPERATURE").mean[0, 20] == 280.0
    write_field(root, "TEMPERATURE", [290.0])
    assert load_profile(cache, root, "TEMPERATURE").mean[0, 20] == 290.0


def test_limited_run_does_not_replace_full_profile_cache(tmp_path):
    root = tmp_path / "inputs"
    cache = tmp_path / "temperature.npz"
    write_field(root, "TEMPERATURE", [280.0])
    full = load_profile(cache, root, "TEMPERATURE")
    original = cache.read_bytes()
    limited = load_profile(cache, root, "TEMPERATURE", max_days=1)
    np.testing.assert_array_equal(full.mean, limited.mean)
    assert cache.read_bytes() == original


def test_temperature_overlay_retains_offline_cache_support(tmp_path):
    from alaro_analysis.data.hydrometeor_profiles import load_or_compute_diurnal

    cache = tmp_path / "temperature.npz"
    expected = np.full((2, 24), 280.0)
    np.savez(cache, mean=expected, counts=np.ones((2, 24), dtype=int))
    mean, _, low, high = load_or_compute_diurnal(
        cache,
        tmp_path / "offline",
        "TEMPERATURE",
        None,
        False,
        None,
        -4,
    )
    np.testing.assert_array_equal(mean, expected)
    assert (low, high) == (1, 1)
