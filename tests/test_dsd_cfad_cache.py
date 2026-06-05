from __future__ import annotations

import numpy as np

from alaro_analysis.data.cache import array_fingerprint, signature


def test_signature_is_order_independent_and_deterministic():
    a = signature({"x": 1, "y": [1, 2], "arr": np.arange(5.0)})
    b = signature({"arr": np.arange(5.0), "y": [1, 2], "x": 1})
    assert a == b and isinstance(a, str) and a


def test_signature_changes_with_each_parameter():
    base = signature({"min_qr": 1e-7, "arr": np.arange(5.0)})
    assert base != signature({"min_qr": 2e-7, "arr": np.arange(5.0)})  # scalar change
    assert base != signature({"min_qr": 1e-7, "arr": np.arange(6.0)})  # array shape
    assert base != signature({"min_qr": 1e-7, "arr": np.arange(5.0) + 1})  # array values


def test_array_fingerprint_distinguishes_shape_and_values():
    assert array_fingerprint(np.zeros(4)) != array_fingerprint(np.zeros(5))
    assert array_fingerprint(np.zeros(4)) != array_fingerprint(np.ones(4))


def test_dsd_cfad_cache_roundtrip_and_signature(tmp_path):
    from alaro_analysis.workflows.dsd_cfad_profiles import (
        CfadGrid,
        ExperimentCfad,
        PANEL_FIELDS,
        load_experiment_cfad,
        read_cache_signature,
        save_experiment_cfad,
    )

    grids = {
        field: CfadGrid(
            height_km=np.arange(3.0),
            x_edges=np.arange(4.0),
            hist=np.ones((2, 3), dtype=np.int64),
            counts=np.ones((2, 3), dtype=np.int64),
            n_profiles=5,
            source="src",
        )
        for field in PANEL_FIELDS
    }
    cfad = ExperimentCfad(grids=grids, temperature_k=np.arange(3.0), freezing_level_km=4.5)
    path = tmp_path / "control_tag.npz"
    save_experiment_cfad(path, cfad, sig="sig-abc")

    assert read_cache_signature(path) == "sig-abc"
    loaded = load_experiment_cfad(path)
    assert loaded.freezing_level_km == 4.5
    np.testing.assert_array_equal(
        loaded.grids[PANEL_FIELDS[0]].hist, np.ones((2, 3), dtype=np.int64)
    )
