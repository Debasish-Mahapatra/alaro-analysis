import json

import numpy as np
import pytest

from alaro_analysis.data.cache import save_npz_compressed, signature
from alaro_analysis.data.cache_context import CacheContext


def test_cache_tracks_sources_and_sample_selection_but_not_worker_count(tmp_path):
    source = tmp_path / "source.nc"
    source.write_bytes(b"first")
    target = tmp_path / "product.npz"
    context = CacheContext.from_netcdf("test", {"max_days": None, "workers": 32}, [source])
    context.save(target, {"mean": np.array([3.0]), "counts": np.array([2])})
    changed_workers = CacheContext.from_netcdf("test", {"max_days": None, "workers": 1}, [source])
    assert changed_workers.valid(target)
    assert set(context.load(target)) == {"mean", "counts"}
    short_run = CacheContext.from_netcdf("test", {"max_days": 1}, [source])
    assert not short_run.valid(target)
    source.write_bytes(b"replacement")
    assert not CacheContext.from_netcdf("test", {"max_days": None}, [source]).valid(target)
    with np.load(target) as arrays:
        assert json.loads(str(arrays["__metadata__"]))["schema_version"] == 2


def test_nested_bin_fingerprint_uses_all_values():
    first = np.arange(5000.0)
    second = first.copy()
    second[2500] += 0.1
    assert signature({"bins": {"x": first}}) != signature({"bins": {"x": second}})


def test_failed_atomic_save_keeps_previous_cache(tmp_path, monkeypatch):
    target = tmp_path / "complete.npz"
    save_npz_compressed(target, mean=[42.0])
    previous = target.read_bytes()

    def interrupted_write(stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", interrupted_write)
    with pytest.raises(OSError, match="disk full"):
        save_npz_compressed(target, mean=[99.0])
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]
