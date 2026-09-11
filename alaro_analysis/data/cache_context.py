"""Validated cache policy for products with several source variables.

Build one context per run (or experiment), before computation. Its inventory is
fixed for that run; subsequent runs rescan sources, so replaced or added files
cannot reuse an earlier result. Numerical arrays keep their existing NPZ keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .cache import (
    load_cache,
    read_cache_signature,
    save_cache,
    signature,
    source_fingerprint,
)

# Presentation and execution choices do not change the numerical population.
_EXECUTION_SETTINGS = {
    "config",
    "show_config",
    "workers",
    "n_workers",
    "dpi",
    "output",
    "output_dir",
    "intermediate_dir",
    "cache_dir",
    "overwrite_intermediate",
    "force",
    "no_cache",
    "recompute",
}


def netcdf_inventory(roots: Iterable[Path]) -> list[Path]:
    """List files within selected variable roots, including absent roots."""
    files: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_dir():
            files.update(root.rglob("*.nc"))
        else:
            files.add(root)
    return sorted(files)


@dataclass(frozen=True)
class CacheContext:
    """Scientific settings and source inventory associated with a run."""

    product: str
    settings: Mapping[str, Any]
    sources: str
    calculation_version: int = 2

    @classmethod
    def from_netcdf(cls, product: str, settings: Mapping[str, Any], roots: Iterable[Path]):
        scientific = {k: v for k, v in settings.items() if k not in _EXECUTION_SETTINGS}
        return cls(product, scientific, source_fingerprint(netcdf_inventory(roots)))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "calculation_version": self.calculation_version,
            "product": self.product,
            "settings": dict(self.settings),
            "source_inventory": self.sources,
        }

    def cache_signature(self, path: Path) -> str:
        return signature({**self.metadata(), "product_path": str(path.resolve())})

    def valid(self, path: Path) -> bool:
        return read_cache_signature(path) == self.cache_signature(path)

    def arrays(self, path: Path) -> dict[str, np.ndarray]:
        return {
            "__signature__": np.array([self.cache_signature(path)]),
            "__metadata__": np.array(json.dumps(self.metadata(), sort_keys=True, default=str)),
        }

    def load(self, path: Path) -> dict[str, np.ndarray]:
        if not self.valid(path):
            raise ValueError(f"Cache provenance does not match this run: {path}")
        return {k: v for k, v in load_cache(path).items() if not k.startswith("__")}

    def save(self, path: Path, payload: dict[str, Any]) -> None:
        save_cache(path, {**payload, **self.arrays(path)})
