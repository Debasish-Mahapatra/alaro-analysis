from __future__ import annotations

import argparse
from pathlib import Path

from alaro_analysis.common.cli_config import (
    add_config_argument,
    load_config,
    parse_configured_args,
    resolve_config_path,
    workflow_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("/default/out"))
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--seasons", nargs="+", default=["wet", "dry"])
    parser.add_argument("--overwrite-intermediate", action="store_true")
    return parser


def test_missing_config_falls_back_to_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None, cwd=tmp_path) is None

    args = parse_configured_args(_parser(), "surface", argv=[])
    assert args.output_dir == Path("/default/out")
    assert args.n_workers == 16


def test_current_directory_config_lookup(tmp_path: Path):
    config_path = tmp_path / "alaro.toml"
    config_path.write_text("[defaults]\nn_workers = 8\n", encoding="utf-8")

    assert resolve_config_path(None, cwd=tmp_path) == config_path


def test_current_directory_config_is_applied(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alaro.toml").write_text(
        "[defaults]\nn_workers = 8\n",
        encoding="utf-8",
    )

    args = parse_configured_args(_parser(), "surface", argv=[])
    assert args.n_workers == 8


def test_explicit_config_path_loads(tmp_path: Path):
    config_path = tmp_path / "custom.toml"
    config_path.write_text("[defaults]\nn_workers = 4\n", encoding="utf-8")

    assert resolve_config_path(config_path) == config_path
    assert load_config(config_path)["defaults"]["n_workers"] == 4


def test_workflow_config_overrides_defaults(tmp_path: Path):
    config_path = tmp_path / "alaro.toml"
    config_path.write_text(
        """
[defaults]
n_workers = 8
seasons = ["wet"]

[workflows.hydrometeor]
n_workers = 24
""".strip(),
        encoding="utf-8",
    )

    merged = workflow_config(load_config(config_path), "hydrometeor")
    assert merged["n_workers"] == 24
    assert merged["seasons"] == ["wet"]


def test_cli_values_override_config(tmp_path: Path):
    config_path = tmp_path / "alaro.toml"
    config_path.write_text(
        """
[defaults]
output_dir = "/config/out"
n_workers = 8
seasons = ["wet"]
overwrite_intermediate = true
""".strip(),
        encoding="utf-8",
    )

    args = parse_configured_args(
        _parser(),
        "surface",
        argv=[
            "--config",
            str(config_path),
            "--output-dir",
            "/cli/out",
            "--seasons",
            "dry",
        ],
    )

    assert args.output_dir == Path("/cli/out")
    assert args.n_workers == 8
    assert args.seasons == ["dry"]
    assert args.overwrite_intermediate is True
