"""Shared CLI config loading for ALARO workflow commands."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "alaro.toml"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common --config argument to a workflow parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional TOML config file. If omitted, ./alaro.toml is used "
            "when present."
        ),
    )


def parse_configured_args(
    parser: argparse.ArgumentParser,
    workflow_name: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI args and apply optional TOML config defaults.

    Command line values win over config values. Config values win over
    argparse defaults.
    """
    supplied_dests = supplied_cli_dests(parser, argv)
    args = parser.parse_args(argv)

    try:
        config_path = resolve_config_path(args.config)
        if config_path is None:
            args.config = None
            return args

        config = load_config(config_path)
        apply_config(args, parser, config, workflow_name, supplied_dests)
        args.config = config_path
        return args
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
        raise AssertionError("argparse parser.error should exit") from exc


def resolve_config_path(
    explicit_path: str | Path | None,
    *,
    cwd: Path | None = None,
) -> Path | None:
    """Resolve the config path according to project lookup rules."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    base = Path.cwd() if cwd is None else cwd
    candidate = base / DEFAULT_CONFIG_NAME
    if candidate.exists():
        return candidate
    return None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an ALARO TOML config file."""
    config_path = Path(path).expanduser()
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a TOML table: {config_path}")
    return config


def workflow_config(config: Mapping[str, Any], workflow_name: str) -> dict[str, Any]:
    """Return defaults merged with a workflow-specific config table."""
    merged: dict[str, Any] = {}

    defaults = config.get("defaults", {})
    if defaults is not None:
        if not isinstance(defaults, Mapping):
            raise ValueError("[defaults] must be a TOML table")
        merged.update(defaults)

    workflows = config.get("workflows", {})
    if workflows is not None:
        if not isinstance(workflows, Mapping):
            raise ValueError("[workflows] must be a TOML table")
        table = workflows.get(workflow_name)
        if table is None:
            table = workflows.get(workflow_name.replace("_", "-"))
        if table is not None:
            if not isinstance(table, Mapping):
                raise ValueError(f"[workflows.{workflow_name}] must be a TOML table")
            merged.update(table)

    return merged


def apply_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: Mapping[str, Any],
    workflow_name: str,
    supplied_dests: set[str],
) -> None:
    """Apply config values to parsed args unless the CLI supplied the value."""
    values = workflow_config(config, workflow_name)
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help", "config")
    }

    for dest, raw_value in values.items():
        if dest in supplied_dests or dest not in actions or not hasattr(args, dest):
            continue
        value = convert_value(raw_value, actions[dest])
        validate_choices(dest, value, actions[dest])
        setattr(args, dest, value)


def supplied_cli_dests(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> set[str]:
    """Return argparse destinations explicitly present on the command line."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    option_dests = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }

    supplied: set[str] = set()
    for token in tokens:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None:
            supplied.add(dest)
    return supplied


def convert_value(raw_value: Any, action: argparse.Action) -> Any:
    """Convert TOML values using the argparse action metadata."""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return as_bool(raw_value)

    if action.nargs in ("+", "*") or action.nargs is not None:
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        return [convert_scalar(v, action.type) for v in values]

    return convert_scalar(raw_value, action.type)


def convert_scalar(raw_value: Any, converter: Any) -> Any:
    if converter is None:
        return raw_value
    if isinstance(raw_value, converter) if isinstance(converter, type) else False:
        return raw_value
    return converter(raw_value)


def as_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected boolean config value, got {raw_value!r}")


def validate_choices(dest: str, value: Any, action: argparse.Action) -> None:
    choices = action.choices
    if choices is None:
        return

    values = value if isinstance(value, list) else [value]
    invalid = [v for v in values if v not in choices]
    if invalid:
        allowed = ", ".join(str(v) for v in choices)
        raise ValueError(
            f"Invalid config value for {dest}: {invalid[0]!r}. "
            f"Expected one of: {allowed}"
        )
