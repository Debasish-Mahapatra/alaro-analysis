"""Shared CLI config loading for ALARO workflow commands."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alaro_analysis.common.experiments import EXPERIMENT_CATALOG, canonical_experiment
from alaro_analysis.common.paths import DATA_ROOT, RUNS_ROOT, WORKSPACE_ROOT

DEFAULT_CONFIG_NAME = "alaro.toml"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add common config and inspection options, safely when called twice."""
    if "--config" not in parser._option_string_actions:
        parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="TOML configuration (default: ./alaro.toml when present).",
        )
    if "--show-config" not in parser._option_string_actions:
        parser.add_argument(
            "--show-config",
            action="store_true",
            help="Print resolved settings as JSON and exit before processing.",
        )


def parse_configured_args(
    parser: argparse.ArgumentParser,
    workflow_name: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Resolve built-in defaults < TOML defaults < workflow TOML < CLI.

    Defaults are applied before argparse parses the command, so abbreviations,
    aliases, repeated flags and explicit false values follow argparse's own
    precedence rules. Workflow keys are strict; unrelated shared defaults are
    ignored so one config can serve several analyses.
    """
    import json

    for action in parser._actions:
        if action.dest in {"experiment", "experiments"} and action.type in (None, str):
            if action.choices and set(action.choices).issubset(EXPERIMENT_CATALOG):
                action.type = canonical_experiment
    add_config_argument(parser)
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--help" in tokens or "-h" in tokens:
        return parser.parse_args(tokens)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--config", type=Path)
    selected, _ = selector.parse_known_args(tokens)
    try:
        config_path = resolve_config_path(selected.config)
        if config_path is not None:
            config = load_config(config_path)
            defaults = argparse.Namespace(
                **{
                    action.dest: parser.get_default(action.dest)
                    for action in parser._actions
                    if action.dest != argparse.SUPPRESS
                }
            )
            relocate_path_defaults(defaults, config.get("paths", {}))
            apply_config(defaults, parser, config, workflow_name, set())
            parser.set_defaults(**vars(defaults))
        args = parser.parse_args(tokens)
        args.config = config_path
        if args.show_config:
            print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
            parser.exit()
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
        if action.dest not in (argparse.SUPPRESS, "help", "config", "show_config")
    }

    workflows = config.get("workflows") or {}
    specific = (
        workflows.get(workflow_name, workflows.get(workflow_name.replace("_", "-"), {})) or {}
    )
    unknown = set(specific) - actions.keys()
    if unknown:
        raise ValueError(
            f"Unknown keys in [workflows.{workflow_name}]: {', '.join(sorted(unknown))}"
        )

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
        option: action.dest for action in parser._actions for option in action.option_strings
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
            f"Invalid config value for {dest}: {invalid[0]!r}. Expected one of: {allowed}"
        )


def relocate_path_defaults(args: argparse.Namespace, paths: Mapping[str, Any]) -> None:
    """Relocate built-in path defaults; explicit workflow and CLI values win."""
    roots = {"data_root": DATA_ROOT, "runs_root": RUNS_ROOT, "workspace_root": WORKSPACE_ROOT}
    if not isinstance(paths, Mapping):
        raise ValueError("[paths] must be a TOML table")
    unknown = set(paths) - roots.keys()
    if unknown:
        raise ValueError(f"Unknown [paths] keys: {', '.join(sorted(unknown))}")
    replacements = sorted(
        ((roots[key], Path(value).expanduser()) for key, value in paths.items()),
        key=lambda pair: len(pair[0].parts),
        reverse=True,
    )
    for key, value in vars(args).items():
        if not isinstance(value, (str, Path)):
            continue
        for old, new in replacements:
            try:
                relative = Path(value).relative_to(old)
            except ValueError:
                continue
            relocated = new / relative
            setattr(args, key, relocated if isinstance(value, Path) else str(relocated))
            break
