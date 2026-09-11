"""List, inspect or execute a named study recipe without remembering its path."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from importlib.resources import files


def catalog() -> list[dict]:
    return json.loads(files("alaro_analysis.recipes").joinpath("catalog.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "show", "run"), nargs="?", default="list")
    parser.add_argument("name", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    entries = catalog()
    if args.action == "list":
        for entry in entries:
            if args.name is None or entry["family"] == args.name:
                print(f"{entry['family']:14} {entry['name']}\n  {entry['description']}")
        return 0
    matches = [entry for entry in entries if entry["name"] == args.name]
    if not matches:
        parser.error(f"Unknown recipe {args.name!r}; use 'alaro-recipe list'")
    entry = matches[0]
    if args.action == "show":
        print(json.dumps(entry, indent=2))
        return 0
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    original = sys.argv
    try:
        sys.argv = [entry["module"], *arguments]
        runpy.run_module(entry["module"], run_name="__main__")
    finally:
        sys.argv = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
