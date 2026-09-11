"""Guard dependency direction so future scripts cannot become hidden libraries."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "alaro_analysis"


def test_reusable_modules_do_not_import_executable_workflows_or_recipes():
    violations = []
    for family in ("analysis", "common", "data", "dsd", "plotting"):
        for path in (PACKAGE / family).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                else:
                    continue
                if any(
                    name.startswith(("alaro_analysis.workflows", "alaro_analysis.recipes"))
                    for name in imports
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, violations


def test_recipe_catalog_and_legacy_launchers_resolve_to_packaged_modules():
    entries = json.loads((PACKAGE / "recipes" / "catalog.json").read_text())
    assert len({entry["name"] for entry in entries}) == len(entries)
    for entry in entries:
        module = ROOT / (entry["module"].replace(".", "/") + ".py")
        assert module.is_file(), entry
        tree = ast.parse(module.read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), entry
        wrapper = ROOT / entry["legacy_script"]
        assert entry["module"] in wrapper.read_text(), entry
