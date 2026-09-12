"""The deployed Streamlit bundle must import only what it ships.

`snowflake.yml` uploads `flakegraph_app`, its assets and two configs — not the
`kg_processor` package. A module-level import of the product therefore breaks the
entire application inside Snowflake with "No module named 'kg_processor'", before
a single page renders.

Function-level imports are the established exception: they run only on code paths
where the package is present, and `graph_store` documents that deliberately.
"""

from __future__ import annotations

import ast
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2] / "app"
_PRODUCT_PACKAGE = "kg_processor"


def _module_level_product_imports(tree: ast.Module) -> list[str]:
    """Return product imports reachable at import time."""

    found: list[str] = []
    for node in tree.body:
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.If):
            # A TYPE_CHECKING guard never executes at runtime.
            continue
        found.extend(name for name in modules if name.split(".")[0] == _PRODUCT_PACKAGE)
    return found


def test_the_app_bundle_does_not_import_the_product_at_module_level() -> None:
    violations: list[str] = []
    for path in sorted(_APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations.extend(
            f"{path.relative_to(_APP_ROOT)}: {module}"
            for module in _module_level_product_imports(tree)
        )

    assert violations == []
