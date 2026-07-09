from __future__ import annotations

import ast
import re
from pathlib import Path

import kg_processor

_SOURCE_ROOT = Path("src/kg_processor")


def test_production_modules_have_review_oriented_module_docstrings() -> None:
    missing: list[str] = []

    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(str(path))

    assert missing == []


def test_public_production_surfaces_have_review_oriented_docstrings() -> None:
    missing: list[str] = []

    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    missing.append(f"{path}:{node.name}")
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not child.name.startswith("_")
                        and not ast.get_docstring(child)
                    ):
                        missing.append(f"{path}:{node.name}.{child.name}")
            elif (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
                and not ast.get_docstring(node)
            ):
                missing.append(f"{path}:{node.name}")

    assert missing == []


def test_readme_keeps_quick_start_paths_current_and_direct() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "TODO" not in readme
    assert "# FlakeGraph" in readme
    assert "docs/assets/flakegraph-logo.png" in readme
    assert "turns documents into a grounded knowledge graph" in readme
    assert "provider-independent" in readme
    assert "Use Python 3.13" in readme
    assert "uv run flakegraph worker" in readme

    required_paths = [
        "configs/local-smoke.yaml",
        "configs/local-mineru-oss.yaml",
        "configs/README.md",
        "configs/snowflake-cortex.yaml",
        "data/samples/martial-arts-overview.pdf",
        "docs/snowflake-setup.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ]
    for path in required_paths:
        assert path in readme
        assert Path(path).exists()


def test_config_readme_documents_every_yaml_profile() -> None:
    config_readme = Path("configs/README.md").read_text(encoding="utf-8")
    config_files = sorted(Path("configs").glob("*.yaml"))

    assert config_files
    assert ".github/docker-compose.smoke.yaml" in config_readme
    for path in config_files:
        assert f"`{path.name}`" in config_readme


def test_third_party_notices_cover_runtime_license_boundaries() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    required_phrases = [
        "FlakeGraph source code is licensed under the MIT License",
        "does not re-license",
        "mineru[pipeline]",
        "MinerU Open Source License",
        "additional terms",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Apache-2.0",
        "Azure OpenAI",
        "vLLM",
        "Snowflake",
        "Cortex",
        "tesseract-ocr",
        "poppler-utils",
        "GPL/LGPL/MIT",
    ]

    for phrase in required_phrases:
        assert phrase in notices
    assert kg_processor.__version__


def test_snowflake_docs_stay_public_and_template_based() -> None:
    setup_notes = Path("docs/snowflake-setup.md").read_text(encoding="utf-8")

    required_phrases = [
        "account-neutral",
        "KG_PROCESSOR_ROLE",
        "CREATE COMPUTE POOL",
        "KG_PROCESSOR_CPU_POOL",
        "KG_*",
        "image repository",
        "KG_RUN_SNOWFLAKE_LIVE",
    ]

    for phrase in required_phrases:
        assert phrase in setup_notes

    forbidden_patterns = [
        r"https://app\.snowflake\.com/[^/\s`]+/[^)\s`]+",
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        r"sha256:[0-9a-f]{64}",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, setup_notes, flags=re.IGNORECASE) is None
