from __future__ import annotations

import ast
import re
from pathlib import Path

import kg_processor

_SOURCE_ROOT = Path("src/kg_processor")


def test_production_modules_have_module_docstrings() -> None:
    missing: list[str] = []

    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(str(path))

    assert missing == []


def test_public_production_surfaces_have_docstrings() -> None:
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
    assert "app/assets/flakegraph-logo.png" in readme
    assert "docs/assets/flakegraph-pipeline.svg" in readme
    assert "docs/algorithm.md" in readme
    assert "turns documents into evidence-backed knowledge graphs" in readme
    assert "provider interfaces" in readme
    assert "Python 3.14" in readme
    assert "qwen3.6:35b-a3b-q4_K_M" in readme
    assert "qwen3.5:9b" in readme
    assert "https://docs.ollama.com/quickstart" in readme
    assert "data/martial_arts/files" in readme
    assert 'uv tool install --python 3.13 "mineru[pipeline]==3.4.4"' in readme
    assert "uv sync --extra app --extra local-embeddings" in readme
    assert "uv run streamlit run app/streamlit_app.py" in readme
    assert "The CLI remains available for headless and" in readme

    required_paths = [
        "configs/local-mineru-oss.yaml",
        "app/README.md",
        "app/streamlit_app.py",
        "configs/README.md",
        "configs/snowflake-cortex.yaml",
        "data/martial_arts/files/martial-arts-overview.pdf",
        "docs/architecture.md",
        "docs/algorithm.md",
        "docs/kubernetes-fleet.md",
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
    for path in config_files:
        assert f"`{path.name}`" in config_readme


def test_explanatory_guides_keep_their_mermaid_visuals_well_formed() -> None:
    """Protect the diagrams that make the public guides easier to scan.

    Mermaid CLI rendering is performed during documentation review because it
    requires a browser. This focused contract catches missing visuals and
    unclosed Mermaid fences in the fast Python suite without pretending to
    duplicate Mermaid's parser.
    """

    public_markdown = [
        *Path(".").glob("*.md"),
        *Path("configs").rglob("*.md"),
        *Path("data").rglob("*.md"),
        *Path("docs").rglob("*.md"),
    ]
    visual_guides = [
        path
        for path in sorted(set(public_markdown))
        if "```mermaid" in path.read_text(encoding="utf-8")
    ]
    assert visual_guides, "public documentation should contain explanatory diagrams"

    for path in visual_guides:
        guide = path.read_text(encoding="utf-8")
        diagrams = re.findall(r"```mermaid\n(.*?)\n```", guide, flags=re.DOTALL)

        assert diagrams, f"{path} should contain at least one explanatory diagram"
        assert len(diagrams) == guide.count("```mermaid"), (
            f"{path} contains an unclosed Mermaid code fence"
        )
        assert all(
            diagram.lstrip().startswith(("flowchart ", "sequenceDiagram")) for diagram in diagrams
        ), f"{path} contains a Mermaid block without an explicit diagram type"
        reserved_identifiers = {
            "class",
            "click",
            "direction",
            "end",
            "flowchart",
            "graph",
            "linkstyle",
            "style",
            "subgraph",
        }
        declared_identifiers = {
            match.casefold()
            for diagram in diagrams
            for match in re.findall(
                r"(?m)^\s{4}([A-Za-z][A-Za-z0-9_]*)\s*(?=\[|\(|\{|-->|<-->)",
                diagram,
            )
        }
        assert declared_identifiers.isdisjoint(reserved_identifiers), (
            f"{path} uses a Mermaid grammar keyword as a node identifier: "
            f"{sorted(declared_identifiers & reserved_identifiers)}"
        )


def test_readme_pipeline_visual_is_a_self_contained_accessible_svg() -> None:
    """Keep the primary algorithm overview reviewable and GitHub-independent."""

    visual = Path("docs/assets/flakegraph-pipeline.svg").read_text(encoding="utf-8")

    assert visual.startswith("<svg")
    assert "<title>" in visual
    assert "<desc>" in visual
    assert "Documents in. Evidence-backed graph out." in visual
    assert "DOCUMENT-WIDE TWO-PHASE EXTRACTION" in visual
    assert "Document entity inventory barrier" in visual
    assert "CORPUS FINALIZATION" in visual
    assert "LOCAL" in visual
    assert "KUBERNETES" in visual
    assert "SNOWFLAKE" in visual
    assert "<image" not in visual
    assert re.findall(r"https?://[^\"\s]+", visual) == ["http://www.w3.org/2000/svg"]


def test_third_party_notices_cover_runtime_license_boundaries() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    required_phrases = [
        "FlakeGraph source code is licensed under the Apache License 2.0",
        "does not re-license",
        "mineru[pipeline]",
        "MinerU Open Source License",
        "additional terms",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Apache-2.0",
        "vLLM",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "Azure OpenAI",
        "vLLM",
        "NVIDIA's vLLM container",
        "NGC",
        "KEDA",
        "Snowflake",
        "Cortex",
        "plotly",
        "Plotly.js",
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
