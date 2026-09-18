"""Render every Streamlit page the way Snowflake will.

A parameter the deployed Streamlit does not support, or a chart it cannot draw,
raises only when the page is rendered — never at import, and never in a unit test
that exercises the backend alone. `st.popover(key=...)` reached a deployed app and
crashed the graph page exactly this way, and nothing in the suite could have
caught it, because nothing rendered that page.

These tests drive the app through Streamlit's own harness, pinned to the version
`app/environment.yml` deploys, so a page that cannot render in Snowflake cannot
pass here either.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from flakegraph_app.models import RuntimeMode, SourceKind
from flakegraph_app.run_catalog import write_run_record
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup

from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.domain.graph import (
    Chunk,
    Community,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_APP_ROOT = _REPOSITORY_ROOT / "app"
_APP_SCRIPT = _APP_ROOT / "streamlit_app.py"


@pytest.fixture
def completed_run() -> Iterator[str]:
    """Publish one finished local run, then remove it.

    The app resolves its catalog from the repository root, so the fixture is
    written there under a unique id rather than into a temporary directory.
    """

    run_id = f"apptest-{uuid.uuid4().hex[:12]}"
    state_root = _REPOSITORY_ROOT / ".flakegraph" / "app"
    run_directory = state_root / "runs" / run_id
    output_path = state_root / "artifacts" / run_id
    try:
        LocalArtifactsWriter(output_path).write(_graph_batch("apptest-graph"))
        write_run_record(
            run_directory,
            {
                "run_id": run_id,
                "graph_id": "apptest-graph",
                "runtime": "local",
                "status": "succeeded",
                "output_path": str(output_path),
            },
        )
        yield run_id
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)
        shutil.rmtree(output_path, ignore_errors=True)


def _graph_batch(graph_id: str) -> GraphWriteBatch:
    """Build the smallest graph the explorer will still draw."""

    chunk = Chunk(
        id="chunk-1",
        graph_id=graph_id,
        file_id="file-1",
        document_id="doc-1",
        page_number=1,
        chunk_index=0,
        content="Jigoro Kano founded Judo at the Kodokan.",
        start_offset=0,
        end_offset=40,
        token_count=8,
        content_hash="hash-1",
    )
    nodes = [
        GraphNode(
            id=f"node-{index}",
            graph_id=graph_id,
            normalized_name=name.lower(),
            name=name,
            primary_type=node_type,
            types=[node_type],
            description=f"{name} description",
            source_chunk_ids=["chunk-1"],
        )
        for index, (name, node_type) in enumerate(
            [("Jigoro Kano", "PERSON"), ("Judo", "MARTIAL_ART")], start=1
        )
    ]
    edge = GraphEdge(
        id="edge-1",
        graph_id=graph_id,
        source_node_id="node-1",
        target_node_id="node-2",
        relation_type="FOUNDED",
        description="Kano founded Judo.",
        weight=1.0,
        confidence=0.95,
        source_file_id="file-1",
        source_file_ids=["file-1"],
        source_chunk_ids=["chunk-1"],
        evidence_count=1,
    )
    return GraphWriteBatch(
        graph_id=graph_id,
        documents=[
            {
                "id": "doc-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "source_uri": "local://doc-1",
                "checksum": "hash-1",
                "mime_type": "text/plain",
                "size_bytes": 40,
            }
        ],
        pages=[
            {
                "id": "page-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "document_id": "doc-1",
                "page_number": 1,
                "text": chunk.content,
            }
        ],
        chunks=[chunk],
        nodes=nodes,
        edges=[edge],
        evidence=[
            Evidence(
                id="evidence-1",
                graph_id=graph_id,
                subject_id="edge-1",
                subject_kind="edge",
                chunk_id="chunk-1",
                file_id="file-1",
                page_number=1,
                start_offset=0,
                end_offset=40,
                quote=chunk.content,
            )
        ],
        entity_sources=[
            {
                "id": f"source-{node.id}",
                "graph_id": graph_id,
                "node_id": node.id,
                "chunk_id": "chunk-1",
                "file_id": "file-1",
                "mention_count": 1,
                "per_file_description": f"{node.name} appears in this document.",
            }
            for node in nodes
        ],
        communities=[
            Community(
                id="community-1",
                graph_id=graph_id,
                stable_key="judo",
                level=0,
                title="Judo",
                summary="Founding of Judo.",
                rating=1.0,
                member_node_ids=["node-1", "node-2"],
            )
        ],
        community_findings=[],
        run_report={"run_id": "apptest", "graph_id": graph_id},
    )


def _app() -> AppTest:
    return AppTest.from_file(str(_APP_SCRIPT), default_timeout=240)


def test_the_ingestion_page_renders() -> None:
    """The landing page must survive on the deployed Streamlit version."""

    app = _app()
    app.run()

    assert not app.exception
    assert "Build a graph" in [title.value for title in app.title]


@pytest.mark.parametrize("runtime", ["Local", "Kubernetes", "Snowflake"])
def test_every_runtime_renders(runtime: str) -> None:
    """Selecting a runtime must not raise, even where its backend is unavailable.

    Snowflake mode outside Snowflake has no Snowpark session; the app is expected
    to explain that rather than fail to render.
    """

    app = _app()
    app.session_state["navigation_runtime"] = runtime.lower()
    app.run()

    assert not app.exception


def test_the_completed_graph_page_renders(completed_run: str) -> None:
    """The page that crashed in Snowflake, rendered end to end.

    This is the regression guard for unsupported Streamlit parameters: the graph
    heading, its rename control, the explorer, and the chart all draw here.
    """

    app = _app()
    app.session_state["navigation_runtime"] = "local"
    app.session_state["active_page"] = "run"
    app.session_state["selected_run_id"] = completed_run
    app.run()

    assert not app.exception
    assert "apptest-graph" in [title.value for title in app.title]

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Entities"] == "2"
    assert metrics["Relations"] == "1"
    assert metrics["Evidence"] == "1"

    assert [tab.label for tab in app.tabs][:3] == ["Explorer", "Consumption", "Run details"]
    # The explorer's controls must exist, or the canvas has silently gone missing.
    assert {"Entity types", "Relation types", "Communities"}.issubset(
        {widget.label for widget in app.multiselect}
    )


@pytest.fixture
def single_select_button_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the harness rerun a page holding a segmented control.

    Its button-group model was written for ``st.feedback`` and reads every value
    as a list; a segmented control holds one value, which it would iterate as
    characters when it serialises the widget for the next run.
    """

    def indices(self: ButtonGroup[Any]) -> list[int]:
        values = self.value if isinstance(self.value, list) else [self.value]
        return [self.options.index(self.format_func(value)) for value in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))


@pytest.mark.usefixtures("single_select_button_groups")
def test_a_sidebar_action_keeps_a_half_filled_source_form(completed_run: str) -> None:
    """Opening the removal dialog must not wipe the bucket form beside it.

    A sidebar button that changed state and called ``st.rerun`` ended the run
    before the form's widgets were seen, and Streamlit swept their state as
    stale; the operator came back to an empty form.
    """

    app = _app()
    app.session_state["navigation_runtime"] = "local"
    app.session_state["active_page"] = "new"
    app.run()
    # Filled the way an operator fills it: a value the page seeds itself is
    # not widget state, and would survive the sweep this test is about.
    # A single-select control, whose harness model is typed for lists.
    app.button_group(key="ingest_source_kind").set_value(SourceKind.S3).run()  # type: ignore[arg-type]
    app.text_input(key="ingest_s3_bucket").set_value("test-corpora").run()
    app.button(key=f"forget_local_{completed_run}").click().run()

    assert not app.exception
    assert app.session_state["pending_forget_run_id"] == completed_run
    assert app.session_state["ingest_source_kind"] == SourceKind.S3
    assert app.session_state["ingest_s3_bucket"] == "test-corpora"


def test_the_cluster_manager_renders() -> None:
    """Registering a cluster must be possible without leaving the application."""

    app = _app()
    app.session_state["runtime_selector"] = RuntimeMode.KUBERNETES
    app.session_state["navigation_runtime"] = "kubernetes"
    app.session_state["active_page"] = "clusters"
    app.run()

    assert not app.exception
    assert "Clusters" in [title.value for title in app.title]


def test_only_the_snowflake_runtime_is_offered_inside_snowflake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other runtimes cannot work there, so offering them would mislead.

    Local and Kubernetes drive work by running CLI processes against a checkout
    and a kubeconfig, neither of which exists in Streamlit in Snowflake.
    """

    def _session() -> object:
        """Stand in for a Snowpark session, which only exists inside Snowflake."""

        return object()

    monkeypatch.setattr("flakegraph_app.backends.factory.active_snowflake_session", _session)

    app = _app()
    app.run()

    runtimes = [selectbox for selectbox in app.selectbox if selectbox.label == "Runtime"]
    assert runtimes
    assert [str(option) for option in runtimes[0].options] == ["Snowflake"]
