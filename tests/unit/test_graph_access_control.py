"""Contracts for who may see which graph in the Snowflake control plane.

Streamlit in Snowflake runs with owner's rights, so the warehouse returns every
row the app owner can read and these predicates are the only thing deciding what
a viewer is shown. Each test states a way that boundary can be lost.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pytest
from flakegraph_app.backends.snowflake import SnowflakeBackend
from flakegraph_app.configuration import redacted_url
from flakegraph_app.graph_store import GRAPH_DATA_TABLES
from flakegraph_app.models import (
    IngestionRequest,
    OutputDestination,
    ProviderSelection,
    RuntimeMode,
    SnowflakeOutput,
    SourceKind,
    StorageKind,
)
from flakegraph_app.viewer import Viewer, visible_graph_predicate
from streamlit.testing.v1 import AppTest

from kg_processor.adapters import snowflake as adapter
from kg_processor.adapters.files import snowflake_stage
from kg_processor.adapters.jobs.snowflake import SnowflakeJobFileProgressSink
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig, validate_stage_location
from kg_processor.application.progress import ProgressEvent
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.application.spark_finalization import _with_provider_environment


def test_an_unidentified_viewer_is_shown_no_private_graph() -> None:
    """Fail closed when Snowflake does not name the viewer.

    Identity is resolved from ``st.user``, which carries nothing outside
    Streamlit in Snowflake. Treating that absence as "everyone" would hand a
    misconfigured deployment every private graph in the account.
    """

    clause, parameters = visible_graph_predicate(Viewer(user_name=""))

    assert clause == "G.OWNER IS NULL"
    assert parameters == []


def test_a_viewer_sees_their_own_graphs_and_those_shared_with_them() -> None:
    """Admit exactly the owner, the named grantee, and the granted role."""

    viewer = Viewer(user_name="ALICE", roles=frozenset({"RESEARCHERS"}))

    clause, parameters = visible_graph_predicate(viewer)

    assert "UPPER(G.OWNER) = ?" in clause
    assert "KG_GRAPH_ACL" in clause
    # The owner comparison, then one placeholder for the viewer and one for each
    # role they hold, so a share can name either.
    assert parameters == ["ALICE", "ALICE", "RESEARCHERS"]


def test_the_predicate_never_inlines_an_identity_into_sql() -> None:
    """Bind identity as parameters, because a username is attacker-influenced.

    Snowflake usernames come from the identity provider and can contain quotes.
    Interpolated into the predicate, one could close the literal and widen the
    visibility test to every graph.
    """

    viewer = Viewer(user_name="O'BRIEN' OR '1'='1", roles=frozenset())

    clause, parameters = visible_graph_predicate(viewer)

    assert "O'BRIEN" not in clause
    assert "O'BRIEN' OR '1'='1" in parameters


def test_a_graph_nobody_owns_stays_visible() -> None:
    """Keep graphs written before ownership was recorded readable.

    Hiding them would make existing work disappear on upgrade, which reads as
    data loss rather than as a policy change.
    """

    clause, _ = visible_graph_predicate(Viewer(user_name="ALICE"))

    assert "G.OWNER IS NULL" in clause


def test_listing_graphs_applies_the_visibility_predicate() -> None:
    """Filter the history query itself, not the rows it returns.

    Owner's rights means an unfiltered query returns every graph in the account.
    """


    statements: list[tuple[str, Any]] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append((statement, params))
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE", roles=frozenset({"RESEARCHERS"}))

    backend.list_runs(limit=10)

    listing = next(sql for sql, _ in statements if "FROM KG_JOB J" in sql)
    assert "WHERE" in listing
    assert "KG_GRAPH_ACL" in listing


def test_opening_a_graph_is_authorized_independently_of_the_listing() -> None:
    """Gate the open, since a graph is opened by an identifier held in state.

    A viewer who never saw a graph in their history can still hold its ID, so a
    filtered list is not by itself a boundary.
    """


    statements: list[tuple[str, Any]] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append((statement, params))
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")

    assert backend.can_open_graph("someone-elses-graph") is False
    check = statements[-1][0]
    assert "J.GRAPH_ID = ?" in check
    assert "OWNER" in check


def test_sharing_rejects_a_target_that_is_neither_user_nor_role() -> None:
    """Refuse a grantee type the visibility predicate cannot evaluate."""


    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            raise AssertionError("must not reach SQL")

    backend = SnowflakeBackend(Session())

    with pytest.raises(ValueError, match="user or a role"):
        backend.share_graph("graph-1", "EVERYONE", "public")


def test_claiming_a_graph_never_takes_one_that_is_already_owned() -> None:
    """Leave an existing owner in place when someone else re-runs their graph."""


    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="BOB")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "flakegraph_app.backends.snowflake.current_viewer",
            lambda: Viewer(user_name="BOB"),
        )
        backend._claim_graph("graph-1")

    merge = statements[-1]
    assert "WHEN MATCHED AND target.OWNER IS NULL" in merge


def test_resolving_viewer_roles_escapes_the_username_it_must_interpolate() -> None:
    """Escape the one identifier Snowflake will not accept as a parameter.

    ``SHOW GRANTS TO USER`` takes an identifier, not a bindable value, and
    usernames arrive from the identity provider. An unescaped quote would end
    the identifier and let the rest of the name be read as SQL.
    """


    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            return _Result([])

    SnowflakeBackend(Session()).viewer_roles('ALICE" OR "1"="1')

    assert statements[-1] == 'SHOW GRANTS TO USER "ALICE"" OR ""1""=""1"'


_SHARING_SCRIPT = """
from flakegraph_app.ui.run_workspace import _render_sharing
from flakegraph_app.viewer import Viewer
import streamlit as st


class Backend:
    capabilities = frozenset({"share"})

    def viewer(self):
        return Viewer(user_name=st.session_state["who"])

    def graph_owner(self, graph_id):
        return "ALICE"

    def graph_shares(self, graph_id):
        return [("USER", "BOB")]


_render_sharing(Backend(), "graph-1")
"""


def _sharing_buttons(who: str) -> set[str]:
    """Render the sharing panel as one viewer and return every control offered."""

    app = AppTest.from_string(_SHARING_SCRIPT)
    app.session_state["who"] = who
    app.run(timeout=30)
    return {item.label for item in app.button} | {
        str(getattr(item, "label", "")) for item in app.get("form_submit_button")
    }


def test_only_the_owner_is_offered_the_sharing_controls() -> None:
    """Show the share form to the owner and to nobody else.

    A viewer who can see a shared graph must not be able to widen that share:
    the panel is the control surface for the boundary, so rendering it for a
    non-owner would invite them to grant access they do not hold.
    """

    assert "Share" in _sharing_buttons("ALICE")
    assert "Share" not in _sharing_buttons("BOB")


def test_a_staged_object_name_cannot_redirect_the_download(tmp_path: Path) -> None:
    """Refuse a stage object whose name carries a second file transfer target.

    The name of an object inside a stage is chosen by whoever writes to that
    stage, and it decides the GET statement. Carrying its own ``file://`` target
    and a comment, it redirects the download to a path of the writer's choosing
    inside the worker container.
    """

    hostile = "@KG_DOCS/app/j1/x.pdf file:///app/configs; -- .pdf"

    with pytest.raises(ValueError, match="unquoted identifiers"):
        validate_stage_location(hostile)


def test_provider_credentials_reach_spark_executors_as_secret_references() -> None:
    """Keep an API key out of every executor Pod spec.

    Spark renders ``executorEnv`` as literal environment entries on each executor
    Pod, readable by anything able to read pods in the namespace — including the
    executors, which run over untrusted document text.
    """

    class Builder:
        def __init__(self) -> None:
            self.settings: dict[str, str] = {}

        def config(self, key: str, value: str) -> Builder:
            self.settings[key] = value
            return self

    class Distributed:
        spark_provider_secret = "flakegraph-providers"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KG_LLM_API_KEY", "sk-live-secret")
        patch.setenv("KG_LLM_MODEL", "some-model")
        builder = _with_provider_environment(Builder(), Distributed())

    assert builder.settings["spark.executorEnv.KG_LLM_MODEL"] == "some-model"
    assert "spark.executorEnv.KG_LLM_API_KEY" not in builder.settings
    assert (
        builder.settings["spark.kubernetes.executor.secretKeyRef.KG_LLM_API_KEY"]
        == "flakegraph-providers:KG_LLM_API_KEY"
    )
    assert "sk-live-secret" not in str(builder.settings)


def test_a_credential_carried_in_an_endpoint_query_is_recognized() -> None:
    """Detect the credential shapes real providers put in a query string.

    Google, Azure and GitLab all authenticate by query parameter, and the guard
    that keeps literal secrets out of generated run configuration decides from
    this vocabulary. A pagination cursor must still pass, or ordinary runs are
    refused over a parameter that is not a secret at all.
    """

    secret_bearing = [
        "https://generativelanguage.googleapis.com/v1?key=AIzaLIVE",
        "https://x.cognitiveservices.azure.com/ocr?subscription-key=abc",
        "https://gitlab.com/api/v4/x?private_token=glpat-SECRET",
        "https://s3.example.com/o?X-Amz-Signature=deadbeef",
    ]
    for url in secret_bearing:
        assert redacted_url(url) != url, url

    for url in ["https://api.example.com/v1?next_token=cursor", "https://api.example.com/v1?page=2"]:
        assert redacted_url(url) == url, url


def _stage_request(*, job_id: str, graph_id: str) -> Any:
    """Build a Snowflake-stage ingestion request naming one run and graph."""

    return IngestionRequest(
        runtime=RuntimeMode.SNOWFLAKE,
        job_id=job_id,
        graph_id=graph_id,
        source_kind=SourceKind.SNOWFLAKE_STAGE,
        source={"stage": "KG_DOCS", "prefix": "app/BOB/job"},
        ocr=ProviderSelection("snowflake_cortex"),
        llm=ProviderSelection("snowflake_cortex", model="mistral-large2"),
        embedding=ProviderSelection(
            "snowflake_cortex", model="snowflake-arctic-embed-l-v2.0", dimension=1024
        ),
        output=OutputDestination(
            StorageKind.SNOWFLAKE,
            Path("out"),
            snowflake=SnowflakeOutput(
                account="ACC",
                user="U",
                database="D",
                schema="S",
                warehouse="W",
                bulk_stage="KG_LOAD_STAGE",
            ),
        ),
    )


def test_a_run_id_belonging_to_another_graph_is_refused() -> None:
    """Refuse to adopt someone else's run id.

    Every write in submission is keyed on the run id, which is a free-text
    field. Adopting an existing run drops its job service, inserts documents
    into its queue, and writes a failure of the submitter's choosing onto it —
    the destructive power that cancelling was gated against.
    """

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            if "FROM KG_GRAPH G WHERE G.GRAPH_ID" in statement:
                return _Result([("ok",)])
            if statement.startswith("SELECT GRAPH_ID FROM KG_JOB"):
                return _Result([{"GRAPH_ID": "alices-graph"}])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="BOB")

    with pytest.raises(PermissionError, match="already belongs to another graph"):
        backend.submit(_stage_request(job_id="alices-run", graph_id="bobs-graph"))


def test_a_staged_object_that_expands_without_bound_is_rejected(tmp_path: Path) -> None:
    """Refuse a staged object whose compressed form hides an unbounded expansion.

    A gzip member declares nothing about its expanded size, so a small staged
    object can fill the worker's disk and take the run down with it.
    """

    bomb = tmp_path / "bomb.txt.gz"
    with gzip.open(bomb, "wb") as handle:
        handle.write(b"\0" * 4096)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(snowflake_stage, "_MAX_DECOMPRESSED_BYTES", 1024)
        with pytest.raises(ValueError, match="expands beyond"):
            snowflake_stage._restore_original_bytes(bomb)

    assert not (tmp_path / "bomb.txt").exists()


class _OwnedSession:
    """A Snowflake session where every graph belongs to ALICE."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []

    def sql(self, statement: str, params: Any = None) -> Any:
        self.statements.append((statement, params))
        if "FROM KG_GRAPH G WHERE G.GRAPH_ID" in statement:
            # The administration predicate matches nothing: BOB is not the owner.
            return _Result([])
        if statement.startswith("SELECT 1 FROM KG_GRAPH WHERE GRAPH_ID"):
            return _Result([("known",)])
        if statement.startswith("SELECT GRAPH_ID FROM KG_JOB"):
            return _Result([{"GRAPH_ID": "alices-graph"}])
        return _Result([])


def _backend_as(who: str) -> SnowflakeBackend:
    """Build a backend whose viewer is fixed, over a graph owned by ALICE."""

    backend = SnowflakeBackend(_OwnedSession())
    backend._viewer = Viewer(user_name=who)
    return backend


def test_a_share_does_not_authorize_cancelling_the_owners_run() -> None:
    """Keep a read share from stopping a billed ingestion.

    A shared run appears in the grantee's history with a Cancel control. If the
    control plane authorises on visibility, one click ends the owner's
    long-running, Cortex-billed run.
    """

    with pytest.raises(PermissionError, match="read access only"):
        _backend_as("BOB").cancel("alices-run")


def test_a_share_does_not_authorize_deleting_the_owners_run() -> None:
    """Keep a read share from removing the owner's run from history."""

    with pytest.raises(PermissionError, match="read access only"):
        _backend_as("BOB").forget("alices-run")


def test_a_share_does_not_authorize_renaming_the_owners_graph() -> None:
    """Keep a read share from renaming a graph for everyone who can see it."""

    with pytest.raises(PermissionError, match="read access only"):
        _backend_as("BOB").rename_graph("alices-graph", "Renamed")


def test_reading_someone_elses_quiet_run_never_writes_a_verdict_onto_it() -> None:
    """Do not let opening a shared run mark the owner's job failed."""

    backend = _backend_as("BOB")
    backend._fail_lost_run("alices-run", "worker is gone")

    assert not [sql for sql, _ in backend.session.statements if sql.startswith("UPDATE KG_JOB")]


def test_a_stage_folder_belonging_to_another_viewer_cannot_be_listed() -> None:
    """Keep one viewer from enumerating another's uploaded documents.

    The application lists a stage with the app owner's privileges, so without
    this check a viewer could point the stage browser at someone else's upload
    folder and ingest their documents into a graph of their own.
    """

    backend = _backend_as("BOB")

    with pytest.raises(PermissionError, match="other users' uploads"):
        backend.list_source_objects({"stage": "KG_DOCS", "prefix": "app/ALICE/job-1"})


def test_a_stage_path_smuggled_through_the_stage_field_is_still_refused() -> None:
    """Authorise the resolved location, not one form field.

    A stage identifier may carry a path of its own, and it is concatenated with
    the prefix before the LIST runs. Checking only the prefix let a viewer put
    another viewer's folder in the stage box and read it.
    """

    backend = _backend_as("BOB")

    for stage, prefix in [
        ("KG_DOCS/app/ALICE/job-1", ""),
        ("KG_DOCS/app", "ALICE/job-1"),
        ("KG_DOCS", ""),
    ]:
        with pytest.raises(PermissionError, match="other users' uploads"):
            backend.list_source_objects({"stage": stage, "prefix": prefix})


def test_uploading_into_another_viewers_folder_is_refused() -> None:
    """Keep one viewer from seeding a folder another viewer's runs read from."""

    with pytest.raises(PermissionError, match="other users' uploads"):
        _backend_as("BOB").upload_files([], "KG_DOCS", "app/ALICE/job-1")


def test_a_role_share_does_not_admit_a_user_of_the_same_name() -> None:
    """Match a share in the namespace it was granted in.

    Snowflake users and roles share a namespace of names — a service account and
    its functional role are routinely identical — so matching on the name alone
    widens a share beyond the principal the owner named.
    """

    clause, parameters = visible_graph_predicate(
        Viewer(user_name="ANALYSTS", roles=frozenset())
    )

    assert "GRANTEE_TYPE" in clause
    assert "'USER'" in clause
    # No role arm at all, so a ROLE-typed share row cannot match this viewer.
    assert "'ROLE'" not in clause
    assert parameters == ["ANALYSTS", "ANALYSTS"]


class _Result:
    """The subset of a Snowpark result the backend uses."""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def collect(self) -> list[Any]:
        return self.rows

    def limit(self, count: int) -> _Result:
        del count
        return self


def test_a_pooled_connection_is_replaced_before_its_credential_expires() -> None:
    """Reopen a connection on a schedule rather than holding it for a whole run.

    A Snowflake session cannot outlive the credential that opened it, and inside
    SPCS that is an OAuth token with a fixed expiry. Held for the length of a
    long run, the connection dies mid-flight and every document in progress
    fails at once — which is exactly how one 5-hour run lost 21 documents in
    eight seconds, each on its first attempt.
    """

    opened: list[int] = []

    class Connection:
        def cursor(self) -> Any: ...
        def commit(self) -> Any: ...
        def rollback(self) -> Any: ...

        def close(self) -> Any:
            return None

    def factory(**_kwargs: object) -> Any:
        opened.append(1)
        return Connection()

    pool = adapter.ReusableSnowflakeConnections(_connection_config(), factory)
    now = 0.0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter, "monotonic", lambda: now)
        pool.get()
        pool.get()
        assert len(opened) == 1, "a fresh connection must be reused"
        now = adapter._MAX_CONNECTION_AGE_SECONDS + 1
        pool.get()
        assert len(opened) == 2, "a connection older than the limit must be replaced"
        pool.get()

    assert len(opened) == 2, "the replacement must then be reused"


def _connection_config() -> Any:
    """Build the minimal connection contract the pool needs."""

    return SnowflakeConnectionConfig(
        account="a",
        host=None,
        user="u",
        password="p",
        authenticator=None,
        private_key_path=None,
        database="D",
        schema_name="S",
        role=None,
        warehouse=None,
        oauth_token_path=None,
    )


def test_retrying_a_run_requeues_its_failed_documents_and_starts_a_worker() -> None:
    """Requeue and relaunch together, because neither alone finishes the run.

    Documents left FAILED are claimed by nobody, and a job reset without a
    worker sits PENDING forever. Only documents that failed are returned, so a
    retry costs the lost work rather than the whole corpus.
    """

    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            if "CONFIG:snowflake:compute_pool" in statement:
                return _Result([_Row({
                    "POOL": "POOL_A",
                    "SPEC_STAGE": "@DB.SCH.SPECS",
                    "DB": "DB",
                    "SCH": "SCH",
                })])
            if "SELECT GRAPH_ID FROM KG_JOB" in statement:
                return _Result([_Row({"GRAPH_ID": "graph-1"})])
            if "COUNT(*) AS QUEUED" in statement:
                return _Result([_Row({"QUEUED": 4})])
            if statement.startswith("SELECT 1 FROM KG_GRAPH"):
                return _Result([_Row({"1": 1})])
            if "FROM KG_JOB J" in statement or "KG_JOB_FILE F" in statement:
                return _Result([_Row({
                    "ID": "run-1",
                    "GRAPH_ID": "graph-1",
                    "STATUS": "PENDING",
                    "STAGE": "queued",
                    "FILE_STATUS": "QUEUED",
                    "FILES": 4,
                })])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")
    backend.retry("run-1")

    requeue = next(s for s in statements if "UPDATE KG_JOB_FILE SET STATUS = 'QUEUED'" in s)
    assert "STATUS = 'FAILED'" in requeue, "only failed documents may be returned"
    assert any("UPDATE KG_JOB SET STATUS = 'PENDING'" in s for s in statements)
    assert any(s.startswith("EXECUTE JOB SERVICE") for s in statements), "a worker must be started"


class _Row:
    """One Snowpark row."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


def test_a_running_worker_reports_what_it_has_spent_so_far() -> None:
    """Carry running totals in the progress the worker already writes.

    Consumption is otherwise accumulated in memory and published only when the
    run finishes, so the long runs whose spend is worth watching are exactly the
    ones that show nothing until it is too late to act on.
    """


    written: list[dict[str, Any]] = []

    class Manager:
        def update_job_progress(
            self, job_id: str, graph_id: str, worker_id: str, payload: dict[str, Any]
        ) -> None:
            written.append(payload)

        def update_job_file_progress(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    sink = SnowflakeJobFileProgressSink(
        Manager(),  # type: ignore[arg-type]
        "run-1",
        "graph-1",
        "worker-1",
        [],
        lambda: {"calls": 7, "total_tokens": 1234, "billed_usd": 0.5},
    )
    sink.emit(
        ProgressEvent(
            job_id="run-1",
            graph_id="graph-1",
            stage="graph_extraction",
            status="progress",
        )
    )

    assert written, "a stage transition is always written"
    assert written[-1]["consumption"] == {
        "calls": 7,
        "total_tokens": 1234,
        "billed_usd": 0.5,
    }


def test_a_worker_that_cannot_price_its_work_still_reports_progress() -> None:
    """Never let consumption reporting cost a run its progress signal.

    Progress is what tells an operator the run is alive. A failure while
    summarising spend must not take that away.
    """


    written: list[dict[str, Any]] = []

    class Manager:
        def update_job_progress(
            self, job_id: str, graph_id: str, worker_id: str, payload: dict[str, Any]
        ) -> None:
            written.append(payload)

        def update_job_file_progress(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def broken() -> dict[str, object]:
        raise RuntimeError("rate card unavailable")

    sink = SnowflakeJobFileProgressSink(
        Manager(),  # type: ignore[arg-type]
        "run-1",
        "graph-1",
        "worker-1",
        [],
        broken,
    )
    sink.emit(
        ProgressEvent(
            job_id="run-1",
            graph_id="graph-1",
            stage="graph_extraction",
            status="progress",
        )
    )

    assert written[-1]["stage"] == "graph_extraction"
    assert "consumption" not in written[-1]


def test_a_dead_worker_is_noticed_in_the_history_not_only_when_opened() -> None:
    """Check liveness where the reader actually looks.

    A run advances only while its job service is alive. Checking that solely on
    the run's own page left a dead worker listed as running for as long as
    nobody opened it — which is precisely the case where the reader needs to be
    told, because nothing else will tell them.
    """

    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            if "FROM KG_JOB J" in statement:
                return _Result([_Row({
                    "ID": "run-1",
                    "GRAPH_ID": "graph-1",
                    "GRAPH_NAME": "graph-1",
                    "STATUS": "RUNNING",
                    "ERROR": None,
                    "CREATED_AT": "2026-01-01",
                    "UPDATED_AT": "2026-01-01",
                    "OWNER": None,
                    "DOCUMENTS_TOTAL": 5,
                    "DOCUMENTS_COMPLETED": 0,
                    "DOCUMENTS_FAILED": 0,
                    "DOCUMENTS_PENDING": 5,
                    "SECONDS_SINCE_UPDATE": 7_200,
                })])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            SnowflakeBackend, "_lost_worker_error", lambda _self, _run: "The worker stopped."
        )
        runs = backend.list_runs(limit=10)

    assert runs[0].status == "failed"
    assert runs[0].error == "The worker stopped."


def test_a_busy_run_is_not_interrogated_by_the_history() -> None:
    """Spend the extra round trip only on runs that have gone quiet.

    A healthy history must stay one query, or watching a run costs more the more
    runs there are.
    """

    checked: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            if "FROM KG_JOB J" in statement:
                return _Result([_Row({
                    "ID": "run-1",
                    "GRAPH_ID": "graph-1",
                    "GRAPH_NAME": "graph-1",
                    "STATUS": "RUNNING",
                    "ERROR": None,
                    "CREATED_AT": "2026-01-01",
                    "UPDATED_AT": "2026-01-01",
                    "OWNER": None,
                    "DOCUMENTS_TOTAL": 5,
                    "DOCUMENTS_COMPLETED": 0,
                    "DOCUMENTS_FAILED": 0,
                    "DOCUMENTS_PENDING": 5,
                    "SECONDS_SINCE_UPDATE": 3,
                })])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            SnowflakeBackend,
            "_lost_worker_error",
            lambda _self, run: checked.append(run),
        )
        runs = backend.list_runs(limit=10)

    assert runs[0].status == "running"
    assert checked == [], "a run that just reported progress must not be interrogated"


def test_deleting_a_graph_clears_every_table_that_records_it() -> None:
    """Pin the app's table list to the schema that creates them.

    The deployed Streamlit bundle ships without ``kg_processor``, so the app
    cannot read the schema at runtime and the list is written out. A table added
    to the schema later would otherwise keep its rows for a graph the operator
    believes they deleted — data surviving a deletion, unreachable and unnoticed.
    """

    schema_tables = {
        table for table, columns in snowflake_schema_columns().items() if "GRAPH_ID" in columns
    }

    assert set(GRAPH_DATA_TABLES) == schema_tables


def test_a_graph_deletion_is_refused_for_someone_elses_graph() -> None:
    """Deleting is the owner's act, like every other change to a graph."""

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            # The ownership test finds no graph this viewer may administer, and
            # the graph is nonetheless known — so it belongs to somebody else.
            if "FROM KG_GRAPH G" in statement:
                return _Result([])
            if "FROM KG_GRAPH WHERE GRAPH_ID" in statement:
                return _Result([_Row({"1": 1})])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="BOB")

    with pytest.raises(PermissionError, match="another user"):
        backend.delete_graph("alices-graph")


def test_a_graph_deletion_removes_rows_from_every_graph_table() -> None:
    """Clear all of them, so nothing survives pointing at a deleted graph."""

    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            if "SELECT 1 FROM KG_GRAPH G" in statement:
                return _Result([_Row({"1": 1})])
            if "UNION ALL" in statement:
                return _Result([_Row({"TABLE_NAME": "KG_NODE", "ROW_COUNT": 97})])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")

    removed = backend.delete_graph("graph-1")

    cleared = {
        statement.split("DELETE FROM ")[1].split(" ")[0]
        for statement in statements
        if statement.startswith("DELETE FROM ")
    }
    assert cleared == set(GRAPH_DATA_TABLES)
    assert removed == {"KG_NODE": 97}


def test_deleting_a_graph_also_clears_its_staged_documents_and_service() -> None:
    """Remove what a graph left outside the tables, not only its rows.

    Each run uploads the operator's documents into a stage folder, stages a
    service specification, and leaves a job service Snowflake keeps after it
    stops. Clearing only the tables leaves the source documents themselves in
    the account — precisely what somebody deleting a graph means to remove.
    """

    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            if "SELECT 1 FROM KG_GRAPH G" in statement:
                return _Result([_Row({"1": 1})])
            if "stage_prefix" in statement:
                return _Result([_Row({
                    "ID": "run-1",
                    "UPLOAD_PREFIX": "app/ALICE/run-1",
                    "DOCUMENT_STAGE": "@DB.SCH.KG_DOCS",
                    "SPEC_STAGE": "@DB.SCH.KG_SERVICE_SPECS",
                    "DB": "DB",
                    "SCH": "SCH",
                })])
            if "UNION ALL" in statement:
                return _Result([])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")

    backend.delete_graph("graph-1")

    assert any(s.startswith("REMOVE @DB.SCH.KG_DOCS/app/ALICE/run-1") for s in statements)
    assert any("flakegraph-app-" in s and s.startswith("REMOVE") for s in statements)
    assert any(s.startswith("DROP SERVICE IF EXISTS DB.SCH.") for s in statements)


def test_a_curated_source_stage_is_never_emptied_by_a_deletion() -> None:
    """Delete what the application uploaded, never a corpus an operator keeps.

    A run over a folder somebody else curates was built from those documents; it
    does not own them, and removing them would destroy a corpus every other run
    still reads.
    """

    statements: list[str] = []

    class Session:
        def sql(self, statement: str, params: Any = None) -> Any:
            statements.append(statement)
            if "SELECT 1 FROM KG_GRAPH G" in statement:
                return _Result([_Row({"1": 1})])
            if "stage_prefix" in statement:
                return _Result([_Row({
                    "ID": "run-1",
                    "UPLOAD_PREFIX": "corporate/reference-corpus",
                    "DOCUMENT_STAGE": "@DB.SCH.KG_DOCS",
                    "SPEC_STAGE": "@DB.SCH.KG_SERVICE_SPECS",
                    "DB": "DB",
                    "SCH": "SCH",
                })])
            return _Result([])

    backend = SnowflakeBackend(Session())
    backend._viewer = Viewer(user_name="ALICE")

    backend.delete_graph("graph-1")

    assert not any("corporate/reference-corpus" in s for s in statements)
