"""Snowflake-native control plane built on the active Snowpark session."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from flakegraph_app.configuration import build_run_config, redacted_config
from flakegraph_app.graph_store import (
    EMBEDDING_TABLES,
    GRAPH_DATA_TABLES,
    GRAPH_METRICS_QUERY,
    GRAPH_REVIEW_ROW_LIMITS,
    graph_count_query,
    normalize_row,
    review_projection,
)
from flakegraph_app.models import (
    ClusterSnapshot,
    GraphDataset,
    IngestionRequest,
    NodeWorkAssignment,
    RunSnapshot,
    SourceObject,
    StageProgress,
    StorageKind,
)
from flakegraph_app.run_catalog import validate_graph_name
from flakegraph_app.sources import SUPPORTED_SUFFIXES
from flakegraph_app.spcs import build_spcs_launch, service_name_for_job
from flakegraph_app.viewer import (
    UPLOAD_NAMESPACE,
    Viewer,
    administrable_graph_predicate,
    current_viewer,
    may_read_stage_path,
    visible_graph_predicate,
)

_SUBMISSION_BATCH_SIZE = 10_000
_SAFE_STAGE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")
_SAFE_STAGE_PREFIX_RE = re.compile(r"^[A-Za-z0-9_./=$+:%-]*$")
_SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
# Snowflake keeps a job service after it stops. Only these states mean the run is
# over and its name is free; anything else is still holding the compute pool.
_TERMINAL_SERVICE_STATUSES = frozenset(
    {"DONE", "FAILED", "CANCELLED", "SUSPENDED", "DELETING", "INTERNAL_ERROR"}
)
# Deliberately narrower than the set above. Reusing a name and declaring a run
# dead are different questions, and the second one is destructive: a wrong answer
# marks a working run failed and stores that. SUSPENDED and DELETING are states a
# service can pass through while its work is still expected to continue, so only
# states a service can never do further work from count as a dead worker.
_DEAD_SERVICE_STATUSES = frozenset({"DONE", "FAILED", "CANCELLED", "INTERNAL_ERROR"})
# Job states that still expect a worker to do something. A run in one of these
# with work left in its queue is only correct while its worker is alive.
_UNSETTLED_JOB_STATUSES = frozenset({"pending", "running"})
# How long a run may go without its job row changing before the app stops
# assuming a worker is behind it. Comfortably longer than the worker's own
# progress interval, so a busy run is never interrogated.
_QUIET_RUN_SECONDS = 120


def _live_consumption(progress: object) -> Mapping[str, object]:
    """Read the spend a running worker last reported, if it reported one."""

    payload = _variant_mapping(progress)
    totals = payload.get("consumption") if payload else None
    return totals if isinstance(totals, Mapping) else {}


def _share_id(graph_id: str, grantee_type: str, grantee: str) -> str:
    """Derive one stable identity per (graph, grantee) so sharing is idempotent."""

    return hashlib.sha256(f"{graph_id.strip()}|{grantee_type}|{grantee}".encode()).hexdigest()


def _quoted(identifier: str) -> str:
    """Escape a Snowflake identifier for use inside double quotes."""

    return identifier.strip().replace('"', '""')


def _vector_width(data_type: object) -> int | None:
    """Read the fixed width Snowflake reports for a VECTOR column.

    ``SHOW COLUMNS`` describes a column's type as JSON, and only a vector carries
    a dimension. ``INFORMATION_SCHEMA`` is not usable here: its ``DATA_TYPE`` says
    only ``VECTOR`` and omits the width that has to be compared.
    """

    if isinstance(data_type, Mapping):
        declared = data_type.get("dimension")
        return declared if isinstance(declared, int) else None
    if not isinstance(data_type, str) or "dimension" not in data_type:
        return None
    try:
        parsed = json.loads(data_type)
    except ValueError:
        return None
    declared = parsed.get("dimension") if isinstance(parsed, Mapping) else None
    return declared if isinstance(declared, int) else None


class SnowflakeBackend:
    """List stages, submit SPCS queue work, and query graph tables in Snowflake."""

    def __init__(self, session: Any) -> None:
        """Retain an active Snowpark session supplied by the host runtime."""

        self.session = session
        self._viewer: Viewer | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        """Describe Snowflake-native controls available to the UI."""

        return frozenset(
            {
                "upload",
                "snowflake_stage",
                "progress",
                "graph",
                "spcs",
                "cancel",
                "rename",
                "forget",
                "share",
                "delete_graph",
                "retry",
            }
        )

    def list_source_objects(
        self,
        source: Mapping[str, str],
        *,
        limit: int = 1_000,
    ) -> Sequence[SourceObject]:
        """List supported files from one internal stage and optional prefix."""

        self._require_readable_stage(source.get("stage", ""), source.get("prefix", ""))
        stage = _stage_location(source.get("stage", ""), source.get("prefix"))
        rows = self.session.sql(f"LIST {stage}").limit(limit).collect()
        return [item for row in rows if (item := _source_object(row)) is not None]

    def current_context(self) -> Mapping[str, str]:
        """Return the session's own account, user, database, schema and role.

        Deployed in Snowflake the app already knows all of this. Asking an
        operator to retype it invites typos in fields that cannot be wrong.
        """

        try:
            row = self.session.sql(
                "SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_DATABASE(), "
                "CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()"
            ).collect()[0]
        except Exception:
            return {}
        values = list(row)
        keys = ("account", "user", "database", "schema", "warehouse", "role")
        return {key: str(value) for key, value in zip(keys, values, strict=False) if value}

    def _show_names(self, statement: str) -> Sequence[str]:
        """Return the ``name`` column of a SHOW statement, or nothing on failure.

        Discovery is a convenience: a role without permission to list warehouses
        must still be able to type one rather than be blocked by an empty list.
        """

        try:
            rows = self.session.sql(statement).collect()
        except Exception:
            return []
        names = []
        for row in rows:
            item = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            name = str(item.get("name") or "")
            if name:
                names.append(name)
        return sorted(set(names))

    def list_databases(self) -> Sequence[str]:
        """Return databases visible to the session."""

        return self._show_names("SHOW DATABASES")

    def list_schemas(self, database: str = "") -> Sequence[str]:
        """Return schemas, scoped to a database when one is given."""

        target = database.strip()
        if target and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", target):
            return []
        return self._show_names(
            f"SHOW SCHEMAS IN DATABASE {target}" if target else "SHOW SCHEMAS"
        )

    def list_warehouses(self) -> Sequence[str]:
        """Return warehouses visible to the session."""

        return self._show_names("SHOW WAREHOUSES")

    def list_roles(self) -> Sequence[str]:
        """Return roles granted to the current user."""

        return self._show_names("SHOW ROLES")

    def list_compute_pools(self) -> Sequence[str]:
        """Return compute pools this role may use."""

        return self._show_names("SHOW COMPUTE POOLS")

    def compute_pool_capacity(self, pool: str) -> Mapping[str, float]:
        """Return the vCPU and memory one node of a pool provides.

        A job service gets a node to itself, and SPCS bills the node whether the
        container asks for it or not. Sizing the container from the pool's own
        instance family stops a run being confined to a fraction of hardware that
        has already been paid for: this account's GEN_X64_G2_4 offers 3 vCPU and
        13 GiB, against a container that requested half a core and three.

        An empty result means discovery failed, and the caller keeps its own
        defaults rather than guessing at a node it could not read.
        """

        name = pool.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name):
            return {}
        try:
            pools = self.session.sql(f"SHOW COMPUTE POOLS LIKE '{name}'").collect()
            if not pools:
                return {}
            family = str(normalize_row(pools[0].as_dict()).get("instance_family") or "")
            families = self.session.sql("SHOW COMPUTE POOL INSTANCE FAMILIES").collect()
        except Exception:
            return {}
        for row in families:
            item = normalize_row(row.as_dict())
            if str(item.get("name") or "").upper() != family.upper():
                continue
            try:
                return {
                    "vcpu": float(str(item.get("vcpu") or 0)),
                    "memory_gib": float(str(item.get("memory_gib") or 0)),
                }
            except ValueError:
                return {}
        return {}

    def list_image_repositories(self) -> Sequence[str]:
        """Return image repositories as the fully qualified names SPCS expects."""

        try:
            rows = self.session.sql("SHOW IMAGE REPOSITORIES").collect()
        except Exception:
            return []
        names = []
        for row in rows:
            item = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            name = str(item.get("name") or "")
            database = str(item.get("database_name") or "")
            schema = str(item.get("schema_name") or "")
            if name and database and schema:
                names.append(f"{database}.{schema}.{name}")
            elif name:
                names.append(name)
        return sorted(set(names))

    def list_images(self, repository: str) -> Sequence[str]:
        """Return image:tag pairs held in one repository, most recent first.

        Choosing the image from what has been pushed avoids the class of failure
        that a typed tag produces: a service that starts and immediately dies
        pulling something that was never published.

        Order is the other half of that. Sorted by name, the first option is the
        oldest tag ever pushed, and the picker offers it as the default — so an
        operator who never touches the field runs the very first image built for
        this account. That happened: a run submitted from the app launched
        ``flakegraph:0.1.0`` from the previous morning, which failed on startup
        and left the run with nothing to explain it. Newest first makes the
        default the image that was most recently published.
        """

        target = repository.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}", target):
            return []
        try:
            rows = self.session.sql(f"SHOW IMAGES IN IMAGE REPOSITORY {target}").collect()
        except Exception:
            return []
        images: list[tuple[str, str]] = []
        for row in rows:
            item = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            name = str(item.get("image_name") or "")
            created = str(item.get("created_on") or "")
            tags = str(item.get("tags") or "")
            for tag in [part.strip() for part in tags.split(",") if part.strip()]:
                images.append((created, f"{name}:{tag}"))
        ordered: list[str] = []
        seen: set[str] = set()
        # Secondary sort on the reference keeps the order total when a repository
        # reports no push time, rather than leaving it to dictionary chance.
        for _, image in sorted(images, key=lambda pair: (pair[0], pair[1]), reverse=True):
            if image not in seen:
                seen.add(image)
                ordered.append(image)
        return ordered

    def list_stages(self) -> Sequence[str]:
        """Return the internal stages visible in the current schema.

        Offering the stages that exist prevents the failure this replaced: an
        upload to a stage that was never created fails inside Snowflake's storage
        client with a bare SystemError that names nothing.
        """

        rows = self.session.sql("SHOW STAGES").collect()
        names = []
        for row in rows:
            item = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            name = str(item.get("name") or "")
            if name:
                names.append(name)
        return sorted(names)

    def upload_files(self, paths: Sequence[Path], stage: str, prefix: str = "uploads") -> int:
        """Upload app-received files to an internal stage through Snowpark.

        The bytes are streamed rather than handed over as a local path. Snowpark's
        ``put`` delegates to the connector's file transfer, which cannot read the
        app's filesystem inside Streamlit in Snowflake and fails there with a bare
        ``SystemError`` from ``open_stream``. ``put_stream`` uploads the content
        directly and works in both places.
        """

        # Uploading into another viewer's folder would place this viewer's
        # documents where only that viewer can reach them, and lets one viewer
        # seed a folder another viewer's runs read from.
        self._require_readable_stage(stage, prefix)
        # Checked before the first byte moves: Snowflake reports an upload to a
        # missing stage as a SystemError out of its storage client, which names
        # neither the stage nor the problem.
        available = set(self.list_stages())
        target = stage.strip().strip("@").split(".")[-1].upper()
        if available and target not in available:
            raise RuntimeError(
                f"Internal stage {target} does not exist in this schema. "
                f"Available stages: {', '.join(sorted(available))}."
            )
        location = _stage_location(stage, prefix)
        uploaded = 0
        for path in paths:
            with path.open("rb") as handle:
                results = self.session.file.put_stream(
                    handle,
                    f"{location}/{path.name}",
                    auto_compress=False,
                    overwrite=True,
                )
            _validate_upload_results(_as_results(results), path.name)
            uploaded += 1
        return uploaded

    def preflight(self, request: IngestionRequest) -> Mapping[str, object]:
        """Check required queue and graph tables plus stage visibility cheaply."""

        required = {"KG_GRAPH", "KG_JOB", "KG_JOB_FILE", "KG_NODE", "KG_EDGE"}
        rows = self.session.sql("SHOW TABLES").collect()
        visible = {
            str((row.as_dict() if hasattr(row, "as_dict") else dict(row)).get("name", "")).upper()
            for row in rows
        }
        errors = [
            f"Required Snowflake table is not visible: {name}"
            for name in sorted(required - visible)
        ]
        if request.source_kind.value == "Snowflake stage":
            try:
                self.list_source_objects(
                    {
                        "stage": str(request.source.get("stage", "")),
                        "prefix": str(request.source.get("prefix", "")),
                    },
                    limit=1,
                )
            except Exception as exc:
                errors.append(f"Stage cannot be listed: {exc}")
        mismatched: dict[str, int] = {}
        try:
            mismatched = {
                table: width
                for table, width in self._embedding_widths().items()
                if width != request.embedding.dimension
            }
        except Exception as exc:
            errors.append(f"Embedding dimension cannot be checked: {exc}")
        if mismatched:
            errors.append(
                f"Embedding dimension {request.embedding.dimension} does not fit the graph "
                f"tables, which store {sorted(set(mismatched.values()))}: "
                f"{', '.join(sorted(mismatched))}. Choose a model of the stored width, or "
                "recreate the tables at this width."
            )
        try:
            launch = build_spcs_launch(build_run_config(request))
            self.session.sql(f"LIST {launch.spec_stage}").limit(1).collect()
        except Exception as exc:
            errors.append(f"SPCS launch configuration is not ready: {exc}")
        return {
            "ok": not errors,
            "runtime": "snowflake",
            "checks": [
                {"name": "canonical_tables", "ok": not (required - visible)},
                {"name": "embedding_dimension", "ok": not mismatched},
                {"name": "active_session", "ok": True},
            ],
            "errors": errors,
        }

    def graph_embedding_width(self) -> int | None:
        """Return the width the graph tables store, when they agree on one.

        The embedding controls default to a model of this width, so an operator
        is offered a run that can actually be written rather than one that
        preflight will refuse.
        """

        try:
            widths = set(self._embedding_widths().values())
        except Exception:
            return None
        return widths.pop() if len(widths) == 1 else None

    def _embedding_widths(self) -> dict[str, int]:
        """Return the vector width each graph table already stores.

        A VECTOR column fixes its width when the table is created and the writer
        is the last stage of a run, so a mismatch is otherwise discovered only
        after every document has been parsed, extracted and summarised, and the
        whole Cortex spend of the run is rejected at the write.

        A column whose width cannot be read is reported as unknown rather than as
        agreement: a guard that exists to stop an expensive run must not pass a
        run it could not actually check.
        """

        widths: dict[str, int] = {}
        for row in self.session.sql("SHOW COLUMNS LIKE 'EMBEDDING'").collect():
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            table = str(record.get("table_name", "")).upper()
            # A schema is shared. Another table's embedding column says nothing
            # about where this graph is written, and reading it would let an
            # unrelated design decide whether a FlakeGraph run may start.
            if table not in EMBEDDING_TABLES:
                continue
            declared = _vector_width(record.get("data_type"))
            if declared is None:
                raise RuntimeError(
                    f"{table}.EMBEDDING does not report a vector width: {record.get('data_type')}"
                )
            widths[table] = declared
        return widths

    def submit(self, request: IngestionRequest) -> RunSnapshot:
        """Create the queue, stage its immutable spec, and start an SPCS job."""

        # Authorisation precedes every other step, so a refusal costs nothing and
        # cannot be reached through a failure in configuration building.
        #
        # Writing into a graph is not something a share confers: the worker
        # merges into that graph's rows and, holding the whole file queue,
        # replaces them outright.
        self._require_administration(request.graph_id)
        # Every write below is keyed on the run id, which is a free-text field.
        # An existing run belonging to someone else must not be adopted: doing so
        # drops their job service, inserts documents into their queue, and writes
        # a failure of the submitter's choosing onto their live run.
        existing_graph = self._run_graph_id(request.job_id)
        if existing_graph and existing_graph != request.graph_id.strip():
            raise PermissionError(
                "That run ID already belongs to another graph. Choose a different run ID."
            )
        if existing_graph:
            self._require_administration(existing_graph)
        if request.source_kind.value == "Snowflake stage":
            self._require_readable_stage(
                str(request.source.get("stage", "")),
                str(request.source.get("prefix", "")),
            )
        source = {
            "stage": str(request.source.get("stage", "")),
            "prefix": str(request.source.get("prefix", "")),
        }
        config = build_run_config(request)
        launch = build_spcs_launch(config)
        config_json = json.dumps(redacted_config(launch.runtime_config), sort_keys=True)
        self._claim_graph(request.graph_id)
        self.session.sql(
            "MERGE INTO KG_JOB target USING ("
            "SELECT ? AS ID, ? AS GRAPH_ID, PARSE_JSON(?) AS CONFIG"
            ") source ON target.ID = source.ID "
            "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS, CONFIG) "
            "VALUES (source.ID, source.GRAPH_ID, 'PENDING', source.CONFIG)",
            params=[request.job_id, request.graph_id, config_json],
        ).collect()
        try:
            temporary_name = (
                f"KG_JOB_FILE_SUBMISSION_{hashlib.sha256(request.job_id.encode()).hexdigest()[:12]}"
            )
            submitted = 0
            batch: list[tuple[str, ...]] = []
            for item in self._iter_submission_objects(source):
                batch.append(_submission_row(request, item))
                if len(batch) >= _SUBMISSION_BATCH_SIZE:
                    self._merge_submission_batch(temporary_name, batch)
                    submitted += len(batch)
                    batch = []
            if batch:
                self._merge_submission_batch(temporary_name, batch)
                submitted += len(batch)
            self._drop_submission_table(temporary_name)
            if submitted == 0:
                raise ValueError("The selected Snowflake stage prefix contains no supported files")
            upload_result = self.session.file.put_stream(
                BytesIO(launch.spec_yaml.encode("utf-8")),
                f"{launch.spec_stage}/{launch.spec_file}",
                auto_compress=False,
                overwrite=True,
            )
            if upload_result is not None:
                _validate_upload_results(_as_results(upload_result), launch.spec_file)
            self._release_finished_job_service(launch.service_identifier)
            self.session.sql(launch.execute_sql).collect()
        except Exception as error:
            # The cause is recorded, not replaced. A fixed message forces whoever
            # reads the failed run to reproduce it before they can even guess at
            # what went wrong.
            self.session.sql(
                "UPDATE KG_JOB SET STATUS = 'FAILED', "
                "ERROR = OBJECT_CONSTRUCT('message', ?, 'type', ?), "
                "UPDATED_AT = CURRENT_TIMESTAMP() WHERE ID = ?",
                params=[
                    f"SPCS job submission failed: {error}"[:2000],
                    type(error).__name__,
                    request.job_id,
                ],
            ).collect()
            raise
        return self.status(request.job_id)

    def _iter_submission_objects(self, source: Mapping[str, str]) -> Iterator[SourceObject]:
        """Stream LIST output so submission memory stays independent of corpus size."""

        stage = _stage_location(source.get("stage", ""), source.get("prefix"))
        query = self.session.sql(f"LIST {stage}")
        iterator = getattr(query, "to_local_iterator", None)
        rows = iterator() if callable(iterator) else query.collect()
        for row in rows:
            item = _source_object(row)
            if item is not None:
                yield item

    def _release_finished_job_service(self, identifier: str) -> None:
        """Free a finished job service's name so the same graph can run again.

        The name is derived from the job id on purpose: the worker's lease owner
        and the service holding its logs both stay findable from the run alone.
        Snowflake keeps a completed job service until it is dropped, so the
        second submission of a graph — a retry, or a rerun after fixing the
        input — failed with a bare ``Object ... already exists`` that named an
        internal service and explained nothing.

        A service that is still working is never dropped. That would destroy the
        run the caller is trying to repeat, so it is reported instead.
        """

        database, _, remainder = identifier.partition(".")
        schema, _, name = remainder.partition(".")
        if not _SERVICE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Refusing to inspect an unsafe service name: {name}")
        rows = self.session.sql(
            f"SHOW SERVICES LIKE '{name}' IN SCHEMA {database}.{schema}"
        ).collect()
        if not rows:
            return
        status = str(normalize_row(rows[0].as_dict()).get("status", "")).upper()
        if status and status not in _TERMINAL_SERVICE_STATUSES:
            raise RuntimeError(
                f"A job for this graph is already {status.lower()} as {identifier}. "
                "Wait for it to finish or cancel it before submitting again."
            )
        self.session.sql(f"DROP SERVICE IF EXISTS {identifier}").collect()

    def _drop_submission_table(self, name: str) -> None:
        """Remove the transient staging table once its rows are merged.

        A transient table outlives the statement that made it, so leaving it
        behind would accumulate one scratch table per submission.
        """

        try:
            self.session.sql(f"DROP TABLE IF EXISTS {name}").collect()
        except Exception:
            # The rows are already merged; a leftover scratch table is untidy
            # rather than harmful, and must not fail a successful submission.
            return

    def _merge_submission_batch(
        self,
        temporary_name: str,
        rows: Sequence[tuple[str, ...]],
    ) -> None:
        """Merge one bounded metadata batch into the durable Snowflake queue."""

        frame = self.session.create_dataframe(
            rows,
            schema=["ID", "JOB_ID", "GRAPH_ID", "FILE_ID", "SOURCE_URI", "BLOB_PATH", "CHECKSUM"],
        )
        # Transient rather than temporary: Streamlit in Snowflake executes as a
        # stored procedure, where a temporary table is rejected outright with
        # "Unsupported statement type 'temporary TABLE'". A transient table is
        # accepted in both contexts and is dropped once submission completes.
        frame.write.mode("overwrite").save_as_table(temporary_name, table_type="transient")
        self.session.sql(
            f"MERGE INTO KG_JOB_FILE target USING {temporary_name} source "
            "ON target.ID = source.ID "
            "WHEN MATCHED THEN UPDATE SET SOURCE_URI = source.SOURCE_URI, "
            "BLOB_PATH = source.BLOB_PATH, CHECKSUM = source.CHECKSUM, "
            "UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT "
            "(ID, JOB_ID, GRAPH_ID, FILE_ID, SOURCE_URI, BLOB_PATH, CHECKSUM, STATUS) "
            "VALUES (source.ID, source.JOB_ID, source.GRAPH_ID, source.FILE_ID, "
            "source.SOURCE_URI, source.BLOB_PATH, source.CHECKSUM, 'QUEUED')"
        ).collect()

    def list_runs(self, *, limit: int = 100) -> Sequence[RunSnapshot]:
        """Return recent Snowflake jobs and file-level completion counts in one query."""

        bounded_limit = max(1, min(limit, 500))
        # Owner's rights would otherwise return every graph in the account to
        # every viewer, so the predicate is the only thing standing between one
        # person's corpus and another's.
        visibility, parameters = visible_graph_predicate(self.viewer(), alias="G")
        rows = self.session.sql(
            "SELECT J.ID, J.GRAPH_ID, COALESCE(G.DISPLAY_NAME, J.GRAPH_ID) AS GRAPH_NAME, "
            "J.STATUS, J.ERROR, J.CREATED_AT, J.UPDATED_AT, G.OWNER, "
            "COUNT(F.ID) AS DOCUMENTS_TOTAL, "
            "COUNT_IF(UPPER(F.STATUS) IN ('DONE', 'SUCCEEDED')) AS DOCUMENTS_COMPLETED, "
            "COUNT_IF(UPPER(F.STATUS) = 'FAILED') AS DOCUMENTS_FAILED, "
            "COUNT_IF(UPPER(F.STATUS) IN ('QUEUED', 'CLAIMED')) AS DOCUMENTS_PENDING, "
            "TIMESTAMPDIFF('second', COALESCE(J.UPDATED_AT, J.CREATED_AT), "
            "CURRENT_TIMESTAMP()) AS SECONDS_SINCE_UPDATE "
            "FROM KG_JOB J LEFT JOIN KG_JOB_FILE F ON F.JOB_ID = J.ID "
            "LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID "
            f"WHERE {visibility} "
            "GROUP BY J.ID, J.GRAPH_ID, G.DISPLAY_NAME, G.OWNER, J.STATUS, J.ERROR, "
            "J.CREATED_AT, J.UPDATED_AT "
            f"ORDER BY COALESCE(J.UPDATED_AT, J.CREATED_AT) DESC LIMIT {bounded_limit}",
            params=parameters,
        ).collect()
        snapshots = []
        for row in rows:
            item = normalize_row(row.as_dict())
            status = str(item["status"]).lower()
            error = _job_error_text(item.get("error"))
            # A run only advances because its job service is alive. Checking that
            # solely when the run is opened left a dead worker listed as running
            # for as long as nobody looked, which is exactly when it matters. Only
            # a run that has gone quiet with work outstanding is worth the extra
            # round trip, so a healthy history still costs one query.
            if (
                status in _UNSETTLED_JOB_STATUSES
                and int(item.get("documents_pending") or 0)
                and int(item.get("seconds_since_update") or 0) >= _QUIET_RUN_SECONDS
            ):
                lost = self._lost_worker_error(str(item["id"]))
                if lost is not None:
                    self._fail_lost_run(str(item["id"]), lost)
                    status, error = "failed", lost
            snapshots.append(
                RunSnapshot(
                    run_id=str(item["id"]),
                    graph_id=str(item["graph_id"]),
                    graph_name=str(item.get("graph_name") or item["graph_id"]),
                    status=status,
                    started_at=str(item.get("created_at")) if item.get("created_at") else None,
                    updated_at=str(item.get("updated_at")) if item.get("updated_at") else None,
                    documents_total=int(item.get("documents_total") or 0),
                    documents_completed=int(item.get("documents_completed") or 0),
                    documents_failed=int(item.get("documents_failed") or 0),
                    storage_kind=StorageKind.SNOWFLAKE,
                    storage_location="Snowflake",
                    error=error,
                )
            )
        return snapshots

    def status(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Read job state and grouped progress in one bounded warehouse query."""

        _ = config_path
        rows = self.session.sql(
            "SELECT J.ID, J.GRAPH_ID, COALESCE(G.DISPLAY_NAME, J.GRAPH_ID) AS GRAPH_NAME, "
            "J.STATUS, J.ERROR, J.CREATED_AT, J.UPDATED_AT, J.PROGRESS, "
            "TIMESTAMPDIFF(second, J.UPDATED_AT, CURRENT_TIMESTAMP()) AS SECONDS_SINCE_UPDATE, "
            "COALESCE(F.STAGE, 'queued') AS STAGE, F.STATUS AS FILE_STATUS, "
            "COUNT(F.ID) AS FILE_COUNT "
            "FROM KG_JOB J LEFT JOIN KG_JOB_FILE F ON F.JOB_ID = J.ID "
            "LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID "
            "WHERE J.ID = ? "
            "GROUP BY J.ID, J.GRAPH_ID, G.DISPLAY_NAME, J.STATUS, J.ERROR, "
            "J.CREATED_AT, J.UPDATED_AT, J.PROGRESS, "
            "COALESCE(F.STAGE, 'queued'), F.STATUS "
            "ORDER BY STAGE, FILE_STATUS",
            params=[run_id],
        ).collect()
        if not rows:
            raise ValueError(f"Unknown Snowflake job: {run_id}")
        job = normalize_row(rows[0].as_dict())
        grouped: dict[str, dict[str, int]] = {}
        for row in rows:
            item = normalize_row(row.as_dict())
            file_status = item.get("file_status")
            file_count = int(item.get("file_count") or 0)
            if file_status and file_count:
                grouped.setdefault(str(item["stage"]), {})[str(file_status).lower()] = file_count
        stages = []
        for stage, statuses in grouped.items():
            total = sum(statuses.values())
            done = statuses.get("done", 0)
            state = (
                "failed"
                if statuses.get("failed")
                else (
                    "running"
                    if statuses.get("claimed")
                    else ("completed" if done == total else "queued")
                )
            )
            stages.append(StageProgress(stage, state, done, total))
        stages = _with_live_stage_progress(stages, job.get("progress"))
        totals = {
            status: sum(values.get(status, 0) for values in grouped.values())
            for status in ("queued", "claimed", "done", "failed")
        }
        # Every file registered for the run, in whatever state it reached.
        # Cancellation rewrites queued and claimed rows to CANCELLED, and the run
        # history counts file rows outright, so a total assembled from only the
        # states work passes through would report no documents beside a stage
        # breakdown and a sidebar entry that both still list them.
        documents_total = sum(sum(values.values()) for values in grouped.values())
        job_status = str(job["status"]).lower()
        job_error = _job_error_text(job.get("error"))
        # Only a run that has gone quiet is worth a second round trip. A worker
        # writes progress every few seconds, so a healthy run keeps its job row
        # fresh and status stays the single aggregate query it was.
        quiet_for = int(job.get("seconds_since_update") or 0)
        if (
            job_status in _UNSETTLED_JOB_STATUSES
            and totals["queued"] + totals["claimed"]
            and quiet_for >= _QUIET_RUN_SECONDS
        ):
            lost = self._lost_worker_error(run_id)
            if lost is not None:
                self._fail_lost_run(run_id, lost)
                job_status, job_error = "failed", lost
        return RunSnapshot(
            run_id=run_id,
            graph_id=str(job["graph_id"]),
            graph_name=str(job.get("graph_name") or job["graph_id"]),
            status=job_status,
            started_at=str(job.get("created_at")) if job.get("created_at") else None,
            updated_at=str(job.get("updated_at")) if job.get("updated_at") else None,
            stages=stages,
            documents_total=documents_total,
            documents_completed=totals["done"],
            documents_failed=totals["failed"],
            storage_kind=StorageKind.SNOWFLAKE,
            storage_location="Snowflake",
            error=job_error,
            raw={"consumption": _live_consumption(job.get("progress"))},
        )

    def _lost_worker_error(self, run_id: str) -> str | None:
        """Describe the run's worker if it has stopped without draining the queue.

        A run only advances because its SPCS job service is alive. When that
        service dies — an image that cannot start, an exhausted container, a
        crash — nothing writes to the queue and nothing writes an error, so the
        app keeps reporting exactly what it last saw: pending, zero of one, for
        as long as anyone leaves it open. The service's own terminal state is the
        missing signal, and it is addressable from the run id alone.

        Returns None whenever the worker might still be doing the work, so a live
        run is never reported as lost.
        """

        name = service_name_for_job(run_id)
        try:
            rows = self.session.sql(f"SHOW SERVICES LIKE '{name}'").collect()
        except Exception:
            # Discovery needs privileges a viewer may not hold. Not being able to
            # look is not evidence that the worker died.
            return None
        if not rows:
            return None
        service = normalize_row(rows[0].as_dict())
        state = str(service.get("status") or "").upper()
        if state not in _DEAD_SERVICE_STATUSES:
            return None
        return (
            f"The worker for this run stopped without processing its documents "
            f"(job service {name} reported {state}). Its container logs are held "
            f"by that service. Re-submitting the graph starts a new worker; if it "
            f"stops again, check the image the run selected."
        )

    def _fail_lost_run(self, run_id: str, message: str) -> None:
        """Record a lost worker on the job so every view agrees the run is over.

        Written rather than only displayed: the sidebar reads stored status, and a
        run that reads failed on its own page while still listing as pending
        beside it is worse than either answer alone. Queued files are deliberately
        left queued so a re-submission can still claim them.
        """

        if not self.can_administer_graph(self._run_graph_id(run_id)):
            # Reading someone else's run must not write a verdict onto it.
            return
        with suppress(Exception):
            self.session.sql(
                "UPDATE KG_JOB SET STATUS = 'FAILED', "
                "ERROR = OBJECT_CONSTRUCT('message', ?, 'type', 'worker_stopped'), "
                "UPDATED_AT = CURRENT_TIMESTAMP() "
                "WHERE ID = ? AND STATUS NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')",
                params=[message, run_id],
            ).collect()

    def cancel(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Prevent queued files from being claimed and mark their parent cancelled."""

        _ = config_path
        self._require_administration(self._run_graph_id(run_id))
        self.session.sql(
            "UPDATE KG_JOB SET STATUS = 'CANCELLED', LEASE_OWNER = NULL, "
            "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHERE ID = ? AND STATUS NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')",
            params=[run_id],
        ).collect()
        self.session.sql(
            "UPDATE KG_JOB_FILE SET STATUS = 'CANCELLED', WORKER_ID = NULL, "
            "LEASE_UNTIL = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHERE JOB_ID = ? AND STATUS IN ('QUEUED', 'CLAIMED')",
            params=[run_id],
        ).collect()
        return self.status(run_id)

    def retry(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Requeue a failed run's documents and start a worker to finish them.

        The queue is the durable state: documents that failed are returned to it
        and the job is made claimable again. The original service specification
        is still on the spec stage and its name derives from the run, so the same
        worker can be started again without rebuilding the request — which the
        application could not do anyway, since what it stored of the run's
        configuration is redacted.

        Documents that already succeeded are left alone, so a retry costs only
        the work that was lost.
        """

        del config_path
        self._require_administration(self._run_graph_id(run_id))
        launch = self._stored_launch(run_id)
        requeued = self._requeue_failed_documents(run_id)
        if requeued:
            self._release_finished_job_service(launch["service"])
            self.session.sql(
                "EXECUTE JOB SERVICE\n"
                f"  IN COMPUTE POOL {launch['pool']}\n"
                f"  NAME = {launch['service']}\n"
                "  ASYNC = TRUE\n"
                f"  FROM {launch['stage']}\n"
                f"  SPEC = '{launch['spec']}';"
            ).collect()
        return self.status(run_id)

    def _stored_launch(self, run_id: str) -> dict[str, str]:
        """Rebuild the job service identity recorded when the run was submitted."""

        rows = self.session.sql(
            "SELECT CONFIG:snowflake:compute_pool::string AS POOL, "
            "CONFIG:snowflake:service_spec_stage::string AS SPEC_STAGE, "
            "CONFIG:snowflake:database::string AS DB, "
            "CONFIG:snowflake:schema::string AS SCH "
            "FROM KG_JOB WHERE ID = ?",
            params=[run_id.strip()],
        ).collect()
        if not rows:
            raise ValueError("This run is no longer in the queue")
        record = rows[0].as_dict() if hasattr(rows[0], "as_dict") else dict(rows[0])
        pool = str(record.get("POOL") or "")
        stage = str(record.get("SPEC_STAGE") or "")
        database = str(record.get("DB") or "")
        schema = str(record.get("SCH") or "")
        if not (pool and stage and database and schema):
            raise ValueError(
                "This run does not record the compute pool and spec stage a retry needs"
            )
        for identifier in (pool, database, schema):
            if not _SERVICE_NAME_PATTERN.fullmatch(identifier):
                raise ValueError("This run records an unusable Snowflake object name")
        _stage_location(stage)
        job = run_id.strip()
        digest = hashlib.sha256(job.encode()).hexdigest()[:16]
        return {
            "pool": pool,
            "stage": stage,
            "spec": f"flakegraph-app-{digest}.yaml",
            "service": f"{database}.{schema}.{service_name_for_job(job)}",
        }

    def _requeue_failed_documents(self, run_id: str) -> int:
        """Return failed documents to the queue and make the job claimable again."""

        graph_id = self._run_graph_id(run_id)
        self.session.sql(
            "UPDATE KG_JOB_FILE SET STATUS = 'QUEUED', WORKER_ID = NULL, LEASE_UNTIL = NULL, "
            "ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED'",
            params=[run_id.strip(), graph_id],
        ).collect()
        rows = self.session.sql(
            "SELECT COUNT(*) AS QUEUED FROM KG_JOB_FILE WHERE JOB_ID = ? AND STATUS = 'QUEUED'",
            params=[run_id.strip()],
        ).collect()
        record = rows[0].as_dict() if rows and hasattr(rows[0], "as_dict") else {}
        queued = int(record.get("QUEUED") or 0)
        if queued:
            self.session.sql(
                "UPDATE KG_JOB SET STATUS = 'PENDING', ERROR = NULL, LEASE_OWNER = NULL, "
                "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() WHERE ID = ?",
                params=[run_id.strip()],
            ).collect()
        return queued

    def recover(self, run_id: str, config_path: Path | None = None) -> str:
        """Defer Snowflake service recovery to its managed job control plane."""

        del run_id, config_path
        raise NotImplementedError("Infrastructure recovery is managed by Snowflake")

    def forget(self, run_id: str, config_path: Path | None = None) -> None:
        """Remove one terminal job and its queue rows without touching graph data."""

        _ = config_path
        self._require_administration(self._run_graph_id(run_id))
        snapshot = self.status(run_id)
        if snapshot.status.lower() not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("An active Snowflake job must be cancelled before removal")
        self.session.sql("DELETE FROM KG_JOB_FILE WHERE JOB_ID = ?", params=[run_id]).collect()
        self.session.sql("DELETE FROM KG_JOB WHERE ID = ?", params=[run_id]).collect()

    def graph_row_counts(self, graph_id: str) -> dict[str, int]:
        """Return how many rows each table still holds for one graph.

        Shown before a graph is removed so the reader is told what will go,
        rather than asked to trust a verb.
        """

        target = graph_id.strip()
        if not target:
            return {}
        union = " UNION ALL ".join(
            f"SELECT '{table}' AS TABLE_NAME, COUNT(*) AS ROW_COUNT "
            f"FROM {table} WHERE GRAPH_ID = ?"
            for table in GRAPH_DATA_TABLES
        )
        rows = self.session.sql(union, params=[target] * len(GRAPH_DATA_TABLES)).collect()
        counts: dict[str, int] = {}
        for row in rows:
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            count = int(record.get("ROW_COUNT") or 0)
            if count:
                counts[str(record.get("TABLE_NAME"))] = count
        return counts

    def _release_graph_storage(self, graph_id: str) -> None:
        """Remove the staged files and job services a graph's runs left behind.

        A graph is not only rows. Each run uploaded the operator's documents into
        a stage folder, staged a service specification, and started a job service
        that Snowflake keeps after it stops. Clearing the tables alone leaves the
        source documents themselves sitting in the account — the very thing
        somebody deleting a graph means to remove.

        Best effort by design: a stage that no longer exists, or a service
        already dropped, must not stop the deletion the operator asked for.
        """

        rows = self.session.sql(
            "SELECT ID, CONFIG['files']['stage_prefix']::string AS UPLOAD_PREFIX, "
            "CONFIG['snowflake']['stage']::string AS DOCUMENT_STAGE, "
            "CONFIG['snowflake']['service_spec_stage']::string AS SPEC_STAGE, "
            "CONFIG['snowflake']['database']::string AS DB, "
            "CONFIG['snowflake']['schema']::string AS SCH "
            "FROM KG_JOB WHERE GRAPH_ID = ?",
            params=[graph_id],
        ).collect()
        for row in rows:
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            run_id = str(record.get("ID") or "")
            self._remove_staged_uploads(
                str(record.get("DOCUMENT_STAGE") or ""),
                str(record.get("UPLOAD_PREFIX") or ""),
            )
            self._remove_service_spec(str(record.get("SPEC_STAGE") or ""), run_id)
            self._drop_job_service(
                str(record.get("DB") or ""),
                str(record.get("SCH") or ""),
                run_id,
            )

    def _remove_staged_uploads(self, stage: str, prefix: str) -> None:
        """Delete one run's uploaded documents from the stage they were put in."""

        # Only the application's own upload namespace is ever removed. A run over
        # a stage folder an operator curates must leave that corpus untouched:
        # the graph was built from those documents, it does not own them.
        if not stage or not prefix or not prefix.startswith(f"{UPLOAD_NAMESPACE}/"):
            return
        with suppress(Exception):
            self.session.sql(f"REMOVE {_stage_location(stage, prefix)}").collect()

    def _remove_service_spec(self, spec_stage: str, run_id: str) -> None:
        """Delete the service specification staged for one run."""

        if not spec_stage or not run_id:
            return
        digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        with suppress(Exception):
            self.session.sql(
                f"REMOVE {_stage_location(spec_stage)}/flakegraph-app-{digest}.yaml"
            ).collect()

    def _drop_job_service(self, database: str, schema: str, run_id: str) -> None:
        """Drop the job service Snowflake retains after one run has stopped."""

        if not (database and schema and run_id):
            return
        for identifier in (database, schema):
            if not _SERVICE_NAME_PATTERN.fullmatch(identifier):
                return
        service = f"{database}.{schema}.{service_name_for_job(run_id)}"
        with suppress(Exception):
            self.session.sql(f"DROP SERVICE IF EXISTS {service}").collect()

    def delete_graph(self, graph_id: str) -> dict[str, int]:
        """Remove a graph and everything recorded about it.

        Forgetting a run takes it out of the history and deliberately leaves the
        graph behind, which is right for a run and wrong for a graph. Without
        this, a graph could only ever be hidden: its entities, evidence and
        source text stayed in the account with nothing in the application still
        pointing at them, and an operator who pressed a bin icon was left
        believing their documents were gone when they were not.
        """

        target = graph_id.strip()
        if not target:
            raise ValueError("graph ID must not be empty")
        self._require_administration(target)
        removed = self.graph_row_counts(target)
        # Read before the rows go: the staged uploads, specification file and
        # job service of each run are addressed from what those rows record, and
        # nothing else in the account remembers where they are.
        self._release_graph_storage(target)
        for table in GRAPH_DATA_TABLES:
            self.session.sql(
                f"DELETE FROM {table} WHERE GRAPH_ID = ?",  # noqa: S608
                params=[target],
            ).collect()
        return removed

    def viewer(self) -> Viewer:
        """Return the signed-in viewer together with the roles Snowflake grants them."""

        if self._viewer is not None:
            return self._viewer
        identity = current_viewer()
        self._viewer = (
            Viewer(
                user_name=identity.user_name,
                email=identity.email,
                roles=frozenset(self.viewer_roles(identity.user_name)),
            )
            if identity.is_identified
            else identity
        )
        return self._viewer

    def can_administer_graph(self, graph_id: str) -> bool:
        """Report whether this viewer may change one graph, rather than read it.

        A share is read access. Cancelling a billed run, forgetting it, renaming
        the graph or writing more documents into it are the owner's to do.
        """

        allowed, parameters = administrable_graph_predicate(self.viewer(), alias="G")
        rows = self.session.sql(
            "SELECT 1 FROM KG_GRAPH G WHERE G.GRAPH_ID = ? "
            f"AND {allowed} LIMIT 1",
            params=[graph_id.strip(), *parameters],
        ).collect()
        if rows:
            return True
        # A graph with no KG_GRAPH row has never been claimed, so nobody owns it
        # and the legacy rule applies rather than a refusal.
        known = self.session.sql(
            "SELECT 1 FROM KG_GRAPH WHERE GRAPH_ID = ? LIMIT 1",
            params=[graph_id.strip()],
        ).collect()
        return not known

    def _require_readable_stage(self, stage: str, prefix: str) -> None:
        """Refuse to read another viewer's uploads out of a shared stage.

        The application lists a stage with the app owner's privileges, so the
        resolved path is authorised rather than either field: a stage identifier
        can carry a path, and the stage root contains every viewer's folder.
        """

        if not may_read_stage_path(self.viewer(), _stage_relative_path(stage, prefix)):
            raise PermissionError(
                "That stage folder holds other users' uploads. Point at your own upload "
                "folder, or give a prefix outside it."
            )

    def _require_administration(self, graph_id: str) -> None:
        """Refuse an operation that would change a graph the viewer does not own."""

        if not self.can_administer_graph(graph_id):
            raise PermissionError(
                "This graph belongs to another user. Sharing grants read access only."
            )

    def _run_graph_id(self, run_id: str) -> str:
        """Return the graph one run belongs to, for authorising run-level writes."""

        rows = self.session.sql(
            "SELECT GRAPH_ID FROM KG_JOB WHERE ID = ?",
            params=[run_id.strip()],
        ).collect()
        if not rows:
            return ""
        record = rows[0].as_dict() if hasattr(rows[0], "as_dict") else dict(rows[0])
        return str(record.get("GRAPH_ID") or "")

    def can_open_graph(self, graph_id: str) -> bool:
        """Report whether this viewer may open one graph by its identifier.

        Listing alone is not a boundary: a graph is opened by identifier held in
        session state, so the same predicate has to gate the open itself.
        """

        visibility, parameters = visible_graph_predicate(self.viewer(), alias="G")
        rows = self.session.sql(
            "SELECT 1 FROM KG_JOB J LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID "
            f"WHERE J.GRAPH_ID = ? AND {visibility} LIMIT 1",
            params=[graph_id.strip(), *parameters],
        ).collect()
        return bool(rows)

    def _claim_graph(self, graph_id: str) -> None:
        """Record the submitting viewer as the graph's owner, once.

        Ownership is claimed at submission rather than at completion so a run
        that fails still belongs to whoever started it. An existing owner is
        never overwritten: a re-run of someone else's graph does not take it.
        """

        viewer = current_viewer()
        if not viewer.is_identified:
            return
        self.session.sql(
            "MERGE INTO KG_GRAPH target USING (SELECT ? AS GRAPH_ID, ? AS OWNER) source "
            "ON target.GRAPH_ID = source.GRAPH_ID "
            "WHEN MATCHED AND target.OWNER IS NULL THEN UPDATE SET OWNER = source.OWNER, "
            "UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT (GRAPH_ID, DISPLAY_NAME, OWNER) "
            "VALUES (source.GRAPH_ID, source.GRAPH_ID, source.OWNER)",
            params=[graph_id.strip(), viewer.user_name],
        ).collect()

    def viewer_roles(self, user_name: str) -> list[str]:
        """Return the roles Snowflake has granted one named user.

        Read for the viewer rather than from ``CURRENT_ROLE``, which inside an
        owner's-rights app reports the role that owns the application.
        """

        if not user_name.strip():
            return []
        try:
            rows = self.session.sql(f'SHOW GRANTS TO USER "{_quoted(user_name)}"').collect()
        except Exception:
            return []
        roles = set()
        for row in rows:
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            granted = record.get("role") or record.get("name")
            if isinstance(granted, str) and granted.strip():
                roles.add(granted.strip().strip('"').upper())
        return sorted(roles)

    def document_failures(self, run_id: str, *, limit: int = 20) -> list[tuple[str, int]]:
        """Return each distinct reason documents failed, with how many share it.

        The queue records a reason per document, and a run usually fails for one
        reason repeated: reporting only a count leaves the operator with nothing
        to act on and no way to tell a bad document from a dead session.
        """

        rows = self.session.sql(
            "SELECT ERROR, COUNT(*) AS DOCUMENTS FROM KG_JOB_FILE "
            "WHERE JOB_ID = ? AND STATUS = 'FAILED' AND ERROR IS NOT NULL "
            f"GROUP BY ERROR ORDER BY DOCUMENTS DESC LIMIT {max(1, min(limit, 50))}",
            params=[run_id.strip()],
        ).collect()
        failures = []
        for row in rows:
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            reason = _variant_mapping(record.get("ERROR"))
            message = str(reason.get("error_message") or record.get("ERROR") or "").strip()
            if message:
                failures.append((message, int(record.get("DOCUMENTS") or 0)))
        return failures

    def graph_owner(self, graph_id: str) -> str | None:
        """Return who owns one graph, or ``None`` when nobody has claimed it."""

        rows = self.session.sql(
            "SELECT OWNER FROM KG_GRAPH WHERE GRAPH_ID = ?",
            params=[graph_id.strip()],
        ).collect()
        if not rows:
            return None
        record = rows[0].as_dict() if hasattr(rows[0], "as_dict") else dict(rows[0])
        owner = record.get("OWNER")
        return str(owner) if owner else None

    def graph_shares(self, graph_id: str) -> list[tuple[str, str]]:
        """Return the grantee type and name of every share on one graph."""

        rows = self.session.sql(
            "SELECT GRANTEE_TYPE, GRANTEE FROM KG_GRAPH_ACL WHERE GRAPH_ID = ? "
            "ORDER BY GRANTEE_TYPE, GRANTEE",
            params=[graph_id.strip()],
        ).collect()
        shares = []
        for row in rows:
            record = row.as_dict() if hasattr(row, "as_dict") else dict(row)
            shares.append((str(record.get("GRANTEE_TYPE", "")), str(record.get("GRANTEE", ""))))
        return shares

    def share_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Grant one user or role access to a graph."""

        viewer = current_viewer()
        normalized_type = grantee_type.strip().upper()
        if normalized_type not in {"USER", "ROLE"}:
            raise ValueError("Share target must be a user or a role")
        normalized_grantee = grantee.strip().upper()
        if not normalized_grantee:
            raise ValueError("Share target must not be empty")
        self.session.sql(
            "MERGE INTO KG_GRAPH_ACL target USING (SELECT ? AS ID, ? AS GRAPH_ID, "
            "? AS GRANTEE_TYPE, ? AS GRANTEE, ? AS GRANTED_BY) source "
            "ON target.ID = source.ID "
            "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, GRANTEE_TYPE, GRANTEE, GRANTED_BY) "
            "VALUES (source.ID, source.GRAPH_ID, source.GRANTEE_TYPE, source.GRANTEE, "
            "source.GRANTED_BY)",
            params=[
                _share_id(graph_id, normalized_type, normalized_grantee),
                graph_id.strip(),
                normalized_type,
                normalized_grantee,
                viewer.user_name,
            ],
        ).collect()

    def unshare_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Withdraw one share from a graph."""

        self.session.sql(
            "DELETE FROM KG_GRAPH_ACL WHERE ID = ?",
            params=[
                _share_id(graph_id, grantee_type.strip().upper(), grantee.strip().upper())
            ],
        ).collect()

    def rename_graph(self, graph_id: str, display_name: str) -> str:
        """Upsert a durable Snowflake display name without rewriting graph rows."""

        normalized_name = validate_graph_name(display_name)
        normalized_id = graph_id.strip()
        if not normalized_id:
            raise ValueError("graph ID must not be empty")
        self._require_administration(normalized_id)
        self.session.sql(
            "MERGE INTO KG_GRAPH target USING (SELECT ? AS GRAPH_ID, ? AS DISPLAY_NAME) source "
            "ON target.GRAPH_ID = source.GRAPH_ID "
            "WHEN MATCHED THEN UPDATE SET DISPLAY_NAME = source.DISPLAY_NAME, "
            "UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT (GRAPH_ID, DISPLAY_NAME) "
            "VALUES (source.GRAPH_ID, source.DISPLAY_NAME)",
            params=[normalized_id, normalized_name],
        ).collect()
        return normalized_name

    def cluster(self, namespace: str) -> ClusterSnapshot | None:
        """Return no Kubernetes telemetry for a Snowflake-native backend."""

        _ = namespace
        return None

    def node_assignments(
        self,
        namespace: str,
        node_name: str,
    ) -> Sequence[NodeWorkAssignment]:
        """Return no Kubernetes assignments for the Snowflake-native backend."""

        _ = namespace, node_name
        return []

    def load_graph(self, location: str, graph_id: str | None = None) -> GraphDataset:
        """Load a bounded review projection while retaining full graph counts."""

        _ = location
        selected_graph = graph_id or self._latest_graph_id()
        tables: dict[str, list[dict[str, Any]]] = {}
        full_counts: dict[str, int] = {}
        count_rows = self.session.sql(
            graph_count_query(GRAPH_REVIEW_ROW_LIMITS),
            params=[selected_graph] * len(GRAPH_REVIEW_ROW_LIMITS),
        ).collect()
        full_counts = {
            str(item["table_name"]).removeprefix("KG_").lower(): int(item["row_count"])
            for item in (normalize_row(row.as_dict()) for row in count_rows)
        }
        for table, limit in GRAPH_REVIEW_ROW_LIMITS.items():
            if table == "KG_EDGE":
                continue
            order = "DEGREE DESC NULLS LAST, ID" if table == "KG_NODE" else "ID"
            projection = (
                f"SELECT {review_projection(table)} FROM {table} WHERE GRAPH_ID = ? "
                f"ORDER BY {order} LIMIT {limit}"
            )
            rows = self.session.sql(
                projection,
                params=[selected_graph],
            ).collect()
            tables[table] = [normalize_row(row.as_dict()) for row in rows]
        node_limit = GRAPH_REVIEW_ROW_LIMITS["KG_NODE"]
        edge_limit = GRAPH_REVIEW_ROW_LIMITS["KG_EDGE"]
        edge_rows = self.session.sql(
            "WITH REVIEW_NODES AS ("
            "SELECT ID FROM KG_NODE WHERE GRAPH_ID = ? "
            f"ORDER BY DEGREE DESC NULLS LAST, ID LIMIT {node_limit}"
            ") SELECT E.* EXCLUDE EMBEDDING FROM KG_EDGE E "
            "JOIN REVIEW_NODES SOURCE ON SOURCE.ID = E.SOURCE_NODE_ID "
            "JOIN REVIEW_NODES TARGET ON TARGET.ID = E.TARGET_NODE_ID "
            "WHERE E.GRAPH_ID = ? "
            f"ORDER BY E.ID LIMIT {edge_limit}",
            params=[selected_graph, selected_graph],
        ).collect()
        edges = [normalize_row(row.as_dict()) for row in edge_rows]
        report_rows = self.session.sql(
            "SELECT REPORT FROM KG_RUN_REPORT WHERE GRAPH_ID = ? ORDER BY UPDATED_AT DESC LIMIT 1",
            params=[selected_graph],
        ).collect()
        report = _variant_value(report_rows[0], "REPORT") if report_rows else {}
        report["app_full_counts"] = full_counts
        metrics_rows = self.session.sql(
            GRAPH_METRICS_QUERY,
            params=[selected_graph],
        ).collect()
        return GraphDataset(
            graph_id=selected_graph,
            nodes=tables["KG_NODE"],
            edges=edges,
            communities=tables["KG_COMMUNITY"],
            evidence=tables["KG_EVIDENCE"],
            run_report=report,
            graph_metrics=_variant_value(metrics_rows[0], "METRICS") if metrics_rows else {},
        )

    def load_run_graph(
        self,
        snapshot: RunSnapshot,
        config_path: Path | None = None,
    ) -> GraphDataset:
        """Load the selected run's graph from canonical Snowflake tables."""

        _ = config_path
        return self.load_graph("current", snapshot.graph_id)

    def _latest_graph_id(self) -> str:
        """Resolve the most recently updated graph visible to the active role."""

        rows = self.session.sql(
            "SELECT GRAPH_ID FROM KG_NODE GROUP BY GRAPH_ID ORDER BY MAX(UPDATED_AT) DESC LIMIT 1"
        ).collect()
        if not rows:
            raise ValueError("No graph nodes are visible in the current Snowflake schema")
        return str(normalize_row(rows[0].as_dict())["graph_id"])


def _with_live_stage_progress(
    stages: Sequence[StageProgress],
    raw_progress: object,
) -> list[StageProgress]:
    """Replace the running stage's file counts with the worker's own counts.

    File rows only move between queued, claimed, and done, so every stage of a
    run over one claimed batch reports the same zero progress until the batch
    finishes. A stage that legitimately runs for many minutes is then
    indistinguishable from a stalled worker. The worker records what it is
    actually doing on the job row, which is the only source that can tell them
    apart; absent it, the file-derived counts are still shown.
    """

    progress = _variant_mapping(raw_progress)
    if not progress:
        return list(stages)
    stage_name = str(progress.get("stage") or "")
    if not stage_name:
        return list(stages)
    counts = progress.get("counts")
    completed, total = _progress_counts(counts if isinstance(counts, Mapping) else {})
    message = progress.get("message")
    updated: list[StageProgress] = []
    seen = False
    for stage in stages:
        if stage.stage != stage_name:
            updated.append(stage)
            continue
        seen = True
        updated.append(
            StageProgress(
                stage.stage,
                stage.status,
                completed if total else stage.completed,
                total or stage.total,
                stage.elapsed_ms,
                str(message) if message else stage.message,
            )
        )
    if not seen:
        # The worker reports a stage the file rows have not recorded yet.
        updated.append(
            StageProgress(
                stage_name,
                str(progress.get("status") or "running"),
                completed,
                total,
                0,
                str(message) if message else None,
            )
        )
    return updated


def _progress_counts(counts: Mapping[str, object]) -> tuple[int, int | None]:
    """Pick the completed/total pair a stage reports, whatever it counts in.

    Stages count in their own units — batches, reports, candidates, files — so
    the pair is discovered rather than assumed.
    """

    for completed_key, total_key in (
        ("batches_completed", "batches_total"),
        ("reports_completed", "reports_total"),
        ("completed", "total"),
        ("files_processed", "files_total"),
    ):
        if completed_key in counts and total_key in counts:
            try:
                # VARIANT counters arrive as untyped objects; the except clause
                # below is what actually rejects a value that is not a number.
                return int(counts[completed_key]), int(counts[total_key])  # type: ignore[call-overload]
            except (TypeError, ValueError):
                return 0, None
    return 0, None


def _job_error_text(value: object) -> str | None:
    """Render a stored job error as a sentence rather than the JSON it is kept in.

    KG_JOB.ERROR is an object so the cause stays machine-readable, but the app
    puts it straight in front of a person. Handed the raw column they were shown
    a JSON literal with the explanation quoted inside it.
    """

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    payload = _variant_mapping(value)
    if not payload:
        return str(value).strip() or None
    message = payload.get("message")
    if message:
        return str(message)
    failed_files = payload.get("failed_files")
    if failed_files is not None:
        count = int(str(failed_files))
        documents = "document" if count == 1 else "documents"
        # Deliberately does not say the retries were exhausted. The job row
        # records how many documents failed and nothing about how many attempts
        # each had; a run whose worker died leaves them on their first attempt,
        # and telling the reader they were retried to exhaustion sends them
        # looking for a cause rather than pressing Retry.
        return f"{count} {documents} failed. Open the failures below for the reason."
    return "; ".join(f"{key}: {item}" for key, item in payload.items()) or None


def _variant_mapping(value: object) -> Mapping[str, object]:
    """Read a Snowflake VARIANT that arrives as either a mapping or JSON text."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def active_snowflake_session() -> Any | None:
    """Return Snowflake's active Streamlit session, or ``None`` outside Snowflake."""

    try:
        # The local app extra does not need Snowpark. Delay this import so local
        # and Kubernetes users do not inherit a Snowflake runtime dependency.
        from snowflake.snowpark.context import get_active_session  # noqa: PLC0415

        return get_active_session()
    except Exception:
        return None


def _source_object(row: object) -> SourceObject | None:
    """Normalize one LIST row and discard unsupported stage objects."""

    values = row.as_dict() if hasattr(row, "as_dict") else dict(cast(Any, row))
    name = str(values.get("name") or values.get("NAME") or "")
    if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        return None
    return SourceObject(
        uri=f"@{name.lstrip('@')}",
        name=name,
        size_bytes=int(values.get("size") or values.get("SIZE") or 0),
        modified_at=str(values.get("last_modified") or values.get("LAST_MODIFIED") or "") or None,
        checksum=str(
            values.get("md5") or values.get("MD5") or values.get("etag") or values.get("ETAG") or ""
        )
        or None,
    )


def _submission_row(request: IngestionRequest, item: SourceObject) -> tuple[str, ...]:
    """Return one deterministic durable queue row without retaining sibling objects."""

    checksum = item.checksum or _stable_id("stage_file_checksum", item.uri)
    file_id = _stable_id("stage_file", item.uri, checksum)
    return (
        _stable_id("snowflake_job_file", request.job_id, file_id),
        request.job_id,
        request.graph_id,
        file_id,
        item.uri,
        _relative_stage_name(item.name, str(request.source.get("stage", ""))),
        checksum,
    )


def _as_results(results: object) -> Sequence[object]:
    """Normalize put_stream's single result to the sequence put returns.

    A PutResult is a NamedTuple, so testing for a tuple splits one result into
    its eight fields and reports each as a separate upload of unknown status.
    A result is recognised by carrying a status, not by being iterable.
    """

    if hasattr(results, "status"):
        return [results]
    return list(results) if isinstance(results, list | tuple) else [results]


def _validate_upload_results(results: Sequence[object], object_name: str) -> None:
    """Require every overwrite request to report a fresh upload, never SKIPPED."""

    if not results:
        raise RuntimeError(f"Snowflake upload returned no status for {object_name}")
    statuses = []
    for item in results:
        if hasattr(item, "as_dict"):
            value = item.as_dict().get("status") or item.as_dict().get("STATUS")
        elif isinstance(item, Mapping):
            value = item.get("status") or item.get("STATUS")
        else:
            value = getattr(item, "status", "")
        statuses.append(str(value or "").upper())
    rejected = [status or "UNKNOWN" for status in statuses if status != "UPLOADED"]
    if rejected:
        raise RuntimeError(f"Snowflake did not overwrite {object_name}: " + ", ".join(rejected))


def _stable_id(prefix: str, *parts: object, length: int = 32) -> str:
    """Reproduce the processing package's public deterministic identifier contract."""

    joined = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(joined.encode()).hexdigest()[:length]}"


def _stage_relative_path(stage: str, prefix: str | None = None) -> str:
    """Return the path inside the stage that a stage/prefix pair addresses.

    A stage identifier may carry a path of its own, and it is concatenated with
    the prefix before the statement runs. Authorisation has to read the result,
    not either field alone.
    """

    normalized = stage.strip().lstrip("@").rstrip("/")
    _, separator, embedded = normalized.partition("/")
    return "/".join(
        part
        for part in (embedded if separator else "", (prefix or "").strip("/"))
        if part
    )


def _stage_location(stage: str, prefix: str | None = None) -> str:
    """Validate a stage identifier before interpolating it into LIST or PUT SQL."""

    normalized = stage.strip().lstrip("@").rstrip("/")
    stage_name, separator, stage_prefix = normalized.partition("/")
    if not _SAFE_STAGE_RE.fullmatch(stage_name):
        raise ValueError("Snowflake stage must be an identifier, not arbitrary SQL")
    suffix = "/".join(
        part
        for part in (stage_prefix if separator else "", prefix.strip("/") if prefix else "")
        if part
    )
    if suffix and (".." in Path(suffix).parts or not _SAFE_STAGE_PREFIX_RE.fullmatch(suffix)):
        raise ValueError("Snowflake stage prefix contains unsafe path characters")
    return f"@{stage_name}/{suffix}" if suffix else f"@{stage_name}"


def _relative_stage_name(raw_name: str, stage: str) -> str:
    """Strip the stage namespace from LIST output for worker blob access."""

    name = raw_name.lstrip("@")
    stage_name = stage.lstrip("@").rstrip("/")
    if name.startswith(stage_name + "/"):
        return name[len(stage_name) + 1 :]
    return name.split("/", 1)[1] if "/" in name else name


def _variant_value(row: Any, key: str) -> dict[str, Any]:
    """Normalize a Snowpark VARIANT cell into a dictionary."""

    values = row.as_dict() if hasattr(row, "as_dict") else dict(row)
    value = values.get(key) or values.get(key.lower()) or {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}
