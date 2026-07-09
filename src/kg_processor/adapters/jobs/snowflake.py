"""Snowflake job and per-file lease coordination adapter."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    connect_snowflake,
)
from kg_processor.application.progress import ProgressEvent
from kg_processor.domain.jobs import JobFileClaim, JobFileResult


class JobClaimResult(BaseModel):
    """Outcome returned after attempting to lease a graph-level KG_JOB row."""

    claimed: bool
    job_id: str
    status: str | None = None
    lease_owner: str | None = None


class SnowflakeJobManager:
    """Coordinates graph-level and file-level leases in Snowflake tables."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        """Store Snowflake connection details and an optional test connector."""

        self.config = config
        self.connector_factory = connector_factory

    def claim_job(
        self,
        job_id: str,
        graph_id: str,
        lease_owner: str,
        lease_seconds: int,
        config: dict[str, Any],
    ) -> JobClaimResult:
        """Create a pending job if needed and attempt to acquire its lease."""

        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(
                build_create_pending_job_statement(),
                [job_id, graph_id, _json(config)],
            )
            cursor.execute(
                build_claim_job_statement(lease_seconds),
                [lease_owner, job_id],
            )
            cursor.execute(build_select_claimed_job_statement(), [job_id, lease_owner])
            row = cursor.fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()
        if row is None:
            return JobClaimResult(claimed=False, job_id=job_id)
        return JobClaimResult(
            claimed=True,
            job_id=str(row[0]),
            status=str(row[1]),
            lease_owner=str(row[2]),
        )

    def complete_job(self, job_id: str, lease_owner: str, report: dict[str, Any]) -> None:
        """Mark a leased graph-level job as succeeded and attach its run report."""

        self._update_terminal_job(
            build_complete_job_statement(),
            [json.dumps(report, sort_keys=True), job_id, lease_owner],
        )

    def fail_job(self, job_id: str, lease_owner: str, error: dict[str, Any]) -> None:
        """Mark a leased graph-level job as failed with a structured error."""

        self._update_terminal_job(
            build_fail_job_statement(),
            [json.dumps(error, sort_keys=True), job_id, lease_owner],
        )

    def heartbeat_job(self, job_id: str, lease_owner: str, lease_seconds: int) -> None:
        """Extend a graph-level job lease while a worker is still alive."""

        self._update_terminal_job(
            build_heartbeat_job_statement(lease_seconds),
            [job_id, lease_owner],
        )

    def claim_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        lease_seconds: int,
        batch_size: int,
    ) -> list[JobFileClaim]:
        """Claim the next available KG_JOB_FILE rows for a worker."""

        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(build_mark_job_running_statement(), [job_id, graph_id])
            cursor.execute(
                build_claim_job_files_statement(lease_seconds, batch_size),
                [worker_id, job_id, graph_id, job_id, graph_id],
            )
            cursor.execute(
                build_select_claimed_job_files_statement(batch_size),
                [job_id, graph_id, worker_id],
            )
            rows = cursor.fetchall()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()
        return [_job_file_claim_from_row(row) for row in rows]

    def complete_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        results: list[JobFileResult],
    ) -> None:
        """Mark claimed file rows done and persist per-file processing metadata."""

        if not results:
            return
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            statement = build_complete_job_file_statement()
            for result in results:
                cursor.execute(
                    statement,
                    [
                        result.rows_written,
                        _json(result.row_counts),
                        result.stage,
                        result.ocr_provider,
                        result.llm_provider,
                        result.embedding_model,
                        result.embedding_dimension,
                        _job_file_audit_json(result),
                        job_id,
                        graph_id,
                        result.file_id,
                        worker_id,
                    ],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def fail_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        error: dict[str, Any],
    ) -> None:
        """Mark claimed file rows failed with the same structured error payload."""

        if not file_ids:
            return
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            statement = build_fail_job_file_statement()
            error_json = json.dumps(error, sort_keys=True)
            for file_id in file_ids:
                cursor.execute(statement, [error_json, job_id, graph_id, file_id, worker_id])
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def heartbeat_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        lease_seconds: int,
    ) -> None:
        """Extend leases for claimed file rows owned by the current worker."""

        if not file_ids:
            return
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(
                build_heartbeat_job_files_statement(lease_seconds, len(file_ids)),
                [job_id, graph_id, worker_id, *file_ids],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def update_job_file_progress(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        stage: str,
    ) -> None:
        """Update the non-terminal stage text for claimed file rows."""

        if not file_ids:
            return
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(
                build_update_job_file_progress_statement(len(file_ids)),
                [stage, job_id, graph_id, worker_id, *file_ids],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def complete_job_if_file_queue_drained(
        self,
        job_id: str,
        graph_id: str,
        report: dict[str, Any],
    ) -> None:
        """Complete the graph-level job once no queued or claimed file rows remain."""

        self._update_terminal_job(
            build_complete_job_if_file_queue_drained_statement(),
            [json.dumps(report, sort_keys=True), job_id, graph_id, job_id, graph_id],
        )

    def _update_terminal_job(self, sql: str, params: list[object]) -> None:
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()


class SnowflakeJobFileProgressSink:
    """Persist live pipeline progress onto claimed KG_JOB_FILE rows.

    KG_JOB_FILE is the table the orchestration app naturally polls while a
    distributed file queue is running. The sink keeps STAGE current for claimed
    rows without changing terminal status; completion and failure still go
    through the explicit job manager methods.
    """

    def __init__(
        self,
        job_manager: SnowflakeJobManager,
        job_id: str,
        graph_id: str,
        worker_id: str,
        claimed_file_ids: list[str],
    ) -> None:
        self.job_manager = job_manager
        self.job_id = job_id
        self.graph_id = graph_id
        self.worker_id = worker_id
        self.claimed_file_ids = list(dict.fromkeys(claimed_file_ids))
        self.claimed_file_id_set = set(self.claimed_file_ids)

    def emit(self, event: ProgressEvent) -> None:
        """Translate a pipeline progress event into KG_JOB_FILE.STAGE updates."""

        file_ids = self._target_file_ids(event)
        if not file_ids:
            return
        self.job_manager.update_job_file_progress(
            self.job_id,
            self.graph_id,
            self.worker_id,
            file_ids,
            _progress_stage(event),
        )

    def _target_file_ids(self, event: ProgressEvent) -> list[str]:
        if event.file_id:
            if event.file_id not in self.claimed_file_id_set:
                return []
            return [event.file_id]
        return self.claimed_file_ids


def build_create_pending_job_statement() -> str:
    """Return SQL that idempotently creates a pending KG_JOB row."""

    return (
        "MERGE INTO KG_JOB target USING ("
        "SELECT ? AS ID, ? AS GRAPH_ID, PARSE_JSON(?) AS CONFIG"
        ") source ON target.ID = source.ID "
        "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS, CONFIG) "
        "VALUES (source.ID, source.GRAPH_ID, 'PENDING', source.CONFIG)"
    )


def build_mark_job_running_statement() -> str:
    """Return SQL that moves or creates a KG_JOB row in RUNNING state."""

    return (
        "MERGE INTO KG_JOB target USING ("
        "SELECT ? AS ID, ? AS GRAPH_ID"
        ") source ON target.ID = source.ID "
        "WHEN MATCHED THEN UPDATE SET STATUS = 'RUNNING', UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS) "
        "VALUES (source.ID, source.GRAPH_ID, 'RUNNING')"
    )


def build_claim_job_statement(lease_seconds: int) -> str:
    """Return SQL that atomically claims a graph-level job lease."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return (
        "UPDATE KG_JOB SET STATUS = 'RUNNING', LEASE_OWNER = ?, "
        f"LEASE_EXPIRES_AT = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "ATTEMPT = COALESCE(ATTEMPT, 0) + 1, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND (STATUS IN ('PENDING', 'FAILED') OR "
        "(STATUS = 'RUNNING' AND LEASE_EXPIRES_AT < CURRENT_TIMESTAMP()))"
    )


def build_select_claimed_job_statement() -> str:
    """Return SQL that confirms the current worker owns a claimed KG_JOB row."""

    return (
        "SELECT ID, STATUS, LEASE_OWNER FROM KG_JOB "
        "WHERE ID = ? AND STATUS = 'RUNNING' AND LEASE_OWNER = ?"
    )


def build_complete_job_statement() -> str:
    """Return SQL that completes a running KG_JOB row for its lease owner."""

    return (
        "UPDATE KG_JOB SET STATUS = 'SUCCEEDED', ERROR = NULL, CONFIG = "
        "OBJECT_INSERT(COALESCE(CONFIG, OBJECT_CONSTRUCT()), 'run_report', PARSE_JSON(?), TRUE), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING'"
    )


def build_fail_job_statement() -> str:
    """Return SQL that records failure on a running KG_JOB row."""

    return (
        "UPDATE KG_JOB SET STATUS = 'FAILED', ERROR = PARSE_JSON(?), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING'"
    )


def build_heartbeat_job_statement(lease_seconds: int) -> str:
    """Return SQL that refreshes a graph-level job lease."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return (
        "UPDATE KG_JOB SET "
        f"LEASE_EXPIRES_AT = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING'"
    )


def build_claim_job_files_statement(lease_seconds: int, batch_size: int) -> str:
    """Return SQL that claims the next queued or expired file rows."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'CLAIMED', WORKER_ID = ?, "
        f"LEASE_UNTIL = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "ATTEMPTS = COALESCE(ATTEMPTS, 0) + 1, ERROR = NULL, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND FILE_ID IN ("
        "SELECT FILE_ID FROM ("
        "SELECT FILE_ID FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND "
        "(STATUS = 'QUEUED' OR (STATUS = 'CLAIMED' AND LEASE_UNTIL < CURRENT_TIMESTAMP())) "
        "ORDER BY UPDATED_AT, FILE_ID "
        f"LIMIT {batch_size}"
        ")"
        ")"
    )


def build_select_claimed_job_files_statement(batch_size: int) -> str:
    """Return SQL that fetches file rows just claimed by a worker."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (
        "SELECT JOB_ID, GRAPH_ID, FILE_ID, COALESCE(SOURCE_URI, BLOB_PATH), CHECKSUM, "
        "STATUS, WORKER_ID, COALESCE(ATTEMPTS, 0) "
        "FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'CLAIMED' AND WORKER_ID = ? "
        "ORDER BY UPDATED_AT, FILE_ID "
        f"LIMIT {batch_size}"
    )


def build_complete_job_file_statement() -> str:
    """Return SQL that completes one claimed KG_JOB_FILE row."""

    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'DONE', LEASE_UNTIL = NULL, "
        "ROWS_WRITTEN = ?, ROW_COUNTS = PARSE_JSON(?), STAGE = ?, "
        "OCR_PROVIDER = ?, LLM_PROVIDER = ?, EMBED_MODEL = ?, EMBED_DIM = ?, "
        "AUDIT = PARSE_JSON(?), ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND FILE_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED'"
    )


def build_fail_job_file_statement() -> str:
    """Return SQL that fails one claimed KG_JOB_FILE row."""

    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'FAILED', LEASE_UNTIL = NULL, "
        "ERROR = PARSE_JSON(?), UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND FILE_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED'"
    )


def build_heartbeat_job_files_statement(lease_seconds: int, file_count: int) -> str:
    """Return SQL that refreshes leases for a fixed set of file rows."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if file_count <= 0:
        raise ValueError("file_count must be positive")
    placeholders = ", ".join("?" for _ in range(file_count))
    return (
        "UPDATE KG_JOB_FILE SET "
        f"LEASE_UNTIL = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' "
        f"AND FILE_ID IN ({placeholders})"
    )


def build_update_job_file_progress_statement(file_count: int) -> str:
    """Return SQL that writes progress stage text for claimed file rows."""

    if file_count <= 0:
        raise ValueError("file_count must be positive")
    placeholders = ", ".join("?" for _ in range(file_count))
    return (
        "UPDATE KG_JOB_FILE SET STAGE = ?, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' "
        f"AND FILE_ID IN ({placeholders})"
    )


def build_complete_job_if_file_queue_drained_statement() -> str:
    """Return SQL that completes a KG_JOB only after its file queue is drained."""

    return (
        "UPDATE KG_JOB SET STATUS = 'SUCCEEDED', ERROR = NULL, CONFIG = "
        "OBJECT_INSERT(COALESCE(CONFIG, OBJECT_CONSTRUCT()), 'run_report', PARSE_JSON(?), TRUE), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND GRAPH_ID = ? AND NOT EXISTS ("
        "SELECT 1 FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS IN ('QUEUED', 'CLAIMED')"
        ")"
    )


def _job_file_claim_from_row(row: object) -> JobFileClaim:
    if not isinstance(row, tuple | list):
        raise ValueError("KG_JOB_FILE claim rows must be sequences")
    return JobFileClaim(
        job_id=str(row[0]),
        graph_id=str(row[1]),
        file_id=str(row[2]),
        source_uri=str(row[3]) if row[3] is not None else None,
        checksum=str(row[4]) if row[4] is not None else None,
        status=str(row[5]),
        worker_id=str(row[6]) if row[6] is not None else None,
        attempts=int(row[7] or 0),
    )


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _job_file_audit_json(result: JobFileResult) -> str:
    # Keep KG_JOB_FILE.AUDIT self-contained so the orchestration app can display
    # per-file outcomes without joining back to the graph report tables.
    return _json(
        {
            "file_id": result.file_id,
            "rows_written": result.rows_written,
            "row_counts": result.row_counts,
            "stage": result.stage,
            "providers": {
                "ocr": result.ocr_provider,
                "llm": result.llm_provider,
                "embedding_model": result.embedding_model,
                "embedding_dimension": result.embedding_dimension,
            },
            "metadata": result.audit,
        }
    )


def _progress_stage(event: ProgressEvent) -> str:
    return f"{event.stage}:{event.status}"
