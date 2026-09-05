"""Local CLI control-plane backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flakegraph_app import progress as progress_module
from flakegraph_app.backends.base import GraphAccessDefaults
from flakegraph_app.configuration import environment_for_request, write_run_config
from flakegraph_app.graph_store import load_local_graph, load_snowflake_graph
from flakegraph_app.models import (
    ARTIFACTS_UNAVAILABLE_STATUS,
    ClusterSnapshot,
    GraphDataset,
    IngestionRequest,
    NodeWorkAssignment,
    ProgressEvent,
    RunSnapshot,
    SourceObject,
    StorageKind,
)
from flakegraph_app.processes import PROCESS_REGISTRY, ManagedProcess, cancel_recorded_process
from flakegraph_app.progress import LocalProgress, read_local_progress
from flakegraph_app.run_catalog import (
    graph_name,
    list_run_records,
    read_run_record,
    write_run_record,
)
from flakegraph_app.run_catalog import (
    rename_graph as persist_graph_name,
)
from flakegraph_app.sources import list_azure_objects, list_local_objects, list_s3_objects

# Kept as a module attribute for app-service tests and third-party diagnostics
# that patched the former reader directly before cumulative checkpoints existed.
read_jsonl_events = progress_module.read_jsonl_events


STATE_ROOT_ENVIRONMENT = "FLAKEGRAPH_APP_STATE_ROOT"


def _configured_state_root() -> Path | None:
    """Return the operator's chosen state directory, if they named one."""

    configured = os.environ.get(STATE_ROOT_ENVIRONMENT, "").strip()
    return Path(configured) if configured else None


class LocalBackend(GraphAccessDefaults):
    """Run the processing engine as a supervised local CLI child process."""

    def __init__(self, repository_root: Path, state_root: Path | None = None) -> None:
        """Configure command working directory and app-owned state storage.

        State defaults to a directory inside the checkout, which is right for a
        developer running from a clone and wrong for a container, where the code
        is on a read-only filesystem. Deployments name a writable location
        instead; without one, hiding a run or registering a cluster fails at the
        moment the user asks for it rather than at startup.
        """

        self.repository_root = repository_root.resolve()
        configured = state_root or _configured_state_root()
        self.state_root = (configured or repository_root / ".flakegraph" / "app").resolve()
        # Set by runtimes whose listing can fall back to a less authoritative
        # source; the sidebar shows it so a degraded history says so.
        self.listing_warning: str | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        """Describe controls available when Streamlit runs beside FlakeGraph."""

        return frozenset(
            {"upload", "local", "azure_blob", "s3", "progress", "graph", "forget", "rename"}
        )

    def list_source_objects(
        self,
        source: Mapping[str, str],
        *,
        limit: int = 1_000,
    ) -> Sequence[SourceObject]:
        """Dispatch metadata listing to the selected source browser."""

        kind = source.get("kind", "local")
        if kind == "local":
            return list_local_objects(Path(source.get("path", "")), limit)
        if kind == "s3":
            return list_s3_objects(source, limit)
        if kind == "azure_blob":
            return list_azure_objects(source, limit)
        raise ValueError(f"Source browser is not available locally: {kind}")

    def graph_embedding_width(self) -> int | None:
        """Report no fixed width: this destination stores vectors as written."""

        return None

    def preflight(self, request: IngestionRequest) -> Mapping[str, object]:
        """Generate the effective profile and invoke the core preflight command."""

        config_path = self._config_path(request.job_id)
        write_run_config(request, config_path)
        result = subprocess.run(
            ["uv", "run", "flakegraph", "preflight", "--config", str(config_path)],
            cwd=self.repository_root,
            env=environment_for_request(request),
            capture_output=True,
            text=True,
            check=False,
        )
        payload = _last_json_object(result.stdout) or {
            "ok": False,
            "errors": [result.stderr.strip() or "Preflight returned no JSON result"],
        }
        if result.returncode and payload.get("ok") is not False:
            payload = {**payload, "ok": False, "errors": [result.stderr.strip()]}
        return payload

    def submit(self, request: IngestionRequest) -> RunSnapshot:
        """Start the worker and return immediately so Streamlit remains responsive."""

        run_directory = self.state_root / "runs" / request.job_id
        config_path = write_run_config(request, run_directory / "config.yaml")
        managed = PROCESS_REGISTRY.start(
            run_id=request.job_id,
            graph_id=request.graph_id,
            command=[
                "uv",
                "run",
                "flakegraph",
                "worker",
                "--config",
                str(config_path),
                "--progress",
                "json",
            ],
            cwd=self.repository_root,
            config_path=config_path,
            output_path=request.output.workspace_path,
            environment=environment_for_request(request),
            events_path=run_directory / "events.jsonl",
            stdout_path=run_directory / "stdout.log",
            storage_kind=request.output.kind,
            storage_location=request.output.location,
        )
        return self._snapshot(managed)

    def list_runs(self, *, limit: int = 100) -> Sequence[RunSnapshot]:
        """Return newest local runs without parsing every historical event log.

        Sidebar rows require identity, terminal state, timestamps, and storage only.
        Detailed stage events are loaded by ``status`` after a run is selected. A
        newly terminated managed process is summarized once so its durable catalog
        state remains correct; subsequent listings stay metadata-only.
        """

        snapshots = []
        for run_directory, record in list_run_records(self.state_root, limit):
            if str(record.get("runtime") or "local") != "local":
                continue
            run_id = str(record.get("run_id") or run_directory.name)
            managed = PROCESS_REGISTRY.get(run_id)
            if managed and (
                managed.process.poll() is None
                or str(record.get("status", "")).lower()
                in {
                    "planning",
                    "pending",
                    "queued",
                    "running",
                    "processing",
                    "submitted",
                }
            ):
                snapshots.append(self._snapshot(managed))
            else:
                snapshots.append(
                    self._catalog_snapshot(run_directory, record, include_progress=False)
                )
        return snapshots

    def status(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Aggregate current process state and recent structured progress events."""

        _ = config_path
        managed = PROCESS_REGISTRY.get(run_id)
        if managed is not None:
            return self._snapshot(managed)
        run_directory = self.state_root / "runs" / run_id
        record = read_run_record(run_directory)
        if not record:
            raise ValueError(f"Unknown local run: {run_id}")
        return self._catalog_snapshot(run_directory, record)

    def cancel(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Persist sticky cancellation and stop a live or safely recovered worker."""

        _ = config_path
        managed = PROCESS_REGISTRY.get(run_id)
        if managed is not None:
            return self._snapshot(PROCESS_REGISTRY.cancel(run_id), forced_status="cancelled")
        run_directory = self.state_root / "runs" / run_id
        record = read_run_record(run_directory)
        if not record:
            raise ValueError(f"Unknown local run: {run_id}")
        write_run_record(
            run_directory,
            {
                "status": "cancelled",
                "cancellation_requested_at": datetime.now(UTC).isoformat(),
            },
        )
        cancel_recorded_process(record)
        return self._catalog_snapshot(run_directory, read_run_record(run_directory))

    def retry(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Reject in-place retries because local child processes are not durable queues."""

        del run_id, config_path
        raise NotImplementedError("Retry is available only for durable distributed runs")

    def recover(self, run_id: str, config_path: Path | None = None) -> str:
        """Reject infrastructure recovery for local child-process runs."""

        del run_id, config_path
        raise NotImplementedError("Infrastructure recovery is available only for Kubernetes runs")

    def forget(self, run_id: str, config_path: Path | None = None) -> None:
        """Remove terminal app-owned metadata while preserving generated artifacts."""

        _ = config_path
        snapshot = self.status(run_id)
        if snapshot.status.lower() in {"running", "queued", "processing", "submitted"}:
            raise ValueError("An active run must be cancelled before removal")
        PROCESS_REGISTRY.forget(run_id)
        run_directory = (self.state_root / "runs" / run_id).resolve()
        if run_directory.parent != (self.state_root / "runs").resolve():
            raise ValueError("Refusing to remove a run outside the app state directory")
        shutil.rmtree(run_directory, ignore_errors=True)

    def rename_graph(self, graph_id: str, display_name: str) -> str:
        """Persist one display name for every local or fleet run of a graph."""

        return persist_graph_name(self.state_root, graph_id, display_name)

    def cluster(self, namespace: str) -> ClusterSnapshot | None:
        """Return no cluster telemetry for a local process backend."""

        _ = namespace
        return None

    def node_assignments(
        self,
        namespace: str,
        node_name: str,
    ) -> Sequence[NodeWorkAssignment]:
        """Return no node assignments because local execution has no fleet nodes."""

        _ = namespace, node_name
        return []

    def load_graph(self, location: str, graph_id: str | None = None) -> GraphDataset:
        """Load a standard local artifact directory for exploration."""

        _ = graph_id
        return load_local_graph(self.artifact_directory(location))

    def artifact_directory(self, location: str) -> Path:
        """Resolve a recorded artifact location against the repository it belongs to.

        Runs record where their graph was written, and a profile may express that
        relative to the repository rather than absolutely. Resolving it against
        the process working directory instead would make a graph openable or not
        depending on where the server happened to be started from.
        """

        path = Path(location).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        return path.resolve()

    def load_run_graph(
        self,
        snapshot: RunSnapshot,
        config_path: Path | None = None,
    ) -> GraphDataset:
        """Load the artifact directory recorded when a local run was submitted."""

        if snapshot.storage_kind == StorageKind.SNOWFLAKE:
            if config_path is None:
                value = str(snapshot.raw.get("config_path") or "")
                config_path = Path(value) if value else None
            if config_path is None:
                raise ValueError(f"Run {snapshot.run_id} does not record its Snowflake profile")
            return load_snowflake_graph(config_path, snapshot.graph_id)
        if not snapshot.output_path:
            raise ValueError(f"Run {snapshot.run_id} does not record an output directory")
        return self.load_graph(snapshot.output_path, snapshot.graph_id)

    def _config_path(self, run_id: str) -> Path:
        """Return the app-owned generated profile path for one run."""

        return self.state_root / "runs" / run_id / "config.yaml"

    def _snapshot(self, managed: ManagedProcess, forced_status: str | None = None) -> RunSnapshot:
        """Build a stable view from one child process and its append-only event log."""

        progress = read_local_progress(managed.events_path)
        events = progress.events
        return_code = managed.process.poll()
        record = read_run_record(managed.events_path.parent)
        if forced_status or record.get("cancellation_requested_at"):
            status = forced_status or "cancelled"
        elif return_code is None:
            status = "running"
        elif return_code == 0:
            status = "succeeded"
        else:
            status = "failed"
        latest = events[-1] if events else None
        error = None
        if status == "failed":
            error = _failure_summary(managed.events_path, events)
        snapshot = RunSnapshot(
            run_id=managed.run_id,
            graph_id=managed.graph_id,
            status=status,
            started_at=managed.started_at,
            updated_at=(
                progress.updated_at
                or (latest.timestamp if latest else datetime.now(UTC).isoformat())
            ),
            graph_name=graph_name(self.state_root, managed.graph_id),
            stages=progress.stages,
            events=events[-200:],
            documents_total=progress.documents_total,
            documents_completed=progress.documents_completed,
            documents_failed=progress.documents_failed,
            output_path=str(managed.output_path),
            storage_kind=managed.storage_kind,
            storage_location=managed.storage_location,
            error=error,
            raw={"config_path": str(managed.config_path)},
        )
        write_run_record(
            managed.events_path.parent,
            {
                "status": snapshot.status,
                "updated_at": snapshot.updated_at,
                "output_path": snapshot.output_path,
                "config_path": str(managed.config_path),
                "error": snapshot.error,
                "storage_kind": snapshot.storage_kind.value,
                "storage_location": snapshot.storage_location,
                "documents_total": snapshot.documents_total,
                "documents_completed": snapshot.documents_completed,
                "documents_failed": snapshot.documents_failed,
            },
        )
        return snapshot

    def _catalog_snapshot(
        self,
        run_directory: Path,
        record: Mapping[str, Any],
        *,
        durable_runtime: bool = False,
        include_progress: bool = True,
    ) -> RunSnapshot:
        """Reconstruct a run, loading detailed events only for an opened workspace."""

        progress = (
            read_local_progress(run_directory / "events.jsonl")
            if include_progress
            else LocalProgress([], [], None, 0, 0, None)
        )
        events = progress.events
        output_path = str(record.get("output_path") or "") or None
        storage_kind = _storage_kind(record.get("storage_kind"))
        recorded_status = str(record.get("status") or "unknown").lower()
        if record.get("cancellation_requested_at") or recorded_status == "cancelled":
            status = "cancelled"
        elif recorded_status == "failed":
            status = "failed"
        elif durable_runtime or recorded_status == "succeeded":
            status = recorded_status
        elif _pid_is_running(record.get("pid")):
            status = "running"
        elif output_path and _graph_artifacts_exist(self.artifact_directory(output_path)):
            # Legacy records may predate durable terminal-state persistence.
            status = "succeeded"
        else:
            status = "interrupted"
        if (
            status == "succeeded"
            and not durable_runtime
            and storage_kind == StorageKind.LOCAL
            and output_path
            and not _graph_artifacts_exist(self.artifact_directory(output_path))
        ):
            # A local run writes its graph once and never fetches it again, so an
            # absent directory means the graph is not reachable from this host.
            # A durable runtime is excluded deliberately: it re-exports the graph
            # from the fleet's object store on demand, where a missing local copy
            # is the normal state rather than a lost one.
            status = ARTIFACTS_UNAVAILABLE_STATUS
        documents_total = (
            progress.documents_total
            if include_progress
            else _optional_int(record.get("documents_total"))
        )
        documents_completed = (
            progress.documents_completed
            if include_progress
            else int(record.get("documents_completed") or 0)
        )
        documents_failed = (
            progress.documents_failed
            if include_progress
            else int(record.get("documents_failed") or 0)
        )
        graph_id = str(record.get("graph_id") or record.get("run_id") or run_directory.name)
        recorded_graph_name = record.get("graph_name")
        return RunSnapshot(
            run_id=str(record.get("run_id") or run_directory.name),
            graph_id=graph_id,
            status=status,
            started_at=str(record.get("started_at")) if record.get("started_at") else None,
            updated_at=(
                progress.updated_at
                if progress.updated_at
                else str(record.get("updated_at") or record.get("started_at") or "") or None
            ),
            graph_name=(
                graph_name(self.state_root, graph_id)
                or (str(recorded_graph_name) if recorded_graph_name else None)
            ),
            stages=progress.stages if include_progress else (),
            events=events[-200:],
            documents_total=documents_total,
            documents_completed=documents_completed,
            documents_failed=documents_failed,
            output_path=output_path,
            storage_kind=storage_kind,
            storage_location=(
                str(record.get("storage_location"))
                if record.get("storage_location")
                else output_path
            ),
            error=str(record.get("error")) if record.get("error") else None,
            raw={"config_path": str(record.get("config_path") or "")},
        )


def _last_json_object(output: str) -> dict[str, Any]:
    """Parse a pretty-printed CLI JSON object while ignoring surrounding text."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _failure_summary(
    events_path: Path,
    events: Sequence[ProgressEvent],
    *,
    limit: int = 500,
) -> str:
    """State one cause for a stopped local worker rather than quoting its log.

    Structured progress and the worker's own diagnostics share a single stream,
    so its tail interleaves JSON records with traceback frames — a wall of text
    in which the one useful line is not the last. The cause is either the stage
    that reported failure or the final line of the traceback, and the complete
    log stays on disk for anyone who wants it.
    """

    failed = next((event for event in reversed(events) if event.status == "failed"), None)
    stage = failed.stage.replace("_", " ") if failed is not None else ""
    if failed is not None and failed.message:
        return _bounded(f"The {stage} stage failed: {failed.message}", limit)
    diagnostic = _last_diagnostic_line(events_path)
    if diagnostic and failed is not None:
        return _bounded(f"The {stage} stage failed: {diagnostic}", limit)
    if diagnostic:
        return _bounded(diagnostic, limit)
    if failed is not None:
        return f"The {stage} stage failed without reporting a cause."
    return "The worker stopped without reporting a cause."


def _last_diagnostic_line(path: Path, lines: int = 200) -> str | None:
    """Return the worker's own final message from a log shared with progress JSON.

    Progress records are complete JSON objects and traceback frames are indented
    continuation lines; what remains is the message the worker meant a person to
    read, of which the last is the one that ended the run.
    """

    if not path.exists():
        return None
    for line in reversed(progress_module._tail_lines(path, lines)):  # noqa: SLF001
        text = line.strip()
        if not text or text.startswith(("{", 'File "', "Traceback")):
            continue
        return text
    return None


def _bounded(message: str, limit: int) -> str:
    """Keep one failure sentence readable inside a status panel."""

    collapsed = " ".join(message.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1].rstrip()}…"


def _graph_artifacts_exist(path: Path) -> bool:
    """Return whether a local directory contains the minimum explorer dataset."""

    return (path / "nodes.parquet").is_file() and (path / "edges.parquet").is_file()


def _pid_is_running(raw_pid: object) -> bool:
    """Check a recorded local PID without signalling or taking ownership of it."""

    try:
        pid = int(str(raw_pid))
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def _storage_kind(value: object) -> StorageKind:
    """Normalize current and legacy catalog values into the public storage enum."""

    try:
        return StorageKind(str(value or StorageKind.LOCAL.value))
    except ValueError:
        return StorageKind.LOCAL


def _optional_int(value: object) -> int | None:
    """Normalize an optional catalog counter without treating zero as missing."""

    if value is None:
        return None
    if not isinstance(value, int | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
