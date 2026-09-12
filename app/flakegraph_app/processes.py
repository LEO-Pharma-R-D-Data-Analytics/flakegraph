"""Durable-enough process supervision for local Streamlit server sessions."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flakegraph_app.models import StorageKind
from flakegraph_app.run_catalog import write_run_record


@dataclass
class ManagedProcess:
    """One child process plus its stable on-disk logs and submission metadata."""

    run_id: str
    graph_id: str
    process: subprocess.Popen[bytes]
    events_path: Path
    output_path: Path
    stdout_path: Path
    config_path: Path
    started_at: str
    storage_kind: StorageKind
    storage_location: str


class ProcessRegistry:
    """Thread-safe registry shared across Streamlit script reruns."""

    def __init__(self) -> None:
        """Create an empty registry guarded by a reentrant lock."""

        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        run_id: str,
        graph_id: str,
        command: Sequence[str],
        cwd: Path,
        config_path: Path,
        output_path: Path,
        environment: Mapping[str, str],
        events_path: Path,
        stdout_path: Path,
        storage_kind: StorageKind,
        storage_location: str,
    ) -> ManagedProcess:
        """Start one non-shell command with separate machine and summary logs."""

        with self._lock:
            existing = self._processes.get(run_id)
            if existing and existing.process.poll() is None:
                raise ValueError(f"Run is already active: {run_id}")
            events_path.parent.mkdir(parents=True, exist_ok=True)
            (events_path.parent / "progress-summary.json").unlink(missing_ok=True)
            event_handle = events_path.open("wb")
            stdout_handle = stdout_path.open("wb")
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=dict(environment),
                    stdout=stdout_handle,
                    stderr=event_handle,
                    start_new_session=True,
                )
            finally:
                # Popen duplicates the descriptors for the child. Closing the
                # parent copies prevents one long-lived Streamlit session from
                # leaking two handles per completed run.
                event_handle.close()
                stdout_handle.close()
            managed = ManagedProcess(
                run_id=run_id,
                graph_id=graph_id,
                process=process,
                events_path=events_path,
                output_path=output_path,
                stdout_path=stdout_path,
                config_path=config_path,
                started_at=datetime.now(UTC).isoformat(),
                storage_kind=storage_kind,
                storage_location=storage_location,
            )
            self._processes[run_id] = managed
            self._write_metadata(managed)
            return managed

    def get(self, run_id: str) -> ManagedProcess | None:
        """Return a managed process without exposing the mutable registry."""

        with self._lock:
            return self._processes.get(run_id)

    def cancel(self, run_id: str) -> ManagedProcess:
        """Persist cancellation before terminating the owned process group."""

        with self._lock:
            managed = self._processes.get(run_id)
            if managed is None:
                raise ValueError(f"Unknown local run: {run_id}")
            requested_at = datetime.now(UTC).isoformat()
            write_run_record(
                managed.events_path.parent,
                {"status": "cancelled", "cancellation_requested_at": requested_at},
            )
            if managed.process.poll() is None:
                _terminate_owned_process(managed.process)
            return managed

    def forget(self, run_id: str) -> None:
        """Release a terminal process record so its catalog can be removed safely."""

        with self._lock:
            managed = self._processes.get(run_id)
            if managed is not None and managed.process.poll() is None:
                raise ValueError("An active local run must be cancelled before removal")
            self._processes.pop(run_id, None)

    @staticmethod
    def _write_metadata(managed: ManagedProcess) -> None:
        """Persist non-secret process identity for operator inspection."""

        write_run_record(
            managed.events_path.parent,
            {
                "run_id": managed.run_id,
                "graph_id": managed.graph_id,
                "runtime": "local",
                "status": "running",
                "pid": managed.process.pid,
                "process_identity": process_identity(managed.process.pid),
                "started_at": managed.started_at,
                "updated_at": managed.started_at,
                "config_path": str(managed.config_path),
                "output_path": str(managed.output_path),
                "storage_kind": managed.storage_kind.value,
                "storage_location": managed.storage_location,
            },
        )


def process_identity(pid: int) -> str | None:
    """Return a stable token for the current lifetime of a local process ID."""

    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    process_description = result.stdout.strip()
    if result.returncode or not process_description:
        return None
    return hashlib.sha256(f"{pid}:{process_description}".encode()).hexdigest()


def cancel_recorded_process(record: Mapping[str, object]) -> bool:
    """Signal a post-restart worker only when its persisted identity still matches."""

    try:
        pid = int(str(record.get("pid") or ""))
    except ValueError:
        return False
    expected = str(record.get("process_identity") or "")
    if pid <= 0 or not expected or process_identity(pid) != expected:
        return False
    try:
        if os.getpgid(pid) != pid:
            return False
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the worker session, falling back to the direct child handle."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.terminate()


PROCESS_REGISTRY = ProcessRegistry()
