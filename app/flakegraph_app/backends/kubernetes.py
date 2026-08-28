"""Kubernetes fleet control and observability backend."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from base64 import b64decode
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import yaml
from flakegraph_app.backends.local import LocalBackend, _last_json_object, _storage_kind
from flakegraph_app.cluster_catalog import ClusterTarget, read_catalog
from flakegraph_app.configuration import (
    build_run_config,
    run_ontology_profile,
    sensitive_config_key,
    write_run_config,
)
from flakegraph_app.graph_store import load_local_graph
from flakegraph_app.models import (
    DISTRIBUTED_STAGE_ORDER,
    ClusterSnapshot,
    GraphDataset,
    IngestionRequest,
    NodeStatus,
    NodeWorkAssignment,
    RunSnapshot,
    StageProgress,
    StorageKind,
    WorkloadStatus,
)
from flakegraph_app.run_catalog import (
    graph_name,
    hidden_run_ids,
    hide_run,
    list_run_records,
    read_run_record,
    write_run_record,
)
from psycopg import connect

_KUBECTL_METRIC_COLUMNS = 3
_KUBECTL_NODE_METRIC_COLUMNS = 5
_KUBECTL_TIMEOUT_SECONDS = 15
_CONTROL_COMMAND_TIMEOUT_SECONDS = 60
_LONG_CONTROL_COMMAND_TIMEOUT_SECONDS = 30 * 60
_KUBERNETES_NAMESPACE_ENV = "FLAKEGRAPH_APP_KUBERNETES_NAMESPACE"
_KUBERNETES_SECRET_SENTINEL = "configured-in-kubernetes"
_REQUIRED_WORKER_COMPONENTS = frozenset({"worker-prepare", "worker-extract", "worker-finalize"})
_VLLM_MIN_SERVE_ARGUMENTS = 2
_ENVIRONMENT_PLACEHOLDER_PATTERN = re.compile(r"\$\{[A-Z_][A-Z0-9_]*\}")
_ARTIFACT_ENVIRONMENT_NAMES = (
    "KG_DISTRIBUTED_ARTIFACT_URI",
    "KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL",
    "KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID",
    "KG_DISTRIBUTED_ARTIFACT_SECRET_ACCESS_KEY",
    "KG_DISTRIBUTED_ARTIFACT_REGION",
)
_DATABASE_ENVIRONMENT_NAME = "KG_DISTRIBUTED_DATABASE_URL"
_PARSING_ENDPOINT_ENVIRONMENT_NAME = "KG_MINERU_API_URL"
_POSTGRES_DEFAULT_PORT = 5432
_PORT_FORWARD_STARTUP_SECONDS = 15.0
_KUBERNETES_SERVICE_NAMESPACE_LABELS = 2
_KUBERNETES_DNS_MIN_LABELS = 3
_STALE_HISTORY_WARNING = (
    "Showing this host's own record of fleet runs, not the fleet's: {reason}. "
    "Graphs submitted from elsewhere are missing, and the states shown may be out of date."
)
_STALE_HISTORY_REASON_LIMIT = 200


class KubernetesBackend(LocalBackend):
    """Submit durable distributed runs and inspect their Kubernetes workloads."""

    def _cluster_target(self) -> ClusterTarget:
        """Return how kubectl must be pointed at the selected cluster."""

        catalog = read_catalog(self.state_root)
        active = catalog.active()
        return active.target() if active else ClusterTarget()

    def _cluster_environment(self) -> dict[str, str]:
        """Return the environment targeting the selected cluster.

        The selection is authoritative over the ambient environment: a cluster
        chosen in the app must be the one commands run against, or the selector
        would only appear to work.
        """

        catalog = read_catalog(self.state_root)
        active = catalog.active()
        return active.environment() if active else {}

    @property
    def namespace(self) -> str:
        """Return the namespace of the cluster this backend acts on.

        Fleet reads are confined to it, so the pages that display a namespace and
        the commands that use one must resolve it the same way.
        """

        return self._namespace()

    def _namespace(self) -> str:
        """Return the selected cluster's namespace, or the ambient default."""

        return self._cluster_environment().get(
            _KUBERNETES_NAMESPACE_ENV,
            os.environ.get(_KUBERNETES_NAMESPACE_ENV, "flakegraph"),
        )

    @property
    def capabilities(self) -> frozenset[str]:
        """Add fleet diagnostics and durable cancellation to local source controls."""

        return super().capabilities | {
            "cluster",
            "distributed",
            "cancel",
            "model_serving",
            "forget",
            "node_assignments",
            "recover",
            "retry",
        }

    def graph_embedding_width(self) -> int | None:
        """Report no fixed width: this destination stores vectors as written."""

        return None

    def fleet_profile(self) -> Mapping[str, Any]:
        """Return the processing profile the deployed workers actually mount.

        The ingestion form uses this to default its provider fields to what the
        fleet runs. A worker only claims a run whose semantic configuration
        equals its own, so a form defaulting to anything else asks the operator
        to reconcile two configurations by hand before their first submission —
        and reports it as three separate errors when they do not.

        Returns an empty mapping when the fleet cannot be read. Defaults are a
        convenience; preflight remains the authority on whether a run matches.
        """

        try:
            target = self._cluster_target()
            namespace = self._namespace()
            config_maps = _kubectl_json(
                target=target,
                arguments=[
                    "get",
                    "configmap",
                    "-n",
                    namespace,
                    "-l",
                    "app.kubernetes.io/name=flakegraph",
                    "-o",
                    "json",
                ],
            ).get("items", [])
        except Exception:
            return {}
        for item in config_maps:
            data = item.get("data", {})
            if isinstance(data, Mapping) and "config.yaml" in data:
                try:
                    return _load_deployed_profile(item)
                except RuntimeError:
                    return {}
        return {}

    def fleet_ocr_options(self) -> Mapping[str, Any]:
        """Return the OCR routing a run must carry to be claimed by these workers.

        The form offers the OCR provider, but not the composition beneath it: for
        the fallback adapter, which concrete parser reads a page the primary
        cannot. That composition is part of the digest a worker claims by, so a
        run keeping whatever the base profile happens to say is claimed by
        nothing and sits queued with nothing on the page to explain it. The base
        profile is a local default — a worker parses through the deployed
        document plane instead — and the fleet is the only authority on which.

        The parsing endpoint is read from the worker contract rather than the
        mounted profile because the chart supplies it through the environment,
        where it also overrides any value the profile carries. It is excluded
        from the digest, so it is transport, not identity: it travels only so a
        submitted run states the endpoint its own preflight requires.

        Returns an empty mapping when the fleet cannot be read. These are
        defaults; preflight remains the authority on whether a run matches.
        """

        section = self.fleet_profile().get("ocr")
        options: dict[str, Any] = {}
        if isinstance(section, Mapping):
            options = {
                str(key): value
                for key, value in section.items()
                if str(key) != "provider"
                and str(key) not in _DEPLOYMENT_LOCAL_KEYS["ocr"]
                and not sensitive_config_key(str(key))
            }
        if not options:
            return {}
        with suppress(Exception):
            endpoint = self._fleet_parsing_endpoint()
            if endpoint:
                options["mineru_api_url"] = endpoint
        return options

    def _fleet_parsing_endpoint(self) -> str | None:
        """Read the parsing endpoint literal the deployed workers are given."""

        deployments = _kubectl_json(
            target=self._cluster_target(),
            arguments=[
                "get",
                "deployments",
                "-n",
                self._namespace(),
                "-l",
                "app.kubernetes.io/name=flakegraph",
                "-o",
                "json",
            ],
        ).get("items", [])
        for item in deployments:
            if not str(_component(item)).startswith("worker-"):
                continue
            containers = (
                item.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            )
            for container in containers:
                for entry in container.get("env", []):
                    if str(entry.get("name")) != _PARSING_ENDPOINT_ENVIRONMENT_NAME:
                        continue
                    value = str(entry.get("value") or "").strip()
                    if value:
                        return value
        return None

    def preflight(self, request: IngestionRequest) -> Mapping[str, object]:
        """Validate the submit host and the fleet without requiring worker packages.

        Source discovery and artifact staging execute beside Streamlit, so ordinary
        path and configuration checks still run locally. OCR, extraction, and
        embeddings execute in Kubernetes; their binaries and Python packages are
        therefore validated by the deployed-image contract rather than the Mac's
        optional extras.
        """

        config_path = write_run_config(
            request,
            self._config_path(_validated_run_id(request.job_id)),
        )
        environment = _orchestrator_environment(request)
        result = subprocess.run(
            [
                "uv",
                "run",
                "flakegraph",
                "preflight",
                "--config",
                str(config_path),
                "--orchestrator",
            ],
            cwd=self.repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        payload = _last_json_object(result.stdout) or {
            "ok": False,
            "checks": [],
            "errors": [result.stderr.strip() or "Coordinator preflight returned no JSON result"],
        }
        namespace = self._namespace()
        try:
            fleet = _fleet_preflight(request, namespace, self._cluster_target())
        except Exception as exc:
            fleet = {"ok": False, "checks": [], "errors": [str(exc)]}
        checks = [str(item) for item in payload.get("checks", [])]
        checks.extend(_message_items(fleet, "checks"))
        errors = [str(item) for item in payload.get("errors", [])]
        errors.extend(_message_items(fleet, "errors"))
        if result.returncode and not errors:
            errors.append(result.stderr.strip() or "Coordinator preflight failed")
        return {"ok": not errors, "checks": checks, "errors": errors}

    def submit(self, request: IngestionRequest) -> RunSnapshot:
        """Submit a task graph to PostgreSQL through the existing distributed CLI."""

        config_path = write_run_config(
            request,
            self._config_path(_validated_run_id(request.job_id)),
        )
        payload = self._run_json(
            [
                "uv",
                "run",
                "flakegraph",
                "distributed",
                "submit",
                "--config",
                str(config_path),
                "--run-id",
                request.job_id,
            ],
            _orchestrator_environment(request),
        )
        snapshot = replace(
            _distributed_snapshot(payload, config_path),
            graph_name=graph_name(self.state_root, request.graph_id),
            output_path=str(request.output.workspace_path),
            storage_kind=request.output.kind,
            storage_location=request.output.location,
        )
        self._record_snapshot(snapshot, config_path)
        return snapshot

    def list_runs(self, *, limit: int = 100) -> Sequence[RunSnapshot]:
        """Return durable PostgreSQL history when a fleet profile is available."""

        self.listing_warning = None
        hidden = hidden_run_ids(self.state_root)
        catalog = [
            self._catalog_snapshot(
                run_directory,
                record,
                durable_runtime=True,
                include_progress=False,
            )
            for run_directory, record in list_run_records(self.state_root, limit)
            if str(record.get("runtime")) == "kubernetes"
            and str(record.get("run_id") or run_directory.name) not in hidden
        ]
        config_path = self._fleet_config_path(catalog)
        if config_path is None:
            self.listing_warning = _STALE_HISTORY_WARNING.format(
                reason="no fleet profile could be found on this host"
            )
            return catalog
        try:
            payload = self._run_json(
                [
                    "uv",
                    "run",
                    "flakegraph",
                    "distributed",
                    "list",
                    "--config",
                    str(config_path),
                    "--limit",
                    str(limit),
                ]
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            # Falling back keeps the sidebar populated when the fleet is briefly
            # unreachable, but the two sources are not interchangeable: the local
            # catalog holds only runs this host submitted, and a copied catalog
            # may hold none that exist. Substituting one for the other without
            # saying so presents another machine's history as this fleet's.
            self.listing_warning = _STALE_HISTORY_WARNING.format(reason=_bounded_reason(str(exc)))
            return catalog
        records = {snapshot.run_id: snapshot for snapshot in catalog}
        return [
            replace(
                _distributed_overview_snapshot(
                    item,
                    self._run_config_path(str(item.get("id") or "unknown"), config_path),
                    records.get(str(item.get("id"))),
                    _run_output_path(
                        self.repository_root,
                        str(item.get("id") or "unknown"),
                    ),
                ),
                graph_name=graph_name(self.state_root, str(item.get("graph_id") or "")),
            )
            for item in payload.get("runs", [])
            if str(item.get("id") or "") not in hidden
        ]

    def status(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Query the bounded distributed status contract without loading task payloads."""

        path = self._run_config_path(run_id, config_path)
        payload = self._run_json(
            [
                "uv",
                "run",
                "flakegraph",
                "distributed",
                "status",
                "--config",
                str(path),
                "--run-id",
                run_id,
            ]
        )
        record = read_run_record(self.state_root / "runs" / run_id)
        snapshot = replace(
            _distributed_snapshot(payload, path),
            graph_name=graph_name(
                self.state_root,
                str((payload.get("run") or {}).get("graph_id") or ""),
            ),
            output_path=(
                str(record.get("output_path") or "")
                or str(self.repository_root / "out" / "app" / run_id)
            ),
            storage_kind=_storage_kind(record.get("storage_kind")),
            storage_location=(
                str(record.get("storage_location"))
                if record.get("storage_location")
                else str(record.get("output_path") or "") or None
            ),
        )
        self._record_snapshot(snapshot, path)
        return snapshot

    def cancel(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Cancel queued and running distributed tasks through the coordination store."""

        path = self._run_config_path(run_id, config_path)
        self._run_json(
            [
                "uv",
                "run",
                "flakegraph",
                "distributed",
                "cancel",
                "--config",
                str(path),
                "--run-id",
                run_id,
            ]
        )
        return self.status(run_id, path)

    def retry(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Requeue failed fleet work while retaining every successful prerequisite.

        The distributed store resets only terminally failed tasks or publications.
        Completed tasks and immutable artifacts remain intact, which makes this an
        inexpensive recovery operation rather than a duplicate corpus ingestion.
        """

        path = self._run_config_path(run_id, config_path)
        self._run_json(
            [
                "uv",
                "run",
                "flakegraph",
                "distributed",
                "retry",
                "--config",
                str(path),
                "--run-id",
                run_id,
            ]
        )
        return self.status(run_id, path)

    def recover(self, run_id: str, config_path: Path | None = None) -> str:
        """Reconcile worker deployments required by this run's queued stages.

        Queued tasks are already durable and must not be duplicated. Recovery
        therefore validates service accounts, node placement, rollout failures,
        and pod startup before nudging only an unavailable worker pool. Healthy
        pools are left untouched because they may be serving another graph.
        """

        snapshot = self.status(run_id, self._run_config_path(run_id, config_path))
        components = _queued_worker_components(snapshot)
        if not components:
            return "No queued processing stage currently requires infrastructure recovery."
        namespace = self._namespace()
        return _recover_kubernetes_workers(namespace, components, self._cluster_target())

    def forget(self, run_id: str, config_path: Path | None = None) -> None:
        """Hide a terminal durable run while preserving its audit and graph data."""

        path = self._run_config_path(run_id, config_path)
        snapshot = self.status(run_id, path)
        if snapshot.status.lower() in {"running", "queued", "planning", "processing"}:
            raise ValueError("An active run must be cancelled before removal")
        hide_run(self.state_root, run_id)
        run_directory = (self.state_root / "runs" / run_id).resolve()
        if run_directory.parent != (self.state_root / "runs").resolve():
            raise ValueError("Refusing to remove a run outside the app state directory")
        shutil.rmtree(run_directory, ignore_errors=True)

    def load_run_graph(
        self,
        snapshot: RunSnapshot,
        config_path: Path | None = None,
    ) -> GraphDataset:
        """Export a completed distributed artifact once, then load it locally."""

        path = self._run_config_path(snapshot.run_id, config_path)
        if snapshot.storage_kind == StorageKind.SNOWFLAKE:
            return super().load_run_graph(snapshot, path)
        output = Path(
            snapshot.output_path or self.repository_root / "out" / "app" / snapshot.run_id
        )
        if not (output / "nodes.parquet").is_file() or not (output / "edges.parquet").is_file():
            namespace = self._namespace()
            with _artifact_export_environment(namespace, self._cluster_target()) as environment:
                self._run_json(
                    [
                        "uv",
                        "run",
                        "flakegraph",
                        "distributed",
                        "export",
                        "--config",
                        str(path),
                        "--run-id",
                        snapshot.run_id,
                        "--output",
                        str(output),
                    ],
                    environment,
                )
        return load_local_graph(output)

    def cluster(self, namespace: str) -> ClusterSnapshot:
        """Return fleet telemetry with independent Kubernetes reads in parallel."""

        namespace = _validated_fleet_namespace(namespace, self._namespace())
        target = self._cluster_target()
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="flakegraph-fleet") as pool:
            context_future = pool.submit(
                _kubectl_json,
                arguments=["config", "current-context"],
                target=target,
                raw=True,
            )
            nodes_future = pool.submit(
                _kubectl_json, arguments=["get", "nodes", "-o", "json"], target=target
            )
            pods_future = pool.submit(
                _kubectl_json,
                arguments=[
                    "get",
                    "pods",
                    "-n",
                    namespace,
                    "-l",
                    "app.kubernetes.io/name=flakegraph",
                    "-o",
                    "json",
                ],
                target=target,
            )
            pod_metrics_future = pool.submit(
                _kubectl_json,
                arguments=["top", "pods", "-n", namespace, "--no-headers"],
                target=target,
                raw=True,
            )
            node_metrics_future = pool.submit(
                _kubectl_json,
                arguments=["top", "nodes", "--no-headers"],
                target=target,
                raw=True,
            )
        context = str(context_future.result()).strip()
        nodes = nodes_future.result()
        pods = pods_future.result()
        pod_metrics: dict[str, Mapping[str, Any]] = {}
        node_metrics: dict[str, Mapping[str, Any]] = {}
        warnings: list[str] = []
        try:
            raw_metrics = pod_metrics_future.result()
            for line in raw_metrics.splitlines():
                columns = line.split()
                if len(columns) >= _KUBECTL_METRIC_COLUMNS:
                    pod_metrics[columns[0]] = {"cpu": columns[1], "memory": columns[2]}
        except RuntimeError:
            warnings.append("Pod CPU and memory metrics are unavailable; install metrics-server.")
        try:
            raw_node_metrics = node_metrics_future.result()
            for line in raw_node_metrics.splitlines():
                columns = line.split()
                if len(columns) >= _KUBECTL_NODE_METRIC_COLUMNS:
                    node_metrics[columns[0]] = {
                        "cpu": columns[1],
                        "cpu_percent": columns[2],
                        "memory": columns[3],
                        "memory_percent": columns[4],
                    }
        except RuntimeError:
            warnings.append("Node CPU and memory metrics are unavailable; install metrics-server.")

        node_items = nodes.get("items", [])
        workloads = [_pod_status(item, pod_metrics) for item in pods.get("items", [])]
        gpu_metrics, gpu_warning = _gpu_utilization_by_node(namespace, workloads, target)
        if gpu_warning:
            warnings.append(gpu_warning)
        node_statuses = [
            _node_status(item, workloads, node_metrics, gpu_metrics) for item in node_items
        ]
        failed = [item.name for item in workloads if item.phase not in {"Running", "Succeeded"}]
        if failed:
            warnings.append(f"Workloads needing attention: {', '.join(failed[:6])}")
        return ClusterSnapshot(
            context=context,
            namespace=namespace,
            nodes_ready=sum(_node_ready(item) for item in node_items),
            nodes_total=len(node_items),
            nodes=node_statuses,
            workloads=workloads,
            warnings=warnings,
        )

    def node_assignments(
        self,
        namespace: str,
        node_name: str,
    ) -> Sequence[NodeWorkAssignment]:
        """Query only currently leased tasks for worker pods on one selected node.

        The fleet overview deliberately avoids task-ledger reads. This method is
        called only from an explicit node-detail action and selects bounded identity
        columns without loading task payloads or artifacts.
        """

        namespace = _validated_fleet_namespace(namespace, self._namespace())
        target = self._cluster_target()
        pods = _kubectl_json(
            target=target,
            arguments=[
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                "app.kubernetes.io/name=flakegraph",
                "--field-selector",
                f"spec.nodeName={node_name}",
                "-o",
                "json",
            ],
        )
        worker_ids = sorted(
            str(item.get("metadata", {}).get("name"))
            for item in pods.get("items", [])
            if str(
                item.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component", "")
            ).startswith("worker")
            and item.get("status", {}).get("phase") == "Running"
        )
        if not worker_ids:
            return []
        with (
            _fleet_database_environment(namespace, self._cluster_target()) as environment,
            connect(environment[_DATABASE_ENVIRONMENT_NAME], connect_timeout=5) as connection,
        ):
            rows = connection.execute(
                """
                SELECT task.lease_owner, task.run_id, run.graph_id, task.id,
                       task.stage, task.scope_id, task.updated_at
                FROM flakegraph_task AS task
                JOIN flakegraph_run AS run ON run.id = task.run_id
                WHERE task.status = 'running'
                  AND task.lease_owner = ANY(%s)
                ORDER BY task.lease_owner, task.updated_at DESC
                LIMIT 100
                """,
                (worker_ids,),
            ).fetchall()
        return [
            NodeWorkAssignment(
                worker_id=str(row[0]),
                run_id=str(row[1]),
                graph_id=str(row[2]),
                task_id=str(row[3]),
                stage=str(row[4]),
                scope_id=str(row[5]),
                updated_at=str(row[6]) if row[6] else None,
            )
            for row in rows
        ]

    def _run_json(
        self,
        command: Sequence[str],
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run one bounded CLI command and return its JSON result or a clear failure."""

        namespace = self._namespace()
        timeout = (
            _LONG_CONTROL_COMMAND_TIMEOUT_SECONDS
            if any(part in {"submit", "export"} for part in command)
            else _CONTROL_COMMAND_TIMEOUT_SECONDS
        )
        # Overlay, never replace: a subprocess still needs PATH and HOME to run.
        merged = {**os.environ, **self._cluster_environment(), **(environment or {})}
        with _fleet_database_environment(
            namespace,
            self._cluster_target(),
            merged,
        ) as resolved_environment:
            result = subprocess.run(
                list(command),
                cwd=self.repository_root,
                env=resolved_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        payload = _last_json_object(result.stdout)
        if result.returncode:
            raise RuntimeError(_command_failure_message(result.stderr, result.stdout))
        if not payload:
            raise RuntimeError("FlakeGraph distributed command returned no JSON object")
        return payload

    def _record_snapshot(self, snapshot: RunSnapshot, config_path: Path) -> None:
        """Persist enough fleet metadata to rebuild the app sidebar after restart."""

        run_id = _validated_run_id(snapshot.run_id)
        write_run_record(
            self.state_root / "runs" / run_id,
            {
                "run_id": snapshot.run_id,
                "graph_id": snapshot.graph_id,
                "graph_name": snapshot.graph_name,
                "runtime": "kubernetes",
                "status": snapshot.status,
                "started_at": snapshot.started_at,
                "updated_at": snapshot.updated_at,
                "config_path": str(config_path),
                "output_path": snapshot.output_path,
                "error": snapshot.error,
                "storage_kind": snapshot.storage_kind.value,
                "storage_location": snapshot.storage_location,
            },
        )

    def _fleet_config_path(self, catalog: Sequence[RunSnapshot]) -> Path | None:
        """Resolve the profile used to connect the app to the durable fleet store."""

        configured = os.environ.get("FLAKEGRAPH_APP_KUBERNETES_CONFIG", "").strip()
        candidates = [configured] if configured else []
        candidates.extend(str(item.raw.get("config_path") or "") for item in catalog)
        candidates.append(str(self.repository_root / "configs" / "app-defaults.yaml"))
        for value in candidates:
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.repository_root / path
            if path.is_file():
                return path.resolve()
        return None

    def _run_config_path(self, run_id: str, fallback: Path | None = None) -> Path:
        """Prefer the immutable run-owned profile over a fleet discovery profile.

        A shared profile may list every run because database connectivity is common,
        but status diagnostics and retry compatibility must use the exact generated
        profile belonging to the selected run. The fallback supports older remote
        entries whose app-owned profile is no longer present on this host.
        """

        owned = self._config_path(_validated_run_id(run_id))
        return owned if owned.is_file() else (fallback or owned)


def _distributed_snapshot(payload: Mapping[str, Any], config_path: Path) -> RunSnapshot:
    """Translate the bounded distributed diagnostics shape into the common UI model."""

    run = payload.get("run") or {}
    counts = payload.get("task_counts") or []
    stages = _distributed_stages(counts)
    diagnostics = payload.get("diagnostics") or {}
    warnings = [_diagnostic_warning(item) for item in diagnostics.get("warnings", [])]
    documents_total, documents_completed, documents_failed = _distributed_document_counts(counts)
    return RunSnapshot(
        run_id=str(run.get("id") or run.get("run_id") or "unknown"),
        graph_id=str(run.get("graph_id") or "unknown"),
        status=str(run.get("status", "unknown")),
        started_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        stages=stages,
        documents_total=documents_total,
        documents_completed=documents_completed,
        documents_failed=documents_failed,
        warnings=warnings,
        error=_distributed_error_message(payload.get("error")),
        raw={**dict(payload), "config_path": str(config_path)},
    )


def _validated_run_id(value: str) -> str:
    """Reject path traversal and platform-hostile durable run identifiers."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("run ID must be a safe 1-128 character identifier")
    return value


def _run_output_path(repository_root: Path, run_id: str) -> Path:
    """Return an app-owned per-run output path after validating its segment."""

    return repository_root / "out" / "app" / _validated_run_id(run_id)


def _diagnostic_warning(value: object) -> str:
    """Combine one bounded warning and its remediation for the run workspace."""

    if not isinstance(value, Mapping):
        return str(value)
    message = str(value.get("message") or value)
    remediation = str(value.get("remediation") or "").strip()
    return f"{message} {remediation}" if remediation else message


def _queued_worker_components(snapshot: RunSnapshot) -> set[str]:
    """Map queued pipeline stages to the worker pools capable of claiming them."""

    components: set[str] = set()
    for stage in snapshot.stages:
        if stage.status.lower() != "queued":
            continue
        if stage.stage == "prepare_document":
            components.add("worker-prepare")
        elif stage.stage == "finalize_graph":
            components.add("worker-finalize")
        else:
            components.add("worker-extract")
    return components


def _validated_fleet_namespace(namespace: str, allowed: str) -> str:
    """Constrain UI-supplied fleet access to the selected cluster's namespace.

    ``allowed`` is the namespace of the cluster the operator chose, so registering
    a cluster on a namespace of its own does not lock the fleet page out.
    """

    if namespace != allowed:
        raise RuntimeError(f"Fleet access is restricted to Kubernetes namespace {allowed}")
    return allowed


def _recover_kubernetes_workers(
    namespace: str,
    components: set[str],
    target: ClusterTarget,
) -> str:
    """Validate and nudge unavailable worker pools without touching task state.

    A healthy pool may be occupied by another graph, so it is deliberately left
    alone. An unhealthy pool is restarted only after static prerequisites pass;
    deterministic failures such as a missing service account or image pull error
    are surfaced for operator correction instead of entering a restart loop.
    """
    namespace = _validated_fleet_namespace(namespace, namespace)

    deployments = _kubectl_json(
        target=target,
        arguments=[
            "get",
            "deployments",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=flakegraph",
            "-o",
            "json",
        ],
    ).get("items", [])
    by_component = {_component(item): item for item in deployments if _component(item)}
    service_accounts = {
        str(item.get("metadata", {}).get("name", ""))
        for item in _kubectl_json(
            target=target, arguments=["get", "serviceaccounts", "-n", namespace, "-o", "json"]
        ).get("items", [])
    }
    nodes = _kubectl_json(target=target, arguments=["get", "nodes", "-o", "json"]).get("items", [])
    pods = _kubectl_json(
        target=target,
        arguments=[
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=flakegraph",
            "-o",
            "json",
        ],
    ).get("items", [])
    pods_by_component: dict[str, list[Mapping[str, Any]]] = {}
    for pod in pods:
        pods_by_component.setdefault(_component(pod), []).append(pod)

    issues: list[str] = []
    unavailable: list[str] = []
    healthy: list[str] = []
    reconciling: list[str] = []
    for component in sorted(components):
        deployment = by_component.get(component)
        if deployment is None:
            issues.append(f"Kubernetes fleet is missing the {component} deployment")
            continue
        component_pods = pods_by_component.get(component, [])
        issues.extend(
            _worker_recovery_issues(
                deployment,
                component_pods,
                service_accounts,
                nodes,
            )
        )
        available = int(deployment.get("status", {}).get("availableReplicas", 0) or 0)
        if available > 0:
            healthy.append(component)
        elif any(
            pod.get("status", {}).get("phase") in {"Pending", "Running"} for pod in component_pods
        ):
            reconciling.append(component)
        else:
            unavailable.append(component)
    if issues:
        raise RuntimeError(" ".join(dict.fromkeys(issues)))

    restarted: list[str] = []
    for component in unavailable:
        deployment = by_component[component]
        name = str(deployment.get("metadata", {}).get("name", component))
        desired = int(deployment.get("spec", {}).get("replicas", 0) or 0)
        if desired == 0:
            _kubectl_json(
                target=target,
                arguments=["scale", "deployment", name, "-n", namespace, "--replicas=1"],
                raw=True,
            )
        else:
            _kubectl_json(
                target=target,
                arguments=["rollout", "restart", f"deployment/{name}", "-n", namespace],
                raw=True,
            )
        restarted.append(component)
    if restarted:
        return (
            f"Restarted unavailable worker pools: {', '.join(restarted)}. "
            "Queued work remains durable and will resume automatically."
        )
    if reconciling:
        return (
            f"Kubernetes is already reconciling worker pools: {', '.join(reconciling)}. "
            "No tasks or active workers were changed; queued work resumes automatically."
        )
    return (
        f"Required worker pools are healthy: {', '.join(healthy)}. This run is "
        "waiting for shared capacity and will resume automatically."
    )


def _worker_recovery_issues(
    deployment: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    service_accounts: set[str],
    nodes: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return deterministic rollout blockers that restarting cannot repair."""

    name = str(deployment.get("metadata", {}).get("name", "worker deployment"))
    pod_spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
    service_account = str(pod_spec.get("serviceAccountName") or "default")
    issues = []
    if service_account not in service_accounts:
        issues.append(f"{name} references missing service account {service_account}.")
    selector = pod_spec.get("nodeSelector") or {}
    if selector and not any(
        _node_ready(node) and _labels_match(node.get("metadata", {}).get("labels", {}), selector)
        for node in nodes
    ):
        issues.append(f"{name} has no ready node matching selector {dict(selector)}.")
    for condition in deployment.get("status", {}).get("conditions", []):
        if condition.get("type") == "ReplicaFailure" and condition.get("status") == "True":
            message = str(condition.get("message") or condition.get("reason") or "replica failure")
            issues.append(f"{name} cannot create a worker pod: {message}")
    for pod in pods:
        statuses = [
            *pod.get("status", {}).get("initContainerStatuses", []),
            *pod.get("status", {}).get("containerStatuses", []),
        ]
        for status in statuses:
            waiting = status.get("state", {}).get("waiting") or {}
            reason = str(waiting.get("reason") or "")
            if reason in {"CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff"}:
                message = str(waiting.get("message") or reason)
                issues.append(f"{name} pod startup failed: {message}")
    return issues


def _labels_match(labels: Mapping[str, object], selector: Mapping[str, object]) -> bool:
    """Return whether all exact-match node selector labels are present."""

    return all(str(labels.get(key)) == str(value) for key, value in selector.items())


def _distributed_error_message(value: object) -> str | None:
    """Render one bounded redacted run failure as an actionable operator message.

    Distributed workers persist errors after passing them through the shared
    redaction layer. The UI keeps the failure type and task identity because both
    materially help an operator correct the cause before selecting retry.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        text = str(value).strip()
        return text or None
    error_type = str(value.get("error_type") or value.get("type") or "Run failure")
    message = _distributed_error_summary(
        str(value.get("error_message") or value.get("message") or "")
    )
    task_id = str(value.get("task_id") or value.get("publication_id") or "").strip()
    context = f"Task {task_id}" if task_id else "Distributed run"
    if message:
        return f"{context} failed with {error_type}: {message}"
    return f"{context} failed with {error_type}."


def _distributed_error_summary(message: str, *, limit: int = 500) -> str:
    """Extract one operator-facing cause from a potentially enormous traceback.

    Java and remote-provider exceptions often stringify as a short wrapper, a
    useful server ``Message:``, and hundreds of stack frames. Run status should
    lead with the server's cause instead of expanding the page with transport
    internals. The complete redacted payload remains in ``RunSnapshot.raw`` for
    the terminal page's opt-in technical-details panel.
    """

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    flattened = " ".join(lines)
    server_message = re.search(
        r"\bMessage:\s*(.+?)(?=\s+Received status:|\s+at\s+[\w.$]+\(|$)",
        flattened,
    )
    summary = server_message.group(1).strip() if server_message else lines[0]
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 1].rstrip()}…"


def _distributed_overview_snapshot(
    overview: Mapping[str, Any],
    config_path: Path,
    catalog_snapshot: RunSnapshot | None,
    default_output_path: Path,
) -> RunSnapshot:
    """Translate one bounded distributed run overview for sidebar navigation."""

    run_id = str(overview.get("id") or "unknown")
    output_path = (
        catalog_snapshot.output_path
        if catalog_snapshot and catalog_snapshot.output_path
        else str(default_output_path)
    )
    counts = overview.get("task_counts") or []
    documents_total, documents_completed, documents_failed = _distributed_document_counts(counts)
    return RunSnapshot(
        run_id=run_id,
        graph_id=str(overview.get("graph_id") or run_id),
        status=str(overview.get("status") or "unknown"),
        started_at=str(overview.get("created_at")) if overview.get("created_at") else None,
        updated_at=str(overview.get("updated_at")) if overview.get("updated_at") else None,
        stages=_distributed_stages(counts),
        documents_total=documents_total,
        documents_completed=documents_completed,
        documents_failed=documents_failed,
        output_path=output_path,
        storage_kind=(catalog_snapshot.storage_kind if catalog_snapshot else StorageKind.LOCAL),
        storage_location=(catalog_snapshot.storage_location if catalog_snapshot else output_path),
        raw={"config_path": str(config_path)},
    )


def _distributed_stages(counts: Sequence[Mapping[str, Any]]) -> list[StageProgress]:
    """Aggregate distributed task buckets into one status row per pipeline stage."""

    grouped: dict[str, dict[str, int]] = {}
    task_progress: dict[str, Mapping[str, Any]] = {}
    started: dict[str, list[datetime]] = {}
    completed: dict[str, list[datetime]] = {}
    for item in counts:
        stage = str(item.get("stage", "unknown"))
        status = str(item.get("status", "unknown"))
        grouped.setdefault(stage, {})[status] = int(item.get("count", 0))
        progress = item.get("progress")
        if isinstance(progress, Mapping):
            task_progress[stage] = progress
        if started_at := _parse_timestamp(item.get("started_at")):
            started.setdefault(stage, []).append(started_at)
        if completed_at := _parse_timestamp(item.get("completed_at")):
            completed.setdefault(stage, []).append(completed_at)
    stages = []
    observed_at = datetime.now(UTC)
    for stage, statuses in grouped.items():
        total = sum(statuses.values())
        succeeded = statuses.get("succeeded", 0)
        state = (
            "failed"
            if statuses.get("failed")
            else (
                "running"
                if statuses.get("running")
                else ("completed" if total == succeeded else "queued")
            )
        )
        stage_started = min(started.get(stage, []), default=None)
        stage_completed = max(completed.get(stage, []), default=None)
        end = observed_at if state == "running" else stage_completed
        elapsed_ms = (
            max(0, round((end - stage_started).total_seconds() * 1000))
            if stage_started is not None and end is not None
            else 0
        )
        displayed_completed = succeeded
        displayed_total = total
        message = None
        progress = task_progress.get(stage)
        if state == "running" and progress is not None:
            phase_index = max(1, int(progress.get("phase_index") or 1))
            phase_total = max(phase_index, int(progress.get("phase_total") or phase_index))
            inner_completed = int(progress.get("completed") or 0)
            inner_total = int(progress.get("total") or 0)
            phase_finished = bool(inner_total and inner_completed >= inner_total)
            displayed_completed = min(
                phase_total,
                phase_index - 1 + int(phase_finished),
            )
            displayed_total = phase_total
            message = str(progress.get("message") or "") or None
            if message and inner_total > 1:
                message = f"{message} · {inner_completed}/{inner_total}"
        stages.append(
            StageProgress(
                stage=stage,
                status=state,
                completed=displayed_completed,
                total=displayed_total,
                elapsed_ms=elapsed_ms,
                message=message,
            )
        )
    stage_order = {stage: index for index, stage in enumerate(DISTRIBUTED_STAGE_ORDER)}
    return sorted(stages, key=lambda item: stage_order.get(item.stage, len(stage_order)))


def _parse_timestamp(value: object) -> datetime | None:
    """Parse optional ISO timestamps returned by bounded distributed status calls."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _distributed_document_counts(
    counts: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int, int]:
    """Derive corpus-level counters from durable per-document task stages.

    Every discovered document owns one ``prepare_document`` task, so that stage
    defines the total even while later dynamic work is still being planned. A
    document is complete only after ``compact_document`` succeeds; preparation
    alone must not make the top-level summary claim that ingestion is finished.
    """

    grouped: dict[str, dict[str, int]] = {}
    for item in counts:
        stage = str(item.get("stage", "unknown"))
        status = str(item.get("status", "unknown"))
        grouped.setdefault(stage, {})[status] = int(item.get("count", 0))
    preparation = grouped.get("prepare_document", {})
    compaction = grouped.get("compact_document", {})
    total = sum(preparation.values()) or None
    completed = compaction.get("succeeded", 0)
    failed = preparation.get("failed", 0) + compaction.get("failed", 0)
    return total, completed, failed


def _kubectl_json(
    arguments: Sequence[str],
    *,
    target: ClusterTarget,
    raw: bool = False,
) -> Any:
    """Execute a bounded kubectl call and parse JSON unless raw output is requested.

    The cluster is selected per call rather than through the ambient environment,
    so a read always reaches the cluster the operator chose.

    A stale VPN route or unreachable API server must not leave the Streamlit process
    waiting indefinitely. The timeout applies to each small inventory request and is
    reported through the same operator-facing error path as other kubectl failures.
    """

    command = ["kubectl", *target.arguments(), *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_KUBECTL_TIMEOUT_SECONDS,
            env=target.environment(os.environ),
        )
    except subprocess.TimeoutExpired as exc:
        rendered = " ".join(command)
        raise RuntimeError(f"Kubernetes did not respond within 15 seconds: {rendered}") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "kubectl command failed")
    return result.stdout if raw else json.loads(result.stdout)


def _node_ready(node: Mapping[str, Any]) -> bool:
    """Return whether Kubernetes reports its Ready condition as true."""

    conditions = node.get("status", {}).get("conditions", [])
    return any(item.get("type") == "Ready" and item.get("status") == "True" for item in conditions)


def _pod_status(
    pod: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> WorkloadStatus:
    """Normalize one pod and optional metrics-server observation."""

    metadata = pod.get("metadata", {})
    status = pod.get("status", {})
    spec = pod.get("spec", {})
    name = str(metadata.get("name", "unknown"))
    container_statuses = status.get("containerStatuses", [])
    usage = metrics.get(name, {})
    containers = spec.get("containers", [])
    container = containers[0] if containers else {}
    component = str(metadata.get("labels", {}).get("app.kubernetes.io/component", "worker"))
    return WorkloadStatus(
        name=name,
        component=component,
        phase=str(status.get("phase", "Unknown")),
        node=str(spec.get("nodeName")) if spec.get("nodeName") else None,
        ready=bool(container_statuses) and all(item.get("ready") for item in container_statuses),
        restarts=sum(int(item.get("restartCount", 0)) for item in container_statuses),
        cpu=str(usage.get("cpu")) if usage.get("cpu") else None,
        memory=str(usage.get("memory")) if usage.get("memory") else None,
        image=str(container.get("image")) if container.get("image") else None,
        model=_served_model(container) if component == "model-serving" else None,
    )


def _node_status(
    node: Mapping[str, Any],
    workloads: Sequence[WorkloadStatus],
    metrics: Mapping[str, Mapping[str, Any]],
    gpu_metrics: Mapping[str, str] | None = None,
) -> NodeStatus:
    """Combine one Kubernetes node with capacity, usage, and pod placement data.

    Kubernetes exposes hardware inventory and live resource usage through separate
    APIs. Keeping their reconciliation here gives Streamlit a small, vendor-neutral
    value object and also lets the page remain useful when Metrics Server or NVIDIA
    feature labels are not installed.
    """

    metadata = node.get("metadata", {})
    status = node.get("status", {})
    labels = metadata.get("labels", {})
    capacity = status.get("capacity", {}) or status.get("allocatable", {})
    name = str(metadata.get("name", "unknown"))
    usage = metrics.get(name, {})
    placed = [item for item in workloads if item.node == name]
    model_servers = [item for item in placed if item.component == "model-serving"]
    active_model_server = next((item for item in model_servers if item.ready), None)
    representative_model_server = active_model_server or (
        model_servers[0] if model_servers else None
    )
    return NodeStatus(
        name=name,
        ready=_node_ready(node),
        node_class=_node_class(labels),
        gpu_count=_resource_count(capacity.get("nvidia.com/gpu")),
        gpu_model=_first_label(
            labels,
            "nvidia.com/gpu.product",
            "nvidia.com/gpu.family",
            "nvidia.com/gpu.machine",
        ),
        gpu_percent=(gpu_metrics or {}).get(name),
        cpu_capacity=_optional_string(capacity.get("cpu")),
        memory_capacity=_optional_string(capacity.get("memory")),
        cpu_usage=_optional_string(usage.get("cpu")),
        cpu_percent=_optional_string(usage.get("cpu_percent")),
        memory_usage=_optional_string(usage.get("memory")),
        memory_percent=_optional_string(usage.get("memory_percent")),
        workload_count=len(placed),
        worker_count=sum(item.component.startswith("worker") for item in placed),
        model_server_ready=active_model_server is not None,
        model=representative_model_server.model if representative_model_server else None,
        model_image=representative_model_server.image if representative_model_server else None,
    )


def _gpu_utilization_by_node(
    namespace: str,
    workloads: Sequence[WorkloadStatus],
    target: ClusterTarget,
) -> tuple[dict[str, str], str | None]:
    """Read optional live GPU utilization from each ready model-serving pod.

    Kubernetes Metrics Server does not expose NVIDIA engine utilization. The model
    container already owns the allocated device and provides ``nvidia-smi``, so a
    bounded read there avoids node-level privileges or another monitoring stack.
    Missing tooling remains an optional telemetry gap rather than a fleet failure.
    """

    candidates = [
        workload
        for workload in workloads
        if workload.component == "model-serving" and workload.ready and workload.node
    ]
    utilization: dict[str, str] = {}
    unavailable = False
    if candidates:
        with ThreadPoolExecutor(
            max_workers=min(8, len(candidates)),
            thread_name_prefix="flakegraph-gpu",
        ) as pool:
            futures = [
                pool.submit(_gpu_utilization, namespace, workload, target)
                for workload in candidates
            ]
        for future in futures:
            node_name, value = future.result()
            if node_name and value is not None:
                utilization[node_name] = value
            else:
                unavailable = True
    warning = (
        "GPU utilization is unavailable for one or more model-serving nodes."
        if unavailable
        else None
    )
    return utilization, warning


def _gpu_utilization(
    namespace: str,
    workload: WorkloadStatus,
    target: ClusterTarget,
) -> tuple[str | None, str | None]:
    """Read one model server's accelerator load for parallel fleet aggregation."""

    try:
        output = str(
            _kubectl_json(
                target=target,
                arguments=[
                    "exec",
                    "-n",
                    namespace,
                    workload.name,
                    "--",
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                raw=True,
            )
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    except (RuntimeError, ValueError):
        return workload.node, None
    if not values:
        return workload.node, None
    return workload.node, f"{round(sum(values) / len(values))}%"


def _fleet_preflight(
    request: IngestionRequest,
    namespace: str,
    target: ClusterTarget,
) -> dict[str, object]:
    """Validate Kubernetes capacity, worker controllers, and processing compatibility.

    Worker deployments are intentionally allowed to sit at zero replicas because
    KEDA activates them from durable queue demand. A zero-replica deployment is only
    accepted when its ScaledObject reports Ready. Provider and model selections must
    match the immutable worker ConfigMap so one run cannot mix semantic profiles.
    """

    checks: list[str] = []
    errors: list[str] = []
    context = str(
        _kubectl_json(target=target, arguments=["config", "current-context"], raw=True)
    ).strip()
    nodes = _kubectl_json(target=target, arguments=["get", "nodes", "-o", "json"]).get("items", [])
    ready_nodes = sum(_node_ready(item) for item in nodes)
    _record_requirement(
        errors,
        checks,
        bool(nodes) and ready_nodes == len(nodes),
        f"Kubernetes context {context} has {ready_nodes}/{len(nodes)} ready nodes",
        f"Kubernetes context {context} has only {ready_nodes}/{len(nodes)} ready nodes",
    )

    deployments = _kubectl_json(
        target=target,
        arguments=[
            "get",
            "deployments",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=flakegraph",
            "-o",
            "json",
        ],
    ).get("items", [])
    by_component = {_component(item): item for item in deployments if _component(item)}
    service_accounts = {
        str(item.get("metadata", {}).get("name", ""))
        for item in _kubectl_json(
            target=target, arguments=["get", "serviceaccounts", "-n", namespace, "-o", "json"]
        ).get("items", [])
    }
    try:
        scaled_objects = _kubectl_json(
            target=target,
            arguments=[
                "get",
                "scaledobjects.keda.sh",
                "-n",
                namespace,
                "-l",
                "app.kubernetes.io/name=flakegraph",
                "-o",
                "json",
            ],
        ).get("items", [])
    except RuntimeError:
        scaled_objects = []
    scaled_by_target = {
        str(item.get("spec", {}).get("scaleTargetRef", {}).get("name", "")): item
        for item in scaled_objects
    }
    config_maps = _validate_worker_deployments(
        by_component,
        service_accounts,
        nodes,
        scaled_by_target,
        checks,
        errors,
    )

    _record_requirement(
        errors,
        checks,
        len(config_maps) == 1,
        f"Worker pools share processing profile: {next(iter(config_maps), 'unknown')}",
        "Worker pools must mount one shared FlakeGraph processing ConfigMap",
    )
    if len(config_maps) == 1:
        config_name = next(iter(config_maps))
        config_map = _kubectl_json(
            target=target,
            arguments=["get", "configmap", config_name, "-n", namespace, "-o", "json"],
        )
        deployed_profile = _load_deployed_profile(config_map)
        profile_errors = _profile_mismatches(request, deployed_profile)
        # The named comparisons above cover the fields an operator chooses in the
        # form. The digest a worker claims by covers entire sections, so the
        # effective configuration is compared too — otherwise a run differing
        # only in a field the form never shows passes preflight and is never
        # claimed.
        with suppress(Exception):
            profile_errors.extend(
                _semantic_mismatches(build_run_config(request), deployed_profile)
            )
        # The ontology is part of the digest a worker claims by, so a run whose
        # ontology differs from the fleet's is never claimed by anything. It does
        # not fail: it sits queued forever with nothing on the page to say why,
        # which is the one outcome preflight exists to prevent.
        profile_errors.extend(
            _ontology_mismatches(request, deployed_profile, namespace, target, deployments)
        )
        errors.extend(profile_errors)
        if not profile_errors:
            checks.append("Selected OCR, LLM, embedding, and ontology match fleet workers")

    if request.llm.provider == "vllm_local" and _loopback_endpoint(request.llm.endpoint):
        stateful_sets = _kubectl_json(
            target=target,
            arguments=[
                "get",
                "statefulsets",
                "-n",
                namespace,
                "-l",
                "app.kubernetes.io/name=flakegraph",
                "-o",
                "json",
            ],
        ).get("items", [])
        model_server = next(
            (item for item in stateful_sets if _component(item) == "model-serving"),
            None,
        )
        desired = int(model_server.get("spec", {}).get("replicas", 0) or 0) if model_server else 0
        ready = (
            int(model_server.get("status", {}).get("readyReplicas", 0) or 0) if model_server else 0
        )
        _record_requirement(
            errors,
            checks,
            model_server is not None and desired > 0 and ready == desired,
            f"Chart-managed vLLM model servers are ready: {ready}/{desired}",
            "Selected vLLM endpoint is node-local, but the fleet has no ready "
            "model-serving StatefulSet",
        )

    _validate_fleet_credentials(request, namespace, by_component, checks, errors, target)
    return {"ok": not errors, "checks": checks, "errors": errors}


def _validate_worker_deployments(
    by_component: Mapping[str, Mapping[str, Any]],
    service_accounts: set[str],
    nodes: Sequence[Mapping[str, Any]],
    scaled_by_target: Mapping[str, Mapping[str, Any]],
    checks: list[str],
    errors: list[str],
) -> set[str]:
    """Validate queue workers and return their mounted processing profiles."""

    config_maps: set[str] = set()
    for component in sorted(_REQUIRED_WORKER_COMPONENTS):
        deployment = by_component.get(component)
        if deployment is None:
            errors.append(f"Kubernetes fleet is missing the {component} deployment")
            continue
        name = str(deployment.get("metadata", {}).get("name", component))
        image = _controller_image(deployment)
        _record_requirement(
            errors,
            checks,
            bool(image),
            f"{component} worker image is configured: {image}",
            f"{component} deployment has no worker image",
        )
        desired = int(deployment.get("spec", {}).get("replicas", 0) or 0)
        pod_spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
        service_account = str(pod_spec.get("serviceAccountName") or "default")
        _record_requirement(
            errors,
            checks,
            service_account in service_accounts,
            f"{component} service account is available: {service_account}",
            f"{component} references missing service account: {service_account}",
        )
        selector = pod_spec.get("nodeSelector") or {}
        _record_requirement(
            errors,
            checks,
            not selector
            or any(
                _node_ready(node)
                and _labels_match(node.get("metadata", {}).get("labels", {}), selector)
                for node in nodes
            ),
            f"{component} has ready node capacity for selector {dict(selector)}",
            f"{component} has no ready node matching selector {dict(selector)}",
        )
        available = int(deployment.get("status", {}).get("availableReplicas", 0) or 0)
        if desired > 0:
            _record_requirement(
                errors,
                checks,
                available == desired,
                f"{component} worker replicas are available: {available}/{desired}",
                f"{component} worker replicas are unavailable: {available}/{desired}",
            )
        scaled = scaled_by_target.get(name)
        if desired == 0:
            _record_requirement(
                errors,
                checks,
                bool(scaled and _condition_true(scaled, "Ready")),
                f"{component} is ready for queue-driven autoscaling",
                f"{component} is scaled to zero without a Ready KEDA ScaledObject",
            )
        config_name = _mounted_config_map(deployment)
        if config_name:
            config_maps.add(config_name)
    return config_maps


def _validate_fleet_credentials(
    request: IngestionRequest,
    namespace: str,
    by_component: Mapping[str, Mapping[str, Any]],
    checks: list[str],
    errors: list[str],
    target: ClusterTarget,
) -> None:
    """Verify both Secret presence and finalizer environment mapping for output auth."""

    required_credentials = {
        selection.api_key_environment_variable
        for selection in (request.ocr, request.llm, request.embedding)
        if selection.api_key_environment_variable
    }
    required_credentials.update(request.output.credential_environment_variables)
    if not required_credentials:
        return
    secrets = _kubectl_json(
        target=target, arguments=["get", "secrets", "-n", namespace, "-o", "json"]
    ).get("items", [])
    available_keys = {str(key) for secret in secrets for key in secret.get("data", {})}
    for credential in sorted(required_credentials):
        _record_requirement(
            errors,
            checks,
            credential in available_keys,
            f"Fleet credential is available: {credential}",
            f"Fleet credential is missing from namespace {namespace}: {credential}",
        )
    finalizer_environment = _controller_environment_names(by_component.get("worker-finalize", {}))
    for credential in request.output.credential_environment_variables:
        _record_requirement(
            errors,
            checks,
            credential in finalizer_environment,
            f"Finalizer receives output credential: {credential}",
            f"Finalizer does not map output credential environment variable: {credential}",
        )


def _orchestrator_environment(request: IngestionRequest) -> dict[str, str]:
    """Resolve credential placeholders without copying Kubernetes secret values locally."""

    environment = dict(os.environ)
    for selection in (request.ocr, request.llm, request.embedding):
        name = selection.api_key_environment_variable
        if name:
            environment.setdefault(name, _KUBERNETES_SECRET_SENTINEL)
    for name in request.output.credential_environment_variables:
        environment.setdefault(name, _KUBERNETES_SECRET_SENTINEL)
    return environment


@contextmanager
def _fleet_database_environment(
    namespace: str,
    target: ClusterTarget,
    base_environment: Mapping[str, str] | None = None,
) -> Iterator[dict[str, str]]:
    """Expose the fleet coordination database to one local control-plane command.

    Worker deployments already define the authoritative database environment
    contract. A local Streamlit process should not require users to duplicate its
    secret in their shell, so the adapter resolves that exact deployment entry in
    memory and forwards an internal Service to loopback for the duration of one
    bounded CLI call. An explicitly supplied local URL remains authoritative.
    """

    environment = dict(base_environment or os.environ)
    database_url = environment.get(_DATABASE_ENVIRONMENT_NAME, "").strip()
    if not database_url:
        database_url = _fleet_database_url(namespace, target)
        environment[_DATABASE_ENVIRONMENT_NAME] = database_url

    service = _database_cluster_service(database_url, namespace)
    if service is None:
        yield environment
        return

    service_name, service_namespace, remote_port = service
    forward_target = _service_port_forward_target(service_name, service_namespace, target)
    local_port = _available_local_port()
    process = subprocess.Popen(
        [
            "kubectl",
            *target.arguments(),
            "port-forward",
            "--namespace",
            service_namespace,
            "--address",
            "127.0.0.1",
            forward_target,
            f"{local_port}:{remote_port}",
        ],
        env=target.environment(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_port_forward(process, local_port)
        environment[_DATABASE_ENVIRONMENT_NAME] = _replace_url_host(database_url, local_port)
        yield environment
    finally:
        _terminate_forward(process)


def _fleet_database_url(namespace: str, target: ClusterTarget) -> str:
    """Resolve the database URI from a worker's Kubernetes environment mapping."""

    deployments = _kubectl_json(
        target=target,
        arguments=[
            "get",
            "deployments",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=flakegraph",
            "-o",
            "json",
        ],
    ).get("items", [])
    for deployment in deployments:
        component = _component(deployment)
        if component not in _REQUIRED_WORKER_COMPONENTS:
            continue
        containers = (
            deployment.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        )
        entry = next(
            (
                item
                for container in containers
                for item in container.get("env", [])
                if item.get("name") == _DATABASE_ENVIRONMENT_NAME
            ),
            None,
        )
        if entry is not None:
            return _resolve_environment_entry(namespace, entry, target)
    raise RuntimeError(
        "Kubernetes workers do not expose the distributed database connection; "
        "check the Helm database secret configuration"
    )


def _database_cluster_service(
    database_url: str,
    default_namespace: str,
) -> tuple[str, str, int] | None:
    """Identify a short or fully qualified Kubernetes PostgreSQL Service host."""

    parsed = urlparse(database_url)
    host = parsed.hostname or ""
    if not host or host in {"localhost", "127.0.0.1", "::1"} or _ip_address(host):
        return None
    try:
        port = parsed.port or _POSTGRES_DEFAULT_PORT
    except ValueError as exc:
        raise RuntimeError("Kubernetes database URL contains an invalid port") from exc
    labels = host.rstrip(".").split(".")
    if len(labels) == 1:
        return labels[0], default_namespace, port
    if len(labels) == _KUBERNETES_SERVICE_NAMESPACE_LABELS:
        return labels[0], labels[1], port
    if len(labels) >= _KUBERNETES_DNS_MIN_LABELS and labels[2] == "svc":
        return labels[0], labels[1], port
    return None


def _replace_url_host(database_url: str, local_port: int) -> str:
    """Point a credential-bearing URI at loopback without decoding user info."""

    parsed = urlparse(database_url)
    user_info, separator, _host = parsed.netloc.rpartition("@")
    prefix = f"{user_info}{separator}" if separator else ""
    return urlunparse(parsed._replace(netloc=f"{prefix}127.0.0.1:{local_port}"))


def _ip_address(host: str) -> bool:
    """Return whether a host is already a numeric address rather than Service DNS."""

    try:
        socket.inet_pton(socket.AF_INET, host)
    except OSError:
        try:
            socket.inet_pton(socket.AF_INET6, host)
        except OSError:
            return False
    return True


def _terminate_forward(process: subprocess.Popen[bytes]) -> None:
    """Stop a temporary kubectl forward and reap it even after command failures."""

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _command_failure_message(stderr: str, stdout: str) -> str:
    """Return a concise final exception instead of exposing a Python traceback."""

    lines = [line.strip() for line in (stderr or stdout).splitlines() if line.strip()]
    if not lines:
        return "FlakeGraph distributed command failed without an error message"
    message = lines[-1]
    for prefix in ("ValueError: ", "RuntimeError: "):
        if message.startswith(prefix):
            return message.removeprefix(prefix)
    return message


def _bounded_reason(message: str) -> str:
    """Keep one cause readable inside a sidebar notice."""

    collapsed = " ".join(message.split())
    if not collapsed:
        return "the fleet coordination store could not be reached"
    if len(collapsed) <= _STALE_HISTORY_REASON_LIMIT:
        return collapsed
    return f"{collapsed[: _STALE_HISTORY_REASON_LIMIT - 1].rstrip()}…"


@contextmanager
def _artifact_export_environment(namespace: str, target: ClusterTarget) -> Iterator[dict[str, str]]:
    """Expose fleet artifact storage to one local export command.

    Worker pods receive their S3-compatible storage settings from the Helm release,
    while a locally running Streamlit process normally has only the coordination
    database connection. This bridge reads the finalizer's reviewed environment
    mapping, resolves referenced Secret keys in memory, and temporarily forwards an
    internal Kubernetes Service when its DNS name is not routable from the submit
    host. The forwarding process and resolved values live only for the duration of
    the export and are never written to the generated run configuration.
    """

    environment = dict(os.environ)
    environment.update(_fleet_artifact_environment(namespace, target))
    endpoint = environment.get("KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL", "").strip()
    service = _cluster_service(endpoint)
    if service is None:
        yield environment
        return

    service_name, service_namespace, remote_port = service
    forward_target = _service_port_forward_target(service_name, service_namespace, target)
    local_port = _available_local_port()
    process = subprocess.Popen(
        [
            "kubectl",
            *target.arguments(),
            "port-forward",
            "--namespace",
            service_namespace,
            "--address",
            "127.0.0.1",
            forward_target,
            f"{local_port}:{remote_port}",
        ],
        env=target.environment(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_port_forward(process, local_port)
        parsed = urlparse(endpoint)
        environment["KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL"] = urlunparse(
            parsed._replace(netloc=f"127.0.0.1:{local_port}")
        )
        yield environment
    finally:
        _terminate_forward(process)


def _fleet_artifact_environment(namespace: str, target: ClusterTarget) -> dict[str, str]:
    """Read only artifact-store variables from the deployed finalizer contract."""

    deployments = _kubectl_json(
        target=target,
        arguments=[
            "get",
            "deployments",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=flakegraph",
            "-o",
            "json",
        ],
    ).get("items", [])
    finalizer = next(
        (item for item in deployments if _component(item) == "worker-finalize"),
        None,
    )
    if finalizer is None:
        raise RuntimeError("Kubernetes fleet has no finalizer deployment for graph export")
    containers = finalizer.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    entries = {
        str(entry.get("name")): entry
        for container in containers
        for entry in container.get("env", [])
        if str(entry.get("name")) in _ARTIFACT_ENVIRONMENT_NAMES
    }
    resolved = {
        name: _resolve_environment_entry(namespace, entries[name], target)
        for name in _ARTIFACT_ENVIRONMENT_NAMES
        if name in entries
    }
    if not resolved.get("KG_DISTRIBUTED_ARTIFACT_URI"):
        raise RuntimeError("Kubernetes finalizer has no shared artifact store configured")
    return resolved


def _resolve_environment_entry(
    namespace: str,
    entry: Mapping[str, Any],
    target: ClusterTarget,
) -> str:
    """Resolve a literal or Secret-backed Kubernetes environment entry in memory."""

    if entry.get("value") is not None:
        return str(entry["value"])
    reference = entry.get("valueFrom", {}).get("secretKeyRef", {})
    secret_name = str(reference.get("name") or "")
    secret_key = str(reference.get("key") or "")
    if not secret_name or not secret_key:
        raise RuntimeError(f"Kubernetes environment entry {entry.get('name')} is not resolvable")
    secret = _kubectl_json(
        target=target, arguments=["get", "secret", secret_name, "-n", namespace, "-o", "json"]
    )
    encoded = str(secret.get("data", {}).get(secret_key) or "")
    if not encoded:
        raise RuntimeError(f"Kubernetes Secret {secret_name} does not contain key {secret_key}")
    try:
        return b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Kubernetes Secret {secret_name} key {secret_key} is not valid UTF-8 data"
        ) from exc


def _cluster_service(endpoint: str) -> tuple[str, str, int] | None:
    """Return an internal Service target encoded in a Kubernetes DNS endpoint."""

    if not endpoint:
        return None
    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    labels = host.split(".")
    if len(labels) < _KUBERNETES_DNS_MIN_LABELS or labels[2] != "svc":
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise RuntimeError("Kubernetes artifact endpoint contains an invalid port") from exc
    return labels[0], labels[1], port


def _available_local_port() -> int:
    """Ask the operating system for an unused loopback port for a short forward."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _service_port_forward_target(
    service_name: str,
    namespace: str,
    target: ClusterTarget,
) -> str:
    """Prefer a ready Service endpoint pod over kubectl's selector-based choice.

    ``kubectl port-forward service`` resolves the Service selector to an arbitrary
    pod rather than following EndpointSlice readiness. Stateful databases can use
    a broad selector while exposing only the current primary as ready, so selecting
    the endpoint explicitly avoids forwarding to a replica or unavailable node.
    Services without pod-backed EndpointSlices retain kubectl's normal fallback.
    """

    try:
        slices = _kubectl_json(
            target=target,
            arguments=[
                "get",
                "endpointslices.discovery.k8s.io",
                "-n",
                namespace,
                "-l",
                f"kubernetes.io/service-name={service_name}",
                "-o",
                "json",
            ],
        ).get("items", [])
    except RuntimeError:
        return f"service/{service_name}"
    pod_names = sorted(
        str(endpoint.get("targetRef", {}).get("name", ""))
        for endpoint_slice in slices
        for endpoint in endpoint_slice.get("endpoints", [])
        if endpoint.get("conditions", {}).get("ready") is True
        and endpoint.get("targetRef", {}).get("kind") == "Pod"
        and endpoint.get("targetRef", {}).get("name")
    )
    return f"pod/{pod_names[0]}" if pod_names else f"service/{service_name}"


def _wait_for_port_forward(process: subprocess.Popen[bytes], port: int) -> None:
    """Wait until kubectl accepts loopback connections or report startup failure."""

    deadline = time.monotonic() + _PORT_FORWARD_STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = (
                process.stderr.read().decode("utf-8", "replace").strip()
                if process.stderr
                else ""
            )
            raise RuntimeError(output or "Kubernetes service port-forward exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Timed out while opening the Kubernetes service port-forward")


def _message_items(payload: Mapping[str, object], key: str) -> list[str]:
    """Normalize one preflight message list without trusting an external JSON shape."""

    value = payload.get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _record_requirement(
    errors: list[str],
    checks: list[str],
    condition: bool,
    ok_message: str,
    error_message: str,
) -> None:
    """Append one fleet assertion to the same JSON contract as core preflight."""

    (checks if condition else errors).append(ok_message if condition else error_message)


def _component(resource: Mapping[str, Any]) -> str:
    """Return the standard FlakeGraph component label for a controller."""

    return str(
        resource.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component", "")
    )


def _condition_true(resource: Mapping[str, Any], condition_type: str) -> bool:
    """Return whether a Kubernetes resource reports one named condition as true."""

    return any(
        item.get("type") == condition_type and item.get("status") == "True"
        for item in resource.get("status", {}).get("conditions", [])
    )


def _controller_image(resource: Mapping[str, Any]) -> str | None:
    """Read the primary container image from a Deployment or StatefulSet."""

    containers = resource.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not containers or not containers[0].get("image"):
        return None
    return str(containers[0]["image"])


def _controller_environment_names(resource: Mapping[str, Any]) -> set[str]:
    """Return environment names explicitly mapped into a controller's first container."""

    containers = resource.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not containers:
        return set()
    return {str(item.get("name")) for item in containers[0].get("env", []) if item.get("name")}


def _mounted_config_map(resource: Mapping[str, Any], volume_name: str = "config") -> str | None:
    """Return the ConfigMap a worker mounts under one volume name."""

    volumes = resource.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    for volume in volumes:
        if volume.get("name") != volume_name:
            continue
        name = volume.get("configMap", {}).get("name")
        return str(name) if name else None
    return None


def _load_deployed_profile(config_map: Mapping[str, Any]) -> Mapping[str, Any]:
    """Parse the mounted worker profile without exposing any Secret resources."""

    text = config_map.get("data", {}).get("config.yaml")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Fleet processing ConfigMap does not contain config.yaml")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise RuntimeError("Fleet processing config.yaml must contain a mapping")
    return loaded


# Keys inside the compared sections that describe how a worker reaches
# infrastructure rather than what graph it produces. The runtime excludes these
# from the digest workers claim by, so comparing them would report differences
# that never prevent a claim.
_DEPLOYMENT_LOCAL_KEYS: Mapping[str, frozenset[str]] = {
    "ocr": frozenset(
        {
            "mineru_api_key",
            "mineru_api_url",
            "mineru_command",
            "mineru_server_url",
            "model_cache_dir",
            "tesseract_command",
            "tesseract_pdf_renderer_command",
            "timeout_seconds",
        }
    ),
    "llm": frozenset({"api_key", "endpoint", "timeout_seconds"}),
    "embedding": frozenset({"api_key", "batch_size", "device", "endpoint"}),
    "graph": frozenset(
        {
            "community_report_parallelism",
            "description_merge_parallelism",
            "extraction_parallelism",
            "resolution_parallelism",
        }
    ),
}


def _semantic_mismatches(
    effective: Mapping[str, Any],
    deployed: Mapping[str, Any],
) -> list[str]:
    """Report every digest-relevant field on which a run and the fleet disagree.

    A worker claims only runs whose semantic configuration hashes to its own, and
    that hash covers whole configuration sections. Comparing a handful of named
    fields therefore passes runs that no worker can ever claim: this check once
    reported success on a run that differed in ``mineru_method``,
    ``mineru_backend`` and ``fail_on_quality_error``, and the run sat queued with
    nothing on the page to explain it.

    Only keys both sides state are compared. A key the fleet omits takes the
    runtime's default, and resolving that default needs the runtime's own
    settings model, which this application deliberately cannot import — guessing
    would reject working configurations. The observed failure disagreed on three
    fields and two were stated by both sides, so either would have blocked it.
    """

    mismatches: list[str] = []
    for section in ("ocr", "llm", "embedding", "graph", "extractors"):
        run_section = effective.get(section)
        fleet_section = deployed.get(section)
        if not isinstance(run_section, Mapping):
            continue
        excluded = _DEPLOYMENT_LOCAL_KEYS.get(section, frozenset())
        fleet_values = fleet_section if isinstance(fleet_section, Mapping) else {}
        for key in sorted(set(run_section) | set(fleet_values)):
            if key in excluded:
                continue
            selected = run_section.get(key)
            actual = fleet_values.get(key)
            if key not in fleet_values or key not in run_section:
                continue
            if isinstance(actual, str) and _environment_placeholder(actual):
                continue
            if selected != actual:
                mismatches.append(
                    f"{section}.{key} does not match fleet workers: "
                    f"selected {selected!r}, deployed {actual!r}"
                )
    return mismatches


def _profile_mismatches(
    request: IngestionRequest,
    deployed: Mapping[str, Any],
) -> list[str]:
    """Explain semantic provider differences between one run and homogeneous workers."""

    comparisons = [
        ("OCR provider", request.ocr.provider, _profile_value(deployed, "ocr", "provider")),
        ("LLM provider", request.llm.provider, _profile_value(deployed, "llm", "provider")),
        ("LLM model", request.llm.model, _profile_value(deployed, "llm", "model")),
        (
            "embedding provider",
            request.embedding.provider,
            _profile_value(deployed, "embedding", "provider"),
        ),
        (
            "embedding model",
            request.embedding.model,
            _profile_value(deployed, "embedding", "model"),
        ),
    ]
    return [
        f"{label} does not match fleet workers: selected {selected!r}, deployed {actual!r}"
        for label, selected, actual in comparisons
        if selected and actual and not _environment_placeholder(actual) and selected != actual
    ]


def _ontology_mismatches(
    request: IngestionRequest,
    deployed: Mapping[str, Any],
    namespace: str,
    target: ClusterTarget,
    deployments: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Report an ontology the fleet workers could not agree with.

    A submitted run carries its ontology inline so it describes itself wherever it
    executes, while workers mount theirs as a file. Comparing the two therefore
    means reading the mounted ConfigMap rather than the worker's config alone.
    """

    selected = run_ontology_profile(request)
    mounted_name = next(
        (name for item in deployments if (name := _mounted_config_map(item, "ontology"))),
        None,
    )
    deployed_reference = deployed.get("ontology")
    has_deployed = bool(mounted_name) or (
        isinstance(deployed_reference, Mapping)
        and any(deployed_reference.get(key) for key in ("profile", "profile_path"))
    )

    if selected and not has_deployed:
        return [
            "Selected ontology is not deployed to the fleet: workers mount no ontology, "
            "so no worker can claim this run. Install the chart with "
            "--set-file ontology.content=<profile.yaml>."
        ]
    if has_deployed and not selected:
        return [
            "Fleet workers mount an ontology but this run selects none, "
            "so no worker can claim this run."
        ]
    if not selected or not mounted_name:
        return []

    ontology_map = _kubectl_json(
        target=target,
        arguments=["get", "configmap", mounted_name, "-n", namespace, "-o", "json"],
    )
    data = ontology_map.get("data", {})
    text = next((value for value in data.values() if isinstance(value, str)), "")
    loaded = yaml.safe_load(text) or {} if text.strip() else {}
    if not isinstance(loaded, Mapping):
        return [f"Fleet ontology ConfigMap {mounted_name} does not contain a mapping"]
    if dict(loaded) != dict(selected):
        return [
            f"Selected ontology differs from the one fleet workers mount ({mounted_name}), "
            "so no worker can claim this run."
        ]
    return []


def _profile_value(profile: Mapping[str, Any], section: str, key: str) -> str | None:
    """Read one optional scalar from a deployed YAML profile."""

    value = profile.get(section, {})
    if not isinstance(value, Mapping) or value.get(key) in (None, ""):
        return None
    return str(value[key])


def _loopback_endpoint(endpoint: str | None) -> bool:
    """Identify node-local model endpoints that require chart-managed serving."""

    if not endpoint:
        return False
    return urlparse(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}


def _environment_placeholder(value: str) -> bool:
    """Return whether a deployed scalar is supplied by the pod environment."""

    return _ENVIRONMENT_PLACEHOLDER_PATTERN.fullmatch(value) is not None


def _served_model(container: Mapping[str, Any]) -> str | None:
    """Read the public model name from common vLLM command-line layouts."""

    arguments = [str(item) for item in container.get("args", [])]
    try:
        option_index = arguments.index("--served-model-name")
    except ValueError:
        option_index = -1
    if option_index >= 0 and option_index + 1 < len(arguments):
        return arguments[option_index + 1]
    if len(arguments) >= _VLLM_MIN_SERVE_ARGUMENTS and arguments[0] == "serve":
        return arguments[1]
    return None


def _node_class(labels: Mapping[str, Any]) -> str:
    """Choose the most useful portable hardware or scheduling class label."""

    explicit = _first_label(
        labels,
        "flakegraph.io/node-class",
        "node.kubernetes.io/instance-type",
        "beta.kubernetes.io/instance-type",
    )
    if explicit:
        return explicit
    roles = [
        key.removeprefix("node-role.kubernetes.io/")
        for key in labels
        if str(key).startswith("node-role.kubernetes.io/")
    ]
    return ", ".join(sorted(role for role in roles if role)) or "kubernetes-node"


def _first_label(labels: Mapping[str, Any], *names: str) -> str | None:
    """Return the first populated label from an ordered compatibility list."""

    for name in names:
        value = labels.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _resource_count(value: Any) -> int:
    """Parse integral Kubernetes resources while tolerating absent vendor capacity."""

    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _optional_string(value: Any) -> str | None:
    """Normalize optional Kubernetes scalar values for display models."""

    return str(value) if value not in (None, "") else None
