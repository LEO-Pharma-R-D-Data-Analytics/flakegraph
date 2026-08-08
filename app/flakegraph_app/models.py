"""Portable value objects shared by application backends and Streamlit pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# Measured against this deployment's Cortex endpoint by sweeping concurrency for
# a representative completion. Throughput rose to a peak at sixteen in flight and
# then collapsed: at twenty-four it was below the level of four, with median
# latency roughly tripled. The default sits below that peak so two graphs built
# at once still share the account's budget without crossing into collapse; the
# ceiling is the provider's own per-account, per-model limit, not the machine.
DEFAULT_PROVIDER_PARALLELISM = 12

# Distributed tasks are created dynamically, so database result order is not a
# reliable presentation order. Keep the execution contract in one shared place
# for adapters and UI renderers that need to explain the pipeline to operators.
DISTRIBUTED_STAGE_ORDER: tuple[str, ...] = (
    "prepare_document",
    "extract_document_context",
    "extract_entity_window",
    "compact_entity_inventory",
    "extract_relation_window",
    "compact_document",
    "finalize_graph",
)

# A local or Snowflake worker runs the whole pipeline in one process and reports
# these stages, which are a different vocabulary from the distributed task names
# above. Both the JSONL progress reader and the run page order by it, so it lives
# here rather than beside either reader.
PIPELINE_STAGE_ORDER: tuple[str, ...] = (
    "file_source",
    "ocr",
    "chunking",
    "chunk_embeddings",
    "graph_extraction",
    "filtering",
    "merge",
    "graph_embeddings",
    "community_detection",
    "quality",
    "write",
)

# A run reports one vocabulary or the other, never both, so one lookup orders
# either without the renderer needing to know which runtime produced it.
RUN_STAGE_ORDER: tuple[str, ...] = DISTRIBUTED_STAGE_ORDER + PIPELINE_STAGE_ORDER


class RuntimeMode(StrEnum):
    """Execution environments controlled or observed by the application."""

    LOCAL = "Local"
    KUBERNETES = "Kubernetes"
    SNOWFLAKE = "Snowflake"


class SourceKind(StrEnum):
    """Input source families exposed by the ingestion form."""

    UPLOAD = "Upload"
    LOCAL = "Local path"
    AZURE_BLOB = "Azure Blob"
    S3 = "S3-compatible bucket"
    SNOWFLAKE_STAGE = "Snowflake stage"


class StorageKind(StrEnum):
    """Durable graph destinations shown consistently across all runtimes."""

    LOCAL = "Local files"
    SNOWFLAKE = "Snowflake"


@dataclass(frozen=True)
class SourceObject:
    """One selectable object returned by a local or remote source browser."""

    uri: str
    name: str
    size_bytes: int
    modified_at: str | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class ProviderSelection:
    """User-selected adapter and non-secret provider connection settings.

    ``dimension`` is meaningful only for embedding providers. Keeping it on the
    immutable selection makes the vector shape part of the reviewed request
    instead of an implicit model-side default that can drift between runtimes.
    """

    provider: str
    model: str | None = None
    endpoint: str | None = None
    api_key_environment_variable: str | None = None
    dimension: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject dimensions that cannot describe a valid embedding vector."""

        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")


@dataclass(frozen=True)
class SnowflakeOutput:
    """Non-secret Snowflake target plus a reference to its runtime credential.

    The credential value never enters Streamlit session state, generated YAML, or
    the run catalog. Local processes resolve the named environment variable; fleet
    workers resolve the same name from a Kubernetes Secret mounted as environment
    variables.
    """

    account: str
    user: str
    database: str
    schema: str
    warehouse: str
    bulk_stage: str
    role: str | None = None
    host: str | None = None
    authenticator: str | None = None
    credential_environment_variable: str | None = None
    credential_field: str | None = None

    @property
    def location(self) -> str:
        """Return a compact, non-secret destination label for catalogs and help text."""

        return f"{self.database}.{self.schema}"


@dataclass(frozen=True)
class OutputDestination:
    """Select one durable graph writer independently from the execution runtime."""

    kind: StorageKind
    workspace_path: Path
    snowflake: SnowflakeOutput | None = None

    @property
    def writer_provider(self) -> str:
        """Map the user-facing destination to the processing-core writer adapter."""

        return "snowflake_bulk" if self.kind == StorageKind.SNOWFLAKE else "local_artifacts"

    @property
    def location(self) -> str:
        """Return the durable location displayed with a completed graph."""

        if self.snowflake is not None:
            return self.snowflake.location
        return str(self.workspace_path)

    @property
    def credential_environment_variables(self) -> tuple[str, ...]:
        """Return secret references that the selected runtime must make available."""

        name = self.snowflake.credential_environment_variable if self.snowflake else None
        return (name,) if name else ()


@dataclass(frozen=True)
class IngestionRequest:
    """Complete backend-neutral description of one requested graph run."""

    runtime: RuntimeMode
    job_id: str
    graph_id: str
    source_kind: SourceKind
    source: Mapping[str, Any]
    ocr: ProviderSelection
    llm: ProviderSelection
    embedding: ProviderSelection
    output: OutputDestination
    base_config_path: Path | None = None
    include_globs: Sequence[str] = ("**/*",)
    cache_provider: str = "local"
    # How many model calls each processing stage may have in flight. It bounds
    # the pipeline rather than the machine hosting it, so it belongs to the run
    # and applies whichever runtime executes it.
    provider_parallelism: int = DEFAULT_PROVIDER_PARALLELISM
    runtime_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    """Normalized progress event emitted by any processing runtime."""

    timestamp: str
    stage: str
    status: str
    file_id: str | None = None
    message: str | None = None
    elapsed_ms: int | None = None
    counts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageProgress:
    """Latest status and accumulated duration for one pipeline stage."""

    stage: str
    status: str
    completed: int = 0
    total: int | None = None
    elapsed_ms: int = 0
    message: str | None = None


@dataclass(frozen=True)
class RunSnapshot:
    """Compact application-facing status for a local or remote graph run."""

    run_id: str
    graph_id: str
    status: str
    started_at: str | None
    updated_at: str | None
    graph_name: str | None = None
    stages: Sequence[StageProgress] = ()
    events: Sequence[ProgressEvent] = ()
    documents_total: int | None = None
    documents_completed: int = 0
    documents_failed: int = 0
    output_path: str | None = None
    storage_kind: StorageKind = StorageKind.LOCAL
    storage_location: str | None = None
    warnings: Sequence[str] = ()
    error: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Return the human-readable name while preserving the stable graph ID."""

        return self.graph_name or self.graph_id


@dataclass(frozen=True)
class WorkloadStatus:
    """One Kubernetes workload or model-serving pod shown to an operator."""

    name: str
    component: str
    phase: str
    node: str | None
    ready: bool
    restarts: int
    cpu: str | None = None
    memory: str | None = None
    image: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class NodeStatus:
    """One Kubernetes compute node with hardware and placed-workload context."""

    name: str
    ready: bool
    node_class: str
    gpu_count: int
    gpu_model: str | None
    cpu_capacity: str | None
    memory_capacity: str | None
    gpu_percent: str | None = None
    cpu_usage: str | None = None
    cpu_percent: str | None = None
    memory_usage: str | None = None
    memory_percent: str | None = None
    workload_count: int = 0
    worker_count: int = 0
    model_server_ready: bool = False
    model: str | None = None
    model_image: str | None = None


@dataclass(frozen=True)
class NodeWorkAssignment:
    """One currently leased task shown in a node's on-demand work details."""

    worker_id: str
    run_id: str
    graph_id: str
    task_id: str
    stage: str
    scope_id: str
    updated_at: str | None = None


@dataclass(frozen=True)
class ClusterSnapshot:
    """Bounded operational view of the selected Kubernetes namespace."""

    context: str
    namespace: str
    nodes_ready: int
    nodes_total: int
    nodes: Sequence[NodeStatus]
    workloads: Sequence[WorkloadStatus]
    warnings: Sequence[str] = ()
    observed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class GraphDataset:
    """Graph artifacts and supporting evidence loaded for interactive review."""

    graph_id: str
    nodes: Sequence[Mapping[str, Any]]
    edges: Sequence[Mapping[str, Any]]
    communities: Sequence[Mapping[str, Any]] = ()
    evidence: Sequence[Mapping[str, Any]] = ()
    documents: Sequence[Mapping[str, Any]] = ()
    chunks: Sequence[Mapping[str, Any]] = ()
    run_report: Mapping[str, Any] = field(default_factory=dict)
    graph_metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def counts(self) -> Mapping[str, int]:
        """Return principal full-dataset counts for summary metrics."""

        full_counts = self.run_report.get("app_full_counts")
        if isinstance(full_counts, Mapping):
            return {
                "documents": int(full_counts.get("document", len(self.documents))),
                "nodes": int(full_counts.get("node", len(self.nodes))),
                "edges": int(full_counts.get("edge", len(self.edges))),
                "communities": int(full_counts.get("community", len(self.communities))),
                "evidence": int(full_counts.get("evidence", len(self.evidence))),
            }
        return {
            "documents": len(self.documents),
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "communities": len(self.communities),
            "evidence": len(self.evidence),
        }
