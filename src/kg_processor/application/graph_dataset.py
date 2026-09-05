"""Read a partitioned graph manifest for explicit local export or inspection."""

from __future__ import annotations

from io import BytesIO
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    EdgeObservation,
    EntitySource,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)
from kg_processor.ports.blob_store import BlobStore


class GraphDatasetReader:
    """Materialize an explicitly requested graph version from Parquet partitions.

    Spark publication leaves table files partitioned in shared object storage.
    This reader exists for bounded local export and the static explorer, both of
    which inherently require a finite in-memory selection. Large downstream loads
    should consume the manifest partitions without passing through this reader.
    """

    def __init__(self, blob_store: BlobStore) -> None:
        """Retain the same authenticated object adapter used by distributed workers."""

        self.blob_store = blob_store

    def read(self, manifest: GraphDatasetManifest) -> GraphWriteBatch:
        """Load every table named by a manifest into the portable writer contract."""

        rows = {name: self._table_rows(manifest, name) for name in manifest.tables}
        return GraphWriteBatch(
            graph_id=manifest.graph_id,
            documents=rows.get("documents", []),
            pages=rows.get("pages", []),
            blocks=rows.get("blocks", []),
            assets=rows.get("assets", []),
            chunks=[Chunk.model_validate(row) for row in rows.get("chunks", [])],
            nodes=[GraphNode.model_validate(row) for row in rows.get("nodes", [])],
            edges=[GraphEdge.model_validate(row) for row in rows.get("edges", [])],
            edge_observations=[
                EdgeObservation.model_validate(row) for row in rows.get("edge_observations", [])
            ],
            evidence=[Evidence.model_validate(row) for row in rows.get("evidence", [])],
            entity_sources=[
                EntitySource.model_validate(row) for row in rows.get("entity_sources", [])
            ],
            communities=[Community.model_validate(row) for row in rows.get("communities", [])],
            community_findings=[
                CommunityFinding.model_validate(row) for row in rows.get("community_findings", [])
            ],
            run_report={
                "run_id": manifest.run_id,
                "engine": manifest.engine,
                "timings_seconds": manifest.timings_seconds,
            },
            graph_metrics=manifest.metrics,
        )

    def _table_rows(
        self,
        manifest: GraphDatasetManifest,
        name: str,
    ) -> list[dict[str, Any]]:
        """Read and concatenate one logical table's immutable Parquet files."""

        table_manifest = manifest.table(name)
        if not table_manifest.files:
            return []
        read_table = cast(Any, pq.read_table)
        tables = [
            read_table(BytesIO(self.blob_store.get(_public_s3_uri(item.uri))))
            for item in table_manifest.files
        ]
        table = pa.concat_tables(tables, promote_options="default")
        return [dict(row) for row in table.to_pylist()]


def _public_s3_uri(uri: str) -> str:
    """Translate Hadoop's internal S3A scheme back to the artifact-store contract."""

    return f"s3://{uri[6:]}" if uri.startswith("s3a://") else uri
