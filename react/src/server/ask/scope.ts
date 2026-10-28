import { evidenceEntityId, evidenceRelationId } from "@/lib/evidence";
import { communityMemberIds, filterGraph } from "../graph-filter";
import type { GraphDataset } from "../protocol/schema";
import { documentIdOf, edgeEnds, nodeName } from "./retrieve";
import type { AskScope } from "./types";

export function scopeDataset(dataset: GraphDataset, scope?: AskScope): GraphDataset {
  if (!isScoped(scope)) {
    return dataset;
  }
  let next = dataset;
  if (scope?.documentIds && scope.documentIds.length > 0) {
    next = restrictToDocuments(next, scope.documentIds);
  }
  if (!scope?.search && !(scope?.communityIds && scope.communityIds.length > 0)) {
    return next;
  }
  const filtered = filterGraph(next, {
    search: scope.search,
    communityIds: scope.communityIds,
    includeIsolates: true,
    limit: Number.POSITIVE_INFINITY,
  });
  const nodeIds = new Set(filtered.nodes.map((node) => String(node.id ?? "")));
  const edgeIds = new Set(filtered.edges.map((edge) => String(edge.id ?? "")));
  const nodes = next.nodes.filter((node) => nodeIds.has(String(node.id ?? "")));
  const edges = next.edges.filter((edge) => edgeIds.has(String(edge.id ?? "")));
  const chunkIds = new Set<string>();
  for (const row of [...nodes, ...edges]) {
    const ids = row.source_chunk_ids ?? row.sourceChunkIds;
    if (Array.isArray(ids)) ids.forEach((id) => chunkIds.add(String(id)));
  }
  const evidence = next.evidence.filter((row) => {
    const relationId = evidenceRelationId(row);
    const entityId = evidenceEntityId(row);
    return relationId ? edgeIds.has(relationId) : entityId ? nodeIds.has(entityId) : chunkIds.has(String(row.chunk_id ?? ""));
  });
  for (const row of evidence) if (row.chunk_id) chunkIds.add(String(row.chunk_id));
  const chunks = next.chunks.filter((row) => {
    const entityId = evidenceEntityId(row);
    return chunkIds.has(String(row.id ?? row.chunk_id ?? "")) || Boolean(entityId && nodeIds.has(entityId));
  });
  const documentIds = new Set([...evidence, ...chunks].map(documentIdOf));
  return {
    ...next,
    nodes,
    edges,
    evidence,
    chunks,
    documents: next.documents.filter((row) => [row.id, row.document_id, row.file_id].some((id) => id != null && documentIds.has(String(id)))),
    communities: scope.communityIds?.length
      ? next.communities.filter((community) => scope.communityIds!.includes(String(community.id ?? "")))
      : next.communities.filter((community) => {
          const members = communityMemberIds(community);
          return members.some((id) => nodeIds.has(id));
        }),
  };
}

export function isScoped(scope?: AskScope): boolean {
  return Boolean(
    scope?.search ||
      (scope?.communityIds && scope.communityIds.length > 0) ||
      (scope?.documentIds && scope.documentIds.length > 0),
  );
}

export function mergeAskScope(base?: AskScope, extra?: AskScope): AskScope | undefined {
  if (!base && !extra) {
    return undefined;
  }
  const documentIds = intersectIds(extra?.documentIds, base?.documentIds);
  const communityIds = extra?.communityIds ?? base?.communityIds;
  const search = extra?.search ?? base?.search;
  if (!search && !(communityIds && communityIds.length) && !(documentIds && documentIds.length)) {
    return undefined;
  }
  return { search, communityIds, documentIds };
}

export function intersectIds(requested?: string[], existing?: string[]): string[] | undefined {
  if (requested && requested.length > 0 && existing && existing.length > 0) {
    const allowed = new Set(existing);
    return requested.filter((id) => allowed.has(id));
  }
  if (requested && requested.length > 0) {
    return requested;
  }
  if (existing && existing.length > 0) {
    return existing;
  }
  return undefined;
}

function restrictToDocuments(dataset: GraphDataset, documentIds: string[]): GraphDataset {
  const allowed = new Set(documentIds);
  const documents = dataset.documents.filter((document) => allowed.has(String(document.id ?? "")));
  const evidence = dataset.evidence.filter((row) => allowed.has(documentIdOf(row)));
  const chunks = dataset.chunks.filter((row) => allowed.has(documentIdOf(row)));
  const relationIds = new Set<string>();
  const entityIds = new Set<string>();
  for (const row of [...evidence, ...chunks]) {
    const relationId = evidenceRelationId(row) ?? "";
    const entityId = evidenceEntityId(row) ?? "";
    if (relationId) {
      relationIds.add(relationId);
    }
    if (entityId) {
      entityIds.add(entityId);
    }
  }
  const linkedEdges = dataset.edges.filter((edge) => relationIds.has(String(edge.id ?? "")));
  for (const edge of linkedEdges) {
    const ends = edgeEnds(edge);
    entityIds.add(ends.source);
    entityIds.add(ends.target);
  }
  if (entityIds.size === 0 && (evidence.length > 0 || chunks.length > 0)) {
    const haystack = [...evidence, ...chunks]
      .map((row) => String(row.quote ?? row.sentence ?? row.content ?? row.text ?? "").toLowerCase())
      .join(" ");
    for (const node of dataset.nodes) {
      const name = nodeName(node).toLowerCase();
      if (name && haystack.includes(name)) {
        entityIds.add(String(node.id ?? ""));
      }
    }
  }
  const nodes = dataset.nodes.filter((node) => entityIds.has(String(node.id ?? "")));
  const nodeIds = new Set(nodes.map((node) => String(node.id ?? "")));
  const edges = dataset.edges.filter((edge) => {
    const ends = edgeEnds(edge);
    const id = String(edge.id ?? "");
    return nodeIds.has(ends.source) && nodeIds.has(ends.target) && (relationIds.size === 0 || relationIds.has(id));
  });
  const communities = dataset.communities.filter((community) => {
    const members = communityMemberIds(community);
    return members.some((id) => nodeIds.has(id));
  });
  return { ...dataset, documents, evidence, chunks, nodes, edges, communities };
}
