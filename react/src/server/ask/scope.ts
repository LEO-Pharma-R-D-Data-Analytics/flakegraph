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
  return {
    ...next,
    nodes: next.nodes.filter((node) => nodeIds.has(String(node.id ?? ""))),
    edges: next.edges.filter((edge) => edgeIds.has(String(edge.id ?? ""))),
    evidence: next.evidence.filter((row) => {
      const relationId = String(row.relation_id ?? "");
      const entityId = String(row.entity_id ?? row.node_id ?? "");
      return (!relationId || edgeIds.has(relationId)) && (!entityId || nodeIds.has(entityId));
    }),
    chunks: next.chunks.filter((row) => {
      const entityId = String(row.entity_id ?? row.node_id ?? "");
      return !entityId || nodeIds.has(entityId);
    }),
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
    const relationId = String(row.relation_id ?? "");
    const entityId = String(row.entity_id ?? row.node_id ?? "");
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
