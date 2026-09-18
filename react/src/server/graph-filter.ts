import type { GraphDataset } from "./protocol/schema";

const MAX_CANVAS_NODES = 1_200;

export interface GraphFilters {
  search?: string;
  nodeTypes?: string[];
  relationTypes?: string[];
  communityIds?: string[];
  minimumConfidence?: number;
  includeIsolates?: boolean;
}

export interface GraphFacets {
  nodeTypes: string[];
  relationTypes: string[];
  communityIds: string[];
}

export function graphFacets(dataset: GraphDataset): GraphFacets {
  const nodeTypes = unique(dataset.nodes.map((node) => String(node.primary_type ?? node.type ?? "Unknown")));
  const relationTypes = unique(
    dataset.edges.map((edge) => String(edge.relation_type ?? edge.relationType ?? "RELATED")),
  );
  const communityIds = unique(dataset.communities.map((community) => String(community.id ?? "")));
  return { nodeTypes, relationTypes, communityIds };
}

export function filterGraph(
  dataset: GraphDataset,
  filters: GraphFilters = {},
): { nodes: Record<string, unknown>[]; edges: Record<string, unknown>[] } {
  const search = (filters.search ?? "").trim().toLowerCase();
  const allowedTypes = new Set(filters.nodeTypes ?? []);
  const allowedRelations = new Set(filters.relationTypes ?? []);
  const allowedCommunities = new Set(filters.communityIds ?? []);
  const membership = communityMembership(dataset.communities);
  const minimumConfidence = filters.minimumConfidence ?? 0;
  const includeIsolates = filters.includeIsolates ?? true;
  const nodes = dataset.nodes.filter((raw) => {
    const nodeId = String(raw.id ?? "");
    const haystack = `${raw.name ?? ""} ${raw.description ?? ""} ${Array.isArray(raw.aliases) ? raw.aliases.join(" ") : ""}`.toLowerCase();
    if (search && !haystack.includes(search)) {
      return false;
    }
    if (allowedTypes.size > 0 && !allowedTypes.has(String(raw.primary_type ?? raw.type ?? "Unknown"))) {
      return false;
    }
    if (allowedCommunities.size > 0) {
      const owned = membership.get(nodeId) ?? new Set<string>();
      if (![...allowedCommunities].some((id) => owned.has(id))) {
        return false;
      }
    }
    return true;
  });
  const visible = new Set(nodes.map((node) => String(node.id ?? "")));
  let edges = dataset.edges.filter((edge) => {
    const source = String(edge.source_node_id ?? edge.source ?? "");
    const target = String(edge.target_node_id ?? edge.target ?? "");
    const relation = String(edge.relation_type ?? edge.relationType ?? "");
    const confidence = asFloat(edge.confidence);
    return (
      visible.has(source) &&
      visible.has(target) &&
      (allowedRelations.size === 0 || allowedRelations.has(relation)) &&
      confidence >= minimumConfidence
    );
  });
  if (!includeIsolates) {
    const connected = new Set<string>();
    for (const edge of edges) {
      connected.add(String(edge.source_node_id ?? edge.source ?? ""));
      connected.add(String(edge.target_node_id ?? edge.target ?? ""));
    }
    return {
      nodes: nodes.filter((node) => connected.has(String(node.id ?? ""))).slice(0, MAX_CANVAS_NODES),
      edges,
    };
  }
  const cappedNodes = nodes.slice(0, MAX_CANVAS_NODES);
  const cappedIds = new Set(cappedNodes.map((node) => String(node.id ?? "")));
  edges = edges.filter(
    (edge) =>
      cappedIds.has(String(edge.source_node_id ?? edge.source ?? "")) &&
      cappedIds.has(String(edge.target_node_id ?? edge.target ?? "")),
  );
  return { nodes: cappedNodes, edges };
}

export function communityMembership(communities: readonly Record<string, unknown>[]): Map<string, Set<string>> {
  const membership = new Map<string, Set<string>>();
  for (const community of communities) {
    const id = String(community.id ?? "");
    const members = variantSequence(community.members ?? community.member_ids ?? community.nodes);
    for (const member of members) {
      const set = membership.get(member) ?? new Set<string>();
      set.add(id);
      membership.set(member, set);
    }
  }
  return membership;
}

export function variantSequence(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.map((item) => String(item));
      }
    } catch {
      return value.split(",").map((item) => item.trim()).filter(Boolean);
    }
  }
  return [];
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function asFloat(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : 0;
}

export const CANVAS_NODE_LIMIT = MAX_CANVAS_NODES;
