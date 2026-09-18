import { readFile } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";
import { GRAPH_REVIEW_ROW_LIMITS, type GraphDataset } from "./protocol/schema";

export async function loadLocalGraph(directory: string): Promise<GraphDataset> {
  if (!existsSync(directory)) {
    throw new Error(`No graph artifact directory exists at this location: ${directory}`);
  }
  const parquetNodes = path.join(directory, "nodes.parquet");
  const jsonNodes = path.join(directory, "nodes.json");
  if (!existsSync(parquetNodes) && !existsSync(jsonNodes)) {
    throw new Error(`Graph directory must contain nodes.parquet or nodes.json: ${directory}`);
  }
  const [nodes, nodeCount] = await readTable(directory, "nodes", GRAPH_REVIEW_ROW_LIMITS.nodes);
  const nodeIds = new Set(nodes.map((node) => String(node.id ?? "")));
  const [edges, edgeCount] = await readTable(directory, "edges", GRAPH_REVIEW_ROW_LIMITS.edges);
  const visibleEdges = edges.filter(
    (edge) => nodeIds.has(String(edge.source_node_id ?? edge.source ?? "")) &&
      nodeIds.has(String(edge.target_node_id ?? edge.target ?? "")),
  );
  const [communities, communityCount] = await readTable(
    directory,
    "communities",
    GRAPH_REVIEW_ROW_LIMITS.communities,
  );
  const [evidence, evidenceCount] = await readTable(
    directory,
    "evidence",
    GRAPH_REVIEW_ROW_LIMITS.evidence,
  );
  const documents = await readOptionalTable(directory, "documents");
  const chunks = await readOptionalTable(directory, "chunks");
  const runReport = await readJsonObject(path.join(directory, "run_report.json"));
  runReport.app_full_counts = {
    node: nodeCount,
    edge: edgeCount,
    community: communityCount,
    evidence: evidenceCount,
    document: documents.length,
    chunk: chunks.length,
  };
  const graphId = String(
    runReport.graph_id ?? runReport.graphId ?? (nodes[0]?.graph_id as string | undefined) ?? path.basename(directory),
  );
  return {
    graphId,
    nodes,
    edges: visibleEdges.length > 0 ? visibleEdges : edges,
    communities,
    evidence,
    documents,
    chunks,
    runReport,
    graphMetrics: await readJsonObject(path.join(directory, "graph_metrics.json")),
  };
}

export function graphArtifactsExist(directory: string): boolean {
  return (
    existsSync(path.join(directory, "nodes.parquet")) && existsSync(path.join(directory, "edges.parquet"))
  ) || (
    existsSync(path.join(directory, "nodes.json")) && existsSync(path.join(directory, "edges.json"))
  );
}

async function readTable(
  directory: string,
  name: string,
  limit: number,
): Promise<[Record<string, unknown>[], number]> {
  const parquet = path.join(directory, `${name}.parquet`);
  const json = path.join(directory, `${name}.json`);
  if (existsSync(parquet)) {
    const rows = await readParquetRows(parquet, limit);
    const count = await countParquet(parquet);
    return [rows, count];
  }
  if (existsSync(json)) {
    const rows = await readJsonArray(json);
    return [rows.slice(0, limit), rows.length];
  }
  return [[], 0];
}

async function readOptionalTable(directory: string, name: string): Promise<Record<string, unknown>[]> {
  const [rows] = await readTable(directory, name, Number.MAX_SAFE_INTEGER);
  return rows;
}

async function readParquetRows(file: string, limit: number): Promise<Record<string, unknown>[]> {
  const { asyncBufferFromFile, parquetReadObjects } = await import("hyparquet");
  const buffer = await asyncBufferFromFile(file);
  const rows = await parquetReadObjects({
    file: buffer,
    rowEnd: limit,
  });
  return rows.map(normalizeRow);
}

async function countParquet(file: string): Promise<number> {
  const { asyncBufferFromFile, parquetReadObjects } = await import("hyparquet");
  const buffer = await asyncBufferFromFile(file);
  const rows = await parquetReadObjects({ file: buffer });
  return rows.length;
}

async function readJsonArray(file: string): Promise<Record<string, unknown>[]> {
  const value = JSON.parse(await readFile(file, "utf8")) as unknown;
  return Array.isArray(value) ? value.map((row) => normalizeRow(row as Record<string, unknown>)) : [];
}

async function readJsonObject(file: string): Promise<Record<string, unknown>> {
  if (!existsSync(file)) {
    return {};
  }
  const value = JSON.parse(await readFile(file, "utf8")) as unknown;
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    result[key] = normalizeValue(value);
  }
  return result;
}

function normalizeValue(value: unknown): unknown {
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(value)) {
    return value.toString("utf8");
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.map(normalizeValue);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nested]) => [key, normalizeValue(nested)]),
    );
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  return value;
}
