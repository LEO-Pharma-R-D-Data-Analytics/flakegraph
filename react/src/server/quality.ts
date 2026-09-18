import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { compareToGold, type GoldComparison, type GoldGraph } from "./gold";
import { graphCounts, type GraphDataset, type RunSnapshot } from "./protocol/schema";

const GOLD_HINTS: Array<{ pattern: RegExp; file: string }> = [
  { pattern: /martial/i, file: "data/martial_arts/gold.json" },
  { pattern: /deep/i, file: "data/deep_learning_papers/gold.json" },
];

export interface QualityReport {
  nodeCount: number;
  edgeCount: number;
  documentCount: number;
  documentsFailed: number;
  zeroEntityFailure: boolean;
  gold: GoldComparison | null;
  shareBlockedReason: string | null;
}

export async function qualityForGraph(options: {
  snapshot: RunSnapshot;
  dataset: GraphDataset;
  repositoryRoot: string;
}): Promise<QualityReport> {
  const counts = graphCounts(options.dataset);
  const gold = await loadGoldForGraph(
    `${options.snapshot.graphId} ${options.snapshot.graphName ?? ""}`,
    options.repositoryRoot,
  );
  const comparison = gold ? compareToGold(options.dataset, gold) : null;
  const zeroEntityFailure = counts.nodes === 0;
  let shareBlockedReason: string | null = null;
  if (zeroEntityFailure) {
    shareBlockedReason = "This run produced 0 entities, so it is not ready to share.";
  } else if (comparison && comparison.missingRequired.length > 0) {
    shareBlockedReason = `${comparison.missingRequired.length} required gold relations are missing.`;
  }
  return {
    nodeCount: counts.nodes,
    edgeCount: counts.edges,
    documentCount: counts.documents,
    documentsFailed: options.snapshot.documentsFailed ?? 0,
    zeroEntityFailure,
    gold: comparison,
    shareBlockedReason,
  };
}

async function loadGoldForGraph(haystack: string, repositoryRoot: string): Promise<GoldGraph | null> {
  for (const hint of GOLD_HINTS) {
    if (!hint.pattern.test(haystack)) {
      continue;
    }
    const file = path.join(repositoryRoot, hint.file);
    if (!existsSync(file)) {
      continue;
    }
    return JSON.parse(await readFile(file, "utf8")) as GoldGraph;
  }
  return null;
}
