import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { desc, inArray, sql } from "drizzle-orm";
import { appEnv } from "../env";
import * as schema from "./schema";

export function createDrizzle() {
  const url = appEnv().databaseUrl;
  if (!url) {
    return null;
  }
  const client = postgres(url, { max: 4, prepare: false });
  return drizzle(client, { schema });
}

export async function listPostgresRuns(limit = 100) {
  const db = createDrizzle();
  if (!db) {
    return [];
  }
  return db.select().from(schema.flakegraphRun).orderBy(desc(schema.flakegraphRun.updatedAt)).limit(limit);
}

/**
 * Per-run document counters from the durable stages: every discovered
 * document owns one prepare task, and a document is complete once its
 * compaction succeeds. One grouped query for every listed run.
 */
export async function documentCountsByRun(
  runIds: readonly string[],
): Promise<Map<string, { total: number | null; completed: number; failed: number }>> {
  const db = createDrizzle();
  const counts = new Map<string, { total: number | null; completed: number; failed: number }>();
  if (!db || runIds.length === 0) {
    return counts;
  }
  const rows = await db
    .select({
      runId: schema.flakegraphTask.runId,
      stage: schema.flakegraphTask.stage,
      status: schema.flakegraphTask.status,
      count: sql<number>`count(*)::int`,
    })
    .from(schema.flakegraphTask)
    .where(inArray(schema.flakegraphTask.runId, [...runIds]))
    .groupBy(schema.flakegraphTask.runId, schema.flakegraphTask.stage, schema.flakegraphTask.status);
  for (const row of rows) {
    const entry = counts.get(row.runId) ?? { total: null, completed: 0, failed: 0 };
    if (row.stage === "prepare_document") {
      entry.total = (entry.total ?? 0) + row.count;
      if (row.status === "failed") {
        entry.failed += row.count;
      }
    } else if (row.stage === "compact_document") {
      if (row.status === "succeeded") {
        entry.completed += row.count;
      } else if (row.status === "failed") {
        entry.failed += row.count;
      }
    }
    counts.set(row.runId, entry);
  }
  return counts;
}
