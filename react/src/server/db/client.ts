import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { desc } from "drizzle-orm";
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
