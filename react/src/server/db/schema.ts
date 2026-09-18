import { pgTable, text, jsonb, timestamp } from "drizzle-orm/pg-core";

export const flakegraphRun = pgTable("flakegraph_run", {
  id: text("id").primaryKey(),
  graphId: text("graph_id").notNull(),
  configJson: jsonb("config_json").notNull(),
  configDigest: text("config_digest").notNull(),
  status: text("status").notNull(),
  errorJson: jsonb("error_json"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull(),
});

export const flakegraphTask = pgTable("flakegraph_task", {
  id: text("id").primaryKey(),
  runId: text("run_id").notNull(),
  stage: text("stage").notNull(),
  scopeId: text("scope_id").notNull(),
  status: text("status").notNull(),
  leaseOwner: text("lease_owner"),
  updatedAt: timestamp("updated_at", { withTimezone: true }),
});
