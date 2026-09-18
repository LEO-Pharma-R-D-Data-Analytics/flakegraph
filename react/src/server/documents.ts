import type { ProgressEvent } from "./protocol/schema";

export type DocumentPhase =
  | "queued"
  | "running"
  | "scanned"
  | "skipped"
  | "ocr-failed"
  | "extracted-0"
  | "indexed"
  | "poison"
  | "cancelled";

export interface DocumentStatus {
  fileId: string;
  /** The source's own filename when the run recorded one. */
  name?: string;
  phase: DocumentPhase;
  detail: string;
}

/** The durable stages whose scope is one document, in pipeline order. */
export const DOCUMENT_STAGES = ["prepare_document", "extract_document_context", "compact_document"] as const;

interface DocumentTask {
  stage: string;
  scopeId: string;
  status: string;
  attempts: number;
  lastError: unknown;
  sourceMetadata: unknown;
}

/**
 * One status per document from a fleet run's task rows.
 *
 * A document is indexed once its compaction succeeded, poisoned or OCR-failed
 * by whichever of its tasks failed, in progress while one runs, and scanned
 * once parsed but not yet compacted. The prepare task's source artifact names
 * the file.
 */
export function documentStatusesFromTasks(tasks: readonly DocumentTask[]): DocumentStatus[] {
  const byFile = new Map<string, DocumentTask[]>();
  for (const task of tasks) {
    const rows = byFile.get(task.scopeId) ?? [];
    rows.push(task);
    byFile.set(task.scopeId, rows);
  }
  const statuses: DocumentStatus[] = [];
  for (const [fileId, rows] of byFile) {
    const name = sourceFilename(rows.find((row) => row.stage === "prepare_document")?.sourceMetadata);
    const failed = rows.find((row) => row.status === "failed");
    const status = (phase: DocumentPhase, detail: string) => ({ fileId, ...(name ? { name } : {}), phase, detail });
    if (failed) {
      const reason = errorMessage(failed.lastError) || `${titleStage(failed.stage)} failed`;
      statuses.push(
        failed.stage === "prepare_document"
          ? status("ocr-failed", reason)
          : status("poison", `${reason} (attempt ${failed.attempts})`),
      );
      continue;
    }
    if (rows.some((row) => row.status === "cancelled")) {
      statuses.push(status("cancelled", "Cancelled before it finished"));
      continue;
    }
    if (rows.some((row) => row.stage === "compact_document" && row.status === "succeeded")) {
      statuses.push(status("indexed", "Extracted and compacted into the graph"));
      continue;
    }
    const running = rows.find((row) => row.status === "running");
    if (running) {
      statuses.push(status("running", titleStage(running.stage)));
      continue;
    }
    if (rows.some((row) => row.stage === "prepare_document" && row.status === "succeeded")) {
      statuses.push(status("scanned", "Extraction pending"));
      continue;
    }
    statuses.push(status("queued", "Waiting for a worker"));
  }
  return statuses.sort((left, right) => (left.name ?? left.fileId).localeCompare(right.name ?? right.fileId));
}

/** Mark the files an operator quarantined, whichever runtime reported them. */
export function withSkippedFiles(statuses: readonly DocumentStatus[], skipped: readonly string[]): DocumentStatus[] {
  if (skipped.length === 0) {
    return [...statuses];
  }
  return statuses.map((item) =>
    skipped.includes(item.fileId)
      ? { ...item, phase: "skipped" as const, detail: "Quarantined. Retry will not send this file." }
      : item,
  );
}

function sourceFilename(metadata: unknown): string | undefined {
  if (!metadata || typeof metadata !== "object") {
    return undefined;
  }
  const input = (metadata as Record<string, unknown>).input_file;
  if (!input || typeof input !== "object") {
    return undefined;
  }
  const record = input as Record<string, unknown>;
  const name = record.filename ?? record.source_uri;
  return typeof name === "string" && name ? name : undefined;
}

function errorMessage(error: unknown): string {
  if (!error || typeof error !== "object") {
    return typeof error === "string" ? error : "";
  }
  const record = error as Record<string, unknown>;
  const message = record.error_message ?? record.message;
  const type = record.error_type;
  if (typeof message === "string" && message) {
    return typeof type === "string" && type ? `${type}: ${message}` : message;
  }
  return "";
}

function titleStage(stage: string): string {
  return stage.replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase());
}

export function documentStatusesFromEvents(
  events: readonly ProgressEvent[],
  skipped: readonly string[] = [],
): DocumentStatus[] {
  const byFile = new Map<string, ProgressEvent[]>();
  for (const event of events) {
    if (!event.fileId) {
      continue;
    }
    const rows = byFile.get(event.fileId) ?? [];
    rows.push(event);
    byFile.set(event.fileId, rows);
  }
  const statuses: DocumentStatus[] = [];
  for (const [fileId, rows] of byFile) {
    if (skipped.includes(fileId)) {
      statuses.push({ fileId, phase: "skipped", detail: "Quarantined. Retry will not send this file." });
      continue;
    }
    const failed = rows.find((row) => row.status === "failed" || row.status === "error");
    if (failed) {
      statuses.push({
        fileId,
        phase: failed.stage.includes("ocr") ? "ocr-failed" : "poison",
        detail: failed.message || "Failed during extraction",
      });
      continue;
    }
    const extractedZero = rows.find((row) => /0 entit/i.test(row.message ?? ""));
    if (extractedZero) {
      statuses.push({ fileId, phase: "extracted-0", detail: extractedZero.message || "Extracted 0 entities" });
      continue;
    }
    const extracted = rows.some((row) => row.stage.includes("extract") && row.status === "completed");
    statuses.push({
      fileId,
      phase: extracted ? "indexed" : "scanned",
      detail: extracted ? "Indexed" : "Seen by OCR",
    });
  }
  return statuses.sort((left, right) => left.fileId.localeCompare(right.fileId));
}
