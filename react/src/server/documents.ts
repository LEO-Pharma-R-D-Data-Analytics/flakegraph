import type { ProgressEvent } from "./protocol/schema";

export type DocumentPhase = "scanned" | "skipped" | "ocr-failed" | "extracted-0" | "indexed" | "poison";

export interface DocumentStatus {
  fileId: string;
  phase: DocumentPhase;
  detail: string;
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
