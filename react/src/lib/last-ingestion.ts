import { useMemo } from "react";
import { readStorage, useStorageItem, writeStorage } from "./browser-storage";

const STORAGE_KEY = "flakegraph.last-ingestion";

export interface LastIngestionDraft {
  runtime: string;
  graphName: string;
  sourceKind: string;
  sourcePath: string;
  ocrProvider: string;
  llmProvider: string;
  embeddingProvider: string;
  savedAt: string;
}

function storageKey(runtime: string): string {
  return `${STORAGE_KEY}.${runtime}`;
}

function parseDraft(raw: string | null): LastIngestionDraft | null {
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as LastIngestionDraft;
  } catch {
    return null;
  }
}

export function readLastIngestion(runtime: string): LastIngestionDraft | null {
  return parseDraft(readStorage("local", storageKey(runtime)));
}

/** The last submitted draft for a runtime, kept current as it is rewritten. */
export function useLastIngestion(runtime: string): LastIngestionDraft | null {
  const raw = useStorageItem("local", storageKey(runtime));
  return useMemo(() => parseDraft(raw), [raw]);
}

export function writeLastIngestion(draft: LastIngestionDraft): void {
  writeStorage("local", storageKey(draft.runtime), JSON.stringify(draft));
}
