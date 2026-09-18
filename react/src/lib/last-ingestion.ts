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

export function readLastIngestion(runtime: string): LastIngestionDraft | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(`${STORAGE_KEY}.${runtime}`);
    if (!raw) {
      return null;
    }
    return JSON.parse(raw) as LastIngestionDraft;
  } catch {
    return null;
  }
}

export function writeLastIngestion(draft: LastIngestionDraft): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(`${STORAGE_KEY}.${draft.runtime}`, JSON.stringify(draft));
}
