import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 10 || unit === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) {
    return "unknown";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const deltaSeconds = Math.round((date.getTime() - Date.now()) / 1000);
  const abs = Math.abs(deltaSeconds);
  const formatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (abs < 60) {
    return formatter.format(deltaSeconds, "second");
  }
  if (abs < 3600) {
    return formatter.format(Math.round(deltaSeconds / 60), "minute");
  }
  if (abs < 86400) {
    return formatter.format(Math.round(deltaSeconds / 3600), "hour");
  }
  return formatter.format(Math.round(deltaSeconds / 86400), "day");
}

export function formatCount(value: number | null | undefined): string {
  return Number(value ?? 0).toLocaleString();
}

export function titleCaseStage(stage: string): string {
  return stage
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const DOCUMENT_PHASES: Record<string, string> = {
  "ocr-failed": "OCR failed",
  "extracted-0": "No text extracted",
  skipped: "Skipped",
  poison: "Poisoned file",
  queued: "Queued",
  running: "In progress",
  extracted: "Text extracted",
  embedded: "Embedded",
  done: "Done",
};

export function formatDocumentPhase(phase: string): string {
  return DOCUMENT_PHASES[phase] ?? titleCaseStage(phase);
}

export function documentPhaseNeedsSkip(phase: string): boolean {
  return phase === "poison" || phase === "ocr-failed" || phase === "extracted-0";
}
