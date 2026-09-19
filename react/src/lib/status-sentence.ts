import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  isActiveStatus,
  isSuccessStatus,
  type RunSnapshot,
  type StageProgress,
} from "@/server/protocol/schema";

export function blockingStage(stages: readonly StageProgress[]): StageProgress | null {
  const active = stages.find((stage) => {
    const status = stage.status.toLowerCase();
    return status !== "completed" && status !== "succeeded" && status !== "skipped";
  });
  return active ?? stages.at(-1) ?? null;
}

export function statusSentence(
  snapshot: Pick<
    RunSnapshot,
    "status" | "error" | "documentsCompleted" | "documentsTotal" | "documentsFailed" | "stages"
  >,
  options: { nodeCount?: number | null; pendingReason?: string | null } = {},
): string {
  const status = snapshot.status.toLowerCase();
  const stage = blockingStage(snapshot.stages ?? []);
  const docs =
    snapshot.documentsTotal != null
      ? `${snapshot.documentsCompleted}/${snapshot.documentsTotal} documents`
      : `${snapshot.documentsCompleted} documents`;

  if (status === "cancelling") {
    return `Cancelling · workers may still be holding GPUs · wait, then inspect leftover leases`;
  }
  if (isActiveStatus(status)) {
    if (status === "pending" || status === "queued" || status === "planning") {
      return options.pendingReason
        ? `Queued · ${options.pendingReason}`
        : `Queued · waiting for a worker`;
    }
    // A listing carries no stages; the sentence then names the run's own
    // status rather than a stage it cannot see.
    if (!stage) {
      return `${humanize(status)} · ${docs}`;
    }
    return `${humanize(stage.status || status)} · ${humanize(stage.stage)} · ${docs}`;
  }
  if (status === "interrupted") {
    return `Interrupted · process gone, last progress frozen at ${docs}`;
  }
  if (status === ARTIFACTS_UNAVAILABLE_STATUS) {
    return `Catalog kept · files are not on this host`;
  }
  if (isSuccessStatus(status)) {
    if (options.nodeCount === 0) {
      return `Pipeline finished · 0 entities. OCR or schema likely failed`;
    }
    if ((snapshot.documentsFailed ?? 0) > 0) {
      return `Finished with gaps · ${snapshot.documentsFailed} documents failed`;
    }
    if (options.nodeCount) {
      return `Ready to explore · ${options.nodeCount} entities`;
    }
    return `Ready to explore · ${docs}`;
  }
  if (status === "cancelled") {
    return `Cancelled · no further workers will be claimed`;
  }
  const reason = snapshot.error?.trim() || "see run details";
  return `Failed · ${reason}`;
}

export function catalogEmptyCopy(options: {
  loading: boolean;
  total: number;
  visible: number;
  search: string;
  storageFilter: string;
  mine: boolean;
  identified: boolean;
  analyst?: boolean;
}): { title: string; detail: string } | null {
  if (options.loading) {
    return null;
  }
  if (options.mine && !options.identified) {
    return {
      title: "0 yours",
      detail: `${options.total} graphs in this catalog. Unidentified sessions cannot claim ownership — switch to All to browse.`,
    };
  }
  if (options.mine && options.visible === 0) {
    return {
      title: "0 yours",
      detail: `${options.total} graphs in this catalog. None list you as owner — switch to All to browse shared or unowned graphs.`,
    };
  }
  if (options.total === 0) {
    return {
      title: "No graphs yet",
      detail: "Start with the martial-arts sample or drop a folder of documents.",
    };
  }
  if (options.search.trim() && options.visible === 0) {
    return {
      title: `No graphs match ‘${options.search.trim()}’`,
      detail: "Clear search to see the rest of this catalog.",
    };
  }
  if (options.storageFilter !== "all" && options.visible === 0) {
    return {
      title: `Filters hide all ${options.total} graphs`,
      detail: "Switch storage to All to see graphs on other destinations.",
    };
  }
  if (options.analyst && options.visible === 0) {
    return {
      title: "No production graphs",
      detail: "This catalog has no gold-passed perspective yet. Ask a builder to publish one after gold.",
    };
  }
  return null;
}

function humanize(value: string): string {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
