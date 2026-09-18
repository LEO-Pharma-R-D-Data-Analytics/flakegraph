/**
 * How an evidence or chunk row names what it grounds and where it came from.
 *
 * The pipeline's parquet says `subject_id` with a `subject_kind` of node or
 * edge and a `file_id`; the gold format and older exports say `entity_id`,
 * `relation_id` and `document_id`. Every reader goes through here.
 */
export function evidenceEntityId(row: Record<string, unknown>): string | null {
  const explicit = row.entity_id ?? row.node_id;
  if (explicit != null && String(explicit)) {
    return String(explicit);
  }
  if (String(row.subject_kind ?? "") === "node" && row.subject_id != null) {
    return String(row.subject_id);
  }
  return null;
}

export function evidenceRelationId(row: Record<string, unknown>): string | null {
  const explicit = row.relation_id ?? row.edge_id;
  if (explicit != null && String(explicit)) {
    return String(explicit);
  }
  if (String(row.subject_kind ?? "") === "edge" && row.subject_id != null) {
    return String(row.subject_id);
  }
  return null;
}

export function evidenceDocumentId(row: Record<string, unknown>): string {
  const value = row.document_id ?? row.file_id ?? row.documentId;
  return value == null ? "" : String(value);
}

/** Document id → the name a reader knows it by: its filename, else its source, else the id. */
export function documentNameIndex(documents: readonly Record<string, unknown>[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const document of documents) {
    const name = documentDisplayName(document);
    for (const key of [document.id, document.file_id, document.document_id]) {
      if (key != null && String(key) && name) {
        names.set(String(key), name);
      }
    }
  }
  return names;
}

export function documentDisplayName(document: Record<string, unknown>): string {
  const filename = document.filename ?? document.name ?? document.title;
  if (filename != null && String(filename)) {
    return String(filename);
  }
  const uri = document.source_uri ?? document.path ?? document.uri;
  if (uri != null && String(uri)) {
    const tail = String(uri).split(/[\\/]/).filter(Boolean).pop();
    return tail || String(uri);
  }
  return String(document.id ?? document.file_id ?? "");
}

/**
 * Entity id → the document that grounds it: the file of its first evidence
 * quote, else the file of the first chunk it was extracted from.
 */
export function entityDocumentIndex(dataset: {
  nodes: readonly Record<string, unknown>[];
  evidence: readonly Record<string, unknown>[];
  chunks: readonly Record<string, unknown>[];
}): Map<string, string> {
  const documents = new Map<string, string>();
  for (const row of dataset.evidence) {
    const entityId = evidenceEntityId(row);
    const documentId = evidenceDocumentId(row);
    if (entityId && documentId && !documents.has(entityId)) {
      documents.set(entityId, documentId);
    }
  }
  const chunkDocument = new Map<string, string>();
  for (const chunk of dataset.chunks) {
    const id = chunk.id == null ? "" : String(chunk.id);
    const documentId = evidenceDocumentId(chunk);
    if (id && documentId) {
      chunkDocument.set(id, documentId);
    }
  }
  for (const node of dataset.nodes) {
    const id = node.id == null ? "" : String(node.id);
    if (!id || documents.has(id)) {
      continue;
    }
    const chunkIds = Array.isArray(node.source_chunk_ids) ? node.source_chunk_ids : [];
    for (const chunkId of chunkIds) {
      const documentId = chunkDocument.get(String(chunkId));
      if (documentId) {
        documents.set(id, documentId);
        break;
      }
    }
  }
  return documents;
}
