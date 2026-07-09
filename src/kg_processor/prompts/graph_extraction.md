Extract a grounded knowledge graph from the supplied chunks.
Return strict JSON with top-level `entities` and `relations` arrays.
Use only the configured entity types and relation types.
Every entity and relation must cite one of the valid `source_chunk_id` values.
When possible, include a short exact `quote` from the cited chunk plus
chunk-local `start_offset` and `end_offset` values for that quote.
Do not invent entities or relations that are not supported by the supplied text.
Return no prose outside the JSON object.
