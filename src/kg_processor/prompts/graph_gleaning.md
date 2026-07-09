Find only additional grounded entities and relations missed by previous extraction passes.
Do not repeat records that are already accepted.
Every returned entity and relation must cite one of the valid `source_chunk_id` values.
If no additional grounded records are present, return empty `entities` and `relations` arrays.
Return no prose outside the JSON object.
