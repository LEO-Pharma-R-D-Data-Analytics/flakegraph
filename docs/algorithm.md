# How FlakeGraph Builds A Knowledge Graph

FlakeGraph converts source documents into canonical nodes and evidence-backed
edges through one provider-neutral processing contract. The same logical stages
run locally, across a Kubernetes fleet, or with Snowflake providers and storage.

[![Detailed FlakeGraph pipeline showing document preparation, entity-first relation extraction, corpus finalization, provider adapters, and runtime mappings](assets/flakegraph-pipeline.svg)](assets/flakegraph-pipeline.svg)

## Processing Stages

| Stage | Input | Work performed | Durable result |
| --- | --- | --- | --- |
| Prepare | Source files | OCR or text extraction, layout normalization, stable document identity, bibliography-aware chunking, and bounded document windows | Documents, pages, blocks, assets, and chunks with source offsets |
| Document context | Bounded front matter | Identify ontology-declared focal subjects such as the paper, method, product, or organization once per document; profile a dataset instead (catalogue, measurements, or other) | Reusable grounded context mentions for every body window; a dataset's kind and summary |
| Entity extraction | Bounded document windows | Extract grounded entity mentions in parallel, with bounded gleaning and cue-based audits | Independently mergeable mention observations |
| Entity inventory | Every entity window for one document | Deduplicate mentions into one immutable document-wide endpoint vocabulary | A complete entity inventory shared by relation windows |
| Relation extraction | Bounded document windows plus the document inventory | Extract, ground, and optionally verify relations in parallel | Independently mergeable relation observations with quotes, stated values, confidence, and provenance |
| Finalize | All observation shards in the corpus | Resolve identity, assemble canonical edges, remove invalid or isolated records, merge descriptions, embed graph records, detect communities, and evaluate quality | Canonical graph tables, evidence, communities, metrics, and traces |
| Publish | Validated graph tables | Write a complete version and atomically move the graph head | Local artifacts, Snowflake tables, or an exported distributed graph |

## The Two-Pass Contract

“Two-pass” describes two parallel phases per document. It is not two complete
serial runs over the corpus.

1. **Entity pass.** Independently queueable windows emit typed mentions grounded
   to chunk spans. FlakeGraph applies ontology constraints and can run bounded
   gleaning or cue-based audits for missed mentions. Local threads or fleet
   workers process these windows concurrently.
2. **Inventory barrier.** FlakeGraph deduplicates all accepted mentions for one
   document. This is a metadata-only operation with no provider calls.
3. **Relation pass.** Every window receives the complete document inventory and
   a table of which relations each of its entity types may start and end. A
   relation references accepted IDs, so it can connect an endpoint introduced
   in another window. With `graph.relation_new_endpoints` (off by default),
   when its other end is a named thing of an ontology type the inventory lacks,
   the response may add it as a new endpoint (up to six a call); an added
   endpoint is grounded like any entity and joins the inventory, even if the
   relation that needed it is rejected. Relation windows
   remain independent and run concurrently. Each observation carries source
   evidence, and candidates requiring semantic judgment pass through the
   configured verifier.

Before these passes, the document-context stage inspects bounded front matter.
It makes document-level subjects available to body windows without repeating
the title and contribution framing in every model call. Bibliography-only
windows are excluded from ordinary extraction. Kubernetes packs a small bounded
number of logical windows into each lease to reduce queue and object-store
cardinality; model calls remain separate and use bounded in-task parallelism.

## Evidence Grounding

Every entity and relation keeps an exact span of source text as its evidence.
Grounding is the deterministic check - string matching, no model call - that
makes this true. The model returns a quote and names the chunk it came from;
grounding then:

1. Looks for the quote in that chunk, tolerating differences in whitespace,
   letter case, compatibility characters such as ligatures, and line-break
   hyphens (a quote may keep the hyphen or drop it), but never a changed or
   missing word. HTML character references a parser left in the text ("&amp;")
   read as their characters, and table markup - HTML cell tags and cell bars -
   reads as whitespace, so a quote that reads across a table row as prose
   ("Product Name: POLAWAX") matches the cells it spans.
2. For a relation, requires each endpoint's name, alias, or local surface as a
   whole-word match inside the quote, the two mentions not overlapping. Letters
   and digits within a name may be written together or apart ("GG918",
   "GG-918"); a name's last word may carry a plural or genitive ending ("LEO
   Pharmas", "tablets"); an underscore separates words, as in file names and
   codes ("SOP_004858"). A name that contains the other endpoint's name is
   evidence of their relation on its own: "10% urea foam" is a foam.
3. Otherwise finds the shortest run of one to three sentences in that chunk
   that names both endpoints. A sentence ends at a line break, at the end of an
   HTML table row, or at terminal punctuation followed by whitespace, so a
   decimal such as "0.10" stays whole.
4. Otherwise accepts a quote given in pieces joined by "...", when every piece
   occurs exactly in the chunk and the pieces together name both endpoints; the
   evidence is the exact stretch of source from the first piece to the last.
   This is how a form, certificate, label, or table states a fact: the subject
   in a header, the value in a field below.
5. Drops the record as `ungrounded_quote`.

An entity is grounded the same way, with its name in the quote. When the model
put the name in its own words - "pH measurement" for "the pH of the foams was
measured" - the entity is kept if its quote is verbatim and every word of the
name is a word of the quote, short words exactly and longer ones up to their
ending (`repaired_rephrased_name`).

An entity type marked `document_subject` names what a document itself is or
describes - a paper's own title, the material a certificate or data sheet
covers. When the document-context pass finds such an entity, a statement about
it may leave its name out: its field or sentence grounds the target alone.

With `graph.llm_grounding_fallback`, relations still ungrounded get one more
chance: one call per window shows the cited text as numbered units (the same
sentences, rows, and lines) and the model answers only with unit numbers. Code
confirms both endpoints occur in the chosen units, and the evidence is the exact
source stretch they span, recorded with `evidence_method: llm_located`. When the
chosen units name one endpoint, the stretch may extend within the chunk to the
nearest unit naming the other - a paragraph's topic and a detail several
sentences later - bounded like a quote in pieces (`llm_grounding_extended`). The
verifier then judges these relations like any other.

With `graph.entity_grounding_fallback`, entities string grounding could not
place get the same treatment: the window's chunks are shown as numbered units,
the model points at the lines that name each one, and code requires those lines
to lie in one chunk and share a word stem with the name. The evidence is the
exact stretch; it is `llm_located` when it contains the name and
`llm_confirmed` when it states the thing in other words ("stable for 4 weeks"
for stability). Every evidence row records its `method` - `llm`,
`llm_located`, `llm_confirmed`, or `ontology_cue` - so a consumer can keep only
evidence that names its subject verbatim.

With `graph.relation_relabel`, a statement whose relation does not allow its two
entity types is offered, in one call per window, the relations the ontology
does allow between them in either direction; the model picks one or none. A
relabelled statement passes every check again and keeps the type the model
first gave it as `proposed_relation_type`. One it declines is rejected as
`domain_or_range_violation`.

An ontology may name a `fallback_relation`, typically `RELATED_TO`. A
statement whose own relation does not allow its two entity types is then kept
under the fallback, and the type the model stated stays on the observation as
`proposed_relation_type`; the console shows it on the edge. Without one, such a
statement is rejected as `domain_or_range_violation`.

Every record validation turns away is listed in the `rejected_records` table:
its document, page, reason, and what it stated - an entity with its type, or a
relation with both ends, their types, and its quote. Only a repeat of a record
the graph kept is left out. The run report summarizes them by reason and names
the relation type rules the model broke most, with an example of each, so an
ontology's author can see which rules cost statements.

A relation call that returns its full record limit is continued with what it
has already found, and an empty answer over a large inventory is retried once
with another seed (`relation_continuation_max_passes`,
`relation_empty_retry_min_entities`).

## Values On Relations

A relation type marked `quantities` carries the values its relation states: a
limit, a range, a result. Each is kept with its exact text, a kind, a unit, a
comparator, and bounds parsed from the text, on the relation's observation. A
value whose text the evidence does not contain is dropped. Once an ontology has
such a relation type, an entity whose name is only a value - "≤ 0.10 %",
"154 - 162" - is rejected, except under a type marked `identifier`, whose names
are codes or numbers (a batch or lot number).

## Folders And File Names

Where a file sits says what it belongs to: an image named `IMG_0392.jpg` in
`Formulation Lab/Trial 3/Pictures/` belongs to that trial. With
`files.location_page` (on by default) every document starts with a page 0 that
states its location relative to the source root - its folders and its file
name. Extraction reads that page like any other text: a folder or file name that
names a project, an experiment, a material, or a batch becomes an entity grounded
in it, a folder's thing is part of its parent folder's thing, and the
document-context pass may take a document's subject from its file name. The
document row records the `folder`.

## Documents, Datasets, And Images

Each source file is a document, a dataset, or an image, by its type. A dataset -
a spreadsheet or delimited table - is profiled from its name and opening rows:

| Profile | Entity pass | Relation pass |
| --- | --- | --- |
| `catalogue`: rows name things of the ontology's types | yes | no |
| `measurements`: rows record values for named things | yes | yes |
| `other`: calculations, templates, schedules, forms | yes | no |

An image keeps the picture itself as its asset. One with no text to read - or
whose parse holds only a reference to the picture - is an image document with no
pages, not a failed parse; its location page still places it. Every dataset
and image lists the related files beside it: the same folder's documents that
share an identifier (a batch or order number) in their names, or the whole
folder when it is small. An image in a folder of its own (a `Pictures`
subfolder) is related to the documents one level up.

A PDF whose text layer interleaves two columns line by line is read by the
fallback OCR provider's secondary parser, which reads the page layout
(`ocr.fallback_max_interleaved_column_page_ratio`).

Extraction deliberately stops at observations. Mentions such as “BERT,” “the
BERT model,” and a longer formal name remain distinct until the complete corpus
is available for resolution. A name ending in a parenthesized abbreviation -
“Sodium lauryl sulphate (SLS)” - is recorded as the name with the abbreviation
as an alias; a mark such as (TM) or a grade such as (PEG 400) stays in the name.
Within one chunk a name names one thing: when a response types the same name
twice there, the first type stands.

## Corpus Finalization

Finalization turns mergeable observations into a graph:

1. Filter malformed, low-confidence, ungrounded, and ontology-invalid records.
2. Generate bounded identity candidates using aliases, normalized lexical
   forms, and semantic embeddings.
3. Auto-merge only high-confidence candidates and use optional LLM adjudication
   for ambiguous pairs.
4. Compute identity components and choose stable canonical names, types, IDs,
   aliases, and descriptions.
5. Map relation endpoints to canonical node IDs, remove disallowed self-loops,
   aggregate repeated triples, and preserve every supporting evidence record.
6. Optionally remove isolated entities, then calculate node degree and graph
   structure.
7. Merge source-specific descriptions and generate embeddings for retrieval.
8. Detect communities from canonical relations plus a bounded, linear-size
   co-mention topology; generate grounded reports and embed those reports.
9. Run evidence, endpoint, ontology, uniqueness, and structural quality checks.
10. Publish a complete graph version atomically, so readers never observe a
    partially finalized graph.

Local execution performs this contract in one process. Kubernetes stores
immutable stage shards in S3-compatible storage and uses Spark for partitioned
finalization. Snowflake execution can select Cortex OCR, LLM, and embedding
adapters and publish through direct or staged Snowflake writers.

## Replaceable Providers

The algorithm depends on ports rather than vendor SDKs. File sources, OCR, LLM,
embedding, cache, and writer adapters are selected independently in YAML. A
local run can therefore use MinerU with vLLM and write local artifacts, while a
different deployment uses Snowflake stages, Cortex, and `KG_*` tables without
forking the processing rules.

The implementation boundaries and provider extension contract are documented
in [Architecture](architecture.md). Fleet scheduling, leases, autoscaling, and
Spark execution are documented in [Kubernetes fleet deployment](kubernetes-fleet.md).
