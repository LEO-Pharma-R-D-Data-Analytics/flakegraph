# FlakeGraph Application

The Streamlit application is the operational interface for FlakeGraph. It
configures ingestion, performs preflight checks, submits work, follows document
and stage progress, shows Kubernetes workloads, and explores completed graphs.
The UI is a control plane: the processing engine remains in the FlakeGraph
worker image and all provider behavior remains behind the existing adapters.

```mermaid
flowchart LR
    user["Operator"] --> app["Streamlit control plane"]
    app --> local["Local CLI worker"]
    app --> fleet["Kubernetes task queue"]
    app --> snow["Snowflake job tables"]
    snow --> spcs["Asynchronous SPCS worker"]
    local --> output{"Output destination"}
    fleet --> output
    snow --> output
    output --> files["Portable local artifacts"]
    output --> graph["Canonical Snowflake KG tables"]
    files --> history["Run history"]
    graph --> history["Run history"]
    history --> active["Active run progress"]
    history --> explorer["Completed graph explorer"]
```

The sidebar is organized around graphs rather than application pages. **New
graph** opens ingestion. Every submitted run then remains available under
**Graphs**: active runs open live progress, while successful runs open the
integrated explorer with run details on a secondary tab. Kubernetes also exposes
a dedicated **Fleet** destination for infrastructure health and placement. Search
and storage filters keep larger histories manageable, and every entry identifies
whether its graph is stored as local artifacts or in Snowflake. Graphs can be
renamed from their workspace; the friendly name changes across their run-history
entries while the stable graph ID and stored data remain unchanged.

Run-wide finalization reports durable phase progress rather than one opaque task.
The active row identifies Spark startup, artifact reads, input materialization,
embeddings, entity resolution, graph assembly, enrichment, communities, quality
checks, partitioned table writes, destination publication, and manifest storage.

## Local

From the repository root:

```bash
uv tool install --python 3.13 "mineru[pipeline]==3.4.4"
uv sync --extra app --extra local-embeddings
uv run streamlit run app/streamlit_app.py
```

The local backend launches `uv run flakegraph` as a child process and reads its
JSON progress stream. App-owned uploads, generated profiles, run history, and
logs are kept under `.flakegraph/app/`, which is ignored by Git. Completed local
runs remain discoverable after the app restarts. Provider credentials stay in
environment variables; the generated profile contains only variable references.
Credential-reference fields are masked by default and can be revealed explicitly
with the eye control inside the field.

Every runtime starts from `configs/app-defaults.yaml`. This profile contains no
sample paths, graph IDs, dataset ontology, or benchmark quality exceptions. The
form overlays its source and provider selections, while the processing core
supplies the shared extraction, verification, resolution, and community defaults.

The **Destination** section is independent from the selected runtime. Local and
Kubernetes runs can write portable Parquet/JSON artifacts or bulk-publish the
final graph to Snowflake. Snowflake coordinates are stored with the generated run
profile, while password or OAuth values are resolved from the named environment
variable locally and from an explicitly mapped Kubernetes Secret in fleet mode.

Kubernetes mode uses the current `kubectl` context and the selected base profile.
The app submits work through `flakegraph distributed` and reads its bounded
status contract. The Fleet view lists every registered node with readiness,
hardware class, advertised NVIDIA GPU capacity, assigned FlakeGraph workers,
colocated vLLM model and image, and Metrics Server CPU and memory usage. Missing
Metrics Server or NVIDIA labels remove only those optional measurements; node and
workload health remain visible.
When a completed fleet run uses local output, its final distributed artifact is
exported into the recorded local directory on first inspection. Snowflake output
is published from Spark partitions with bounded memory and opens directly from
the canonical `KG_*` tables.
Set `FLAKEGRAPH_APP_KUBERNETES_CONFIG` to a fleet profile available to the app
process so the sidebar can list all recent runs from PostgreSQL after an app
restart or replica replacement. Without it, the app still lists runs submitted
through its local control-plane state.

Set `FLAKEGRAPH_APP_KUBERNETES_NAMESPACE` when FlakeGraph is installed outside
the default `flakegraph` namespace. The namespace can also be changed directly
from the Fleet view.

## Snowflake

The app is intentionally Python 3.11 compatible and does not import the Python
3.14 worker package. In Snowflake it uses the active Snowpark session to list and
upload internal-stage files, populate `KG_JOB` and `KG_JOB_FILE`, stage a
credential-free service specification, start an asynchronous SPCS job, monitor
queue progress, and query `KG_*` graph tables. The compute pool, stages, graph
tables, image repository, and worker image must exist as described in
[Snowflake setup](../docs/snowflake-setup.md); individual job services are
created by the app when ingestion starts.

`snowflake.yml` ships account-neutral defaults. Supply the target warehouse at
deploy time rather than editing the file, so one checkout serves several
accounts:

```bash
snow streamlit deploy flakegraph_app --replace --open \
  --env query_warehouse=YOUR_WAREHOUSE
```

Stage and identifier remain editable in the repository-level `snowflake.yml`
when an account needs different names.

The warehouse-runtime manifest pins Streamlit 1.52.2, a currently supported
Snowflake version. The graph canvas loads a bounded review projection for large
graphs while its summary metrics continue to report full table counts.
