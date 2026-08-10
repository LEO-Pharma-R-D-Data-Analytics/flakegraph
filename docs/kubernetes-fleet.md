# Kubernetes Fleet Deployment

FlakeGraph uses Kubernetes for scheduling, PostgreSQL for durable task leases,
KEDA for queue-driven worker scaling, S3-compatible storage for immutable
artifacts, and Spark for graph-wide finalization. Providers may run in worker
containers, in chart-managed vLLM pods, or behind external endpoints.

```mermaid
flowchart LR
    submitter["Submitter"] --> database["PostgreSQL tasks and graph versions"]
    database --> keda["KEDA worker demand"]
    keda --> prepare
    keda --> extract
    keda --> finalizer
    submitter --> objects["S3-compatible artifacts"]
    prepare["Prepare workers"] <--> database
    contextQueue["Document-context tasks"] --> extract["Shared extraction worker pool"]
    windowQueue["Window-extraction tasks"] --> extract
    extract <--> database
    prepare <--> objects
    extract <--> objects
    extract --> models["vLLM or external LLM"]
    finalizer["Finalization coordinator"] --> spark["Spark executors"]
    finalizer --> database
    spark <--> objects
    spark --> models
```

## Components

| Component | Owner |
| --- | --- |
| Prepare, context/extract, and finalize workers | FlakeGraph Helm chart |
| Queue-driven scale-to-zero | KEDA PostgreSQL scaler |
| Spark driver, executor template, and RBAC | FlakeGraph Helm chart |
| Local vLLM service | Optional FlakeGraph StatefulSet |
| PostgreSQL | Managed service or optional CloudNativePG cluster |
| S3-compatible storage | External managed or self-hosted service |
| OCR, embedding, and external model services | Selected provider deployment |
| GPU drivers and NVIDIA device plugin | Cluster administrator |

FlakeGraph does not install object storage or hide its lifecycle inside the
application chart. PostgreSQL stores small coordination records; source bytes,
stage shards, and graph tables move through object storage.

## Work Distribution

Workers pull compatible tasks rather than receiving fixed document partitions.
For 100 documents, submission creates 100 preparation tasks. Each preparation
result adds one context task; that task identifies the source document once and
then adds extraction tasks for its actual body windows. Context and window tasks
share the extraction worker pool, and idle workers continuously claim the next
eligible item.

```mermaid
flowchart LR
    documents["100 documents"] --> prepareQueue["100 preparation tasks"]
    prepareQueue --> preparePool["Preparation workers"]
    preparePool --> contextQueue["100 document-context tasks"]
    contextQueue --> extractionPool["Extraction workers"]
    contextQueue --> entityQueue["Batched entity-window tasks"]
    entityQueue --> extractionPool
    extractionPool --> inventoryQueue["One entity-inventory task per document"]
    inventoryQueue --> relationQueue["Batched relation-window tasks"]
    relationQueue --> extractionPool
    extractionPool --> compactionQueue["One document compaction task"]
    compactionQueue --> compactionPool["Extraction workers, compaction stage"]
    compactionPool --> barrier{"All document shards complete?"}
    barrier -- "Yes" --> driver["Leased finalization coordinator"]
    driver --> spark["Partitioned Spark finalization"]
    spark --> version["Atomically published graph version"]
```

`prepare_document` performs OCR, normalization, and chunking.
`extract_document_context` identifies ontology-declared focal entities from
bounded front matter and removes a validated bibliography suffix.
`extract_entity_window` produces grounded mentions. A bounded lease may contain
several logical windows, whose provider calls remain independently parallel.
`compact_entity_inventory` deduplicates every mention for the document and fans
out `extract_relation_window` tasks. Each relation window receives that complete
inventory, so it can connect endpoints discovered elsewhere in the document.
`compact_document` assembles relation observations and the inventory into one
immutable document shard. Both compaction stages perform no model inference.
`finalize_graph` starts Spark jobs for identity resolution, graph
assembly, connected-entity selection, quality checks, embeddings, communities,
and versioned Parquet output. Its leased coordinator persists one bounded
phase-progress record, so status clients can show the current operation and
table-write counter without scanning Spark logs or graph-sized task data.

Task claims use renewable PostgreSQL leases. A stopped worker's task becomes
claimable after lease expiry, up to the configured attempt budget. Finalization
uses the complete corpus because independent document-level graphs cannot
resolve identities or communities consistently.

The default five-minute task lease is renewed once per minute and does not cap
task duration. It bounds hard-node-loss recovery while allowing long OCR and
Spark work to continue for as long as its worker remains healthy.

KEDA queries dependency-ready queued tasks and active leases for each worker
pool. The query stops once it reaches that pool's configured maximum useful
demand: counting the remaining millions of tasks cannot request more replicas
and would only load PostgreSQL. Active leases remain in the bounded demand, so
KEDA can remove idle pods without dropping the capacity that owns in-flight
work. After extraction drains, the prepare and extract pools reach zero and
Spark executors receive the fleet's CPU and memory. Spark has higher scheduling
priority than workers as a fallback during the short autoscaler convergence
window; model servers remain higher priority than both.

## Prerequisites

- Kubernetes 1.34 or newer with reliable cross-node networking and DNS
- KEDA 2.20.1
- PostgreSQL 14 or newer
- S3-compatible storage reachable by workers and Spark executors
- Worker and Spark images for every node architecture
- NVIDIA drivers and device plugin when local GPU providers are enabled
- A registry or mirror reachable by every node
- Provider and storage credentials stored in Kubernetes Secrets

For high availability, use at least three control-plane/database members and
spread control-plane, DNS, database, model, and storage replicas across failure
domains. Kubernetes server nodes remain ordinary schedulable workers unless an
operator deliberately taints them, so control-plane redundancy does not remove
three DGX systems from the compute pool.

### Create A K3s Cluster

Kubernetes installation is infrastructure ownership rather than a FlakeGraph
runtime concern. For a new bare-metal fleet, K3s provides a compact supported
path. Use its current
[requirements](https://docs.k3s.io/installation/requirements) and
[embedded-etcd HA](https://docs.k3s.io/datastore/ha-embedded) guides as the
source of truth. The production shape is:

1. Give every host a unique stable name and address, synchronize time, and make
   the documented Kubernetes, etcd, DNS, and overlay-network ports reachable.
2. Put a stable DNS name or load balancer in front of three K3s server nodes.
3. Initialize the first server with embedded etcd, then join two more servers.
4. Join the remaining hosts as agents. Do not taint the three servers when all
   DGX systems should also execute FlakeGraph workloads.
5. Install the NVIDIA GPU Operator or a validated equivalent device-plugin and
   container-runtime integration.

The commands below show the K3s topology. Keep the token in a secret manager,
pin `INSTALL_K3S_VERSION` to the release validated by the site, and replace the
stable API endpoint rather than using one server's transient address:

```bash
# First of three server nodes
curl -sfL https://get.k3s.io | \
  K3S_TOKEN="$K3S_TOKEN" INSTALL_K3S_VERSION="$K3S_VERSION" \
  sh -s - server --cluster-init --tls-san "$K3S_API_HOST"

# Second and third server nodes
curl -sfL https://get.k3s.io | \
  K3S_TOKEN="$K3S_TOKEN" K3S_URL="https://$K3S_API_HOST:6443" \
  INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - server \
  --tls-san "$K3S_API_HOST"

# Every remaining worker node
curl -sfL https://get.k3s.io | \
  K3S_TOKEN="$K3S_TOKEN" K3S_URL="https://$K3S_API_HOST:6443" \
  INSTALL_K3S_VERSION="$K3S_VERSION" sh -
```

Private-overlay deployments may additionally need K3s `node-ip`, advertised
address, and Flannel interface settings. Those depend on the chosen network and
must be identical in intent across every host; verify cross-node pod, service,
DNS, API-server, and etcd connectivity before installing FlakeGraph.

Use capability labels rather than hostnames in public values files:

```bash
kubectl label node <gpu-node> flakegraph.io/node-class=nvidia-spark
kubectl label node <cpu-node> flakegraph.io/node-class=cpu-control
```

### Scripted Bring-Up For NVIDIA DGX Spark Nodes

`deploy/spark/` automates the topology above for GB10 hardware, and is worth
reading even if the fleet runs on something else, because two of the problems it
solves are properties of corporate networks rather than of this hardware.

| Script | Runs as | Does |
| --- | --- | --- |
| `stage-artifacts.sh` | operator workstation | Fetches k3s, verifies its published SHA-256, copies it to the node |
| `bootstrap-node.sh` | root, on the node | Container runtime, NVIDIA runtime, k3s, node labels |
| `install-cluster.sh` | operator, on the node | Device plugin, KEDA, CloudNativePG, object storage |

`bootstrap-node.sh --role server` starts embedded etcd rather than the k3s
default, because a default single-node server uses SQLite and can never gain a
second control-plane node — a decision that cannot be revisited later without
rebuilding the cluster. Further nodes take `--role agent`. It is idempotent, so
re-running it verifies a node rather than disturbing it.

Two traps it exists to handle:

**A CIS-hardened image may blacklist `overlay`.** Containers cannot run without
it; both Docker's `overlay2` driver and the containerd k3s embeds mount it for
every layer. Worse, a blacklist written as `install overlay /bin/true` makes
`modprobe overlay` **exit successfully while loading nothing**. Verify with
`grep -w overlay /proc/filesystems`, never with `modprobe`'s exit status.

**A TLS-inspecting network needs its root CA on every node.** Where a proxy
re-signs certificates, a host without those roots fails every download in a way
that reads as a firewall block — transfers stop at zero bytes and return a
redirect to the proxy's own page. Install the roots before concluding that a host
is blocked. Containers carry their own trust stores and need the certificate
copied in separately, and Python is affected worse than most: `curl` verifies
against the system store while `httpx` and `requests` verify against `certifi`,
so the same URL succeeds under one and fails under the other.

## Build Images

A mixed fleet needs both `linux/amd64` and `linux/arm64` manifests:

```bash
docker buildx create --name flakegraph-builder --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag registry.example.com/flakegraph:0.1.0 \
  --push .

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.spark \
  --tag registry.example.com/flakegraph-spark:0.1.0 \
  --push .
```

Pin the resulting image digests in production values.

## PostgreSQL And Storage

For an external database, set `database.secretName` and `database.secretKey` to
a Secret containing a PostgreSQL URI. To use CloudNativePG, install its pinned
operator chart and set `database.cloudNativePG.enabled=true`; the FlakeGraph
chart then creates the declared cluster and uses its application Secret.

Set these object-storage values:

- `artifactStorage.uri`
- `artifactStorage.endpointUrl`
- `artifactStorage.existingSecret`
- `artifactStorage.region`

Before processing data, test put, get, checksum verification, and delete from
both the worker and Spark images. Back up PostgreSQL and object storage as one
system: PostgreSQL contains task/version metadata and object storage contains
the referenced payloads.

### Capacity Sizing

Size PostgreSQL for durable coordination records, indexes, retries, and retained
run versions rather than source-file bytes. A useful first estimate is:

```text
database capacity = projected live metadata x 2 for indexes x 2 for operating headroom
```

Measure `pg_total_relation_size` after a representative sample before loading the
complete corpus. Keep at least 50% free during a run so PostgreSQL can vacuum,
build indexes, and absorb retry or version overlap. The CloudNativePG profile
therefore starts at 100Gi; increase it for corpora whose representative sample
projects beyond 50Gi of live tables and indexes.

Object storage holds source documents, prepared OCR, extraction artifacts, and
partitioned graph tables. Project it independently from a representative sample,
include every retained graph version, and apply the storage service's replication
factor. Object-store capacity commonly exceeds database capacity by an order of
magnitude for PDF-heavy corpora.

## Configure And Install

Keep site-specific values, endpoints, and Secret references outside Git under
the ignored `deploy/private/` directory:

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --version 2.20.1 \
  --set operator.replicaCount=2 \
  --set metricsServer.replicaCount=2 \
  --set webhooks.replicaCount=2 \
  --wait

cp deploy/examples/k3s-spark-values.yaml deploy/private/fleet-values.yaml
cp configs/app-defaults.yaml deploy/private/fleet-config.yaml

helm upgrade --install flakegraph deploy/helm/flakegraph \
  --namespace flakegraph \
  --create-namespace \
  --values deploy/private/fleet-values.yaml \
  --set-file config.content=deploy/private/fleet-config.yaml \
  --set-file ontology.content=configs/ontologies/general.yaml \
  --wait
```

The release is not reported ready until an idempotent database-bootstrap Job
has validated the mounted processing configuration, provider credentials,
provider binaries, and ontology, then applied the coordination schema used by
workers and KEDA. A missing credential, invalid provider setup, unreachable
database, or incompatible schema therefore fails the Helm operation instead of
leaving a superficially installed but inert worker fleet. Paths supplied only by
worker data volumes are intentionally deferred because the bootstrap hook does
not mount corpus or output storage.

Provider Secrets use an explicit environment-variable allowlist. Keep provider
and model identity in the mounted config, and map only credentials from the
Secret:

```yaml
providerSecret:
  name: flakegraph-providers
  optional: false
  env:
    - name: KG_LLM_API_KEY
      key: KG_LLM_API_KEY
      optional: false
    - name: KG_EMBED_API_KEY
      key: KG_EMBED_API_KEY
      optional: true
    - name: KG_SNOWFLAKE_PASSWORD
      key: KG_SNOWFLAKE_PASSWORD
      optional: true
    - name: KG_SNOWFLAKE_OAUTH_TOKEN
      key: KG_SNOWFLAKE_OAUTH_TOKEN
      optional: true
```

The chart never imports a whole Secret with `envFrom`, so an unrelated key such
as `KG_LLM_MODEL` or `KG_EMBED_MODEL` cannot silently replace reviewed YAML.
The Snowflake entries are optional until a submitted run selects Snowflake as its
output. The final task carries only database, schema, stage, and credential-key
names; the claiming worker resolves the credential value from this Secret.

Use the same processing config for submission and workers. Worker identity,
eligible stages, replica counts, and lease timing may differ; extraction,
ontology, model, and graph semantics must match the submitted run.

The chart creates independent preparation, extraction, and finalization
Deployments plus one KEDA `ScaledObject` per pool. PostgreSQL remains the source
of truth; KEDA only adjusts capacity and never owns task state. For an external
database, the URI in `database.secretName` must contain a fully qualified host
reachable from the KEDA namespace. The CloudNativePG profile configures its
fully qualified service automatically.

## Local Model Serving

Set `modelServing.enabled=true` to run one pinned vLLM server per configured
StatefulSet replica. The chart provides:

- one persistent model cache per replica;
- anti-affinity, health probes, and a PodDisruptionBudget;
- a cluster-wide Service for ordinary workers and Spark;
- a node-local Service for colocated extraction workers; and
- `KG_LLM_ENDPOINT` and `KG_LLM_MODEL` injection into workers.

When a pool sets `localModelServing: true`, the chart adds required pod affinity
to this release's model-serving pods using `modelServing.topologyKey`. Workers
therefore cannot schedule on a node where the node-local Service has no endpoint.
Helm rejects local mode when model serving is disabled, and rejects a custom
`podAffinity` that would conflict with the chart-owned rule. Operators may still
set `nodeAffinity`, `podAntiAffinity`, topology spreading, and node selectors.

The default profile serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with the pinned NVIDIA
vLLM image, model revision, FP8 KV cache, FlashInfer attention, Marlin MoE,
chunked prefill, prefix caching, asynchronous scheduling, and MTP speculative
decoding. The `vllm_local` adapter disables free-form reasoning for strict JSON
extraction.

The default `modelServing.server.gpuMemoryUtilization` is `0.50`. On unified-
memory systems, CUDA allocations are not fully represented by Kubernetes pod
memory metrics, and the same physical memory must also accommodate workers,
Spark executors, storage, and system services. Increase this value only for
dedicated inference nodes or after validating finalization under peak load.

```yaml
modelServing:
  enabled: true
  replicas: 4
  nodeSelector:
    flakegraph.io/node-class: nvidia-spark

workers:
  extract:
    # Two workers * two concurrent windows fill each four-sequence server.
    replicas: 8
    autoscaling:
      maxReplicas: 8
    localModelServing: true
    nodeSelector:
      flakegraph.io/node-class: nvidia-spark
```

The public checkpoint does not require authentication. Configure
`modelServing.huggingFaceTokenSecret.name` when a token is needed for rate limits
or a replacement model.

No separate inference router is required. The cluster-wide Service distributes
ordinary traffic, while the node-local Service keeps colocated workers attached
to their local model pod. An external gateway remains supported by disabling
`modelServing` and selecting an OpenAI-compatible endpoint in the normal
provider config.

Validate model identity and acceleration before submitting work:

```bash
kubectl -n flakegraph rollout status statefulset/flakegraph-flakegraph-vllm --timeout=60m
kubectl -n flakegraph get pods,pvc,service -l app.kubernetes.io/instance=flakegraph
kubectl -n flakegraph get --raw \
  /api/v1/namespaces/flakegraph/services/http:flakegraph-flakegraph-vllm:8000/proxy/v1/models
kubectl -n flakegraph logs flakegraph-flakegraph-vllm-0 -c vllm \
  | grep -E 'NVFP4|MARLIN|FLASHINFER'
kubectl -n flakegraph exec flakegraph-flakegraph-vllm-0 -c vllm -- \
  nvidia-smi --query-compute-apps=used_memory --format=csv
```

Change one image, model revision, context limit, or concurrency control at a
time and run a gold-set canary before promotion.

## Submit And Export

From an environment that can reach PostgreSQL and the configured source:

```bash
export KG_DISTRIBUTED_DATABASE_URL='postgresql://...'

uv run flakegraph distributed init --config configs/your-config.yaml
uv run flakegraph distributed submit --config configs/your-config.yaml
uv run flakegraph distributed status --run-id <run-id> --config configs/your-config.yaml
```

Every worker emits one `worker_ready` JSON event containing its eligible stages,
provider/model identities, and semantic `config_digest`; endpoints and credentials
are omitted. `distributed status` compares the caller's digest with the run and
reports machine-readable diagnostics. `CONFIG_DIGEST_MISMATCH` means those
settings cannot claim the run. `QUEUED_WORK_NOT_ADVANCING` means queued work has
not been claimed for at least 60 seconds; inspect pod readiness and confirm that
an eligible worker advertises the run digest.

After completion:

```bash
uv run flakegraph distributed export \
  --run-id <run-id> \
  --output out/fleet-run \
  --config configs/your-config.yaml

uv run flakegraph inspect html --output out/fleet-run --open
```

The submitter stores source bytes in the artifact store, so workers do not need
the submitter's filesystem mount.

## Scaling

Set capacity ceilings once; KEDA handles ordinary scale-up, scale-down, and the
handoff to Spark. The shared extraction pool claims one document-context task
per document, then an entity-window wave, an inventory barrier, and an
independent relation-window wave. Both window waves are dynamically claimed;
bounded task packs may process several logical windows concurrently. A worker
owns one lease at a time, while `graph.extraction_parallelism` bounds provider
calls inside that lease.

For local vLLM, provider capacity is approximately model replicas multiplied by
`modelServing.server.maxNumSeqs`. Size the extraction ceiling as provider capacity
divided by `graph.extraction_parallelism`, rounding down to avoid a permanent
overload. Increase sequence capacity, batched-token capacity, and context length
separately while observing queue depth, generation throughput, GPU utilization,
and memory. Context and compaction tasks use less model concurrency, so KEDA may
temporarily schedule below the ceiling without leaving sustained extraction work
idle.

Spark finalization scales with executor instances and cores. The finalization
coordinator remains one leased task, but graph rows stay partitioned across
executors and object storage. Provider-backed phases use several small work units
per executor slot and bounded in-partition concurrency, allowing Spark to keep
assigning work when individual model requests have variable latency. The default
three-minute LLM request timeout prevents one stalled call from indefinitely
serializing a stage; raise `llm.timeout_seconds` only for a measured provider that
needs a longer decode window. Small corpora may still be dominated by model
startup, the final extraction straggler, Spark startup, and atomic publication.
Executors strongly prefer distinct topology domains but may co-locate if a node
is unavailable, preventing a temporary capacity reduction from leaving the
entire finalization job unschedulable.

For a homogeneous 20-node GPU fleet with one four-sequence vLLM replica per
node, the site-specific values are limited to capacity declarations:

```yaml
modelServing:
  replicas: 20

workers:
  prepare:
    autoscaling:
      # Four four-core OCR workers fit beside vLLM on each 20-core GB10 node.
      maxReplicas: 80
  extract:
    autoscaling:
      # 20 replicas * 4 sequences / 2 calls per extraction task.
      maxReplicas: 40

spark:
  executorInstances: 20
```

The bundled CloudNativePG profile reserves 300 database sessions for these
worker ceilings plus KEDA, Spark, and operator traffic. When using an external
PostgreSQL service, provide an equivalent direct-connection budget or place a
supported transaction-mode pooler in front of it; do not scale worker pods beyond
the database service's connection capacity.

Use node labels and resource requests to describe heterogeneous fleets. Do not
encode hostnames or split document lists per machine; all workers claim from the
same queue, and one slow document cannot strand an assigned partition.

Before the first production corpus, validate that all expected nodes are Ready,
all GPUs are allocatable, model replicas are spread, ScaledObjects are healthy,
and no Spark executor is permanently Pending:

```bash
kubectl get nodes
kubectl get nodes -o json | jq -r \
  '.items[] | [.metadata.name, .status.allocatable["nvidia.com/gpu"]] | @tsv'
kubectl -n flakegraph get statefulset,pod -l app.kubernetes.io/component=model-serving -o wide
kubectl -n flakegraph get scaledobject,hpa
kubectl -n flakegraph describe scaledobject
```

Run a representative canary through OCR, extraction, finalization, export, and
gold-set evaluation after every Kubernetes, driver, model, provider, image, or
chart upgrade. Promote the exact image digests and values only after the canary
and the recovery drill below succeed.

## Recovery

- `distributed status` shows bounded stage counts, configuration compatibility,
  and stalled-queue diagnostics. Add `--include-tasks` for attempts, leases,
  errors, and outputs of individual tasks.
- `kubectl get scaledobject,hpa -n flakegraph` shows worker demand and capacity.
- `distributed cancel` stops unfinished work without deleting successful artifacts.
- `distributed retry --run-id <run-id>` requeues terminally failed tasks after
  the underlying provider, configuration, or storage issue is corrected.
- Worker and executor loss is recovered through task leases or Spark partition
  recomputation; incomplete finalization never publishes a graph version.
- Set pod termination grace longer than the task lease when graceful completion
  is preferred over lease recovery.

Validate recovery by deleting workers during extraction, restarting the
database primary, stopping a provider replica, and deleting a Spark executor
during finalization. Use non-sensitive canary data and confirm that retries do
not create duplicate successful outputs.

For unattended operation, alert on terminal run failures, tasks nearing their
attempt limit, expired leases, KEDA scaler errors, unschedulable pods, database
replica health, object-store errors, model readiness, GPU memory pressure, and
Spark executor loss. Worker loss is expected and recoverable; a terminal task
failure is an operator-visible outcome, never an indefinitely hung run.

The application-layer contracts behind the deployment are described in
[Architecture](architecture.md).
