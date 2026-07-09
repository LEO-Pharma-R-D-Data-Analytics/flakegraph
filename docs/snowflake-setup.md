# Snowflake Setup

This guide is intentionally account-neutral. Use it as a checklist for a target
Snowflake account, but do not commit real account locators, user emails, role
names from private tenants, image digests, stage listings, or access-audit
transcripts back to the repository.

FlakeGraph can run locally without Snowflake. Snowflake mode uses the same
pipeline with Snowflake stages, Cortex functions, Snowflake tables, and
Snowpark Container Services.

## Required Objects

A production Snowflake deployment normally needs these objects:

- A runtime role, for example `KG_PROCESSOR_ROLE`
- A warehouse for metadata queries and table writes
- A database and schema for `KG_*` tables
- A document stage for source files
- A bulk-load stage for staged Parquet/JSON files
- A service-spec stage for generated Snowpark Container Services YAML
- An image repository for the FlakeGraph Docker image
- A dedicated Snowpark Container Services compute pool

Avoid relying on account-provided system compute pools for job services. Create
or grant a dedicated service-compatible pool such as `KG_PROCESSOR_CPU_POOL`.

## Minimum Grants

The runtime role should have:

- `USAGE` on the warehouse, database, schema, stages, image repository, and
  compute pool
- `CREATE TABLE`, `CREATE STAGE`, `CREATE IMAGE REPOSITORY`, and service
  creation privileges where the deployment is expected to manage objects
- `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and `MERGE` capabilities on the graph
  tables
- Cortex privileges, including `SNOWFLAKE.CORTEX_USER` and any account-level AI
  function grants required by the target account
- Either `CREATE COMPUTE POOL ON ACCOUNT` or `USAGE` on a pre-created dedicated
  compute pool

## Configuration

Start from [configs/snowflake-cortex.yaml](../configs/snowflake-cortex.yaml) and
provide account-specific values through environment variables:

```bash
export KG_SNOWFLAKE_ACCOUNT="your-org-your-account"
export KG_SNOWFLAKE_USER="your-user"
export KG_SNOWFLAKE_AUTHENTICATOR="externalbrowser"
export KG_SNOWFLAKE_DATABASE="KG_DB"
export KG_SNOWFLAKE_SCHEMA="GRAPH"
export KG_SNOWFLAKE_ROLE="KG_PROCESSOR_ROLE"
export KG_SNOWFLAKE_WAREHOUSE="KG_PROCESSOR_WH"
export KG_SNOWFLAKE_STAGE="@KG_DB.GRAPH.KG_DOCS"
export KG_SNOWFLAKE_BULK_STAGE="@KG_DB.GRAPH.KG_LOAD_STAGE"
export KG_SNOWFLAKE_IMAGE_REPOSITORY="KG_DB.GRAPH.KG_IMAGES"
export KG_SNOWFLAKE_IMAGE_NAME="flakegraph:latest"
export KG_SNOWFLAKE_COMPUTE_POOL="KG_PROCESSOR_CPU_POOL"
export KG_SNOWFLAKE_SERVICE_NAME="KG_PROCESSOR_JOB"
export KG_SNOWFLAKE_SERVICE_SPEC_STAGE="@KG_DB.GRAPH.KG_SERVICE_SPECS"
export KG_STAGE_PREFIX="incoming"
export KG_LLM_MODEL="llama3.3-70b"
export KG_EMBED_MODEL="snowflake-arctic-embed-l-v2.0"
export KG_EMBED_DIM="1024"
```

For unattended Snowpark Container Services runs, prefer OAuth file auth inside
the container or key-pair auth for a dedicated service user. Keep private keys,
OAuth tokens, passwords, and generated session tokens outside Git.

## Generated SQL And Specs

Render the account setup and runtime artifacts locally before applying them:

```bash
uv run flakegraph snowflake setup-sql --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake objects-sql --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake access-check --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake service-spec --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake execute-job-sql --config configs/snowflake-cortex.yaml
```

`setup-sql` includes role/grant-oriented statements. `objects-sql` renders only
schema objects and is safer for environments where administrators manage grants
separately.

## Live Validation

Once the account objects and grants exist:

1. Push the Docker image to the configured Snowflake image repository.
2. Upload or stage at least one public test document under the configured stage
   prefix.
3. Run `uv run flakegraph snowflake access-check --config configs/snowflake-cortex.yaml`.
4. Execute the rendered job-service SQL.
5. Inspect the `KG_*` tables or export local artifacts for parity comparison.

The repository keeps live Snowflake tests opt-in. Set `KG_RUN_SNOWFLAKE_LIVE=1`
only in an environment where the required account values and permissions are
provided externally.
