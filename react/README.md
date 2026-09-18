# FlakeGraph control plane

TypeScript operator application for FlakeGraph. The processing engine stays in
the existing flakegraph worker image; this package is the Next.js control plane
that replaces Streamlit once it reaches parity.

```text
Browser
  → Next.js (App Router, tRPC, Effect)
      → Runtime module (local | kubernetes | snowflake)
          → flakegraph worker | K8s queue + workers | EXECUTE JOB SERVICE
```

## Run locally

From this directory:

```bash
pnpm install   # or bun install
pnpm dev       # http://localhost:3000
```

`flakegraph` must be on `PATH`, or set `FLAKEGRAPH_CLI` to an executable
invocation such as `uv run flakegraph`. App state is written under
`.flakegraph/app/` at the repository root unless `FLAKEGRAPH_APP_STATE_ROOT`
points elsewhere.

Seed two public datasets (martial arts and deep-learning papers) plus fleet and
Snowflake fixtures:

```bash
pnpm seed
```

## Tests

```bash
bun run test        # Vitest: schemas, catalog, progress, runtimes
bun run test:e2e    # Playwright operator flows against a local fake worker
```

End-to-end tests start the Next.js app with `FLAKEGRAPH_STUB_RUNTIMES=1` and a
fake `flakegraph` CLI so they do not need a GPU or live Snowflake account. The
Ask journeys that need a language model run against whatever the checkout
provides (hosted credentials the e2e server finds, or a local Ollama that
answers a probe) and skip otherwise; `FLAKEGRAPH_ASK_NO_MODEL=1` runs the suite
the way CI does, with no model, where the Ask tab answers from the graph's own
text.

## Environment

| Variable | Role |
| --- | --- |
| `FLAKEGRAPH_APP_STATE_ROOT` | Catalog, uploads, generated YAML |
| `FLAKEGRAPH_APP_DEFAULT_RUNTIME` | `local` or `kubernetes` |
| `FLAKEGRAPH_CLI` | Worker command, default `uv run flakegraph` |
| `FLAKEGRAPH_REPOSITORY_ROOT` | Checkout that contains `configs/app-defaults.yaml` |
| `FLAKEGRAPH_STUB_RUNTIMES` | File-backed Kubernetes and Snowflake modules |
| `FLAKEGRAPH_TRUST_IDENTITY_HEADERS` | Trust oauth2-proxy identity headers |
| `DATABASE_URL` | Optional Postgres for `flakegraph_run` listing |
| `FLAKEGRAPH_APP_KUBERNETES_NAMESPACE` | Default fleet namespace |

Pages never import a runtime module. tRPC procedures call the `ControlPlane`
interface; Effect Schema is the source of truth for `IngestionRequest`,
`RunSnapshot`, `GraphDataset`, `Viewer`, and capabilities.

Snowflake App Runtime deploy stays in this directory (`app.yml` and `manifest.yml`).
Do not move those files to the repository root.
