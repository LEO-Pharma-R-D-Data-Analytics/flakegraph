# Snowflake App Runtime descriptors (experimental)

These two files describe the FlakeGraph console as a Snowflake Native App
running on the Node.js App Runtime: `app.yml` is the `snow` CLI project
definition, `manifest.yml` the application manifest. They are kept here, not
under `react/`, because this deployment path is not supported yet:

- The console's Snowflake runtime (`react/src/server/runtime/snowflake.ts`) is
  a JSON-backed stub. It records runs and answers the catalog from a local
  store; it does not submit work to Snowflake or read graphs from it.
- Snowflake-hosted identity (the `Sf-Context-Current-User` headers) is read
  only when the console runs with `FLAKEGRAPH_SNOWFLAKE_HOSTED=1`, which the
  App Runtime would have to set.

If you want to try it anyway:

1. Build the console: `cd react && bun install && bun run build`. This
   produces `react/.next/standalone`, which `app.yml` uploads as `server`.
2. Copy `app.yml` and `manifest.yml` into `react/` and run `snow app deploy`
   from that directory. The artifact paths are relative to where the command
   runs; deploying from the repository root would upload the Python tree.
3. Set `FLAKEGRAPH_SNOWFLAKE_HOSTED=1` on the application so the console
   treats the Snowflake context headers as its sign-in gate.

Everything the Kubernetes and local runtimes offer (submitting runs, reading
graphs back, publishing to Snowflake tables) is documented in
`react/README.md` and `docs/`.
