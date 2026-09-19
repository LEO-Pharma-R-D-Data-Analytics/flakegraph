import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildRunConfig,
  defaultOntologySelection,
  ontologyProfile,
  ontologySelectionFromProfile,
  runOntologySelection,
} from "./config";
import type { IngestionRequest, OntologySelection } from "./protocol/schema";

function request(overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  return {
    runtime: "local",
    jobId: "job_1",
    graphId: "graph_1",
    graphName: null,
    sourceKind: "local_path",
    source: { path: "/corpus" },
    ocr: { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    llm: { provider: "vllm_local", model: "qwen", endpoint: "http://localhost:8000/v1", apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    embedding: {
      provider: "sentence_transformers",
      model: "mini",
      endpoint: null,
      apiKeyEnvironmentVariable: null,
      dimension: 384,
      options: {},
    },
    output: { kind: "local_files", workspacePath: "/graphs/graph_1", snowflake: null },
    baseConfigPath: null,
    includeGlobs: ["**/*"],
    cacheProvider: "local",
    providerParallelism: 4,
    runtimeOptions: {},
    ontology: null,
    revision: null,
    ...overrides,
  };
}

const selection: OntologySelection = {
  entityTypes: [
    { name: "PERSON", description: "A named human being." },
    { name: "TECHNIQUE", description: "" },
  ],
  relationTypes: [{ name: "TRAINED_UNDER", description: "The source studied under the target." }],
  relations: "guided",
};

describe("the run's ontology", () => {
  it("travels inline as a profile the pipeline validates, with descriptions filled in", async () => {
    const config = await buildRunConfig(request({ ontology: selection }));
    expect(config.ontology).toEqual({
      profile: {
        name: "console",
        description: "Chosen on the console for this graph.",
        mode: "hybrid",
        entity_types: [
          { name: "PERSON", description: "A named human being." },
          { name: "TECHNIQUE", description: "A source-grounded technique." },
        ],
        relation_types: [{ name: "TRAINED_UNDER", description: "The source studied under the target." }],
      },
    });
  });

  it("maps the relation choice onto the profile's mode", () => {
    expect(ontologyProfile({ ...selection, relations: "fixed" }).mode).toBe("closed");
    const open = ontologyProfile({ ...selection, relations: "open" });
    expect(open.mode).toBe("open");
    expect(open.relation_types).toEqual([]);
  });

  it("reads a profile back into the same terms, and is read back from a run's config", async () => {
    const roundTrip = ontologySelectionFromProfile(ontologyProfile(selection));
    expect(roundTrip.entityTypes.map((term) => term.name)).toEqual(["PERSON", "TECHNIQUE"]);
    expect(roundTrip.relations).toBe("guided");
    expect(ontologySelectionFromProfile(ontologyProfile({ ...selection, relations: "fixed" })).relations).toBe("fixed");
    expect(ontologySelectionFromProfile(ontologyProfile({ ...selection, relations: "open" })).relations).toBe("open");

    const directory = await mkdtemp(path.join(tmpdir(), "fg-config-"));
    const configPath = path.join(directory, "config.yaml");
    await writeFile(
      configPath,
      "ontology:\n  profile:\n    name: console\n    mode: closed\n    entity_types:\n      - name: WIDGET\n        description: A widget.\n    relation_types:\n      - name: PART_OF\n        description: Part.\n",
      "utf8",
    );
    expect(await runOntologySelection(configPath)).toEqual({
      entityTypes: [{ name: "WIDGET", description: "A widget." }],
      relationTypes: [{ name: "PART_OF", description: "Part." }],
      relations: "fixed",
    });
    expect(await runOntologySelection(path.join(directory, "missing.yaml"))).toBeNull();
    await writeFile(path.join(directory, "bare.yaml"), "ontology:\n  profile_path: general.yaml\n", "utf8");
    expect(await runOntologySelection(path.join(directory, "bare.yaml"))).toBeNull();
  });

  it("offers the base configuration's profile file as the default", async () => {
    const defaults = await defaultOntologySelection(path.resolve(import.meta.dirname, "../../../configs/app-defaults.yaml"));
    expect(defaults.entityTypes.map((term) => term.name)).toEqual([
      "PERSON",
      "ORGANIZATION",
      "LOCATION",
      "EVENT",
      "CONCEPT",
      "WORK",
      "DATE",
    ]);
    expect(defaults.relations).toBe("guided");
    expect(defaults.relationTypes.map((term) => term.name)).toContain("LOCATED_IN");
  });
});
