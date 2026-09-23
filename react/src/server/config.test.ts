import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildRunConfig, defaultOntologySelection, runOntologySelection } from "./config";
import { ontologyProfile, ontologySelectionFromProfile } from "./ontology-profile";
import type { IngestionRequest, OntologySelection } from "./protocol/schema";
import { appEnv } from "./env";
import { availableSamplePacks } from "./sample-packs";

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
  it("travels inline as a profile the pipeline validates, built on the default profile", async () => {
    // What every confined request carries when the form names no base.
    const baseConfigPath = path.resolve(import.meta.dirname, "../../../configs/app-defaults.yaml");
    const config = await buildRunConfig(request({ ontology: selection, baseConfigPath }));
    expect(config.ontology).toEqual({
      profile: {
        name: "general",
        description: "General-purpose profile for evidence-grounded document graphs.",
        mode: "hybrid",
        entity_types: [
          { name: "PERSON", description: "A named human being." },
          { name: "TECHNIQUE", description: "A source-grounded technique." },
        ],
        relation_types: [{ name: "TRAINED_UNDER", description: "The source studied under the target." }],
      },
    });
  });

  it("keeps what a sample pack's terms carry beyond their names, minus what an edit removed", async () => {
    const pack = await availableSamplePacks(appEnv().repositoryRoot);
    const martial = pack.find((item) => item.path === "data/martial_arts/files");
    expect(martial?.ontology?.entityTypes.map((term) => term.name)).toContain("SCHOOL");
    const withoutSchools = {
      ...martial!.ontology!,
      entityTypes: martial!.ontology!.entityTypes.filter((term) => term.name !== "SCHOOL"),
    };
    const config = await buildRunConfig(
      // As a confined request carries it: resolved against the repository.
      request({ source: { path: path.join(appEnv().repositoryRoot, "data/martial_arts/files") }, ontology: withoutSchools }),
    );
    const profile = (config.ontology as { profile: Record<string, unknown> }).profile;
    expect(profile.name).toBe("martial-arts-history");
    const founded = (profile.relation_types as Array<Record<string, unknown>>).find((item) => item.name === "FOUNDED_BY");
    expect(founded?.evidence_cues).toEqual(["\\b(?:founded|established|created|formed)\\b"]);
    expect(founded?.aliases).toEqual(["established_by", "created_by"]);
    // SCHOOL is gone, so the pipeline must not be told FOUNDED_BY starts at one.
    expect(founded?.source_types).toEqual(["ORGANIZATION"]);
    const martialArt = (profile.entity_types as Array<Record<string, unknown>>).find((item) => item.name === "MARTIAL_ART");
    expect(martialArt?.aliases).toEqual(["style", "system", "combat_sport"]);
  });

  it("drops what refers to a removed term, and aliases that would collide", () => {
    const base = {
      name: "base",
      description: "A base profile.",
      entity_types: [
        { name: "PERSON", description: "" },
        { name: "PLACE", description: "" },
      ],
      relation_types: [
        { name: "PARENT_OF", description: "", inverse: "CHILD_OF", source_types: ["PLACE"], aliases: ["father_of"] },
        { name: "CHILD_OF", description: "", inverse: "PARENT_OF" },
      ],
    };
    const profile = ontologyProfile(
      {
        entityTypes: [{ name: "PERSON", description: "" }],
        relationTypes: [
          { name: "PARENT_OF", description: "" },
          { name: "Father-Of", description: "Stated as father." },
        ],
        relations: "guided",
      },
      base,
    );
    const [parent, father] = profile.relation_types as Array<Record<string, unknown>>;
    expect(parent).toEqual({ name: "PARENT_OF", description: "A source-grounded parent of.", aliases: [] });
    expect(father).toEqual({ name: "Father-Of", description: "Stated as father." });
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

describe("the owner a run records", () => {
  it("names the submitter as the graph's owner, and nobody when there is none", async () => {
    const owned = await buildRunConfig(request(), null, "ALICE@EXAMPLE.COM");
    expect((owned.job as Record<string, unknown>).owner).toBe("ALICE@EXAMPLE.COM");
    const unowned = await buildRunConfig(request());
    expect((unowned.job as Record<string, unknown>).owner).toBeUndefined();
  });
});
