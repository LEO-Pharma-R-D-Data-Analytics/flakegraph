"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Upload } from "lucide-react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { GuideCard } from "@/components/console/guide-card";
import { OntologyEditor, ontologyProblem } from "@/components/console/ontology-editor";
import { PageHeader } from "@/components/console/page-header";
import { SnowflakeGrantsCard } from "@/components/console/operator-tools";
import {
  DEFAULT_PROVIDER_PARALLELISM,
  type Capability,
  type IngestionRequest,
  type OntologySelection,
  type ProviderSelection,
  type RuntimeMode,
  type SourceKind,
  type StorageKind,
} from "@/server/protocol/schema";
import {
  EMBEDDING_PROVIDERS,
  LLM_PROVIDERS,
  OCR_PROVIDERS,
  defaultProvider,
  providerSelection,
  selectionFromOption,
} from "@/server/providers";
import { useStorageItem, writeStorage } from "@/lib/browser-storage";
import { useLastIngestion, writeLastIngestion, type LastIngestionDraft } from "@/lib/last-ingestion";
import { useDebouncedValue } from "@/lib/use-debounced-value";
import { cn, formatBytes } from "@/lib/utils";

/**
 * The form builds a new version of an existing graph instead of a new graph:
 * the graph's identity, name and vocabulary are the base run's, the documents
 * it keeps are decided elsewhere, and this form only adds to them.
 */
export interface RevisionTarget {
  baseRunId: string;
  graphId: string;
  graphName: string | null;
  dropFileIds: string[];
  keptCount: number;
  /** What the base version extracted, when the run recorded it. */
  ontology: OntologySelection | null;
}

interface IngestionFormProps {
  runtime: RuntimeMode;
  capabilities: Set<Capability>;
  lastSuccessRuntime?: string | null;
  suggestionMode?: "off" | "on-request" | "auto-fill";
  revision?: RevisionTarget | null;
  onSubmitted?: (runId: string) => Promise<void> | void;
}

/** A hosted corpus the session says is on this host. */
type SamplePack = { name: string; path: string; graphName: string; why: string };

/** The document source on the form: a source kind, or a sample pack standing in for a local path. */
type SourceChoice = SourceKind | "sample";

const EMPTY_ONTOLOGY: OntologySelection = { entityTypes: [], relationTypes: [], relations: "guided" };

/** The credential slot a fleet's finalizer reads its Snowflake password from. */
const FLEET_SNOWFLAKE_PASSWORD_SLOT = "KG_SNOWFLAKE_PASSWORD";

export function IngestionForm({
  runtime,
  capabilities,
  lastSuccessRuntime,
  suggestionMode = "on-request",
  revision = null,
  onSubmitted,
}: IngestionFormProps) {
  const session = trpc.auth.session.useQuery();
  const workspace = trpc.workspace.get.useQuery();
  // Sample packs are offered only when their files are on the host the
  // runtime reads from; the fleet image carries the small one, a laptop
  // checkout both, a bare install none.
  const samplePacks: readonly SamplePack[] = session.data?.samplePacks ?? [];
  const isSamplePath = (path: string) => samplePacks.some((sample) => sample.path === path);
  const clearPromotion = trpc.workspace.clearPromotion.useMutation();
  const [jobId] = useState(() => cryptoRandom("run"));
  const [graphId, setGraphId] = useState(() => revision?.graphId ?? cryptoRandom("graph"));
  const [graphName, setGraphName] = useState(revision?.graphName ?? "");
  const [choice, setChoice] = useState<SourceChoice>(defaultSourceKind(capabilities));
  const [sourcePath, setSourcePath] = useState("");
  const [uploadPath, setUploadPath] = useState<string | null>(null);
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [azure, setAzure] = useState({ accountUrl: "", container: "", prefix: "" });
  const [s3, setS3] = useState({ bucket: "", prefix: "", endpointUrl: "", region: "" });
  const [stage, setStage] = useState({ stage: "", prefix: "" });
  const [outputKind, setOutputKind] = useState<StorageKind>(capabilities.has("spcs") ? "snowflake" : "local_files");
  const [ocr, setOcr] = useState<ProviderSelection>(() => defaultProvider("ocr"));
  const [llm, setLlm] = useState(() => defaultProvider("llm"));
  const [embedding, setEmbedding] = useState(() => defaultProvider("embedding"));
  const [parallelism, setParallelism] = useState(DEFAULT_PROVIDER_PARALLELISM);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [preview, setPreview] = useState<string>("");
  const [promotionApplied, setPromotionApplied] = useState(false);
  const [snowflake, setSnowflake] = useState({
    account: "",
    user: "",
    database: "",
    schema: "",
    warehouse: "",
    bulkStage: "",
    role: "",
    host: "",
    authenticator: "",
    credentialEnvironmentVariable: "SNOWFLAKE_PASSWORD",
    credentialField: "password",
  });
  const sourceKind: SourceKind = choice === "sample" ? "local_path" : choice;

  // On the fleet the workers' own profile decides the providers: a worker
  // claims only a run whose semantic configuration hashes to its own. The
  // card shows what the fleet runs; the choice is the deployment's.
  const fleet = trpc.fleet.profile.useQuery(undefined, { enabled: runtime === "kubernetes" });
  const fleetProfile = runtime === "kubernetes" ? fleet.data ?? null : null;
  const effectiveOcr = fleetProfile ? fleetSelection(fleetProfile, "ocr", ocr) : ocr;
  const effectiveLlm = fleetProfile ? fleetSelection(fleetProfile, "llm", llm) : llm;
  const effectiveEmbedding = fleetProfile ? fleetSelection(fleetProfile, "embedding", embedding) : embedding;
  const fleetSnowflake = fleetProfile ? fleetSnowflakeAccount(fleetProfile) : null;

  // What the run extracts: the runtime's default vocabulary until it is
  // changed here; a revision keeps the base version's.
  const baseConfigPath = `${session.data?.repositoryRoot ?? ""}/configs/app-defaults.yaml`;
  const defaultOntology = trpc.ingestion.defaultOntology.useQuery(
    { baseConfigPath },
    { enabled: Boolean(session.data?.repositoryRoot) && !revision },
  );
  const ontologyDefaults = defaultOntology.data
    ? {
        selection: {
          entityTypes: defaultOntology.data.entityTypes,
          relationTypes: defaultOntology.data.relationTypes,
          relations: defaultOntology.data.relations,
        },
        source: defaultOntology.data.source,
      }
    : null;
  const [edited, setEdited] = useState<OntologySelection | null>(null);
  // Until someone edits the vocabulary (or clones one), it follows the
  // defaults, including when they arrive or the runtime changes. Derived
  // while rendering rather than from an effect, so the form never paints
  // an empty vocabulary first.
  const [followingDefaults, setFollowingDefaults] = useState(true);
  const [defaultsSeen, setDefaultsSeen] = useState<string | null>(null);
  const defaultsKey = ontologyDefaults ? `${runtime}:${JSON.stringify(ontologyDefaults.selection)}` : null;
  if (defaultsKey && defaultsSeen !== defaultsKey && ontologyDefaults) {
    setDefaultsSeen(defaultsKey);
    if (followingDefaults) {
      setEdited(ontologyDefaults.selection);
    }
  }
  function setOntology(next: OntologySelection) {
    setEdited(next);
    setFollowingDefaults(false);
  }
  // A revision's vocabulary is its base version's, read from that run; it
  // is shown, never edited.
  const ontology = revision ? revision.ontology : edited;
  const effectiveOntology = ontology ?? EMPTY_ONTOLOGY;

  const preflight = trpc.ingestion.preflight.useMutation();
  const previewConfig = trpc.ingestion.config.preview.useMutation();
  const submit = trpc.runs.submit.useMutation();
  const scanPii = trpc.ingestion.scanPii.useMutation();
  const ackPii = trpc.ingestion.acknowledgePii.useMutation({
    onSuccess: () => {
      if (request) {
        scanPii.mutate({ source: request.source, sourceKey });
      }
    },
  });
  // What the run would read, named before it starts: local paths list at
  // once; a bucket or container lists once its fields stop changing, through
  // the pipeline's own listing. A Snowflake stage is only known to the
  // Snowflake runtime.
  const browseSource = useDebouncedValue(browsableSource(), 600);
  // No placeholder from the previous source: a count belongs to the source
  // it was read from, and Start must not open on another bucket's listing.
  const sources = trpc.ingestion.sources.list.useQuery(
    { source: browseSource ?? { kind: "local", path: "" } },
    { enabled: browseSource !== null },
  );

  const lastDraft = useLastIngestion(runtime);
  // A draft is copied into the fields once per runtime: when the workspace
  // asks for it, or when a run page left a clone waiting. Both are derived
  // while rendering rather than from an effect, so the form never paints its
  // empty state first.
  const cloneWaiting = useStorageItem("session", "flakegraph.pending-clone") === "1";
  const [clonedFor, setClonedFor] = useState<string | null>(null);
  if (lastDraft && (suggestionMode === "auto-fill" || cloneWaiting) && clonedFor !== runtime) {
    setClonedFor(runtime);
    applyClone(lastDraft);
  }
  useEffect(() => {
    if (cloneWaiting && (clonedFor === runtime || !lastDraft)) {
      writeStorage("session", "flakegraph.pending-clone", null);
    }
  }, [cloneWaiting, clonedFor, lastDraft, runtime]);

  const envConfirmed = useStorageItem("session", "flakegraph.runtime-ack") === runtime;

  function confirmEnvironment() {
    writeStorage("session", "flakegraph.runtime-ack", runtime);
  }

  const pendingPromotion = workspace.data?.pendingPromotion;
  if (pendingPromotion && pendingPromotion.toRuntime === runtime && !promotionApplied) {
    setPromotionApplied(true);
    setGraphName(pendingPromotion.graphName);
    setGraphId(pendingPromotion.keepGraphId ? pendingPromotion.graphId : cryptoRandom("graph"));
    if (pendingPromotion.toRuntime === "snowflake") {
      setChoice("snowflake_stage");
      setStage({ stage: `app/<user>/${pendingPromotion.graphId}`, prefix: pendingPromotion.graphId });
    } else if (pendingPromotion.toRuntime === "kubernetes") {
      setChoice("s3");
      setS3({ bucket: "flakegraph-corpus", prefix: pendingPromotion.graphId, endpointUrl: "", region: "" });
    }
  }
  useEffect(() => {
    if (promotionApplied) {
      toast.success("Promotion remaps applied. Submit stays a human click.");
    }
  }, [promotionApplied]);

  const request = useMemo((): IngestionRequest | null => {
    // Graph artifacts live under the control plane's own state, the one place
    // it can write on a read-only image; locally that is the checkout's
    // .flakegraph/app, as before.
    const workspacePath = `${session.data?.stateRoot ?? ""}/graphs/${graphId}`;
    const source = sourcePayload();
    // A revision that removes documents needs no source: it keeps what it
    // keeps and adds nothing. One that removes nothing must add something.
    const dropsOnly = Boolean(revision && revision.dropFileIds.length > 0 && !source);
    if (!source && !dropsOnly) {
      return null;
    }
    // On a fleet the account and credential are the fleet's; the form names
    // only where inside that account the graph lands.
    const snowflakeTarget =
      outputKind === "snowflake"
        ? fleetSnowflake
          ? {
              ...snowflake,
              account: fleetSnowflake.account,
              user: fleetSnowflake.user,
              host: fleetSnowflake.host,
              authenticator: fleetSnowflake.authenticator,
              role: snowflake.role || null,
              credentialEnvironmentVariable: FLEET_SNOWFLAKE_PASSWORD_SLOT,
              credentialField: "password",
            }
          : {
              ...snowflake,
              role: snowflake.role || null,
              host: snowflake.host || null,
              authenticator: snowflake.authenticator || null,
              credentialEnvironmentVariable: snowflake.credentialEnvironmentVariable || null,
              credentialField: snowflake.credentialField || null,
            }
        : null;
    return {
      runtime,
      jobId,
      graphId,
      graphName: graphName.trim() || null,
      sourceKind,
      source: source ?? {},
      revision: revision
        ? { baseRunId: revision.baseRunId, dropFileIds: revision.dropFileIds, addDocuments: Boolean(source) }
        : null,
      ocr: effectiveOcr,
      llm: effectiveLlm,
      embedding: effectiveEmbedding,
      output: { kind: outputKind, workspacePath, snowflake: snowflakeTarget },
      baseConfigPath,
      includeGlobs: ["**/*"],
      cacheProvider: capabilities.has("spcs") ? "snowflake" : "local",
      providerParallelism: parallelism,
      runtimeOptions: {},
      ontology: ontology && ontology.entityTypes.length ? ontology : null,
    };

    function sourcePayload(): Record<string, unknown> | null {
      if (choice === "upload") {
        if (!uploadPath) {
          return null;
        }
        return { kind: "local", path: uploadPath };
      }
      if (choice === "local_path" || choice === "sample") {
        if (!sourcePath.trim()) {
          return null;
        }
        return { kind: "local", path: sourcePath.trim() };
      }
      if (choice === "azure_blob") {
        if (!azure.accountUrl.trim() || !azure.container.trim()) {
          return null;
        }
        return { kind: "azure_blob", ...azure };
      }
      if (choice === "s3") {
        if (!s3.bucket.trim()) {
          return null;
        }
        return { kind: "s3", ...s3 };
      }
      if (!stage.stage.trim()) {
        return null;
      }
      return { kind: "snowflake_stage", ...stage };
    }
  }, [
    runtime,
    graphName,
    revision,
    choice,
    sourceKind,
    sourcePath,
    uploadPath,
    azure,
    s3,
    stage,
    outputKind,
    effectiveOcr,
    effectiveLlm,
    effectiveEmbedding,
    parallelism,
    ontology,
    snowflake,
    fleetSnowflake,
    baseConfigPath,
    session.data?.stateRoot,
    jobId,
    graphId,
    capabilities,
  ]);

  // A runtime with different capabilities changes what the form may hold:
  // the output moves to the runtime's own storage, and a local folder or the
  // sample pack is no longer a source.
  const [capabilitiesSeen, setCapabilitiesSeen] = useState(capabilities);
  if (capabilitiesSeen !== capabilities) {
    setCapabilitiesSeen(capabilities);
    setOutputKind(capabilities.has("spcs") ? "snowflake" : "local_files");
  }
  if (choice === "sample" && (!capabilities.has("local") || (session.data && samplePacks.length === 0))) {
    setChoice(defaultSourceKind(capabilities));
    setSourcePath("");
  }
  if (!capabilities.has("local") && choice === "local_path") {
    setChoice(defaultSourceKind(capabilities));
    setSourcePath("");
  }
  // A Snowflake destination needs an account to write into: the fleet's on
  // Kubernetes, the form's elsewhere. When the fleet turns out to hold none,
  // the destination falls back rather than submitting an unwritable run.
  const snowflakeAvailable =
    runtime === "snowflake" || runtime === "local" || (runtime === "kubernetes" && fleetSnowflake !== null);
  if (outputKind === "snowflake" && fleet.isFetched && !snowflakeAvailable) {
    setOutputKind("local_files");
  }

  function resolvedPath() {
    return choice === "upload" ? uploadPath ?? "" : sourcePath;
  }

  function browsableSource(): Record<string, unknown> | null {
    if (choice === "upload" || choice === "local_path" || choice === "sample") {
      return resolvedPath() ? { kind: "local", path: resolvedPath() } : null;
    }
    if (choice === "s3") {
      return s3.bucket.trim() ? { kind: "s3", ...s3 } : null;
    }
    if (choice === "azure_blob") {
      return azure.accountUrl.trim() && azure.container.trim() ? { kind: "azure_blob", ...azure } : null;
    }
    return runtime === "snowflake" && stage.stage.trim() ? { kind: "snowflake_stage", ...stage } : null;
  }

  function chooseSource(next: SourceChoice) {
    if (next === choice) {
      return;
    }
    if (choice === "sample" || next === "sample") {
      setSourcePath("");
    }
    if (next !== "upload") {
      setUploadPath(null);
      setUploadJobId(null);
    }
    setChoice(next);
    if (next === "sample") {
      const selected = samplePacks.find((sample) => sample.path === sourcePath) ?? samplePacks[0];
      if (selected) {
        applySample(selected);
      }
    }
  }

  function applySample(sample: SamplePack) {
    setChoice("sample");
    setSourcePath(sample.path);
    setGraphName((current) => {
      const fromPack = samplePacks.some((item) => item.graphName === current);
      return !current.trim() || fromPack ? sample.graphName : current;
    });
  }

  function applyClone(draft: LastIngestionDraft) {
    setGraphName(draft.graphName);
    const kind = (draft.sourceKind as SourceKind) || "local_path";
    const source = draft.source ?? {};
    const field = (name: string, fallback = "") => (typeof source[name] === "string" ? source[name] : fallback);
    setSourcePath(draft.sourcePath);
    if (kind === "upload") {
      setUploadPath(draft.sourcePath);
      setChoice("upload");
    } else if (kind === "local_path" && isSamplePath(draft.sourcePath) && capabilities.has("local")) {
      setChoice("sample");
    } else {
      setChoice(kind);
    }
    if (kind === "s3") {
      setS3({
        bucket: field("bucket"),
        prefix: field("prefix", draft.sourcePath),
        endpointUrl: field("endpointUrl", field("endpoint_url")),
        region: field("region"),
      });
    } else if (kind === "azure_blob") {
      setAzure({
        accountUrl: field("accountUrl", field("account_url")),
        container: field("container"),
        prefix: field("prefix", draft.sourcePath),
      });
    } else if (kind === "snowflake_stage") {
      setStage({ stage: field("stage", draft.sourcePath), prefix: field("prefix") });
    }
    setOcr(providerSelection("ocr", draft.ocrProvider));
    setLlm(providerSelection("llm", draft.llmProvider));
    setEmbedding(providerSelection("embedding", draft.embeddingProvider));
    if (draft.ontology) {
      setOntology(draft.ontology);
    }
  }

  async function onUpload(files: FileList | null) {
    if (!files?.length) {
      return;
    }
    setChoice("upload");
    setUploading(true);
    const body = new FormData();
    for (const file of files) {
      body.append("files", file);
    }
    if (uploadJobId) {
      body.append("jobId", uploadJobId);
    }
    try {
      const response = await fetch("/api/uploads", { method: "POST", body });
      const payload = (await response.json()) as { path?: string; error?: string; jobId?: string; count?: number };
      if (!response.ok || !payload.path) {
        toast.error(payload.error || "Upload failed");
        return;
      }
      setUploadPath(payload.path);
      if (payload.jobId) {
        setUploadJobId(payload.jobId);
      }
      // The server's count, not the input's: the input is cleared while the
      // upload is in flight, which empties the FileList it handed us.
      const count = payload.count ?? 0;
      toast.success(
        uploadJobId
          ? `Added ${count} file${count === 1 ? "" : "s"} to the same upload folder`
          : `Uploaded ${count} file${count === 1 ? "" : "s"}`,
      );
    } finally {
      setUploading(false);
    }
  }

  async function onPreview() {
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    try {
      const yaml = await previewConfig.mutateAsync(request);
      setPreview(yaml);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Unable to preview configuration");
    }
  }

  async function onPreflight() {
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    try {
      setPreflightRanFor(preflightInputs);
      const result = await preflight.mutateAsync(request);
      if (result.ok) {
        toast.success("Preflight passed. The run is ready to submit.");
      } else {
        toast.error(result.errors.join("; ") || "Preflight failed");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Preflight failed");
    }
  }

  async function onSubmit() {
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    if (runtime === "snowflake" && session.data?.grants?.some((grant) => !grant.ok)) {
      toast.error("Snowflake account preflight is still red.");
      return;
    }
    if (lastSuccessRuntime && lastSuccessRuntime !== runtime && !envConfirmed) {
      toast.error(`Confirm switching from ${lastSuccessRuntime} to ${runtime} before Start.`);
      return;
    }
    try {
      if (request.revision?.addDocuments !== false) {
        const pii = await scanPii.mutateAsync({ source: request.source, sourceKey });
        if (pii.blocked) {
          toast.error("PII hits must be quarantined or acknowledged before embed.");
          return;
        }
        setPreflightRanFor(preflightInputs);
        const result = await preflight.mutateAsync(request);
        if (!result.ok) {
          toast.error(result.errors.join("; ") || "Fix preflight errors before submitting");
          return;
        }
      }
      const snapshot = await submit.mutateAsync(request);
      writeLastIngestion({
        runtime,
        graphName: graphName.trim(),
        sourceKind,
        sourcePath: resolvedPath(),
        source: request.source,
        ocrProvider: ocr.provider,
        llmProvider: llm.provider,
        embeddingProvider: embedding.provider,
        ...(request.ontology ? { ontology: request.ontology } : {}),
        savedAt: new Date().toISOString(),
      });
      toast.success(
        revision
          ? `Building a new version of ${snapshot.graphName || snapshot.graphId}`
          : `Submitted ${snapshot.graphName || snapshot.graphId}`,
      );
      await onSubmitted?.(snapshot.runId);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Unable to start ingestion");
    }
  }

  const sourceChoices = sourceChoiceOptions(capabilities, samplePacks.length > 0 && !revision);
  const objectCount = sources.data?.length ?? 0;
  const sourceKey = `${runtime}:${sourceKind}:${resolvedPath()}`;
  const estimate = trpc.ingestion.estimate.useQuery(
    {
      objectCount,
      runtime,
      ocrProvider: ocr.provider,
      parallelism,
    },
    { enabled: objectCount > 0, placeholderData: (previous) => previous },
  );
  const totalBytes = (sources.data ?? []).reduce((sum, item) => sum + item.sizeBytes, 0);
  const sourceNoun = choice === "s3" ? "objects" : choice === "azure_blob" ? "blobs" : "files";
  const sourcePlace = choice === "s3" || choice === "azure_blob" ? "under this prefix" : "at this path";
  const envChanged = Boolean(lastSuccessRuntime && lastSuccessRuntime !== runtime);
  const grantsBlocked = runtime === "snowflake" && Boolean(session.data?.grants?.some((grant) => !grant.ok));
  // A source the console can name ahead of the run is held to what it lists:
  // Start waits for the listing, and an empty or failed one keeps it closed
  // for the same reason the run itself would stop.
  // A listing whose retry is paused (the tab lost focus between attempts)
  // has neither answered nor failed yet, so it counts as pending too.
  const browsable = browseSource !== null;
  const listingPending = browsable && (sources.isFetching || sources.isPaused) && !sources.data;
  const emptyLocalListing =
    listingPending ||
    (choice === "upload" && !uploadPath) ||
    (browsable && Array.isArray(sources.data) && sources.data.length === 0);
  const listingFailed = browsable && Boolean(sources.error);
  const envBlocked = envChanged && !envConfirmed;
  const incompleteSource = !request;
  const dropsOnly = Boolean(request?.revision && request.revision.addDocuments === false);
  // A version with nothing in it is not a graph: removing every document
  // without adding any is refused here, not after a run.
  const emptyRevision = Boolean(revision && revision.keptCount === 0 && objectCount === 0);
  // The vocabulary must be usable before a run is built with it; a revision
  // keeps its base version's and is not held to this.
  const ontologyBlocked = !revision && ontology !== null && ontologyProblem(ontology) !== null;
  // Hosted in Snowflake, the account and its defaults come with the session;
  // elsewhere the target inside the account has to be named.
  const snowflakeIncomplete =
    outputKind === "snowflake" &&
    runtime !== "snowflake" &&
    (!snowflake.database.trim() ||
      !snowflake.schema.trim() ||
      !snowflake.bulkStage.trim() ||
      (!fleetSnowflake && (!snowflake.account.trim() || !snowflake.user.trim())));
  const startBlocked =
    grantsBlocked ||
    (!dropsOnly && (emptyLocalListing || listingFailed)) ||
    envBlocked ||
    incompleteSource ||
    emptyRevision ||
    ontologyBlocked ||
    snowflakeIncomplete;

  // A preflight verdict describes the inputs it ran against; once any of them
  // changes it is no longer shown, and the next check starts fresh.
  const preflightInputs = JSON.stringify([
    embedding.provider,
    llm.provider,
    ocr.provider,
    parallelism,
    runtime,
    sourceKind,
    sourcePath,
    uploadPath,
    outputKind,
    ontology,
  ]);
  const [preflightRanFor, setPreflightRanFor] = useState<string | null>(null);
  const preflightVerdict = preflightRanFor === preflightInputs ? preflight.data : undefined;

  const kindFields = (
    <>
      {choice === "sample" ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {samplePacks.map((sample) => (
            <ChoiceTile
              key={sample.path}
              name={sample.name}
              title={sample.name}
              description={sample.why}
              selected={sourcePath === sample.path}
              onClick={() => applySample(sample)}
            />
          ))}
        </div>
      ) : null}
      {choice === "upload" ? (
        <FileDropzone busy={uploading} hasFiles={Boolean(uploadPath)} onFiles={(files) => void onUpload(files)} />
      ) : null}
      {choice === "local_path" ? (
        <Field label="Folder path">
          <Input
            aria-label="Folder path"
            placeholder="A folder the workers can read, e.g. /data/contracts"
            value={sourcePath}
            onChange={(event) => setSourcePath(event.target.value)}
          />
        </Field>
      ) : null}
      {choice === "azure_blob" ? (
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Account URL">
            <Input value={azure.accountUrl} onChange={(event) => setAzure({ ...azure, accountUrl: event.target.value })} />
          </Field>
          <Field label="Container">
            <Input value={azure.container} onChange={(event) => setAzure({ ...azure, container: event.target.value })} />
          </Field>
          <Field label="Prefix">
            <Input value={azure.prefix} onChange={(event) => setAzure({ ...azure, prefix: event.target.value })} />
          </Field>
        </div>
      ) : null}
      {choice === "s3" ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Bucket">
            <Input value={s3.bucket} onChange={(event) => setS3({ ...s3, bucket: event.target.value })} />
          </Field>
          <Field label="Prefix">
            <Input value={s3.prefix} onChange={(event) => setS3({ ...s3, prefix: event.target.value })} />
          </Field>
          <Field label="Endpoint">
            <Input value={s3.endpointUrl} onChange={(event) => setS3({ ...s3, endpointUrl: event.target.value })} />
          </Field>
          <Field label="Region">
            <Input value={s3.region} onChange={(event) => setS3({ ...s3, region: event.target.value })} />
          </Field>
        </div>
      ) : null}
      {choice === "snowflake_stage" ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Stage">
            <Input value={stage.stage} onChange={(event) => setStage({ ...stage, stage: event.target.value })} />
          </Field>
          <Field label="Prefix">
            <Input value={stage.prefix} onChange={(event) => setStage({ ...stage, prefix: event.target.value })} />
          </Field>
        </div>
      ) : null}
    </>
  );
  const listing = (
    <>
      {browsable ? (
        listingPending ? (
          <p className="text-sm text-muted-foreground">Listing {sourceNoun}…</p>
        ) : sources.data ? (
          <p className="text-sm text-muted-foreground" data-testid="source-count">
            {sources.data.length} selectable {sources.data.length === 1 ? "object" : "objects"}
            {sources.data.length === 0 ? ` ${sourcePlace}` : totalBytes > 0 ? ` · ${formatBytes(totalBytes)}` : ""}
          </p>
        ) : null
      ) : choice === "snowflake_stage" ? (
        <p className="text-sm text-muted-foreground">Stage contents are confirmed when the job starts.</p>
      ) : choice === "upload" ? null : (
        <p className="text-sm text-muted-foreground">
          {choice === "s3"
            ? "Objects are listed once a bucket is named."
            : choice === "azure_blob"
              ? "Blobs are listed once an account URL and container are named."
              : choice === "sample"
                ? "Pick a pack to list its files."
                : "Files are listed once a folder is named."}
        </p>
      )}
      {sources.error ? <Alert variant="destructive">{sources.error.message}</Alert> : null}
    </>
  );

  return (
    <div className="flex min-h-full flex-1 flex-col gap-6">
      {revision ? null : (
        <PageHeader
          kicker="Ingestion"
          title="Build a graph"
          description={`Point FlakeGraph at documents, choose what to extract, and start. The job runs on ${runtimeLabel(runtime)} and keeps running if you close the browser.`}
        />
      )}

      {promotionApplied && workspace.data?.pendingPromotion ? (
        <GuideCard
          title="Promotion remaps are on the form"
          why={(workspace.data.pendingPromotion.remaps ?? []).map((item) => `${item.from} → ${item.to}`).join(" · ")}
          onDismiss={() => {
            setPromotionApplied(false);
            clearPromotion.mutate();
          }}
          actions={[]}
        />
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>{revision ? "Documents to add" : "Documents"}</CardTitle>
          <CardDescription>
            {revision
              ? "Only these are parsed and extracted; the documents the graph keeps are read from where the earlier run left them. A file the graph already holds is skipped."
              : "Where the documents come from. PDF, markdown, and office documents are parsed; anything else is skipped."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Document source">
            {sourceChoices.map((option) => (
              <button
                key={option.value}
                type="button"
                aria-pressed={choice === option.value}
                onClick={() => chooseSource(option.value)}
                className={cn(
                  "rounded-md border px-3 py-1.5 text-sm transition-colors",
                  choice === option.value
                    ? "border-primary bg-accent font-medium text-foreground ring-1 ring-primary"
                    : "border-border bg-background text-muted-foreground hover:bg-muted/50",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
          {kindFields}
          {listing}
        </CardContent>
      </Card>

      {revision ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Graph</CardTitle>
            <CardDescription>
              A friendly name is optional; the stable graph ID is assigned on submit. The destination is where the
              graph is written, not where the job runs.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Graph ID <span className="font-mono">{graphId}</span> · minted for this Start
            </p>
            <Field label="Display name">
              <Input
                aria-label="Display name"
                value={graphName}
                onChange={(event) => setGraphName(event.target.value)}
                placeholder="Optional display name"
              />
            </Field>
            <Field label="Destination">
              <Select value={outputKind} onValueChange={(value) => setOutputKind(value as StorageKind)}>
                <SelectTrigger aria-label="Destination">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="local_files">Local files</SelectItem>
                  <SelectItem value="snowflake" disabled={!snowflakeAvailable}>
                    Snowflake
                  </SelectItem>
                </SelectContent>
              </Select>
              {runtime === "kubernetes" && fleet.isFetched && !fleetSnowflake ? (
                <span className="text-xs text-muted-foreground" data-testid="snowflake-unavailable">
                  Snowflake is not set up on this fleet: its workers hold no Snowflake account. Set{" "}
                  <span className="font-mono">snowflake.account</span> and <span className="font-mono">user</span> in
                  the fleet&apos;s processing config and the{" "}
                  <span className="font-mono">{FLEET_SNOWFLAKE_PASSWORD_SLOT}</span> key in its providers Secret.
                </span>
              ) : null}
            </Field>
          </CardContent>
        </Card>
      )}

      {outputKind === "snowflake" ? (
        <Card data-testid="snowflake-destination">
          <CardHeader>
            <CardTitle>Snowflake destination</CardTitle>
            <CardDescription>
              {fleetSnowflake
                ? `The fleet writes as ${fleetSnowflake.user} into account ${fleetSnowflake.account}, with the credential its workers hold. Name where inside that account the graph lands.`
                : "Credentials stay as environment variable references; the value is read where the run executes."}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-3">
            {(fleetSnowflake ? FLEET_SNOWFLAKE_FIELDS : ALL_SNOWFLAKE_FIELDS).map((key) => (
              <Field key={key} label={humanizeField(key)}>
                <Input
                  aria-label={humanizeField(key)}
                  value={snowflake[key]}
                  onChange={(event) => setSnowflake({ ...snowflake, [key]: event.target.value })}
                />
              </Field>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {revision && !revision.ontology ? null : (
        <OntologyEditor
          value={effectiveOntology}
          onChange={setOntology}
          defaults={revision ? null : ontologyDefaults}
          locked={
            revision
              ? "A new version is built with the vocabulary of the version it revises, so the documents it keeps and the ones it adds are typed alike. To extract different types, build a new graph."
              : null
          }
        />
      )}

      <Card data-testid="compose-providers">
        <CardHeader>
          <CardTitle>Processing</CardTitle>
          <CardDescription>
            {fleetProfile
              ? `Set by the fleet: its workers run one parser, one model and one embedding model (profile ${fleetProfile.configMap}) and claim only runs built for them. Change the fleet's values to change these.`
              : "The parser, the language model, and the embedding model this run uses. Adaptive layout reads native document text first and sends scans, figures, and sparse pages to MinerU."}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <ProviderFields title="OCR" options={OCR_PROVIDERS} value={effectiveOcr} onChange={setOcr} locked={Boolean(fleetProfile)} />
          <ProviderFields title="LLM" options={LLM_PROVIDERS} value={effectiveLlm} onChange={setLlm} locked={Boolean(fleetProfile)} />
          <ProviderFields
            title="Embeddings"
            options={EMBEDDING_PROVIDERS}
            value={effectiveEmbedding}
            onChange={setEmbedding}
            dimension
            locked={Boolean(fleetProfile)}
          />
        </CardContent>
      </Card>

      {runtime === "snowflake" ? <SnowflakeGrantsCard /> : null}

      {envChanged && !envConfirmed ? (
        <GuideCard
          tone="warning"
          title="This Start uses a different environment"
          why={`The last job on this control plane ran on ${runtimeLabel(lastSuccessRuntime ?? "local")}. This Start will use ${runtimeLabel(runtime)}. Confirm in the bar at the bottom of this page.`}
          actions={[]}
        />
      ) : null}

      <div>
        <button
          type="button"
          aria-expanded={advancedOpen}
          className="flex w-full items-center justify-between rounded-lg border border-border bg-card px-4 py-3 text-left text-sm font-semibold hover:bg-muted/40"
          onClick={() => setAdvancedOpen((open) => !open)}
        >
          Advanced: parallelism
          <span className="text-xs font-normal text-muted-foreground" aria-hidden="true">
            {advancedOpen ? "Hide" : "Show"}
          </span>
        </button>
        {advancedOpen ? (
          <div className="mt-3 space-y-4 rounded-lg border border-border bg-card p-4">
            <Field label="Provider calls in flight">
              <Input
                type="number"
                min={1}
                max={64}
                aria-label="Provider calls in flight"
                value={parallelism}
                onChange={(event) => setParallelism(Number(event.target.value))}
              />
            </Field>
          </div>
        ) : null}
      </div>

      {preflightVerdict ? (
        <Alert variant={preflightVerdict.ok ? "success" : "destructive"}>
          {preflightVerdict.ok
            ? "Preflight passed. The run is ready to submit."
            : preflightVerdict.errors.join(" ")}
        </Alert>
      ) : null}

      {scanPii.data?.hits.length ? (
        scanPii.data.blocked ? (
          <Alert variant="destructive">
            PII {scanPii.data.hits.map((hit) => `${hit.kind} in ${hit.source}`).join("; ")}. Embeddings will not be
            written until you acknowledge residual risk.
            <Button className="ml-2" size="sm" variant="secondary" onClick={() => ackPii.mutate({ sourceKey })}>
              Acknowledge residual risk
            </Button>
          </Alert>
        ) : (
          // What was found stays on screen after the acknowledgement, so the
          // person starting the run still sees what it carries.
          <Alert variant="warning" data-testid="pii-acknowledged">
            PII {scanPii.data.hits.map((hit) => `${hit.kind} in ${hit.source}`).join("; ")} · residual risk acknowledged
            for this source.
          </Alert>
        )
      ) : null}

      {preview ? (
        <Card>
          <CardHeader>
            <CardTitle>Effective configuration</CardTitle>
            <CardDescription>Secrets are redacted. YAML is a downloadable preview, not the first screen.</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea aria-label="YAML configuration text" readOnly value={preview} rows={16} className="font-mono text-xs" />
          </CardContent>
        </Card>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button variant="outline" onClick={() => void onPreview()} disabled={!request || previewConfig.isPending}>
          Preview configuration
        </Button>
        <Button variant="outline" onClick={() => void onPreflight()} disabled={!request || preflight.isPending}>
          Run preflight
        </Button>
        <Button
          variant="outline"
          onClick={() => request && scanPii.mutate({ source: request.source, sourceKey })}
          disabled={!request || scanPii.isPending}
        >
          Scan for PII
        </Button>
      </div>

      <div className="sticky bottom-0 z-10 -mx-4 mt-auto flex items-center justify-between gap-3 border-t border-border bg-background/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="min-w-0 flex-1 space-y-1">
          {grantsBlocked ? (
            <p className="truncate text-sm text-destructive">Snowflake grants are red. Mark them granted before Start.</p>
          ) : null}
          {envBlocked ? (
            <p className="truncate text-sm text-destructive">
              Last job used {runtimeLabel(lastSuccessRuntime ?? "local")}. Confirm {runtimeLabel(runtime)} before Start.
            </p>
          ) : null}
          {listingFailed ? (
            <p className="truncate text-sm text-destructive">
              Listing failed. Start stays disabled until the source lists {sourceNoun}.
            </p>
          ) : null}
          {ontologyBlocked && ontology ? (
            <p className="truncate text-sm text-destructive">{ontologyProblem(ontology)}</p>
          ) : null}
          {snowflakeIncomplete && !incompleteSource ? (
            <p className="truncate text-sm text-destructive">
              Name the Snowflake database, schema and bulk stage before Start.
            </p>
          ) : null}
          {revision ? (
            <p className="truncate text-sm text-muted-foreground" data-testid="revision-summary">
              Keeps {revision.keptCount} document{revision.keptCount === 1 ? "" : "s"}
              {revision.dropFileIds.length ? ` · removes ${revision.dropFileIds.length}` : ""}
              {objectCount > 0
                ? ` · adds ${objectCount}`
                : emptyRevision
                  ? " · a version needs at least one document: keep one, or add some"
                  : incompleteSource && !revision.dropFileIds.length
                    ? " · add documents or remove some to build a new version"
                    : ""}
            </p>
          ) : null}
          {revision ? null : listingPending ? (
            <p className="truncate text-sm text-muted-foreground">Listing files to estimate cost and time…</p>
          ) : emptyLocalListing || incompleteSource ? (
            <p className="truncate text-sm text-destructive">
              {choice === "upload" && !uploadPath
                ? "Drop files before Start."
                : incompleteSource
                  ? "Fill the required source fields before Start."
                  : `No ${sourceNoun} found ${sourcePlace}. Start stays disabled.`}
            </p>
          ) : null}
          {revision ? null : estimate.data && objectCount > 0 ? (
            <p className="truncate text-sm text-muted-foreground" data-testid="credit-envelope">
              Credit envelope {estimate.data.usdLow}–{estimate.data.usdHigh} usd · {estimate.data.minutesLow}–
              {estimate.data.minutesHigh} min
              <span className="hidden sm:inline">. {estimate.data.note}</span>
            </p>
          ) : !startBlocked ? (
            <p className="truncate text-sm text-muted-foreground">
              {objectCount === 0
                ? sources.isFetching || sources.isPaused
                  ? `Listing ${sourceNoun} to estimate cost and time…`
                  : browsable
                    ? `No ${sourceNoun} found ${sourcePlace}.`
                    : choice === "snowflake_stage"
                      ? "Cost is estimated after Start lists the stage."
                      : "Point at files to estimate cost and time."
                : "Estimating cost and time…"}
            </p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {envBlocked ? (
            <Button variant="secondary" onClick={confirmEnvironment}>
              Confirm environment
            </Button>
          ) : null}
          <Button
            onClick={() => void onSubmit()}
            disabled={submit.isPending || preflight.isPending || scanPii.isPending || !session.data || startBlocked}
            title={
              !startBlocked
                ? undefined
                : grantsBlocked
                  ? "Snowflake grants must be marked granted before Start"
                  : envBlocked
                    ? "Confirm the environment change before Start"
                    : emptyRevision
                      ? "Keep at least one document, or add some"
                      : ontologyBlocked && ontology
                        ? (ontologyProblem(ontology) ?? undefined)
                        : snowflakeIncomplete && !incompleteSource
                          ? "Name the Snowflake database, schema and bulk stage before Start"
                          : emptyLocalListing || incompleteSource
                            ? "Fill a usable source before Start"
                            : undefined
            }
          >
            {revision ? "Build new version" : "Start"}
          </Button>
        </div>
      </div>
    </div>
  );
}

type SnowflakeField =
  | "account"
  | "user"
  | "database"
  | "schema"
  | "warehouse"
  | "bulkStage"
  | "role"
  | "host"
  | "authenticator"
  | "credentialEnvironmentVariable"
  | "credentialField";

const ALL_SNOWFLAKE_FIELDS: SnowflakeField[] = [
  "account",
  "user",
  "database",
  "schema",
  "warehouse",
  "bulkStage",
  "role",
  "host",
  "authenticator",
  "credentialEnvironmentVariable",
  "credentialField",
];

/** On a fleet the account, user and credential are the fleet's; only the target is named. */
const FLEET_SNOWFLAKE_FIELDS: SnowflakeField[] = ["database", "schema", "warehouse", "bulkStage", "role"];

function runtimeLabel(runtime: string): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Local";
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}

function ChoiceTile({
  name,
  title,
  description,
  selected,
  onClick,
}: {
  name: string;
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={name}
      aria-pressed={selected}
      onClick={onClick}
      className={cn(
        "flex w-full flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors",
        selected ? "border-primary bg-accent ring-2 ring-primary" : "border-border bg-background hover:bg-muted/50",
      )}
    >
      <span className="flex w-full items-center justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        {selected ? (
          <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground">
            Selected
          </span>
        ) : null}
      </span>
      <span className="text-xs leading-snug text-muted-foreground">{description}</span>
    </button>
  );
}

function FileDropzone({
  onFiles,
  busy = false,
  hasFiles = false,
}: {
  onFiles: (files: FileList) => void;
  busy?: boolean;
  hasFiles?: boolean;
}) {
  const [over, setOver] = useState(false);

  return (
    <label
      data-testid="file-dropzone"
      onDragEnter={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        setOver(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        if (!event.currentTarget.contains(event.relatedTarget as Node)) {
          setOver(false);
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        if (event.dataTransfer.files.length) {
          onFiles(event.dataTransfer.files);
        }
      }}
      className={cn(
        "flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors",
        over ? "border-primary bg-accent" : "border-muted-foreground/40 bg-muted/40 hover:border-primary/50 hover:bg-muted/60",
        busy && "pointer-events-none opacity-60",
      )}
    >
      <input
        type="file"
        multiple
        className="sr-only"
        disabled={busy}
        aria-label="Upload documents"
        onChange={(event) => {
          if (event.target.files?.length) {
            onFiles(event.target.files);
          }
          event.target.value = "";
        }}
      />
      <Upload className="size-5 text-muted-foreground" aria-hidden="true" />
      <p className="text-sm font-medium">
        {busy ? "Uploading…" : hasFiles ? "Drop more files, or click to add" : "Drop files here, or click to browse"}
      </p>
      <p className="text-xs text-muted-foreground">
        PDF, markdown, and office documents. A second drop adds to the same folder.
      </p>
    </label>
  );
}

function ProviderFields({
  title,
  options,
  value,
  onChange,
  dimension = false,
  locked = false,
}: {
  title: string;
  options: typeof OCR_PROVIDERS;
  value: ProviderSelection;
  onChange: (value: ProviderSelection) => void;
  dimension?: boolean;
  locked?: boolean;
}) {
  const selected = options.find((option) => option.name === value.provider);
  if (locked) {
    return (
      <div className="space-y-1" data-testid={`fleet-provider-${title.toLowerCase()}`}>
        <p className="text-sm font-medium">{title}</p>
        <p className="text-sm">{selected?.label ?? value.provider}</p>
        {value.model ? <p className="text-xs text-muted-foreground">{value.model}</p> : null}
        {dimension && value.dimension ? (
          <p className="text-xs text-muted-foreground">{value.dimension} dimensions</p>
        ) : null}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <p className="text-sm font-medium">{title}</p>
      <Select
        value={value.provider}
        onValueChange={(name) => {
          const option = options.find((item) => item.name === name);
          if (!option) {
            return;
          }
          onChange(selectionFromOption(option));
        }}
      >
        <SelectTrigger aria-label={`${title} provider`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.name} value={option.name}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {selected?.needsModel ? (
        <Input
          aria-label={`${title} model`}
          placeholder="Model"
          value={value.model ?? ""}
          onChange={(event) => onChange({ ...value, model: event.target.value || null })}
        />
      ) : null}
      {selected?.needsEndpoint ? (
        <Input
          aria-label={`${title} endpoint`}
          placeholder="Endpoint"
          value={value.endpoint ?? ""}
          onChange={(event) => onChange({ ...value, endpoint: event.target.value || null })}
        />
      ) : null}
      {selected?.needsApiKey ? (
        <Input
          aria-label={`${title} API key environment variable`}
          placeholder="API key environment variable"
          value={value.apiKeyEnvironmentVariable ?? ""}
          onChange={(event) => onChange({ ...value, apiKeyEnvironmentVariable: event.target.value || null })}
        />
      ) : null}
      {dimension ? (
        <Input
          type="number"
          aria-label={`${title} dimension`}
          placeholder="Dimension"
          value={value.dimension ?? ""}
          onChange={(event) => onChange({ ...value, dimension: Number(event.target.value) || null })}
        />
      ) : null}
    </div>
  );
}

function fleetSelection(
  profile: { config: Record<string, unknown> },
  section: "ocr" | "llm" | "embedding",
  fallback: ProviderSelection,
): ProviderSelection {
  const values = profile.config[section];
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return fallback;
  }
  const record = values as Record<string, unknown>;
  if (typeof record.provider !== "string" || !record.provider) {
    return fallback;
  }
  return {
    ...providerSelection(section, record.provider),
    model: typeof record.model === "string" ? record.model : null,
    endpoint: typeof record.endpoint === "string" && !record.endpoint.startsWith("${") ? record.endpoint : null,
    dimension: typeof record.dimension === "number" ? record.dimension : null,
  };
}

/** The Snowflake account a fleet's workers write with, when its profile names one. */
function fleetSnowflakeAccount(profile: { config: Record<string, unknown> }): {
  account: string;
  user: string;
  host: string | null;
  authenticator: string | null;
} | null {
  const values = profile.config.snowflake;
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return null;
  }
  const record = values as Record<string, unknown>;
  const text = (key: string) =>
    typeof record[key] === "string" && !String(record[key]).startsWith("${") ? String(record[key]) : "";
  if (!text("account") || !text("user")) {
    return null;
  }
  return {
    account: text("account"),
    user: text("user"),
    host: text("host") || null,
    authenticator: text("authenticator") || null,
  };
}

function humanizeField(key: string): string {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (letter) => letter.toUpperCase())
    .trim();
}

function defaultSourceKind(capabilities: Set<Capability>): SourceKind {
  if (capabilities.has("snowflake_stage") && !capabilities.has("local")) {
    return "snowflake_stage";
  }
  if (capabilities.has("upload")) {
    return "upload";
  }
  if (capabilities.has("local")) {
    return "local_path";
  }
  return "upload";
}

function sourceChoiceOptions(capabilities: Set<Capability>, samplePacks: boolean) {
  const options: Array<{ value: SourceChoice; label: string }> = [];
  if (capabilities.has("upload")) {
    options.push({ value: "upload", label: "Upload" });
  }
  if (capabilities.has("local") && samplePacks) {
    options.push({ value: "sample", label: "Sample pack" });
  }
  if (capabilities.has("local")) {
    options.push({ value: "local_path", label: "Folder path" });
  }
  if (capabilities.has("azure_blob")) {
    options.push({ value: "azure_blob", label: "Azure Blob" });
  }
  if (capabilities.has("s3")) {
    options.push({ value: "s3", label: "S3-compatible bucket" });
  }
  if (capabilities.has("snowflake_stage")) {
    options.push({ value: "snowflake_stage", label: "Snowflake stage" });
  }
  return options;
}

function cryptoRandom(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`;
  }
  return `${prefix}_${Math.random().toString(16).slice(2, 14)}`;
}
