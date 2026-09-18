import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { appEnv } from "../env";
import { ClusterProfile, IngestionRequest, isSuccessStatus, standardSchema } from "../protocol/schema";
import { availableRuntimes } from "../runtime/resolve";
import { publicApiKeys } from "../api-keys";
import { publicProcedure, router, runEffect } from "./init";
import { qualityForGraph } from "../quality";
import { inspectFromDataset } from "../inspect-report";
import { AskHttpError } from "../ask/errors";
import { completeAsk } from "../ask/complete";
import { perspectiveScope } from "../ask/load";
import { scanPii } from "../pii";
import { estimateConsumption } from "../estimate";
import { documentStatusesFromEvents } from "../documents";
import {
  acknowledgePii,
  applyIncremental,
  assumeIdentity,
  createApiKey,
  createWatch,
  revokeApiKey,
  decideReview,
  enqueueReviews,
  forgetMany,
  loadWorkspace,
  markGrant,
  pinTarget,
  promotePerspective,
  promotionPlan,
  proposeOntology,
  publishVersion,
  recordEstimate,
  recordIdentityIncident,
  recordRuntimeSuccess,
  sampleHighConfidenceReviews,
  setPendingPromotion,
  setSuggestionMode,
  skipFile,
  staffIncident,
  toggleWatch,
  upsertPerspective,
} from "../workspace";
import { ontologyCoverage, type GoldGraph } from "../gold";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

const runIdInput = z.object({ runId: z.string().min(1) });
const graphIdInput = z.object({ graphId: z.string().min(1) });

export const appRouter = router({
  auth: router({
    session: publicProcedure.query(async ({ ctx }) => {
      const workspace = await loadWorkspace();
      return {
        viewer: ctx.viewer,
        runtime: ctx.runtimeName,
        capabilities: [...ctx.controlPlane.capabilities],
        identified: Boolean(ctx.viewer.userName),
        snowflakeHosted: appEnv().snowflakeHosted,
        availableRuntimes: availableRuntimes(),
        repositoryRoot: appEnv().repositoryRoot,
        role: workspace.role,
        suggestionMode: workspace.suggestionMode,
        lastSuccessRuntime: workspace.lastSuccessRuntime,
        lastConfigDigest: workspace.lastConfigDigest[ctx.runtimeName] ?? null,
        grants: workspace.grants,
      };
    }),
    assume: publicProcedure
      .input(
        z.object({
          userName: z.string(),
          email: z.string().optional(),
          roles: z.array(z.string()),
          role: z.enum(["operator", "analyst", "staff"]),
        }),
      )
      .mutation(({ input }) => assumeIdentity(input)),
    setSuggestions: publicProcedure
      .input(z.object({ mode: z.enum(["off", "on-request", "auto-fill"]) }))
      .mutation(({ input }) => setSuggestionMode(input.mode)),
  }),
  runs: router({
    list: publicProcedure
      .input(z.object({ limit: z.number().int().min(1).max(500).optional() }).optional())
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.listRuns(input?.limit ?? 100))),
    get: publicProcedure.input(runIdInput).query(({ ctx, input }) => runEffect(ctx.controlPlane.getRun(input.runId))),
    submit: publicProcedure.input(standardSchema(IngestionRequest)).mutation(async ({ ctx, input }) => {
      const request = requiredInput(input);
      const snapshot = await runEffect(ctx.controlPlane.submit(request));
      await recordRuntimeSuccess(
        request.runtime,
        `${request.sourceKind}:${String(
          (request.source as Record<string, unknown>).path ??
            (request.source as Record<string, unknown>).prefix ??
            (request.source as Record<string, unknown>).stage ??
            "",
        )}`,
      );
      return snapshot;
    }),
    cancel: publicProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.cancel(input.runId)),
    ),
    retry: publicProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.retry(input.runId)),
    ),
    recover: publicProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.recover(input.runId)),
    ),
    forget: publicProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.forget(input.runId)),
    ),
    forgetMany: publicProcedure
      .input(z.object({ runIds: z.array(z.string().min(1)).min(1) }))
      .mutation(async ({ ctx, input }) => {
        if (!ctx.controlPlane.capabilities.has("forget")) {
          throw new TRPCError({ code: "FORBIDDEN", message: "This runtime cannot forget graphs." });
        }
        return { forgotten: await forgetMany(input.runIds) };
      }),
    skipFile: publicProcedure
      .input(z.object({ runId: z.string().min(1), fileId: z.string().min(1) }))
      .mutation(({ input }) => skipFile(input.runId, input.fileId)),
    documents: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const workspace = await loadWorkspace();
      return documentStatusesFromEvents(snapshot.events, workspace.skippedFiles[input.runId] ?? []);
    }),
  }),
  ingestion: router({
    preflight: publicProcedure.input(standardSchema(IngestionRequest)).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.preflight(requiredInput(input))),
    ),
    sources: router({
      list: publicProcedure
        .input(
          z.object({
            source: z.record(z.string(), z.unknown()),
            limit: z.number().int().min(1).max(5_000).optional(),
          }),
        )
        .query(({ ctx, input }) => runEffect(ctx.controlPlane.listSourceObjects(input.source, input.limit))),
    }),
    config: router({
      preview: publicProcedure.input(standardSchema(IngestionRequest)).mutation(({ ctx, input }) =>
        runEffect(ctx.controlPlane.previewConfig(requiredInput(input))),
      ),
    }),
    estimate: publicProcedure
      .input(
        z.object({
          objectCount: z.number().int().min(0),
          runtime: z.string(),
          ocrProvider: z.string(),
          parallelism: z.number(),
          runId: z.string().optional(),
        }),
      )
      .query(async ({ input }) => {
        const estimate = estimateConsumption(input);
        if (input.runId) {
          await recordEstimate(input.runId, estimate);
        }
        return estimate;
      }),
    scanPii: publicProcedure
      .input(z.object({ source: z.record(z.string(), z.unknown()), sourceKey: z.string() }))
      .mutation(async ({ ctx, input }) => {
        const objects = await runEffect(ctx.controlPlane.listSourceObjects(input.source));
        const hits = await scanPii(objects);
        const workspace = await loadWorkspace();
        return {
          hits,
          blocked: hits.length > 0 && !workspace.piiAcknowledged.includes(input.sourceKey),
        };
      }),
    acknowledgePii: publicProcedure
      .input(z.object({ sourceKey: z.string() }))
      .mutation(({ input }) => acknowledgePii(input.sourceKey)),
    ontology: publicProcedure
      .input(z.object({ intent: z.string().min(8) }))
      .mutation(async ({ input }) => {
        const gold = await loadMartialArtsGold();
        const goldTypes = [...new Set((gold?.entities ?? []).map((entity) => entity.type.toUpperCase()))];
        const proposal = proposeOntology(input.intent, goldTypes);
        return {
          ...proposal,
          coverage: gold ? ontologyCoverage(proposal.types, gold) : null,
        };
      }),
  }),
  graphs: router({
    load: publicProcedure
      .input(z.object({ location: z.string().min(1), graphId: z.string().nullable().optional() }))
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.loadGraph(input.location, input.graphId))),
    loadRun: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      return runEffect(ctx.controlPlane.loadRunGraph(snapshot));
    }),
    rename: publicProcedure
      .input(z.object({ graphId: z.string().min(1), displayName: z.string().min(1) }))
      .mutation(({ ctx, input }) => runEffect(ctx.controlPlane.renameGraph(input.graphId, input.displayName))),
    delete: publicProcedure.input(graphIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.deleteGraph(input.graphId)),
    ),
    share: publicProcedure
      .input(z.object({ graphId: z.string(), granteeType: z.string(), grantee: z.string() }))
      .mutation(({ ctx, input }) =>
        runEffect(ctx.controlPlane.shareGraph(input.graphId, input.granteeType, input.grantee)),
      ),
    unshare: publicProcedure
      .input(z.object({ graphId: z.string(), granteeType: z.string(), grantee: z.string() }))
      .mutation(({ ctx, input }) =>
        runEffect(ctx.controlPlane.unshareGraph(input.graphId, input.granteeType, input.grantee)),
      ),
    owner: publicProcedure.input(graphIdInput).query(({ ctx, input }) =>
      runEffect(ctx.controlPlane.graphOwner(input.graphId)),
    ),
    shares: publicProcedure.input(graphIdInput).query(({ ctx, input }) =>
      runEffect(ctx.controlPlane.graphShares(input.graphId)),
    ),
    quality: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      return qualityForGraph({
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
      });
    }),
    inspect: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      const quality = await qualityForGraph({
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
      });
      return inspectFromDataset(snapshot, dataset, quality);
    }),
    ask: publicProcedure
      .input(
        z.object({
          runId: z.string().min(1),
          question: z.string().min(2),
          mode: z.enum(["local", "global", "hybrid", "drift", "auto"]).default("local"),
          perspectiveId: z.string().optional(),
          documentIds: z.array(z.string().min(1)).optional(),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
        if (!isSuccessStatus(snapshot.status)) {
          throw new TRPCError({
            code: "CONFLICT",
            message: `Graph ${input.runId} is ${snapshot.status}. Ask requires a finished graph.`,
          });
        }
        const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
        try {
          const scope = await perspectiveScope(input.perspectiveId, snapshot.graphId);
          return await completeAsk({
            dataset,
            question: input.question,
            mode: input.mode === "auto" ? undefined : input.mode,
            scope,
            documentIds: input.documentIds,
          });
        } catch (error) {
          if (error instanceof AskHttpError) {
            throw new TRPCError({
              code: error.status === 400 ? "BAD_REQUEST" : error.status === 409 ? "CONFLICT" : "BAD_REQUEST",
              message: error.message,
            });
          }
          throw error;
        }
      }),
    pin: publicProcedure
      .input(
        z.object({
          graphId: z.string(),
          targetId: z.string(),
          kind: z.enum(["node", "edge"]),
          note: z.string(),
        }),
      )
      .mutation(({ input }) => pinTarget(input)),
    promote: publicProcedure
      .input(
        z.object({
          fromRuntime: z.string(),
          toRuntime: z.enum(["local", "kubernetes", "snowflake"]),
          keepGraphId: z.boolean(),
        }),
      )
      .query(({ input }) => promotionPlan(input.fromRuntime, input.toRuntime, input.keepGraphId)),
    applyPromotion: publicProcedure
      .input(
        z.object({
          fromRuntime: z.string(),
          toRuntime: z.enum(["local", "kubernetes", "snowflake"]),
          keepGraphId: z.boolean(),
          graphId: z.string(),
          graphName: z.string(),
          sourceKind: z.string().default("local_path"),
          sourcePath: z.string().default(""),
        }),
      )
      .mutation(({ input }) => {
        const plan = promotionPlan(input.fromRuntime, input.toRuntime, input.keepGraphId);
        return setPendingPromotion({
          ...plan,
          graphId: input.graphId,
          graphName: input.graphName,
          sourceKind: input.sourceKind,
          sourcePath: input.sourcePath,
        });
      }),
    sampleReviews: publicProcedure.input(runIdInput).mutation(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      return enqueueReviews(sampleHighConfidenceReviews(snapshot.graphId, dataset.edges));
    }),
  }),
  workspace: router({
    get: publicProcedure.query(async () => {
      const state = await loadWorkspace();
      return { ...state, apiKeys: publicApiKeys(state.apiKeys) };
    }),
    perspective: publicProcedure
      .input(
        z.object({
          id: z.string().optional(),
          graphId: z.string(),
          name: z.string(),
          lifecycle: z.enum(["draft", "candidate", "production"]),
          search: z.string(),
          communityIds: z.array(z.string()),
          suggestedQuestions: z.array(z.string()),
        }),
      )
      .mutation(({ input }) => upsertPerspective(input)),
    promotePerspective: publicProcedure
      .input(z.object({ id: z.string(), lifecycle: z.enum(["draft", "candidate", "production"]) }))
      .mutation(({ input }) => promotePerspective(input.id, input.lifecycle)),
    review: publicProcedure
      .input(z.object({ id: z.string(), decision: z.enum(["keep", "drop", "merge", "relabel"]) }))
      .mutation(({ input }) => decideReview(input.id, input.decision)),
    publish: publicProcedure
      .input(z.object({ graphId: z.string(), note: z.string() }))
      .mutation(({ input }) => publishVersion(input.graphId, input.note)),
    toggleWatch: publicProcedure.input(z.object({ id: z.string() })).mutation(({ input }) => toggleWatch(input.id)),
    applyWatch: publicProcedure.input(z.object({ id: z.string() })).mutation(({ input }) => applyIncremental(input.id)),
    createWatch: publicProcedure
      .input(z.object({ graphId: z.string(), prefix: z.string().min(1) }))
      .mutation(({ input }) => createWatch(input)),
    createKey: publicProcedure.input(z.object({ name: z.string().min(1) })).mutation(({ input }) => createApiKey(input.name)),
    revokeKey: publicProcedure.input(z.object({ id: z.string().min(1) })).mutation(({ input }) => revokeApiKey(input.id)),
    identityIncident: publicProcedure
      .input(
        z.object({
          expectedPrincipal: z.string(),
          actualPrincipal: z.string(),
          graphOwner: z.string(),
          grants: z.string(),
        }),
      )
      .mutation(({ input }) => recordIdentityIncident(input)),
    clearPromotion: publicProcedure.mutation(() => setPendingPromotion(null)),
    markGrant: publicProcedure
      .input(z.object({ object: z.string(), ok: z.boolean() }))
      .mutation(({ input }) => markGrant(input.object, input.ok)),
    incident: publicProcedure.query(() => staffIncident()),
  }),
  fleet: router({
    cluster: publicProcedure
      .input(z.object({ namespace: z.string().optional() }).optional())
      .query(({ ctx, input }) =>
        runEffect(ctx.controlPlane.cluster(input?.namespace || appEnv().kubernetesNamespace)),
      ),
    nodeAssignments: publicProcedure
      .input(z.object({ namespace: z.string(), nodeName: z.string() }))
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.nodeAssignments(input.namespace, input.nodeName))),
  }),
  clusters: router({
    list: publicProcedure.query(({ ctx }) => runEffect(ctx.controlPlane.listClusters())),
    upsert: publicProcedure.input(standardSchema(ClusterProfile)).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.upsertCluster(requiredInput(input))),
    ),
    delete: publicProcedure.input(z.object({ name: z.string() })).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.deleteCluster(input.name)),
    ),
    select: publicProcedure.input(z.object({ name: z.string() })).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.selectCluster(input.name)),
    ),
    selected: publicProcedure.query(({ ctx }) => runEffect(ctx.controlPlane.selectedCluster())),
  }),
});

export type AppRouter = typeof appRouter;

function requiredInput<T>(input: T | undefined): T {
  if (input === undefined) {
    throw new TRPCError({ code: "BAD_REQUEST", message: "Invalid input" });
  }
  return input;
}

async function loadMartialArtsGold(): Promise<GoldGraph | null> {
  const file = path.join(appEnv().repositoryRoot, "data/martial_arts/gold.json");
  if (!existsSync(file)) {
    return null;
  }
  return JSON.parse(await readFile(file, "utf8")) as GoldGraph;
}
