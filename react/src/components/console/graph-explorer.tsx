"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { graphCounts } from "@/server/protocol/schema";
import { CANVAS_NODE_LIMIT, capGraph, filterGraph, graphFacets, type GraphFilters } from "@/server/graph-filter";
import { actualUsdFromConsumption, compareEstimateToActual, type ConsumptionEstimate } from "@/server/estimate";
import type { GraphDataset } from "@/server/protocol/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { GraphCanvas } from "@/components/console/graph-canvas";
import { NeighborhoodPicker, neighborhoodsOf } from "@/components/console/neighborhood-picker";
import { communityMemberIds, communityMembership } from "@/server/graph-filter";
import { RecordTable, prettyLabel, type RecordColumn } from "@/components/console/record-table";
import {
  buildGraphML,
  colorForCommunity,
  colorForType,
  findShortestPath,
  graphEdgeEnds,
  graphEdgeId,
  graphNodeId,
  graphNodeLabel,
  hopNeighborhood,
  type ColorMode,
  type FocusDepth,
  type GraphHover,
  type LayoutMode,
} from "@/lib/graph-geometry";
import { cn } from "@/lib/utils";
import { documentNameIndex, evidenceDocumentId, evidenceEntityId, evidenceRelationId } from "@/lib/evidence";

/** The overview draws only the best-connected core; a focus widens the cap. */
const OVERVIEW_NODE_LIMIT = 400;

export function GraphExplorer({
  dataset,
  perspectives = [],
  analyst = false,
  estimate,
  missingGold = [],
  goldPending = false,
  goldPresent = false,
  initialSearch = "",
  onPin,
}: {
  dataset: GraphDataset;
  perspectives?: Array<{
    id: string;
    name: string;
    search: string;
    communityIds: string[];
    lifecycle: string;
    suggestedQuestions?: string[];
  }>;
  analyst?: boolean;
  estimate?: ConsumptionEstimate | null;
  missingGold?: Array<{ id: string; source: string; target: string; relationType: string }>;
  goldPending?: boolean;
  goldPresent?: boolean;
  initialSearch?: string;
  onPin?: (targetId: string, kind: "node" | "edge") => void;
}) {
  const [search, setSearch] = useState(initialSearch);
  // A jump from the palette replaces whatever was typed.
  const [seededWith, setSeededWith] = useState(initialSearch);
  if (seededWith !== initialSearch) {
    setSeededWith(initialSearch);
    setSearch(initialSearch);
  }
  const [nodeTypes, setNodeTypes] = useState<string[]>([]);
  const [relationTypes, setRelationTypes] = useState<string[]>([]);
  const [communityIds, setCommunityIds] = useState<string[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [includeIsolates, setIncludeIsolates] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNeighborhood, setShowNeighborhood] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [perspectiveId, setPerspectiveId] = useState<string>("");
  // Empty until named: a pre-filled name invites typing into the middle of
  // it, and a perspective is worth naming for what it shows.
  const [perspectiveName, setPerspectiveName] = useState("");
  const [colorMode, setColorMode] = useState<ColorMode>("type");
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("force");
  const [focusDepth, setFocusDepth] = useState<FocusDepth>(null);
  const [showMinimap, setShowMinimap] = useState(false);
  const [hiddenTypes, setHiddenTypes] = useState<string[]>([]);
  const [pathEndpoints, setPathEndpoints] = useState<string[]>([]);
  const [hover, setHover] = useState<GraphHover | null>(null);
  const [flashNonce, setFlashNonce] = useState(0);
  // The chips above the canvas read the workspace; a saved or promoted
  // perspective has to reach them without a reload.
  const utils = trpc.useUtils();
  const savePerspective = trpc.workspace.perspective.useMutation({
    onSuccess: async () => {
      toast.success("Perspective saved as draft. Publish to production after gold is green.");
      await utils.workspace.get.invalidate();
    },
  });
  const promotePerspective = trpc.workspace.promotePerspective.useMutation({
    onSuccess: async () => {
      toast.success("Perspective lifecycle updated.");
      await utils.workspace.get.invalidate();
    },
  });
  const namedPerspectives = analyst
    ? perspectives.filter((item) => item.lifecycle === "production")
    : perspectives;
  const facets = useMemo(() => graphFacets(dataset), [dataset]);
  const membership = useMemo(() => communityMembership(dataset.communities ?? []), [dataset.communities]);
  const documentNames = useMemo(() => documentNameIndex(dataset.documents ?? []), [dataset.documents]);
  const filters: GraphFilters = useMemo(
    () => ({
      search,
      nodeTypes,
      relationTypes,
      communityIds,
      minimumConfidence: confidence,
      includeIsolates,
      // The whole match stays available for neighborhoods and exports; only
      // the canvas is capped below.
      limit: Number.POSITIVE_INFINITY,
    }),
    [search, nodeTypes, relationTypes, communityIds, confidence, includeIsolates],
  );
  const filtered = useMemo(() => filterGraph(dataset, filters), [dataset, filters]);
  const counts = graphCounts(dataset);
  const selectedNode = dataset.nodes.find((node) => String(node.id) === selectedId);
  const selectedEdge = dataset.edges.find((edge) => String(edge.id) === selectedId);
  const consumption = dataset.graphMetrics.consumption;
  const focused =
    showNeighborhood ||
    search.trim().length >= 2 ||
    nodeTypes.length > 0 ||
    relationTypes.length > 0 ||
    communityIds.length > 0;
  const focusedGraph = useMemo(() => {
    if (!showNeighborhood || !selectedId) {
      return filtered;
    }
    return hopNeighborhood(filtered.nodes, filtered.edges, selectedId);
  }, [filtered, selectedId, showNeighborhood]);
  const canvasLimit = focused ? CANVAS_NODE_LIMIT : OVERVIEW_NODE_LIMIT;
  const canvas = useMemo(
    () => capGraph(focusedGraph.nodes, focusedGraph.edges, canvasLimit),
    [canvasLimit, focusedGraph.edges, focusedGraph.nodes],
  );
  const pathHighlight = useMemo(() => {
    if (pathEndpoints.length !== 2) {
      return null;
    }
    const [src, dst] = pathEndpoints;
    return findShortestPath(
      canvas.edges.map((edge, index) => ({ id: graphEdgeId(edge, index), ...graphEdgeEnds(edge) })),
      src ?? "",
      dst ?? "",
    );
  }, [canvas.edges, pathEndpoints]);
  const typeLegend = useMemo(() => {
    const counts = new Map<string, number>();
    for (const node of canvas.nodes) {
      const type = String(node.primary_type ?? node.type ?? "Unknown");
      counts.set(type, (counts.get(type) ?? 0) + 1);
    }
    return [...counts.entries()].sort((left, right) => left[0].localeCompare(right[0]));
  }, [canvas.nodes]);
  const hiddenTypeSet = useMemo(() => new Set(hiddenTypes), [hiddenTypes]);
  const hoverNode = hover?.kind === "node" ? dataset.nodes.find((node) => graphNodeId(node) === hover.id) : null;
  const hoverEdge = hover?.kind === "edge" ? dataset.edges.find((edge, index) => graphEdgeId(edge, index) === hover.id) : null;
  const pathNames = pathEndpoints.map((id) => {
    const node = dataset.nodes.find((item) => graphNodeId(item) === id);
    return node ? graphNodeLabel(node) : id;
  });
  const neighborhoods = useMemo(() => neighborhoodsOf(dataset.communities ?? []), [dataset.communities]);
  const communityTitles = useMemo(
    () =>
      Object.fromEntries(
        (dataset.communities ?? []).map((community) => [String(community.id ?? ""), String(community.title ?? community.id ?? "")]),
      ),
    [dataset.communities],
  );
  const filtersActive =
    search.trim().length > 0 ||
    nodeTypes.length > 0 ||
    relationTypes.length > 0 ||
    communityIds.length > 0 ||
    confidence > 0 ||
    showNeighborhood ||
    hiddenTypes.length > 0 ||
    pathEndpoints.length > 0 ||
    focusDepth != null ||
    !includeIsolates ||
    Boolean(perspectiveId);

  function clearFilters() {
    setSearch("");
    setNodeTypes([]);
    setRelationTypes([]);
    setCommunityIds([]);
    setConfidence(0);
    setIncludeIsolates(true);
    setShowNeighborhood(false);
    setPerspectiveId("");
    setSelectedId(null);
    setFocusDepth(null);
    setHiddenTypes([]);
    setPathEndpoints([]);
  }

  function selectAndFly(id: string | null) {
    setSelectedId(id);
    if (id) {
      setFlashNonce((current) => current + 1);
    }
  }

  function toggleHiddenType(type: string) {
    setHiddenTypes((current) => (current.includes(type) ? current.filter((item) => item !== type) : [...current, type]));
  }

  function onShiftClickNode(id: string) {
    setPathEndpoints((current) => {
      if (current.includes(id)) {
        return current.filter((item) => item !== id);
      }
      if (current.length >= 2) {
        return [id];
      }
      return [...current, id];
    });
  }

  return (
    <div className="space-y-3">
      {namedPerspectives.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Perspectives</span>
          {namedPerspectives.map((item) => (
            <Button
              key={item.id}
              size="sm"
              variant={perspectiveId === item.id ? "default" : "outline"}
              onClick={() => {
                setPerspectiveId(item.id);
                setSearch(item.search);
                setCommunityIds(item.communityIds);
              }}
            >
              {item.name} · {item.lifecycle}
            </Button>
          ))}
          {perspectiveId && !analyst ? (
            <Button
              size="sm"
              variant="secondary"
              disabled={goldPending || !goldPresent || missingGold.length > 0}
              onClick={() => {
                if (goldPending) {
                  toast.error("Quality is still loading. Wait for gold before publishing.");
                  return;
                }
                if (!goldPresent) {
                  toast.error("No gold dataset is attached to this graph. Publish after gold is green.");
                  return;
                }
                if (missingGold.length > 0) {
                  toast.error(`Gold still missing ${missingGold.length} required relations. Publish after gold.`);
                  return;
                }
                promotePerspective.mutate({ id: perspectiveId, lifecycle: "production" });
              }}
            >
              Publish perspective
            </Button>
          ) : null}
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[16rem] flex-1">
          <Input
            aria-label="Search entities"
            className="h-9"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Entity name or description"
          />
        </div>
        {!analyst ? (
          <>
            <Input
              aria-label="Perspective name"
              className="h-9 max-w-[12rem]"
              value={perspectiveName}
              onChange={(event) => setPerspectiveName(event.target.value)}
              placeholder="Perspective name"
            />
            <Button
              size="sm"
              variant="outline"
              disabled={!perspectiveName.trim() || savePerspective.isPending}
              title={perspectiveName.trim() ? undefined : "Name the perspective before saving it"}
              onClick={() =>
                savePerspective.mutate(
                  {
                    graphId: dataset.graphId,
                    name: perspectiveName.trim(),
                    lifecycle: "draft",
                    search,
                    communityIds,
                    suggestedQuestions: search ? [`What connects to ${search}?`] : [],
                  },
                  { onSuccess: () => setPerspectiveName("") },
                )
              }
            >
              Save as perspective
            </Button>
          </>
        ) : null}
      </div>
      {neighborhoods.length > 0 ? (
        <NeighborhoodPicker
          neighborhoods={neighborhoods}
          value={communityIds}
          onChange={setCommunityIds}
          trailing={
            selectedId ? (
              <Button
                size="sm"
                variant={showNeighborhood ? "default" : "ghost"}
                className="h-7 text-xs"
                aria-pressed={showNeighborhood}
                onClick={() => setShowNeighborhood((current) => !current)}
              >
                {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
              </Button>
            ) : null
          }
        />
      ) : selectedId ? (
        <Button
          size="sm"
          variant={showNeighborhood ? "default" : "outline"}
          aria-pressed={showNeighborhood}
          onClick={() => setShowNeighborhood((current) => !current)}
        >
          {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
        </Button>
      ) : null}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          {counts.documents} docs · {counts.nodes} entities · {counts.edges} relations · {counts.evidence} evidence
        </p>
        <div className="flex flex-wrap gap-2">
          {filtersActive ? (
            <Button size="sm" variant="outline" onClick={clearFilters}>
              Clear filters
            </Button>
          ) : null}
          {canvas.nodes.length > 0 ? (
            <>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                const blob = new Blob([JSON.stringify({ nodes: focusedGraph.nodes, edges: focusedGraph.edges }, null, 2)], {
                  type: "application/json",
                });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `${dataset.graphId}-subgraph.json`;
                link.click();
                URL.revokeObjectURL(url);
                toast.success("Exported subgraph");
              }}
            >
              Export subgraph
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                const blob = new Blob([buildGraphML(focusedGraph.nodes, focusedGraph.edges)], { type: "application/graphml+xml" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `${dataset.graphId}.graphml`;
                link.click();
                URL.revokeObjectURL(url);
                toast.success("Exported GraphML");
              }}
            >
              Export GraphML
            </Button>
            </>
          ) : null}
        </div>
      </div>
      <details
        className="rounded-md border border-border p-3"
        onToggle={(event) => setFiltersOpen((event.currentTarget as HTMLDetailsElement).open)}
      >
        <summary className="cursor-pointer text-sm font-medium">Filters</summary>
        <div className={filtersOpen ? "mt-3 grid gap-3 md:grid-cols-3" : ""}>
          <MultiSelect
            label="Entity types"
            options={facets.nodeTypes}
            value={nodeTypes}
            onChange={setNodeTypes}
            visible={filtersOpen}
          />
          <MultiSelect
            label="Relation types"
            options={facets.relationTypes}
            value={relationTypes}
            onChange={setRelationTypes}
            visible={filtersOpen}
          />
          <MultiSelect
            label="Communities"
            options={facets.communityIds}
            labels={communityTitles}
            value={communityIds}
            onChange={setCommunityIds}
            visible={filtersOpen}
          />
          {filtersOpen ? (
            <>
              <label className="grid gap-2 text-sm">
                <span className="text-sm font-medium leading-none">Minimum confidence ({confidence.toFixed(2)})</span>
                <Slider
                  aria-label="Minimum confidence"
                  value={[confidence]}
                  max={1}
                  step={0.05}
                  onValueChange={(value) => setConfidence(value[0] ?? 0)}
                />
              </label>
              <label className="flex items-center gap-2 text-sm">
                <Switch checked={includeIsolates} onCheckedChange={setIncludeIsolates} aria-label="Include isolated entities" />
                Include isolated entities
              </label>
            </>
          ) : null}
        </div>
        {filtersOpen && filtersActive ? (
          <Button className="mt-3" size="sm" variant="outline" onClick={clearFilters}>
            Clear filters
          </Button>
        ) : null}
      </details>
      {focusedGraph.nodes.length > 0 ? (
        <p className="text-sm text-muted-foreground" data-testid="explore-scope">
          Showing {canvas.nodes.length.toLocaleString()} of {canvas.totalNodes.toLocaleString()} entities on the canvas
          {showNeighborhood && selectedId ? " in the selected neighborhood" : ""}.
          {canvas.totalNodes > canvas.nodes.length
            ? " The best-connected ones are drawn; search or a neighborhood focuses the rest. Summary metrics cover the full graph."
            : " Summary metrics cover the full graph."}
        </p>
      ) : (
        <p className="text-sm text-muted-foreground">
          No entities in this filter.
          {filtersActive ? " Clear filters to return to the overview." : ""}
        </p>
      )}
      {canvas.nodes.length > 0 ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md border border-border bg-card px-3 py-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Color by</span>
            <Tabs value={colorMode} onValueChange={(value) => setColorMode(value as ColorMode)}>
              <TabsList className="h-7">
                <TabsTrigger value="type" className="h-6 px-2 text-xs">
                  Type
                </TabsTrigger>
                <TabsTrigger value="community" className="h-6 px-2 text-xs" disabled={neighborhoods.length === 0}>
                  Community
                </TabsTrigger>
                <TabsTrigger value="degree" className="h-6 px-2 text-xs">
                  Degree
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Layout</span>
            <Tabs value={layoutMode} onValueChange={(value) => setLayoutMode(value as LayoutMode)}>
              <TabsList className="h-7">
                <TabsTrigger value="force" className="h-6 px-2 text-xs">
                  Force
                </TabsTrigger>
                <TabsTrigger value="clustered" className="h-6 px-2 text-xs" disabled={neighborhoods.length === 0}>
                  Clustered
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Focus</span>
            <Tabs
              value={focusDepth === null ? "off" : String(focusDepth)}
              onValueChange={(value) => {
                if (value === "off") {
                  setFocusDepth(null);
                  return;
                }
                const depth = Number(value);
                if (depth === 1 || depth === 2 || depth === 3) {
                  setFocusDepth(depth);
                }
              }}
            >
              <TabsList className="h-7">
                <TabsTrigger value="off" className="h-6 px-2 text-xs">
                  Focus off
                </TabsTrigger>
                <TabsTrigger value="1" className="h-6 px-2 text-xs" disabled={!selectedId}>
                  1 hop
                </TabsTrigger>
                <TabsTrigger value="2" className="h-6 px-2 text-xs" disabled={!selectedId}>
                  2 hops
                </TabsTrigger>
                <TabsTrigger value="3" className="h-6 px-2 text-xs" disabled={!selectedId}>
                  3 hops
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>
      ) : null}
      <p className="text-xs text-muted-foreground">
        Scroll to zoom, drag to pan, double-click an entity to fly in. Shift-click two entities to find a path.
      </p>
      <div className={cn("grid items-start gap-4", selectedNode || selectedEdge ? "lg:grid-cols-[2fr_1fr]" : "")}>
        <div className="relative">
          <GraphCanvas
            nodes={canvas.nodes}
            edges={canvas.edges}
            selectedId={selectedId}
            onSelect={setSelectedId}
            compact={canvas.nodes.length === 0}
            colorMode={colorMode}
            layoutMode={layoutMode}
            focusDepth={showNeighborhood ? 1 : focusDepth}
            pathHighlight={pathHighlight}
            hiddenTypes={hiddenTypeSet}
            membership={membership}
            flashNonce={flashNonce}
            showMinimap={showMinimap}
            onToggleMinimap={() => setShowMinimap((current) => !current)}
            onHover={setHover}
            onShiftClickNode={onShiftClickNode}
          />
          {canvas.nodes.length === 0 ? (
            <p className="pointer-events-none absolute inset-0 flex items-center justify-center p-6 text-center text-sm font-medium text-foreground">
              No entities to draw.{filtersActive ? " Clear filters to return to the overview." : ""}
            </p>
          ) : null}
          {pathEndpoints.length > 0 ? (
            <div className="absolute top-2 left-2 z-10 max-w-xs rounded-md border border-border bg-background/95 px-2 py-1.5 text-[11px] shadow-sm backdrop-blur-sm">
              <p className="font-semibold">Path finder</p>
              <p>
                {pathNames[0] ?? "—"}
                {pathEndpoints.length > 1 ? ` → ${pathNames[1]}` : " → shift-click another entity"}
              </p>
              {pathEndpoints.length === 2 ? (
                <p className="text-muted-foreground" role="status">
                  {pathHighlight && pathHighlight.edgeIds.size > 0
                    ? `${pathHighlight.edgeIds.size} hop${pathHighlight.edgeIds.size === 1 ? "" : "s"}`
                    : "No path found"}
                </p>
              ) : null}
              <button type="button" className="text-primary hover:underline" onClick={() => setPathEndpoints([])}>
                Clear path
              </button>
            </div>
          ) : null}
          {typeLegend.length > 0 && colorMode === "type" ? (
            <div className="absolute bottom-2 left-2 z-10 max-w-[min(100%,28rem)] rounded-md border border-border bg-background/95 px-2 py-1.5 shadow-sm backdrop-blur-sm">
              <div className="mb-1 flex items-center justify-between gap-3">
                <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Entity types</p>
                {hiddenTypes.length > 0 ? (
                  <button type="button" className="text-[10px] font-medium text-primary hover:underline" onClick={() => setHiddenTypes([])}>
                    Show all
                  </button>
                ) : null}
              </div>
              <ul className="flex flex-wrap gap-1.5" aria-label="Entity type colors">
                {typeLegend.map(([type, count]) => {
                  const hidden = hiddenTypes.includes(type);
                  return (
                    <li key={type}>
                      <button
                        type="button"
                        aria-pressed={!hidden}
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 text-[11px]",
                          hidden ? "text-muted-foreground line-through opacity-50" : "text-foreground",
                        )}
                        onClick={() => toggleHiddenType(type)}
                      >
                        <span className="size-2.5 shrink-0 rounded-full" style={{ background: colorForType(type) }} aria-hidden="true" />
                        {prettyLabel(type)}
                        <span className="text-muted-foreground">{count}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : colorMode === "community" && neighborhoods.length > 0 ? (
            <ul className="absolute bottom-2 left-2 z-10 flex max-w-[min(100%,28rem)] flex-wrap gap-1.5 rounded-md border border-border bg-background/95 px-2 py-1.5 text-[11px] shadow-sm" aria-label="Community colors">
              {neighborhoods.slice(0, 8).map((community) => (
                <li key={community.id} className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 shrink-0 rounded-full" style={{ background: colorForCommunity(community.id) }} aria-hidden="true" />
                  {prettyLabel(community.title)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        {selectedNode || selectedEdge ? (
        <Card data-testid="explore-selection">
          <CardHeader>
            <CardTitle>Selection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {selectedNode ? (
              <>
                <p className="font-medium">{String(selectedNode.name ?? selectedNode.id)}</p>
                <p className="text-muted-foreground">{String(selectedNode.primary_type ?? selectedNode.type ?? "")}</p>
                <p>{String(selectedNode.description ?? "")}</p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant={showNeighborhood ? "default" : "outline"}
                    aria-pressed={showNeighborhood}
                    onClick={() => setShowNeighborhood((current) => !current)}
                  >
                    {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onPin?.(String(selectedNode.id), "node")}
                  >
                    Looks wrong
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const evidence =
                        dataset.evidence.find((row) => evidenceEntityId(row) === String(selectedNode.id)) ??
                        dataset.evidence.find((row) => {
                          const quote = String(row.quote ?? "").toLowerCase();
                          const name = String(selectedNode.name ?? "").toLowerCase();
                          return name && quote.includes(name);
                        });
                      const quote = String(evidence?.quote ?? selectedNode.description ?? selectedNode.name ?? "");
                      const documentId = evidence ? documentNames.get(evidenceDocumentId(evidence)) ?? evidenceDocumentId(evidence) : "";
                      void navigator.clipboard.writeText(`${documentId} · ${quote}`.trim());
                      toast.success("Copied citation");
                    }}
                  >
                    Copy citation
                  </Button>
                </div>
              </>
            ) : selectedEdge ? (
              <>
                <p className="font-medium">{String(selectedEdge.relation_type ?? selectedEdge.relationType)}</p>
                <p>
                  {String(selectedEdge.source_node_id ?? selectedEdge.source)} →{" "}
                  {String(selectedEdge.target_node_id ?? selectedEdge.target)}
                </p>
                {selectedEdge.description ? (
                  <p className="text-muted-foreground">{String(selectedEdge.description)}</p>
                ) : null}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant={showNeighborhood ? "default" : "outline"}
                    aria-pressed={showNeighborhood}
                    onClick={() => setShowNeighborhood((current) => !current)}
                  >
                    {showNeighborhood ? "Hide neighborhood" : "Show neighborhood"}
                  </Button>
                  {onPin ? (
                    <Button size="sm" variant="outline" onClick={() => onPin(String(selectedEdge.id), "edge")}>
                      Looks wrong
                    </Button>
                  ) : null}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const evidence = dataset.evidence.find(
                        (row) => evidenceRelationId(row) === String(selectedEdge.id),
                      );
                      const quote = String(evidence?.quote ?? selectedEdge.description ?? selectedEdge.relation_type ?? "");
                      const documentId = evidence ? documentNames.get(evidenceDocumentId(evidence)) ?? evidenceDocumentId(evidence) : "";
                      void navigator.clipboard.writeText(`${documentId} · ${quote}`.trim());
                      toast.success("Copied citation");
                    }}
                  >
                    Copy citation
                  </Button>
                </div>
              </>
            ) : null}
          </CardContent>
        </Card>
        ) : null}
      </div>
      {hover && (hoverNode || hoverEdge) ? (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded-md border border-border bg-background px-2 py-1.5 text-[11px] shadow-md"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
          role="tooltip"
        >
          {hoverNode ? (
            <>
              <p className="font-semibold">{graphNodeLabel(hoverNode)}</p>
              <p className="text-muted-foreground">{String(hoverNode.primary_type ?? hoverNode.type ?? "")}</p>
              {hoverNode.description ? (
                <p className="mt-0.5 line-clamp-3 text-muted-foreground">{String(hoverNode.description)}</p>
              ) : null}
            </>
          ) : hoverEdge ? (
            <>
              <p className="font-semibold">{String(hoverEdge.relation_type ?? hoverEdge.relationType ?? "")}</p>
              <p className="text-muted-foreground">
                {String(hoverEdge.source_node_id ?? hoverEdge.source)} → {String(hoverEdge.target_node_id ?? hoverEdge.target)}
              </p>
              {hoverEdge.description ? (
                <p className="mt-0.5 line-clamp-3 text-muted-foreground">{String(hoverEdge.description)}</p>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
      <Tabs defaultValue="entities">
        <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
          <TabsTrigger value="entities">Entities</TabsTrigger>
          <TabsTrigger value="relations">Relations</TabsTrigger>
          <TabsTrigger value="communities">Communities</TabsTrigger>
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="consumption">Consumption</TabsTrigger>
        </TabsList>
        <ExploreTables
          dataset={dataset}
          graph={focusedGraph}
          documentNames={documentNames}
          missingGold={missingGold}
          selectedId={selectedId}
          onSelect={selectAndFly}
        />
        <TabsContent value="consumption">
          <ConsumptionPanel consumption={consumption} estimate={estimate ?? null} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ConsumptionPanel({
  consumption,
  estimate,
}: {
  consumption: unknown;
  estimate: ConsumptionEstimate | null;
}) {
  if (!consumption || typeof consumption !== "object") {
    return (
      <p className="text-sm text-muted-foreground">
        This graph reports no recorded usage, so there are no tokens, pages or cost to show.
      </p>
    );
  }
  const record = consumption as {
    totals?: { usd?: number; unpriced_calls?: number };
    events?: Array<Record<string, unknown>>;
  };
  const usd = record.totals?.usd;
  const unpriced = record.totals?.unpriced_calls ?? 0;
  const comparison = compareEstimateToActual(estimate ?? undefined, actualUsdFromConsumption(consumption));
  return (
    <div className="space-y-3">
      <p className="text-sm" data-testid="estimate-vs-actual">
        {usd != null ? `Derived cost ${usd} usd` : "Usage measured, cost not priced"}
        {unpriced ? ` · ${unpriced} unpriced calls` : ""}
        {comparison ? ` · ${comparison.label}` : ""}
      </p>
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          const blob = new Blob([JSON.stringify({ estimate, actual: record }, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = "consumption-export.json";
          link.click();
          URL.revokeObjectURL(url);
          toast.success("Exported consumption");
        }}
      >
        Export consumption
      </Button>
      {(record.events ?? []).length === 0 ? (
        <p className="text-sm text-muted-foreground">No per-call events were recorded for this graph.</p>
      ) : (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Stage</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>Tokens</TableHead>
            <TableHead>Locality</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {(record.events ?? []).map((event, index) => (
            <TableRow key={index}>
              <TableCell>{String(event.stage ?? "")}</TableCell>
              <TableCell>{String(event.model ?? event.provider ?? "")}</TableCell>
              <TableCell>{String(event.total_tokens ?? "")}</TableCell>
              <TableCell>{String(event.locality ?? "")}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      )}
    </div>
  );
}

function MultiSelect({
  label,
  options,
  labels = {},
  value,
  onChange,
  visible = true,
}: {
  label: string;
  options: string[];
  /** Display names for options that are ids, such as community titles. */
  labels?: Record<string, string>;
  value: string[];
  onChange: (value: string[]) => void;
  visible?: boolean;
}) {
  const labelOf = (option: string) => labels[option] ?? prettyLabel(option);
  return (
    // Content sits at the top of its column: the three facet columns share a
    // row, and the tallest one must not stretch the others' chips.
    <div className={visible ? "grid content-start gap-1.5 text-sm" : "sr-only"}>
      <span className="text-sm font-medium leading-none">{label}</span>
      <select
        multiple
        tabIndex={-1}
        aria-label={label}
        className="sr-only"
        value={value}
        onChange={(event) => onChange([...event.target.selectedOptions].map((option) => option.value))}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {labelOf(option)}
          </option>
        ))}
      </select>
      {visible ? (
      <div className="flex flex-wrap content-start items-start gap-1.5">
        {options.length === 0 ? (
          <p className="text-xs text-muted-foreground">None on this graph.</p>
        ) : (
          options.map((option) => {
            const selected = value.includes(option);
            return (
              <button
                key={option}
                type="button"
                aria-pressed={selected}
                className={cn(
                  "rounded-md border px-2 py-1 text-xs",
                  selected ? "border-primary bg-accent text-foreground" : "border-border bg-background text-muted-foreground hover:bg-muted/60",
                )}
                onClick={() =>
                  onChange(selected ? value.filter((item) => item !== option) : [...value, option])
                }
              >
                {labelOf(option)}
              </button>
            );
          })
        )}
      </div>
      ) : null}
    </div>
  );
}

/**
 * The four record tabs under the canvas. Entities and relations are the
 * filtered graph itself; communities and evidence follow it, so a filter
 * that hides an entity also hides the communities it alone populated and
 * the quotes that only ground it. Tables never see more than the graph
 * loaded, so "total" is the loaded count, not the summary metric.
 */
function ExploreTables({
  dataset,
  graph,
  documentNames,
  missingGold,
  selectedId,
  onSelect,
}: {
  dataset: GraphDataset;
  graph: { nodes: readonly Record<string, unknown>[]; edges: readonly Record<string, unknown>[] };
  documentNames: Map<string, string>;
  missingGold: ReadonlyArray<{ id: string }>;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const nodeNames = useMemo(
    () => new Map(dataset.nodes.map((node) => [graphNodeId(node), graphNodeLabel(node)] as const)),
    [dataset.nodes],
  );
  const edgesById = useMemo(
    () => new Map(dataset.edges.map((edge, index) => [graphEdgeId(edge, index), edge] as const)),
    [dataset.edges],
  );
  const visibleNodeIds = useMemo(() => new Set(graph.nodes.map((node) => graphNodeId(node))), [graph.nodes]);
  const visibleEdgeIds = useMemo(() => new Set(graph.edges.map((edge, index) => graphEdgeId(edge, index))), [graph.edges]);
  // Only a filter that removed something narrows the other two tabs; a row
  // whose subject the graph never had stays visible either way.
  const narrowed = graph.nodes.length < dataset.nodes.length || graph.edges.length < dataset.edges.length;
  const communities = useMemo(() => {
    if (!narrowed) {
      return dataset.communities;
    }
    return dataset.communities.filter((community) => {
      const members = communityMemberIds(community);
      return members.length === 0 || members.some((member) => visibleNodeIds.has(member));
    });
  }, [dataset.communities, narrowed, visibleNodeIds]);
  const evidence = useMemo(() => {
    if (!narrowed) {
      return dataset.evidence;
    }
    return dataset.evidence.filter((row) => {
      const relationId = evidenceRelationId(row);
      const entityId = evidenceEntityId(row);
      if (!relationId && !entityId) {
        return true;
      }
      return (relationId != null && visibleEdgeIds.has(relationId)) || (entityId != null && visibleNodeIds.has(entityId));
    });
  }, [dataset.evidence, narrowed, visibleEdgeIds, visibleNodeIds]);
  const highlightIds = useMemo(() => new Set(missingGold.map((item) => item.id)), [missingGold]);

  // Column definitions close over the name indexes, and the table re-sorts
  // whenever they change, so they are rebuilt only when an index does; the
  // canvas hover re-renders this component far more often than that.
  const columns = useMemo(() => {
    const nameOf = (id: unknown) => (id == null ? "" : (nodeNames.get(String(id)) ?? String(id)));
    const relationLabel = (edge: Record<string, unknown>) => {
      const ends = graphEdgeEnds(edge);
      return `${nameOf(ends.source)} ${String(edge.relation_type ?? edge.relationType ?? "")} ${nameOf(ends.target)}`.trim();
    };
    const entityColumns: RecordColumn[] = [
      { key: "name", value: (row) => graphNodeLabel(row) },
      { key: "primary_type", label: "Type", kind: "type", value: (row) => row.primary_type ?? row.type ?? "", color: colorForType },
      { key: "description", kind: "long" },
      { key: "id", kind: "id" },
    ];
    const relationColumns: RecordColumn[] = [
      {
        key: "source_node_id",
        label: "Source",
        value: (row) => nameOf(graphEdgeEnds(row).source),
        title: (row) => String(graphEdgeEnds(row).source),
      },
      { key: "relation_type", label: "Relation", kind: "type", value: (row) => row.relation_type ?? row.relationType ?? "" },
      {
        key: "target_node_id",
        label: "Target",
        value: (row) => nameOf(graphEdgeEnds(row).target),
        title: (row) => String(graphEdgeEnds(row).target),
      },
      { key: "confidence", kind: "number" },
      { key: "id", kind: "id" },
    ];
    const communityColumns: RecordColumn[] = [
      { key: "title", value: (row) => row.title ?? row.id ?? "" },
      { key: "size", kind: "number", value: (row) => communityMemberIds(row).length },
      { key: "summary", kind: "long" },
    ];
    const evidenceColumns: RecordColumn[] = [
      {
        key: "document_id",
        label: "Document",
        width: "w-56",
        value: (row) => documentNames.get(evidenceDocumentId(row)) ?? evidenceDocumentId(row),
        title: (row) => evidenceDocumentId(row) || undefined,
      },
      { key: "quote", kind: "long" },
      {
        key: "supports",
        width: "w-72",
        value: (row) => {
          const relationId = evidenceRelationId(row);
          const edge = relationId ? edgesById.get(relationId) : undefined;
          if (edge) {
            return relationLabel(edge);
          }
          return relationId ?? nameOf(evidenceEntityId(row));
        },
        title: (row) => evidenceRelationId(row) ?? evidenceEntityId(row) ?? undefined,
      },
    ];
    return { entities: entityColumns, relations: relationColumns, communities: communityColumns, evidence: evidenceColumns };
  }, [documentNames, edgesById, nodeNames]);

  return (
    <>
      <TabsContent value="entities">
        <RecordTable
          rows={graph.nodes}
          total={dataset.nodes.length}
          columns={columns.entities}
          noun={{ one: "entity", many: "entities" }}
          onSelect={onSelect}
          selectedId={selectedId}
        />
      </TabsContent>
      <TabsContent value="relations">
        <RecordTable
          rows={graph.edges}
          total={dataset.edges.length}
          columns={columns.relations}
          noun={{ one: "relation", many: "relations" }}
          onSelect={onSelect}
          selectedId={selectedId}
          highlightIds={highlightIds}
        />
      </TabsContent>
      <TabsContent value="communities">
        <RecordTable rows={communities} total={dataset.communities.length} columns={columns.communities} noun={{ one: "community", many: "communities" }} />
      </TabsContent>
      <TabsContent value="evidence">
        <RecordTable
          rows={evidence}
          total={dataset.evidence.length}
          columns={columns.evidence}
          noun={{ one: "evidence row", many: "evidence rows" }}
          onSelect={onSelect}
          selectedId={selectedId}
          selectionId={(row) => evidenceRelationId(row) ?? evidenceEntityId(row)}
        />
      </TabsContent>
    </>
  );
}

