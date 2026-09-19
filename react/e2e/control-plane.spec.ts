import { readFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";

async function confirmEnvironmentIfNeeded(page: Page) {
  const confirm = page.getByRole("button", { name: "Confirm environment" });
  await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  if (await confirm.isVisible()) {
    await confirm.click();
    await expect(confirm).toHaveCount(0);
  }
}

async function chooseYourFiles(page: Page) {
  const yours = page.getByRole("button", { name: "Your files", exact: true });
  if (await yours.isVisible() && (await yours.getAttribute("aria-pressed")) !== "true") {
    await yours.click();
  }
}

async function useSamplePack(page: Page, name: "Martial arts" | "Deep learning papers" = "Martial arts") {
  await page.getByRole("button", { name: "Sample pack", exact: true }).click();
  const tile = page.getByRole("button", { name, exact: true });
  await expect(tile).toBeVisible();
  if ((await tile.getAttribute("aria-pressed")) !== "true") {
    await tile.click();
  }
}

async function openMoreSources(page: Page) {
  await chooseYourFiles(page);
  const summary = page.locator("summary").filter({ hasText: /folder path or object storage/i });
  await expect(summary).toBeVisible();
  const details = page.locator("details").filter({ has: summary });
  if ((await details.getAttribute("open")) === null) {
    await summary.click();
  }
}

// React Query parks retries while the document is hidden; this flips what
// it reads without needing a second window.
async function setVisibility(page: Page, state: "hidden" | "visible") {
  await page.evaluate((next) => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => next });
    Object.defineProperty(document, "hidden", { configurable: true, get: () => next === "hidden" });
    document.dispatchEvent(new Event("visibilitychange", { bubbles: true }));
  }, state);
}

async function useFolderPath(page: Page, path: string) {
  await openMoreSources(page);
  await page.getByLabel("Source kind").click();
  await page.getByRole("option", { name: "Local path" }).click();
  await page.getByLabel("Local path").fill(path);
}

test.describe("local martial arts graph", () => {
  test("lists the seeded graph and opens the explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await expect(page.getByTestId("status-sentence")).toContainText("Ready to explore");
    await expect(page.getByRole("tab", { name: "Quality" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Export review bundle" })).toBeVisible();
    await expect(page.getByTestId("guide-card")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Show neighborhood" })).toHaveCount(0);
    await page.getByRole("tab", { name: "Quality" }).click();
    await expect(page.getByText(/required gold relations/i)).toBeVisible();
    await page.getByRole("tab", { name: "Explore" }).click();
    await expect(page.getByRole("tab", { name: "Entities" })).toBeVisible();
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await page.getByPlaceholder("Entity name or description").fill("Kano");
    await expect(page.getByRole("cell", { name: "Jigoro Kano" }).first()).toBeVisible();
  });

  test("zooms, fits, and recolors the graph explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Explore" }).click();
    const canvas = page.getByTestId("graph-canvas");
    await expect(canvas).toBeVisible();
    await expect(page.getByText(/scroll to zoom/i)).toBeVisible();
    const before = await canvas.getAttribute("data-zoom");
    await page.getByRole("button", { name: "Zoom in" }).click();
    await expect.poll(async () => canvas.getAttribute("data-zoom")).not.toBe(before);
    await page.getByRole("button", { name: "Fit graph" }).click();
    await page.getByRole("tab", { name: "Degree" }).click();
    await expect(page.getByRole("tab", { name: "Degree" })).toHaveAttribute("data-state", "active");
    await page.getByRole("button", { name: "Show minimap" }).click();
    await expect(page.getByTestId("graph-minimap")).toBeVisible();
    await expect(page.getByRole("button", { name: "Export GraphML" })).toBeVisible();
  });

  test("exports the focused subgraph as JSON and GraphML", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByPlaceholder("Entity name or description").fill("judo");
    const [json] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export subgraph" }).click(),
    ]);
    expect(json.suggestedFilename()).toMatch(/-subgraph\.json$/);
    const subgraph = JSON.parse(await readFile(await json.path(), "utf8")) as { nodes: unknown[]; edges: unknown[] };
    expect(subgraph.nodes.length).toBeGreaterThan(0);
    expect(subgraph.nodes.length).toBeLessThan(20);
    const [graphml] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export GraphML" }).click(),
    ]);
    expect(graphml.suggestedFilename()).toMatch(/\.graphml$/);
    const xml = await readFile(await graphml.path(), "utf8");
    expect(xml).toContain("<graphml");
    expect(xml.match(/<node /g)?.length).toBe(subgraph.nodes.length);
    expect(xml.match(/<edge /g)?.length).toBe(subgraph.edges.length);
  });

  test("renames a graph from the workspace", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_rename_target");
    await page.getByRole("button", { name: "Rename", exact: true }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByLabel("Graph name").fill("Martial arts corpus");
    await page.getByRole("button", { name: "Save name" }).click();
    await expect(page.getByRole("heading", { name: "Martial arts corpus" })).toBeVisible();
  });

  test("filters entities, relations, communities, and consumption", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.locator("summary").filter({ hasText: "Filters" }).click();
    await page.getByLabel("Entity types").selectOption("PERSON");
    await expect(page.getByRole("cell", { name: "Jigoro Kano" }).first()).toBeVisible();
    await expect(page.getByRole("cell", { name: "Judo", exact: true })).toHaveCount(0);
    await page.getByRole("button", { name: "Clear filters" }).first().click();
    await expect(page.getByRole("cell", { name: "Judo", exact: true }).first()).toBeVisible();
    await page.getByRole("tab", { name: "Relations" }).click();
    await page.getByLabel("Entity types").selectOption([]);
    await expect(page.getByRole("cell", { name: "DEVELOPED_BY" }).first()).toBeVisible();
    await page.getByRole("tab", { name: "Communities" }).click();
    await expect(page.getByRole("cell", { name: "PERSON" }).first()).toBeVisible();
    // Community filter chips carry the community's title, not its id.
    const communityFacet = page.getByLabel("Communities", { exact: true });
    await expect(communityFacet.locator("option", { hasText: "PERSON" })).toHaveCount(1);
    await expect(page.getByRole("button", { name: /^Community community_/ })).toHaveCount(0);
    await page.getByRole("tab", { name: "Consumption" }).click();
    await expect(page.getByText(/usd/i).first()).toBeVisible();
  });

  test("searches the sidebar catalog across datasets", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Search graphs").fill("Deep learning");
    await expect(page.getByRole("button", { name: /Deep learning papers Ready to/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Martial arts history Ready to/ })).toHaveCount(0);
  });

  test("explains a catalog search miss instead of looking empty", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Search graphs").fill("xyzzy-no-such-graph");
    await expect(page.getByTestId("catalog-empty")).toContainText("xyzzy-no-such-graph");
  });

  test("forgets a completed graph from history", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_forget_me");
    await expect(page.getByRole("heading", { name: "Forgettable graph" })).toBeVisible();
    await page.getByRole("button", { name: "Forget Forgettable graph" }).click();
    await page.getByRole("button", { name: "Confirm forget Forgettable graph" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Forget Forgettable graph" })).toHaveCount(0);
  });
});

test.describe("deep learning papers graph", () => {
  test("opens a second dataset with different counts", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_deep_learning");
    await expect(page.getByRole("heading", { name: "Deep learning papers" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Entities" })).toBeVisible();
    await page.getByPlaceholder("Entity name or description").fill("ImageNet");
    await expect(page.getByRole("cell", { name: "ImageNet" }).first()).toBeVisible();
    const martial = await page.request.get("/api/health");
    expect(martial.ok()).toBeTruthy();
  });
});

test.describe("ingestion", () => {
  test("lists a local corpus and runs preflight", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("button", { name: "Your files", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveCount(0);
    await useSamplePack(page);
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("source-count")).toContainText("selectable object", { timeout: 20_000 });
    await page.getByRole("button", { name: "Preview configuration" }).click();
    await expect(page.getByText("Effective configuration")).toBeVisible();
    await page.getByRole("button", { name: "Run preflight" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Preflight passed" })).toBeVisible();
  });

  test("submits a fake worker and reaches a completed graph", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await page.getByLabel("Display name").fill("Smoke graph");
    await confirmEnvironmentIfNeeded(page);
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Smoke graph" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("succeeded", { exact: true }).first()).toBeVisible({
      timeout: 30_000,
    });
  });

  test("uploads a markdown file and lists it as a source object", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await page.locator('input[type="file"]').setInputFiles({
      name: "note.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Judo was developed by Jigoro Kano.\n"),
    });
    await expect(page.getByTestId("source-count")).toContainText("selectable object", { timeout: 20_000 });
  });

  test("shows Azure and S3 source fields from local capabilities", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await openMoreSources(page);
    await page.getByLabel("Source kind").click();
    await page.getByRole("option", { name: "Azure Blob" }).click();
    await expect(page.getByLabel("Account URL")).toBeVisible();
    await expect(page.getByText("Blobs are listed once an account URL and container are named.")).toBeVisible();
    await page.getByLabel("Source kind").click();
    await page.getByRole("option", { name: "S3-compatible bucket" }).click();
    await expect(page.getByLabel("Bucket")).toBeVisible();
    await expect(page.getByText("Objects are listed once a bucket is named.")).toBeVisible();
  });

  test("lists a bucket before Start and holds Start to what it lists", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await openMoreSources(page);
    await page.getByLabel("Source kind").click();
    await page.getByRole("option", { name: "S3-compatible bucket" }).click();
    const start = page.getByRole("button", { name: "Start", exact: true });
    await expect(start).toBeDisabled();
    await page.getByLabel("Bucket").fill("test-corpora");
    await page.getByLabel("Prefix").fill("martial_arts/");
    // The fake CLI answers `sources list` with two supported objects.
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
    await expect(page.getByTestId("source-count")).toContainText("41 KB");
    await expect(page.getByTestId("credit-envelope")).toBeVisible({ timeout: 20_000 });
    await expect(start).toBeEnabled();
    // A bucket that cannot be listed closes Start again and says why. The
    // tab is hidden while the first attempt fails, which parks the retry:
    // a parked listing is still pending, never "no objects" with Start open.
    await setVisibility(page, "hidden");
    await page.getByLabel("Bucket").fill("missing");
    await expect(page.getByText("Listing objects…")).toBeVisible({ timeout: 20_000 });
    await expect(start).toBeDisabled();
    await page.waitForTimeout(1_500);
    await expect(page.getByText("Listing objects…")).toBeVisible();
    await expect(start).toBeDisabled();
    await setVisibility(page, "visible");
    await expect(page.getByText(/NoSuchBucket/)).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText(/Listing failed/)).toBeVisible();
    await expect(start).toBeDisabled();
    // And clearing the bucket leaves nothing to list.
    await page.getByLabel("Bucket").fill("   ");
    await expect(page.getByText("Objects are listed once a bucket is named.")).toBeVisible();
    await expect(start).toBeDisabled();
  });
});

test.describe("unavailable and failed runs", () => {
  test("explains missing artifacts instead of opening the explorer", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_missing_artifacts");
    await expect(page.getByText("files are not available on this host")).toBeVisible();
  });

  test("shows a failed run error", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_failed_llm");
    await expect(page.getByTestId("status-sentence")).toContainText("LLM endpoint timed out");
    // The catalog row carries that error on one line; it must truncate in
    // its column rather than widen the list into a sideways scroll.
    const overflow = await page.evaluate(() => {
      const viewport = document.querySelector("aside [data-radix-scroll-area-viewport]");
      return viewport ? viewport.scrollWidth - viewport.clientWidth : -1;
    });
    expect(overflow).toBe(0);
    await expect(page.getByTestId("guide-card")).toContainText("This run did not finish");
    await expect(page.getByTestId("guide-card")).toContainText("Clone the config");
    await expect(page.getByRole("button", { name: "Forget this graph" })).toBeVisible();
    await page.getByRole("button", { name: "Forget this graph" }).click();
    await expect(page.getByRole("button", { name: "Confirm forget this graph" })).toBeVisible();
    await expect(page.getByRole("alert").filter({ hasText: "LLM endpoint timed out" })).toHaveCount(0);
    await page.getByRole("button", { name: "New graph from this config" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByLabel("Display name")).toHaveValue("LLM timeout");
    await expect(page.getByRole("button", { name: "Sample pack", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("LLM provider")).toContainText("Ollama");
    await expect(page.getByLabel("OCR provider")).toContainText("Built-in document text only");
  });
});

test.describe("active progress", () => {
  test("shows live OCR progress for an in-flight run", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_active_ocr");
    await expect(page.getByRole("heading", { name: "Active OCR run" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Progress" })).toBeVisible();
    await expect(page.getByText(/ocr/i).first()).toBeVisible();
  });
});

test.describe("kubernetes fleet", () => {
  test("shows stub nodes and cluster catalog", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=fleet");
    await expect(page.getByRole("heading", { name: "Compute fleet" })).toBeVisible();
    // The dashboards are one click away when the deployment names them.
    await expect(page.getByRole("link", { name: "Open Grafana" })).toHaveAttribute("href", "https://grafana.example.test");
    await expect(page.getByRole("heading", { name: "gpu-a" })).toBeVisible();
    await expect(page.getByText("Fleet martial arts")).toBeVisible();
    await page.getByRole("button", { name: "Clusters" }).click();
    await expect(page.getByRole("heading", { name: "Registered clusters" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "lab", exact: true })).toBeVisible();
  });

  test("registers another cluster from the catalog", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=clusters");
    await page.getByRole("textbox", { name: "Cluster name", exact: true }).fill("lab-west");
    await page.getByRole("button", { name: "Save cluster" }).click();
    await expect(page.getByRole("cell", { name: /lab-west/ })).toBeVisible();
  });

  test("submits a stub fleet job", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    await useSamplePack(page);
    await page.getByLabel("Display name").fill("Fleet smoke");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Fleet smoke" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
  });

  test("clones a bucket source back into compose with every field", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    await openMoreSources(page);
    await page.getByLabel("Source kind").click();
    await page.getByRole("option", { name: "S3-compatible bucket" }).click();
    await page.getByLabel("Bucket", { exact: true }).fill("test-corpora");
    await page.getByLabel("Prefix", { exact: true }).fill("martial_arts/");
    await page.getByLabel("Endpoint", { exact: true }).fill("http://minio.local:9000");
    await page.getByLabel("Region", { exact: true }).fill("us-east-1");
    await page.getByLabel("Display name").fill("Bucket clone");
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Bucket clone" })).toBeVisible({ timeout: 30_000 });
    // Back on compose, "Clone last config" restores the bucket as it was named,
    // not just its kind.
    await page.goto("/?runtime=kubernetes&page=new");
    await page.getByRole("button", { name: "Clone last config" }).click();
    await expect(page.getByLabel("Bucket", { exact: true })).toHaveValue("test-corpora");
    await expect(page.getByLabel("Prefix", { exact: true })).toHaveValue("martial_arts/");
    await expect(page.getByLabel("Endpoint", { exact: true })).toHaveValue("http://minio.local:9000");
    await expect(page.getByLabel("Region", { exact: true })).toHaveValue("us-east-1");
    await expect(page.getByLabel("Display name")).toHaveValue("Bucket clone");
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
  });

  test("builds a new version of a finished fleet graph from its Edit tab", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible();
    await page.getByRole("tab", { name: "Edit" }).click();
    const editor = page.getByTestId("graph-editor");
    await expect(editor.getByText("karate-history.md")).toBeVisible();
    const confirm = page.getByRole("button", { name: "Confirm environment" });
    if (await confirm.isVisible()) {
      await confirm.click();
    }
    // Nothing removed and nothing added: there is no version to build yet.
    await expect(page.getByTestId("revision-summary")).toContainText("Keeps 2 documents");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeDisabled();
    // Removing one is enough on its own.
    await page.getByLabel("Remove karate-history.md").check();
    await expect(page.getByTestId("removal-summary")).toContainText("1 of 2 documents will be left out");
    await expect(page.getByTestId("revision-summary")).toContainText("removes 1");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeEnabled();
    // Adding a file is counted too, and the form is the compose form: no name, no ontology.
    await page.locator('input[type="file"]').setInputFiles({
      name: "aikido.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Aikido was developed by Morihei Ueshiba.\n"),
    });
    await expect(page.getByTestId("revision-summary")).toContainText("adds 1", { timeout: 20_000 });
    await expect(page.getByLabel("Display name")).toHaveCount(0);
    await page.getByRole("button", { name: "Build new version" }).click();
    await expect(page.getByText(/Building a new version of Fleet judo/)).toBeVisible();
    // The new run keeps the graph's name and identity and is queued on the fleet.
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible({ timeout: 30_000 });
    await expect(page).not.toHaveURL(/run=run_k8s_done/);
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("main")).toContainText("graph_k8s_done");
  });

  test("removing a document alone is enough to build a new version, and versions are listed", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await expect(page.getByRole("heading", { name: "Fleet judo" })).toBeVisible();
    await page.getByRole("tab", { name: "Versions" }).click();
    // The fleet keeps the versions; the stub has published none, and the
    // workspace's own version labels do not appear on the fleet.
    await expect(page.getByTestId("graph-versions")).toContainText("No version has been published");
    await expect(page.getByRole("button", { name: "Publish new version" })).toHaveCount(0);
    await expect(page.getByText("Watch new files")).toHaveCount(0);
    await page.getByRole("tab", { name: "Edit" }).click();
    await page.getByLabel("Remove judo-history.md").check();
    const confirm = page.getByRole("button", { name: "Confirm environment" });
    if (await confirm.isVisible()) {
      await confirm.click();
    }
    await expect(page.getByTestId("revision-summary")).toContainText("Keeps 1 document · removes 1");
    await page.getByRole("button", { name: "Build new version" }).click();
    await expect(page.getByText(/Building a new version of Fleet judo/)).toBeVisible();
    await expect(page).not.toHaveURL(/run=run_k8s_done/);
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible({ timeout: 30_000 });
  });

  test("cancels an in-flight fleet run and retries a failed one", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_martial");
    await expect(page.getByRole("heading", { name: "Fleet martial arts" })).toBeVisible();
    await page.getByRole("button", { name: "Recover workers" }).click();
    await expect(page.getByText(/Reconciled infrastructure/i)).toBeVisible();
    await page.getByTestId("guide-card").getByRole("button", { name: "Cancel job" }).click();
    await expect(page.getByRole("main").getByText("cancelled", { exact: true }).first()).toBeVisible();
    // A cancelled run keeps what it finished and can be picked up again.
    await expect(page.getByTestId("guide-card")).toContainText("Resume this run where it stopped");
    await page.getByTestId("guide-card").getByRole("button", { name: "Resume run" }).click();
    await expect(page.getByText("Run resumed")).toBeVisible();
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_failed");
    await expect(page.getByTestId("status-sentence")).toContainText("Worker lease expired");
    await page.getByTestId("guide-card").getByRole("button", { name: "Retry failed documents" }).click();
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
  });

  test("switches from local to kubernetes without wiping compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByLabel("Display name").fill("Keep me");
    await page.getByLabel("Runtime").click();
    await page.getByRole("option", { name: "Kubernetes" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
    await expect(page.getByLabel("Display name")).toHaveValue("Keep me");
    await expect(page.getByTestId("identity-chip")).toContainText("Kubernetes");
    const ack = await page.evaluate(() => sessionStorage.getItem("flakegraph.runtime-ack"));
    expect(ack).not.toBe("kubernetes");
    await page.getByLabel("Runtime").click();
    await page.getByRole("option", { name: "Snowflake" }).click();
    await expect(page.getByLabel("Display name")).toHaveValue("Keep me");
    await expect(page.getByRole("button", { name: "Confirm environment" })).toBeVisible();
  });
});

test.describe("identity and catalog scope", () => {
  test("opens the identity dialog and explains forget versus delete", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await expect(page.getByRole("dialog", { name: "Session identity" })).toBeVisible();
    await expect(page.getByRole("dialog")).toContainText("Forget");
    await expect(page.getByRole("dialog")).toContainText("Delete graph");
    await expect(page.getByRole("dialog")).toContainText("This laptop catalog lists every local graph");
    await expect(page.getByTestId("identity-chip")).toContainText("laptop catalog");
    await expect(page.getByTestId("identity-chip")).not.toContainText("private graphs hidden");
  });

  test("Mine is empty while unidentified, All still lists graphs", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    const signOut = page.getByRole("button", { name: "Sign out" });
    if (await signOut.isVisible()) {
      await signOut.click();
    } else {
      await page.keyboard.press("Escape");
    }
    await page.getByRole("button", { name: "Mine", exact: true }).click();
    await expect(page.getByRole("button", { name: "Mine", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("catalog-empty")).toContainText("0 yours");
    await page.getByRole("button", { name: "All", exact: true }).click();
    await expect(page.getByText("Martial arts history")).toBeVisible();
  });

  test("Mine lists ALICE graphs and hides unowned catalog rows", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign in as ALICE" }).click();
    await page.getByRole("button", { name: "Mine", exact: true }).click();
    await expect(page.getByRole("button", { name: "Mine", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText("Martial arts history")).toBeVisible();
    await expect(page.getByText("Copied host graph")).toHaveCount(0);
    await page.getByRole("button", { name: "All", exact: true }).click();
    await expect(page.getByText("Copied host graph")).toBeVisible();
  });
});

test.describe("snowflake sharing", () => {
  test("hides share controls on local runtime and shows them on snowflake", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("tab", { name: "Sharing" })).toHaveCount(0);
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_martial");
    await expect(page.getByRole("heading", { name: "Shared martial arts" })).toBeVisible();
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByText("ANALYST")).toBeVisible();
    await page.getByLabel("Grantee", { exact: true }).fill("CAROL");
    await expect(page.getByTestId("share-preview")).toContainText("USER CAROL will see");
  });

  test("hides a privately owned graph from an unidentified viewer", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=new");
    await page.getByTestId("identity-chip").click();
    const signOut = page.getByRole("button", { name: "Sign out" });
    if (await signOut.isVisible()) {
      await signOut.click();
    } else {
      await page.keyboard.press("Escape");
    }
    await expect(page.getByTestId("identity-chip")).toContainText("private graphs hidden");
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_private");
    await expect(page.getByText(/not visible/i)).toBeVisible();
  });

  test("shares, unshares, and deletes a stored graph", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_martial");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await page.getByLabel("Grantee", { exact: true }).fill("CAROL");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await expect(page.getByText("USER CAROL")).toBeVisible();
    await page.getByRole("listitem").filter({ hasText: "USER CAROL" }).getByRole("button", { name: "Unshare" }).click();
    await expect(page.getByRole("listitem").filter({ hasText: "USER CAROL" })).toHaveCount(0);
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_delete");
    await expect(page.getByRole("heading", { name: "Disposable snowflake graph" })).toBeVisible();
    await page.getByRole("button", { name: "Delete graph" }).click();
    await page.getByRole("button", { name: "Confirm delete" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  });

  test("submits a stub snowflake job from an uploaded document", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=new");
    await confirmEnvironmentIfNeeded(page);
    await expect(page.getByLabel("Stage", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Sample pack", exact: true })).toHaveCount(0);
    await expect(page.getByLabel("LLM model")).toHaveValue("unsloth/Qwen3.8-27B-NVFP4");
    await page.getByLabel("Display name").fill("Snowflake smoke");
    await page.getByLabel("Source kind").click();
    await page.getByRole("option", { name: "Upload" }).click();
    await page.locator('input[type="file"]').setInputFiles({
      name: "note.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("Judo was developed by Jigoro Kano.\n"),
    });
    await expect(page.getByTestId("source-count")).toContainText("1 selectable");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Snowflake smoke" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("main").getByText("queued", { exact: false }).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Recover workers" })).toHaveCount(0);
  });

  test("disables Recover workers on a queued Snowflake job", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_queued");
    await expect(page.getByRole("heading", { name: "Queued Snowflake job" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Recover workers" })).toHaveCount(0);
    await expect(page.getByTestId("guide-card")).toContainText("has not started");
  });

  test("blocks share when quality is red", async ({ page }) => {
    await page.goto("/?runtime=snowflake&page=run&run=run_snow_thin");
    await expect(page.getByRole("heading", { name: "Empty share candidate" })).toBeVisible();
    await expect(page.getByTestId("guide-card")).toContainText("entity count is 0");
    await page.getByRole("tab", { name: "Sharing" }).click();
    await expect(page.getByText(/not ready to share/i)).toBeVisible();
    await expect(page.getByRole("button", { name: "Share", exact: true })).toHaveCount(0);
  });
});

test.describe("remaining report journeys", () => {
  test("serves inspect HTML for the martial arts graph", async ({ request }) => {
    const response = await request.get("/api/inspect?run=run_martial_arts");
    expect(response.ok()).toBeTruthy();
    expect(await response.text()).toContain("CLI inspect artifact");
  });

  test("asks a local question and opens the Judo perspective", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Ask" }).click();
    await page.getByLabel("Ask the graph").fill("Who developed judo?");
    await page.getByRole("button", { name: "Ask", exact: true }).click();
    await expect(page.getByTestId("ask-answer")).toBeVisible({ timeout: 120_000 });
    await page.getByRole("tab", { name: "Explore" }).click();
    await expect(page.getByRole("button", { name: /Judo/ }).first()).toBeVisible();
    await page.getByRole("button", { name: /Judo/ }).first().click();
    await expect(page.getByPlaceholder("Entity name or description")).toHaveValue("judo");
  });

  test("saves a named perspective and reopens it", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    const save = page.getByRole("button", { name: "Save as perspective" });
    // Nothing to save until the perspective is named.
    await expect(save).toBeDisabled();
    await page.getByPlaceholder("Entity name or description").fill("judo");
    await page.getByLabel("Perspective name").fill("Judo lens");
    await save.click();
    await expect(page.getByText("Perspective saved as draft", { exact: false })).toBeVisible();
    await expect(page.getByLabel("Perspective name")).toHaveValue("");
    await page.getByPlaceholder("Entity name or description").fill("");
    await page.getByRole("button", { name: "Judo lens · draft" }).click();
    await expect(page.getByPlaceholder("Entity name or description")).toHaveValue("judo");
  });

  test("shows a credit envelope on compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await expect(page.getByTestId("credit-envelope")).toBeVisible();
    await expect(page.getByRole("button", { name: "Scan for PII" })).toBeVisible();
    await expect(page.getByText("Describe the graph you want")).toBeVisible();
  });

  test("selects a sample pack and required providers on compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Your files", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Sample pack" })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "How to process" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Fast / cheap" })).toHaveCount(0);
    await expect(page.getByTestId("compose-providers")).toBeVisible();
    await expect(page.getByLabel("OCR provider")).toContainText("Adaptive layout");
    await expect(page.getByLabel("LLM provider")).toContainText("vLLM");
    await expect(page.getByLabel("LLM model")).toHaveValue("unsloth/Qwen3.8-27B-NVFP4");
    await expect(page.getByLabel("Embeddings model")).toHaveValue("sentence-transformers/all-MiniLM-L6-v2");
    await useSamplePack(page);
    await expect(page.getByRole("button", { name: "Sample pack", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toHaveCount(0);
    await expect(page.getByLabel("OCR provider")).toContainText("Adaptive layout");
    await expect(page.getByLabel("LLM model")).toHaveValue("unsloth/Qwen3.8-27B-NVFP4");
    await page.getByLabel("LLM provider").click();
    await page.getByRole("option", { name: "Ollama" }).click();
    await expect(page.getByLabel("LLM model")).toHaveValue("llama3.2");
    // Only packs whose files are on this host are offered: the papers have
    // to be downloaded first, so they are not.
    await expect(page.getByRole("button", { name: "Deep learning papers", exact: true })).toHaveCount(0);
    await expect(page.getByLabel("Display name")).toHaveValue("Martial arts history");
    await expect(page.getByLabel("LLM model")).toHaveValue("llama3.2");
    await page.getByRole("button", { name: "Your files", exact: true }).click();
    await expect(page.getByRole("button", { name: "Your files", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveCount(0);
  });

  test("logo returns home from a graph workspace", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await page.getByRole("button", { name: "FlakeGraph home" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
  });

  test("opens the sidebar on a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Open navigation" })).toBeVisible();
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toBeVisible();
  });

  test("skips a poison document on an active run", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_active_ocr");
    await expect(page.getByText(/doc_poison_scan · OCR failed/i)).toBeVisible();
    await page.getByRole("button", { name: "Skip file" }).click();
    await expect(page.getByText(/doc_poison_scan · Skipped/i)).toBeVisible();
  });

  test("analyst chrome hides New graph after CAROL signs in", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign in as CAROL" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Perspectives" })).toBeVisible();
    await page.getByRole("button", { name: "Open Judo" }).click();
    await expect(page.getByRole("heading", { name: "Martial arts history" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Rename", exact: true })).toHaveCount(0);
    await page.getByRole("tab", { name: "Versions" }).click();
    await expect(page.getByText("Watch new files")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Publish new version" })).toHaveCount(0);
    await expect(page.getByLabel("Runtime")).toHaveCount(0);
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page.getByRole("button", { name: "New graph" })).toBeVisible();
  });

  test("opens inspect HTML with a gold compare section", async ({ request }) => {
    const response = await request.get("/api/inspect?run=run_martial_arts");
    const body = await response.text();
    expect(body).toContain("Gold compare");
  });

  test("reviews an uncertain triple", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Review" }).click();
    await expect(page.getByText(/never rewrite the stored graph/i)).toBeVisible();
    await expect(page.getByText(/does not apply these decisions/i)).toBeVisible();
    await expect(page.getByText("rel_judo_developed_by")).toBeVisible();
    await page.getByRole("button", { name: "Keep", exact: true }).click();
    await expect(page.getByRole("button", { name: "Keep", exact: true })).toBeVisible();
    await expect(page.getByText(/Recorded keep/i)).toBeVisible();
  });

  test("ingests new files from a watch", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Versions" }).click();
    await expect(page.getByText("data/martial_arts/files")).toBeVisible();
    await page.getByRole("button", { name: "Ingest new files" }).click();
    await expect(page.getByText(/incremental/i).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Process on compose" })).toBeVisible();
  });

  test("applies promotion remaps onto compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Versions" }).click();
    await expect(page.getByTestId("promotion-remaps")).toBeVisible();
    await page.getByRole("button", { name: "Apply remaps" }).click();
    await expect(page.getByText("Promotion remaps are on the form")).toBeVisible();
  });

  test("opens the command palette with Cmd+K", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await expect(page.getByRole("button", { name: "Jump" })).toBeVisible();
    await page.getByRole("button", { name: "Jump" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Jump" })).toBeVisible();
  });

  test("Jump entity search does not stick after picking another graph", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("button", { name: "Jump" }).click();
    await page.getByLabel("Command palette").fill("Judo");
    await page.getByRole("option", { name: /Entity Judo/ }).click();
    await expect(page.getByLabel("Search entities")).toHaveValue("Judo");
    await page.getByRole("button", { name: /Deep learning papers Ready to/ }).click();
    await expect(page.getByRole("heading", { name: "Deep learning papers" })).toBeVisible();
    await expect(page.getByLabel("Search entities")).toHaveValue("");
  });

  test("staff can build an identity incident note", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Staff" }).click();
    await page.getByRole("button", { name: "Incident" }).click();
    await expect(page.getByRole("heading", { name: "Staff incident" })).toBeVisible();
    await page.getByRole("button", { name: "Build incident note" }).click();
    await expect(page.getByText("Identity mismatch incident")).toBeVisible();
  });

  test("creates an SDK key that does not use SSO", async ({ page }) => {
    await page.goto("/?runtime=local&page=keys");
    await expect(page.getByText(/Machine credentials/i)).toBeVisible();
    await expect(page.getByRole("heading", { name: "Where the API docs are" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "How to use a key" })).toBeVisible();
    await expect(page.getByRole("link", { name: "tRPC HTTP" })).toHaveAttribute("href", "https://trpc.io/docs/rpc");
    await expect(page.getByTestId("sdk-docs")).toContainText("/api/docs");
    await expect(page.getByTestId("sdk-procedure-catalog")).toContainText("graphs.ask");
    await expect(page.getByRole("tab", { name: "List graphs" })).toHaveAttribute("data-state", "active");
    await expect(page.getByRole("tab", { name: "Python" })).toHaveCount(0);
    await expect(page.getByTestId("sdk-example-language")).toBeVisible();
    await expect(page.getByTestId("sdk-key-example")).toContainText("Authorization: Bearer");
    await expect(page.getByTestId("sdk-key-example")).toContainText("/api/trpc/runs.list");
    await expect(page.getByTestId("sdk-key-example")).not.toContainText("CONTROL_PLANE");
    await expect(page.getByTestId("sdk-key-example")).toContainText("FLAKEGRAPH_API_KEY");
    await page.getByRole("button", { name: "TypeScript", exact: true }).click();
    await expect(page.getByTestId("sdk-key-example")).toContainText("fetch(");
    await page.getByRole("button", { name: "Python", exact: true }).click();
    await expect(page.getByTestId("sdk-key-example")).toContainText("urllib.request");
    await page.getByRole("tab", { name: "Ask a graph" }).click();
    await expect(page.getByTestId("sdk-ask-example")).toContainText("graphs.ask");
    await expect(page.getByTestId("sdk-ask-example")).toContainText("urllib.request");
    await page.getByRole("button", { name: "curl", exact: true }).click();
    await expect(page.getByTestId("sdk-ask-example")).toContainText("/api/trpc/graphs.ask");
    const docs = await page.request.get("/api/docs");
    expect(docs.ok()).toBeTruthy();
    const catalog = await docs.json();
    expect(catalog.endpoint).toContain("/api/trpc");
    expect(catalog.procedures.some((item: { name: string }) => item.name === "runs.list")).toBeTruthy();
    await page.getByRole("button", { name: "Create a key" }).click();
    await expect(page.getByText(/Copy now:/)).toBeVisible();
    const secretLine = await page.getByText(/Copy now:/).innerText();
    const secret = secretLine.replace(/^.*Copy now:\s*/, "").trim();
    expect(secret).toMatch(/^fg_/);
    const listed = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: `Bearer ${secret}` } },
    );
    expect(listed.ok()).toBeTruthy();
    const denied = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: "Bearer fg_revoked_or_wrong" } },
    );
    expect(denied.status()).toBe(401);
    await expect(page.getByRole("button", { name: "Revoke" })).toBeVisible();
    await page.getByRole("button", { name: "Revoke" }).click();
    await page.getByRole("button", { name: "Confirm revoke" }).click();
    await expect(page.getByText("No machine keys yet")).toBeVisible();
    const afterRevoke = await page.request.get(
      `/api/trpc/runs.list?input=${encodeURIComponent(JSON.stringify({ json: { limit: 5 } }))}`,
      { headers: { Authorization: `Bearer ${secret}` } },
    );
    expect(afterRevoke.status()).toBe(401);
  });

  test("forgets selected catalog rows in bulk", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    const checkbox = page.locator('input[type="checkbox"][aria-label^="Select "]').first();
    await expect(checkbox).toBeVisible();
    const label = await checkbox.getAttribute("aria-label");
    expect(label).toBeTruthy();
    await checkbox.check();
    await page.getByRole("button", { name: "Forget selected" }).click();
    await page.getByRole("button", { name: "Confirm forget" }).click();
    await expect(page.getByLabel(label!)).toHaveCount(0);
  });

  test("shows a cancelling leftover-lease sentence", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_cancelling");
    await expect(page.getByTestId("status-sentence")).toContainText("Cancelling");
    await expect(page.getByTestId("status-sentence")).toContainText("leftover leases");
  });

  test("scans a sample pack for PII before embed", async ({ page, request }) => {
    // The seed writes the pack under the server's state root, wherever that is.
    const session = await request.get(
      "/api/trpc/auth.session?batch=1&input=%7B%220%22%3A%7B%22json%22%3Anull%7D%7D",
      { headers: { "x-flakegraph-runtime": "local" } },
    );
    const [{ result }] = (await session.json()) as [{ result: { data: { json: { stateRoot: string } } } }];
    await page.goto("/?runtime=local&page=new");
    await useFolderPath(page, `${result.data.json.stateRoot}/pii-pack`);
    await page.getByRole("button", { name: "Scan for PII" }).click();
    await expect(page.getByText(/PII email/i)).toBeVisible();
    // Acknowledging keeps the finding on screen rather than making it vanish.
    await page.getByRole("button", { name: "Acknowledge residual risk" }).click();
    await expect(page.getByTestId("pii-acknowledged")).toContainText(/PII email .* residual risk acknowledged/);
    await expect(page.getByRole("button", { name: "Acknowledge residual risk" })).toHaveCount(0);
  });

  test("compares estimate vs actual on consumption", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Consumption" }).click();
    await expect(page.getByTestId("estimate-vs-actual")).toBeVisible();
  });

  test("proposes an ontology with gold coverage", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByRole("button", { name: "Propose ontology" }).click();
    await expect(page.getByTestId("ontology-coverage")).toBeVisible();
  });

  test("shows a 1-hop neighborhood from a selected relation", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Relations" }).click();
    await page.getByRole("cell", { name: "rel_001" }).click();
    const selection = page.getByTestId("explore-selection");
    await expect(selection.getByRole("button", { name: "Looks wrong" })).toBeVisible();
    await selection.getByRole("button", { name: "Show neighborhood" }).click();
    await expect(page.getByTestId("explore-scope")).toContainText("in the selected neighborhood");
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
  });

  test("welcome back ignores the seed OCR fixture after ALICE signs in", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await page.getByTestId("identity-chip").click();
    await page.getByRole("button", { name: "Sign in as ALICE" }).click();
    await expect(page.getByTestId("welcome-back")).toBeVisible();
    await expect(page.getByTestId("welcome-back")).not.toContainText("Active OCR");
  });
});

test.describe("health", () => {
  test("serves the kubernetes health route", async ({ request }) => {
    const response = await request.get("/api/health");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.service).toBe("flakegraph-control-plane");
  });
});
