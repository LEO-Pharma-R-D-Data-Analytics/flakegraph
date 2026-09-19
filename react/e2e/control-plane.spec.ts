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

/** Pick where the documents come from; the choices are always on screen. */
async function chooseSource(page: Page, name: string) {
  const tile = page.getByRole("group", { name: "Document source" }).getByRole("button", { name, exact: true });
  await expect(tile).toBeVisible();
  if ((await tile.getAttribute("aria-pressed")) !== "true") {
    await tile.click();
  }
}

async function useSamplePack(page: Page, name: "Martial arts" | "Deep learning papers" = "Martial arts") {
  await chooseSource(page, "Sample pack");
  const tile = page.getByRole("button", { name, exact: true });
  await expect(tile).toBeVisible();
  if ((await tile.getAttribute("aria-pressed")) !== "true") {
    await tile.click();
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
  await chooseSource(page, "Folder path");
  await page.getByLabel("Folder path").fill(path);
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
    await expect(page.getByRole("group", { name: "Document source" }).getByRole("button", { name: "Upload" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
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
    await chooseSource(page, "Azure Blob");
    await expect(page.getByLabel("Account URL")).toBeVisible();
    await expect(page.getByText("Blobs are listed once an account URL and container are named.")).toBeVisible();
    await chooseSource(page, "S3-compatible bucket");
    await expect(page.getByLabel("Bucket")).toBeVisible();
    await expect(page.getByText("Objects are listed once a bucket is named.")).toBeVisible();
  });

  test("lists a bucket before Start and holds Start to what it lists", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await chooseSource(page, "S3-compatible bucket");
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
  test("shows stub nodes", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=fleet");
    await expect(page.getByRole("heading", { name: "Compute fleet" })).toBeVisible();
    // The dashboards are one click away when the deployment names them.
    await expect(page.getByRole("link", { name: "Open Grafana" })).toHaveAttribute("href", "https://grafana.example.test");
    await expect(page.getByRole("heading", { name: "gpu-a" })).toBeVisible();
    await expect(page.getByText("Fleet martial arts")).toBeVisible();
    // The console manages the fleet it runs in; there is no cluster to pick.
    await expect(page.getByRole("button", { name: "Clusters" })).toHaveCount(0);
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
    await chooseSource(page, "S3-compatible bucket");
    await page.getByLabel("Bucket", { exact: true }).fill("test-corpora");
    await page.getByLabel("Prefix", { exact: true }).fill("martial_arts/");
    await page.getByLabel("Endpoint", { exact: true }).fill("http://minio.local:9000");
    await page.getByLabel("Region", { exact: true }).fill("us-east-1");
    await page.getByLabel("Display name").fill("Bucket clone");
    await expect(page.getByTestId("source-count")).toContainText("2 selectable objects", { timeout: 20_000 });
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Bucket clone" })).toBeVisible({ timeout: 30_000 });
    // A run page hands its configuration back to compose with the bucket as
    // it was named, not just its kind.
    await page.getByTestId("guide-card").getByRole("button", { name: "Cancel job" }).click();
    await expect(page.getByRole("main").getByText("cancelled", { exact: true }).first()).toBeVisible();
    await page.getByTestId("guide-card").getByRole("button", { name: "New graph from this config" }).click();
    await expect(page.getByRole("heading", { name: "Build a graph" })).toBeVisible();
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
    // The whole set can be left out at once and kept again; then one is enough on its own.
    await editor.getByRole("button", { name: /Remove all/ }).click();
    await expect(page.getByTestId("removal-summary")).toContainText("2 of 2 documents will be left out");
    await editor.getByRole("button", { name: "Keep all" }).click();
    await expect(page.getByTestId("removal-summary")).toHaveCount(0);
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
    // The fleet keeps the versions: two runs published this graph, the later
    // one is the head, and the one on screen says so. The workspace's own
    // version labels do not appear on the fleet.
    const versions = page.getByTestId("graph-versions");
    await expect(versions).toContainText("v2");
    await expect(versions).toContainText("head");
    await expect(versions.getByRole("row").filter({ hasText: "run_k8s_done" }).filter({ hasText: "viewing" })).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Publish new version" })).toHaveCount(0);
    await expect(page.getByText("Watch new files")).toHaveCount(0);
    // Opening the head from the list lands on that run; opening the older
    // one again shows it is behind the head before it is edited.
    await versions.getByRole("row").filter({ hasText: "run_k8s_done_v2" }).getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/run=run_k8s_done_v2/);
    await page.getByRole("tab", { name: "Edit" }).click();
    await expect(page.getByTestId("editing-behind-head")).toHaveCount(0);
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("tab", { name: "Edit" }).click();
    await expect(page.getByTestId("editing-behind-head")).toContainText("version 1 of 2, not the head");
    await page.getByLabel("Remove judo-history.md").check();
    const confirm = page.getByRole("button", { name: "Confirm environment" });
    if (await confirm.isVisible()) {
      await confirm.click();
    }
    await expect(page.getByTestId("revision-summary")).toContainText("Keeps 1 document · removes 1");
    // Removing the last one too, with nothing added, is refused: a version
    // needs at least one document.
    await page.getByLabel("Remove karate-history.md").check();
    await expect(page.getByTestId("revision-summary")).toContainText("a version needs at least one document");
    await expect(page.getByRole("button", { name: "Build new version" })).toBeDisabled();
    await page.getByLabel("Remove karate-history.md").uncheck();
    await expect(page.getByRole("button", { name: "Build new version" })).toBeEnabled();
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
    await chooseSource(page, "Upload");
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
    await expect(page.getByLabel("Describe the graph you want")).toBeVisible();
  });

  test("selects a sample pack and required providers on compose", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    const sourceGroup = page.getByRole("group", { name: "Document source" });
    await expect(sourceGroup.getByRole("button", { name: "Upload" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    // Every source is one click away; nothing is folded behind a disclosure.
    for (const name of ["Upload", "Sample pack", "Folder path", "Azure Blob", "S3-compatible bucket"]) {
      await expect(sourceGroup.getByRole("button", { name, exact: true })).toBeVisible();
    }
    await expect(page.locator("summary")).toHaveCount(0);
    await expect(page.getByText("Run like last time")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Fast / cheap" })).toHaveCount(0);
    await expect(page.getByTestId("compose-providers")).toBeVisible();
    await expect(page.getByLabel("OCR provider")).toContainText("Adaptive layout");
    await expect(page.getByLabel("LLM provider")).toContainText("vLLM");
    await expect(page.getByLabel("LLM model")).toHaveValue("unsloth/Qwen3.8-27B-NVFP4");
    await expect(page.getByLabel("Embeddings model")).toHaveValue("Qwen/Qwen3-Embedding-0.6B");
    await expect(page.getByLabel("Embeddings dimension")).toHaveValue("1024");
    await useSamplePack(page);
    await expect(sourceGroup.getByRole("button", { name: "Sample pack", exact: true })).toHaveAttribute("aria-pressed", "true");
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
    await chooseSource(page, "Upload");
    await expect(page.getByTestId("file-dropzone")).toBeVisible();
    await expect(page.getByRole("button", { name: "Martial arts", exact: true })).toHaveCount(0);
  });

  test("chooses what to extract: types, relations, reset, and the run carries them", async ({ page }) => {
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await confirmEnvironmentIfNeeded(page);
    const editor = page.getByTestId("ontology-editor");
    const entityTypes = editor.getByTestId("entity-types");
    // The default profile arrives as editable chips.
    await expect(entityTypes.getByRole("button", { name: "PERSON", exact: true })).toBeVisible();
    await expect(editor).toContainText("These are the types from the default profile.");
    // Add one (normalised to a type name), remove one.
    await entityTypes.getByLabel("Add entity type").fill("martial technique");
    await entityTypes.getByLabel("Add entity type").press("Enter");
    await expect(entityTypes.getByRole("button", { name: "MARTIAL_TECHNIQUE", exact: true })).toBeVisible();
    await entityTypes.getByRole("button", { name: "Remove DATE" }).click();
    await expect(entityTypes.getByRole("button", { name: "DATE", exact: true })).toHaveCount(0);
    // Describe a type.
    await entityTypes.getByRole("button", { name: "MARTIAL_TECHNIQUE", exact: true }).click();
    await entityTypes.getByLabel("Description of MARTIAL_TECHNIQUE").fill("A named throw, strike, or hold.");
    await entityTypes.getByRole("button", { name: "Save" }).click();
    await expect(editor).toContainText("Changed from the default profile.");
    // Relations can be detected automatically, in which case no list is asked for.
    await editor.getByRole("radio", { name: /Detected automatically/ }).click();
    await expect(editor.getByTestId("relation-types")).toHaveCount(0);
    // A fixed list with nothing in it cannot be built with.
    await editor.getByRole("radio", { name: /Only this list/ }).click();
    const relationTypes = editor.getByTestId("relation-types");
    for (const name of ["RELATED_TO", "PART_OF", "LOCATED_IN", "CREATED_BY", "INFLUENCED_BY", "OCCURRED_AT"]) {
      await relationTypes.getByRole("button", { name: `Remove ${name}` }).click();
    }
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeDisabled();
    await expect(editor).toContainText("Add at least one relation type");
    await relationTypes.getByLabel("Add relation type").fill("TRAINED_UNDER");
    await relationTypes.getByLabel("Add relation type").press("Enter");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    // The run's configuration carries exactly this vocabulary, inline.
    await page.getByRole("button", { name: "Preview configuration" }).click();
    const yaml = page.getByLabel("YAML configuration text");
    await expect(yaml).toContainText("MARTIAL_TECHNIQUE");
    await expect(yaml).toContainText("A named throw, strike, or hold.");
    await expect(yaml).toContainText("mode: closed");
    await expect(yaml).toContainText("TRAINED_UNDER");
    await expect(yaml).not.toContainText("name: DATE");
    // Reset returns to the defaults.
    await editor.getByRole("button", { name: "Reset to the default profile" }).click();
    await expect(entityTypes.getByRole("button", { name: "DATE", exact: true })).toBeVisible();
    await expect(editor).toContainText("These are the types from the default profile.");
  });

  test("a revision keeps the vocabulary of the version it revises", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=run&run=run_k8s_done");
    await page.getByRole("tab", { name: "Edit" }).click();
    const editor = page.getByTestId("ontology-editor");
    await expect(editor).toContainText("built with the vocabulary of the version it revises");
    await expect(editor.getByTestId("entity-types")).toContainText("SCHOOL");
    await expect(editor.getByTestId("relation-types")).toContainText("FOUNDED_BY");
    await expect(editor.getByLabel("Add entity type")).toHaveCount(0);
    await expect(editor.getByRole("button", { name: "Remove SCHOOL" })).toHaveCount(0);
  });

  test("offers Snowflake as a destination only where an account can write", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    await confirmEnvironmentIfNeeded(page);
    // The stub fleet holds no Snowflake account: the option is there, closed, and says why.
    await expect(page.getByTestId("snowflake-unavailable")).toContainText("no Snowflake account");
    await page.getByLabel("Destination").click();
    await expect(page.getByRole("option", { name: "Snowflake" })).toHaveAttribute("data-disabled", "");
    await page.keyboard.press("Escape");
    // Locally the account is named on the form, and Start waits for the target.
    await page.goto("/?runtime=local&page=new");
    await useSamplePack(page);
    await confirmEnvironmentIfNeeded(page);
    await page.getByLabel("Destination").click();
    await page.getByRole("option", { name: "Snowflake" }).click();
    await expect(page.getByTestId("snowflake-destination")).toBeVisible();
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeDisabled();
    await expect(page.getByText("Name the Snowflake database, schema and bulk stage before Start.")).toBeVisible();
    await page.getByLabel("Account").fill("xy123");
    await page.getByLabel("User").fill("ALICE");
    await page.getByLabel("Database").fill("FG");
    await page.getByLabel("Schema").fill("PUBLIC");
    await page.getByLabel("Bulk Stage").fill("@stage");
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeEnabled();
    await page.getByRole("button", { name: "Preview configuration" }).click();
    await expect(page.getByLabel("YAML configuration text")).toContainText("provider: snowflake_bulk");
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
    // The document needing a decision is listed on the Progress card itself.
    const attention = page.getByTestId("documents-needing-attention");
    await expect(attention).toContainText("One document needs a decision");
    await expect(attention).toContainText("doc_poison_scan");
    await attention.getByRole("button", { name: "Skip file" }).click();
    // Skipped, it needs nothing more: the list goes, the summary counts it.
    await expect(page.getByTestId("documents-needing-attention")).toHaveCount(0);
    await expect(page.getByTestId("phase-summary")).toContainText("Skipped");
    await page.getByText(/^All \d+ documents$/).click();
    await expect(page.getByTestId("document-table")).toContainText("doc_poison_scan");
    await expect(page.getByTestId("document-table")).toContainText("Quarantined");
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
    const bar = page.getByTestId("catalog-selection");
    const checkbox = page.locator('input[type="checkbox"][aria-label^="Select "]:not([disabled]):not([aria-label="Select all shown graphs"])').first();
    await expect(checkbox).toBeVisible();
    const label = await checkbox.getAttribute("aria-label");
    expect(label).toBeTruthy();
    await checkbox.check();
    await expect(bar).toContainText("1 of");
    // The forget asks first, naming what goes, and Cancel keeps everything.
    await page.getByRole("button", { name: "Forget selected" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("Forget this graph?");
    await expect(dialog).toContainText(label!.replace(/^Select /, ""));
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByLabel(label!)).toBeChecked();
    await page.getByRole("button", { name: "Forget selected" }).click();
    await page.getByRole("button", { name: "Confirm forget" }).click();
    await expect(page.getByLabel(label!)).toHaveCount(0);
    await expect(bar).toContainText("Select all");
  });

  test("keeps a 300-document run readable while it runs", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_large_corpus");
    // The Progress card sums the corpus up and lists only what needs a decision.
    const summary = page.getByTestId("phase-summary");
    await expect(summary).toContainText("6");
    await expect(summary).toContainText("OCR failed");
    await expect(summary).toContainText("Indexed");
    const attention = page.getByTestId("documents-needing-attention");
    // paper-043 and paper-086 are here only because the whole events file is
    // read: the snapshot's tail of 200 events does not reach them.
    await expect(attention).toContainText("6 documents need a decision");
    await expect(attention).toContainText("paper-043.pdf");
    await expect(attention.getByRole("button", { name: "Skip file" })).toHaveCount(5);
    await expect(attention).toContainText("1 more in the list below");
    // Every document is one click away, a page at a time, searchable, filterable.
    await expect(page.getByTestId("document-table")).toBeHidden();
    await page.getByText("All 300 documents").click();
    await expect(page.getByTestId("document-table")).toBeVisible();
    const table = page.getByTestId("document-table");
    await expect(table.getByRole("row")).toHaveCount(51);
    await expect(table).toContainText("Showing 50 of 300");
    await table.getByRole("button", { name: /Show 50 more/ }).click();
    await expect(table.getByRole("row")).toHaveCount(101);
    await table.getByLabel("Search documents").fill("paper-29");
    await expect(page.getByTestId("document-table-count")).toContainText("10 of 300 documents");
    await table.getByLabel("Search documents").fill("");
    await table.getByLabel("Phase filter").click();
    await page.getByRole("option", { name: /Needs attention/ }).click();
    await expect(page.getByTestId("document-table-count")).toContainText("6 of 300 documents");
    await expect(table.getByRole("button", { name: "Skip file" })).toHaveCount(6);
  });

  test("selects every shown graph at once, leaving running ones out", async ({ page }) => {
    await page.goto("/?runtime=kubernetes&page=new");
    const bar = page.getByTestId("catalog-selection");
    const rowBoxes = 'input[type="checkbox"][aria-label^="Select "]:not([aria-label="Select all shown graphs"])';
    const rows = page.locator(rowBoxes);
    const forgettable = page.locator(`${rowBoxes}:not([disabled])`);
    const running = page.locator(`${rowBoxes}[disabled]`);
    await expect(forgettable.first()).toBeVisible();
    const shown = await forgettable.count();
    expect(await running.count()).toBeGreaterThan(0);
    await bar.getByLabel("Select all shown graphs").check();
    await expect(bar).toContainText(`${shown} of ${shown} selected`);
    for (const box of await forgettable.all()) {
      await expect(box).toBeChecked();
    }
    for (const box of await running.all()) {
      await expect(box).not.toBeChecked();
    }
    // A filter hides some of the selection without dropping it.
    await page.getByLabel("Search graphs").fill("judo");
    await expect(bar).toContainText("not shown");
    await page.getByLabel("Search graphs").fill("");
    // The dialog lists the whole selection; Clear puts it away.
    await page.getByRole("button", { name: "Forget selected" }).click();
    await expect(page.getByRole("dialog")).toContainText(`Forget ${shown} graphs?`);
    await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click();
    await bar.getByRole("button", { name: "Clear" }).click();
    await expect(bar).toContainText(`Select all ${shown}`);
    expect(await rows.count()).toBeGreaterThan(0);
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
    await page.getByLabel("Describe the graph you want").fill("people, schools, and techniques in these histories");
    await page.getByRole("button", { name: "Suggest types" }).click();
    const proposal = page.getByTestId("ontology-proposal");
    await expect(proposal).toBeVisible();
    // The review panel marks what the lists do not hold yet, and only that.
    const editor = page.getByTestId("ontology-editor");
    const entityTypes = editor.getByTestId("entity-types");
    await expect(proposal).toContainText("SCHOOL");
    await expect(proposal.getByText("new", { exact: true }).first()).toBeVisible();
    await proposal.getByRole("button", { name: "Add to mine" }).click();
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toBeVisible();
    await expect(proposal.getByText("new", { exact: true })).toHaveCount(0);
    await expect(editor).toContainText("Changed from the default profile.");
    // A description is saved with Enter and abandoned with Cancel.
    await entityTypes.getByRole("button", { name: "SCHOOL", exact: true }).click();
    await entityTypes.getByLabel("Description of SCHOOL").fill("A named place of instruction.");
    await entityTypes.getByLabel("Description of SCHOOL").press("Enter");
    await expect(entityTypes.getByLabel("Description of SCHOOL")).toHaveCount(0);
    await expect(entityTypes.getByRole("button", { name: "SCHOOL", exact: true })).toHaveAttribute(
      "title",
      "A named place of instruction.",
    );
    await entityTypes.getByRole("button", { name: "SCHOOL", exact: true }).click();
    await entityTypes.getByRole("button", { name: "Cancel" }).click();
    await expect(entityTypes.getByLabel("Description of SCHOOL")).toHaveCount(0);
    // The panel can be put away without taking anything from it.
    await proposal.getByRole("button", { name: "Dismiss suggestion" }).click();
    await expect(page.getByTestId("ontology-proposal")).toHaveCount(0);
  });

  test("picks neighborhoods from a ranked list rather than a wall of chips", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    const picker = page.getByTestId("neighborhood-picker");
    // Nothing chosen: one button, no chips.
    await expect(picker.getByRole("button", { name: "Choose neighborhoods" })).toBeVisible();
    await expect(picker.getByRole("option")).toHaveCount(0);
    await picker.getByRole("button", { name: "Choose neighborhoods" }).click();
    const options = picker.getByRole("option");
    const total = await options.count();
    expect(total).toBeGreaterThan(1);
    // Largest first, each with its size.
    await expect(options.first()).toContainText(/\d+ entit/);
    await picker.getByLabel("Find a neighborhood").fill("person");
    await expect(picker.getByRole("option")).toHaveCount(1);
    await picker.getByRole("option", { name: /PERSON/ }).click();
    await picker.getByLabel("Find a neighborhood").fill("");
    await picker.getByRole("option", { name: /MARTIAL_ART/ }).click();
    await picker.getByRole("button", { name: "Done" }).click();
    // The chosen ones are chips on the row; the canvas scope follows them.
    await expect(picker.getByRole("option")).toHaveCount(0);
    await expect(picker.getByRole("button", { name: "2 chosen" })).toBeVisible();
    await expect(picker.getByRole("button", { name: "Remove PERSON" })).toBeVisible();
    await expect(picker.getByRole("button", { name: "Remove MARTIAL_ART" })).toBeVisible();
    await picker.getByRole("button", { name: "Remove PERSON" }).click();
    await expect(picker.getByRole("button", { name: "1 chosen" })).toBeVisible();
    await picker.getByRole("button", { name: "Remove MARTIAL_ART" }).click();
    await expect(picker.getByRole("button", { name: "Choose neighborhoods" })).toBeVisible();
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
