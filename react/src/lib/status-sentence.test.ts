import { describe, expect, it } from "vitest";
import { catalogEmptyCopy, statusSentence } from "./status-sentence";

describe("statusSentence", () => {
  it("writes a blocking sentence for running work", () => {
    expect(
      statusSentence({
        status: "running",
        error: null,
        documentsCompleted: 3,
        documentsTotal: 10,
        documentsFailed: 0,
        stages: [{ stage: "ocr", status: "running", completed: 3, total: 10, elapsedMs: 0, message: null }],
      }),
    ).toMatch(/Ocr · 3\/10 documents/i);
  });

  it("does not call a zero-entity finish succeeded", () => {
    expect(
      statusSentence(
        {
          status: "succeeded",
          error: null,
          documentsCompleted: 10,
          documentsTotal: 10,
          documentsFailed: 0,
          stages: [],
        },
        { nodeCount: 0 },
      ),
    ).toMatch(/0 entities/i);
  });

  it("prefers entity count over a zero document total", () => {
    expect(
      statusSentence(
        {
          status: "succeeded",
          error: null,
          documentsCompleted: 0,
          documentsTotal: 0,
          documentsFailed: 0,
          stages: [],
        },
        { nodeCount: 74 },
      ),
    ).toMatch(/74 entities/);
  });
});

describe("catalogEmptyCopy", () => {
  it("distinguishes first-run empty from a search miss", () => {
    expect(
      catalogEmptyCopy({
        loading: false,
        total: 0,
        visible: 0,
        search: "",
        storageFilter: "all",
        mine: false,
        identified: false,
      })?.title,
    ).toBe("No graphs yet");
    expect(
      catalogEmptyCopy({
        loading: false,
        total: 4,
        visible: 0,
        search: "foo",
        storageFilter: "all",
        mine: false,
        identified: false,
      })?.title,
    ).toContain("foo");
  });

  it("tells unidentified users Mine is empty while All still has graphs", () => {
    expect(
      catalogEmptyCopy({
        loading: false,
        total: 4,
        visible: 4,
        search: "",
        storageFilter: "all",
        mine: true,
        identified: false,
      })?.title,
    ).toBe("0 yours");
  });

  it("tells identified users Mine is empty when they own none of the catalog", () => {
    expect(
      catalogEmptyCopy({
        loading: false,
        total: 4,
        visible: 0,
        search: "",
        storageFilter: "all",
        mine: true,
        identified: true,
      })?.detail,
    ).toMatch(/None list you as owner/);
  });

  it("tells analysts when the catalog has no production perspectives", () => {
    expect(
      catalogEmptyCopy({
        loading: false,
        total: 4,
        visible: 0,
        search: "",
        storageFilter: "all",
        mine: false,
        identified: true,
        analyst: true,
      })?.title,
    ).toBe("No production graphs");
  });
});
