import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { resetAppEnv } from "../env";
import { createContext } from "./init";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

async function context(headers: Record<string, string>, trust: boolean) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-context-"));
  process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS = trust ? "1" : "0";
  process.env.FLAKEGRAPH_APP_DEFAULT_RUNTIME = "kubernetes";
  process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
  resetAppEnv();
  return createContext({ headers: new Headers(headers) });
}

describe("request context", () => {
  it("takes the gate's identity when the gate is trusted", async () => {
    const ctx = await context({ "x-auth-request-email": "alice@example.test", "x-auth-request-user": "alice" }, true);
    expect(ctx.viewer.userName).toBe("ALICE");
    expect(ctx.viewer.email).toBe("alice@example.test");
  });

  it("ignores identity headers nobody vouched for", async () => {
    const ctx = await context({ "x-auth-request-user": "mallory" }, false);
    expect(ctx.viewer.userName).toBe("");
  });

  it("opens on the configured runtime when the address names none", async () => {
    const ctx = await context({}, false);
    expect(ctx.runtimeName).toBe("kubernetes");
    const named = await context({ "x-flakegraph-runtime": "local" }, false);
    expect(named.runtimeName).toBe("local");
  });
});
