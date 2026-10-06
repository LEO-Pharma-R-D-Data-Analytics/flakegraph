import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { TRPCError } from "@trpc/server";
import { afterEach, describe, expect, it } from "vitest";
import { resetAppEnv } from "../env";
import { createContext } from "./init";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

async function context(headers: Record<string, string>, flags: Record<string, string>) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-context-"));
  delete process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS;
  delete process.env.FLAKEGRAPH_DEMO_IDENTITIES;
  delete process.env.FLAKEGRAPH_APP_ANONYMOUS_READ;
  process.env.FLAKEGRAPH_APP_DEFAULT_RUNTIME = "kubernetes";
  process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
  Object.assign(process.env, flags);
  resetAppEnv();
  return createContext({ headers: new Headers(headers) });
}

describe("request context", () => {
  it("takes the gate's identity when the gate is trusted", async () => {
    const ctx = await context(
      { "x-auth-request-email": "alice@example.test", "x-auth-request-user": "c7s1v9-opaque-subject" },
      { FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" },
    );
    // The address, not the provider's opaque subject id, is who the viewer is.
    expect(ctx.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(ctx.viewer.email).toBe("alice@example.test");
    expect(ctx.authorization).toBe("gate");
  });

  it("refuses identity headers nobody vouched for", async () => {
    await expect(context({ "x-auth-request-user": "mallory" }, {})).rejects.toBeInstanceOf(TRPCError);
    // Even where anonymous reads are open, the header does not make them mallory.
    const ctx = await context({ "x-auth-request-user": "mallory" }, { FLAKEGRAPH_APP_ANONYMOUS_READ: "1" });
    expect(ctx.viewer.userName).toBe("");
    expect(ctx.authorization).toBe("anonymous");
  });

  it("opens on the configured runtime when the address names none", async () => {
    const ctx = await context({}, { FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    expect(ctx.runtimeName).toBe("kubernetes");
    expect(ctx.authorization).toBe("demo");
    const named = await context({ "x-flakegraph-runtime": "local" }, { FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    expect(named.runtimeName).toBe("local");
  });
});
