import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { authorizeRequest, RequestRefused } from "./auth";
import { resetAppEnv } from "./env";
import { createApiKey } from "./workspace";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

async function gatedConsole(options: { requireSignIn: boolean }) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-auth-"));
  process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS = "1";
  process.env.FLAKEGRAPH_APP_REQUIRE_SIGN_IN = options.requireSignIn ? "1" : "0";
  resetAppEnv();
}

describe("authorizeRequest", () => {
  it("serves the gate's viewer, and refuses a caller the gate never saw when a sign-in is required", async () => {
    await gatedConsole({ requireSignIn: true });
    const gated = await authorizeRequest(new Headers({ "x-auth-request-email": "alice@example.test" }));
    expect(gated.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(gated.machine).toBeNull();
    await expect(authorizeRequest(new Headers())).rejects.toBeInstanceOf(RequestRefused);
  });

  it("still serves an anonymous caller where no sign-in is required", async () => {
    await gatedConsole({ requireSignIn: false });
    const open = await authorizeRequest(new Headers());
    expect(open.viewer.userName).toBe("");
  });

  it("serves a machine by its key and refuses a key it does not hold", async () => {
    await gatedConsole({ requireSignIn: true });
    const { secret } = await createApiKey("ci");
    const machine = await authorizeRequest(new Headers({ authorization: `Bearer ${secret}` }));
    expect(machine.machine?.name).toBe("ci");
    expect(machine.viewer.userName).toBe("");
    await expect(authorizeRequest(new Headers({ "x-flakegraph-api-key": "fg_nope" }))).rejects.toThrow(
      /missing or revoked/,
    );
  });
});
