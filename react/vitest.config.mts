import path from "node:path";
import { defineConfig } from "vitest/config";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [tsconfigPaths()],
  resolve: {
    alias: {
      "server-only": path.resolve(__dirname, "src/test-stubs/server-only.ts"),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    restoreMocks: true,
    env: {
      FLAKEGRAPH_ASK_IGNORE_HUB_SECRETS: "1",
      FLAKEGRAPH_ASK_DISABLE_OLLAMA: "1",
    },
  },
});
