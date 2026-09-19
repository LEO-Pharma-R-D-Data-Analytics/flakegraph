import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["snowflake-sdk", "@kubernetes/client-node", "hyparquet", "postgres"],
};

export default nextConfig;
