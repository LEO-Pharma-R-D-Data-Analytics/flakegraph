"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink } from "@trpc/client";
import { createTRPCReact } from "@trpc/react-query";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { useState, type ReactNode } from "react";
import superjson from "superjson";
import type { AppRouter } from "@/server/trpc/router";
import type { RuntimeMode } from "@/server/protocol/schema";
import { Toaster } from "@/components/ui/sonner";

export const trpc = createTRPCReact<AppRouter>();

// The runtime travels as a header only when the address names one; otherwise
// the server's configured default applies, which on a deployed control plane
// is the fleet it fronts.
export function getRuntimeHeader(): RuntimeMode | null {
  if (typeof window === "undefined") {
    return null;
  }
  const params = new URLSearchParams(window.location.search);
  const runtime = params.get("runtime");
  if (runtime === "kubernetes" || runtime === "snowflake" || runtime === "local") {
    return runtime;
  }
  return null;
}

export function AppProviders({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { refetchOnWindowFocus: false, retry: 1 },
        },
      }),
  );
  const [client] = useState(() =>
    trpc.createClient({
      links: [
        httpBatchLink({
          url: "/api/trpc",
          transformer: superjson,
          headers() {
            const runtime = getRuntimeHeader();
            return runtime ? { "x-flakegraph-runtime": runtime } : {};
          },
          // While the control plane restarts, the edge answers in prose
          // ("no available server"); the parse error that follows says
          // nothing to anyone. Name the outage instead.
          fetch: async (input, init) => {
            const response = await fetch(input, init);
            const type = response.headers.get("content-type") ?? "";
            if (!response.ok && !type.includes("json")) {
              const text = (await response.text()).trim().slice(0, 120);
              throw new Error(
                response.status >= 500
                  ? `The control plane is not answering (${response.status}${text ? `: ${text}` : ""}). It may be restarting; try again in a moment.`
                  : `The control plane refused the request (${response.status}${text ? `: ${text}` : ""}).`,
              );
            }
            return response;
          },
        }),
      ],
    }),
  );
  return (
    <trpc.Provider client={client} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        <NuqsAdapter>
          {children}
          <Toaster />
        </NuqsAdapter>
      </QueryClientProvider>
    </trpc.Provider>
  );
}
