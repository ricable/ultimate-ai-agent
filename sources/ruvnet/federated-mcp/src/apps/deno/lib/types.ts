/// <reference lib="deno.ns" />

import { ServerInfo } from "../../../packages/core/types.ts";

export interface ServerStats {
  connections: number;
  edgeFunctionsEnabled: boolean;
  activeEdgeFunctions: string[];
  deployedUrls: Record<string, string>;
  selectedProvider?: string;
  lastDeployment?: string;
}

export interface EdgeProviderConfig {
  name: string;
  requiredEnvVars: { name: string; description: string }[];
  isConfigured: () => boolean;
}

export const serverInfo: ServerInfo = {
  name: "deno-mcp-server",
  version: "1.0.0",
  capabilities: {
    models: ["gpt-3.5-turbo", "gpt-4"],
    protocols: ["json-rpc", "http", "websocket"],
    features: [
      "task-execution",
      "federation", 
      "intent-detection",
      "meeting-info",
      "webhook-handler"
    ]
  }
};

export const stats: ServerStats = {
  connections: 0,
  edgeFunctionsEnabled: false,
  activeEdgeFunctions: [],
  deployedUrls: {},
  selectedProvider: undefined
};

export const edgeProviders: Record<string, EdgeProviderConfig> = {
  supabase: {
    name: "Supabase",
    requiredEnvVars: [
      { name: "SUPABASE_PROJECT_ID", description: "Your Supabase project ID" },
      { name: "SUPABASE_ACCESS_TOKEN", description: "Your Supabase access token" }
    ],
    isConfigured: () => !!Deno.env.get("SUPABASE_PROJECT_ID") && !!Deno.env.get("SUPABASE_ACCESS_TOKEN")
  },
  cloudflare: {
    name: "Cloudflare Workers",
    requiredEnvVars: [
      { name: "CLOUDFLARE_API_TOKEN", description: "Your Cloudflare API token" },
      { name: "CLOUDFLARE_ACCOUNT_ID", description: "Your Cloudflare account ID" }
    ],
    isConfigured: () => !!Deno.env.get("CLOUDFLARE_API_TOKEN") && !!Deno.env.get("CLOUDFLARE_ACCOUNT_ID")
  },
  flyio: {
    name: "Fly.io",
    requiredEnvVars: [
      { name: "FLY_API_TOKEN", description: "Your Fly.io API token" },
      { name: "FLY_APP_NAME", description: "Your Fly.io application name" }
    ],
    isConfigured: () => !!Deno.env.get("FLY_API_TOKEN") && !!Deno.env.get("FLY_APP_NAME")
  }
};
