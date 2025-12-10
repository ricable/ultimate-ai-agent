import { AgentIntegrationConfig } from "./types/integration";

// Simplified agents configuration for OpenCode demo
// Remove external dependencies that aren't installed
export const agentsIntegrations: AgentIntegrationConfig[] = [
  {
    id: "opencode-demo",
    agents: async () => {
      return {
        agentic_chat: {
          // Mock agent for demo purposes
          // Will be replaced with actual OpenCode agent integration
        } as any,
      };
    },
  },
];