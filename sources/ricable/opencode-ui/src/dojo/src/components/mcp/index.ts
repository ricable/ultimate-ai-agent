/**
 * MCP (Model Context Protocol) Components
 * 
 * Comprehensive set of components for managing MCP servers including
 * configuration, monitoring, templates, and testing utilities.
 */

export { ServerConfigForm } from "./server-config-form";
export { ServerDashboard } from "./server-dashboard";
export { ServerHealthMonitor } from "./server-health-monitor";
export { ServerTemplates } from "./server-templates";
export { TestingUtils } from "./testing-utils";

// Re-export types for convenience
export type {
  MCPServerConfig,
  MCPServerWithStatus,
  MCPServerTemplate,
  MCPServerStatus,
  MCPServerHealth,
  MCPServerMetrics,
  MCPServerTestResult,
  MCPServerValidation,
  MCPTool,
  MCPResource,
  MCPServerFormData,
  MCPServerFilters,
  MCPServerSort,
  MCPServerOperation,
  MCPServerBackup,
  MCPViewState,
  MCPServerEvent,
  MCPToolEvent
} from "@/lib/types/mcp";