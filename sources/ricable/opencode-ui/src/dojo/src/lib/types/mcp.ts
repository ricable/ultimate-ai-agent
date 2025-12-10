/**
 * Enhanced MCP (Model Context Protocol) Server Types
 * 
 * Comprehensive TypeScript definitions for MCP server management,
 * configuration, monitoring, and integration with OpenCode backend.
 */

// Core MCP Server Types
export interface MCPServerConfig {
  id: string;
  name: string;
  description?: string;
  type: "local" | "remote";
  enabled: boolean;
  
  // Local server configuration
  command?: string[];
  environment?: Record<string, string>;
  workingDirectory?: string;
  
  // Remote server configuration
  url?: string;
  headers?: Record<string, string>;
  authentication?: {
    type: "bearer" | "basic" | "api-key";
    token?: string;
    username?: string;
    password?: string;
  };
  
  // Advanced configuration
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  autoRestart?: boolean;
  
  // Metadata
  tags?: string[];
  category?: string;
  version?: string;
  author?: string;
  documentation?: string;
  created_at?: number;
  updated_at?: number;
}

export interface MCPServerStatus {
  id: string;
  status: "connected" | "disconnected" | "connecting" | "error" | "unknown";
  connected_at?: number;
  last_ping?: number;
  last_error?: string;
  uptime?: number;
  restart_count?: number;
  process_id?: number;
  memory_usage?: number;
  cpu_usage?: number;
  tools_count?: number;
  resources_count?: number;
}

export interface MCPServerHealth {
  id: string;
  healthy: boolean;
  response_time?: number;
  last_health_check?: number;
  error_count?: number;
  warning_count?: number;
  performance_score?: number;
  availability_percentage?: number;
}

export interface MCPServerMetrics {
  id: string;
  requests_total: number;
  requests_successful: number;
  requests_failed: number;
  avg_response_time: number;
  min_response_time: number;
  max_response_time: number;
  bytes_sent: number;
  bytes_received: number;
  last_24h: {
    requests: number;
    errors: number;
    avg_response_time: number;
  };
  last_7d: {
    requests: number;
    errors: number;
    avg_response_time: number;
  };
}

export interface MCPTool {
  id: string;
  server_id: string;
  name: string;
  description: string;
  input_schema: any;
  output_schema?: any;
  enabled: boolean;
  category?: string;
  tags?: string[];
  usage_count?: number;
  last_used?: number;
  error_count?: number;
  avg_execution_time?: number;
}

export interface MCPResource {
  id: string;
  server_id: string;
  name: string;
  description: string;
  type: string;
  uri: string;
  metadata?: Record<string, any>;
  last_accessed?: number;
  access_count?: number;
}

export interface MCPServerTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  author: string;
  version: string;
  documentation?: string;
  icon?: string;
  popularity?: number;
  rating?: number;
  downloads?: number;
  config: Partial<MCPServerConfig>;
  requirements?: {
    dependencies?: string[];
    environment?: string[];
    platforms?: string[];
  };
  examples?: Array<{
    name: string;
    description: string;
    config: Partial<MCPServerConfig>;
  }>;
}

export interface MCPServerValidation {
  valid: boolean;
  errors: Array<{
    field: string;
    message: string;
    code: string;
  }>;
  warnings: Array<{
    field: string;
    message: string;
    code: string;
  }>;
  suggestions?: Array<{
    field: string;
    message: string;
    value: any;
  }>;
}

export interface MCPServerTestResult {
  success: boolean;
  status: "reachable" | "unreachable" | "timeout" | "auth_failed" | "invalid_response";
  response_time?: number;
  error?: string;
  details?: {
    tools_discovered?: number;
    resources_discovered?: number;
    capabilities?: string[];
  };
  timestamp: number;
}

export interface MCPServerLog {
  id: string;
  server_id: string;
  timestamp: number;
  level: "debug" | "info" | "warn" | "error";
  message: string;
  category?: string;
  metadata?: Record<string, any>;
}

// Health Monitoring Types
export interface MCPHealthAlert {
  id: string;
  server_id: string;
  type: "critical" | "warning" | "info";
  metric: "response_time" | "cpu_usage" | "memory_usage" | "error_rate" | "uptime";
  message: string;
  value: number;
  threshold: number;
  acknowledged: boolean;
  timestamp: number;
}

export interface MCPHealthThreshold {
  metric: "response_time" | "cpu_usage" | "memory_usage" | "error_rate" | "uptime";
  enabled: boolean;
  warning_threshold: number;
  critical_threshold: number;
}

// Configuration and Management Types
export interface MCPServerFilters {
  status?: MCPServerStatus["status"][];
  type?: MCPServerConfig["type"][];
  category?: string[];
  tags?: string[];
  enabled?: boolean;
  search?: string;
}

export interface MCPServerSort {
  field: "name" | "created_at" | "updated_at" | "status" | "type" | "uptime";
  direction: "asc" | "desc";
}

export interface MCPServerOperation {
  id: string;
  server_id: string;
  operation: "start" | "stop" | "restart" | "update" | "delete" | "test";
  status: "pending" | "running" | "completed" | "failed";
  progress?: number;
  error?: string;
  started_at: number;
  completed_at?: number;
}

export interface MCPServerBackup {
  id: string;
  name: string;
  description?: string;
  servers: MCPServerConfig[];
  created_at: number;
  created_by: string;
  version: string;
  checksum: string;
}

// UI State Management Types
export interface MCPViewState {
  selectedServer?: string;
  selectedServers: string[];
  showAddDialog: boolean;
  showImportDialog: boolean;
  showExportDialog: boolean;
  showTemplateDialog: boolean;
  activeTab: "servers" | "templates" | "logs" | "metrics";
  filters: MCPServerFilters;
  sort: MCPServerSort;
  view: "grid" | "list" | "details";
  refreshing: boolean;
  lastRefresh?: number;
}

export interface MCPServerFormData {
  name: string;
  description: string;
  type: "local" | "remote";
  enabled: boolean;
  
  // Local server fields
  command: string;
  args: string;
  environment: string;
  workingDirectory: string;
  
  // Remote server fields
  url: string;
  headers: string;
  authType: "none" | "bearer" | "basic" | "api-key";
  authToken: string;
  authUsername: string;
  authPassword: string;
  
  // Advanced fields
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  autoRestart: boolean;
  
  // Metadata
  tags: string;
  category: string;
  version: string;
  author: string;
  documentation: string;
}

// Event Types for Real-time Updates
export interface MCPServerEvent {
  type: "server_added" | "server_updated" | "server_deleted" | "server_status_changed" | "server_error";
  server_id: string;
  data?: any;
  timestamp: number;
}

export interface MCPToolEvent {
  type: "tool_executed" | "tool_error" | "tool_added" | "tool_removed";
  server_id: string;
  tool_id: string;
  data?: any;
  timestamp: number;
}

// Utility Types
export type MCPServerWithStatus = MCPServerConfig & {
  status: MCPServerStatus;
  health: MCPServerHealth;
  metrics: MCPServerMetrics;
  tools: MCPTool[];
  resources: MCPResource[];
};

export type MCPServerSummary = Pick<MCPServerConfig, "id" | "name" | "type" | "enabled"> & {
  status: MCPServerStatus["status"];
  tools_count: number;
  last_connected?: number;
};

// Constants
export const MCP_SERVER_CATEGORIES = [
  "development",
  "productivity",
  "communication",
  "file-management",
  "automation",
  "monitoring",
  "analytics",
  "security",
  "integration",
  "custom"
] as const;

export const MCP_SERVER_STATUSES = [
  "connected",
  "disconnected", 
  "connecting",
  "error",
  "unknown"
] as const;

export const MCP_LOG_LEVELS = [
  "debug",
  "info", 
  "warn",
  "error"
] as const;