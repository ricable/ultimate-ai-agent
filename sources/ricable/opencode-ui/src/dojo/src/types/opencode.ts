/**
 * Comprehensive TypeScript types for OpenCode API client
 * This extends the base types with additional interfaces needed for the enhanced client
 */

// View System Types
export type OpenCodeView = 
  | "welcome" 
  | "projects" 
  | "providers" 
  | "agents" 
  | "settings" 
  | "session" 
  | "usage-dashboard" 
  | "mcp"
  | "tools";

// Extended Session Types
export interface SessionUpdate {
  type: "message" | "status" | "error" | "tool_execution" | "stream_start" | "stream_end" | "heartbeat";
  data: any;
  timestamp: number;
}

export interface OpenCodeStreamMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  provider?: string;
  model?: string;
  tokens?: {
    input: number;
    output: number;
  };
  cost?: number;
  tool_calls?: Array<{
    id: string;
    name: string;
    arguments: any;
    result?: any;
  }>;
  metadata?: Record<string, any>;
  
  // Legacy properties for backwards compatibility
  type?: string;
  subtype?: string;
  isMeta?: boolean;
  leafUuid?: string;
  summary?: string;
  message?: any;
  cwd?: string;
  tools?: string[];
  session_id?: string;
  is_error?: boolean;
  result?: any;
  error?: string;
  cost_usd?: number;
  total_cost_usd?: number;
  duration_ms?: number;
  num_turns?: number;
  usage?: any;
}

export interface SessionSubscription {
  sessionId: string;
  unsubscribe: () => void;
  isActive: boolean;
}

export interface ShareLink {
  url: string;
  expires_at: number;
  password_protected: boolean;
  view_count: number;
}

// Extended Provider Types
export interface AuthResult {
  success: boolean;
  message?: string;
  provider_id: string;
  expires_at?: number;
}

export interface ProviderSubscription {
  unsubscribe: () => void;
  isActive: boolean;
}

// Tool System Extensions
export interface ToolResult {
  success: boolean;
  result: any;
  error?: string;
  cost?: number;
  execution_time: number;
  tool_id: string;
}

export interface ToolExecutionRequest {
  tool_id: string;
  params: Record<string, any>;
  session_id?: string;
  require_approval?: boolean;
  timeout?: number;
}

export interface ToolSubscription {
  unsubscribe: () => void;
  isActive: boolean;
}

// LSP Integration Types
export interface LSPServer {
  id: string;
  name: string;
  command: string;
  args: string[];
  enabled: boolean;
  status: "running" | "stopped" | "error";
  languages: string[];
  features: string[];
}

export interface LSPDiagnostic {
  file_path: string;
  line: number;
  column: number;
  severity: "error" | "warning" | "info" | "hint";
  message: string;
  source: string;
  code?: string;
}

// Custom Commands
export interface CustomCommand {
  id: string;
  name: string;
  description: string;
  command: string;
  args: Record<string, string>;
  enabled: boolean;
  shortcuts?: string[];
}

// WebSocket Management
export interface WebSocketManager {
  connect: (url: string, protocols?: string[]) => WebSocket;
  disconnect: (ws: WebSocket) => void;
  reconnect: (ws: WebSocket) => WebSocket;
  getStatus: (ws: WebSocket) => "connecting" | "open" | "closing" | "closed";
}

// Event System
export interface EventEmitter {
  on: (event: string, callback: Function) => () => void;
  off: (event: string, callback: Function) => void;
  emit: (event: string, data: any) => void;
  once: (event: string, callback: Function) => () => void;
}

// HTTP Client Configuration
export interface HTTPClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  headers: Record<string, string>;
}

export interface HTTPClient {
  get: <T>(url: string, config?: RequestInit) => Promise<T>;
  post: <T>(url: string, data?: any, config?: RequestInit) => Promise<T>;
  put: <T>(url: string, data?: any, config?: RequestInit) => Promise<T>;
  delete: <T>(url: string, config?: RequestInit) => Promise<T>;
  patch: <T>(url: string, data?: any, config?: RequestInit) => Promise<T>;
}

// Session Templates and Management
export interface SessionTemplate {
  id: string;
  name: string;
  description: string;
  provider: string;
  model: string;
  system_prompt?: string;
  tools: string[];
  configuration: Record<string, any>;
  created_by: string;
  is_public: boolean;
}

// Extended Configuration Types
export interface ModelConfig {
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop_sequences?: string[];
}

export interface ProviderEndpoint {
  id: string;
  url: string;
  auth_type: "api_key" | "oauth" | "basic" | "bearer";
  headers?: Record<string, string>;
  timeout?: number;
}

// Enhanced Error Types
export interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
  retry_after?: number;
  request_id?: string;
}

// Streaming Types
export interface StreamChunk {
  type: "delta" | "complete" | "error";
  content?: string;
  metadata?: Record<string, any>;
  finish_reason?: string;
}

export interface StreamOptions {
  session_id: string;
  message: string;
  model_config?: ModelConfig;
  tools_enabled?: boolean;
}

// Usage and Analytics
export interface UsageStats {
  total_sessions: number;
  total_messages: number;
  total_cost: number;
  avg_response_time: number;
  most_used_provider: string;
  most_used_model: string;
  today: {
    sessions: number;
    messages: number;
    cost: number;
  };
  this_week: {
    sessions: number;
    messages: number;
    cost: number;
  };
  this_month: {
    sessions: number;
    messages: number;
    cost: number;
  };
}

export interface CostBreakdown {
  provider_id: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_cost: number;
  requests: number;
  avg_cost_per_request: number;
}

// Connection and Health
export interface ConnectionStatus {
  status: "connected" | "connecting" | "disconnected" | "error";
  last_connected: number;
  reconnect_attempts: number;
  latency: number;
  server_version: string;
}

export interface ServerInfo {
  version: string;
  uptime: number;
  providers_count: number;
  active_sessions: number;
  features: string[];
  limits: {
    max_sessions: number;
    max_message_length: number;
    max_file_size: number;
  };
}

// Project Management Extensions
export interface ProjectStats {
  total_sessions: number;
  total_messages: number;
  total_cost: number;
  last_activity: number;
  file_count: number;
  languages: string[];
}

export interface ProjectTemplate {
  id: string;
  name: string;
  description: string;
  structure: {
    files: string[];
    folders: string[];
  };
  default_config: Record<string, any>;
  session_templates: string[];
}

// File System Integration
export interface FileOperationResult {
  success: boolean;
  path: string;
  operation: "read" | "write" | "delete" | "create" | "move" | "copy";
  size?: number;
  modified_at?: number;
  error?: string;
}

export interface DirectoryListing {
  path: string;
  files: Array<{
    name: string;
    type: "file" | "directory";
    size: number;
    modified_at: number;
    permissions: string;
  }>;
}

// Enhanced Checkpoint System Types
export interface CheckpointMetadata {
  userPrompt?: string;
  model?: string;
  provider?: string;
  totalTokens: number;
  inputTokens?: number;
  outputTokens?: number;
  cost?: number;
  executionTime?: number;
  fileChanges: number;
  toolsUsed?: string[];
  filesModified?: string[];
  snapshotSize?: number;
  parentCheckpointId?: string;
}

export interface Checkpoint {
  id: string;
  session_id: string;
  name: string;
  description?: string;
  timestamp: number;
  message_index: number;
  type: 'manual' | 'auto' | 'milestone' | 'fork';
  metadata: CheckpointMetadata;
  parent_checkpoint_id?: string;
  children?: string[]; // IDs of child checkpoints
  created_at: number;
}

export interface TimelineNode {
  checkpoint: Checkpoint;
  children: TimelineNode[];
  parent?: TimelineNode;
  depth: number;
}

export interface SessionTimeline {
  session_id: string;
  root_node?: TimelineNode;
  current_checkpoint_id?: string;
  total_checkpoints: number;
  auto_checkpoint_enabled: boolean;
  checkpoint_strategy: CheckpointStrategy;
}

export type CheckpointStrategy = 'manual' | 'per_prompt' | 'per_tool' | 'smart';

export interface CheckpointResult {
  checkpoint: Checkpoint;
  files_processed: number;
  warnings: string[];
  success: boolean;
}

export interface FileSnapshot {
  checkpoint_id: string;
  file_path: string;
  content: string;
  hash: string;
  is_deleted: boolean;
  permissions?: number;
  size: number;
  encoding?: string;
}

export interface CheckpointDiff {
  from_checkpoint: Checkpoint;
  to_checkpoint: Checkpoint;
  modified_files: FileDiff[];
  added_files: string[];
  deleted_files: string[];
  token_delta: number;
  cost_delta: number;
  total_changes: number;
}

export interface FileDiff {
  path: string;
  old_content: string;
  new_content: string;
  additions: number;
  deletions: number;
  changes: DiffHunk[];
}

export interface DiffHunk {
  old_start: number;
  old_count: number;
  new_start: number;
  new_count: number;
  lines: DiffLine[];
}

export interface DiffLine {
  type: 'context' | 'added' | 'removed';
  content: string;
  old_line_number?: number;
  new_line_number?: number;
}

export interface CheckpointExportFormat {
  format: 'json' | 'markdown' | 'pdf' | 'zip';
  include_files: boolean;
  include_metadata: boolean;
  include_diffs: boolean;
  password_protected?: boolean;
}

export interface CheckpointExportResult {
  success: boolean;
  file_path?: string;
  download_url?: string;
  error?: string;
  size_bytes: number;
}

export interface CheckpointForkOptions {
  name?: string;
  description?: string;
  preserve_messages: boolean;
  preserve_files: boolean;
  create_branch: boolean;
}

export interface CheckpointSearchFilters {
  keyword?: string;
  type?: CheckpointType[];
  date_range?: {
    start: number;
    end: number;
  };
  provider?: string;
  model?: string;
  has_tools?: boolean;
  min_cost?: number;
  max_cost?: number;
}

export type CheckpointType = 'manual' | 'auto' | 'milestone' | 'fork';

export interface CheckpointStats {
  total_checkpoints: number;
  manual_checkpoints: number;
  auto_checkpoints: number;
  milestone_checkpoints: number;
  fork_checkpoints: number;
  total_size_bytes: number;
  avg_files_per_checkpoint: number;
  most_active_day: string;
  cost_breakdown: {
    total: number;
    by_provider: Record<string, number>;
    by_model: Record<string, number>;
  };
}

// Agent Management System
export interface Agent {
  id: string;
  name: string;
  description?: string;
  icon: string;
  system_prompt: string;
  default_task?: string;
  model: string;
  provider: string;
  temperature?: number;
  max_tokens?: number;
  tools: string[];
  capabilities: AgentCapability[];
  sandbox_enabled: boolean;
  permissions: AgentPermissions;
  created_at: string;
  updated_at: string;
  created_by: string;
  tags: string[];
  category: AgentCategory;
  is_public: boolean;
  usage_count: number;
  rating: number;
  version: string;
}

export interface AgentCapability {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  config?: Record<string, any>;
}

export interface AgentPermissions {
  file_read: boolean;
  file_write: boolean;
  network_access: boolean;
  system_commands: boolean;
  environment_access: boolean;
  package_management: boolean;
  custom_tools: string[];
}

export type AgentCategory = 
  | "coding" 
  | "data-analysis" 
  | "writing" 
  | "research" 
  | "automation" 
  | "testing" 
  | "deployment" 
  | "documentation" 
  | "security" 
  | "custom";

export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  category: AgentCategory;
  icon: string;
  system_prompt_template: string;
  default_config: Partial<Agent>;
  required_tools: string[];
  recommended_models: string[];
  created_by: string;
  downloads: number;
  rating: number;
  tags: string[];
}

export interface AgentRun {
  id: string;
  agent_id: string;
  agent_name: string;
  agent_icon: string;
  session_id: string;
  task: string;
  model: string;
  provider: string;
  project_path: string;
  status: AgentRunStatus;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  error_message?: string;
  metrics: AgentRunMetrics;
  output_summary?: string;
}

export type AgentRunStatus = 
  | "pending" 
  | "initializing" 
  | "running" 
  | "paused" 
  | "completed" 
  | "failed" 
  | "cancelled" 
  | "timeout";

export interface AgentRunMetrics {
  total_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  message_count?: number;
  tool_executions?: number;
  errors_count?: number;
  avg_response_time?: number;
}

export interface AgentPerformanceMetrics {
  agent_id: string;
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  avg_duration_ms: number;
  total_cost_usd: number;
  avg_cost_per_run: number;
  total_tokens: number;
  success_rate: number;
  last_used: string;
  most_used_provider: string;
  most_used_model: string;
  common_tasks: Array<{
    task: string;
    count: number;
    avg_duration: number;
    success_rate: number;
  }>;
}

export interface AgentExecutionRequest {
  agent_id: string;
  task: string;
  project_path?: string;
  model_override?: string;
  provider_override?: string;
  config_override?: Partial<ModelConfig>;
  tools_override?: string[];
  context?: Record<string, any>;
  timeout_ms?: number;
}

export interface AgentExecutionContext {
  session_id: string;
  project_path: string;
  environment: Record<string, string>;
  available_tools: string[];
  sandbox_config: SandboxConfig;
}

export interface SandboxConfig {
  enabled: boolean;
  allowed_paths: string[];
  blocked_paths: string[];
  allowed_commands: string[];
  blocked_commands: string[];
  network_restrictions: NetworkRestriction[];
  environment_variables: Record<string, string>;
  timeout_seconds: number;
  memory_limit_mb?: number;
  cpu_limit_percent?: number;
}

export interface NetworkRestriction {
  type: "allow" | "block";
  pattern: string;
  description?: string;
}

export interface AgentMarketplace {
  featured_agents: Agent[];
  categories: Array<{
    id: AgentCategory;
    name: string;
    description: string;
    agent_count: number;
  }>;
  trending_agents: Agent[];
  new_agents: Agent[];
  top_rated_agents: Agent[];
}

export interface AgentTestResult {
  agent_id: string;
  test_name: string;
  test_description: string;
  input_task: string;
  expected_outcome: string;
  actual_outcome: string;
  status: "passed" | "failed" | "skipped";
  duration_ms: number;
  error_message?: string;
  metrics: AgentRunMetrics;
  timestamp: string;
}

export interface AgentTestSuite {
  id: string;
  name: string;
  description: string;
  agent_id: string;
  tests: AgentTestCase[];
  last_run: string;
  pass_rate: number;
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
}

export interface AgentTestCase {
  id: string;
  name: string;
  description: string;
  input_task: string;
  expected_outputs: string[];
  validation_rules: ValidationRule[];
  timeout_ms: number;
  enabled: boolean;
}

export interface ValidationRule {
  type: "contains" | "matches" | "not_contains" | "json_schema" | "custom";
  value: string;
  description: string;
}

export interface AgentComparisonResult {
  agents: Agent[];
  metrics: Record<string, AgentPerformanceMetrics>;
  benchmark_tasks: Array<{
    task: string;
    results: Record<string, AgentTestResult>;
  }>;
  recommendation: {
    best_overall: string;
    best_for_speed: string;
    best_for_accuracy: string;
    best_for_cost: string;
  };
}

export interface AgentCollaboration {
  id: string;
  name: string;
  description: string;
  agents: Array<{
    agent_id: string;
    role: string;
    order: number;
    handoff_conditions: string[];
  }>;
  workflow: CollaborationWorkflow;
  created_by: string;
  created_at: string;
}

export interface CollaborationWorkflow {
  steps: WorkflowStep[];
  error_handling: ErrorHandlingStrategy;
  timeout_ms: number;
  parallel_execution: boolean;
}

export interface WorkflowStep {
  id: string;
  agent_id: string;
  task_template: string;
  inputs: WorkflowInput[];
  outputs: WorkflowOutput[];
  conditions: WorkflowCondition[];
}

export interface WorkflowInput {
  name: string;
  type: "string" | "number" | "file" | "json";
  source: "user" | "previous_step" | "context";
  required: boolean;
}

export interface WorkflowOutput {
  name: string;
  type: "string" | "number" | "file" | "json";
  destination: "next_step" | "user" | "storage";
  format?: string;
}

export interface WorkflowCondition {
  type: "success" | "failure" | "custom";
  action: "continue" | "retry" | "skip" | "stop";
  max_retries?: number;
}

export type ErrorHandlingStrategy = "stop" | "continue" | "retry" | "fallback";

// Agent Development and Sharing
export interface AgentShareLink {
  id: string;
  agent_id: string;
  url: string;
  expires_at?: string;
  password_protected: boolean;
  download_count: number;
  created_at: string;
}

export interface AgentFeedback {
  id: string;
  agent_id: string;
  user_id: string;
  rating: number;
  comment?: string;
  categories: FeedbackCategory[];
  helpful_count: number;
  created_at: string;
}

export type FeedbackCategory = 
  | "accuracy" 
  | "speed" 
  | "usefulness" 
  | "documentation" 
  | "ease_of_use";

export interface AgentAnalytics {
  agent_id: string;
  timeframe: "hour" | "day" | "week" | "month";
  usage_data: Array<{
    timestamp: string;
    executions: number;
    success_rate: number;
    avg_duration: number;
    total_cost: number;
  }>;
  provider_breakdown: Record<string, {
    usage_count: number;
    success_rate: number;
    avg_cost: number;
  }>;
  task_categories: Record<string, {
    count: number;
    avg_duration: number;
    success_rate: number;
  }>;
  error_patterns: Array<{
    error_type: string;
    count: number;
    last_occurrence: string;
  }>;
}

// Agent Configuration and Management
export interface AgentConfigProfile {
  id: string;
  name: string;
  description: string;
  agent_id: string;
  config: {
    model: string;
    provider: string;
    temperature: number;
    max_tokens: number;
    tools: string[];
    permissions: AgentPermissions;
    sandbox_config: SandboxConfig;
  };
  is_default: boolean;
  created_at: string;
}

export interface AgentBackup {
  id: string;
  agent_id: string;
  version: string;
  backup_data: Agent;
  created_at: string;
  created_by: string;
  description?: string;
  auto_backup: boolean;
}

export interface AgentDeployment {
  id: string;
  agent_id: string;
  environment: "development" | "staging" | "production";
  endpoint_url?: string;
  api_key?: string;
  status: "deployed" | "deploying" | "failed" | "stopped";
  deployed_at: string;
  config: Record<string, any>;
}