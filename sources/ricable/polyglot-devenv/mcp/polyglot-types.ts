/**
 * Types for the Polyglot Development Environment MCP Server
 */

export interface EnvironmentInfo {
  name: string;
  path: string;
  type: "python" | "typescript" | "rust" | "go" | "nushell";
  status: "active" | "inactive" | "error";
  devboxConfig?: DevboxConfig;
  lastModified?: Date;
}

export interface DevboxConfig {
  packages: string[];
  shell?: {
    init_hook?: string[];
    scripts?: Record<string, string>;
  };
  env?: Record<string, string>;
}

export interface CommandResult {
  success: boolean;
  output: string;
  error?: string;
  exitCode: number;
  duration: number;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

export interface ToolExecutionResult extends CommandResult {
  toolName: string;
  environment?: string;
  operation?: string;
  progress?: number;
  stage?: string;
  estimatedTimeRemaining?: number;
}

export interface PerformanceMetric {
  timestamp: Date;
  eventType: string;
  environment: string;
  duration: number;
  status: "success" | "failure" | "timeout";
  details?: Record<string, unknown>;
}

export interface DevPodWorkspace {
  name: string;
  id: string;
  environment: string;
  status: "running" | "stopped" | "creating" | "error";
  created: Date;
  provider: string;
}

export interface SecurityScanResult {
  environment: string;
  findings: SecurityFinding[];
  scanTime: Date;
  status: "clean" | "warnings" | "critical";
}

export interface SecurityFinding {
  type: "secret" | "vulnerability" | "misconfiguration";
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  file?: string;
  line?: number;
  suggestion?: string;
}

export interface ValidationResult {
  environment: string;
  checks: ValidationCheck[];
  overallStatus: "passed" | "warnings" | "failed";
  summary: string;
}

export interface ValidationCheck {
  name: string;
  status: "passed" | "warning" | "failed";
  message: string;
  details?: string;
}

export interface HookExecution {
  id: string;
  hookType: string;
  trigger: string;
  timestamp: Date;
  status: "success" | "failure";
  output: string;
  duration: number;
}

export interface PRPInfo {
  name: string;
  path: string;
  environment: string;
  status: "draft" | "ready" | "executing" | "completed" | "failed";
  lastModified: Date;
  validation?: ValidationResult;
}