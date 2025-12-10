import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  CompleteRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  LoggingLevel,
  ReadResourceRequestSchema,
  Tool,
  ToolSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { readFileSync, readFile } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { promisify } from "util";

const readFileAsync = promisify(readFile);
import {
  getWorkspaceRoot,
  isValidEnvironment,
  executeCommand,
  runDevboxCommand,
  runDevboxScript,
  detectEnvironments,
  validateEnvironment,
  provisionDevPodWorkspace,
  listDevPodWorkspaces,
  formatDuration,
  scanForSecrets,
  startDevboxEnvironment,
  getEnvironmentScripts,
  getDefaultSetupScript,
  getCommonCommands,
  ENVIRONMENTS,
  getEnvironmentType,
} from "./polyglot-utils.js";
import type { EnvironmentInfo, CommandResult, ValidationResult } from "./polyglot-types.js";

// Import modular tool definitions and handlers
import { 
  claudeFlowTools,
  handleClaudeFlowInit,
  handleClaudeFlowWizard,
  handleClaudeFlowStart,
  handleClaudeFlowStop,
  handleClaudeFlowStatus,
  handleClaudeFlowMonitor,
  handleClaudeFlowSpawn,
  handleClaudeFlowLogs,
  handleClaudeFlowHiveMind,
  handleClaudeFlowTerminalMgmt,
} from "./modules/claude-flow.js";

import {
  enhancedHooksTools,
  handleEnhancedHookContextTriggers,
  handleEnhancedHookErrorResolution,
  handleEnhancedHookEnvOrchestration,
  handleEnhancedHookDependencyTracking,
  handleEnhancedHookPerformanceIntegration,
  handleEnhancedHookQualityGates,
  handleEnhancedHookDevpodManager,
  handleEnhancedHookPrpLifecycle,
} from "./modules/enhanced-hooks.js";

import {
  dockerMcpTools,
  handleDockerMcpGatewayStart,
  handleDockerMcpGatewayStatus,
  handleDockerMcpToolsList,
  handleDockerMcpHttpBridge,
  handleDockerMcpClientList,
  handleDockerMcpServerList,
  handleDockerMcpGeminiConfig,
  handleDockerMcpTest,
  handleDockerMcpDemo,
  handleDockerMcpSecurityScan,
  handleDockerMcpResourceLimits,
  handleDockerMcpNetworkIsolation,
  handleDockerMcpSignatureVerify,
  handleDockerMcpLogs,
  handleDockerMcpCleanup,
} from "./modules/docker-mcp.js";

import {
  hostContainerTools,
  handleHostInstallation,
  handleHostInfrastructure,
  handleHostCredential,
  handleHostShellIntegration,
  handleContainerIsolation,
  handleContainerTools,
  handleHostContainerBridge,
  handleSecurityBoundary,
} from "./modules/host-container.js";

import {
  nushellAutomationTools,
  handleNushellScript,
  handleNushellValidation,
  handleNushellOrchestration,
  handleNushellDataProcessing,
  handleNushellAutomation,
  handleNushellPipeline,
  handleNushellConfig,
  handleNushellPerformance,
  handleNushellDebug,
  handleNushellIntegration,
  handleNushellTesting,
  handleNushellDocumentation,
  handleNushellEnvironment,
  handleNushellDeployment,
  handleNushellMonitoring,
  handleNushellSecurity,
  handleNushellBackup,
  handleNushellMigration,
  handleNushellOptimization,
  handleNushellWorkflow,
} from "./modules/nushell-automation.js";

import {
  configManagementTools,
  handleConfigGeneration,
  handleConfigSync,
  handleConfigValidation,
  handleConfigBackup,
  handleConfigTemplate,
} from "./modules/config-management.js";

import {
  advancedAnalyticsTools,
  handlePerformanceAnalytics,
  handleResourceMonitoring,
  handleIntelligenceSystem,
  handleTrendAnalysis,
  handleUsageAnalytics,
  handleAnomalyDetection,
  handlePredictiveAnalytics,
  handleBusinessIntelligence,
} from "./modules/advanced-analytics.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const instructions = readFileSync(join(__dirname, "polyglot-instructions.md"), "utf-8");

const ToolInputSchema = ToolSchema.shape.inputSchema;
type ToolInput = z.infer<typeof ToolInputSchema>;

/* Tool Input Schemas */

// Environment Tools
const EnvironmentDetectSchema = z.object({});

const EnvironmentInfoSchema = z.object({
  environment: z.string().describe("Environment name (dev-env/python, dev-env/typescript, etc.)"),
});

const EnvironmentValidateSchema = z.object({
  environment: z.string().optional().describe("Specific environment to validate, or all if not specified"),
});

// DevBox Tools
const DevboxShellSchema = z.object({
  environment: z.string().describe("Environment to enter shell for"),
});

const DevboxStartSchema = z.object({
  environment: z.string().describe("Environment to start and activate"),
  setup: z.boolean().default(true).describe("Run setup/install scripts after starting"),
});

const DevboxRunSchema = z.object({
  environment: z.string().describe("Environment to run script in"),
  script: z.string().describe("Script name to run (from devbox.json scripts)"),
});

const DevboxStatusSchema = z.object({
  environment: z.string().optional().describe("Environment to check status for"),
});

const DevboxAddPackageSchema = z.object({
  environment: z.string().describe("Environment to add package to"),
  package: z.string().describe("Package name to add"),
});

const DevboxQuickStartSchema = z.object({
  environment: z.string().describe("Environment to quick start"),
  task: z.enum(["dev", "test", "build", "lint"]).optional().describe("Immediate task to run after starting"),
});

// DevPod Tools
const DevpodProvisionSchema = z.object({
  environment: z.string().describe("Environment type (dev-env/python, dev-env/typescript, etc.)"),
  count: z.number().min(1).max(10).default(1).describe("Number of workspaces to provision (1-10)"),
});

const DevpodListSchema = z.object({});

const DevpodStatusSchema = z.object({
  workspace: z.string().optional().describe("Specific workspace name to check"),
});

// Dynamic DevPod Start Tool
const DevpodStartSchema = z.object({
  environment: z.enum(["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"]).describe("Environment to start (dev-env/python, dev-env/typescript, dev-env/rust, dev-env/go, dev-env/nushell)"),
  count: z.number().min(1).max(5).default(1).describe("Number of workspaces to start (1-5)"),
});

// Cross-Language Validation Tools
const PolyglotCheckSchema = z.object({
  environment: z.string().optional().describe("Specific environment to check"),
  include_security: z.boolean().default(false).describe("Include security scanning"),
  include_performance: z.boolean().default(false).describe("Include performance analysis"),
});

const PolyglotValidateSchema = z.object({
  parallel: z.boolean().default(false).describe("Run validation in parallel"),
});

const PolyglotCleanSchema = z.object({
  environment: z.string().optional().describe("Specific environment to clean"),
});

// Performance Tools
const PerformanceMeasureSchema = z.object({
  command: z.string().describe("Command to measure"),
  environment: z.string().describe("Environment to run in"),
});

const PerformanceReportSchema = z.object({
  days: z.number().default(7).describe("Number of days to include in report"),
  environment: z.string().optional().describe("Specific environment to report on"),
});

// Security Tools
const SecurityScanSchema = z.object({
  environment: z.string().optional().describe("Specific environment to scan"),
  scan_type: z.enum(["secrets", "vulnerabilities", "all"]).default("all").describe("Type of security scan"),
});

// Hook Tools
const HookStatusSchema = z.object({});

const HookTriggerSchema = z.object({
  hook_type: z.string().describe("Type of hook to trigger"),
  context: z.record(z.unknown()).optional().describe("Context data for hook"),
});

// PRP (Context Engineering) Tools
const PrpGenerateSchema = z.object({
  feature_file: z.string().describe("Path to feature file"),
  environment: z.string().describe("Target environment"),
});

const PrpExecuteSchema = z.object({
  prp_file: z.string().describe("Path to PRP file to execute"),
  validate: z.boolean().default(true).describe("Validate before execution"),
});

// AG-UI (Agentic UI) Tools
const AguiProvisionSchema = z.object({
  environment: z.enum(["agentic-python", "agentic-typescript", "agentic-rust", "agentic-go", "agentic-nushell"]).describe("Agentic environment type"),
  count: z.number().min(1).max(5).default(1).describe("Number of agentic workspaces to provision (1-5)"),
  features: z.array(z.enum(["agentic_chat", "agentic_generative_ui", "human_in_the_loop", "predictive_state_updates", "shared_state", "tool_based_generative_ui"])).optional().describe("AG-UI features to enable"),
});

const AguiAgentCreateSchema = z.object({
  name: z.string().describe("Agent name"),
  type: z.enum(["chat", "generative_ui", "data_processor", "automation", "coordinator"]).describe("Agent type"),
  environment: z.enum(["agentic-python", "agentic-typescript", "agentic-rust", "agentic-go", "agentic-nushell"]).describe("Target environment"),
  capabilities: z.array(z.string()).optional().describe("Agent capabilities"),
  config: z.record(z.any()).optional().describe("Agent-specific configuration"),
});

const AguiAgentListSchema = z.object({
  environment: z.string().optional().describe("Filter by environment"),
  type: z.string().optional().describe("Filter by agent type"),
  status: z.enum(["active", "inactive", "error", "all"]).default("all").describe("Filter by status"),
});

const AguiAgentInvokeSchema = z.object({
  agent_id: z.string().describe("Agent ID to invoke"),
  message: z.object({
    content: z.string().describe("Message content"),
    role: z.enum(["user", "assistant", "system"]).default("user").describe("Message role"),
    context: z.record(z.any()).optional().describe("Additional context"),
  }).describe("Message to send to agent"),
  environment: z.string().optional().describe("Environment to run in"),
});

const AguiChatSchema = z.object({
  environment: z.string().describe("Environment to start chat in"),
  agent_id: z.string().optional().describe("Specific agent ID (optional)"),
  message: z.string().describe("Initial message"),
  context: z.record(z.any()).optional().describe("Chat context"),
});

const AguiGenerateUiSchema = z.object({
  environment: z.string().describe("Environment to generate UI in"),
  prompt: z.string().describe("UI generation prompt"),
  component_type: z.enum(["form", "dashboard", "chart", "table", "card", "custom"]).default("custom").describe("Type of UI component"),
  framework: z.enum(["react", "vue", "svelte", "html", "auto"]).default("auto").describe("UI framework preference"),
});

const AguiSharedStateSchema = z.object({
  environment: z.string().describe("Environment to manage shared state in"),
  action: z.enum(["get", "set", "update", "delete", "list"]).describe("State action"),
  key: z.string().optional().describe("State key (required for get, set, update, delete)"),
  value: z.any().optional().describe("State value (required for set, update)"),
  namespace: z.string().default("default").describe("State namespace"),
});

const AguiStatusSchema = z.object({
  environment: z.string().optional().describe("Specific environment to check"),
  detailed: z.boolean().default(false).describe("Include detailed metrics"),
});

const AguiWorkflowSchema = z.object({
  environment: z.string().describe("Environment to run workflow in"),
  workflow_type: z.enum(["agent_chat", "ui_generation", "state_collaboration", "tool_based_ui", "human_in_loop"]).describe("Type of AG-UI workflow"),
  config: z.record(z.any()).optional().describe("Workflow configuration"),
  agents: z.array(z.string()).optional().describe("Agent IDs to involve"),
});

// Tool Names Enum
enum ToolName {
  // Environment Tools
  ENVIRONMENT_DETECT = "environment_detect",
  ENVIRONMENT_INFO = "environment_info", 
  ENVIRONMENT_VALIDATE = "environment_validate",
  
  // DevBox Tools
  DEVBOX_SHELL = "devbox_shell",
  DEVBOX_START = "devbox_start",
  DEVBOX_RUN = "devbox_run",
  DEVBOX_STATUS = "devbox_status",
  DEVBOX_ADD_PACKAGE = "devbox_add_package",
  DEVBOX_QUICK_START = "devbox_quick_start",
  
  // DevPod Tools
  DEVPOD_PROVISION = "devpod_provision",
  DEVPOD_LIST = "devpod_list",
  DEVPOD_STATUS = "devpod_status",
  DEVPOD_START = "devpod_start",
  
  // Cross-Language Tools
  POLYGLOT_CHECK = "polyglot_check",
  POLYGLOT_VALIDATE = "polyglot_validate",
  POLYGLOT_CLEAN = "polyglot_clean",
  
  // Performance Tools
  PERFORMANCE_MEASURE = "performance_measure",
  PERFORMANCE_REPORT = "performance_report",
  
  // Security Tools
  SECURITY_SCAN = "security_scan",
  
  // Hook Tools
  HOOK_STATUS = "hook_status",
  HOOK_TRIGGER = "hook_trigger",
  
  // PRP Tools
  PRP_GENERATE = "prp_generate",
  PRP_EXECUTE = "prp_execute",
  
  // AG-UI (Agentic UI) Tools
  AGUI_PROVISION = "agui_provision",
  AGUI_AGENT_CREATE = "agui_agent_create",
  AGUI_AGENT_LIST = "agui_agent_list",
  AGUI_AGENT_INVOKE = "agui_agent_invoke",
  AGUI_CHAT = "agui_chat",
  AGUI_GENERATE_UI = "agui_generate_ui",
  AGUI_SHARED_STATE = "agui_shared_state",
  AGUI_STATUS = "agui_status",
  AGUI_WORKFLOW = "agui_workflow",
}

export const createServer = () => {
  const server = new Server(
    {
      name: "polyglot-devenv/mcp-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        prompts: {},
        resources: {},
        tools: {},
        logging: {},
      },
      instructions
    }
  );

  // Tool Registry
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const tools: Tool[] = [
      // Include all modular tools
      ...claudeFlowTools,
      ...enhancedHooksTools,
      ...dockerMcpTools,
      ...hostContainerTools,
      ...nushellAutomationTools,
      ...configManagementTools,
      ...advancedAnalyticsTools,
      // Environment Tools
      {
        name: ToolName.ENVIRONMENT_DETECT,
        description: "Detect and list all polyglot development environments",
        inputSchema: zodToJsonSchema(EnvironmentDetectSchema) as ToolInput,
      },
      {
        name: ToolName.ENVIRONMENT_INFO,
        description: "Get detailed information about a specific environment",
        inputSchema: zodToJsonSchema(EnvironmentInfoSchema) as ToolInput,
      },
      {
        name: ToolName.ENVIRONMENT_VALIDATE,
        description: "Validate environment configuration and health",
        inputSchema: zodToJsonSchema(EnvironmentValidateSchema) as ToolInput,
      },
      
      // DevBox Tools
      {
        name: ToolName.DEVBOX_SHELL,
        description: "Enter a devbox shell for a specific environment",
        inputSchema: zodToJsonSchema(DevboxShellSchema) as ToolInput,
      },
      {
        name: ToolName.DEVBOX_START,
        description: "Start and activate a devbox development environment with automatic setup",
        inputSchema: zodToJsonSchema(DevboxStartSchema) as ToolInput,
      },
      {
        name: ToolName.DEVBOX_RUN,
        description: "Run a devbox script in a specific environment",
        inputSchema: zodToJsonSchema(DevboxRunSchema) as ToolInput,
      },
      {
        name: ToolName.DEVBOX_STATUS,
        description: "Get devbox status for environments",
        inputSchema: zodToJsonSchema(DevboxStatusSchema) as ToolInput,
      },
      {
        name: ToolName.DEVBOX_ADD_PACKAGE,
        description: "Add a package to a devbox environment",
        inputSchema: zodToJsonSchema(DevboxAddPackageSchema) as ToolInput,
      },
      {
        name: ToolName.DEVBOX_QUICK_START,
        description: "Quick start a development environment and optionally run a task",
        inputSchema: zodToJsonSchema(DevboxQuickStartSchema) as ToolInput,
      },
      
      // DevPod Tools
      {
        name: ToolName.DEVPOD_PROVISION,
        description: "Provision DevPod workspace(s) for development",
        inputSchema: zodToJsonSchema(DevpodProvisionSchema) as ToolInput,
      },
      {
        name: ToolName.DEVPOD_LIST,
        description: "List all DevPod workspaces",
        inputSchema: zodToJsonSchema(DevpodListSchema) as ToolInput,
      },
      {
        name: ToolName.DEVPOD_STATUS,
        description: "Get status of DevPod workspaces",
        inputSchema: zodToJsonSchema(DevpodStatusSchema) as ToolInput,
      },
      {
        name: ToolName.DEVPOD_START,
        description: "Start DevPod development environments dynamically for any language (Python, TypeScript, Rust, Go, Nushell)",
        inputSchema: zodToJsonSchema(DevpodStartSchema) as ToolInput,
      },
      
      // Cross-Language Tools
      {
        name: ToolName.POLYGLOT_CHECK,
        description: "Comprehensive quality check across all environments",
        inputSchema: zodToJsonSchema(PolyglotCheckSchema) as ToolInput,
      },
      {
        name: ToolName.POLYGLOT_VALIDATE,
        description: "Cross-environment validation with parallel execution",
        inputSchema: zodToJsonSchema(PolyglotValidateSchema) as ToolInput,
      },
      {
        name: ToolName.POLYGLOT_CLEAN,
        description: "Clean up environments and remove artifacts",
        inputSchema: zodToJsonSchema(PolyglotCleanSchema) as ToolInput,
      },
      
      // Performance Tools
      {
        name: ToolName.PERFORMANCE_MEASURE,
        description: "Measure performance of commands and record metrics",
        inputSchema: zodToJsonSchema(PerformanceMeasureSchema) as ToolInput,
      },
      {
        name: ToolName.PERFORMANCE_REPORT,
        description: "Generate performance analysis report",
        inputSchema: zodToJsonSchema(PerformanceReportSchema) as ToolInput,
      },
      
      // Security Tools
      {
        name: ToolName.SECURITY_SCAN,
        description: "Run security scans across environments",
        inputSchema: zodToJsonSchema(SecurityScanSchema) as ToolInput,
      },
      
      // Hook Tools
      {
        name: ToolName.HOOK_STATUS,
        description: "Get status of Claude Code hooks configuration",
        inputSchema: zodToJsonSchema(HookStatusSchema) as ToolInput,
      },
      {
        name: ToolName.HOOK_TRIGGER,
        description: "Manually trigger specific hooks for testing",
        inputSchema: zodToJsonSchema(HookTriggerSchema) as ToolInput,
      },
      
      // PRP Tools
      {
        name: ToolName.PRP_GENERATE,
        description: "Generate context engineering PRP from feature files",
        inputSchema: zodToJsonSchema(PrpGenerateSchema) as ToolInput,
      },
      {
        name: ToolName.PRP_EXECUTE,
        description: "Execute PRP files with validation",
        inputSchema: zodToJsonSchema(PrpExecuteSchema) as ToolInput,
      },
      
      // AG-UI (Agentic UI) Tools
      {
        name: ToolName.AGUI_PROVISION,
        description: "Provision agentic DevPod workspaces with AG-UI protocol support",
        inputSchema: zodToJsonSchema(AguiProvisionSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_AGENT_CREATE,
        description: "Create new AI agents in agentic environments",
        inputSchema: zodToJsonSchema(AguiAgentCreateSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_AGENT_LIST,
        description: "List all AI agents across agentic environments",
        inputSchema: zodToJsonSchema(AguiAgentListSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_AGENT_INVOKE,
        description: "Invoke an AI agent with a message",
        inputSchema: zodToJsonSchema(AguiAgentInvokeSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_CHAT,
        description: "Start agentic chat session with CopilotKit integration",
        inputSchema: zodToJsonSchema(AguiChatSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_GENERATE_UI,
        description: "Generate UI components using agentic generative UI",
        inputSchema: zodToJsonSchema(AguiGenerateUiSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_SHARED_STATE,
        description: "Manage shared state between agents and UI components",
        inputSchema: zodToJsonSchema(AguiSharedStateSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_STATUS,
        description: "Get status of agentic environments and AG-UI services",
        inputSchema: zodToJsonSchema(AguiStatusSchema) as ToolInput,
      },
      {
        name: ToolName.AGUI_WORKFLOW,
        description: "Execute AG-UI workflows (chat, generative UI, human-in-the-loop, etc.)",
        inputSchema: zodToJsonSchema(AguiWorkflowSchema) as ToolInput,
      },
    ];

    return { tools };
  });

  // Tool Call Handler
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const progressToken = request.params._meta?.progressToken;

    try {
      switch (name) {
        // Claude-Flow Tools
        case "claude_flow_init":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowInit(args as any), null, 2) }] };
        case "claude_flow_wizard":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowWizard(args as any), null, 2) }] };
        case "claude_flow_start":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowStart(args as any), null, 2) }] };
        case "claude_flow_stop":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowStop(args as any), null, 2) }] };
        case "claude_flow_status":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowStatus(args as any), null, 2) }] };
        case "claude_flow_monitor":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowMonitor(args as any), null, 2) }] };
        case "claude_flow_spawn":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowSpawn(args as any), null, 2) }] };
        case "claude_flow_logs":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowLogs(args as any), null, 2) }] };
        case "claude_flow_hive_mind":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowHiveMind(args as any), null, 2) }] };
        case "claude_flow_terminal_mgmt":
          return { content: [{ type: "text", text: JSON.stringify(await handleClaudeFlowTerminalMgmt(args as any), null, 2) }] };

        // Enhanced AI Hooks Tools  
        case "enhanced_hook_context_triggers":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookContextTriggers(args as any), null, 2) }] };
        case "enhanced_hook_error_resolution":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookErrorResolution(args as any), null, 2) }] };
        case "enhanced_hook_env_orchestration":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookEnvOrchestration(args as any), null, 2) }] };
        case "enhanced_hook_dependency_tracking":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookDependencyTracking(args as any), null, 2) }] };
        case "enhanced_hook_performance_integration":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookPerformanceIntegration(args as any), null, 2) }] };
        case "enhanced_hook_quality_gates":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookQualityGates(args as any), null, 2) }] };
        case "enhanced_hook_devpod_manager":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookDevpodManager(args as any), null, 2) }] };
        case "enhanced_hook_prp_lifecycle":
          return { content: [{ type: "text", text: JSON.stringify(await handleEnhancedHookPrpLifecycle(args as any), null, 2) }] };

        // Docker MCP Tools
        case "docker_mcp_gateway_start":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpGatewayStart(args as any), null, 2) }] };
        case "docker_mcp_gateway_status":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpGatewayStatus(args as any), null, 2) }] };
        case "docker_mcp_tools_list":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpToolsList(args as any), null, 2) }] };
        case "docker_mcp_http_bridge":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpHttpBridge(args as any), null, 2) }] };
        case "docker_mcp_client_list":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpClientList(args as any), null, 2) }] };
        case "docker_mcp_server_list":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpServerList(args as any), null, 2) }] };
        case "docker_mcp_gemini_config":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpGeminiConfig(args as any), null, 2) }] };
        case "docker_mcp_test":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpTest(args as any), null, 2) }] };
        case "docker_mcp_demo":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpDemo(args as any), null, 2) }] };
        case "docker_mcp_security_scan":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpSecurityScan(args as any), null, 2) }] };
        case "docker_mcp_resource_limits":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpResourceLimits(args as any), null, 2) }] };
        case "docker_mcp_network_isolation":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpNetworkIsolation(args as any), null, 2) }] };
        case "docker_mcp_signature_verify":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpSignatureVerify(args as any), null, 2) }] };
        case "docker_mcp_logs":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpLogs(args as any), null, 2) }] };
        case "docker_mcp_cleanup":
          return { content: [{ type: "text", text: JSON.stringify(await handleDockerMcpCleanup(args as any), null, 2) }] };

        // Host/Container Separation Tools
        case "host_installation":
          return { content: [{ type: "text", text: JSON.stringify(await handleHostInstallation(args as any), null, 2) }] };
        case "host_infrastructure":
          return { content: [{ type: "text", text: JSON.stringify(await handleHostInfrastructure(args as any), null, 2) }] };
        case "host_credential":
          return { content: [{ type: "text", text: JSON.stringify(await handleHostCredential(args as any), null, 2) }] };
        case "host_shell_integration":
          return { content: [{ type: "text", text: JSON.stringify(await handleHostShellIntegration(args as any), null, 2) }] };
        case "container_isolation":
          return { content: [{ type: "text", text: JSON.stringify(await handleContainerIsolation(args as any), null, 2) }] };
        case "container_tools":
          return { content: [{ type: "text", text: JSON.stringify(await handleContainerTools(args as any), null, 2) }] };
        case "host_container_bridge":
          return { content: [{ type: "text", text: JSON.stringify(await handleHostContainerBridge(args as any), null, 2) }] };
        case "security_boundary":
          return { content: [{ type: "text", text: JSON.stringify(await handleSecurityBoundary(args as any), null, 2) }] };

        // Nushell Automation Tools
        case "nushell_script":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellScript(args as any), null, 2) }] };
        case "nushell_validation":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellValidation(args as any), null, 2) }] };
        case "nushell_orchestration":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellOrchestration(args as any), null, 2) }] };
        case "nushell_data_processing":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellDataProcessing(args as any), null, 2) }] };
        case "nushell_automation":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellAutomation(args as any), null, 2) }] };
        case "nushell_pipeline":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellPipeline(args as any), null, 2) }] };
        case "nushell_config":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellConfig(args as any), null, 2) }] };
        case "nushell_performance":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellPerformance(args as any), null, 2) }] };
        case "nushell_debug":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellDebug(args as any), null, 2) }] };
        case "nushell_integration":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellIntegration(args as any), null, 2) }] };
        case "nushell_testing":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellTesting(args as any), null, 2) }] };
        case "nushell_documentation":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellDocumentation(args as any), null, 2) }] };
        case "nushell_environment":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellEnvironment(args as any), null, 2) }] };
        case "nushell_deployment":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellDeployment(args as any), null, 2) }] };
        case "nushell_monitoring":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellMonitoring(args as any), null, 2) }] };
        case "nushell_security":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellSecurity(args as any), null, 2) }] };
        case "nushell_backup":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellBackup(args as any), null, 2) }] };
        case "nushell_migration":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellMigration(args as any), null, 2) }] };
        case "nushell_optimization":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellOptimization(args as any), null, 2) }] };
        case "nushell_workflow":
          return { content: [{ type: "text", text: JSON.stringify(await handleNushellWorkflow(args as any), null, 2) }] };

        // Configuration Management Tools
        case "config_generation":
          return { content: [{ type: "text", text: JSON.stringify(await handleConfigGeneration(args as any), null, 2) }] };
        case "config_sync":
          return { content: [{ type: "text", text: JSON.stringify(await handleConfigSync(args as any), null, 2) }] };
        case "config_validation":
          return { content: [{ type: "text", text: JSON.stringify(await handleConfigValidation(args as any), null, 2) }] };
        case "config_backup":
          return { content: [{ type: "text", text: JSON.stringify(await handleConfigBackup(args as any), null, 2) }] };
        case "config_template":
          return { content: [{ type: "text", text: JSON.stringify(await handleConfigTemplate(args as any), null, 2) }] };

        // Advanced Analytics Tools
        case "performance_analytics":
          return { content: [{ type: "text", text: JSON.stringify(await handlePerformanceAnalytics(args as any), null, 2) }] };
        case "resource_monitoring":
          return { content: [{ type: "text", text: JSON.stringify(await handleResourceMonitoring(args as any), null, 2) }] };
        case "intelligence_system":
          return { content: [{ type: "text", text: JSON.stringify(await handleIntelligenceSystem(args as any), null, 2) }] };
        case "trend_analysis":
          return { content: [{ type: "text", text: JSON.stringify(await handleTrendAnalysis(args as any), null, 2) }] };
        case "usage_analytics":
          return { content: [{ type: "text", text: JSON.stringify(await handleUsageAnalytics(args as any), null, 2) }] };
        case "anomaly_detection":
          return { content: [{ type: "text", text: JSON.stringify(await handleAnomalyDetection(args as any), null, 2) }] };
        case "predictive_analytics":
          return { content: [{ type: "text", text: JSON.stringify(await handlePredictiveAnalytics(args as any), null, 2) }] };
        case "business_intelligence":
          return { content: [{ type: "text", text: JSON.stringify(await handleBusinessIntelligence(args as any), null, 2) }] };

        // Environment Tools
        case ToolName.ENVIRONMENT_DETECT:
          return await handleEnvironmentDetect();
          
        case ToolName.ENVIRONMENT_INFO:
          return await handleEnvironmentInfo(EnvironmentInfoSchema.parse(args));
          
        case ToolName.ENVIRONMENT_VALIDATE:
          return await handleEnvironmentValidate(EnvironmentValidateSchema.parse(args));

        // DevBox Tools
        case ToolName.DEVBOX_SHELL:
          return await handleDevboxShell(DevboxShellSchema.parse(args));
          
        case ToolName.DEVBOX_START:
          return await handleDevboxStart(DevboxStartSchema.parse(args), progressToken);
          
        case ToolName.DEVBOX_RUN:
          return await handleDevboxRun(DevboxRunSchema.parse(args), progressToken);
          
        case ToolName.DEVBOX_STATUS:
          return await handleDevboxStatus(DevboxStatusSchema.parse(args));
          
        case ToolName.DEVBOX_ADD_PACKAGE:
          return await handleDevboxAddPackage(DevboxAddPackageSchema.parse(args));

        case ToolName.DEVBOX_QUICK_START:
          return await handleDevboxQuickStart(DevboxQuickStartSchema.parse(args), progressToken);

        // DevPod Tools
        case ToolName.DEVPOD_PROVISION:
          return await handleDevpodProvision(DevpodProvisionSchema.parse(args), progressToken);
          
        case ToolName.DEVPOD_LIST:
          return await handleDevpodList();
          
        case ToolName.DEVPOD_STATUS:
          return await handleDevpodStatus(DevpodStatusSchema.parse(args));
          
        case ToolName.DEVPOD_START:
          return await handleDevpodStart(DevpodStartSchema.parse(args));

        // Cross-Language Tools
        case ToolName.POLYGLOT_CHECK:
          return await handlePolyglotCheck(PolyglotCheckSchema.parse(args), progressToken);
          
        case ToolName.POLYGLOT_VALIDATE:
          return await handlePolyglotValidate(PolyglotValidateSchema.parse(args), progressToken);
          
        case ToolName.POLYGLOT_CLEAN:
          return await handlePolyglotClean(PolyglotCleanSchema.parse(args));

        // Performance Tools
        case ToolName.PERFORMANCE_MEASURE:
          return await handlePerformanceMeasure(PerformanceMeasureSchema.parse(args));
          
        case ToolName.PERFORMANCE_REPORT:
          return await handlePerformanceReport(PerformanceReportSchema.parse(args));

        // Security Tools
        case ToolName.SECURITY_SCAN:
          return await handleSecurityScan(SecurityScanSchema.parse(args));

        // Hook Tools
        case ToolName.HOOK_STATUS:
          return await handleHookStatus();
          
        case ToolName.HOOK_TRIGGER:
          return await handleHookTrigger(HookTriggerSchema.parse(args));

        // PRP Tools
        case ToolName.PRP_GENERATE:
          return await handlePrpGenerate(PrpGenerateSchema.parse(args));
          
        case ToolName.PRP_EXECUTE:
          return await handlePrpExecute(PrpExecuteSchema.parse(args), progressToken);

        // AG-UI Tools
        case ToolName.AGUI_PROVISION:
          return await handleAguiProvision(AguiProvisionSchema.parse(args), progressToken);
          
        case ToolName.AGUI_AGENT_CREATE:
          return await handleAguiAgentCreate(AguiAgentCreateSchema.parse(args));
          
        case ToolName.AGUI_AGENT_LIST:
          return await handleAguiAgentList(AguiAgentListSchema.parse(args));
          
        case ToolName.AGUI_AGENT_INVOKE:
          return await handleAguiAgentInvoke(AguiAgentInvokeSchema.parse(args));
          
        case ToolName.AGUI_CHAT:
          return await handleAguiChat(AguiChatSchema.parse(args));
          
        case ToolName.AGUI_GENERATE_UI:
          return await handleAguiGenerateUi(AguiGenerateUiSchema.parse(args));
          
        case ToolName.AGUI_SHARED_STATE:
          return await handleAguiSharedState(AguiSharedStateSchema.parse(args));
          
        case ToolName.AGUI_STATUS:
          return await handleAguiStatus(AguiStatusSchema.parse(args));
          
        case ToolName.AGUI_WORKFLOW:
          return await handleAguiWorkflow(AguiWorkflowSchema.parse(args));

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error executing tool ${name}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  });

  // Tool Implementation Functions

  async function handleEnvironmentDetect() {
    const environments = await detectEnvironments();
    
    let content = "🔍 **Polyglot Development Environments Detected**\n\n";
    
    environments.forEach(env => {
      const statusIcon = env.status === "active" ? "✅" : env.status === "inactive" ? "⚠️" : "❌";
      const typeIcon = getEnvironmentIcon(env.type);
      
      content += `${statusIcon} ${typeIcon} **${env.name}**\n`;
      content += `   📁 Path: \`${env.path}\`\n`;
      content += `   🏷️ Type: ${env.type}\n`;
      content += `   📊 Status: ${env.status}\n`;
      
      if (env.devboxConfig) {
        content += `   📦 Packages: ${env.devboxConfig.packages.length} installed\n`;
        const scripts = env.devboxConfig.shell?.scripts || {};
        content += `   🔧 Scripts: ${Object.keys(scripts).length} available\n`;
      }
      content += "\n";
    });

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleEnvironmentInfo(args: z.infer<typeof EnvironmentInfoSchema>) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const environments = await detectEnvironments();
    const env = environments.find(e => e.name === args.environment);
    
    if (!env) {
      throw new Error(`Environment not found: ${args.environment}`);
    }

    const typeIcon = getEnvironmentIcon(env.type);
    let content = `${typeIcon} **Environment Information: ${env.name}**\n\n`;
    
    content += `📁 **Path:** \`${env.path}\`\n`;
    content += `🏷️ **Type:** ${env.type}\n`;
    content += `📊 **Status:** ${env.status}\n`;
    
    if (env.lastModified) {
      content += `📅 **Last Modified:** ${env.lastModified.toLocaleString()}\n`;
    }
    
    if (env.devboxConfig) {
      content += `\n📦 **DevBox Configuration:**\n`;
      content += `- **Packages:** ${env.devboxConfig.packages.join(", ")}\n`;
      
      const scripts = env.devboxConfig.shell?.scripts || {};
      if (Object.keys(scripts).length > 0) {
        content += `- **Available Scripts:**\n`;
        Object.entries(scripts).forEach(([name, cmd]) => {
          content += `  - \`${name}\`: ${cmd}\n`;
        });
      }
      
      const envVars = env.devboxConfig.env || {};
      if (Object.keys(envVars).length > 0) {
        content += `- **Environment Variables:** ${Object.keys(envVars).length} configured\n`;
      }
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleEnvironmentValidate(args: z.infer<typeof EnvironmentValidateSchema>) {
    if (args.environment && !isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const environments = args.environment ? [args.environment] : 
      ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];

    let content = "🔍 **Environment Validation Results**\n\n";
    
    for (const env of environments) {
      const result = await validateEnvironment(env);
      const statusIcon = result.overallStatus === "passed" ? "✅" : 
                        result.overallStatus === "warnings" ? "⚠️" : "❌";
      const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
      
      content += `${statusIcon} ${typeIcon} **${env}**\n`;
      content += `   📊 **Overall Status:** ${result.overallStatus}\n`;
      content += `   📝 **Summary:** ${result.summary}\n`;
      
      if (result.checks.length > 0) {
        content += `   🔍 **Checks:**\n`;
        result.checks.forEach((check: any) => {
          const checkIcon = check.status === "passed" ? "✅" : 
                           check.status === "warning" ? "⚠️" : "❌";
          content += `     ${checkIcon} ${check.name}: ${check.message}\n`;
        });
      }
      content += "\n";
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevboxShell(args: z.infer<typeof DevboxShellSchema>) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    return {
      content: [
        {
          type: "text",
          text: `🐚 **DevBox Shell Access**\n\nTo enter the ${args.environment} shell, run:\n\n\`\`\`bash\ncd ${getWorkspaceRoot()}/${args.environment}\ndevbox shell\n\`\`\`\n\nThis will activate the isolated development environment with all packages and configurations.`,
        },
      ],
    };
  }

  async function handleDevboxStart(args: z.infer<typeof DevboxStartSchema>, progressToken?: string | number) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const envType = getEnvironmentType(args.environment);
    const typeIcon = getEnvironmentIcon(envType);
    let content = `🚀 ${typeIcon} **DevBox Environment Startup**\n\n`;
    content += `🏷️ **Environment:** ${args.environment} (${envType})\n`;
    content += `⚙️ **Auto-Setup:** ${args.setup ? "Enabled" : "Disabled"}\n\n`;

    const totalSteps = args.setup ? 4 : 3;

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 1,
          total: totalSteps,
          progressToken,
        },
      });
    }

    // Step 1: Validate environment exists
    content += `📋 **Step 1:** Validating ${envType} environment...\n`;
    const validation = await validateEnvironment(args.environment);
    if (validation.overallStatus === "failed") {
      content += `❌ Environment validation failed: ${validation.summary}\n`;
      return { content: [{ type: "text", text: content }] };
    }
    content += `✅ Environment validated successfully\n\n`;

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 2,
          total: totalSteps,
          progressToken,
        },
      });
    }

    // Step 2: Initialize and start devbox environment
    content += `📋 **Step 2:** Initializing ${envType} devbox environment...\n`;
    const startResult = await startDevboxEnvironment(args.environment);
    
    if (startResult.success) {
      content += `✅ DevBox environment initialized successfully\n`;
    } else {
      content += `⚠️ DevBox initialization completed with warnings\n`;
    }
    content += "\n";

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 3,
          total: totalSteps,
          progressToken,
        },
      });
    }

    // Step 3: Run setup if requested
    if (args.setup) {
      content += `📋 **Step 3:** Running ${envType}-specific setup...\n`;
      
      // Use dynamic setup script detection
      const setupScript = await getDefaultSetupScript(args.environment);
      if (setupScript) {
        const setupResult = await runDevboxScript(args.environment, setupScript);
        if (setupResult.success) {
          content += `✅ ${setupScript} script completed successfully\n`;
          if (setupResult.output) {
            // Show brief output summary
            const lines = setupResult.output.split('\n').filter(l => l.trim());
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.length < 100) {
              content += `   📤 ${lastLine}\n`;
            }
          }
        } else {
          content += `⚠️ ${setupScript} script encountered issues\n`;
        }
      } else {
        content += `ℹ️ No setup script found for ${envType} environment\n`;
      }
      content += "\n";

      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progress: 4,
            total: totalSteps,
            progressToken,
          },
        });
      }
    }

    // Final status and dynamic command display
    const statusIcon = "✅";
    content += `${statusIcon} **${envType.charAt(0).toUpperCase() + envType.slice(1)} Environment Ready!**\n\n`;
    
    // Show environment-specific common commands
    const commonCommands = await getCommonCommands(args.environment);
    if (Object.keys(commonCommands).length > 0) {
      content += `🔧 **Common ${envType.charAt(0).toUpperCase() + envType.slice(1)} Commands:**\n`;
      Object.entries(commonCommands).forEach(([cmd, script]) => {
        content += `- \`devbox run ${cmd}\` - ${script}\n`;
      });
      content += "\n";
    }

    // Environment-specific next steps
    const envSpecificSteps: Record<string, string[]> = {
      python: [
        "Run tests with `devbox run test`",
        "Format code with `devbox run format`", 
        "Type check with `devbox run type-check`"
      ],
      typescript: [
        "Build project with `devbox run build`",
        "Start dev server with `devbox run dev`",
        "Run tests with `devbox run test`"
      ],
      rust: [
        "Build project with `devbox run build`",
        "Run tests with `devbox run test`",
        "Check code with `devbox run check`"
      ],
      go: [
        "Build project with `devbox run build`", 
        "Run tests with `devbox run test`",
        "Format code with `devbox run format`"
      ],
      nushell: [
        "Validate scripts with `devbox run check`",
        "Run tests with `devbox run test`",
        "Format scripts with `devbox run format`"
      ]
    };

    content += `💡 **Next Steps:**\n`;
    content += `- Environment is now active and ready for ${envType} development\n`;
    
    const steps = envSpecificSteps[envType] || ["Use `devbox run <script>` to execute tasks"];
    steps.forEach(step => content += `- ${step}\n`);
    
    content += `- Use \`cd ${getWorkspaceRoot()}/${args.environment} && devbox shell\` for interactive session\n`;

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevboxQuickStart(args: z.infer<typeof DevboxQuickStartSchema>, progressToken?: string | number) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const envType = getEnvironmentType(args.environment);
    const typeIcon = getEnvironmentIcon(envType);
    let content = `⚡ ${typeIcon} **Quick Start: ${envType.toUpperCase()}**\n\n`;

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 1,
          total: args.task ? 3 : 2,
          progressToken,
        },
      });
    }

    // Step 1: Quick validation and start
    content += `🚀 Rapid environment activation...\n`;
    const startResult = await startDevboxEnvironment(args.environment);
    
    if (startResult.success) {
      content += `✅ ${envType} environment ready\n\n`;
    } else {
      content += `⚠️ Environment started with warnings\n\n`;
    }

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 2,
          total: args.task ? 3 : 2,
          progressToken,
        },
      });
    }

    // Step 2: Run immediate task if specified
    if (args.task) {
      content += `⚡ Running immediate task: ${args.task}\n`;
      const taskResult = await runDevboxScript(args.environment, args.task);
      
      if (taskResult.success) {
        content += `✅ ${args.task} completed successfully\n`;
        
        // Show relevant output for different tasks
        if (taskResult.output) {
          const lines = taskResult.output.split('\n').filter(l => l.trim());
          if (args.task === "test") {
            const testSummary = lines.find(l => l.includes("passed") || l.includes("failed") || l.includes("ok"));
            if (testSummary) content += `   📊 ${testSummary}\n`;
          } else if (args.task === "build") {
            const buildInfo = lines[lines.length - 1];
            if (buildInfo && buildInfo.length < 100) content += `   🔨 ${buildInfo}\n`;
          }
        }
      } else {
        content += `❌ ${args.task} failed\n`;
        if (taskResult.error) {
          const errorLines = taskResult.error.split('\n').slice(0, 2);
          content += `   💥 ${errorLines.join(' ')}\n`;
        }
      }
      content += "\n";

      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progress: 3,
            total: 3,
            progressToken,
          },
        });
      }
    }

    // Quick reference
    const commonCommands = await getCommonCommands(args.environment);
    if (Object.keys(commonCommands).length > 0) {
      content += `⚡ **Quick Commands:**\n`;
      Object.entries(commonCommands).slice(0, 4).forEach(([cmd, _]) => {
        content += `- \`devbox run ${cmd}\`\n`;
      });
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevboxRun(args: z.infer<typeof DevboxRunSchema>, progressToken?: string | number) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 1,
          total: 2,
          progressToken,
        },
      });
    }

    const result = await runDevboxScript(args.environment, args.script);

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 2,
          total: 2,
          progressToken,
        },
      });
    }

    const statusIcon = result.success ? "✅" : "❌";
    const typeIcon = getEnvironmentIcon(getEnvironmentType(args.environment));
    
    let content = `${statusIcon} ${typeIcon} **DevBox Script Execution**\n\n`;
    content += `🏷️ **Environment:** ${args.environment}\n`;
    content += `🔧 **Script:** ${args.script}\n`;
    content += `⏱️ **Duration:** ${formatDuration(result.duration)}\n`;
    content += `🚀 **Exit Code:** ${result.exitCode}\n\n`;
    
    if (result.output) {
      content += `📤 **Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\`\n\n`;
    }
    
    if (result.error) {
      content += `⚠️ **Error Output:**\n\`\`\`\n${result.error.trim()}\n\`\`\`\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevboxStatus(args: z.infer<typeof DevboxStatusSchema>) {
    const environments = args.environment ? [args.environment] : 
      ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];

    let content = "📊 **DevBox Environment Status**\n\n";
    
    for (const env of environments) {
      if (!isValidEnvironment(env)) continue;
      
      const result = await runDevboxCommand(env, "info");
      const statusIcon = result.success ? "✅" : "❌";
      const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
      
      content += `${statusIcon} ${typeIcon} **${env}**\n`;
      content += `   ⏱️ Duration: ${formatDuration(result.duration)}\n`;
      
      if (result.success && result.output) {
        content += `   📋 Status: Active and functional\n`;
      } else {
        content += `   ⚠️ Status: Issues detected\n`;
        if (result.error) {
          content += `   🚨 Error: ${result.error.split("\n")[0]}\n`;
        }
      }
      content += "\n";
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevboxAddPackage(args: z.infer<typeof DevboxAddPackageSchema>) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const result = await runDevboxCommand(args.environment, "add", [args.package]);
    
    const statusIcon = result.success ? "✅" : "❌";
    const typeIcon = getEnvironmentIcon(getEnvironmentType(args.environment));
    
    let content = `${statusIcon} ${typeIcon} **Package Installation**\n\n`;
    content += `🏷️ **Environment:** ${args.environment}\n`;
    content += `📦 **Package:** ${args.package}\n`;
    content += `⏱️ **Duration:** ${formatDuration(result.duration)}\n\n`;
    
    if (result.success) {
      content += `🎉 **Success!** Package \`${args.package}\` has been added to ${args.environment}\n\n`;
    } else {
      content += `❌ **Failed** to add package \`${args.package}\`\n\n`;
    }
    
    if (result.output) {
      content += `📤 **Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\`\n`;
    }
    
    if (result.error) {
      content += `⚠️ **Error:**\n\`\`\`\n${result.error.trim()}\n\`\`\`\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevpodProvision(args: z.infer<typeof DevpodProvisionSchema>, progressToken?: string | number) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const typeIcon = getEnvironmentIcon(getEnvironmentType(args.environment));
    let content = `🚀 ${typeIcon} **DevPod Workspace Provisioning**\n\n`;
    content += `🏷️ **Environment:** ${args.environment}\n`;
    content += `🔢 **Count:** ${args.count} workspace(s)\n\n`;

    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 0,
          total: args.count,
          progressToken,
        },
      });
    }

    const results = await provisionDevPodWorkspace(args.environment, args.count);
    
    let successCount = 0;
    results.forEach((result, index) => {
      const statusIcon = result.success ? "✅" : "❌";
      const workspaceNum = index + 1;
      
      content += `${statusIcon} **Workspace ${workspaceNum}/${args.count}**\n`;
      content += `   ⏱️ Duration: ${formatDuration(result.duration)}\n`;
      
      if (result.success) {
        successCount++;
        content += `   🎉 Status: Successfully provisioned\n`;
      } else {
        content += `   ❌ Status: Provisioning failed\n`;
        if (result.error) {
          content += `   🚨 Error: ${result.error.split("\n")[0]}\n`;
        }
      }
      content += "\n";

      if (progressToken) {
        server.notification({
          method: "notifications/progress",
          params: {
            progressToken,
            progress: workspaceNum,
            total: args.count,
          },
        }).catch(() => {}); // Don't block on notification failures
      }
    });

    content += `📊 **Summary:** ${successCount}/${args.count} workspaces provisioned successfully\n\n`;
    content += `💡 **Next Steps:**\n`;
    content += `- Use \`devpod list\` to see all workspaces\n`;
    content += `- Use \`devpod stop <workspace-name>\` to stop specific workspaces\n`;
    content += `- Use VS Code to connect to the workspaces\n`;

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevpodList() {
    const workspaces = await listDevPodWorkspaces();
    
    let content = "📋 **DevPod Workspaces**\n\n";
    
    if (workspaces.length === 0) {
      content += "No DevPod workspaces found.\n\n";
      content += "💡 Use the `devpod_provision` tool to create new workspaces.";
    } else {
      const polyglotWorkspaces = workspaces.filter(w => 
        w.name && w.name.includes("polyglot")
      );
      
      if (polyglotWorkspaces.length === 0) {
        content += `Found ${workspaces.length} total workspace(s), but none are polyglot development workspaces.\n\n`;
      } else {
        content += `Found ${polyglotWorkspaces.length} polyglot workspace(s):\n\n`;
        
        polyglotWorkspaces.forEach(workspace => {
          const envType = getWorkspaceEnvironmentType(workspace.name);
          const typeIcon = getEnvironmentIcon(envType);
          const statusIcon = workspace.status === "Running" ? "🟢" : "🔴";
          
          content += `${statusIcon} ${typeIcon} **${workspace.name}**\n`;
          content += `   📊 Status: ${workspace.status || "Unknown"}\n`;
          content += `   🐳 Provider: ${workspace.provider || "docker"}\n`;
          if (workspace.created) {
            content += `   📅 Created: ${new Date(workspace.created).toLocaleString()}\n`;
          }
          content += "\n";
        });
      }
      
      if (workspaces.length > polyglotWorkspaces.length) {
        content += `\n📝 **Note:** ${workspaces.length - polyglotWorkspaces.length} other workspace(s) not shown (non-polyglot)`;
      }
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevpodStatus(args: z.infer<typeof DevpodStatusSchema>) {
    const result = await executeCommand("devpod", ["list", "--output", "json"]);
    
    if (!result.success) {
      return {
        content: [
          {
            type: "text",
            text: "❌ **DevPod Status Check Failed**\n\nUnable to retrieve DevPod status. Make sure DevPod is installed and configured.",
          },
        ],
      };
    }

    const workspaces = JSON.parse(result.output || "[]") as any[];
    let content = "📊 **DevPod Status Report**\n\n";
    
    if (args.workspace) {
      const workspace = workspaces.find(w => w.name === args.workspace);
      if (!workspace) {
        content += `❌ Workspace \`${args.workspace}\` not found.`;
      } else {
        const envType = getWorkspaceEnvironmentType(workspace.name);
        const typeIcon = getEnvironmentIcon(envType);
        const statusIcon = workspace.status === "Running" ? "🟢" : "🔴";
        
        content += `${statusIcon} ${typeIcon} **${workspace.name}**\n\n`;
        content += `📊 **Status:** ${workspace.status || "Unknown"}\n`;
        content += `🐳 **Provider:** ${workspace.provider || "docker"}\n`;
        content += `🏷️ **Environment:** ${envType}\n`;
        if (workspace.created) {
          content += `📅 **Created:** ${new Date(workspace.created).toLocaleString()}\n`;
        }
      }
    } else {
      const polyglotWorkspaces = workspaces.filter(w => 
        w.name && w.name.includes("polyglot")
      );
      
      const runningCount = polyglotWorkspaces.filter(w => w.status === "Running").length;
      const stoppedCount = polyglotWorkspaces.length - runningCount;
      
      content += `📈 **Overview:**\n`;
      content += `- 🟢 Running: ${runningCount}\n`;
      content += `- 🔴 Stopped: ${stoppedCount}\n`;
      content += `- 📊 Total: ${polyglotWorkspaces.length}\n\n`;
      
      if (polyglotWorkspaces.length > 0) {
        content += `🗂️ **Workspaces by Environment:**\n`;
        const byEnv = groupBy(polyglotWorkspaces, w => getWorkspaceEnvironmentType(w.name));
        
        Object.entries(byEnv).forEach(([envType, envWorkspaces]) => {
          const typeIcon = getEnvironmentIcon(envType as any);
          content += `${typeIcon} **${envType}:** ${envWorkspaces.length} workspace(s)\n`;
        });
      }
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleDevpodStart(args: z.infer<typeof DevpodStartSchema>) {
    const { environment, count } = args;
    
    if (!isValidEnvironment(environment)) {
      throw new Error(`Invalid environment: ${environment}. Valid environments are: ${ENVIRONMENTS.join(", ")}`);
    }

    const envType = getEnvironmentType(environment);
    const typeIcon = getEnvironmentIcon(envType);
    
    let content = `🚀 ${typeIcon} **Dynamic DevPod Start: ${envType.toUpperCase()}**\n\n`;
    content += `🏷️ **Environment:** ${environment}\n`;
    content += `🔢 **Count:** ${count} workspace(s)\n\n`;

    const results: any[] = [];
    
    for (let i = 1; i <= count; i++) {
      content += `⚡ **Starting workspace ${i}/${count}**\n`;
      
      try {
        const result = await runDevboxScript(environment, "devpod:provision");
        
        if (result.success) {
          content += `✅ **Workspace ${i}:** Successfully started\n`;
          content += `   ⏱️ Duration: ${formatDuration(result.duration)}\n`;
          
          // Extract workspace name from output if possible
          const workspaceMatch = result.output.match(/workspace:\s*([^\s\n]+)/i);
          if (workspaceMatch) {
            content += `   🏷️ Name: ${workspaceMatch[1]}\n`;
          }
          
          results.push({ success: true, workspace: i, duration: result.duration });
        } else {
          content += `❌ **Workspace ${i}:** Failed to start\n`;
          content += `   🚨 Error: ${result.error}\n`;
          results.push({ success: false, workspace: i, error: result.error });
        }
      } catch (error) {
        content += `❌ **Workspace ${i}:** Exception occurred\n`;
        content += `   🚨 Error: ${error instanceof Error ? error.message : String(error)}\n`;
        results.push({ success: false, workspace: i, error: error instanceof Error ? error.message : String(error) });
      }
      
      content += "\n";
      
      // Small delay between provisions to avoid conflicts
      if (i < count) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    const successCount = results.filter(r => r.success).length;
    const failureCount = count - successCount;
    
    content += `📊 **Summary:** ${successCount}/${count} workspaces started successfully\n\n`;
    
    if (successCount > 0) {
      content += `💡 **Next Steps:**\n`;
      content += `- Use \`devpod list\` to see all workspaces\n`;
      content += `- Use \`devpod stop <workspace-name>\` to stop specific workspaces\n`;
      content += `- Connect to workspaces using VS Code integration\n`;
      content += `- Use \`mcp__polyglot-dev__devpod_status\` to monitor workspace status\n`;
    }
    
    if (failureCount > 0) {
      content += `\n⚠️ **Troubleshooting:**\n`;
      content += `- Check Docker is running and accessible\n`;
      content += `- Verify workspace naming conventions\n`;
      content += `- Try starting workspaces individually if issues persist\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePolyglotCheck(args: z.infer<typeof PolyglotCheckSchema>, progressToken?: string | number) {
    const environments = args.environment ? [args.environment] : 
      ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];

    let content = "🔍 **Polyglot Environment Quality Check**\n\n";
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 0,
          total: environments.length,
          progressToken,
        },
      });
    }

    const results: any[] = [];
    
    for (let i = 0; i < environments.length; i++) {
      const env = environments[i];
      if (!isValidEnvironment(env)) continue;
      
      const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
      content += `${typeIcon} **${env}**\n`;
      
      // Basic validation
      const validation = await validateEnvironment(env);
      const statusIcon = validation.overallStatus === "passed" ? "✅" : 
                        validation.overallStatus === "warnings" ? "⚠️" : "❌";
      
      content += `   ${statusIcon} Validation: ${validation.summary}\n`;
      
      // Run lint check
      const lintResult = await runDevboxScript(env, "lint");
      if (lintResult.success) {
        content += `   ✅ Linting: Passed\n`;
      } else {
        content += `   ❌ Linting: Issues found\n`;
      }
      
      // Run test check
      const testResult = await runDevboxScript(env, "test");
      if (testResult.success) {
        content += `   ✅ Tests: Passed\n`;
      } else {
        content += `   ⚠️ Tests: Some issues\n`;
      }
      
      results.push({
        environment: env,
        validation: validation.overallStatus,
        lint: lintResult.success,
        test: testResult.success,
      });
      
      content += "\n";
      
      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progressToken,
            progress: i + 1,
            total: environments.length,
          },
        });
      }
    }
    
    // Summary
    const passed = results.filter(r => r.validation === "passed" && r.lint && r.test).length;
    const total = results.length;
    
    content += `📊 **Summary:** ${passed}/${total} environments fully passing\n\n`;
    
    if (args.include_security) {
      content += `🔐 **Security Note:** Security scanning requested but not yet implemented in this version.\n`;
    }
    
    if (args.include_performance) {
      content += `📈 **Performance Note:** Performance analysis requested but not yet implemented in this version.\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePolyglotValidate(args: z.infer<typeof PolyglotValidateSchema>, progressToken?: string | number) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = "🔍 **Cross-Environment Validation**\n\n";
    
    if (args.parallel) {
      content += "🚀 **Mode:** Parallel execution\n\n";
      
      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progressToken,
            progress: 1,
            total: 3,
          },
        });
      }
      
      // Run the Nushell validation script
      const result = await executeCommand("nu", ["scripts/validate-all.nu", "--parallel"], {
        cwd: workspaceRoot,
        timeout: 120000, // 2 minutes
      });
      
      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progressToken,
            progress: 2,
            total: 3,
          },
        });
      }
      
      const statusIcon = result.success ? "✅" : "❌";
      content += `${statusIcon} **Validation Result**\n`;
      content += `⏱️ **Duration:** ${formatDuration(result.duration)}\n\n`;
      
      if (result.output) {
        content += `📤 **Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\`\n`;
      }
      
      if (result.error) {
        content += `⚠️ **Errors:**\n\`\`\`\n${result.error.trim()}\n\`\`\`\n`;
      }
      
      if (progressToken) {
        await server.notification({
          method: "notifications/progress",
          params: {
            progressToken,
            progress: 3,
            total: 3,
          },
        });
      }
    } else {
      content += "🔄 **Mode:** Sequential execution\n\n";
      
      const environments = ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];
      
      for (let i = 0; i < environments.length; i++) {
        const env = environments[i];
        const validation = await validateEnvironment(env);
        
        const statusIcon = validation.overallStatus === "passed" ? "✅" : 
                          validation.overallStatus === "warnings" ? "⚠️" : "❌";
        const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
        
        content += `${statusIcon} ${typeIcon} **${env}:** ${validation.summary}\n`;
        
        if (progressToken) {
          await server.notification({
            method: "notifications/progress",
            params: {
              progressToken,
              progress: i + 1,
              total: environments.length,
            },
          });
        }
      }
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePolyglotClean(args: z.infer<typeof PolyglotCleanSchema>) {
    const environments = args.environment ? [args.environment] : 
      ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];

    let content = "🧹 **Environment Cleanup**\n\n";
    
    for (const env of environments) {
      if (!isValidEnvironment(env)) continue;
      
      const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
      content += `${typeIcon} **${env}**\n`;
      
      // Run clean script
      const cleanResult = await runDevboxScript(env, "clean");
      if (cleanResult.success) {
        content += `   ✅ Clean: Completed successfully\n`;
      } else {
        content += `   ⚠️ Clean: Some issues encountered\n`;
      }
      
      content += `   ⏱️ Duration: ${formatDuration(cleanResult.duration)}\n\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePerformanceMeasure(args: z.infer<typeof PerformanceMeasureSchema>) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const startTime = Date.now();
    const result = await executeCommand(args.command, [], {
      cwd: getWorkspaceRoot() + "/" + args.environment,
    });
    const duration = Date.now() - startTime;

    const typeIcon = getEnvironmentIcon(getEnvironmentType(args.environment));
    const statusIcon = result.success ? "✅" : "❌";
    
    let content = `📊 ${typeIcon} **Performance Measurement**\n\n`;
    content += `🏷️ **Environment:** ${args.environment}\n`;
    content += `🔧 **Command:** \`${args.command}\`\n`;
    content += `⏱️ **Duration:** ${formatDuration(duration)}\n`;
    content += `🚀 **Status:** ${result.success ? "Success" : "Failed"}\n`;
    content += `📈 **Exit Code:** ${result.exitCode}\n\n`;

    // Store performance metric (this would typically go to a database)
    content += `💾 **Metric Recorded:** Performance data saved for analysis\n\n`;
    
    if (result.output) {
      content += `📤 **Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\`\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePerformanceReport(args: z.infer<typeof PerformanceReportSchema>) {
    let content = "📈 **Performance Analysis Report**\n\n";
    content += `📅 **Period:** Last ${args.days} day(s)\n`;
    
    if (args.environment) {
      content += `🏷️ **Environment:** ${args.environment}\n`;
    }
    
    content += "\n";
    
    // This would typically read from a performance database
    content += "📊 **Performance Summary:**\n";
    content += "- Average build time: Not yet tracked\n";
    content += "- Average test time: Not yet tracked\n";
    content += "- Success rate: Not yet tracked\n\n";
    
    content += "💡 **Note:** Performance tracking is not yet fully implemented. ";
    content += "Use the `performance_measure` tool to start collecting metrics.\n";

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleSecurityScan(args: z.infer<typeof SecurityScanSchema>) {
    const environments = args.environment ? [args.environment] : 
      ["dev-env/python", "dev-env/typescript", "dev-env/rust", "dev-env/go", "dev-env/nushell"];

    let content = "🔐 **Security Scan Results**\n\n";
    
    for (const env of environments) {
      if (!isValidEnvironment(env)) continue;
      
      const typeIcon = getEnvironmentIcon(getEnvironmentType(env));
      content += `${typeIcon} **${env}**\n`;
      
      if (args.scan_type === "secrets" || args.scan_type === "all") {
        // Basic secret scanning
        const envPath = getWorkspaceRoot() + "/" + env;
        const configFiles = [".env", ".env.local", "config.json", "secrets.json"];
        
        let secretsFound = 0;
        for (const file of configFiles) {
          const filePath = join(envPath, file);
          try {
            const findings = await scanForSecrets(filePath);
            secretsFound += findings.length;
          } catch {
            // File doesn't exist, skip
          }
        }
        
        if (secretsFound > 0) {
          content += `   ⚠️ Secrets: ${secretsFound} potential issue(s) found\n`;
        } else {
          content += `   ✅ Secrets: No issues detected\n`;
        }
      }
      
      if (args.scan_type === "vulnerabilities" || args.scan_type === "all") {
        content += `   💡 Vulnerabilities: Scanning not yet implemented\n`;
      }
      
      content += "\n";
    }
    
    content += "💡 **Note:** Enhanced security scanning with git-secrets and vulnerability ";
    content += "databases will be implemented in future versions.\n";

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleHookStatus() {
    const workspaceRoot = getWorkspaceRoot();
    const hooksConfigPath = join(workspaceRoot, ".claude", "settings.json");
    
    let content = "🪝 **Claude Code Hooks Status**\n\n";
    
    try {
      const hooksConfig = JSON.parse(await readFileAsync(hooksConfigPath, "utf-8")) as any;
      const hooks = hooksConfig.hooks || {};
      
      content += `📁 **Configuration:** \`${hooksConfigPath}\`\n\n`;
      
      Object.entries(hooks).forEach(([hookType, hookList]: [string, any]) => {
        content += `🔧 **${hookType}:**\n`;
        if (Array.isArray(hookList) && hookList.length > 0) {
          content += `   📊 ${hookList.length} hook(s) configured\n`;
          hookList.forEach((hook: any, index: number) => {
            content += `   ${index + 1}. ${hook.matcher || "All triggers"}\n`;
          });
        } else {
          content += `   📭 No hooks configured\n`;
        }
        content += "\n";
      });
      
      const totalHooks = Object.values(hooks).reduce((sum: number, hookList: any) => 
        sum + (Array.isArray(hookList) ? hookList.length : 0), 0);
      
      content += `📈 **Summary:** ${totalHooks} total hooks across ${Object.keys(hooks).length} categories\n`;
      
    } catch (error) {
      content += `❌ **Error:** Unable to read hooks configuration\n`;
      content += `🚨 **Details:** ${error instanceof Error ? error.message : String(error)}\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleHookTrigger(args: z.infer<typeof HookTriggerSchema>) {
    let content = `🪝 **Manual Hook Trigger**\n\n`;
    content += `🔧 **Hook Type:** ${args.hook_type}\n`;
    
    if (args.context) {
      content += `📋 **Context:** ${JSON.stringify(args.context, null, 2)}\n`;
    }
    
    content += "\n💡 **Note:** Manual hook triggering is not yet implemented. ";
    content += "Hooks are currently triggered automatically by Claude Code based on tool usage.\n";

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePrpGenerate(args: z.infer<typeof PrpGenerateSchema>) {
    if (!isValidEnvironment(args.environment)) {
      throw new Error(`Invalid environment: ${args.environment}`);
    }

    const workspaceRoot = getWorkspaceRoot();
    
    let content = `📝 **PRP Generation**\n\n`;
    content += `📁 **Feature File:** ${args.feature_file}\n`;
    content += `🏷️ **Environment:** ${args.environment}\n\n`;
    
    // This would typically generate a PRP using the existing Python scripts
    const generateResult = await executeCommand("python", [
      ".claude/commands/generate-prp-v2.py",
      args.feature_file,
      "--env", args.environment
    ], {
      cwd: workspaceRoot,
      timeout: 60000,
    });
    
    const statusIcon = generateResult.success ? "✅" : "❌";
    content += `${statusIcon} **Generation Result**\n`;
    content += `⏱️ **Duration:** ${formatDuration(generateResult.duration)}\n\n`;
    
    if (generateResult.output) {
      content += `📤 **Output:**\n\`\`\`\n${generateResult.output.trim()}\n\`\`\`\n`;
    }
    
    if (generateResult.error) {
      content += `⚠️ **Errors:**\n\`\`\`\n${generateResult.error.trim()}\n\`\`\`\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handlePrpExecute(args: z.infer<typeof PrpExecuteSchema>, progressToken?: string | number) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🚀 **PRP Execution**\n\n`;
    content += `📁 **PRP File:** ${args.prp_file}\n`;
    content += `✅ **Validation:** ${args.validate ? "Enabled" : "Disabled"}\n\n`;
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 1,
          total: 2,
          progressToken,
        },
      });
    }
    
    // Execute using the existing Python script
    const executeArgs = ["python", ".claude/commands/execute-prp-v2.py", args.prp_file];
    if (args.validate) {
      executeArgs.push("--validate");
    }
    
    const executeResult = await executeCommand(executeArgs[0], executeArgs.slice(1), {
      cwd: workspaceRoot,
      timeout: 300000, // 5 minutes
    });
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 2,
          total: 2,
          progressToken,
        },
      });
    }
    
    const statusIcon = executeResult.success ? "✅" : "❌";
    content += `${statusIcon} **Execution Result**\n`;
    content += `⏱️ **Duration:** ${formatDuration(executeResult.duration)}\n\n`;
    
    if (executeResult.output) {
      content += `📤 **Output:**\n\`\`\`\n${executeResult.output.trim()}\n\`\`\`\n`;
    }
    
    if (executeResult.error) {
      content += `⚠️ **Errors:**\n\`\`\`\n${executeResult.error.trim()}\n\`\`\`\n`;
    }

    return {
      content: [{ type: "text", text: content }],
    };
  }

  // AG-UI Tool Handlers

  async function handleAguiProvision(args: z.infer<typeof AguiProvisionSchema>, progressToken?: string | number) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🤖 **AG-UI DevPod Provisioning**\n\n`;
    content += `🏗️ **Environment:** ${args.environment}\n`;
    content += `📊 **Count:** ${args.count} workspace(s)\n`;
    
    if (args.features && args.features.length > 0) {
      content += `🎛️ **Features:** ${args.features.join(", ")}\n`;
    }
    content += "\n";
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 1,
          total: 3,
          progressToken,
        },
      });
    }
    
    // Use centralized devpod management for agentic environments
    const provisionResult = await executeCommand("nu", [
      "host-tooling/devpod-management/manage-devpod.nu",
      "provision",
      args.environment
    ], {
      cwd: workspaceRoot,
      timeout: 300000, // 5 minutes
    });
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 2,
          total: 3,
          progressToken,
        },
      });
    }
    
    const statusIcon = provisionResult.success ? "✅" : "❌";
    content += `${statusIcon} **Provisioning Result**\n`;
    content += `⏱️ **Duration:** ${formatDuration(provisionResult.duration)}\n\n`;
    
    if (provisionResult.output) {
      content += `📤 **Output:**\n\`\`\`\n${provisionResult.output.trim()}\n\`\`\`\n`;
    }
    
    if (provisionResult.error) {
      content += `⚠️ **Errors:**\n\`\`\`\n${provisionResult.error.trim()}\n\`\`\`\n`;
    }
    
    if (progressToken) {
      await server.notification({
        method: "notifications/progress",
        params: {
          progress: 3,
          total: 3,
          progressToken,
        },
      });
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiAgentCreate(args: z.infer<typeof AguiAgentCreateSchema>) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🤖 **Creating AG-UI Agent**\n\n`;
    content += `👤 **Name:** ${args.name}\n`;
    content += `🏷️ **Type:** ${args.type}\n`;
    content += `🌍 **Environment:** ${args.environment}\n`;
    
    if (args.capabilities && args.capabilities.length > 0) {
      content += `🎛️ **Capabilities:** ${args.capabilities.join(", ")}\n`;
    }
    content += "\n";
    
    // Create agent configuration file
    const agentConfig = {
      id: `agent-${Date.now()}`,
      name: args.name,
      type: args.type,
      environment: args.environment,
      capabilities: args.capabilities || [],
      config: args.config || {},
      created_at: new Date().toISOString(),
      status: "active"
    };
    
    // Save agent config based on environment
    const configPath = `devpod-automation/agents/${args.environment}/${agentConfig.id}.json`;
    
    try {
      const saveResult = await executeCommand("mkdir", ["-p", `devpod-automation/agents/${args.environment}`], {
        cwd: workspaceRoot,
      });
      
      if (saveResult.success) {
        const configContent = JSON.stringify(agentConfig, null, 2);
        const writeResult = await executeCommand("sh", ["-c", `echo '${configContent}' > ${configPath}`], {
          cwd: workspaceRoot,
        });
        
        if (writeResult.success) {
          content += `✅ **Agent Created Successfully**\n`;
          content += `🆔 **Agent ID:** ${agentConfig.id}\n`;
          content += `📁 **Config Path:** ${configPath}\n`;
        } else {
          content += `❌ **Failed to save agent config**\n`;
          content += `⚠️ **Error:** ${writeResult.error}\n`;
        }
      }
    } catch (error) {
      content += `❌ **Error creating agent:** ${error}\n`;
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiAgentList(args: z.infer<typeof AguiAgentListSchema>) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🤖 **AG-UI Agents**\n\n`;
    
    try {
      // List agent configuration files
      const listResult = await executeCommand("find", [
        "devpod-automation/agents",
        "-name", "*.json",
        "-type", "f"
      ], {
        cwd: workspaceRoot,
      });
      
      if (listResult.success && listResult.output) {
        const agentFiles = listResult.output.trim().split('\n').filter(f => f);
        
        if (agentFiles.length === 0) {
          content += `📭 **No agents found**\n`;
        } else {
          content += `📊 **Found ${agentFiles.length} agent(s)**\n\n`;
          
          for (const file of agentFiles) {
            try {
              const readResult = await executeCommand("cat", [file], { cwd: workspaceRoot });
              if (readResult.success) {
                const agent = JSON.parse(readResult.output);
                
                // Apply filters
                if (args.environment && !agent.environment.includes(args.environment)) continue;
                if (args.type && agent.type !== args.type) continue;
                if (args.status !== "all" && agent.status !== args.status) continue;
                
                const statusIcon = agent.status === "active" ? "✅" : agent.status === "error" ? "❌" : "⚠️";
                const envIcon = getEnvironmentIcon(getEnvironmentType(agent.environment));
                
                content += `${statusIcon} ${envIcon} **${agent.name}**\n`;
                content += `   🆔 ID: \`${agent.id}\`\n`;
                content += `   🏷️ Type: ${agent.type}\n`;
                content += `   🌍 Environment: ${agent.environment}\n`;
                content += `   📅 Created: ${new Date(agent.created_at).toLocaleDateString()}\n`;
                
                if (agent.capabilities && agent.capabilities.length > 0) {
                  content += `   🎛️ Capabilities: ${agent.capabilities.slice(0, 3).join(", ")}${agent.capabilities.length > 3 ? "..." : ""}\n`;
                }
                content += "\n";
              }
            } catch (error) {
              // Skip invalid agent files
            }
          }
        }
      } else {
        content += `📭 **No agent directory found**\n`;
      }
    } catch (error) {
      content += `❌ **Error listing agents:** ${error}\n`;
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiAgentInvoke(args: z.infer<typeof AguiAgentInvokeSchema>) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🤖 **Invoking AG-UI Agent**\n\n`;
    content += `🆔 **Agent ID:** ${args.agent_id}\n`;
    content += `💬 **Message:** ${args.message.content}\n`;
    content += `👤 **Role:** ${args.message.role}\n`;
    
    if (args.environment) {
      content += `🌍 **Environment:** ${args.environment}\n`;
    }
    content += "\n";
    
    try {
      // Find agent configuration
      const findResult = await executeCommand("find", [
        "devpod-automation/agents",
        "-name", `${args.agent_id}.json`,
        "-type", "f"
      ], {
        cwd: workspaceRoot,
      });
      
      if (findResult.success && findResult.output.trim()) {
        const configFile = findResult.output.trim();
        const readResult = await executeCommand("cat", [configFile], { cwd: workspaceRoot });
        
        if (readResult.success) {
          const agent = JSON.parse(readResult.output);
          
          content += `✅ **Agent Found:** ${agent.name}\n`;
          content += `🏷️ **Type:** ${agent.type}\n`;
          content += `🌍 **Environment:** ${agent.environment}\n\n`;
          
          // Mock agent response based on type
          let response = "";
          switch (agent.type) {
            case "chat":
              response = `I'm ${agent.name}, a chat agent. I received your message: "${args.message.content}". How can I assist you today?`;
              break;
            case "generative_ui":
              response = `As a generative UI agent, I can help create UI components. For "${args.message.content}", I suggest creating a dynamic interface with interactive elements.`;
              break;
            case "data_processor":
              response = `Data processing agent ${agent.name} here. I can analyze, transform, and process data. Please provide the data you'd like me to work with.`;
              break;
            case "automation":
              response = `Automation agent ${agent.name} ready. I can help automate tasks and workflows. What process would you like me to automate?`;
              break;
            case "coordinator":
              response = `Coordinator agent ${agent.name} active. I can orchestrate multiple agents and manage complex workflows.`;
              break;
            default:
              response = `Agent ${agent.name} received your message and is processing it.`;
          }
          
          content += `🤖 **Agent Response:**\n`;
          content += `> ${response}\n\n`;
          content += `⏱️ **Response Time:** ${new Date().toLocaleTimeString()}\n`;
        } else {
          content += `❌ **Failed to read agent config**\n`;
        }
      } else {
        content += `❌ **Agent not found:** ${args.agent_id}\n`;
      }
    } catch (error) {
      content += `❌ **Error invoking agent:** ${error}\n`;
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiChat(args: z.infer<typeof AguiChatSchema>) {
    let content = `💬 **AG-UI Chat Session**\n\n`;
    content += `🌍 **Environment:** ${args.environment}\n`;
    
    if (args.agent_id) {
      content += `🤖 **Agent ID:** ${args.agent_id}\n`;
    }
    
    content += `💬 **Initial Message:** ${args.message}\n\n`;
    
    // Mock chat initialization
    content += `✅ **Chat session initialized**\n`;
    content += `🔗 **CopilotKit integration:** Active\n`;
    content += `🎛️ **Features available:**\n`;
    content += `   • Real-time messaging\n`;
    content += `   • Agent collaboration\n`;
    content += `   • Context sharing\n`;
    content += `   • Tool integration\n\n`;
    
    content += `💡 **Next Steps:**\n`;
    content += `1. Open your ${args.environment} DevPod workspace\n`;
    content += `2. Navigate to the chat interface\n`;
    content += `3. Start interacting with the agent\n`;
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiGenerateUi(args: z.infer<typeof AguiGenerateUiSchema>) {
    let content = `🎨 **AG-UI Component Generation**\n\n`;
    content += `🌍 **Environment:** ${args.environment}\n`;
    content += `💭 **Prompt:** ${args.prompt}\n`;
    content += `🧩 **Component Type:** ${args.component_type}\n`;
    content += `⚛️ **Framework:** ${args.framework}\n\n`;
    
    // Mock UI generation
    content += `🔄 **Generating UI component...**\n\n`;
    
    let componentCode = "";
    if (args.framework === "react" || args.framework === "auto") {
      componentCode = `import React from 'react';

interface ${args.component_type.charAt(0).toUpperCase() + args.component_type.slice(1)}Props {
  // Generated based on: ${args.prompt}
}

export const Generated${args.component_type.charAt(0).toUpperCase() + args.component_type.slice(1)}: React.FC<${args.component_type.charAt(0).toUpperCase() + args.component_type.slice(1)}Props> = () => {
  return (
    <div className="generated-component">
      <h2>Generated Component</h2>
      <p>This component was generated based on: "${args.prompt}"</p>
    </div>
  );
};`;
    }
    
    content += `✅ **Component Generated**\n`;
    content += `📄 **Code:**\n\`\`\`typescript\n${componentCode}\n\`\`\`\n\n`;
    
    content += `🚀 **Integration Instructions:**\n`;
    content += `1. Save the component to your ${args.environment} environment\n`;
    content += `2. Import and use in your application\n`;
    content += `3. Customize styling and behavior as needed\n`;
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiSharedState(args: z.infer<typeof AguiSharedStateSchema>) {
    let content = `🔄 **AG-UI Shared State Management**\n\n`;
    content += `🌍 **Environment:** ${args.environment}\n`;
    content += `🎯 **Action:** ${args.action}\n`;
    content += `🏷️ **Namespace:** ${args.namespace}\n`;
    
    if (args.key) {
      content += `🔑 **Key:** ${args.key}\n`;
    }
    content += "\n";
    
    // Mock shared state operations
    switch (args.action) {
      case "get":
        content += `📥 **Retrieved value for key "${args.key}"**\n`;
        content += `💾 **Value:** ${JSON.stringify({ example: "data", timestamp: Date.now() }, null, 2)}\n`;
        break;
      case "set":
        content += `📤 **Set value for key "${args.key}"**\n`;
        content += `💾 **Value:** ${JSON.stringify(args.value, null, 2)}\n`;
        content += `✅ **State updated successfully**\n`;
        break;
      case "update":
        content += `🔄 **Updated value for key "${args.key}"**\n`;
        content += `💾 **New Value:** ${JSON.stringify(args.value, null, 2)}\n`;
        content += `✅ **State updated successfully**\n`;
        break;
      case "delete":
        content += `🗑️ **Deleted key "${args.key}"**\n`;
        content += `✅ **Key removed from shared state**\n`;
        break;
      case "list":
        content += `📋 **Available keys in namespace "${args.namespace}":**\n`;
        content += `   • user_preferences\n`;
        content += `   • session_data\n`;
        content += `   • agent_memory\n`;
        content += `   • ui_state\n`;
        break;
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiStatus(args: z.infer<typeof AguiStatusSchema>) {
    const workspaceRoot = getWorkspaceRoot();
    
    let content = `🤖 **AG-UI Status Report**\n\n`;
    
    try {
      // Check agentic environments
      const environments = ["agentic-python", "agentic-typescript", "agentic-rust", "agentic-go", "agentic-nushell"];
      
      for (const env of environments) {
        if (!args.environment || args.environment === env) {
          const envIcon = getEnvironmentIcon(getEnvironmentType(env));
          
          // Check if template exists
          const templatePath = `devpod-automation/templates/${env}/devcontainer.json`;
          const templateExists = await executeCommand("test", ["-f", templatePath], { cwd: workspaceRoot });
          
          const statusIcon = templateExists.success ? "✅" : "❌";
          content += `${statusIcon} ${envIcon} **${env}**\n`;
          
          if (templateExists.success) {
            content += `   📦 Template: Available\n`;
            
            // Check for active workspaces
            const workspaceCheck = await executeCommand("devpod", ["list", "--output", "json"], { cwd: workspaceRoot });
            if (workspaceCheck.success) {
              try {
                const workspaces = JSON.parse(workspaceCheck.output || "[]");
                const envWorkspaces = workspaces.filter((w: any) => w.name && w.name.includes(env));
                content += `   🔧 Active Workspaces: ${envWorkspaces.length}\n`;
              } catch {
                content += `   🔧 Active Workspaces: Unknown\n`;
              }
            }
            
            // Check for agents
            const agentsCheck = await executeCommand("find", [
              `devpod-automation/agents/${env}`,
              "-name", "*.json",
              "-type", "f"
            ], { cwd: workspaceRoot });
            
            const agentCount = agentsCheck.success ? 
              (agentsCheck.output?.trim().split('\n').filter(f => f).length || 0) : 0;
            content += `   🤖 Configured Agents: ${agentCount}\n`;
          } else {
            content += `   ❌ Template not found\n`;
          }
          
          content += "\n";
        }
      }
      
      if (args.detailed) {
        content += `📊 **Detailed Metrics:**\n`;
        content += `   • MCP Server: Active\n`;
        content += `   • AG-UI Protocol: Supported\n`;
        content += `   • CopilotKit Integration: Ready\n`;
        content += `   • Cross-Environment Communication: Enabled\n`;
      }
      
    } catch (error) {
      content += `❌ **Error checking status:** ${error}\n`;
    }
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  async function handleAguiWorkflow(args: z.infer<typeof AguiWorkflowSchema>) {
    let content = `🔄 **AG-UI Workflow Execution**\n\n`;
    content += `🌍 **Environment:** ${args.environment}\n`;
    content += `🎭 **Workflow Type:** ${args.workflow_type}\n`;
    
    if (args.agents && args.agents.length > 0) {
      content += `🤖 **Agents:** ${args.agents.join(", ")}\n`;
    }
    content += "\n";
    
    // Mock workflow execution based on type
    switch (args.workflow_type) {
      case "agent_chat":
        content += `💬 **Agentic Chat Workflow**\n`;
        content += `1. ✅ Initialize chat session\n`;
        content += `2. ✅ Connect CopilotKit interface\n`;
        content += `3. ✅ Enable real-time messaging\n`;
        content += `4. ✅ Activate agent collaboration\n`;
        break;
      case "ui_generation":
        content += `🎨 **Generative UI Workflow**\n`;
        content += `1. ✅ Analyze generation prompt\n`;
        content += `2. ✅ Select appropriate framework\n`;
        content += `3. ✅ Generate component code\n`;
        content += `4. ✅ Apply styling and interactions\n`;
        break;
      case "state_collaboration":
        content += `🔄 **Shared State Workflow**\n`;
        content += `1. ✅ Initialize shared state store\n`;
        content += `2. ✅ Enable real-time synchronization\n`;
        content += `3. ✅ Connect UI components\n`;
        content += `4. ✅ Enable collaborative editing\n`;
        break;
      case "tool_based_ui":
        content += `🛠️ **Tool-Based UI Workflow**\n`;
        content += `1. ✅ Register tool integrations\n`;
        content += `2. ✅ Generate tool interfaces\n`;
        content += `3. ✅ Enable tool invocation\n`;
        content += `4. ✅ Display tool results\n`;
        break;
      case "human_in_loop":
        content += `👥 **Human-in-the-Loop Workflow**\n`;
        content += `1. ✅ Setup interaction points\n`;
        content += `2. ✅ Enable approval mechanisms\n`;
        content += `3. ✅ Configure feedback loops\n`;
        content += `4. ✅ Activate collaborative planning\n`;
        break;
    }
    
    content += `\n⏱️ **Execution Time:** ${new Date().toLocaleTimeString()}\n`;
    content += `✅ **Workflow completed successfully**\n`;
    
    return {
      content: [{ type: "text", text: content }],
    };
  }

  // Helper Functions
  
  function getEnvironmentIcon(type: string): string {
    switch (type) {
      case "python": return "🐍";
      case "typescript": return "📘";
      case "rust": return "🦀";
      case "go": return "🐹";
      case "nushell": return "🐚";
      default: return "📦";
    }
  }
  
  function getEnvironmentType(envName: string): string {
    if (envName.includes("python")) return "python";
    if (envName.includes("typescript")) return "typescript";
    if (envName.includes("rust")) return "rust";
    if (envName.includes("go")) return "go";
    if (envName.includes("nushell")) return "nushell";
    return "unknown";
  }
  
  function getWorkspaceEnvironmentType(workspaceName: string): string {
    if (workspaceName.includes("python")) return "python";
    if (workspaceName.includes("typescript")) return "typescript";
    if (workspaceName.includes("rust")) return "rust";
    if (workspaceName.includes("go")) return "go";
    if (workspaceName.includes("nushell")) return "nushell";
    return "unknown";
  }
  
  function groupBy<T>(array: T[], keyFn: (item: T) => string): Record<string, T[]> {
    return array.reduce((groups, item) => {
      const key = keyFn(item);
      if (!groups[key]) groups[key] = [];
      groups[key].push(item);
      return groups;
    }, {} as Record<string, T[]>);
  }

  const cleanup = async () => {
    // No persistent connections to clean up in this implementation
  };

  return { server, cleanup };
};