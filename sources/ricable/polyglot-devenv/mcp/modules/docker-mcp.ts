import { z } from "zod";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { executeCommand, getWorkspaceRoot } from "../polyglot-utils.js";
import type { CommandResult } from "../polyglot-types.js";

// Docker MCP Tool Schemas
export const DockerMcpGatewayStartSchema = z.object({
  port: z.number().default(8080).describe("Port for Docker MCP gateway"),
  background: z.boolean().default(true).describe("Run in background mode"),
  log_level: z.enum(["debug", "info", "warn", "error"]).default("info").describe("Log level"),
});

export const DockerMcpGatewayStatusSchema = z.object({
  detailed: z.boolean().default(false).describe("Include detailed status information"),
});

export const DockerMcpToolsListSchema = z.object({
  category: z.string().optional().describe("Filter tools by category"),
  verbose: z.boolean().default(false).describe("Include detailed tool information"),
});

export const DockerMcpHttpBridgeSchema = z.object({
  port: z.number().default(8080).describe("Port for HTTP/SSE bridge"),
  host: z.string().default("localhost").describe("Host to bind to"),
  cors: z.boolean().default(true).describe("Enable CORS support"),
});

export const DockerMcpClientListSchema = z.object({
  active_only: z.boolean().default(false).describe("Show only active clients"),
});

export const DockerMcpServerListSchema = z.object({
  running_only: z.boolean().default(false).describe("Show only running servers"),
});

export const DockerMcpGeminiConfigSchema = z.object({
  api_key: z.string().optional().describe("Gemini API key (optional, uses env var)"),
  model: z.string().default("gemini-pro").describe("Gemini model to use"),
  test: z.boolean().default(false).describe("Run test configuration"),
});

export const DockerMcpTestSchema = z.object({
  suite: z.enum(["all", "integration", "security", "performance"]).default("all").describe("Test suite to run"),
  verbose: z.boolean().default(false).describe("Verbose test output"),
});

export const DockerMcpDemoSchema = z.object({
  scenario: z.enum(["basic", "advanced", "ai-integration"]).default("basic").describe("Demo scenario"),
  interactive: z.boolean().default(true).describe("Run interactive demo"),
});

export const DockerMcpSecurityScanSchema = z.object({
  target: z.enum(["containers", "images", "network", "all"]).default("all").describe("Security scan target"),
  detailed: z.boolean().default(false).describe("Detailed security report"),
});

export const DockerMcpResourceLimitsSchema = z.object({
  container_id: z.string().optional().describe("Specific container to manage"),
  cpu_limit: z.string().optional().describe("CPU limit (e.g., '1.0')"),
  memory_limit: z.string().optional().describe("Memory limit (e.g., '2GB')"),
  action: z.enum(["set", "get", "reset"]).default("get").describe("Resource management action"),
});

export const DockerMcpNetworkIsolationSchema = z.object({
  action: z.enum(["enable", "disable", "status", "configure"]).describe("Network isolation action"),
  network_name: z.string().optional().describe("Custom network name"),
});

export const DockerMcpSignatureVerifySchema = z.object({
  image: z.string().describe("Docker image to verify"),
  trusted_registry: z.boolean().default(true).describe("Require trusted registry"),
});

export const DockerMcpLogsSchema = z.object({
  component: z.enum(["gateway", "bridge", "clients", "all"]).default("gateway").describe("Component to get logs from"),
  lines: z.number().default(100).describe("Number of log lines"),
  follow: z.boolean().default(false).describe("Follow log output"),
});

export const DockerMcpCleanupSchema = z.object({
  target: z.enum(["containers", "images", "networks", "volumes", "all"]).default("containers").describe("Cleanup target"),
  force: z.boolean().default(false).describe("Force cleanup"),
  unused_only: z.boolean().default(true).describe("Clean only unused resources"),
});

// Docker MCP Tool Definitions
export const dockerMcpTools: Tool[] = [
  {
    name: "docker_mcp_gateway_start",
    description: "Start Docker MCP gateway for centralized tool execution",
    inputSchema: zodToJsonSchema(DockerMcpGatewayStartSchema) as any,
  },
  {
    name: "docker_mcp_gateway_status",
    description: "Check Docker MCP gateway status and health",
    inputSchema: zodToJsonSchema(DockerMcpGatewayStatusSchema) as any,
  },
  {
    name: "docker_mcp_tools_list",
    description: "List available containerized MCP tools",
    inputSchema: zodToJsonSchema(DockerMcpToolsListSchema) as any,
  },
  {
    name: "docker_mcp_http_bridge",
    description: "Start HTTP/SSE bridge for web integration",
    inputSchema: zodToJsonSchema(DockerMcpHttpBridgeSchema) as any,
  },
  {
    name: "docker_mcp_client_list",
    description: "List connected MCP clients",
    inputSchema: zodToJsonSchema(DockerMcpClientListSchema) as any,
  },
  {
    name: "docker_mcp_server_list",
    description: "List running MCP servers",
    inputSchema: zodToJsonSchema(DockerMcpServerListSchema) as any,
  },
  {
    name: "docker_mcp_gemini_config",
    description: "Configure Gemini AI integration",
    inputSchema: zodToJsonSchema(DockerMcpGeminiConfigSchema) as any,
  },
  {
    name: "docker_mcp_test",
    description: "Run Docker MCP integration tests",
    inputSchema: zodToJsonSchema(DockerMcpTestSchema) as any,
  },
  {
    name: "docker_mcp_demo",
    description: "Run Docker MCP demonstration scenarios",
    inputSchema: zodToJsonSchema(DockerMcpDemoSchema) as any,
  },
  {
    name: "docker_mcp_security_scan",
    description: "Scan Docker MCP components for security vulnerabilities",
    inputSchema: zodToJsonSchema(DockerMcpSecurityScanSchema) as any,
  },
  {
    name: "docker_mcp_resource_limits",
    description: "Manage container resource limits and quotas",
    inputSchema: zodToJsonSchema(DockerMcpResourceLimitsSchema) as any,
  },
  {
    name: "docker_mcp_network_isolation",
    description: "Configure network isolation for secure execution",
    inputSchema: zodToJsonSchema(DockerMcpNetworkIsolationSchema) as any,
  },
  {
    name: "docker_mcp_signature_verify",
    description: "Verify cryptographic signatures of Docker images",
    inputSchema: zodToJsonSchema(DockerMcpSignatureVerifySchema) as any,
  },
  {
    name: "docker_mcp_logs",
    description: "Access Docker MCP component logs",
    inputSchema: zodToJsonSchema(DockerMcpLogsSchema) as any,
  },
  {
    name: "docker_mcp_cleanup",
    description: "Clean up Docker MCP resources and containers",
    inputSchema: zodToJsonSchema(DockerMcpCleanupSchema) as any,
  },
];

// Docker MCP Tool Handlers
export async function handleDockerMcpGatewayStart(args: z.infer<typeof DockerMcpGatewayStartSchema>): Promise<CommandResult> {
  const { port, background, log_level } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = background 
      ? `${workspaceRoot}/.claude/start-mcp-gateway.sh --port ${port} --log-level ${log_level} --daemon`
      : `docker mcp gateway run --port ${port} --log-level ${log_level}`;
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Docker MCP gateway started on port ${port}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to start Docker MCP gateway: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpGatewayStatus(args: z.infer<typeof DockerMcpGatewayStatusSchema>): Promise<CommandResult> {
  const { detailed } = args;
  
  try {
    const command = detailed 
      ? "docker mcp gateway status --detailed"
      : "docker mcp gateway status";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || "Docker MCP gateway status retrieved",
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to get gateway status: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpToolsList(args: z.infer<typeof DockerMcpToolsListSchema>): Promise<CommandResult> {
  const { category, verbose } = args;
  
  try {
    let command = "docker mcp tools";
    if (category) {
      command += ` --category ${category}`;
    }
    if (verbose) {
      command += " --verbose";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || "Docker MCP tools listed",
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to list tools: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpHttpBridge(args: z.infer<typeof DockerMcpHttpBridgeSchema>): Promise<CommandResult> {
  const { port, host, cors } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `python3 ${workspaceRoot}/.claude/mcp-http-bridge.py --port ${port} --host ${host}`;
    if (cors) {
      command += " --cors";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `HTTP/SSE bridge started on ${host}:${port}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to start HTTP bridge: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpClientList(args: z.infer<typeof DockerMcpClientListSchema>): Promise<CommandResult> {
  const { active_only } = args;
  
  try {
    const command = active_only 
      ? "docker mcp client ls --active"
      : "docker mcp client ls";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || "MCP clients listed",
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to list clients: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpServerList(args: z.infer<typeof DockerMcpServerListSchema>): Promise<CommandResult> {
  const { running_only } = args;
  
  try {
    const command = running_only 
      ? "docker mcp server list --running"
      : "docker mcp server list";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || "MCP servers listed",
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to list servers: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpGeminiConfig(args: z.infer<typeof DockerMcpGeminiConfigSchema>): Promise<CommandResult> {
  const { api_key, model, test } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `python3 ${workspaceRoot}/.claude/gemini-mcp-config.py --model ${model}`;
    if (api_key) {
      command += ` --api-key "${api_key}"`;
    }
    if (test) {
      command += " --test";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Gemini integration configured with model ${model}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to configure Gemini: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpTest(args: z.infer<typeof DockerMcpTestSchema>): Promise<CommandResult> {
  const { suite, verbose } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `python3 ${workspaceRoot}/.claude/test-mcp-integration.py --suite ${suite}`;
    if (verbose) {
      command += " --verbose";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Docker MCP test suite '${suite}' completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to run tests: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpDemo(args: z.infer<typeof DockerMcpDemoSchema>): Promise<CommandResult> {
  const { scenario, interactive } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `${workspaceRoot}/.claude/demo-mcp-integration.sh --scenario ${scenario}`;
    if (!interactive) {
      command += " --no-interactive";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Docker MCP demo '${scenario}' completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to run demo: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpSecurityScan(args: z.infer<typeof DockerMcpSecurityScanSchema>): Promise<CommandResult> {
  const { target, detailed } = args;
  
  try {
    let command = `docker mcp security scan --target ${target}`;
    if (detailed) {
      command += " --detailed";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Security scan completed for ${target}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to run security scan: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpResourceLimits(args: z.infer<typeof DockerMcpResourceLimitsSchema>): Promise<CommandResult> {
  const { container_id, cpu_limit, memory_limit, action } = args;
  
  try {
    let command = `docker mcp resources ${action}`;
    if (container_id) {
      command += ` --container ${container_id}`;
    }
    if (cpu_limit && action === "set") {
      command += ` --cpu ${cpu_limit}`;
    }
    if (memory_limit && action === "set") {
      command += ` --memory ${memory_limit}`;
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Resource limits ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to manage resource limits: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpNetworkIsolation(args: z.infer<typeof DockerMcpNetworkIsolationSchema>): Promise<CommandResult> {
  const { action, network_name } = args;
  
  try {
    let command = `docker mcp network ${action}`;
    if (network_name) {
      command += ` --name ${network_name}`;
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Network isolation ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to configure network isolation: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpSignatureVerify(args: z.infer<typeof DockerMcpSignatureVerifySchema>): Promise<CommandResult> {
  const { image, trusted_registry } = args;
  
  try {
    let command = `docker mcp verify "${image}"`;
    if (trusted_registry) {
      command += " --trusted-registry";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Image signature verification completed for ${image}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to verify image signature: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpLogs(args: z.infer<typeof DockerMcpLogsSchema>): Promise<CommandResult> {
  const { component, lines, follow } = args;
  
  try {
    let command: string;
    
    if (component === "all") {
      command = `tail -n ${lines} /tmp/docker-mcp-*.log`;
    } else {
      command = `docker mcp logs ${component} --lines ${lines}`;
    }
    
    if (follow) {
      command += " --follow";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Logs retrieved for ${component}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to get logs: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleDockerMcpCleanup(args: z.infer<typeof DockerMcpCleanupSchema>): Promise<CommandResult> {
  const { target, force, unused_only } = args;
  
  try {
    let command = `docker mcp cleanup ${target}`;
    if (force) {
      command += " --force";
    }
    if (unused_only) {
      command += " --unused-only";
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Docker MCP cleanup completed for ${target}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to cleanup resources: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}