import { z } from "zod";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { executeCommand, getWorkspaceRoot } from "../polyglot-utils.js";
import type { CommandResult } from "../polyglot-types.js";

// Host/Container Separation Tool Schemas
export const HostInstallationSchema = z.object({
  component: z.enum(["docker", "devpod", "system-tools", "all"]).describe("Component to install on host"),
  configure: z.boolean().default(true).describe("Configure after installation"),
  optimize: z.boolean().default(false).describe("Optimize installation for performance"),
});

export const HostInfrastructureSchema = z.object({
  action: z.enum(["status", "connect", "configure", "monitor"]).describe("Infrastructure management action"),
  service: z.enum(["kubernetes", "github", "external-apis", "all"]).optional().describe("Specific service to manage"),
  credentials: z.boolean().default(false).describe("Include credential management"),
});

export const HostCredentialSchema = z.object({
  action: z.enum(["list", "add", "update", "delete", "rotate", "validate"]).describe("Credential management action"),
  service: z.string().optional().describe("Service name for credential"),
  credential_type: z.enum(["ssh-key", "api-token", "certificate", "password"]).optional().describe("Type of credential"),
  secure_store: z.boolean().default(true).describe("Use secure credential storage"),
});

export const HostShellIntegrationSchema = z.object({
  action: z.enum(["install", "update", "configure", "status"]).describe("Shell integration action"),
  shell_type: z.enum(["zsh", "bash", "fish", "all"]).default("zsh").describe("Shell type to integrate"),
  aliases: z.boolean().default(true).describe("Install custom aliases"),
  environment_vars: z.boolean().default(true).describe("Set up environment variables"),
});

export const ContainerIsolationSchema = z.object({
  action: z.enum(["validate", "enforce", "monitor", "report"]).describe("Container isolation action"),
  environment: z.string().optional().describe("Specific environment to check"),
  security_level: z.enum(["basic", "strict", "paranoid"]).default("strict").describe("Security level for isolation"),
});

export const ContainerToolsSchema = z.object({
  action: z.enum(["list", "install", "update", "validate", "cleanup"]).describe("Container tools action"),
  environment: z.string().optional().describe("Environment to manage tools in"),
  tool_category: z.enum(["linters", "formatters", "debuggers", "analyzers", "all"]).default("all").describe("Category of tools"),
});

export const HostContainerBridgeSchema = z.object({
  action: z.enum(["setup", "status", "sync", "cleanup"]).describe("Host-container bridge action"),
  bridge_type: z.enum(["filesystem", "network", "process", "all"]).default("all").describe("Type of bridge"),
  bidirectional: z.boolean().default(true).describe("Enable bidirectional communication"),
});

export const SecurityBoundarySchema = z.object({
  action: z.enum(["validate", "enforce", "audit", "report"]).describe("Security boundary action"),
  boundary_type: z.enum(["credential", "filesystem", "network", "process", "all"]).default("all").describe("Type of security boundary"),
  strict_mode: z.boolean().default(true).describe("Enable strict security mode"),
});

// Host/Container Separation Tool Definitions
export const hostContainerTools: Tool[] = [
  {
    name: "host_installation",
    description: "Install and configure Docker, DevPod, and system dependencies on host machine",
    inputSchema: zodToJsonSchema(HostInstallationSchema) as any,
  },
  {
    name: "host_infrastructure",
    description: "Manage infrastructure access (Kubernetes, GitHub, external APIs) from host",
    inputSchema: zodToJsonSchema(HostInfrastructureSchema) as any,
  },
  {
    name: "host_credential",
    description: "Secure credential management isolated on host machine",
    inputSchema: zodToJsonSchema(HostCredentialSchema) as any,
  },
  {
    name: "host_shell_integration",
    description: "Configure host shell aliases, environment setup, and productivity tools",
    inputSchema: zodToJsonSchema(HostShellIntegrationSchema) as any,
  },
  {
    name: "container_isolation",
    description: "Validate and enforce container isolation for secure development",
    inputSchema: zodToJsonSchema(ContainerIsolationSchema) as any,
  },
  {
    name: "container_tools",
    description: "Manage development tools within isolated container environments",
    inputSchema: zodToJsonSchema(ContainerToolsSchema) as any,
  },
  {
    name: "host_container_bridge",
    description: "Setup secure communication bridges between host and containers",
    inputSchema: zodToJsonSchema(HostContainerBridgeSchema) as any,
  },
  {
    name: "security_boundary",
    description: "Validate and enforce security boundaries between host and containers",
    inputSchema: zodToJsonSchema(SecurityBoundarySchema) as any,
  },
];

// Host/Container Separation Tool Handlers
export async function handleHostInstallation(args: z.infer<typeof HostInstallationSchema>): Promise<CommandResult> {
  const { component, configure, optimize } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (component) {
      case "docker":
        command = `nu ${workspaceRoot}/host-tooling/installation/docker-setup.nu --install`;
        if (configure) command += " --configure";
        if (optimize) command += " --optimize";
        break;
      case "devpod":
        command = `nu ${workspaceRoot}/host-tooling/installation/devpod-setup.nu --install`;
        if (configure) command += " --configure";
        break;
      case "system-tools":
        command = `nu ${workspaceRoot}/host-tooling/installation/system-tools-setup.nu --install`;
        if (configure) command += " --configure";
        break;
      case "all":
        command = `nu ${workspaceRoot}/host-tooling/installation/complete-setup.nu --install`;
        if (configure) command += " --configure";
        if (optimize) command += " --optimize";
        break;
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Host installation of ${component} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to install ${component}: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleHostInfrastructure(args: z.infer<typeof HostInfrastructureSchema>): Promise<CommandResult> {
  const { action, service, credentials } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "status":
        command = `nu ${workspaceRoot}/host-tooling/monitoring/infrastructure-status.nu`;
        if (service) command += ` --service ${service}`;
        break;
      case "connect":
        command = `nu ${workspaceRoot}/host-tooling/monitoring/infrastructure-connect.nu`;
        if (service) command += ` --service ${service}`;
        if (credentials) command += " --with-credentials";
        break;
      case "configure":
        command = `nu ${workspaceRoot}/host-tooling/monitoring/infrastructure-configure.nu`;
        if (service) command += ` --service ${service}`;
        break;
      case "monitor":
        command = `nu ${workspaceRoot}/host-tooling/monitoring/infrastructure-monitor.nu`;
        if (service) command += ` --service ${service}`;
        break;
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Infrastructure ${action} completed for ${service || 'all services'}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} infrastructure: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleHostCredential(args: z.infer<typeof HostCredentialSchema>): Promise<CommandResult> {
  const { action, service, credential_type, secure_store } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/host-tooling/credentials/credential-manager.nu --action ${action}`;
    
    if (service) command += ` --service "${service}"`;
    if (credential_type) command += ` --type ${credential_type}`;
    if (secure_store) command += " --secure-store";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Credential ${action} completed${service ? ` for ${service}` : ''}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} credentials: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleHostShellIntegration(args: z.infer<typeof HostShellIntegrationSchema>): Promise<CommandResult> {
  const { action, shell_type, aliases, environment_vars } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/host-tooling/shell-integration/shell-setup.nu --action ${action} --shell ${shell_type}`;
    
    if (aliases) command += " --aliases";
    if (environment_vars) command += " --env-vars";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Shell integration ${action} completed for ${shell_type}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} shell integration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleContainerIsolation(args: z.infer<typeof ContainerIsolationSchema>): Promise<CommandResult> {
  const { action, environment, security_level } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/container-isolation.nu --action ${action} --security-level ${security_level}`;
    
    if (environment) command += ` --environment "${environment}"`;
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Container isolation ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} container isolation: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleContainerTools(args: z.infer<typeof ContainerToolsSchema>): Promise<CommandResult> {
  const { action, environment, tool_category } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/container-tools.nu --action ${action} --category ${tool_category}`;
    
    if (environment) command += ` --environment "${environment}"`;
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Container tools ${action} completed for ${tool_category}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} container tools: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleHostContainerBridge(args: z.infer<typeof HostContainerBridgeSchema>): Promise<CommandResult> {
  const { action, bridge_type, bidirectional } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/host-tooling/bridge/host-container-bridge.nu --action ${action} --type ${bridge_type}`;
    
    if (bidirectional) command += " --bidirectional";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Host-container bridge ${action} completed for ${bridge_type}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} host-container bridge: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleSecurityBoundary(args: z.infer<typeof SecurityBoundarySchema>): Promise<CommandResult> {
  const { action, boundary_type, strict_mode } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/security-boundary.nu --action ${action} --type ${boundary_type}`;
    
    if (strict_mode) command += " --strict";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Security boundary ${action} completed for ${boundary_type}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} security boundary: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}