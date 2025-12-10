import { z } from "zod";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { executeCommand, getWorkspaceRoot } from "../polyglot-utils.js";
import type { CommandResult } from "../polyglot-types.js";

// Configuration Management Tool Schemas
export const ConfigGenerationSchema = z.object({
  action: z.enum(["generate", "update", "validate", "diff", "preview"]).describe("Configuration generation action"),
  target: z.enum(["devbox", "devcontainer", "environment", "all"]).default("all").describe("Configuration target"),
  environment: z.string().optional().describe("Specific environment to generate config for"),
  force: z.boolean().default(false).describe("Force regeneration even if up-to-date"),
  dry_run: z.boolean().default(false).describe("Show what would be generated without writing files"),
});

export const ConfigSyncSchema = z.object({
  action: z.enum(["sync", "push", "pull", "merge", "reset"]).describe("Configuration synchronization action"),
  source: z.enum(["canonical", "environment", "devpod", "remote"]).describe("Source for synchronization"),
  target: z.enum(["canonical", "environment", "devpod", "remote", "all"]).describe("Target for synchronization"),
  environments: z.array(z.string()).optional().describe("Specific environments to sync"),
  conflict_resolution: z.enum(["auto", "manual", "source-wins", "target-wins"]).default("auto").describe("How to resolve conflicts"),
});

export const ConfigValidationSchema = z.object({
  action: z.enum(["validate", "lint", "test", "security", "compliance"]).describe("Configuration validation action"),
  scope: z.enum(["single", "environment", "cross-env", "global"]).default("global").describe("Validation scope"),
  config_type: z.enum(["devbox", "devcontainer", "settings", "secrets", "all"]).default("all").describe("Type of configuration"),
  fix_issues: z.boolean().default(false).describe("Automatically fix issues where possible"),
  strict_mode: z.boolean().default(true).describe("Enable strict validation rules"),
});

export const ConfigBackupSchema = z.object({
  action: z.enum(["backup", "restore", "list", "cleanup", "archive"]).describe("Configuration backup action"),
  backup_name: z.string().optional().describe("Name for backup or restore"),
  include_secrets: z.boolean().default(false).describe("Include sensitive configuration"),
  compression: z.boolean().default(true).describe("Compress backup files"),
  retention_days: z.number().default(30).describe("Number of days to retain backups"),
  remote_backup: z.boolean().default(false).describe("Store backup in remote location"),
});

export const ConfigTemplateSchema = z.object({
  action: z.enum(["create", "update", "apply", "validate", "list"]).describe("Configuration template action"),
  template_name: z.string().describe("Name of configuration template"),
  template_type: z.enum(["devbox", "devcontainer", "environment", "project", "custom"]).describe("Type of template"),
  variables: z.record(z.any()).optional().describe("Template variables and values"),
  output_path: z.string().optional().describe("Output path for generated configuration"),
  inherit_from: z.string().optional().describe("Parent template to inherit from"),
});

// Configuration Management Tool Definitions
export const configManagementTools: Tool[] = [
  {
    name: "config_generation",
    description: "Generate configuration files from canonical definitions with zero drift guarantee",
    inputSchema: zodToJsonSchema(ConfigGenerationSchema) as any,
  },
  {
    name: "config_sync",
    description: "Synchronize configurations across environments with conflict resolution",
    inputSchema: zodToJsonSchema(ConfigSyncSchema) as any,
  },
  {
    name: "config_validation", 
    description: "Comprehensive validation of configurations for consistency and compliance",
    inputSchema: zodToJsonSchema(ConfigValidationSchema) as any,
  },
  {
    name: "config_backup",
    description: "Backup and restore configuration files with versioning and encryption",
    inputSchema: zodToJsonSchema(ConfigBackupSchema) as any,
  },
  {
    name: "config_template",
    description: "Manage configuration templates with inheritance and variable substitution",
    inputSchema: zodToJsonSchema(ConfigTemplateSchema) as any,
  },
];

// Configuration Management Tool Handlers
export async function handleConfigGeneration(args: z.infer<typeof ConfigGenerationSchema>): Promise<CommandResult> {
  const { action, target, environment, force, dry_run } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/context-engineering/devpod/environments/refactor-configs.nu --action ${action} --target ${target}`;
    
    if (environment) command += ` --environment "${environment}"`;
    if (force) command += " --force";
    if (dry_run) command += " --dry-run";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Configuration ${action} completed for ${target}${environment ? ` (${environment})` : ''}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} configuration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleConfigSync(args: z.infer<typeof ConfigSyncSchema>): Promise<CommandResult> {
  const { action, source, target, environments, conflict_resolution } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/sync-configs.nu --action ${action} --source ${source} --target ${target} --conflicts ${conflict_resolution}`;
    
    if (environments && environments.length > 0) {
      command += ` --environments ${environments.join(',')}`;
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Configuration sync ${action} completed: ${source} → ${target}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to sync configuration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleConfigValidation(args: z.infer<typeof ConfigValidationSchema>): Promise<CommandResult> {
  const { action, scope, config_type, fix_issues, strict_mode } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/validate-all.nu --action ${action} --scope ${scope} --type ${config_type}`;
    
    if (fix_issues) command += " --fix";
    if (strict_mode) command += " --strict";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Configuration validation ${action} completed for ${config_type} (${scope} scope)`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to validate configuration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleConfigBackup(args: z.infer<typeof ConfigBackupSchema>): Promise<CommandResult> {
  const { action, backup_name, include_secrets, compression, retention_days, remote_backup } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/config-backup.nu --action ${action} --retention ${retention_days}`;
    
    if (backup_name) command += ` --name "${backup_name}"`;
    if (include_secrets) command += " --include-secrets";
    if (compression) command += " --compress";
    if (remote_backup) command += " --remote";
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Configuration backup ${action} completed${backup_name ? ` (${backup_name})` : ''}`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} configuration backup: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleConfigTemplate(args: z.infer<typeof ConfigTemplateSchema>): Promise<CommandResult> {
  const { action, template_name, template_type, variables, output_path, inherit_from } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command = `nu ${workspaceRoot}/scripts/config-templates.nu --action ${action} --name "${template_name}" --type ${template_type}`;
    
    if (variables) command += ` --variables '${JSON.stringify(variables)}'`;
    if (output_path) command += ` --output "${output_path}"`;
    if (inherit_from) command += ` --inherit "${inherit_from}"`;
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Configuration template ${action} completed: ${template_name} (${template_type})`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to ${action} configuration template: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}