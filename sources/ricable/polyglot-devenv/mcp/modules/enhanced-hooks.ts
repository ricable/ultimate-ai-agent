import { z } from "zod";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { executeCommand, getWorkspaceRoot, createErrorResult, createSuccessResult, validateToolExecution } from "../polyglot-utils.js";
import type { CommandResult } from "../polyglot-types.js";

// Enhanced AI Hooks Tool Schemas
export const EnhancedHookContextTriggersSchema = z.object({
  action: z.enum(["status", "trigger", "configure", "analyze"]).describe("Action to perform"),
  feature_file: z.string().optional().describe("Feature file to analyze for PRP generation"),
  environment: z.string().optional().describe("Target environment for context engineering"),
  cooldown: z.number().default(60).describe("Cooldown period in seconds"),
});

export const EnhancedHookErrorResolutionSchema = z.object({
  action: z.enum(["analyze", "suggest", "learn", "report"]).describe("Error resolution action"),
  error_text: z.string().optional().describe("Error text to analyze"),
  environment: z.string().optional().describe("Environment where error occurred"),
  confidence_threshold: z.number().default(0.7).describe("Minimum confidence for suggestions"),
});

export const EnhancedHookEnvOrchestrationSchema = z.object({
  action: z.enum(["switch", "provision", "optimize", "analytics"]).describe("Environment orchestration action"),
  target_environment: z.string().optional().describe("Target environment to switch to"),
  file_context: z.string().optional().describe("File context for smart switching"),
  auto_provision: z.boolean().default(true).describe("Auto-provision if environment doesn't exist"),
});

export const EnhancedHookDependencyTrackingSchema = z.object({
  action: z.enum(["scan", "monitor", "report", "analyze"]).describe("Dependency tracking action"),
  file_path: z.string().optional().describe("Specific package file to monitor"),
  environment: z.string().optional().describe("Environment to scan"),
  security_check: z.boolean().default(true).describe("Include security vulnerability scan"),
});

export const EnhancedHookPerformanceIntegrationSchema = z.object({
  action: z.enum(["measure", "track", "optimize", "report"]).describe("Performance integration action"),
  command: z.string().optional().describe("Command to measure performance"),
  environment: z.string().optional().describe("Environment for performance tracking"),
  metrics: z.array(z.string()).optional().describe("Specific metrics to track"),
});

export const EnhancedHookQualityGatesSchema = z.object({
  action: z.enum(["validate", "enforce", "configure", "report"]).describe("Quality gates action"),
  environment: z.string().optional().describe("Environment for quality validation"),
  rules: z.array(z.string()).optional().describe("Specific quality rules to check"),
  fail_on_error: z.boolean().default(true).describe("Fail validation on errors"),
});

export const EnhancedHookDevpodManagerSchema = z.object({
  action: z.enum(["lifecycle", "optimize", "monitor", "cleanup"]).describe("DevPod management action"),
  environment: z.string().optional().describe("Environment for DevPod management"),
  resource_limits: z.object({
    max_containers: z.number().optional(),
    memory_limit: z.string().optional(),
    cpu_limit: z.string().optional(),
  }).optional().describe("Resource limits for containers"),
});

export const EnhancedHookPrpLifecycleSchema = z.object({
  action: z.enum(["track", "report", "analyze", "cleanup"]).describe("PRP lifecycle action"),
  prp_file: z.string().optional().describe("Specific PRP file to track"),
  status: z.enum(["generated", "executing", "completed", "failed"]).optional().describe("PRP status to filter"),
  days: z.number().default(7).describe("Number of days for historical analysis"),
});

// Enhanced AI Hooks Tool Definitions
export const enhancedHooksTools: Tool[] = [
  {
    name: "enhanced_hook_context_triggers",
    description: "Auto PRP generation from feature edits with smart environment detection",
    inputSchema: zodToJsonSchema(EnhancedHookContextTriggersSchema) as any,
  },
  {
    name: "enhanced_hook_error_resolution",
    description: "AI-powered error analysis with environment-specific solutions",
    inputSchema: zodToJsonSchema(EnhancedHookErrorResolutionSchema) as any,
  },
  {
    name: "enhanced_hook_env_orchestration",
    description: "Smart environment switching with usage analytics",
    inputSchema: zodToJsonSchema(EnhancedHookEnvOrchestrationSchema) as any,
  },
  {
    name: "enhanced_hook_dependency_tracking",
    description: "Cross-environment dependency monitoring with security scanning",
    inputSchema: zodToJsonSchema(EnhancedHookDependencyTrackingSchema) as any,
  },
  {
    name: "enhanced_hook_performance_integration",
    description: "Advanced performance tracking with optimization recommendations",
    inputSchema: zodToJsonSchema(EnhancedHookPerformanceIntegrationSchema) as any,
  },
  {
    name: "enhanced_hook_quality_gates",
    description: "Cross-language quality enforcement with intelligent validation",
    inputSchema: zodToJsonSchema(EnhancedHookQualityGatesSchema) as any,
  },
  {
    name: "enhanced_hook_devpod_manager",
    description: "Smart container lifecycle management with resource optimization",
    inputSchema: zodToJsonSchema(EnhancedHookDevpodManagerSchema) as any,
  },
  {
    name: "enhanced_hook_prp_lifecycle",
    description: "PRP status tracking and reports with analytics",
    inputSchema: zodToJsonSchema(EnhancedHookPrpLifecycleSchema) as any,
  },
];

// Enhanced AI Hooks Tool Handlers
export async function handleEnhancedHookContextTriggers(args: z.infer<typeof EnhancedHookContextTriggersSchema>): Promise<CommandResult> {
  const { action, feature_file, environment, cooldown } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "status":
        command = `python3 ${workspaceRoot}/.claude/hooks/context-engineering-auto-triggers.py --status`;
        break;
      case "trigger":
        if (!feature_file) {
          return {
            success: false,
            output: "",
            error: "feature_file is required for trigger action",
            exitCode: 1,
            duration: 0,
      timestamp: new Date(),
          };
        }
        command = `python3 ${workspaceRoot}/.claude/hooks/context-engineering-auto-triggers.py --trigger "${feature_file}"`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      case "configure":
        command = `python3 ${workspaceRoot}/.claude/hooks/context-engineering-auto-triggers.py --configure --cooldown ${cooldown}`;
        break;
      case "analyze":
        command = `python3 ${workspaceRoot}/.claude/hooks/context-engineering-auto-triggers.py --analyze`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
          duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Context triggers ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute context triggers: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookErrorResolution(args: z.infer<typeof EnhancedHookErrorResolutionSchema>): Promise<CommandResult> {
  const { action, error_text, environment, confidence_threshold } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "analyze":
        if (!error_text) {
          return {
            success: false,
            output: "",
            error: "error_text is required for analyze action",
            exitCode: 1,
      duration: 0,
      timestamp: new Date(),
          };
        }
        command = `python3 ${workspaceRoot}/.claude/hooks/intelligent-error-resolution.py --analyze "${error_text}"`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        command += ` --confidence ${confidence_threshold}`;
        break;
      case "suggest":
        if (!error_text) {
          return {
            success: false,
            output: "",
            error: "error_text is required for suggest action",
            exitCode: 1,
      duration: 0,
      timestamp: new Date(),
          };
        }
        command = `python3 ${workspaceRoot}/.claude/hooks/intelligent-error-resolution.py --suggest "${error_text}"`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      case "learn":
        command = `python3 ${workspaceRoot}/.claude/hooks/intelligent-error-resolution.py --learn`;
        break;
      case "report":
        command = `python3 ${workspaceRoot}/.claude/hooks/intelligent-error-resolution.py --report`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Error resolution ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute error resolution: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookEnvOrchestration(args: z.infer<typeof EnhancedHookEnvOrchestrationSchema>): Promise<CommandResult> {
  const { action, target_environment, file_context, auto_provision } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "switch":
        if (!target_environment) {
          return {
            success: false,
            output: "",
            error: "target_environment is required for switch action",
            exitCode: 1,
      duration: 0,
      timestamp: new Date(),
          };
        }
        command = `python3 ${workspaceRoot}/.claude/hooks/smart-environment-orchestration.py --switch "${target_environment}"`;
        if (file_context) {
          command += ` --context "${file_context}"`;
        }
        if (auto_provision) {
          command += " --auto-provision";
        }
        break;
      case "provision":
        if (!target_environment) {
          return {
            success: false,
            output: "",
            error: "target_environment is required for provision action",
            exitCode: 1,
      duration: 0,
      timestamp: new Date(),
          };
        }
        command = `python3 ${workspaceRoot}/.claude/hooks/smart-environment-orchestration.py --provision "${target_environment}"`;
        break;
      case "optimize":
        command = `python3 ${workspaceRoot}/.claude/hooks/smart-environment-orchestration.py --optimize`;
        break;
      case "analytics":
        command = `python3 ${workspaceRoot}/.claude/hooks/smart-environment-orchestration.py --analytics`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Environment orchestration ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute environment orchestration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookDependencyTracking(args: z.infer<typeof EnhancedHookDependencyTrackingSchema>): Promise<CommandResult> {
  const { action, file_path, environment, security_check } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "scan":
        command = `python3 ${workspaceRoot}/.claude/hooks/cross-environment-dependency-tracking.py --scan`;
        if (file_path) {
          command += ` --file "${file_path}"`;
        }
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        if (security_check) {
          command += " --security";
        }
        break;
      case "monitor":
        command = `python3 ${workspaceRoot}/.claude/hooks/cross-environment-dependency-tracking.py --monitor`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      case "report":
        command = `python3 ${workspaceRoot}/.claude/hooks/cross-environment-dependency-tracking.py --report`;
        break;
      case "analyze":
        command = `python3 ${workspaceRoot}/.claude/hooks/cross-environment-dependency-tracking.py --analyze`;
        if (security_check) {
          command += " --security";
        }
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Dependency tracking ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute dependency tracking: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookPerformanceIntegration(args: z.infer<typeof EnhancedHookPerformanceIntegrationSchema>): Promise<CommandResult> {
  const { action, command, environment, metrics } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let cmd: string;
    
    switch (action) {
      case "measure":
        if (!command) {
          return {
            success: false,
            output: "",
            error: "command is required for measure action",
            exitCode: 1,
      duration: 0,
      timestamp: new Date(),
          };
        }
        cmd = `python3 ${workspaceRoot}/.claude/hooks/performance-analytics-integration.py --measure "${command}"`;
        if (environment) {
          cmd += ` --environment "${environment}"`;
        }
        break;
      case "track":
        cmd = `python3 ${workspaceRoot}/.claude/hooks/performance-analytics-integration.py --track`;
        if (metrics && metrics.length > 0) {
          cmd += ` --metrics ${metrics.join(",")}`;
        }
        break;
      case "optimize":
        cmd = `python3 ${workspaceRoot}/.claude/hooks/performance-analytics-integration.py --optimize`;
        if (environment) {
          cmd += ` --environment "${environment}"`;
        }
        break;
      case "report":
        cmd = `python3 ${workspaceRoot}/.claude/hooks/performance-analytics-integration.py --report`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(cmd);
    
    return {
      success: result.success,
      output: result.output || `Performance integration ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute performance integration: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookQualityGates(args: z.infer<typeof EnhancedHookQualityGatesSchema>): Promise<CommandResult> {
  const { action, environment, rules, fail_on_error } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "validate":
        command = `python3 ${workspaceRoot}/.claude/hooks/quality-gates-validator.py --validate`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        if (rules && rules.length > 0) {
          command += ` --rules ${rules.join(",")}`;
        }
        if (!fail_on_error) {
          command += " --no-fail";
        }
        break;
      case "enforce":
        command = `python3 ${workspaceRoot}/.claude/hooks/quality-gates-validator.py --enforce`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      case "configure":
        command = `python3 ${workspaceRoot}/.claude/hooks/quality-gates-validator.py --configure`;
        if (rules && rules.length > 0) {
          command += ` --rules ${rules.join(",")}`;
        }
        break;
      case "report":
        command = `python3 ${workspaceRoot}/.claude/hooks/quality-gates-validator.py --report`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `Quality gates ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute quality gates: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookDevpodManager(args: z.infer<typeof EnhancedHookDevpodManagerSchema>): Promise<CommandResult> {
  const { action, environment, resource_limits } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "lifecycle":
        command = `python3 ${workspaceRoot}/.claude/hooks/devpod-resource-manager.py --lifecycle`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      case "optimize":
        command = `python3 ${workspaceRoot}/.claude/hooks/devpod-resource-manager.py --optimize`;
        if (resource_limits) {
          if (resource_limits.max_containers) {
            command += ` --max-containers ${resource_limits.max_containers}`;
          }
          if (resource_limits.memory_limit) {
            command += ` --memory-limit "${resource_limits.memory_limit}"`;
          }
          if (resource_limits.cpu_limit) {
            command += ` --cpu-limit "${resource_limits.cpu_limit}"`;
          }
        }
        break;
      case "monitor":
        command = `python3 ${workspaceRoot}/.claude/hooks/devpod-resource-manager.py --monitor`;
        break;
      case "cleanup":
        command = `python3 ${workspaceRoot}/.claude/hooks/devpod-resource-manager.py --cleanup`;
        if (environment) {
          command += ` --environment "${environment}"`;
        }
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `DevPod manager ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute DevPod manager: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}

export async function handleEnhancedHookPrpLifecycle(args: z.infer<typeof EnhancedHookPrpLifecycleSchema>): Promise<CommandResult> {
  const { action, prp_file, status, days } = args;
  
  const workspaceRoot = getWorkspaceRoot();
  
  try {
    let command: string;
    
    switch (action) {
      case "track":
        command = `python3 ${workspaceRoot}/.claude/hooks/prp-lifecycle-manager.py --track`;
        if (prp_file) {
          command += ` --file "${prp_file}"`;
        }
        if (status) {
          command += ` --status "${status}"`;
        }
        break;
      case "report":
        command = `python3 ${workspaceRoot}/.claude/hooks/prp-lifecycle-manager.py --report --days ${days}`;
        if (status) {
          command += ` --status "${status}"`;
        }
        break;
      case "analyze":
        command = `python3 ${workspaceRoot}/.claude/hooks/prp-lifecycle-manager.py --analyze --days ${days}`;
        break;
      case "cleanup":
        command = `python3 ${workspaceRoot}/.claude/hooks/prp-lifecycle-manager.py --cleanup --days ${days}`;
        break;
      default:
        return {
          success: false,
          output: "",
          error: `Invalid action: ${action}`,
          exitCode: 1,
      duration: 0,
      timestamp: new Date(),
        };
    }
    
    const result = await executeCommand(command);
    
    return {
      success: result.success,
      output: result.output || `PRP lifecycle ${action} completed`,
      error: result.error,
      exitCode: result.success ? 0 : 1,
      duration: 0,
      timestamp: new Date(),
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: `Failed to execute PRP lifecycle: ${error instanceof Error ? error.message : String(error)}`,
      exitCode: 1,
      duration: 0,
      timestamp: new Date(),
    };
  }
}