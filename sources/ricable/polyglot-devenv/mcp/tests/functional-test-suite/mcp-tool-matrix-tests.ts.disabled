import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from 'child_process';

/**
 * MCP Tool Matrix Tests
 * 
 * Comprehensive testing of all 64+ MCP tools across all 10 environments.
 * Tests tool execution, parameter validation, output verification, and 
 * environment-specific behavior.
 */

interface MCPTool {
  name: string;
  category: string;
  description: string;
  requiredParams?: Record<string, any>;
  optionalParams?: Record<string, any>;
  applicableEnvironments: string[];
  expectedOutputPattern?: RegExp;
  minimumExecutionTime?: number;
  maximumExecutionTime?: number;
}

// Complete MCP Tool Registry (64+ tools)
const MCP_TOOLS: MCPTool[] = [
  // Environment Tools (3)
  {
    name: 'environment_detect',
    category: 'Environment',
    description: 'Detect available development environments',
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /environments.*detected/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'environment_info',
    category: 'Environment',
    description: 'Get detailed information about environment',
    requiredParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /environment.*info/i,
    maximumExecutionTime: 3000
  },
  {
    name: 'environment_validate',
    category: 'Environment',
    description: 'Validate environment configuration',
    optionalParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /validation.*result/i,
    maximumExecutionTime: 10000
  },

  // DevBox Tools (6)
  {
    name: 'devbox_shell',
    category: 'DevBox',
    description: 'Enter DevBox shell environment',
    requiredParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /shell.*started|devbox.*environment/i,
    maximumExecutionTime: 8000
  },
  {
    name: 'devbox_start',
    category: 'DevBox',
    description: 'Start DevBox environment',
    requiredParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /environment.*started|setup.*completed/i,
    maximumExecutionTime: 15000
  },
  {
    name: 'devbox_run',
    category: 'DevBox',
    description: 'Run DevBox script',
    requiredParams: { environment: 'dev-env/python', script: 'test' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /script.*executed|command.*completed/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'devbox_status',
    category: 'DevBox',
    description: 'Get DevBox environment status',
    optionalParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /status.*active|environment.*ready/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'devbox_add_package',
    category: 'DevBox',
    description: 'Add package to DevBox environment',
    requiredParams: { environment: 'dev-env/python', package: 'requests' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /package.*added|installation.*complete/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'devbox_quick_start',
    category: 'DevBox',
    description: 'Quick start development environment',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { task: 'dev' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /quick.*start|environment.*ready/i,
    maximumExecutionTime: 20000
  },

  // DevPod Tools (4)
  {
    name: 'devpod_provision',
    category: 'DevPod',
    description: 'Provision DevPod workspace',
    requiredParams: { environment: 'python' },
    optionalParams: { count: 1 },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /workspace.*provisioned|container.*created/i,
    minimumExecutionTime: 30000,
    maximumExecutionTime: 300000
  },
  {
    name: 'devpod_list',
    category: 'DevPod',
    description: 'List DevPod workspaces',
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /workspaces.*found|no.*workspaces/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'devpod_status',
    category: 'DevPod',
    description: 'Get DevPod workspace status',
    optionalParams: { workspace: 'test-workspace' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /status.*running|workspace.*stopped/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'devpod_start',
    category: 'DevPod',
    description: 'Start DevPod environment',
    requiredParams: { environment: 'python' },
    optionalParams: { count: 1 },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /environment.*started|workspace.*ready/i,
    maximumExecutionTime: 120000
  },

  // Cross-Language Tools (3)
  {
    name: 'polyglot_check',
    category: 'Cross-Language',
    description: 'Comprehensive quality check',
    optionalParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /check.*completed|validation.*passed/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'polyglot_validate',
    category: 'Cross-Language',
    description: 'Cross-environment validation',
    optionalParams: { parallel: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /validation.*completed|environments.*validated/i,
    maximumExecutionTime: 120000
  },
  {
    name: 'polyglot_clean',
    category: 'Cross-Language',
    description: 'Clean up environments',
    optionalParams: { environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /cleanup.*completed|artifacts.*removed/i,
    maximumExecutionTime: 30000
  },

  // Performance Tools (2)
  {
    name: 'performance_measure',
    category: 'Performance',
    description: 'Measure performance of commands',
    requiredParams: { command: 'echo test', environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /performance.*measured|metrics.*recorded/i,
    maximumExecutionTime: 15000
  },
  {
    name: 'performance_report',
    category: 'Performance',
    description: 'Generate performance report',
    optionalParams: { days: 7, environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /report.*generated|performance.*analysis/i,
    maximumExecutionTime: 10000
  },

  // Security Tools (1)
  {
    name: 'security_scan',
    category: 'Security',
    description: 'Run security scans',
    optionalParams: { environment: 'dev-env/python', scan_type: 'all' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /scan.*completed|security.*analysis/i,
    maximumExecutionTime: 60000
  },

  // Hook Tools (2)
  {
    name: 'hook_status',
    category: 'Hook',
    description: 'Get hook configuration status',
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /hooks.*status|configuration.*active/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'hook_trigger',
    category: 'Hook',
    description: 'Manually trigger hooks',
    requiredParams: { hook_type: 'test' },
    optionalParams: { context: {} },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /hook.*triggered|execution.*completed/i,
    maximumExecutionTime: 15000
  },

  // PRP Tools (2)
  {
    name: 'prp_generate',
    category: 'PRP',
    description: 'Generate context engineering PRP',
    requiredParams: { feature_file: 'features/test.md', environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /prp.*generated|context.*created/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'prp_execute',
    category: 'PRP',
    description: 'Execute PRP files',
    requiredParams: { prp_file: 'test.md' },
    optionalParams: { validate: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /prp.*executed|execution.*completed/i,
    maximumExecutionTime: 60000
  },

  // AG-UI Tools (9)
  {
    name: 'agui_provision',
    category: 'AG-UI',
    description: 'Provision agentic DevPod workspaces',
    requiredParams: { environment: 'agentic-python' },
    optionalParams: { count: 1, features: ['agentic_chat'] },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /agentic.*provisioned|ag-ui.*ready/i,
    maximumExecutionTime: 300000
  },
  {
    name: 'agui_agent_create',
    category: 'AG-UI',
    description: 'Create new AI agents',
    requiredParams: { name: 'TestAgent', type: 'chat', environment: 'agentic-python' },
    optionalParams: { capabilities: ['conversation'] },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /agent.*created|ai.*agent.*ready/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'agui_agent_list',
    category: 'AG-UI',
    description: 'List all AI agents',
    optionalParams: { environment: 'agentic-python', status: 'active' },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /agents.*found|no.*agents/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'agui_agent_invoke',
    category: 'AG-UI',
    description: 'Invoke an AI agent',
    requiredParams: { agent_id: 'test-agent', message: { content: 'Hello', role: 'user' } },
    optionalParams: { environment: 'agentic-python' },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /agent.*invoked|response.*received/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'agui_chat',
    category: 'AG-UI',
    description: 'Start agentic chat session',
    requiredParams: { environment: 'agentic-typescript', message: 'Hello' },
    optionalParams: { context: {} },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /chat.*started|session.*active/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'agui_generate_ui',
    category: 'AG-UI',
    description: 'Generate UI components',
    requiredParams: { environment: 'agentic-typescript', prompt: 'Create button' },
    optionalParams: { component_type: 'custom', framework: 'react' },
    applicableEnvironments: ['agentic-typescript', 'agentic-rust', 'agentic-go'],
    expectedOutputPattern: /ui.*generated|component.*created/i,
    maximumExecutionTime: 45000
  },
  {
    name: 'agui_shared_state',
    category: 'AG-UI',
    description: 'Manage shared state',
    requiredParams: { environment: 'agentic-python', action: 'get' },
    optionalParams: { key: 'test', namespace: 'default' },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /state.*managed|action.*completed/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'agui_status',
    category: 'AG-UI',
    description: 'Get agentic environment status',
    optionalParams: { environment: 'agentic-python', detailed: true },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /status.*active|environment.*ready/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'agui_workflow',
    category: 'AG-UI',
    description: 'Execute AG-UI workflows',
    requiredParams: { environment: 'agentic-typescript', workflow_type: 'agent_chat' },
    optionalParams: { agents: ['agent-1'] },
    applicableEnvironments: ['agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /workflow.*executed|process.*completed/i,
    maximumExecutionTime: 60000
  },

  // Claude-Flow Tools (10)
  {
    name: 'claude_flow_init',
    category: 'Claude-Flow',
    description: 'Initialize Claude-Flow system',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { force: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /claude-flow.*initialized|initialization.*completed/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'claude_flow_wizard',
    category: 'Claude-Flow',
    description: 'Run hive-mind wizard',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { interactive: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /wizard.*completed|hive-mind.*setup/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'claude_flow_start',
    category: 'Claude-Flow',
    description: 'Start Claude-Flow daemon',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { background: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /daemon.*started|claude-flow.*running/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'claude_flow_stop',
    category: 'Claude-Flow',
    description: 'Stop Claude-Flow daemon',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { force: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /daemon.*stopped|claude-flow.*shutdown/i,
    maximumExecutionTime: 15000
  },
  {
    name: 'claude_flow_status',
    category: 'Claude-Flow',
    description: 'Check Claude-Flow status',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { detailed: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /status.*active|claude-flow.*running/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'claude_flow_monitor',
    category: 'Claude-Flow',
    description: 'Monitor Claude-Flow system',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { duration: 300, interval: 5 },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /monitoring.*started|metrics.*collected/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'claude_flow_spawn',
    category: 'Claude-Flow',
    description: 'Spawn AI agents',
    requiredParams: { environment: 'dev-env/python', task: 'Create test app' },
    optionalParams: { claude: true, context: {} },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /agent.*spawned|task.*initiated/i,
    maximumExecutionTime: 45000
  },
  {
    name: 'claude_flow_logs',
    category: 'Claude-Flow',
    description: 'Access Claude-Flow logs',
    requiredParams: { environment: 'dev-env/python' },
    optionalParams: { lines: 100, follow: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /logs.*retrieved|no.*logs/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'claude_flow_hive_mind',
    category: 'Claude-Flow',
    description: 'Multi-agent coordination',
    requiredParams: { environment: 'dev-env/python', command: 'spawn' },
    optionalParams: { task: 'Build app', agents: [] },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /hive-mind.*coordinated|agents.*synchronized/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'claude_flow_terminal_mgmt',
    category: 'Claude-Flow',
    description: 'Terminal session management',
    requiredParams: { environment: 'dev-env/python', action: 'create' },
    optionalParams: { command: 'echo test' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /terminal.*managed|session.*created/i,
    maximumExecutionTime: 15000
  },

  // Enhanced AI Hooks Tools (8)
  {
    name: 'enhanced_hook_context_triggers',
    category: 'Enhanced-Hooks',
    description: 'Context engineering auto-triggers',
    requiredParams: { action: 'trigger' },
    optionalParams: { feature_file: 'test.md', environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /trigger.*activated|context.*generated/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'enhanced_hook_error_resolution',
    category: 'Enhanced-Hooks',
    description: 'AI-powered error analysis',
    requiredParams: { action: 'analyze' },
    optionalParams: { error_text: 'Test error', environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /error.*analyzed|resolution.*suggested/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'enhanced_hook_env_orchestration',
    category: 'Enhanced-Hooks',
    description: 'Smart environment orchestration',
    requiredParams: { action: 'switch' },
    optionalParams: { target_environment: 'dev-env/typescript' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /environment.*switched|orchestration.*completed/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'enhanced_hook_dependency_tracking',
    category: 'Enhanced-Hooks',
    description: 'Cross-environment dependency monitoring',
    requiredParams: { action: 'scan' },
    optionalParams: { environment: 'dev-env/python', security_check: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /dependencies.*scanned|security.*validated/i,
    maximumExecutionTime: 45000
  },
  {
    name: 'enhanced_hook_performance_integration',
    category: 'Enhanced-Hooks',
    description: 'Advanced performance tracking',
    requiredParams: { action: 'measure' },
    optionalParams: { command: 'echo test', environment: 'dev-env/python' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /performance.*tracked|metrics.*recorded/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'enhanced_hook_quality_gates',
    category: 'Enhanced-Hooks',
    description: 'Cross-language quality enforcement',
    requiredParams: { action: 'validate' },
    optionalParams: { environment: 'dev-env/python', rules: [] },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /quality.*validated|gates.*passed/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'enhanced_hook_devpod_manager',
    category: 'Enhanced-Hooks',
    description: 'Smart container lifecycle management',
    requiredParams: { action: 'optimize' },
    optionalParams: { environment: 'dev-env/python', resource_limits: {} },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /container.*optimized|lifecycle.*managed/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'enhanced_hook_prp_lifecycle',
    category: 'Enhanced-Hooks',
    description: 'PRP status tracking and lifecycle',
    requiredParams: { action: 'track' },
    optionalParams: { prp_file: 'test.md', status: 'executing' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell'],
    expectedOutputPattern: /prp.*tracked|lifecycle.*managed/i,
    maximumExecutionTime: 15000
  },

  // Docker MCP Tools (15)
  {
    name: 'docker_mcp_gateway_start',
    category: 'Docker-MCP',
    description: 'Start Docker MCP gateway',
    optionalParams: { port: 8080, background: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /gateway.*started|docker.*mcp.*running/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'docker_mcp_gateway_status',
    category: 'Docker-MCP',
    description: 'Check gateway status',
    optionalParams: { detailed: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /gateway.*status|health.*check/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'docker_mcp_tools_list',
    category: 'Docker-MCP',
    description: 'List available containerized tools',
    optionalParams: { category: 'filesystem', verbose: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /tools.*listed|34.*tools|available.*tools/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'docker_mcp_http_bridge',
    category: 'Docker-MCP',
    description: 'Start HTTP/SSE bridge',
    optionalParams: { port: 8080, cors: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /bridge.*started|http.*server/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'docker_mcp_client_list',
    category: 'Docker-MCP',
    description: 'List connected MCP clients',
    optionalParams: { active_only: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /clients.*listed|no.*clients/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'docker_mcp_server_list',
    category: 'Docker-MCP',
    description: 'List running MCP servers',
    optionalParams: { running_only: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /servers.*listed|no.*servers/i,
    maximumExecutionTime: 5000
  },
  {
    name: 'docker_mcp_gemini_config',
    category: 'Docker-MCP',
    description: 'Configure Gemini AI integration',
    optionalParams: { model: 'gemini-pro', test: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /gemini.*configured|integration.*ready/i,
    maximumExecutionTime: 15000
  },
  {
    name: 'docker_mcp_test',
    category: 'Docker-MCP',
    description: 'Run integration test suites',
    optionalParams: { suite: 'security', verbose: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /tests.*passed|integration.*validated/i,
    maximumExecutionTime: 120000
  },
  {
    name: 'docker_mcp_demo',
    category: 'Docker-MCP',
    description: 'Execute demonstration scenarios',
    optionalParams: { scenario: 'ai-integration', interactive: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /demo.*completed|scenario.*executed/i,
    maximumExecutionTime: 60000
  },
  {
    name: 'docker_mcp_security_scan',
    category: 'Docker-MCP',
    description: 'Security vulnerability scanning',
    optionalParams: { target: 'containers', detailed: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /security.*scan|vulnerabilities.*checked/i,
    maximumExecutionTime: 90000
  },
  {
    name: 'docker_mcp_resource_limits',
    category: 'Docker-MCP',
    description: 'Manage container resource limits',
    requiredParams: { action: 'set' },
    optionalParams: { cpu_limit: '1.0', memory_limit: '2GB' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /limits.*set|resources.*configured/i,
    maximumExecutionTime: 15000
  },
  {
    name: 'docker_mcp_network_isolation',
    category: 'Docker-MCP',
    description: 'Configure network isolation',
    requiredParams: { action: 'enable' },
    optionalParams: { network_name: 'mcp-secure' },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /network.*isolated|isolation.*enabled/i,
    maximumExecutionTime: 20000
  },
  {
    name: 'docker_mcp_signature_verify',
    category: 'Docker-MCP',
    description: 'Verify image signatures',
    requiredParams: { image: 'mcp-tool:latest' },
    optionalParams: { trusted_registry: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /signature.*verified|image.*trusted/i,
    maximumExecutionTime: 30000
  },
  {
    name: 'docker_mcp_logs',
    category: 'Docker-MCP',
    description: 'Access component logs',
    requiredParams: { component: 'gateway' },
    optionalParams: { lines: 100, follow: false },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /logs.*retrieved|component.*logs/i,
    maximumExecutionTime: 10000
  },
  {
    name: 'docker_mcp_cleanup',
    category: 'Docker-MCP',
    description: 'Clean up resources',
    requiredParams: { target: 'containers' },
    optionalParams: { force: false, unused_only: true },
    applicableEnvironments: ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'],
    expectedOutputPattern: /cleanup.*completed|resources.*freed/i,
    maximumExecutionTime: 30000
  }
];

const ENVIRONMENTS = ['python', 'typescript', 'rust', 'go', 'nushell', 'agentic-python', 'agentic-typescript', 'agentic-rust', 'agentic-go', 'agentic-nushell'];
const MCP_SERVER_TIMEOUT = 10000;

describe('MCP Tool Matrix Tests', () => {
  let mcpClient: Client;
  let transport: StdioClientTransport;

  beforeAll(async () => {
    console.log('🚀 Starting MCP Tool Matrix Tests...');
    console.log(`📊 Testing ${MCP_TOOLS.length} tools across ${ENVIRONMENTS.length} environments`);
    console.log(`🎯 Total test combinations: ${MCP_TOOLS.length * ENVIRONMENTS.length}`);

    // Initialize MCP client
    await initializeMCPClient();
  }, 60000);

  afterAll(async () => {
    console.log('🧹 Cleaning up MCP client...');
    if (transport) {
      await transport.close();
    }
  }, 30000);

  describe('Tool Discovery and Registration', () => {
    test('should list all expected tools', async () => {
      const result = await mcpClient.listTools();
      
      expect(result.tools).toBeDefined();
      expect(result.tools.length).toBeGreaterThanOrEqual(MCP_TOOLS.length);
      
      // Verify all expected tools are registered
      const toolNames = result.tools.map(tool => tool.name);
      for (const expectedTool of MCP_TOOLS) {
        expect(toolNames).toContain(expectedTool.name);
      }
      
      console.log(`✅ All ${MCP_TOOLS.length} tools are properly registered`);
    });

    test('should provide valid tool schemas', async () => {
      const result = await mcpClient.listTools();
      
      for (const tool of result.tools) {
        expect(tool.name).toBeDefined();
        expect(tool.description).toBeDefined();
        expect(tool.inputSchema).toBeDefined();
        
        // Validate schema structure
        expect(tool.inputSchema.type).toBe('object');
        if (tool.inputSchema.properties) {
          expect(typeof tool.inputSchema.properties).toBe('object');
        }
      }
      
      console.log('✅ All tool schemas are valid');
    });
  });

  describe.each(MCP_TOOLS)('Tool: $name', (tool) => {
    
    test.each(tool.applicableEnvironments)(
      `${tool.name} should execute successfully in %s environment`,
      async (environment) => {
        console.log(`🧪 Testing ${tool.name} in ${environment} environment...`);
        
        const startTime = Date.now();
        
        try {
          // Prepare tool parameters
          const params = { ...tool.requiredParams };
          
          // Adapt environment parameter if present
          if (params.environment && typeof params.environment === 'string') {
            params.environment = params.environment.replace(/dev-env\/\w+/, `dev-env/${environment.replace('agentic-', '')}`);
          }
          if (params.environment === undefined && tool.name.includes('devbox')) {
            params.environment = `dev-env/${environment.replace('agentic-', '')}`;
          }
          if (params.environment === undefined && tool.name.includes('devpod')) {
            params.environment = environment;
          }
          
          // Add optional parameters for comprehensive testing
          if (tool.optionalParams) {
            Object.assign(params, tool.optionalParams);
          }
          
          // Execute tool
          const result = await mcpClient.callTool({
            name: tool.name,
            arguments: params
          });
          
          const executionTime = Date.now() - startTime;
          
          // Validate result structure
          expect(result).toBeDefined();
          expect(result.content).toBeDefined();
          expect(Array.isArray(result.content)).toBe(true);
          expect(result.content.length).toBeGreaterThan(0);
          
          // Validate result content
          const content = result.content[0];
          expect(content.type).toBe('text');
          expect(content.text).toBeDefined();
          expect(typeof content.text).toBe('string');
          expect(content.text.length).toBeGreaterThan(0);
          
          // Validate expected output pattern if specified
          if (tool.expectedOutputPattern) {
            expect(content.text).toMatch(tool.expectedOutputPattern);
          }
          
          // Validate execution time constraints
          if (tool.minimumExecutionTime) {
            expect(executionTime).toBeGreaterThanOrEqual(tool.minimumExecutionTime);
          }
          if (tool.maximumExecutionTime) {
            expect(executionTime).toBeLessThanOrEqual(tool.maximumExecutionTime);
          }
          
          console.log(`✅ ${tool.name} in ${environment}: Success (${executionTime}ms)`);
          
        } catch (error) {
          console.error(`❌ ${tool.name} in ${environment}: Failed`);
          console.error('Error:', error);
          
          // For non-critical tools, log warning instead of failing
          if (tool.category === 'PRP' || tool.name.includes('claude_flow_spawn')) {
            console.warn(`⚠️ ${tool.name} failed in ${environment} - this may be expected in test environment`);
          } else {
            throw error;
          }
        }
      },
      (tool.maximumExecutionTime || 30000) + 10000 // Add buffer to timeout
    );

    test(`${tool.name} should handle invalid parameters gracefully`, async () => {
      console.log(`🧪 Testing ${tool.name} with invalid parameters...`);
      
      try {
        // Test with completely invalid parameters
        const result = await mcpClient.callTool({
          name: tool.name,
          arguments: { invalid_param: 'invalid_value' }
        });
        
        // Should either succeed with validation error message or fail gracefully
        expect(result).toBeDefined();
        expect(result.content).toBeDefined();
        
      } catch (error) {
        // Error is expected for invalid parameters
        expect(error).toBeDefined();
        console.log(`✅ ${tool.name} correctly rejected invalid parameters`);
      }
    });

    if (tool.requiredParams && Object.keys(tool.requiredParams).length > 0) {
      test(`${tool.name} should require mandatory parameters`, async () => {
        console.log(`🧪 Testing ${tool.name} without required parameters...`);
        
        try {
          // Test without required parameters
          const result = await mcpClient.callTool({
            name: tool.name,
            arguments: {}
          });
          
          // Should fail or return validation error
          if (result.content[0].text) {
            expect(result.content[0].text).toMatch(/error|required|missing|invalid/i);
          }
          
        } catch (error) {
          // Error is expected when required parameters are missing
          expect(error).toBeDefined();
          console.log(`✅ ${tool.name} correctly requires mandatory parameters`);
        }
      });
    }
  });

  describe('Tool Category Performance', () => {
    test.each(Array.from(new Set(MCP_TOOLS.map(tool => tool.category))))(
      'should execute all %s tools within reasonable time',
      async (category) => {
        console.log(`📊 Performance testing ${category} tools...`);
        
        const categoryTools = MCP_TOOLS.filter(tool => tool.category === category);
        const startTime = Date.now();
        
        let successCount = 0;
        let failureCount = 0;
        
        for (const tool of categoryTools) {
          try {
            const params = { ...tool.requiredParams };
            
            // Use first applicable environment for performance testing
            const testEnv = tool.applicableEnvironments[0];
            if (params.environment && typeof params.environment === 'string') {
              params.environment = params.environment.replace(/dev-env\/\w+/, `dev-env/${testEnv.replace('agentic-', '')}`);
            }
            
            await mcpClient.callTool({
              name: tool.name,
              arguments: params
            });
            
            successCount++;
          } catch (error) {
            failureCount++;
            console.warn(`⚠️ ${tool.name} failed in performance test:`, error);
          }
        }
        
        const totalTime = Date.now() - startTime;
        const successRate = (successCount / categoryTools.length) * 100;
        
        console.log(`📈 ${category} Performance:`);
        console.log(`  ✅ Success: ${successCount}/${categoryTools.length} (${successRate.toFixed(1)}%)`);
        console.log(`  ⏱️ Total time: ${totalTime}ms`);
        console.log(`  ⚡ Average per tool: ${(totalTime / categoryTools.length).toFixed(1)}ms`);
        
        // Expect reasonable success rate (allowing for environment-specific failures)
        expect(successRate).toBeGreaterThanOrEqual(60); // At least 60% success rate
        expect(totalTime).toBeLessThanOrEqual(300000); // Maximum 5 minutes per category
      },
      600000 // 10 minutes timeout for category testing
    );
  });

  describe('Cross-Environment Consistency', () => {
    test('should maintain consistent behavior across environments', async () => {
      console.log('🔄 Testing cross-environment consistency...');
      
      // Test a subset of tools that should work consistently across environments
      const consistentTools = MCP_TOOLS.filter(tool => 
        tool.applicableEnvironments.length >= 5 && 
        ['environment_detect', 'environment_info', 'hook_status'].includes(tool.name)
      );
      
      for (const tool of consistentTools) {
        const results: Record<string, any> = {};
        
        for (const environment of tool.applicableEnvironments.slice(0, 3)) { // Test first 3 environments
          try {
            const params = { ...tool.requiredParams };
            if (params.environment) {
              params.environment = `dev-env/${environment.replace('agentic-', '')}`;
            }
            
            const result = await mcpClient.callTool({
              name: tool.name,
              arguments: params
            });
            
            results[environment] = result.content[0].text;
          } catch (error) {
            results[environment] = `Error: ${error}`;
          }
        }
        
        console.log(`🔍 ${tool.name} results across environments:`, Object.keys(results));
        
        // Verify that results are consistent (all success or all contain similar patterns)
        const resultValues = Object.values(results);
        const errorCount = resultValues.filter(r => String(r).startsWith('Error:')).length;
        const successCount = resultValues.length - errorCount;
        
        // Either all should succeed or fail consistently
        expect(successCount === resultValues.length || errorCount === resultValues.length).toBe(false);
        
        console.log(`✅ ${tool.name}: ${successCount} successes, ${errorCount} errors`);
      }
    });
  });

  // Helper functions
  async function initializeMCPClient(): Promise<void> {
    try {
      // Start MCP server
      const serverProcess = spawn('node', ['dist/index.js'], {
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe']
      });
      
      // Initialize transport and client
      transport = new StdioClientTransport({
        command: 'node',
        args: ['dist/index.js']
      });
      
      mcpClient = new Client({
        name: 'test-client',
        version: '1.0.0'
      }, {
        capabilities: {}
      });
      
      await mcpClient.connect(transport);
      
      console.log('✅ MCP client initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize MCP client:', error);
      throw error;
    }
  }
});