# ruv-swarm MCP Integration

This directory contains the Model Context Protocol (MCP) configuration for ruv-swarm integration with Claude Code.

## Configuration Files

### mcp.json
Contains the complete MCP server configuration for ruv-swarm with:
- stdio protocol for seamless Claude Code communication
- 12 comprehensive swarm orchestration tools
- WebAssembly-powered neural network operations
- Cognitive diversity agent management

## Setup Instructions

The ruv-swarm MCP server has been added to Claude with:

```bash
claude mcp add ruv-swarm npx -- ruv-swarm mcp start --protocol=stdio
```

### Verify Installation

```bash
# List configured MCP servers
claude mcp list

# Get server details
claude mcp get ruv-swarm
```

### Available MCP Tools

🔧 **Swarm Management**
- `swarm_init` - Initialize swarm topology (mesh, hierarchical, ring, star)
- `swarm_status` - Get current swarm status and agent information
- `swarm_monitor` - Monitor swarm activity in real-time

🤖 **Agent Operations**
- `agent_spawn` - Spawn new agents with specified capabilities
- `agent_list` - List all active agents in the swarm
- `agent_metrics` - Get performance metrics for agents

📋 **Task Management**
- `task_orchestrate` - Orchestrate distributed tasks across the swarm
- `task_status` - Check progress of running tasks
- `task_results` - Retrieve results from completed tasks

🔬 **Analytics**
- `benchmark_run` - Execute performance benchmarks
- `features_detect` - Detect runtime features and capabilities
- `memory_usage` - Get current memory usage statistics

## Usage

Once configured, Claude Code can directly interact with ruv-swarm through MCP tools for:
- Distributed agent orchestration
- Neural network swarm operations
- Real-time performance monitoring
- Cognitive diversity task distribution

## Technical Details

- **Protocol**: stdio (JSON-RPC 2024-11-05)
- **Command**: `npx ruv-swarm mcp start --protocol=stdio`
- **Working Directory**: `/workspaces/ruv-FANN/ruv-swarm/npm`
- **Performance**: Sub-millisecond WASM initialization
- **Architecture**: Distributed multi-agent system with cognitive diversity

## Troubleshooting

If the MCP server isn't responding:

1. Ensure ruv-swarm is installed: `npm install -g ruv-swarm`
2. Verify MCP configuration: `claude mcp get ruv-swarm`
3. Check server logs: `npx ruv-swarm mcp status`
4. Restart Claude Code if needed

## Resources

- [ruv-swarm Documentation](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Claude Code Integration Guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/npm#claude-code-integration)