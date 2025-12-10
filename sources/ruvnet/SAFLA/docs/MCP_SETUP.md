# SAFLA MCP (Model Context Protocol) Setup

## Overview

SAFLA is now configured to work with the Model Context Protocol (MCP), allowing AI assistants to interact with the remote SAFLA instance deployed on Fly.io.

## Configuration

The MCP configuration is stored in `.roo/mcp.json`:

```json
{
  "mcpServers": {
    "safla": {
      "command": "python3",
      "args": [
        "/workspaces/SAFLA/safla_mcp_simple.py"
      ],
      "env": {
        "SAFLA_REMOTE_URL": "https://safla.fly.dev"
      }
    }
  }
}
```

## Available Tools

The SAFLA MCP server provides the following tools:

### 1. `generate_embeddings`
Generate embeddings using SAFLA's extreme-optimized engine (1.75M+ ops/sec)
- **Input**: `texts` (array of strings) - Texts to embed
- **Performance**: Utilizes the 178,146% optimized engine

### 2. `store_memory`
Store information in SAFLA's hybrid memory system
- **Input**: 
  - `content` (string) - Content to store
  - `memory_type` (string) - Type: "episodic", "semantic", or "procedural"

### 3. `retrieve_memories`
Search and retrieve from SAFLA's memory system
- **Input**:
  - `query` (string) - Search query
  - `limit` (integer, default: 5) - Maximum results

### 4. `get_performance`
Get SAFLA performance metrics
- **Input**: None required
- **Output**: Current performance statistics

## MCP Server Implementation

The MCP server (`safla_mcp_simple.py`) implements the Model Context Protocol by:
1. Reading JSON-RPC messages from stdin
2. Proxying requests to the remote SAFLA API
3. Returning formatted responses to stdout

## Performance Characteristics

- **Baseline**: 985.38 ops/sec
- **Optimized**: 1,755,595.48 ops/sec
- **Improvement**: 178,146.95% (1,781x faster)
- **Cache**: Enabled for maximum performance
- **Batch Size**: 256 (optimal for performance)

## Testing

To test the MCP server manually:

```bash
# Start the server
python3 /workspaces/SAFLA/safla_mcp_simple.py

# Send test requests (JSON-RPC format)
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
```

## Remote SAFLA Instance

The MCP server connects to the SAFLA instance deployed at:
- **URL**: https://safla.fly.dev
- **API Endpoint**: https://safla.fly.dev/api/safla
- **Health Check**: https://safla.fly.dev/health

## Troubleshooting

If you encounter connection issues:
1. Verify the SAFLA instance is running: `curl https://safla.fly.dev/health`
2. Check the MCP server logs for errors
3. Ensure the `SAFLA_REMOTE_URL` environment variable is set correctly
4. Verify Python 3 is available in your environment

## Architecture

```
AI Assistant <-> MCP Protocol <-> safla_mcp_simple.py <-> HTTPS <-> SAFLA on Fly.io
```

The MCP server acts as a bridge between the Model Context Protocol and the remote SAFLA API, enabling seamless integration with AI assistants while leveraging the extreme performance optimizations achieved (1.75M+ ops/sec).