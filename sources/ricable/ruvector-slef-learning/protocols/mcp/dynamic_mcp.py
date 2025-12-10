"""
Dynamic MCP Implementation with Smart Search and Tool Composition
Based on Docker MCP Gateway patterns for Edge-Native AI SaaS

This module implements:
- mcp-find: Smart search for MCP servers in catalogs
- mcp-add: Dynamic server provisioning
- code-mode: Sandboxed tool composition via code execution
- State persistence across tool calls
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum
import asyncio
import httpx
import json
import uuid
import os


class MCPServerType(Enum):
    """Types of MCP servers."""
    STANDARD = "standard"  # Docker containers with stdio/SSE
    POCI = "poci"  # Per-Operation Container Invocation
    SELF_CONTAINED = "self-contained"  # OCI images with embedded metadata
    REMOTE = "remote"  # External MCP servers via SSE


@dataclass
class MCPServer:
    """MCP Server definition."""
    name: str
    description: str
    image: str
    server_type: MCPServerType = MCPServerType.STANDARD
    capabilities: List[str] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    volumes: Dict[str, str] = field(default_factory=dict)
    enabled: bool = False


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    server: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class CodeModeContext:
    """Context for code-mode execution."""
    session_id: str
    available_servers: List[str]
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)


class DynamicMCPGateway:
    """
    Dynamic MCP Gateway with Smart Search and Tool Composition.

    Implements Docker MCP Gateway patterns for:
    - Dynamic server discovery and provisioning
    - Code-mode sandboxed tool composition
    - Token-efficient tool management
    - State persistence across tool calls
    """

    def __init__(
        self,
        catalog_path: str = "~/.docker/mcp/docker-mcp.yaml",
        registry_path: str = "~/.docker/mcp/registry.yaml",
        gateway_url: str = "http://localhost:8080"
    ):
        self.catalog_path = os.path.expanduser(catalog_path)
        self.registry_path = os.path.expanduser(registry_path)
        self.gateway_url = gateway_url
        self.servers: Dict[str, MCPServer] = {}
        self.enabled_servers: List[str] = []
        self.tools: Dict[str, MCPTool] = {}
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.code_mode_contexts: Dict[str, CodeModeContext] = {}

    async def initialize(self):
        """Initialize the gateway and load catalog."""
        await self._load_catalog()
        await self._load_registry()
        await self._discover_capabilities()

    async def _load_catalog(self):
        """Load server catalog from YAML."""
        # In production, parse actual YAML file
        # For now, define built-in servers
        self.servers = {
            "ruvector": MCPServer(
                name="ruvector",
                description="High-performance vector database with HNSW indexing",
                image="ghcr.io/ruvnet/ruvector-mcp:latest",
                capabilities=["vector-search", "embedding-storage", "similarity-search"],
                tools=[
                    {"name": "vector_insert", "description": "Insert vectors into database"},
                    {"name": "vector_search", "description": "Search for similar vectors"},
                    {"name": "vector_delete", "description": "Delete vectors from database"}
                ]
            ),
            "agentdb": MCPServer(
                name="agentdb",
                description="Agent memory with causal reasoning and skill library",
                image="ghcr.io/ruvnet/agentdb-mcp:latest",
                capabilities=["agent-memory", "causal-reasoning", "reflexion-memory"],
                tools=[
                    {"name": "memory_store", "description": "Store agent memory"},
                    {"name": "memory_recall", "description": "Recall agent memory"},
                    {"name": "skill_learn", "description": "Learn new skill from experience"}
                ]
            ),
            "github-official": MCPServer(
                name="github-official",
                description="GitHub official MCP server for repository operations",
                image="ghcr.io/github/mcp-server:latest",
                capabilities=["git-operations", "issue-management", "pr-management"],
                tools=[
                    {"name": "list_repos", "description": "List repositories"},
                    {"name": "create_issue", "description": "Create an issue"},
                    {"name": "create_pr", "description": "Create a pull request"},
                    {"name": "get_file", "description": "Get file contents"}
                ]
            ),
            "e2b-sandbox": MCPServer(
                name="e2b-sandbox",
                description="Secure code execution in Firecracker microVMs",
                image="ghcr.io/e2b-dev/mcp-server:latest",
                capabilities=["code-execution", "python-runtime", "file-operations"],
                tools=[
                    {"name": "execute_python", "description": "Execute Python code"},
                    {"name": "execute_javascript", "description": "Execute JavaScript code"},
                    {"name": "read_file", "description": "Read file from sandbox"},
                    {"name": "write_file", "description": "Write file to sandbox"}
                ]
            ),
            "markdownify": MCPServer(
                name="markdownify",
                description="Convert web content to markdown",
                image="ghcr.io/mcp/markdownify:latest",
                capabilities=["web-scraping", "content-conversion"],
                tools=[
                    {"name": "fetch_url", "description": "Fetch and convert URL to markdown"},
                    {"name": "convert_html", "description": "Convert HTML to markdown"}
                ]
            )
        }

    async def _load_registry(self):
        """Load enabled servers from registry."""
        # Default enabled servers
        self.enabled_servers = ["ruvector", "agentdb", "e2b-sandbox"]

    async def _discover_capabilities(self):
        """Discover capabilities from enabled servers."""
        for server_name in self.enabled_servers:
            server = self.servers.get(server_name)
            if server:
                server.enabled = True
                for tool in server.tools:
                    tool_key = f"{server_name}:{tool['name']}"
                    self.tools[tool_key] = MCPTool(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        server=server_name,
                        input_schema=tool.get("inputSchema", {})
                    )

    # =========================================================================
    # mcp-find: Smart Search for MCP Servers
    # =========================================================================

    async def mcp_find(
        self,
        query: str,
        capability: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Smart search for MCP servers in the catalog.

        This is the 'mcp-find' dynamic tool that searches 230+ servers
        by name, description, or capability.

        Args:
            query: Search query (name, description keywords)
            capability: Optional capability filter
            limit: Maximum results to return

        Returns:
            List of matching server definitions
        """
        results = []

        for name, server in self.servers.items():
            score = 0

            # Search in name
            if query.lower() in name.lower():
                score += 10

            # Search in description
            if query.lower() in server.description.lower():
                score += 5

            # Search in capabilities
            for cap in server.capabilities:
                if query.lower() in cap.lower():
                    score += 3

            # Filter by capability if specified
            if capability and capability not in server.capabilities:
                continue

            if score > 0:
                results.append({
                    "name": name,
                    "description": server.description,
                    "image": server.image,
                    "capabilities": server.capabilities,
                    "tools_count": len(server.tools),
                    "enabled": server.enabled,
                    "score": score
                })

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # =========================================================================
    # mcp-add: Dynamic Server Provisioning
    # =========================================================================

    async def mcp_add(
        self,
        server_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dynamically add an MCP server to the current session.

        This is the 'mcp-add' dynamic tool that pulls servers
        just-in-time without manual configuration.

        Args:
            server_name: Name of server from catalog
            config: Optional configuration overrides

        Returns:
            Status of the server addition
        """
        if server_name not in self.servers:
            return {
                "success": False,
                "error": f"Server '{server_name}' not found in catalog"
            }

        server = self.servers[server_name]

        # Apply configuration overrides
        if config:
            server.env.update(config.get("env", {}))
            server.volumes.update(config.get("volumes", {}))

        # Enable the server
        if server_name not in self.enabled_servers:
            self.enabled_servers.append(server_name)
            server.enabled = True

            # Register tools
            for tool in server.tools:
                tool_key = f"{server_name}:{tool['name']}"
                self.tools[tool_key] = MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    server=server_name,
                    input_schema=tool.get("inputSchema", {})
                )

        return {
            "success": True,
            "server": server_name,
            "tools_added": len(server.tools),
            "capabilities": server.capabilities,
            "message": f"Server '{server_name}' is now available"
        }

    # =========================================================================
    # mcp-remove: Remove Server from Session
    # =========================================================================

    async def mcp_remove(self, server_name: str) -> Dict[str, Any]:
        """
        Remove an MCP server from the current session.

        Args:
            server_name: Name of server to remove

        Returns:
            Status of the removal
        """
        if server_name not in self.enabled_servers:
            return {
                "success": False,
                "error": f"Server '{server_name}' is not enabled"
            }

        # Disable the server
        self.enabled_servers.remove(server_name)
        server = self.servers.get(server_name)
        if server:
            server.enabled = False

        # Remove tools
        tools_to_remove = [k for k in self.tools if k.startswith(f"{server_name}:")]
        for tool_key in tools_to_remove:
            del self.tools[tool_key]

        return {
            "success": True,
            "server": server_name,
            "tools_removed": len(tools_to_remove),
            "message": f"Server '{server_name}' has been removed"
        }

    # =========================================================================
    # code-mode: Sandboxed Tool Composition
    # =========================================================================

    async def code_mode(
        self,
        code: str,
        servers: List[str],
        context_id: Optional[str] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute code that composes multiple MCP tools in a sandbox.

        This is the 'code-mode' dynamic tool that allows agents to write
        code calling tools from multiple MCP servers in a secure sandbox.

        Key benefits:
        - Secure: Code runs in containerized sandbox
        - Token efficient: Tools don't need to be sent to model each turn
        - State persistent: Variables persist across calls in session

        Args:
            code: JavaScript/Python code to execute
            servers: List of MCP server names to make available
            context_id: Optional context ID for state persistence
            timeout: Execution timeout in seconds

        Returns:
            Execution result with stdout, stderr, and artifacts
        """
        # Create or retrieve context
        if context_id and context_id in self.code_mode_contexts:
            context = self.code_mode_contexts[context_id]
        else:
            context_id = context_id or str(uuid.uuid4())
            context = CodeModeContext(
                session_id=context_id,
                available_servers=servers
            )
            self.code_mode_contexts[context_id] = context

        # Ensure all requested servers are available
        for server_name in servers:
            if server_name not in self.enabled_servers:
                await self.mcp_add(server_name)

        # Build the execution environment
        # In production, this would use Docker to run the code
        execution_env = self._build_execution_environment(servers, context)

        try:
            # Execute code in sandbox
            # This would be a Docker container in production
            result = await self._execute_in_sandbox(
                code=code,
                env=execution_env,
                timeout=timeout
            )

            # Update context with execution history
            context.execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "code": code,
                "result": result
            })

            # Extract any artifacts
            if result.get("artifacts"):
                context.artifacts.extend(result["artifacts"])

            # Persist any variables
            if result.get("variables"):
                context.variables.update(result["variables"])

            return {
                "success": True,
                "context_id": context_id,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "return_value": result.get("return_value"),
                "artifacts": result.get("artifacts", []),
                "execution_time_ms": result.get("execution_time_ms", 0)
            }

        except Exception as e:
            return {
                "success": False,
                "context_id": context_id,
                "error": str(e)
            }

    def _build_execution_environment(
        self,
        servers: List[str],
        context: CodeModeContext
    ) -> Dict[str, Any]:
        """Build the execution environment with MCP tool bindings."""
        env = {
            "MCP_SERVERS": servers,
            "MCP_GATEWAY_URL": self.gateway_url,
            "SESSION_ID": context.session_id,
            "VARIABLES": context.variables
        }

        # Generate tool bindings for each server
        tool_bindings = {}
        for server_name in servers:
            server = self.servers.get(server_name)
            if server:
                tool_bindings[server_name] = {
                    tool["name"]: f"mcp.{server_name}.{tool['name']}"
                    for tool in server.tools
                }

        env["TOOL_BINDINGS"] = tool_bindings
        return env

    async def _execute_in_sandbox(
        self,
        code: str,
        env: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed Docker container.

        In production, this would:
        1. Spin up a Docker container with the code-mode runtime
        2. Mount volumes for state persistence
        3. Inject MCP tool bindings
        4. Execute the code
        5. Return results and artifacts
        """
        # Simulated sandbox execution
        # In production, use Docker SDK or subprocess

        # Wrap code with MCP tool access
        wrapped_code = f"""
# MCP Tool Access Layer
import json
import httpx

class MCPClient:
    def __init__(self, gateway_url, servers):
        self.gateway_url = gateway_url
        self.servers = servers
        self.client = httpx.Client(timeout=30.0)

    def call_tool(self, server, tool, params):
        response = self.client.post(
            f"{{self.gateway_url}}/mcp/{{server}}/{{tool}}",
            json=params
        )
        return response.json()

# Initialize MCP client
mcp = MCPClient("{env['MCP_GATEWAY_URL']}", {env['MCP_SERVERS']})

# Restore session variables
variables = {json.dumps(env.get('VARIABLES', {}))}

# User code execution
{code}
"""
        # This is a placeholder - actual execution would use Docker
        return {
            "stdout": f"Code executed with servers: {env['MCP_SERVERS']}",
            "stderr": "",
            "return_value": None,
            "artifacts": [],
            "execution_time_ms": 100
        }

    # =========================================================================
    # Tool Invocation
    # =========================================================================

    async def invoke_tool(
        self,
        server: str,
        tool: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a tool on an MCP server.

        Args:
            server: Server name
            tool: Tool name
            params: Tool parameters

        Returns:
            Tool execution result
        """
        if server not in self.enabled_servers:
            return {"error": f"Server '{server}' is not enabled"}

        tool_key = f"{server}:{tool}"
        if tool_key not in self.tools:
            return {"error": f"Tool '{tool}' not found on server '{server}'"}

        try:
            response = await self.http_client.post(
                f"{self.gateway_url}/mcp/{server}/{tool}",
                json=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Dynamic Tool Registration for Agents
    # =========================================================================

    def get_dynamic_tools(self) -> List[Dict[str, Any]]:
        """
        Get the dynamic MCP tools for agent integration.

        Returns minimal tool set for token efficiency:
        - mcp-find: Search catalog
        - mcp-add: Add server
        - mcp-remove: Remove server
        - code-mode: Compose tools via code
        """
        return [
            {
                "name": "mcp-find",
                "description": "Search MCP server catalog (230+ servers) by name, description, or capability",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "capability": {
                            "type": "string",
                            "description": "Filter by capability"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp-add",
                "description": "Add MCP server to current session (just-in-time provisioning)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "server_name": {
                            "type": "string",
                            "description": "Server name from catalog"
                        },
                        "config": {
                            "type": "object",
                            "description": "Optional configuration"
                        }
                    },
                    "required": ["server_name"]
                }
            },
            {
                "name": "mcp-remove",
                "description": "Remove MCP server from current session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "server_name": {
                            "type": "string",
                            "description": "Server name to remove"
                        }
                    },
                    "required": ["server_name"]
                }
            },
            {
                "name": "code-mode",
                "description": "Execute code that composes multiple MCP tools in a secure sandbox. Enables multi-tool workflows with state persistence.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python/JavaScript code to execute"
                        },
                        "servers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "MCP servers to make available"
                        },
                        "context_id": {
                            "type": "string",
                            "description": "Context ID for state persistence"
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 60
                        }
                    },
                    "required": ["code", "servers"]
                }
            }
        ]

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Example usage patterns for Dynamic MCPs
DYNAMIC_MCP_EXAMPLES = """
# Example 1: Smart Search for MCP Servers
result = await gateway.mcp_find("github", capability="git-operations")
# Returns: GitHub official server with all its tools

# Example 2: Add Server Just-in-Time
await gateway.mcp_add("github-official")
await gateway.mcp_add("markdownify")
# Servers are now available without restart

# Example 3: Code-Mode for Multi-Tool Composition
code = '''
# Fetch README from GitHub and convert to markdown
readme_content = mcp.call_tool("github-official", "get_file", {
    "repo": "ruvnet/agentic-flow",
    "path": "README.md"
})

# Convert any HTML in the README to clean markdown
clean_markdown = mcp.call_tool("markdownify", "convert_html", {
    "content": readme_content["content"]
})

# Store in vector database for semantic search
mcp.call_tool("ruvector", "vector_insert", {
    "content": clean_markdown,
    "metadata": {"source": "github", "repo": "ruvnet/agentic-flow"}
})

return {"status": "processed", "length": len(clean_markdown)}
'''

result = await gateway.code_mode(
    code=code,
    servers=["github-official", "markdownify", "ruvector"],
    context_id="readme-processor-session"
)
# Code runs in sandbox with access to all three MCP servers
# State persists across calls in the same context
"""
