"""
E2B Sandbox Integration for Edge-Native AI SaaS
Secure code execution in Firecracker microVMs

Docker + E2B Partnership: Building Trusted AI
- Hardware-level isolation via Firecracker
- Full Python/JS ecosystem support
- Artifact persistence and retrieval
- Hybrid local/cloud execution
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import os
import uuid


@dataclass
class SandboxConfig:
    """Configuration for E2B sandbox."""
    template: str = "python3"
    timeout: int = 60
    max_memory_mb: int = 512
    max_cpu_cores: int = 2
    persist_artifacts: bool = True
    network_enabled: bool = True
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    execution_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[Any] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: int = 0
    sandbox_id: Optional[str] = None
    error: Optional[str] = None


class E2BSandbox:
    """
    E2B Sandbox Client for secure AI code execution.

    Uses Firecracker microVMs for hardware-level isolation.
    Provides full access to Python/JS ecosystems including:
    - pandas, numpy, scikit-learn
    - Data visualization libraries
    - File I/O operations
    - Network access (configurable)

    Integration with Docker MCP Gateway:
    - Runs as containerized MCP server
    - Accessible via code-mode tool composition
    - State persistence via volumes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.e2b.dev"
    ):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.base_url = base_url
        self.active_sandboxes: Dict[str, Any] = {}

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60,
        dependencies: Optional[List[str]] = None,
        config: Optional[SandboxConfig] = None
    ) -> ExecutionResult:
        """
        Execute code in a secure E2B sandbox.

        Args:
            code: Code to execute
            language: Programming language (python, javascript)
            timeout: Execution timeout in seconds
            dependencies: pip/npm packages to install
            config: Optional sandbox configuration

        Returns:
            ExecutionResult with output and artifacts
        """
        execution_id = str(uuid.uuid4())
        config = config or SandboxConfig(timeout=timeout)
        start_time = datetime.utcnow()

        try:
            # Create sandbox (in production, use e2b SDK)
            sandbox_id = await self._create_sandbox(config, language)

            # Install dependencies if specified
            if dependencies:
                await self._install_dependencies(sandbox_id, dependencies, language)

            # Execute code
            result = await self._run_code(sandbox_id, code, language, timeout)

            # Collect artifacts
            artifacts = []
            if config.persist_artifacts:
                artifacts = await self._collect_artifacts(sandbox_id)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ExecutionResult(
                execution_id=execution_id,
                success=True,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                return_value=result.get("return_value"),
                artifacts=artifacts,
                execution_time_ms=int(execution_time),
                sandbox_id=sandbox_id
            )

        except Exception as e:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                error=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )

    async def _create_sandbox(
        self,
        config: SandboxConfig,
        language: str
    ) -> str:
        """Create a new sandbox instance."""
        # In production, use e2b SDK:
        # from e2b import Sandbox
        # sandbox = await Sandbox.create(template=config.template)

        sandbox_id = f"sandbox-{uuid.uuid4().hex[:8]}"
        self.active_sandboxes[sandbox_id] = {
            "config": config,
            "language": language,
            "created_at": datetime.utcnow().isoformat()
        }
        return sandbox_id

    async def _install_dependencies(
        self,
        sandbox_id: str,
        dependencies: List[str],
        language: str
    ):
        """Install dependencies in sandbox."""
        if language == "python":
            # pip install
            cmd = f"pip install {' '.join(dependencies)}"
        elif language == "javascript":
            # npm install
            cmd = f"npm install {' '.join(dependencies)}"
        else:
            return

        # Execute install command
        await self._run_code(sandbox_id, cmd, "shell", timeout=120)

    async def _run_code(
        self,
        sandbox_id: str,
        code: str,
        language: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Execute code in sandbox."""
        # Simulated execution - in production use e2b SDK
        # result = await sandbox.run_code(code)

        return {
            "stdout": f"Executed in sandbox {sandbox_id}",
            "stderr": "",
            "return_value": None
        }

    async def _collect_artifacts(
        self,
        sandbox_id: str
    ) -> List[Dict[str, Any]]:
        """Collect artifacts from sandbox."""
        # In production, list and download files
        # files = await sandbox.filesystem.list("/artifacts")
        return []

    async def cleanup(self, sandbox_id: Optional[str] = None):
        """Clean up sandbox(es)."""
        if sandbox_id:
            if sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox_id]
        else:
            self.active_sandboxes.clear()


class LocalWasmSandbox:
    """
    Local WebAssembly sandbox for lightweight code execution.

    Uses SpinKube/Wasm runtime for:
    - Millisecond cold starts
    - Memory-safe execution
    - No network access by default (capability-based)
    - Pure Python only (no C extensions)

    Best for:
    - Simple calculations
    - Text processing
    - JSON manipulation
    - Quick logic execution
    """

    def __init__(self, runtime: str = "wasmtime"):
        self.runtime = runtime

    async def execute(
        self,
        code: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code in local Wasm sandbox.

        Limitations:
        - No pandas, numpy (C extensions)
        - Limited file I/O
        - No network by default
        """
        execution_id = str(uuid.uuid4())

        try:
            # In production, use componentize-py for Wasm
            # This would compile Python to Wasm and execute

            return {
                "execution_id": execution_id,
                "success": True,
                "stdout": "Executed in Wasm sandbox",
                "stderr": "",
                "execution_time_ms": 5  # Millisecond cold start
            }

        except Exception as e:
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e)
            }


class HybridSandboxRouter:
    """
    Intelligent routing between E2B and local Wasm sandboxes.

    Decision criteria:
    - Code complexity (simple logic vs. data science)
    - Dependencies required (C extensions → E2B)
    - Security requirements (air-gapped → local)
    - Latency requirements (instant → Wasm)
    - Cost optimization (local first → E2B fallback)
    """

    def __init__(
        self,
        e2b_sandbox: E2BSandbox,
        wasm_sandbox: LocalWasmSandbox
    ):
        self.e2b = e2b_sandbox
        self.wasm = wasm_sandbox

        # Libraries requiring C extensions (need E2B)
        self.e2b_required_libs = {
            "pandas", "numpy", "scipy", "scikit-learn",
            "matplotlib", "seaborn", "tensorflow", "torch",
            "cv2", "opencv", "PIL", "pillow"
        }

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code to determine best sandbox."""
        import_lines = [
            line.strip() for line in code.split('\n')
            if line.strip().startswith(('import ', 'from '))
        ]

        # Extract imported modules
        modules = set()
        for line in import_lines:
            if line.startswith('import '):
                mod = line.split()[1].split('.')[0]
                modules.add(mod)
            elif line.startswith('from '):
                mod = line.split()[1].split('.')[0]
                modules.add(mod)

        # Check for E2B-required libraries
        needs_e2b = bool(modules & self.e2b_required_libs)

        # Check for file I/O
        has_file_io = 'open(' in code or 'read(' in code or 'write(' in code

        # Check for network
        has_network = 'requests' in code or 'httpx' in code or 'urllib' in code

        return {
            "modules": list(modules),
            "needs_e2b": needs_e2b,
            "has_file_io": has_file_io,
            "has_network": has_network,
            "recommended_sandbox": "e2b" if needs_e2b else "wasm"
        }

    async def execute(
        self,
        code: str,
        sandbox_mode: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute code with intelligent sandbox routing.

        Args:
            code: Code to execute
            sandbox_mode: "auto", "e2b", "wasm", or "local"
            **kwargs: Additional execution parameters

        Returns:
            Execution result
        """
        if sandbox_mode == "auto":
            analysis = self.analyze_code(code)
            sandbox_mode = analysis["recommended_sandbox"]

        if sandbox_mode == "e2b":
            result = await self.e2b.execute(code, **kwargs)
            return {
                "sandbox_type": "e2b",
                **result.__dict__
            }
        else:
            result = await self.wasm.execute(code, **kwargs)
            return {
                "sandbox_type": "wasm",
                **result
            }


# MCP Server definition for E2B sandbox
E2B_MCP_SERVER = {
    "name": "e2b-sandbox",
    "description": "Secure code execution in Firecracker microVMs",
    "version": "1.0.0",
    "tools": [
        {
            "name": "execute_python",
            "description": "Execute Python code with full library support (pandas, numpy, etc.)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "pip packages to install"
                    },
                    "timeout": {"type": "integer", "default": 60}
                },
                "required": ["code"]
            }
        },
        {
            "name": "execute_javascript",
            "description": "Execute JavaScript/Node.js code",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript code to execute"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "npm packages to install"
                    },
                    "timeout": {"type": "integer", "default": 60}
                },
                "required": ["code"]
            }
        },
        {
            "name": "file_read",
            "description": "Read file from sandbox filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "file_write",
            "description": "Write file to sandbox filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "install_packages",
            "description": "Install Python or Node.js packages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Packages to install"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript"],
                        "default": "python"
                    }
                },
                "required": ["packages"]
            }
        }
    ],
    "capabilities": [
        "code-execution",
        "python-runtime",
        "javascript-runtime",
        "file-operations",
        "package-management"
    ]
}
