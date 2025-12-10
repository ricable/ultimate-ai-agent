Unified Agentic Platform (UAP) - Complete Implementation Blueprint for AI Coder
To the AI Coding Agent: You are a world-class AI agent developer. Your task is to implement the Unified Agentic Platform (UAP) based on the following comprehensive blueprint. Follow the instructions sequentially, creating the specified files and directories with the provided content. This document contains all the necessary context, architecture, code, and configuration for the complete implementation.
1. Project Setup & Implementation Guide
First, set up the project structure and core configuration files. This is the foundation upon which the entire platform will be built.
1.1. Create Project Directory Structure
Execute the following commands to create the necessary directories:
bashmkdir -p uap-platform/frontend/src/{components/{agents,chat,common,ui},hooks,types,utils,stores,examples,app}
mkdir -p uap-platform/backend/{frameworks/{copilot,agno,mastra},services,processors,distributed,tests}
mkdir -p uap-platform/scripts
mkdir -p uap-platform/skypilot
mkdir -p uap-platform/docs
mkdir -p uap-platform/.github/workflows

# Create Python __init__.py files for proper module structure
touch uap-platform/backend/__init__.py
touch uap-platform/backend/services/__init__.py
touch uap-platform/backend/frameworks/__init__.py
touch uap-platform/backend/frameworks/{copilot,agno,mastra}/__init__.py
touch uap-platform/backend/processors/__init__.py
touch uap-platform/backend/distributed/__init__.py
touch uap-platform/backend/tests/__init__.py
1.2. Development Environment: devbox.json
Create the main development environment configuration. This file ensures a reproducible environment for all developers and for the agent itself.
File: uap-platform/devbox.json
json{
  "packages": [
    "nodejs@20",
    "python@3.11",
    "git",
    "curl",
    "jq",
    "docker",
    "kubectl",
    "terraform@1.6",
    "teller@1.5.6",
    "nushell@0.90",
    "uv@0.1.15"
  ],
  "shell": {
    "init_hook": [
      "echo 'Welcome to UAP Development Environment'",
      "echo 'Node.js version:' && node --version",
      "echo 'Python version:' && python --version",
      "echo 'Available commands: uap-agent-status, uap-deploy-agent, etc.'",
      "export UAP_ENV=development",
      "export PATH=$PWD/scripts:$PATH",
      "if [ -f frontend/package.json ]; then cd frontend && npm install && cd ..; fi",
      "if [ -f backend/requirements.txt ]; then cd backend && pip install -r requirements.txt && cd ..; fi"
    ],
    "scripts": {
      "dev": [
        "echo 'Starting UAP development servers...'",
        "teller run -- concurrently \"npm --prefix frontend run dev\" \"python -m uvicorn backend.main:app --reload --host 0.0.0.0\""
      ],
      "build": [
        "echo 'Building UAP for production...'",
        "npm --prefix frontend run build",
        "cd backend && python -m build"
      ],
      "test": [
        "echo 'Running UAP test suite...'",
        "npm --prefix frontend test",
        "cd backend && pytest"
      ],
      "deploy": [
        "echo 'Deploying UAP to cloud...'",
        "teller run -- skypilot up -c skypilot/uap-production.yaml"
      ]
    }
  },
  "env": {
    "UAP_VERSION": "3.0.0",
    "NODE_ENV": "development",
    "PYTHONPATH": "./backend"
  }
}
1.3. Frontend Dependencies: package.json
Create the package.json for the React frontend.
File: uap-platform/frontend/package.json
json{
  "name": "uap-frontend",
  "private": true,
  "version": "3.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview",
    "test": "vitest"
  },
  "dependencies": {
    "@ag-ui/client": "^0.1.0",
    "@copilotkit/react-core": "^0.6.0",
    "@copilotkit/react-ui": "^0.6.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-progress": "^1.0.3",
    "@radix-ui/react-slot": "^1.0.2",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "events": "^3.3.0",
    "lucide-react": "^0.372.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "tailwind-merge": "^2.2.2",
    "tailwindcss-animate": "^1.0.7",
    "zustand": "^4.5.2"
  },
  "devDependencies": {
    "@types/node": "^20.12.7",
    "@types/react": "^18.2.66",
    "@types/react-dom": "^18.2.22",
    "@typescript-eslint/eslint-plugin": "^7.2.0",
    "@typescript-eslint/parser": "^7.2.0",
    "@vitejs/plugin-react-swc": "^3.5.0",
    "autoprefixer": "^10.4.19",
    "eslint": "^8.57.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.6",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "typescript": "^5.2.2",
    "vite": "^5.2.0",
    "vitest": "^1.4.0"
  }
}
1.4. Backend Dependencies: requirements.txt
Create the requirements.txt for the Python backend.
File: uap-platform/backend/requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-dotenv==1.0.0
pyjwt==2.8.0
passlib[bcrypt]==1.7.4
pydantic==2.5.3
httpx==0.26.0
websockets==12.0
pytest==7.4.4
pytest-asyncio==0.23.3

# Agent Frameworks (to be added when available)
# agno
# mastra-ai
# copilotkit-sdk

# Infrastructure
# teller-cli
# skypilot
# docling
ray[serve,data]==2.9.1
# mlx-lm

# Other
requests==2.31.0
aiofiles==23.2.1
1.5. Secrets Management: .teller.yml
Create the Teller configuration for universal secrets management.
File: uap-platform/.teller.yml
yaml# .teller.yml
project: uap-production
opts:
  region: us-central1
  stage: production

providers:
  google_secret_manager:
    env_sync:
      path: secretmanager/projects/uap-prod/secrets/
  
  hashicorp_vault:
    env_sync:
      path: kv/data/uap/
      address: https://vault.company.com
      role_id: "{{ .Env.VAULT_ROLE_ID }}"

env:
  # LLM API Keys
  OPENAI_API_KEY:
    provider: google_secret_manager
    path: openai-api-key/versions/latest
    redact_with: '***OPENAI***'
    
  ANTHROPIC_API_KEY:
    provider: google_secret_manager
    path: anthropic-api-key/versions/latest
    redact_with: '***ANTHROPIC***'
    
  # Database Connections
  DATABASE_URL:
    provider: google_secret_manager
    path: database-url/versions/latest
    redact_with: '***DB***'
    
  # Application Secrets
  JWT_SECRET:
    provider: google_secret_manager
    path: jwt-secret/versions/latest
    redact_with: '***JWT***'
    
  # External Service Keys
  TAVILY_API_KEY:
    provider: google_secret_manager
    path: tavily-api-key/versions/latest
    redact_with: '***TAVILY***'
1.6. Environment Variables Template
File: uap-platform/.env.example
bash# Backend
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# LLM Keys (managed by Teller in production)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost/uap

# Security
JWT_SECRET=your-secret-key-here
1.7. README.md
File: uap-platform/README.md
markdown# Unified Agentic Platform (UAP)

## Overview
UAP is an end-to-end platform for building, deploying, and operating AI agents using multiple frameworks.

## Quick Start

### Prerequisites
- Devbox installed
- Docker (optional)

### Development Setup
```bash
# Clone the repository
git clone <repo-url>
cd uap-platform

# Enter the devbox shell
devbox shell

# The environment will auto-setup on first run
# Start development servers
devbox run dev
Production Deployment
bash# Deploy to cloud
devbox run deploy
Architecture

Frontend: React + TypeScript + Vite
Backend: Python + FastAPI
Protocol: AG-UI
Frameworks: CopilotKit, Agno, Mastra

Documentation
See /docs for detailed documentation.

### 1.8. Docker Configuration

**File: uap-platform/Dockerfile**

```dockerfile
# Multi-stage build for production
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
1.9. GitHub Actions CI/CD
File: uap-platform/.github/workflows/ci.yml
yamlname: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20'
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        cd frontend && npm ci
        cd ../backend && pip install -r requirements.txt
    - name: Run tests
      run: |
        cd frontend && npm test
        cd ../backend && pytest
2. Executive Summary
The Unified Agentic Platform (UAP) is an internal, end-to-end ecosystem designed to drastically simplify the lifecycle of building, deploying, and operating sophisticated, multi-agent AI systems. Built with a modern TypeScript-first architecture for the frontend and a high-performance Python backend, UAP provides developers with curated "golden paths" while abstracting infrastructure complexity and championing a secure, component-based architecture.
Core Innovation:

Agent-First Design: Purpose-built for AI agent development and deployment.
Protocol Standardization: AG-UI protocol ensures cross-framework interoperability between the frontend and any backend agent.
Multi-Framework Integration: Unifies CopilotKit, Agno, and Mastra under a single orchestration layer.
Type-Safe Development: Full TypeScript support on the frontend for robust UIs.
Modern Tooling: Vite.js for a blazing-fast development experience and Devbox for reproducible environments.

Base Reference Applications:

Frontend: Dojo App from AG-UI Protocol
Backend: CopilotKit with Agno starter template

3. Problem Statement & Vision
Current Challenges

Fragmented Tooling: Developers manually stitch together local environments, ML frameworks, and deployment tools, creating high cognitive load.
Security Risks: Using .env files for secrets is a common but critical vulnerability.
Infrastructure as Bottleneck: AI/ML engineers are forced to become DevOps experts, slowing down innovation.
Monolithic Logic: Agents are often built as large applications with tightly coupled capabilities that are hard to reuse or scale.
Protocol Fragmentation: Lack of standardized communication between agents and UIs.

Vision & Goals
Vision: To create the industry's most efficient and secure platform for developing and operating component-based, multi-agent AI systems, where infrastructure is invisible, security is automatic, and developer velocity is paramount.
Key Goals:

Reduce Time-to-Production: Decrease agent deployment time from weeks to hours.
Achieve Zero Secret Leaks: Mandate Teller for 100% of workflows.
Unify Execution Environments: Ensure workloads run identically on local Apple Silicon or cloud GPU clusters.
Foster an InnerSource Ecosystem: Make agent "skills" discoverable, versioned, and reusable.
Standardize Communication: Use the AG-UI protocol for all agent-frontend interactions.

4. User Personas & Goals

Alina, the AI/ML Engineer: Needs a terminal-first workflow to iterate on models locally (MLX on Mac) and seamlessly scale to distributed training jobs (A100 spot instances) with cost optimization and real-time monitoring.
Ben, the Backend Developer: Needs reproducible environments, ironclad secrets management (Teller), and simple, type-safe ways to build and orchestrate agent logic using Agno and Mastra.
Chloe, the Cloud & Platform Engineer: Needs to provide developers with a self-service platform using curated deployment templates that are cost-efficient, secure, and observable, all while maintaining guardrails.
Dana, the Frontend Developer: Needs a modern React/Vite environment with standardized protocols (AG-UI) to build responsive and intuitive interfaces for visualizing and interacting with agents in real-time.

5. Technical Architecture Overview
High-Level System Architecture
mermaidgraph TB
    subgraph "Frontend Layer"
        A[React + Vite.js App]
        B[AG-UI Client SDK]
        C[CopilotKit UI Components]
        D[Tailwind CSS + Radix UI]
    end
    
    subgraph "Protocol Layer"
        E[AG-UI Protocol]
        F[WebSocket/SSE Transport]
        G[Authentication & Security]
    end
    
    subgraph "Backend Services (Python)"
        H[CopilotKit Backend]
        I[Agno Agent Framework]
        J[Mastra AI Framework]
        K[Agent Orchestration Service]
    end
    
    subgraph "Infrastructure Layer"
        L[SkyPilot Cloud Orchestration]
        M[Teller Secrets Management]
        N[Vector Database (RAG)]
        O[Model Services (MLX/Ray)]
    end
    
    subgraph "Development Environment"
        P[Devbox Environment]
        Q[Nushell Scripting]
        R[Local LLM Daemon]
    end
    
    A --> B
    B --> E
    E --> F
    F --> K
    K --> H
    K --> I
    K --> J
    H & I & J --> N
    H & I & J --> O
    O --> L
    M --injects secrets--> K
    M --injects secrets--> O
    P --runs--> A
    P --runs--> K
Technology Stack Matrix
LayerTechnologyPurposeVersionStatusFrontendReactUI Framework18+Production ReadyVite.jsBuild Tool & Dev Server6.0+Production ReadyTypeScriptType Safety5.0+Production ReadyAG-UI SDKAgent ProtocolLatestProduction ReadyTailwind CSSStyling Framework3.4+Production ReadyBackendCopilotKitAgent InfrastructureLatestProduction ReadyAgnoMulti-Agent FrameworkLatestProduction ReadyMastraTypeScript AgentsLatestProduction ReadyFastAPIAPI FrameworkLatestProduction ReadyInfrastructureSkyPilotCloud OrchestrationLatestProduction ReadyTellerSecrets ManagementLatestProduction ReadyDoclingDocument ProcessingLatestProduction ReadyRayDistributed MLLatestProduction ReadyDevelopmentDevboxEnvironment ManagementLatestProduction ReadyNushellStructured ScriptingLatestEvaluation PhaseWasmEdgeSecure RuntimeLatestProduction Ready
6. Backend Architecture (Python)
The backend is built using Python and FastAPI. It features a multi-framework orchestration layer to manage agents built with CopilotKit, Agno, and Mastra. Communication with the frontend is standardized via the AG-UI protocol.
6.1. Core API Server
This is the main entry point for the backend, handling HTTP requests, WebSocket connections, and routing tasks to the orchestrator.
File: uap-platform/backend/main.py
python# File: uap-platform/backend/main.py
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime

from .services.agent_orchestrator import UAP_AgentOrchestrator
# from .services.auth import verify_token # Placeholder for auth logic

# Placeholder for auth - in a real app, this would be a robust function
def verify_token(credentials: HTTPAuthorizationCredentials):
    # In a real app, decode and verify the JWT token
    # For now, we'll just check for a static token for simplicity
    if credentials.credentials != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"user_id": "123", "username": "testuser"}

class AgentRequest(BaseModel):
    message: str
    framework: Optional[str] = 'auto'
    context: Optional[Dict[str, Any]] = {}
    stream: Optional[bool] = False

class AgentResponse(BaseModel):
    message: str
    agent_id: str
    framework: str
    timestamp: datetime
    metadata: Dict[str, Any]

app = FastAPI(
    title="UAP Backend API",
    version="3.0.0",
    description="Unified Agentic Platform Backend API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
orchestrator = UAP_AgentOrchestrator()

@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize_services()
    print("UAP Backend API started successfully")

@app.websocket("/ws/agents/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str, token: Optional[str] = None):
    """WebSocket endpoint for real-time agent communication via AG-UI protocol."""
    await websocket.accept()
    connection_id = f"{agent_id}_{datetime.utcnow().timestamp()}"
    orchestrator.register_connection(connection_id, websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            event = json.loads(data)
            # Route AG-UI events to the orchestrator for processing
            await orchestrator.handle_agui_event(connection_id, event)
    except WebSocketDisconnect:
        orchestrator.unregister_connection(connection_id)
        print(f"WebSocket {connection_id} disconnected.")
    except Exception as e:
        print(f"WebSocket error for {connection_id}: {e}")
        orchestrator.unregister_connection(connection_id)

@app.post("/api/agents/{agent_id}/chat", response_model=AgentResponse)
async def chat_with_agent(
    agent_id: str, 
    request: AgentRequest,
    # user: dict = Depends(verify_token) # Uncomment for production
):
    """HTTP endpoint for stateless agent interactions."""
    try:
        response_data = await orchestrator.handle_http_chat(
            agent_id,
            request.message,
            request.framework,
            # {"user_id": user["user_id"], **request.context}
            request.context
        )
        return AgentResponse(
            message=response_data.get("content", ""),
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            framework=response_data.get("framework", "unknown"),
            metadata=response_data.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get status of all agents and frameworks."""
    return await orchestrator.get_system_status()

# Placeholder endpoints for other services
@app.post("/api/documents/analyze")
async def analyze_document_endpoint():
    # This would call the Docling processor
    return {"status": "ok", "message": "Document analysis endpoint placeholder."}

@app.post("/api/workflows/{workflow_name}/execute")
async def execute_workflow_endpoint(workflow_name: str):
    # This would execute a Mastra workflow via the orchestrator
    return {"status": "ok", "message": f"Workflow '{workflow_name}' execution placeholder."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
6.2. Agent Orchestration Service
This service is the brain of the backend. It initializes and manages agent frameworks and routes incoming requests to the appropriate agent.
File: uap-platform/backend/services/agent_orchestrator.py
python# File: uap-platform/backend/services/agent_orchestrator.py
import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket

# Import framework managers (which we will create next)
# from ..frameworks.copilot.agent import CopilotKitManager
# from ..frameworks.agno.agent import AgnoAgentManager
# from ..frameworks.mastra.agent import MastraAgentManager

# --- Mock Framework Managers for initial implementation ---
# Replace these with the real implementations later.
class MockAgentManager:
    def __init__(self, framework_name):
        self.framework_name = framework_name
        print(f"{framework_name} manager initialized.")
    
    async def process_message(self, message, context):
        return {
            "content": f"Response from {self.framework_name}: You said '{message}'",
            "metadata": {"source": self.framework_name, "context_received": bool(context)}
        }
    
    def get_status(self):
        return {"status": "active", "agents": 2}

class UAP_AgentOrchestrator:
    def __init__(self):
        # self.copilot_manager = CopilotKitManager()
        # self.agno_manager = AgnoAgentManager()
        # self.mastra_manager = MastraAgentManager()
        
        # Using mocks for now
        self.copilot_manager = MockAgentManager("CopilotKit")
        self.agno_manager = MockAgentManager("Agno")
        self.mastra_manager = MockAgentManager("Mastra")
        
        self.active_connections: Dict[str, WebSocket] = {}

    async def initialize_services(self):
        # In a real app, this would involve async setup for each manager
        print("All agent frameworks initialized successfully.")

    def register_connection(self, conn_id: str, websocket: WebSocket):
        self.active_connections[conn_id] = websocket
        print(f"Connection {conn_id} registered.")

    def unregister_connection(self, conn_id: str):
        if conn_id in self.active_connections:
            del self.active_connections[conn_id]
            print(f"Connection {conn_id} unregistered.")

    async def handle_agui_event(self, conn_id: str, event: Dict[str, Any]):
        """Handles incoming events from the AG-UI client."""
        event_type = event.get("type")
        
        if event_type == "user_message":
            content = event.get("content", "")
            metadata = event.get("metadata", {})
            framework = metadata.get("framework", "auto")
            
            # Route to the correct framework
            response_data = await self._route_and_process(content, framework, metadata)
            
            # Create a response event and send it back
            response_event = {
                "type": "text_message_content",
                "content": response_data.get("content", "No response."),
                "metadata": response_data.get("metadata", {})
            }
            await self._send_to_connection(conn_id, response_event)

    async def handle_http_chat(self, agent_id: str, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Handles stateless HTTP chat requests."""
        response = await self._route_and_process(message, framework, context)
        response["framework"] = framework if framework != 'auto' else 'copilot' # default
        return response

    async def _route_and_process(self, message: str, framework: str, context: Dict) -> Dict[str, Any]:
        """Intelligently routes a message to the correct agent framework."""
        if framework == 'auto':
            # Simple content-based routing logic
            if 'document' in message.lower() or 'analyze' in message.lower():
                framework = 'agno' # Agno is good for document processing
            elif 'support' in message.lower() or 'help' in message.lower():
                framework = 'mastra' # Mastra for workflow-based support
            else:
                framework = 'copilot' # Default to CopilotKit
        
        # Process with the selected framework
        if framework == 'copilot':
            return await self.copilot_manager.process_message(message, context)
        elif framework == 'agno':
            return await self.agno_manager.process_message(message, context)
        elif framework == 'mastra':
            return await self.mastra_manager.process_message(message, context)
        else:
            return {"content": f"Error: Unknown framework '{framework}'", "metadata": {}}

    async def _send_to_connection(self, conn_id: str, data: Dict[str, Any]):
        """Sends a message to a specific WebSocket connection."""
        if conn_id in self.active_connections:
            websocket = self.active_connections[conn_id]
            try:
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                print(f"Failed to send to {conn_id}: {e}")
                self.unregister_connection(conn_id)

    async def get_system_status(self) -> Dict[str, Any]:
        return {
            "copilot": self.copilot_manager.get_status(),
            "agno": self.agno_manager.get_status(),
            "mastra": self.mastra_manager.get_status(),
            "active_connections": len(self.active_connections)
        }
6.3. Agent Framework Implementations (Stubs)
For now, create stub files for each agent framework. The MockAgentManager in the orchestrator uses these concepts. The full implementation would follow the detailed code from the original PRD.
File: uap-platform/backend/frameworks/agno/agent.py
python# File: uap-platform/backend/frameworks/agno/agent.py
# This file will contain the full AgnoAgentManager class as detailed in the original PRD.
# For now, it's a placeholder.
print("Agno agent module loaded.")
File: uap-platform/backend/frameworks/mastra/agent.py
python# File: uap-platform/backend/frameworks/mastra/agent.py
# This file will contain the full MastraAgentManager class as detailed in the original PRD.
# For now, it's a placeholder.
print("Mastra agent module loaded.")
File: uap-platform/backend/frameworks/copilot/agent.py
python# File: uap-platform/backend/frameworks/copilot/agent.py
# This file will contain the full CopilotKitManager class as detailed in the original PRD.
# For now, it's a placeholder.
print("CopilotKit agent module loaded.")
6.4. Testing Structure
File: uap-platform/backend/tests/test_orchestrator.py
pythonimport pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_agent_chat():
    response = client.post("/api/agents/test-agent/chat", json={
        "message": "Hello",
        "framework": "copilot"
    })
    assert response.status_code == 200

def test_system_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert "copilot" in response.json()
    assert "agno" in response.json()
    assert "mastra" in response.json()
7. Frontend Architecture (React + TypeScript)
The frontend is a modern React application built with Vite and TypeScript, based on the Dojo App reference. It uses the AG-UI protocol to communicate with the backend in real-time.
7.1. HTML Entry Point
File: uap-platform/frontend/index.html
html<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>UAP - Unified Agentic Platform</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
7.2. Vite Configuration
File: uap-platform/frontend/vite.config.ts
typescript// File: uap-platform/frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // Proxy API requests to the Python backend
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000', // Proxy WebSocket connections
        ws: true,
      },
    },
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          'ag-ui': ['@ag-ui/client'],
          copilot: ['@copilotkit/react-core', '@copilotkit/react-ui'],
          ui: ['@radix-ui/react-dialog', 'lucide-react'],
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
})
7.3. TypeScript Configuration
File: uap-platform/frontend/tsconfig.json
json{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
File: uap-platform/frontend/tsconfig.node.json
json{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
7.4. Tailwind Configuration
File: uap-platform/frontend/tailwind.config.js
javascript/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
7.5. PostCSS Configuration
File: uap-platform/frontend/postcss.config.js
javascriptexport default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
7.6. CSS Files
File: uap-platform/frontend/src/index.css
css@tailwind base;
@tailwind components;
@tailwind utilities;
File: uap-platform/frontend/src/App.css
css/* App-specific styles */
#root {
  margin: 0 auto;
  text-align: center;
}
7.7. Main Entry Point
File: uap-platform/frontend/src/main.tsx
typescriptimport React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
7.8. Main Application Component
File: uap-platform/frontend/src/App.tsx
typescript// File: uap-platform/frontend/src/App.tsx
import './App.css'
import { AgentDashboard } from './components/agents/AgentDashboard'
import { Layout } from './components/common/Layout'

function App() {
  return (
    <Layout>
      <h1 className="text-3xl font-bold p-6">Unified Agentic Platform</h1>
      <AgentDashboard />
    </Layout>
  )
}

export default App
7.9. Layout Component
File: uap-platform/frontend/src/components/common/Layout.tsx
typescriptimport { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto max-w-7xl">
        {children}
      </main>
    </div>
  )
}
7.10. Type Definitions
File: uap-platform/frontend/src/types/ag-ui.d.ts
typescriptdeclare module '@ag-ui/client' {
  export interface AGUIEvent {
    type: string;
    content?: string;
    metadata?: Record<string, any>;
  }
  
  export class AGUIClient {
    constructor(config: any);
    on(event: string, handler: (event: AGUIEvent) => void): void;
    connect(): void;
    disconnect(): void;
    sendMessage(message: any): Promise<void>;
  }
}
7.11. AG-UI Protocol Hook
File: uap-platform/frontend/src/hooks/useAGUI.ts
typescript// File: uap-platform/frontend/src/hooks/useAGUI.ts
import { useEffect, useState, useCallback, useRef } from 'react';
import { AGUIClient, AGUIEvent } from '@ag-ui/client';

export function useAGUI(agentId: string) {
  const [client, setClient] = useState<AGUIClient | null>(null);
  const [messages, setMessages] = useState<AGUIEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const agentIdRef = useRef(agentId);

  useEffect(() => {
    // Construct WebSocket URL, handling http/https protocols
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/agents/${agentIdRef.current}`;

    const aguiClient = new AGUIClient({
      endpoint: wsUrl,
      transport: 'websocket',
      reconnect: true,
    });

    aguiClient.on('connection_open', () => setIsConnected(true));
    aguiClient.on('connection_close', () => setIsConnected(false));
    aguiClient.on('error', (err) => console.error('AG-UI Error:', err));
    
    // Listen for all message types and add to state
    const messageTypes = ['text_message_content', 'tool_call_start', 'tool_call_end', 'state_delta'];
    messageTypes.forEach(type => {
        aguiClient.on(type, (event: AGUIEvent) => {
            setMessages(prev => [...prev, event]);
        });
    });

    setClient(aguiClient);
    aguiClient.connect();

    return () => {
      aguiClient.disconnect();
    };
  }, []); // Only run once on mount

  const sendMessage = useCallback(async (content: string, framework: string = 'auto') => {
    if (!client || !isConnected) {
      console.error('Cannot send message: Client not connected.');
      return;
    }
    await client.sendMessage({
      type: 'user_message',
      content,
      metadata: { framework },
    });
  }, [client, isConnected]);

  return { messages, isConnected, sendMessage };
}
7.12. Agent Dashboard Component
File: uap-platform/frontend/src/components/agents/AgentDashboard.tsx
typescript// File: uap-platform/frontend/src/components/agents/AgentDashboard.tsx
import { AgentCard } from './AgentCard';

// This data would typically come from an API call to /api/status
const mockAgents = [
  {
    id: 'research-agent',
    name: 'Research Agent',
    description: 'Specializes in web searches and document analysis using Agno.',
    framework: 'agno',
  },
  {
    id: 'support-agent',
    name: 'Customer Support Agent',
    description: 'Handles customer queries with predefined workflows using Mastra.',
    framework: 'mastra',
  },
  {
    id: 'general-assistant',
    name: 'General Assistant',
    description: 'A general-purpose assistant powered by CopilotKit.',
    framework: 'copilot',
  },
];

export function AgentDashboard() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
      {mockAgents.map((agent) => (
        <AgentCard
          key={agent.id}
          id={agent.id}
          name={agent.name}
          description={agent.description}
          framework={agent.framework}
        />
      ))}
    </div>
  );
}
7.13. Agent Card Component
File: uap-platform/frontend/src/components/agents/AgentCard.tsx
typescript// File: uap-platform/frontend/src/components/agents/AgentCard.tsx
import { useState, useRef } from 'react';
import { useAGUI } from '@/hooks/useAGUI';
// Dummy components for UI, replace with Radix/Shadcn
const Card = ({ children, className }: any) => <div className={`border rounded-lg p-4 shadow-md bg-white ${className}`}>{children}</div>;
const CardHeader = ({ children }: any) => <div className="font-bold text-lg mb-2">{children}</div>;
const CardContent = ({ children }: any) => <div className="text-sm text-gray-700">{children}</div>;
const Input = (props: any) => <input className="border rounded w-full p-2 my-2" {...props} />;
const Button = (props: any) => <button className="bg-blue-500 text-white rounded px-4 py-2 w-full disabled:bg-gray-400" {...props} />;
const Badge = ({ children }: any) => <span className="bg-gray-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full">{children}</span>;


interface AgentCardProps {
  id: string;
  name: string;
  description: string;
  framework: string;
}

export function AgentCard({ id, name, description, framework }: AgentCardProps) {
  const { messages, isConnected, sendMessage } = useAGUI(id);
  const [input, setInput] = useState('');
  const chatWindowRef = useRef<HTMLDivElement>(null);

  const handleSend = () => {
    if (input.trim()) {
      sendMessage(input, framework);
      setInput('');
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <span>{name}</span>
          <Badge>{framework}</Badge>
        </div>
        <div className={`mt-1 h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
      </CardHeader>
      <CardContent>
        <p className="mb-4">{description}</p>
        <div ref={chatWindowRef} className="h-48 overflow-y-auto border rounded p-2 bg-gray-50 mb-2">
          {messages.map((msg, index) => (
            <div key={index} className="text-xs p-1">
              <strong>{msg.type}:</strong> {msg.content}
            </div>
          ))}
        </div>
        <div className="flex space-x-2">
          <Input
            type="text"
            value={input}
            onChange={(e: any) => setInput(e.target.value)}
            onKeyDown={(e: any) => e.key === 'Enter' && handleSend()}
            placeholder="Chat with agent..."
            disabled={!isConnected}
          />
          <Button onClick={handleSend} disabled={!isConnected}>
            Send
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
8. Platform Infrastructure
This section details the tools and configurations for deploying and running the UAP in the cloud.
8.1. Cloud Orchestration with SkyPilot
Use SkyPilot for cost-effective, multi-cloud deployment of the UAP application and ML training jobs.
File: uap-platform/skypilot/uap-production.yaml
yaml# skypilot/uap-production.yaml
resources:
  accelerators: A100:1 # Request at least one GPU for ML tasks
  cloud: [gcp, aws, azure]  # Multi-cloud failover
  region: [us-central1, us-west-2, eastus]
  use_spot: true
  spot_recovery: auto
  
workdir: . # The root of the project

file_mounts:
  /app: . # Mount the entire project directory

setup: |
  # Install system dependencies
  sudo apt-get update && sudo apt-get install -y curl git
  
  # Install devbox
  curl -fsSL https://get.jetify.com/devbox | bash
  
  # Use devbox to install project-specific packages
  devbox install

run: |
  # Use devbox to run the development script
  # Teller will inject secrets before the command runs
  devbox run dev
8.2. Infrastructure and Helper Scripts (Nushell)
Nushell scripts provide powerful, data-aware utilities for managing the development lifecycle.
File: uap-platform/scripts/uap-tools.nu
nushell# File: uap-platform/scripts/uap-tools.nu
# UAP Development Utilities in Nushell

# Get agent status across all frameworks
export def "uap agent-status" [] {
    let status = (
        http get "http://localhost:8000/api/status" 
        | from json
    )
    
    $status | table
}

# Monitor agent performance (stub)
export def "uap monitor" [
    --agent_id: string = "all"  # Specific agent ID or "all"
    --duration: string = "1m"   # Monitoring duration
] {
    print $"Monitoring agents (ID: ($agent_id), duration: ($duration)) - STUB"
    sleep 10sec
    print "Monitoring complete."
}

# Generate agent performance report (stub)
export def "uap generate-report" [
    --format: string = "json"  # Output format (json, csv, html)
] {
    print $"Generating performance report (format: ($format)) - STUB"
    let report = { generated_at: (date now), status: "stub_data" }
    $report | to json
}

# Deploy to SkyPilot
export def "uap deploy" [
    --env: string = "production"  # Environment (production, staging, dev)
] {
    print $"Deploying to ($env) environment..."
    cd ..
    skypilot up -c $"skypilot/uap-($env).yaml"
}

# Check system health
export def "uap health-check" [] {
    let health = (
        http get "http://localhost:8000/health" 
        | from json
    )
    
    print $"System Status: ($health.status)"
    print $"Timestamp: ($health.timestamp)"
}
9. Functional & Technical Requirements
These sections from the original PRD provide the guiding principles for the implementation. The coding agent should use them to understand the intent behind the code.
Functional Requirements (Epics & Stories)

Epic 1: Cross-Framework Agent Development: Seamlessly create, manage, and communicate between agents from CopilotKit, Agno, and Mastra.
Epic 2: Real-Time Agent Interactions: Provide users with a live, streaming view of agent responses and internal state via WebSockets.
Epic 3: Document Intelligence Pipeline: Enable users to upload and process various document formats, extracting structured data for analysis.
Epic 4: Multi-Cloud Deployment: Deploy the platform cost-effectively across multiple cloud providers with high availability.
Epic 5: Advanced Security & Compliance: Ensure zero-trust secrets management, comprehensive audit trails, and compliance with standards like SOC2/GDPR.

Technical & Performance Specifications

Agent Response Time: < 2s (95th percentile) for the first token.
UI Load Time: < 1s Time to Interactive (TTI).
Concurrent Users: 1000+ simultaneous active WebSocket sessions.
Uptime: 99.9% monthly availability.
Security: JWT-based auth, RBAC, data encryption at rest (AES-256) and in transit (TLS 1.3).

10. Implementation Roadmap
This phased plan outlines the development sequence. The agent should follow this order to build the platform incrementally.
Phase 1: Foundation (Current Scope)

Environment Setup: Initialize project with Devbox, Vite, TypeScript, and Tailwind. (✓ Done in Section 1)
Backend Infrastructure: Set up FastAPI server with CORS, WebSocket support, and a basic orchestrator. (✓ Done in Section 6)
Basic Agent Integration: Implement mock/stub managers for each agent framework. (✓ Done in Section 6)
Frontend-Backend Communication: Implement AG-UI hook and basic UI components for real-time chat. (✓ Done in Section 7)

Phase 2: Core Features

Full Agent Integration: Replace mock managers with the full Python implementations of CopilotKit, Agno, and Mastra managers.
Document Processing: Integrate the DoclingProcessor and build the corresponding API and UI for file uploads.
Authentication: Implement robust JWT-based authentication and authorization.

Phase 3: Advanced Features & Cloud Deployment

Cloud Deployment: Configure and test the SkyPilot deployment scripts for production.
ML/AI Enhancements: Integrate the MLXProcessor for local inference on Apple Silicon and the RayClusterManager for distributed workloads.
Security Hardening: Implement full audit logging, encryption key management, and security scanning.

11. Risk Assessment & Success Metrics
These sections provide project management context and are included for completeness.
Risk Assessment

Technical Risks: Framework compatibility, performance bottlenecks, security vulnerabilities.
Mitigation: Incremental integration, feature flags, load testing, and security audits.
Organizational Risks: Skills gaps, changing requirements.
Mitigation: Comprehensive documentation (this document), agile sprints, and modular architecture.

Success Metrics & KPIs

Developer Velocity: Reduce time to deploy a new agent from weeks to hours.
Platform Performance: Maintain <2s p95 response time and 99.9% uptime.
User Experience: Achieve a >95% task completion rate for core workflows.
Business Impact: Achieve a 50% reduction in infrastructure costs and a 300% increase in developer productivity.

12. Validation Checklist for Coding Agent
Before implementation, ensure:

 All directory paths exist before file creation
 All imports have corresponding files/packages
 Environment variables are documented
 Mock implementations work before adding real ones
 Frontend can run independently with mock data
 Backend can run with stub agents
 WebSocket connection is testable

13. Implementation Order for Coding Agent

Create directory structure (including all init.py files)
Create all configuration files (devbox.json, package.json, requirements.txt, etc.)
Create environment files (.env.example, .teller.yml)
Create backend with mock agents (main.py, orchestrator, stubs)
Create frontend infrastructure files (HTML, CSS, configs)
Create minimal frontend components that connect
Test WebSocket communication
Add testing structure
Add deployment configurations
Gradually replace mocks with real implementations

14. Reference Links & Resources
This curated list of resources is vital for understanding the technologies used. All links from the original document are preserved here for reference.
Core Technologies

Vite.js: Official Guide
React: Official Documentation
CopilotKit: GitHub Repository
Agno Framework: GitHub Repository
Mastra AI: GitHub Repository
AG-UI Protocol: GitHub Repository
SkyPilot: GitHub Repository
Teller: GitHub Repository
Docling: GitHub Repository
Devbox: GitHub Repository
Nushell: GitHub Repository
Apple MLX: GitHub Repository
Ray Project: GitHub Repository


Note to AI Coding Agent: This PRD is now complete with all necessary files and configurations. Start with the implementation order in Section 13 and create each file exactly as specified. The mock implementations allow for a working MVP that can be gradually enhanced with real framework integrations.