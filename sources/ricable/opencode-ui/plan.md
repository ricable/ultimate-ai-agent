# Product Requirements Document: Porting Claudia UI to SST OpenCode

## Executive Summary

This PRD outlines the requirements and implementation strategy for porting the Claudia UI frontend (a Rust/Tauri2 desktop application for Claude Code) to serve as a desktop GUI for SST OpenCode. The project leverages OpenCode's modular Go-based architecture and explicit multi-frontend design to create a powerful desktop experience that complements the existing terminal interface.

**Key Advantages of this Port:**
- SST OpenCode is architected with clean API separation, unlike Claude Code's terminal-only design
- OpenCode's Go backend provides 75+ AI provider ecosystem with robust session management
- Claudia provides a proven, feature-rich UI with advanced capabilities like timeline management and sandboxing
- The combination creates the first desktop GUI for OpenCode's comprehensive tool integration system
- Tauri2's lightweight architecture (600KB installers) ensures excellent performance

**Project Scope:** Adapt Claudia's React/Tauri2 frontend to communicate with SST OpenCode's Go backend via a new API layer, maintaining feature parity while leveraging OpenCode's superior multi-provider capabilities, session management, and tool integration.

## Technical Architecture Analysis

### Current Architectures

#### Claudia (Source)
```
┌─────────────────────────────────────┐
│         Tauri2 Application          │
├─────────────────────────────────────┤
│  Frontend (React + TypeScript)      │
│  - Components, State Management     │
│  - Tailwind CSS Styling            │
├─────────────────────────────────────┤
│  IPC Bridge (Tauri Commands)        │
├─────────────────────────────────────┤
│  Backend (Rust)                     │
│  - Claude Code CLI Integration     │
│  - Sandboxing & Security           │
│  - Timeline/Checkpoint Management   │
└─────────────────────────────────────┘
```

#### SST OpenCode (Target)
```
┌─────────────────────────────────────┐
│    OpenCode Go Backend              │
├─────────────────────────────────────┤
│  cmd/           # CLI (Cobra)        │
│  internal/app/  # Core Services     │
│  internal/llm/  # 75+ Providers     │
│  internal/db/   # SQLite Sessions   │
│  internal/tui/  # Bubble Tea UI     │
│  internal/lsp/  # Language Servers  │
├─────────────────────────────────────┤
│  Features:                          │
│  - Session Management (SQLite)     │
│  - Tool Integration (MCP + Built-in)│
│  - Multi-Provider Authentication   │
│  - Custom Commands & Agents        │
│  - Configuration System (JSON)     │
└─────────────────────────────────────┘
```

### Target Architecture
```
┌─────────────────────────────────────┐
│    OpenCode Desktop (Tauri2)        │
├─────────────────────────────────────┤
│  Frontend (React + TypeScript)      │
│  - Adapted Claudia Components      │
│  - OpenCode API Integration        │
│  - Enhanced Provider Management    │
│  - Visual Tool Approval System     │
├─────────────────────────────────────┤
│  Backend Adapter Layer (Rust)       │
│  - API Client Implementation       │
│  - OpenCode Process Management     │
│  - Enhanced Security Features      │
├─────────────────────────────────────┤
│  NEW: OpenCode API Layer (Go)       │
│  - HTTP/WebSocket Server           │
│  - Bridge to internal/app          │
│  - Real-time Session Updates       │
├─────────────────────────────────────┤
│  OpenCode Core (Existing)           │
│  - internal/* modules unchanged    │
│  - SQLite Database                 │
│  - Provider Ecosystem              │
└─────────────────────────────────────┘
```

### Key Technical Differences & Integration Points

| Component | Claudia | SST OpenCode | Migration Strategy |
|-----------|---------|--------------|-------------------|
| **Backend Communication** | CLI wrapper | Go modular backend | Create new API layer in Go |
| **Provider Support** | Claude only | 75+ via Models.dev | Leverage existing provider system |
| **Session Storage** | Local files | SQLite database | Use OpenCode's session management |
| **Authentication** | Claude API key | Multi-provider via `opencode auth` | Integrate with auth.json system |
| **Tool Integration** | Limited | Built-in + MCP tools | Expose rich tool system in GUI |
| **Configuration** | Simple config | JSON schema-based | Leverage OpenCode's config system |
| **Language Support** | Basic | LSP integration | Expose diagnostics and LSP features |

## OpenCode Integration Deep Dive

### 1. OpenCode Backend Architecture
OpenCode's modular Go architecture provides clean integration points:

```go
// Key modules for GUI integration
internal/
├── app/        # Core application services (session, tool execution)
├── config/     # JSON schema-based configuration management
├── db/         # SQLite operations and migrations
├── llm/        # Provider management and tool registry
├── session/    # Session lifecycle management
├── message/    # Message handling and streaming
└── lsp/        # Language Server Protocol integration
```

### 2. Session Management System
OpenCode provides robust session management that surpasses Claudia's capabilities:

**Features to Leverage:**
- **SQLite Persistence**: Automatic session storage and retrieval
- **Session Sharing**: Built-in shareable link generation
- **Parallel Sessions**: Multiple concurrent sessions support
- **Session Metadata**: Rich session information and organization

**GUI Integration Points:**
- Visual session browser with thumbnails
- Session sharing interface with QR codes
- Session import/export functionality
- Cross-device session synchronization

### 3. Multi-Provider Ecosystem
OpenCode's provider system via Models.dev offers unprecedented flexibility:

**Current Providers (75+):**
- **OpenAI**: GPT-4.1, GPT-4.5, GPT-4o, O1, O3, O4 families
- **Anthropic**: Claude 3.5/3.7 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- **Google**: Gemini 2.5, Gemini 2.5 Flash, Gemini 2.0 Flash variants
- **Groq**: Llama 4 Maverick, Llama 4 Scout, QWEN, Deepseek R1
- **Local Models**: Custom provider configuration support

**GUI Enhancements:**
- Provider performance dashboard with cost/speed metrics
- Intelligent provider routing based on task type
- Provider health monitoring and failover
- Cost tracking and budgeting per provider

### 4. Advanced Tool System
OpenCode provides comprehensive tool integration:

#### Built-in Tools
```typescript
// File Operations
interface FileTools {
  glob: (pattern: string, path?: string) => FileMatch[]
  grep: (pattern: string, options: GrepOptions) => SearchResult[]
  ls: (path?: string, ignore?: string[]) => DirectoryListing
  view: (filePath: string, offset?: number, limit?: number) => FileContent
  write: (filePath: string, content: string) => WriteResult
  edit: (filePath: string, params: EditParams) => EditResult
  patch: (filePath: string, diff: string) => PatchResult
}

// System Operations
interface SystemTools {
  bash: (command: string, timeout?: number) => CommandResult
  fetch: (url: string, format: string, timeout?: number) => FetchResult
  agent: (prompt: string) => AgentResult
  diagnostics: (filePath?: string) => DiagnosticInfo[]
}
```

#### MCP (Model Context Protocol) Integration
```json
{
  "mcpServers": {
    "localServer": {
      "type": "stdio",
      "command": "path/to/mcp-server",
      "env": [],
      "args": []
    },
    "remoteServer": {
      "type": "sse",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer token"
      }
    }
  }
}
```

### 5. Configuration System Architecture
OpenCode's JSON schema-based configuration provides robust foundation:

**Configuration Hierarchy:**
1. Global: `~/.config/opencode/config.json`
2. Project: `opencode.json` (Git-aware, team-shared)
3. Local: `./.opencode.json` (directory-specific)

**Schema Structure:**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "theme": "opencode",
  "model": "anthropic/claude-sonnet-4-20250514",
  "autoshare": false,
  "autoupdate": true,
  "provider": {
    "customProvider": {
      "apiKey": "encrypted",
      "disabled": false
    }
  },
  "agents": {
    "primary": { "model": "claude-3.7-sonnet", "maxTokens": 5000 },
    "task": { "model": "claude-3.7-sonnet", "maxTokens": 5000 },
    "title": { "model": "claude-3.7-sonnet", "maxTokens": 80 }
  },
  "mcp": {},
  "lsp": {
    "go": { "disabled": false, "command": "gopls" },
    "typescript": { "command": "typescript-language-server", "args": ["--stdio"] }
  },
  "keybinds": {},
  "shell": {
    "path": "/bin/zsh",
    "args": ["-l"]
  }
}
```

### 6. Agent Rules System (AGENTS.md)
OpenCode's sophisticated agent behavior system:

**Rule Hierarchy:**
1. **Global Rules**: `~/.config/opencode/AGENTS.md` (personal preferences)
2. **Project Rules**: `<PROJECT_ROOT>/AGENTS.md` (team-shared, Git-committed)

**GUI Integration:**
- Visual AGENTS.md editor with syntax highlighting
- Rule template marketplace
- Team collaboration on agent behavior
- A/B testing interface for different rule sets

## Feature Requirements and Mapping

### Core Features to Port from Claudia

#### 1. Enhanced Project & Session Management
**Current (Claudia)**: Visual project dashboard with timestamped sessions
**Target Enhancement**:
- **Multi-Provider Session Creation**: Choose provider, model, and configuration per session
- **SQLite-Backed Persistence**: Leverage OpenCode's robust database system
- **Shareable Session Integration**: GUI for OpenCode's built-in session sharing
- **Cross-Device Continuity**: Session synchronization across devices
- **Session Templates**: Pre-configured session types for different workflows

#### 2. Advanced Chat Interface
**Current**: Claude-specific conversation UI
**Target Requirements**:
- **Provider-Agnostic Messaging**: Handle messages from any of 75+ providers
- **Real-Time Streaming**: WebSocket integration for live responses
- **Multi-Model Conversations**: Switch providers mid-conversation
- **Cost Tracking**: Real-time cost calculation per provider/model
- **Message Annotations**: Add comments, bookmarks, and tags to messages

#### 3. Enhanced Timeline & Checkpoints
**Current**: Git-like versioning with visual timeline
**Target Adaptation**:
- **SQLite-Backed Checkpoints**: Leverage OpenCode's database for checkpoint storage
- **Multi-User Branching**: Collaborative branching support via session sharing
- **Universal Diff Viewer**: Diff support for any provider's code changes
- **Checkpoint Metadata**: Rich information about each checkpoint (provider, cost, performance)
- **Export/Import**: Checkpoint sharing and backup functionality

#### 4. Multi-Agent System
**Current**: CC Agents with custom prompts
**Target Enhancement**:
- **OpenCode Agent Integration**: Leverage primary/task/title agent system
- **Provider-Specific Configurations**: Different agents for different providers
- **Agent Performance Analytics**: Track agent effectiveness across providers
- **Template Marketplace**: Shared agent configurations
- **A/B Testing Interface**: Compare agent configurations

#### 5. Advanced Tool Management
**Current**: Basic MCP configuration UI
**Target**: Full OpenCode tool integration:
- **Built-in Tools Dashboard**: Visual management for file, system, and development tools
- **MCP Server Management**: Easy configuration for local and remote MCP servers
- **Tool Permission Matrix**: Granular control over tool access
- **Tool Execution Monitoring**: Real-time logs and status tracking
- **Custom Tool Builder**: Visual interface for creating custom tools

### New Features Unique to OpenCode

#### 1. Multi-Provider Command Center
- **Provider Performance Dashboard**: Real-time metrics, cost analysis, response times
- **Intelligent Provider Routing**: Automatic provider selection based on task type
- **Provider Health Monitoring**: Status indicators and failover configuration
- **Cost Management**: Budgets, alerts, and spending analytics per provider

#### 2. Language Server Integration
- **LSP Status Dashboard**: Monitor connected language servers
- **Diagnostics Viewer**: Rich display of code diagnostics from LSP
- **Multi-Language Support**: Configure LSP servers for different languages
- **Code Intelligence**: Leverage LSP features for enhanced code understanding

#### 3. Advanced Configuration Management
- **Visual Config Editor**: GUI for OpenCode's JSON configuration
- **Configuration Profiles**: Switch between different configuration sets
- **Team Configuration Sync**: Share project configurations
- **Schema Validation**: Real-time validation against OpenCode schema

#### 4. Custom Commands System
- **Command Builder**: Visual interface for creating custom commands with named arguments
- **Command Marketplace**: Share and discover custom commands
- **Command Analytics**: Track command usage and effectiveness
- **Command Templates**: Pre-built command templates for common workflows

## API Layer Design

### New OpenCode API Server (Go)

We need to add a new API server component to OpenCode's existing architecture:

```go
// cmd/server/main.go - New API server entry point
package main

import (
    "github.com/sst/opencode/internal/api"
    "github.com/sst/opencode/internal/app"
)

func main() {
    // Initialize OpenCode core services
    appServices := app.NewServices()
    
    // Start API server
    apiServer := api.NewServer(appServices)
    apiServer.Start(":8080")
}
```

#### API Endpoints Design

```go
// internal/api/routes.go
package api

type APIServer struct {
    sessionManager *session.Manager
    llmManager     *llm.Manager
    configManager  *config.Manager
    dbManager      *db.Manager
}

// Session Management
// GET    /api/sessions              - List all sessions
// POST   /api/sessions              - Create new session
// GET    /api/sessions/{id}         - Get session details
// DELETE /api/sessions/{id}         - Delete session
// POST   /api/sessions/{id}/share   - Generate share link
// POST   /api/sessions/{id}/message - Send message to session

// Provider Management
// GET    /api/providers             - List available providers
// POST   /api/providers/auth        - Authenticate with provider
// GET    /api/providers/status      - Get provider health status

// Tool Management
// GET    /api/tools                 - List available tools
// POST   /api/tools/execute         - Execute tool with approval
// GET    /api/tools/mcp             - List MCP servers

// Configuration
// GET    /api/config                - Get current configuration
// PUT    /api/config                - Update configuration
// POST   /api/config/validate       - Validate configuration

// WebSocket for real-time updates
// WS     /api/ws/sessions/{id}      - Real-time session updates
```

#### WebSocket Integration

```go
// internal/api/websocket.go
type SessionWebSocket struct {
    sessionID string
    conn      *websocket.Conn
    manager   *session.Manager
}

func (ws *SessionWebSocket) HandleMessage(messageType int, data []byte) {
    switch messageType {
    case websocket.TextMessage:
        // Handle incoming message
        ws.manager.ProcessMessage(ws.sessionID, string(data))
    }
}

func (ws *SessionWebSocket) SendUpdate(update SessionUpdate) {
    // Send real-time updates to frontend
    ws.conn.WriteJSON(update)
}
```

### Frontend API Client (TypeScript)

```typescript
// src/api/opencode-client.ts
export class OpenCodeClient {
    private baseURL: string
    private websockets: Map<string, WebSocket> = new Map()

    // Session Management
    async createSession(config: SessionConfig): Promise<Session> {
        const response = await fetch(`${this.baseURL}/api/sessions`, {
            method: 'POST',
            body: JSON.stringify(config)
        })
        return response.json()
    }

    async sendMessage(sessionId: string, message: string): Promise<void> {
        await fetch(`${this.baseURL}/api/sessions/${sessionId}/message`, {
            method: 'POST',
            body: JSON.stringify({ message })
        })
    }

    // Real-time Session Updates
    subscribeToSession(sessionId: string, callback: (update: SessionUpdate) => void): void {
        const ws = new WebSocket(`ws://localhost:8080/api/ws/sessions/${sessionId}`)
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data)
            callback(update)
        }
        this.websockets.set(sessionId, ws)
    }

    // Provider Management
    async getProviders(): Promise<Provider[]> {
        const response = await fetch(`${this.baseURL}/api/providers`)
        return response.json()
    }

    // Tool Execution
    async executeTool(toolId: string, params: any): Promise<ToolResult> {
        const response = await fetch(`${this.baseURL}/api/tools/execute`, {
            method: 'POST',
            body: JSON.stringify({ toolId, params })
        })
        return response.json()
    }
}
```

## UI/UX Specifications

### Design System Foundation
- **Base**: Adapt Claudia's existing Tailwind CSS design system
- **Enhancement**: OpenCode-specific provider themes and branding
- **Components**: Leverage shadcn/ui for consistency with modern React applications
- **Theme Integration**: Support for OpenCode's built-in theme system (10+ themes)

### Layout Architecture

#### Primary Layout (Enhanced Four-Panel Design)
```
┌─────────┬─────────────────────┬──────────┬─────────┐
│Sidebar  │    Main Editor      │   Chat   │ Tools   │
│         │                     │          │ Panel   │
│Sessions │  ┌───────────────┐ │Messages  │         │
│Providers│  │ Monaco Editor │ │History   │Tool     │
│Agents   │  │               │ │          │Queue    │
│Config   │  └───────────────┘ │Controls  │MCP      │
│Files    │  Timeline/Diffs     │Provider  │LSP      │
│         │  LSP Diagnostics    │Selector  │Logs     │
└─────────┴─────────────────────┴──────────┴─────────┘
```

### Key UI Components

#### 1. Enhanced Provider Selector
- **Multi-Provider Grid**: Visual grid with provider logos, status, and cost indicators
- **Quick Provider Switch**: Hotkey switching between frequently used providers
- **Cost Estimation**: Real-time cost display based on token usage
- **Provider Health**: Connection status and response time indicators
- **Model Selection**: Dropdown within each provider for model selection

#### 2. Advanced Session Management
- **Session Browser**: Grid view with session thumbnails and metadata
- **Session Sharing**: One-click sharing with QR code generation
- **Session Templates**: Pre-configured session types for different workflows
- **Cross-Device Sync**: Visual indicators for sessions available on other devices
- **Session Analytics**: Usage statistics and performance metrics

#### 3. Enhanced Chat Interface
- **Provider Attribution**: Message bubbles with provider logos and model indicators
- **Streaming Indicators**: Real-time typing indicators during message generation
- **Message Actions**: Edit, regenerate, branch conversation options
- **File References**: Rich previews for @-mentioned files
- **Cost Tracking**: Running cost display per message and session

#### 4. Tool Execution Dashboard
- **Tool Queue**: Visual queue of pending tool executions with previews
- **Approval Workflow**: Rich approval interface with tool details and risk assessment
- **Execution Logs**: Real-time logs with filtering and search
- **Tool Performance**: Analytics on tool usage and success rates
- **MCP Server Status**: Health monitoring for local and remote MCP servers

#### 5. Configuration Management Interface
- **Visual Config Editor**: Form-based editor for OpenCode's JSON configuration
- **Schema Validation**: Real-time validation with helpful error messages
- **Configuration Profiles**: Save and switch between different configuration sets
- **Import/Export**: Easy sharing of configuration between team members
- **Diff Viewer**: Compare configuration changes before applying

#### 6. Enhanced Timeline Interface
- **Multi-Provider Timeline**: Visual timeline showing messages from different providers
- **Checkpoint Management**: Visual checkpoint creation with metadata
- **Branch Visualization**: Git-like branching interface for conversation paths
- **Diff Integration**: Side-by-side diff viewer for file changes
- **Export Options**: Multiple export formats (PDF, markdown, JSON)

### Responsive Design Requirements
- **Desktop First**: Optimize for 1920x1080 and above
- **Compact Mode**: Support for smaller screens (1366x768)
- **Multi-Monitor**: Detachable panels for extended setups
- **Panel Resizing**: Flexible panel sizing with saved layouts
- **Keyboard Navigation**: Full keyboard accessibility with customizable shortcuts

## Development Roadmap

### Phase 1: Foundation & API Layer (Weeks 1-4)
**Goal**: Establish OpenCode API server and basic desktop communication

**Week 1-2: OpenCode API Server Development**
- Add new API server component to OpenCode repository
- Implement core REST endpoints for sessions, providers, configuration
- Add WebSocket support for real-time session updates
- Create authentication middleware for multi-provider support
- Set up database integration with existing SQLite schema

**Week 3-4: Desktop App Foundation**
- Fork and configure Claudia repository for OpenCode integration
- Replace Claude Code CLI calls with OpenCode API client
- Implement basic session management with SQLite backend
- Create provider selection interface
- Basic chat functionality with one provider

**Deliverables**:
- OpenCode API server integrated into main repository
- Working desktop app connecting to OpenCode backend
- Basic provider authentication and session management
- Development documentation and API specification

### Phase 2: Core Feature Integration (Weeks 5-8)
**Goal**: Port all Claudia features to leverage OpenCode's capabilities

**Week 5-6: Tool System Integration**
- Implement visual tool approval system
- Add MCP server management interface
- Create tool execution monitoring dashboard
- Integrate LSP diagnostics viewer
- Add file browser with OpenCode's file operations

**Week 7-8: Enhanced Session Management**
- Port timeline and checkpoint system to SQLite backend
- Implement session sharing with OpenCode's share links
- Add multi-provider session support
- Create session templates and quick actions
- Develop configuration management interface

**Deliverables**:
- Feature-complete port with OpenCode integration
- Visual tool management system
- Enhanced session management with sharing
- Configuration UI with schema validation

### Phase 3: OpenCode-Specific Features (Weeks 9-12)
**Goal**: Leverage OpenCode's unique capabilities beyond Claudia's scope

**Week 9-10: Multi-Provider Enhancement**
- Implement provider performance dashboard
- Add intelligent provider routing system
- Create cost tracking and budgeting interface
- Develop provider health monitoring
- Add custom provider configuration UI

**Week 11-12: Advanced Integration**
- Implement custom commands builder with named arguments
- Add AGENTS.md visual editor
- Create agent performance analytics
- Develop LSP configuration interface
- Add theme system integration

**Deliverables**:
- Multi-provider management dashboard
- Custom commands and agents system
- Advanced configuration and theme support
- Performance analytics and monitoring

### Phase 4: Polish & Production (Weeks 13-16)
**Goal**: Production-ready application with comprehensive testing

**Week 13-14: Quality Assurance**
- Comprehensive testing across platforms (Windows, macOS, Linux)
- Performance optimization and memory management
- Security audit for multi-provider credential handling
- Accessibility compliance (WCAG 2.1)
- Integration testing with various OpenCode configurations

**Week 15-16: Release Preparation**
- Auto-update system implementation
- Installation packages for all platforms with code signing
- Comprehensive documentation and video tutorials
- Community beta testing program
- Performance benchmarking and optimization

**Deliverables**:
- Production-ready application with installers
- Complete documentation and user guides
- Performance benchmarks and optimization reports
- Community feedback integration

## Technical Implementation Details

### Enhanced Frontend Architecture

#### State Management with OpenCode Integration
```typescript
// Enhanced Zustand store structure
interface AppState {
  // OpenCode Integration
  opencode: {
    apiClient: OpenCodeClient
    serverStatus: 'connected' | 'disconnected' | 'error'
    serverVersion: string
  }

  // Enhanced Session Management
  sessions: {
    items: Session[]
    active: string | null
    shared: SharedSession[]
    templates: SessionTemplate[]
  }

  // Multi-Provider Management
  providers: {
    available: Provider[]
    authenticated: AuthenticatedProvider[]
    active: string
    performance: ProviderMetrics[]
    health: ProviderHealth[]
  }

  // Tool System
  tools: {
    available: Tool[]
    mcp: MCPServer[]
    execution: ToolExecution[]
    permissions: ToolPermission[]
  }

  // Configuration
  config: {
    current: OpenCodeConfig
    profiles: ConfigProfile[]
    schema: JSONSchema
    validation: ValidationResult[]
  }

  // Real-time Data
  messages: Message[]
  diagnostics: LSPDiagnostic[]
  streaming: StreamingState
}
```

#### Enhanced API Integration Layer
```typescript
// OpenCode API Client with comprehensive coverage
class OpenCodeClient {
  private httpClient: HTTPClient
  private websocketManager: WebSocketManager
  private eventEmitter: EventEmitter

  // Session Management
  async createSession(config: SessionConfig): Promise<Session>
  async getSession(id: string): Promise<Session>
  async getSessions(): Promise<Session[]>
  async deleteSession(id: string): Promise<void>
  async shareSession(id: string): Promise<ShareLink>
  async importSession(shareLink: string): Promise<Session>

  // Real-time Communication
  subscribeToSession(sessionId: string): SessionSubscription
  subscribeToProviderUpdates(): ProviderSubscription
  subscribeToToolExecutions(): ToolSubscription

  // Provider Management
  async getProviders(): Promise<Provider[]>
  async authenticateProvider(provider: string, credentials: any): Promise<AuthResult>
  async getProviderHealth(): Promise<ProviderHealth[]>
  async getProviderMetrics(): Promise<ProviderMetrics[]>

  // Tool System
  async getTools(): Promise<Tool[]>
  async executeTool(request: ToolExecutionRequest): Promise<ToolResult>
  async getMCPServers(): Promise<MCPServer[]>
  async addMCPServer(config: MCPServerConfig): Promise<MCPServer>

  // Configuration
  async getConfig(): Promise<OpenCodeConfig>
  async updateConfig(config: Partial<OpenCodeConfig>): Promise<void>
  async validateConfig(config: OpenCodeConfig): Promise<ValidationResult>

  // LSP Integration
  async getLSPServers(): Promise<LSPServer[]>
  async getDiagnostics(filePath?: string): Promise<LSPDiagnostic[]>

  // Custom Commands
  async getCustomCommands(): Promise<CustomCommand[]>
  async executeCommand(commandId: string, args: Record<string, string>): Promise<void>
}
```

### Enhanced Backend Integration (Rust/Tauri)

#### OpenCode Process Management
```rust
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenCodeManager {
    process: Option<std::process::Child>,
    api_port: u16,
    data_dir: PathBuf,
}

impl OpenCodeManager {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            process: None,
            api_port: 8080,
            data_dir,
        }
    }

    pub fn start_server(&mut self) -> Result<(), String> {
        let mut cmd = Command::new("opencode");
        cmd.args(&["server", "--port", &self.api_port.to_string()])
           .current_dir(&self.data_dir)
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let child = cmd.spawn()
            .map_err(|e| format!("Failed to start OpenCode server: {}", e))?;

        self.process = Some(child);
        Ok(())
    }

    pub fn stop_server(&mut self) -> Result<(), String> {
        if let Some(mut process) = self.process.take() {
            process.kill()
                .map_err(|e| format!("Failed to stop OpenCode server: {}", e))?;
        }
        Ok(())
    }

    pub fn health_check(&self) -> bool {
        // Check if API server is responding
        // Implementation for health checking
        true
    }
}

#[tauri::command]
async fn start_opencode_server(
    manager: State<'_, Mutex<OpenCodeManager>>
) -> Result<(), String> {
    let mut manager = manager.lock().unwrap();
    manager.start_server()
}

#[tauri::command]
async fn get_server_status(
    manager: State<'_, Mutex<OpenCodeManager>>
) -> Result<String, String> {
    let manager = manager.lock().unwrap();
    if manager.health_check() {
        Ok("running".to_string())
    } else {
        Ok("stopped".to_string())
    }
}
```

#### Enhanced Security and Sandboxing
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolExecutionSandbox {
    allowed_paths: Vec<PathBuf>,
    allowed_commands: Vec<String>,
    environment_restrictions: HashMap<String, String>,
    timeout_seconds: u64,
}

impl ToolExecutionSandbox {
    pub fn validate_tool_execution(
        &self,
        tool: &ToolExecutionRequest
    ) -> Result<(), SecurityViolation> {
        // Validate tool execution against security policies
        match tool.tool_id.as_str() {
            "bash" => self.validate_bash_command(&tool.params),
            "write" => self.validate_file_write(&tool.params),
            "edit" => self.validate_file_edit(&tool.params),
            _ => Ok(()),
        }
    }

    fn validate_bash_command(&self, params: &Value) -> Result<(), SecurityViolation> {
        let command = params.get("command")
            .and_then(|v| v.as_str())
            .ok_or(SecurityViolation::InvalidParameters)?;

        // Check against allowed commands and dangerous patterns
        for allowed in &self.allowed_commands {
            if command.starts_with(allowed) {
                return Ok(());
            }
        }

        Err(SecurityViolation::UnauthorizedCommand(command.to_string()))
    }
}

#[tauri::command]
async fn execute_tool_sandboxed(
    tool_request: ToolExecutionRequest,
    sandbox: State<'_, ToolExecutionSandbox>,
    api_client: State<'_, OpenCodeClient>
) -> Result<ToolResult, String> {
    // Validate against sandbox restrictions
    sandbox.validate_tool_execution(&tool_request)
        .map_err(|e| format!("Security violation: {:?}", e))?;

    // Execute via OpenCode API
    api_client.execute_tool(tool_request).await
        .map_err(|e| format!("Tool execution failed: {}", e))
}
```

### OpenCode Backend Extensions

#### API Server Integration
```go
// cmd/server/main.go - New OpenCode API server
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/sst/opencode/internal/api"
    "github.com/sst/opencode/internal/app"
    "github.com/sst/opencode/internal/config"
    "github.com/sst/opencode/internal/db"
)

func main() {
    // Load configuration
    cfg, err := config.Load()
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }

    // Initialize database
    database, err := db.Open(cfg.Data.Directory)
    if err != nil {
        log.Fatalf("Failed to open database: %v", err)
    }
    defer database.Close()

    // Initialize core services
    appServices := app.NewServices(cfg, database)

    // Create API server
    apiServer := api.NewServer(appServices, cfg)

    // Start server
    server := &http.Server{
        Addr:    ":8080",
        Handler: apiServer.Router(),
    }

    // Graceful shutdown
    go func() {
        sigChan := make(chan os.Signal, 1)
        signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
        <-sigChan

        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()

        if err := server.Shutdown(ctx); err != nil {
            log.Printf("Server shutdown error: %v", err)
        }
    }()

    log.Printf("Starting OpenCode API server on :8080")
    if err := server.ListenAndServe(); err != http.ErrServerClosed {
        log.Fatalf("Server failed to start: %v", err)
    }
}
```

#### Enhanced Session Management API
```go
// internal/api/sessions.go
package api

import (
    "encoding/json"
    "net/http"
    "strconv"

    "github.com/gorilla/mux"
    "github.com/gorilla/websocket"
    "github.com/sst/opencode/internal/session"
)

type SessionAPI struct {
    sessionManager *session.Manager
    upgrader       websocket.Upgrader
}

func (s *SessionAPI) RegisterRoutes(r *mux.Router) {
    r.HandleFunc("/sessions", s.listSessions).Methods("GET")
    r.HandleFunc("/sessions", s.createSession).Methods("POST")
    r.HandleFunc("/sessions/{id}", s.getSession).Methods("GET")
    r.HandleFunc("/sessions/{id}", s.deleteSession).Methods("DELETE")
    r.HandleFunc("/sessions/{id}/share", s.shareSession).Methods("POST")
    r.HandleFunc("/sessions/{id}/message", s.sendMessage).Methods("POST")
    r.HandleFunc("/ws/sessions/{id}", s.websocketHandler)
}

func (s *SessionAPI) createSession(w http.ResponseWriter, r *http.Request) {
    var req CreateSessionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    session, err := s.sessionManager.CreateSession(req.Config)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(session)
}

func (s *SessionAPI) websocketHandler(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    sessionID := vars["id"]

    conn, err := s.upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Printf("WebSocket upgrade failed: %v", err)
        return
    }
    defer conn.Close()

    // Create WebSocket session handler
    wsHandler := NewWebSocketHandler(sessionID, conn, s.sessionManager)
    wsHandler.Start()
}
```

#### Provider Management API
```go
// internal/api/providers.go
package api

import (
    "encoding/json"
    "net/http"

    "github.com/sst/opencode/internal/llm"
)

type ProviderAPI struct {
    llmManager *llm.Manager
}

func (p *ProviderAPI) RegisterRoutes(r *mux.Router) {
    r.HandleFunc("/providers", p.listProviders).Methods("GET")
    r.HandleFunc("/providers/auth", p.authenticateProvider).Methods("POST")
    r.HandleFunc("/providers/status", p.getProviderStatus).Methods("GET")
    r.HandleFunc("/providers/metrics", p.getProviderMetrics).Methods("GET")
}

func (p *ProviderAPI) listProviders(w http.ResponseWriter, r *http.Request) {
    providers := p.llmManager.GetAvailableProviders()
    
    response := make([]ProviderInfo, len(providers))
    for i, provider := range providers {
        response[i] = ProviderInfo{
            ID:          provider.ID,
            Name:        provider.Name,
            Models:      provider.AvailableModels(),
            Authenticated: provider.IsAuthenticated(),
            Health:      provider.HealthStatus(),
        }
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (p *ProviderAPI) getProviderMetrics(w http.ResponseWriter, r *http.Request) {
    metrics := p.llmManager.GetProviderMetrics()
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(metrics)
}
```

## Risk Assessment and Mitigation

### Technical Risks

#### 1. OpenCode API Integration Complexity
- **Risk**: Creating stable API layer without disrupting OpenCode's core functionality
- **Mitigation**: 
  - Implement API server as separate binary that imports OpenCode packages
  - Use OpenCode's existing internal interfaces without modification
  - Comprehensive integration testing with OpenCode's test suite
  - Gradual rollout with feature flags

#### 2. Multi-Provider State Management
- **Risk**: Complex state synchronization across 75+ providers
- **Mitigation**:
  - Leverage OpenCode's existing provider management system
  - Implement provider-agnostic state management patterns
  - Add provider health monitoring and automatic failover
  - Cache provider states with background refresh

#### 3. Real-time Performance
- **Risk**: WebSocket performance with multiple concurrent sessions
- **Mitigation**:
  - Implement connection pooling and message queuing
  - Add rate limiting and throttling mechanisms
  - Monitor connection health and implement automatic reconnection
  - Optimize message serialization and compression

#### 4. Cross-Platform Compatibility
- **Risk**: OpenCode binary integration across Windows, macOS, Linux
- **Mitigation**:
  - Bundle OpenCode binary with Tauri application
  - Implement platform-specific startup scripts
  - Add binary health checking and automatic recovery
  - Test on all platforms with CI/CD pipeline

### Business Risks

#### 1. OpenCode Development Velocity
- **Risk**: OpenCode breaking changes affecting desktop integration
- **Mitigation**:
  - Establish close collaboration with OpenCode maintainers
  - Implement version compatibility matrix
  - Add automated testing against OpenCode releases
  - Maintain backward compatibility for at least 2 versions

#### 2. Feature Scope vs Timeline
- **Risk**: Underestimating complexity of OpenCode integration
- **Mitigation**:
  - Phase-based development with MVP milestones
  - Regular assessment of feature complexity
  - Prioritize core functionality over nice-to-have features
  - Build buffer time into timeline estimates

### Security Risks

#### 1. Multi-Provider Credential Management
- **Risk**: Credential leakage across 75+ providers
- **Mitigation**:
  - Leverage OpenCode's existing auth.json system
  - Implement OS-native secure credential storage
  - Add provider-specific encryption keys
  - Regular security audits and penetration testing

#### 2. API Server Security
- **Risk**: Exposed API server creating attack surface
- **Mitigation**:
  - Implement localhost-only binding by default
  - Add authentication tokens for API access
  - Use HTTPS with self-signed certificates
  - Implement request rate limiting and validation

#### 3. Tool Execution Security
- **Risk**: Malicious tool execution via GUI
- **Mitigation**:
  - Leverage OpenCode's existing tool permission system
  - Implement mandatory approval workflows in GUI
  - Add tool execution sandboxing and timeout
  - Audit logging for all tool executions

## Testing Requirements

### Comprehensive Testing Strategy

#### Unit Testing
- **Frontend**: 85% coverage for React components and utilities
- **Backend**: 90% coverage for Rust adapter layer and API clients
- **OpenCode Integration**: 95% coverage for new API server components
- **Focus Areas**: State management, API communication, security features, provider management

#### Integration Testing
- **API Communication**: Mock OpenCode server for frontend testing
- **Multi-Provider Flows**: Test authentication and session management across providers
- **Tool Execution**: Test all built-in tools and MCP integration
- **Configuration Management**: Test config validation and persistence
- **Real-time Features**: WebSocket connection reliability and message handling

#### End-to-End Testing
- **Complete User Workflows**: Session creation to completion across multiple providers
- **Cross-Platform Testing**: Automated testing on Windows, macOS, Linux
- **Performance Testing**: Load testing with multiple concurrent sessions
- **Security Testing**: Penetration testing and vulnerability assessment
- **Accessibility Testing**: WCAG 2.1 compliance verification

#### OpenCode Compatibility Testing
- **Version Compatibility**: Test against OpenCode stable and development versions
- **Configuration Compatibility**: Ensure config files work with both CLI and GUI
- **Database Compatibility**: Test SQLite database sharing between CLI and GUI
- **Feature Parity**: Verify all OpenCode features accessible via GUI

### Testing Tools and Infrastructure

#### Frontend Testing
```typescript
// Jest + React Testing Library + MSW
import { render, screen, waitFor } from '@testing-library/react'
import { rest } from 'msw'
import { setupServer } from 'msw/node'

const server = setupServer(
  rest.get('/api/sessions', (req, res, ctx) => {
    return res(ctx.json({ sessions: mockSessions }))
  }),
  rest.post('/api/sessions', (req, res, ctx) => {
    return res(ctx.json({ id: 'new-session-id' }))
  })
)

describe('SessionManager', () => {
  test('creates new session with selected provider', async () => {
    render(<SessionManager />)
    
    // Test session creation workflow
    await userEvent.click(screen.getByText('New Session'))
    await userEvent.selectOptions(screen.getByLabelText('Provider'), 'anthropic')
    await userEvent.click(screen.getByText('Create'))
    
    await waitFor(() => {
      expect(screen.getByText('Session created')).toBeInTheDocument()
    })
  })
})
```

#### Backend Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_opencode_client_session_creation() {
        let mut mock_server = mockito::Server::new_async().await;
        let mock = mock_server.mock("POST", "/api/sessions")
            .with_status(200)
            .with_body(r#"{"id": "test-session", "status": "active"}"#)
            .create_async().await;

        let client = OpenCodeClient::new(&mock_server.url());
        let result = client.create_session(SessionConfig::default()).await;
        
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_tool_execution_security() {
        let sandbox = ToolExecutionSandbox::new_restrictive();
        let dangerous_request = ToolExecutionRequest {
            tool_id: "bash".to_string(),
            params: json!({"command": "rm -rf /"}),
        };

        let result = sandbox.validate_tool_execution(&dangerous_request);
        assert!(result.is_err());
    }
}
```

## Deployment Strategy

### Enhanced Distribution Strategy

#### Platform-Specific Packages
- **macOS**: Universal binary (.dmg) with Apple notarization and Gatekeeper compliance
- **Windows**: NSIS installer with Microsoft SmartScreen compatibility and code signing
- **Linux**: AppImage (universal), .deb (Debian/Ubuntu), .rpm (Red Hat/SUSE), and Flatpak

#### Auto-Update System with OpenCode Integration
```rust
// Enhanced auto-update system
#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateManager {
    current_version: Version,
    opencode_version: Version,
    update_channel: UpdateChannel,
}

impl UpdateManager {
    pub async fn check_for_updates(&self) -> Result<UpdateInfo, UpdateError> {
        // Check for both GUI and OpenCode updates
        let gui_update = self.check_gui_update().await?;
        let opencode_update = self.check_opencode_update().await?;
        
        Ok(UpdateInfo {
            gui_update,
            opencode_update,
            compatibility_verified: self.verify_compatibility(&gui_update, &opencode_update),
        })
    }

    pub async fn perform_update(&self, update_info: UpdateInfo) -> Result<(), UpdateError> {
        // Coordinate updates of both components
        if let Some(opencode_update) = update_info.opencode_update {
            self.update_opencode(opencode_update).await?;
        }
        
        if let Some(gui_update) = update_info.gui_update {
            self.update_gui(gui_update).await?;
        }
        
        Ok(())
    }
}
```

#### OpenCode Binary Integration
- **Bundled Binary**: Include compatible OpenCode binary with desktop app
- **Version Management**: Automatic OpenCode updates coordinated with GUI updates
- **Fallback Support**: Use system-installed OpenCode if bundled version fails
- **Development Mode**: Support for using local OpenCode development builds

### Infrastructure Requirements

#### Enhanced CDN Distribution
- **Global CDN**: AWS CloudFront with edge locations worldwide
- **Binary Distribution**: Separate channels for GUI and OpenCode components
- **Differential Updates**: Binary diff updates to minimize bandwidth
- **Rollback Support**: Instant rollback capability for problematic releases

#### Analytics and Monitoring
```typescript
// Privacy-respecting analytics
interface AnalyticsEvent {
  event: 'session_created' | 'provider_switched' | 'tool_executed' | 'error_occurred'
  properties: {
    provider?: string // anonymized
    tool_type?: string
    error_type?: string
    session_duration?: number
  }
  timestamp: number
  session_id: string // anonymized
}

class AnalyticsManager {
  private enabled: boolean = false // opt-in only

  async trackEvent(event: AnalyticsEvent): Promise<void> {
    if (!this.enabled) return
    
    // Send anonymized data to analytics endpoint
    await fetch('/api/analytics', {
      method: 'POST',
      body: JSON.stringify(this.anonymize(event))
    })
  }

  private anonymize(event: AnalyticsEvent): AnalyticsEvent {
    // Remove all PII and hash identifiers
    return {
      ...event,
      session_id: this.hash(event.session_id),
      properties: this.sanitizeProperties(event.properties)
    }
  }
}
```

## Success Metrics and KPIs

### Technical Performance Metrics
- **Application Startup**: < 2 seconds to ready state
- **OpenCode Integration**: < 500ms API response times
- **Memory Efficiency**: < 300MB baseline memory usage
- **Crash-Free Rate**: 99.9% crash-free sessions
- **Multi-Provider Performance**: < 100ms provider switching time

### User Adoption and Engagement
- **Initial Adoption**: 40% of OpenCode users trying desktop app within 6 months
- **Retention Rate**: 70% weekly active retention after first month
- **Feature Utilization**: 60% of users using multi-provider features
- **User Satisfaction**: 4.6+ star rating across all platforms
- **Community Growth**: 15+ community contributors within first year

### Business Impact Metrics
- **Market Penetration**: 25,000+ downloads in first quarter
- **Feature Completeness**: 95% feature parity with OpenCode CLI
- **Performance Improvement**: 4x faster onboarding for new users
- **Support Reduction**: 30% reduction in OpenCode support tickets
- **Community Contributions**: 5+ major community-contributed features

### OpenCode Ecosystem Impact
- **Core Integration**: Zero regression in OpenCode CLI functionality
- **API Adoption**: 3+ third-party applications using OpenCode API
- **Documentation Improvement**: Comprehensive API documentation for ecosystem
- **Community Feedback**: Positive feedback from OpenCode maintainers and community

## Conclusion

This comprehensive PRD provides a detailed blueprint for successfully porting Claudia's sophisticated UI to create the first desktop GUI for SST OpenCode. By leveraging OpenCode's modular Go architecture, comprehensive provider ecosystem, and robust session management system, this project will deliver a powerful desktop experience that significantly enhances the accessibility and usability of OpenCode's 75+ AI provider ecosystem.

**Key Success Factors:**

1. **Architecture Alignment**: OpenCode's modular design and API-friendly architecture make it ideal for GUI integration, unlike Claude Code's terminal-only approach.

2. **Feature Amplification**: The desktop GUI will amplify OpenCode's existing strengths (multi-provider support, session management, tool integration) while adding visual workflows impossible in a terminal.

3. **Ecosystem Enhancement**: Rather than competing with the CLI, the desktop app enhances the overall OpenCode ecosystem by providing complementary interaction patterns.

4. **Community Impact**: This project positions OpenCode as the most accessible AI coding platform, lowering barriers to entry while maintaining power user capabilities.

The phased development approach ensures steady progress while managing technical and business risks. The comprehensive testing strategy and deployment plan provide a solid foundation for production deployment. With careful execution of this plan, OpenCode Desktop will become the preferred interface for developers seeking a visual, efficient, and powerful AI coding assistant that leverages the full breadth of available AI providers.

This PRD serves as a comprehensive guide for LLM coding agents to implement a production-ready desktop application that successfully bridges the gap between Claudia's proven UI patterns and OpenCode's superior backend capabilities.