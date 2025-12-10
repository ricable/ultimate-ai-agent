OpenCode Repository Guide for LLM Agents
Project Overview
OpenCode is a powerful terminal-based AI coding assistant built in Go, designed to provide intelligent coding assistance directly in the terminal. It's in early development but offers a robust TUI (Terminal User Interface) with extensive AI provider support.
Key Characteristics

Language: Go (requires Go 1.24.0+)
Architecture: Modular design with clean separation of concerns
UI Framework: Bubble Tea for terminal interface
Database: SQLite for persistent storage
Status: Early development, features may change

Repository Structure
opencode/
├── cmd/                    # CLI interface using Cobra
├── internal/
│   ├── app/               # Core application services
│   ├── config/            # Configuration management
│   ├── db/                # Database operations and migrations
│   ├── llm/               # LLM providers and tools integration
│   ├── tui/               # Terminal UI components and layouts
│   ├── logging/           # Logging infrastructure
│   ├── message/           # Message handling
│   ├── session/           # Session management
│   └── lsp/               # Language Server Protocol integration
├── .opencode.json         # Project configuration
└── README.md
Core Features and Architecture
1. Multi-Provider AI Support

75+ AI Providers via Models.dev integration
Primary Providers: OpenAI, Anthropic Claude, Google Gemini, AWS Bedrock, Groq, Azure OpenAI
Local Model Support: Yes, with custom provider configuration
Authentication: Managed via opencode auth login and stored in ~/.local/share/opencode/auth.json

2. Session Management

Persistent Sessions: SQLite database storage
Session Operations: Create, list, switch, share sessions
Session Sharing: Generate shareable links for debugging/reference
Multiple Sessions: Parallel sessions supported

3. Tool Integration
OpenCode provides built-in tools accessible to AI:
File Operations

glob: Find files by pattern
grep: Search file contents
ls: List directory contents
view: View file contents
write: Write to files
edit: Edit files
patch: Apply patches

System Operations

bash: Execute shell commands
fetch: Fetch data from URLs
agent: Run sub-tasks with AI agent

Development Tools

diagnostics: Get LSP diagnostics information

4. Configuration System
Configuration Locations (Priority Order)

~/.config/opencode/config.json (Global)
opencode.json (Project root, Git-aware)
./.opencode.json (Local directory)

Key Configuration Sections
json{
  "$schema": "https://opencode.ai/config.json",
  "theme": "opencode",
  "model": "anthropic/claude-sonnet-4-20250514",
  "autoshare": false,
  "autoupdate": true,
  "provider": {},
  "keybinds": {},
  "mcp": {},
  "agents": {
    "primary": { "model": "claude-3.7-sonnet", "maxTokens": 5000 },
    "task": { "model": "claude-3.7-sonnet", "maxTokens": 5000 },
    "title": { "model": "claude-3.7-sonnet", "maxTokens": 80 }
  }
}
5. Model Context Protocol (MCP) Integration

Local Servers: Stdio communication
Remote Servers: SSE (Server-Sent Events)
Tool Discovery: Automatic tool detection from MCP servers
Security: Permission-based tool execution

6. Language Server Protocol (LSP)

Multi-language Support: Configurable LSP servers
Diagnostics Integration: Error checking exposed to AI
File Watching: Automatic change notifications
Current Limitation: Only diagnostics exposed to AI (full LSP features available in codebase)

Command Line Interface
Installation Commands
bash# npm/bun/pnpm/yarn
npm install -g opencode-ai

# Install script
curl -fsSL https://opencode.ai/install | bash

# Homebrew (macOS)
brew install sst/tap/opencode

# Arch Linux
paru -S opencode-bin

# Go install
go install github.com/sst/opencode@latest
Usage Commands
bash# Start OpenCode
opencode

# Start with debug logging
opencode -d

# Start with specific working directory
opencode -c /path/to/project

# Non-interactive mode (scripting)
opencode -p "Explain Go context" -f json -q

# Authentication
opencode auth login
opencode auth list
opencode auth logout <provider>

# Updates
opencode update
opencode update --version 0.1.0
CLI Flags
FlagShortDescription--help-hDisplay help--debug-dEnable debug mode--cwd-cSet working directory--prompt-pNon-interactive prompt--output-format-fOutput format (text, json)--quiet-qHide spinner
Key Bindings System
Leader Key Concept

Default Leader: Ctrl+X
Usage Pattern: Press leader key, then action key
Example: Ctrl+X → n (new session)

Essential Keybinds
ActionKeybindDescriptionSession ManagementNew Session<leader>nCreate new sessionList Sessions<leader>lSession pickerShare Session<leader>sGenerate share linkSwitch SessionCtrl+ASession switcherUI NavigationHelp<leader>h or ?Toggle helpEditor<leader>eOpen editorModel List<leader>mModel selectorTheme List<leader>tTheme pickerCommunicationSend MessageEnter or Ctrl+SSubmit inputNew LineShift+EnterAdd line breakInterruptEscCancel generationSystemQuitCtrl+C or <leader>qExit applicationCommandsCtrl+KCommand dialog
Custom Commands System
Command Structure
Custom commands are Markdown files with predefined prompts:
Locations

User Commands: ~/.config/opencode/commands/ (prefix: user:)
Project Commands: <PROJECT>/.opencode/commands/ (prefix: project:)

Named Arguments
Use $VARIABLE_NAME placeholders for dynamic input:
markdown# Fetch Context for Issue $ISSUE_NUMBER
RUN gh issue view $ISSUE_NUMBER --json title,body,comments
RUN git grep --author="$AUTHOR_NAME" -n .
Organization

Subdirectories supported: git/commit.md → user:git:commit
Execute via Ctrl+K command dialog

Agent Rules System (AGENTS.md)
Purpose
Define behavioral rules and context for AI agents.
Locations (Priority Order)

Global: ~/.config/opencode/AGENTS.md (personal rules)
Project: <PROJECT_ROOT>/AGENTS.md (team-shared rules)

Usage

Created via /init command in OpenCode
Committed to Git for team sharing
Scanned automatically to understand project context

Example Content
markdown# Project Context
This is an SST v3 monorepo with TypeScript.
The project uses bun workspaces for package management.

# Development Guidelines
- Use TypeScript strict mode
- Follow conventional commits
- Write tests for new features
Theme System
Built-in Themes

opencode (default)
catppuccin
dracula
flexoki
gruvbox
monokai
onedark
tokyonight
tron

Custom Themes
json{
  "tui": {
    "theme": "custom",
    "customTheme": {
      "primary": "#ffcc00",
      "secondary": "#00ccff",
      "accent": { "dark": "#aa00ff", "light": "#ddccff" },
      "error": "#ff0000"
    }
  }
}
Theme Loading Priority

Built-in themes (embedded)
~/.config/opencode/themes/*.json
<project>/.opencode/themes/*.json
./.opencode/themes/*.json

Development Guidelines for LLM Agents
When Working with OpenCode Repository
1. Architecture Understanding

Respect the modular design: Changes should fit within existing package boundaries
Database operations: Use the established migration system in internal/db
UI components: Follow Bubble Tea patterns in internal/tui
Provider integration: Extend internal/llm for new AI providers

2. Configuration Changes

Always validate against schema: Use https://opencode.ai/config.json
Maintain backward compatibility: Add new fields as optional
Consider both global and project configs: Changes should work in both contexts

3. Tool Development

Follow existing tool patterns: Study internal/llm/tools
Implement proper permissions: User approval required for sensitive operations
Error handling: Provide clear error messages and recovery options
Security considerations: Validate inputs, sandbox execution when possible

4. Testing Considerations

Terminal UI testing: Consider automated testing limitations
Cross-platform compatibility: Test on Linux, macOS, Windows
Provider testing: Mock external API calls
Database testing: Use test databases, clean up after tests

5. Documentation Requirements

Update README.md: For user-facing changes
Schema updates: Update config schema for new options
Command documentation: Update CLI help text
Architecture decisions: Document significant design choices

Common Development Patterns
Adding a New Tool

Define tool interface in internal/llm/tools
Implement tool logic with proper error handling
Add tool to registry in internal/llm
Update documentation and tests
Consider permission requirements

Adding a New Provider

Implement provider interface in internal/llm/providers
Add configuration schema
Update Models.dev integration if needed
Add authentication flow
Test with real API calls

UI Component Development

Follow Bubble Tea Model/Update/View pattern
Handle all message types appropriately
Implement proper focus management
Add keyboard navigation
Test across different terminal sizes

Security Considerations
For Agent Development

Input validation: Always validate user inputs
Command execution: Use safe command execution patterns
File operations: Respect file permissions and sandboxing
API keys: Never log or expose API keys
MCP tools: Implement proper permission checking

For Configuration

Sensitive data: Store in appropriate secure locations
File permissions: Set restrictive permissions on config files
Schema validation: Validate all configuration inputs
Default security: Secure defaults, opt-in for dangerous features

Advanced Features
Non-Interactive Mode
Perfect for scripting and automation:
bash# Get JSON response
opencode -p "Explain Go interfaces" -f json -q

# Pipe to other tools
opencode -p "Generate unit tests for main.go" | tee tests.go
Session Sharing

Generate public links for debugging
Share context with team members
Reference sessions in documentation

Multi-Agent Support

Configure different agents for different tasks
Separate models for primary, task, and title generation
Customize token limits per agent type

Best Practices for LLM Agents

Understand the user's context: Check for AGENTS.md files and project structure
Respect configuration: Use existing config patterns and schemas
Follow Go conventions: Proper error handling, interface design, package organization
Maintain UX consistency: Follow established UI patterns and keybinds
Security first: Validate inputs, request permissions, avoid dangerous defaults
Document changes: Update relevant documentation for user-facing changes
Test thoroughly: Consider cross-platform compatibility and edge cases

This guide should provide LLM agents with comprehensive understanding of the OpenCode repository structure, features, and development patterns needed to contribute effectively to the project.