# CLAUDE.md - OpenCode Desktop UI

This file provides comprehensive guidance for Claude Code when working with the OpenCode Desktop UI application.

## Project Overview

OpenCode Desktop UI is a Next.js-based frontend application that provides a modern, intuitive interface for the OpenCode AI coding agent system. It features comprehensive session management, multi-provider support, advanced tooling, and real-time collaboration capabilities.

## Development Commands

### Primary Application (Dojo - OpenCode Desktop UI)
```bash
cd src/dojo
pnpm install
pnpm run dev             # Next.js development server (localhost:3000)
pnpm run build           # Production build
pnpm run lint            # ESLint code quality checks
pnpm run test            # Run test suite (Vitest)
pnpm run test:e2e        # End-to-end tests (Playwright)
pnpm run test:coverage   # Test coverage report
pnpm run typecheck       # TypeScript type checking
```

### OpenCode Backend (if developing locally)
```bash
# Install OpenCode globally
npm install -g opencode-ai@latest

# Or run from source
cd code/opencode
bun install
bun run packages/opencode/src/index.ts

# Start OpenCode server for API integration
opencode serve --port 8080 --print-logs
```

### Documentation
```bash
cd docs
mintlify dev --port=4000  # Development server
```

## Architecture Overview

This repository contains multiple interconnected projects focused on AI coding agents and desktop interfaces:

### 1. Claudia (Desktop GUI for Claude Code) - Reference Implementation
- **Tech Stack**: Tauri 2 + React 18 + TypeScript + Rust backend
- **Purpose**: Powerful desktop GUI for Claude Code with advanced features
- **Key Features**: 
  - Project & session management with visual timeline
  - Custom AI agents with sandboxed execution
  - Usage analytics dashboard with cost tracking
  - MCP server management
  - Advanced security with OS-level sandboxing

### 2. OpenCode (Multi-Provider AI Agent) - Backend System
- **Tech Stack**: TypeScript + Go TUI + Client/Server architecture
- **Purpose**: Open-source terminal AI agent supporting 75+ providers
- **Key Components**:
  - `packages/opencode/`: Core TypeScript implementation
  - `packages/tui/`: Go-based terminal user interface
  - `packages/web/`: Astro-based documentation site
  - `packages/function/`: API functions

### 3. Dojo (OpenCode Desktop UI) - Primary Application
- **Tech Stack**: Next.js 15 + TypeScript + OpenCode API integration
- **Purpose**: Desktop GUI for OpenCode multi-provider AI agent system  
- **Current Status**: ✅ **FEATURE COMPLETE** - Full Claudia UI port with enhancements
- **Complete Feature Set**: 
  - **Multi-Provider Session Management**: Support for 75+ AI providers with intelligent routing
  - **Real-Time Chat Interface**: Advanced message streaming with tool execution approval
  - **Enhanced Timeline System**: Tree-style timeline with branching visualization and checkpoints
  - **Advanced Checkpoints**: Full diff comparison, forking, and restoration capabilities
  - **Comprehensive Usage Analytics**: Interactive charts, cost tracking, and multi-format export
  - **MCP Server Management**: Complete server configuration, monitoring, and marketplace
  - **Agent Management System**: Agent creation, execution monitoring, and performance analytics
  - **Provider Intelligence**: Health monitoring, cost estimation, and smart routing
  - **Configuration Management**: JSON schema validation with visual editors
  - **Tool Integration**: Built-in tools, MCP tools, and custom tool development
  - **Security Features**: Tool sandboxing, permission management, and audit logging
  - **Modern UX**: Responsive design, dark/light themes, accessibility compliance

## Current Implementation Status ✅

### Core Features (100% Complete)
- ✅ **OpenCode API Integration**: Full HTTP/WebSocket client with automatic mock mode fallback
- ✅ **Multi-Provider Support**: 75+ providers including local models (Ollama, llama.cpp, LM Studio)
- ✅ **Session Management**: Create, manage, share, and organize sessions with SQLite backend
- ✅ **Enhanced Timeline**: Tree-style visualization with branching and checkpoint restoration
- ✅ **Provider Dashboard**: Health monitoring, cost tracking, intelligent routing
- ✅ **Tool System**: Comprehensive tool management with MCP server integration
- ✅ **Agent Management**: Creation, execution monitoring, performance analytics
- ✅ **Usage Analytics**: Interactive charts, cost tracking, multi-format export
- ✅ **MCP Integration**: Server configuration, monitoring, marketplace templates
- ✅ **Security**: Tool sandboxing, permission management, audit logging

### Technical Implementation (100% Complete)
- ✅ **Frontend Architecture**: Next.js 15 with TypeScript, shadcn/ui components
- ✅ **State Management**: Zustand store with real-time updates via WebSocket
- ✅ **API Client**: Comprehensive OpenCode client with mock mode for development
- ✅ **Testing**: Vitest unit tests, Playwright E2E, MSW API mocking
- ✅ **Type Safety**: Complete TypeScript definitions for all OpenCode APIs
- ✅ **Responsive Design**: Mobile-first design with dark/light theme support
- ✅ **Accessibility**: WCAG 2.1 compliance with keyboard navigation

## Key Configuration Files

### OpenCode Configuration
- `code/opencode/opencode.json`: Main configuration with experimental hooks
- `code/opencode/packages/opencode/config.schema.json`: Complete configuration schema
- Supports provider configurations, MCP servers, keybinds, themes

### Dojo Configuration
- `src/dojo/package.json`: Frontend dependencies and build scripts
- `src/dojo/next.config.ts`: Next.js configuration with optimizations
- `src/dojo/tailwind.config.ts`: Tailwind CSS configuration with design system
- `src/dojo/tsconfig.json`: TypeScript configuration with strict mode

## Project Structure Overview

### Monorepo Organization
- `code/`: Reference implementations (claudia, opencode source)
- `src/dojo/`: **Primary OpenCode Desktop UI application**
- `docs/`: Mintlify documentation for the project
- Each project maintains its own package.json and build system

### Dojo Frontend Architecture (Primary Codebase)
```
src/dojo/src/
├── app/                        # Next.js 15 app router pages
│   ├── page.tsx               # Main application entry point
│   ├── layout.tsx             # Global layout with providers
│   └── globals.css            # Global styles and design tokens
├── components/
│   ├── opencode/              # Core OpenCode functionality
│   │   ├── opencode-layout.tsx        # Main application layout
│   │   ├── session-timeline.tsx       # Enhanced timeline system
│   │   ├── provider-selector.tsx      # Multi-provider interface
│   │   ├── session-creation.tsx       # Session management
│   │   ├── tool-dashboard.tsx         # Tool system interface
│   │   ├── advanced-checkpoint-manager.tsx # Checkpoint system
│   │   ├── checkpoint-diff-viewer.tsx      # Diff comparison
│   │   └── checkpoint-fork-manager.tsx     # Branching system
│   ├── views/                 # Main application views
│   │   ├── projects-view.tsx          # Project management
│   │   ├── session-view.tsx           # Chat interface
│   │   ├── providers-view.tsx         # Provider dashboard
│   │   ├── usage-dashboard-view.tsx   # Analytics dashboard
│   │   ├── mcp-view.tsx              # MCP server management
│   │   ├── agents-view.tsx           # Agent system
│   │   ├── tools-view.tsx            # Tool management
│   │   └── settings-view.tsx         # Configuration
│   ├── mcp/                   # MCP server components
│   │   ├── server-config-form.tsx     # Configuration forms
│   │   ├── server-dashboard.tsx       # Monitoring interface
│   │   ├── server-templates.tsx       # Template marketplace
│   │   └── testing-utils.tsx          # Testing utilities
│   ├── agents/                # Agent management components
│   │   ├── agent-analytics.tsx        # Performance metrics
│   │   ├── agent-testing.tsx          # Testing framework
│   │   ├── agent-marketplace.tsx      # Discovery interface
│   │   └── agent-execution-monitor.tsx # Real-time monitoring
│   └── ui/                    # Shared shadcn/ui components
├── lib/
│   ├── opencode-client.ts     # **Core API client with mock mode**
│   ├── session-store.ts       # **Zustand state management**
│   ├── analytics-utils.ts     # Analytics processing utilities
│   ├── utils.ts              # Common utility functions
│   └── types/                # TypeScript definitions
│       ├── opencode.ts       # Core OpenCode API types
│       ├── mcp.ts           # MCP system types
│       └── agents.ts        # Agent system types
├── hooks/                    # Custom React hooks
├── test/                     # Testing utilities and mocks
└── types/                    # Additional type definitions
```

## Development Workflow & Guidelines

### Code Standards
- **TypeScript-first**: All code must be TypeScript with strict type checking
- **Component Architecture**: Follow shadcn/ui patterns with composition
- **State Management**: Use Zustand for global state, React hooks for local state
- **API Integration**: All OpenCode API calls go through the centralized client
- **Testing**: Unit tests for utilities, integration tests for components
- **Accessibility**: WCAG 2.1 compliance with proper ARIA labels

### Local Development Setup
1. **Frontend**: `cd src/dojo && pnpm install && pnpm run dev`
2. **Backend**: OpenCode server auto-detected or mock mode fallback
3. **Testing**: `pnpm test` for unit tests, `pnpm test:e2e` for E2E
4. **Type Checking**: `pnpm typecheck` for TypeScript validation

### OpenCode Integration
- **API Client**: Automatic server detection with graceful mock mode fallback
- **Real-time Updates**: WebSocket integration for live session updates
- **Provider Management**: Support for 75+ providers including local models
- **Tool System**: Integration with OpenCode's built-in and MCP tools
- **Session Persistence**: SQLite backend integration for robust data storage

## Provider System (OpenCode Integration)

### Supported Providers (75+)
- **Foundation Models**: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
- **High Performance**: Groq, Together AI, Fireworks AI
- **Local Models**: Ollama, llama.cpp, LM Studio, LocalAI, Text Generation WebUI
- **Specialized**: Cohere, Mistral, Perplexity, DeepSeek, AI21, Hugging Face
- **Custom Providers**: NPM package specification and custom endpoints

### Enhanced Features
- **Intelligent Routing**: Automatic provider selection based on task type
- **Cost Optimization**: Real-time cost tracking with budget management
- **Health Monitoring**: Connection status and performance metrics
- **Local Model Support**: Zero-cost local inference with privacy features
- **Fallback Chains**: Automatic failover between providers

## Testing Infrastructure

### Comprehensive Test Coverage
- **Unit Tests**: Vitest for components and utilities (85%+ coverage target)
- **Integration Tests**: React Testing Library with MSW API mocking
- **E2E Tests**: Playwright for complete user workflows
- **API Testing**: Mock server integration for OpenCode API testing
- **Type Testing**: TypeScript strict mode for compile-time validation

### Test Commands
```bash
cd src/dojo
pnpm test              # Run all unit tests
pnpm test:coverage     # Generate coverage report
pnpm test:e2e          # Run Playwright E2E tests
pnpm test:watch        # Watch mode for development
```

## Deployment & Build

### Production Build
```bash
cd src/dojo
pnpm run build         # Next.js production build
pnpm run start         # Production server
```

### Build Optimization
- **Next.js 15**: Latest features with performance optimizations
- **Tree Shaking**: Automatic code splitting and bundling
- **Image Optimization**: Automatic image compression and WebP conversion
- **Bundle Analysis**: Built-in analyzer for performance monitoring

## Key Implementation Notes

### OpenCode API Client (`src/dojo/src/lib/opencode-client.ts`)
- **Automatic Server Detection**: Checks for OpenCode server on startup
- **Mock Mode Fallback**: Graceful degradation when server unavailable
- **WebSocket Support**: Real-time updates for session changes
- **Error Handling**: Comprehensive error recovery and user feedback
- **Type Safety**: Complete TypeScript coverage for all API endpoints

### Session Store (`src/dojo/src/lib/session-store.ts`)
- **Zustand State Management**: Reactive state with persistence
- **Real-time Sync**: WebSocket integration for live updates
- **Multi-Provider Support**: Seamless switching between providers
- **Tool Integration**: Complete tool system with approval workflows
- **Analytics Integration**: Real-time usage and cost tracking

### UI Component System
- **shadcn/ui Foundation**: Modern, accessible component library
- **Custom Components**: OpenCode-specific functionality
- **Theme System**: Dark/light mode with system preference detection
- **Responsive Design**: Mobile-first approach with breakpoint optimization
- **Animation System**: Smooth transitions with Framer Motion integration

## Development Status Summary

🎉 **PROJECT STATUS: FEATURE COMPLETE**

The OpenCode Desktop UI (Dojo) is now a fully functional, production-ready application that successfully ports all Claudia UI features to work with OpenCode's multi-provider architecture. The application includes:

- ✅ Complete UI/UX parity with Claudia reference design
- ✅ Enhanced multi-provider support (75+ providers)
- ✅ Advanced session management with real-time collaboration
- ✅ Comprehensive tool system with MCP integration
- ✅ Agent management with performance analytics
- ✅ Usage analytics with interactive dashboards
- ✅ Modern development stack with comprehensive testing
- ✅ Production-ready deployment configuration

**Next Steps**: The application is ready for production deployment and can serve as the primary desktop interface for OpenCode users seeking a visual, feature-rich alternative to the terminal interface.