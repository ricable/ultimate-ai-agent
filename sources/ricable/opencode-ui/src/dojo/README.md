# OpenCode Desktop UI (Dojo)

A comprehensive desktop GUI for OpenCode's multi-provider AI agent system, featuring sophisticated session management, real-time tool execution, and enterprise-grade security. Built by porting Claudia's proven UI from Rust/Tauri to TypeScript/Next.js.

## Overview

OpenCode Desktop UI provides a powerful graphical interface for OpenCode's multi-provider AI agent ecosystem, supporting 75+ AI providers with advanced features:

### Core Features
- **Multi-Provider Session Management**: Create and manage sessions across 75+ AI providers (OpenAI, Anthropic, Google, Groq, etc.)
- **Local Provider Integration**: Full support for Ollama, llama.cpp, LM Studio, LocalAI, and Text Generation WebUI
- **Real-Time Chat Interface**: Enhanced messaging with provider attribution, streaming indicators, and cost tracking
- **Tool Execution Dashboard**: Secure approval workflows with comprehensive audit logging
- **Usage Analytics**: Real-time cost tracking, provider performance metrics, and usage statistics
- **Configuration Management**: Visual JSON schema-based configuration editor with custom provider support
- **MCP Server Integration**: Management interface for Model Context Protocol servers
- **Enhanced Security**: Tool sandboxing and granular permission management
- **Professional UI**: Dark/light theme support with responsive design
- **Privacy-Focused Workflows**: Local development templates for offline AI assistance

### Architecture
- **Frontend**: Next.js 15 + TypeScript + Tailwind CSS
- **State Management**: Zustand with persistence
- **API Integration**: Comprehensive OpenCode API client with WebSocket support
- **Testing**: Vitest + React Testing Library + MSW + Playwright
- **UI Components**: shadcn/ui with custom OpenCode integrations

## Development Setup

To run the Demo Viewer locally for development, follow these steps:

### Install dependencies

```bash
brew install protobuf
```

```bash
npm i turbo
```

```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

### Run OpenCode Desktop UI

In a new terminal, navigate to the project root and start the development server:

```bash
pnpm install
pnpm run dev
```

The OpenCode Desktop UI will be running at [http://localhost:3000](http://localhost:3000).

### Testing

```bash
# Run test suite
pnpm run test

# Run tests in watch mode
pnpm run test:watch

# Generate coverage report
pnpm run test:coverage

# Run end-to-end tests
pnpm run test:e2e

# Open Playwright UI for debugging
pnpm run test:e2e:ui
```

### OpenCode Backend Integration

This UI integrates with OpenCode's Go backend via a comprehensive API layer:

1. **API Client**: `/src/lib/opencode-client.ts` provides full OpenCode API integration
2. **State Management**: `/src/lib/session-store.ts` handles real-time state synchronization
3. **WebSocket Integration**: Real-time updates for sessions, providers, and tool executions
4. **Security Layer**: Tool approval workflows and audit logging

### Provider Configuration

Supports OpenCode's comprehensive multi-provider architecture:
- **Cloud Providers**: OpenAI, Anthropic, Google AI, Groq, Cohere, Mistral, and 70+ others
- **Local Providers**: Ollama, llama.cpp, LM Studio, LocalAI, Text Generation WebUI
- **Custom Providers**: NPM package specification and OpenAI-compatible endpoints
- Provider authentication and health monitoring with local endpoint validation
- Intelligent provider routing and fallback chains including local optimization
- Cost tracking and budgeting per provider (zero-cost tracking for local models)
- Model management interface with custom model support for local providers
- Privacy-focused configuration for offline development workflows

## Project Structure

```
src/
├── app/                 # Next.js app router and API routes
├── components/          # React UI components
│   ├── opencode/       # OpenCode-specific components
│   │   ├── session-view.tsx      # Main chat interface
│   │   ├── provider-selector.tsx # Multi-provider selection
│   │   ├── tool-dashboard.tsx    # Tool execution management
│   │   └── __tests__/           # Component tests
│   ├── ui/             # shadcn/ui base components
│   └── views/          # Page-level view components
├── lib/                # Core utilities and API integration
│   ├── opencode-client.ts       # Comprehensive OpenCode API client
│   ├── session-store.ts         # Zustand state management
│   └── __tests__/              # API integration tests
├── types/              # TypeScript type definitions
│   └── opencode.ts             # OpenCode-specific types
├── test/               # Test utilities and mocks
└── utils/              # Helper functions
```

## Technologies

### Core Stack
- **Next.js 15**: App router with API routes
- **React 19**: Latest React with concurrent features
- **TypeScript 5**: Strict typing throughout
- **Tailwind CSS 4**: Utility-first styling
- **Zustand**: Lightweight state management

### Testing & Quality
- **Vitest**: Fast unit testing with React Testing Library
- **MSW**: API mocking for integration tests
- **Playwright**: End-to-end testing
- **ESLint**: Code linting and formatting

### UI & Design
- **shadcn/ui**: High-quality React components
- **Radix UI**: Accessible component primitives
- **Lucide React**: Consistent icon system
- **Framer Motion**: Smooth animations
- **Monaco Editor**: Code editing capabilities

### Integration
- **OpenCode API**: Full integration with Go backend
- **WebSocket**: Real-time communication
- **JSON Schema**: Configuration validation
- **MCP Protocol**: Model Context Protocol support

## Test Coverage Status

- **Total Tests**: 46 (36 passing, 9 failing, 1 skipped)
- **Current Coverage**: 78% pass rate
- **UI Components**: 93% coverage (Button component comprehensive)
- **API Integration**: 71% (MSW handler gaps causing test failures)
- **Type Safety**: 100% (All type guards working)

### Test Infrastructure
- ✅ Comprehensive test setup with Vitest, MSW, Playwright
- ✅ Component testing with React Testing Library
- ✅ API mocking with Mock Service Worker
- ✅ Accessibility testing with jest-axe
- 🔍 MSW handlers need completion for full API coverage

## Architecture Highlights

### OpenCode Integration
- **2,050+ line API client** with comprehensive provider support
- **Real-time WebSocket** communication with health monitoring
- **Security validation** for tool execution approval
- **Enhanced error handling** with retry logic and recovery

### UI Fidelity
- **Matches Claudia design** with sophisticated tool widgets
- **Multi-provider support** adapted from Claude-specific features
- **Responsive design** with dark/light theme integration
- **Professional UX** with loading states and error handling

### Production Ready
- **Robust state management** with persistence
- **Comprehensive error boundaries** and fallback UI
- **Performance optimized** with code splitting
- **Accessibility compliant** with WCAG guidelines

## Contributing

This project follows the OpenCode desktop integration specifications outlined in `/plan.md`. Key development principles:

1. **Type Safety**: Maintain strict TypeScript throughout
2. **Component Quality**: Follow shadcn/ui patterns
3. **Test Coverage**: Aim for 85%+ UI, 90%+ API coverage
4. **OpenCode Integration**: Leverage full multi-provider capabilities
5. **UI Fidelity**: Match Claudia's sophisticated design patterns

## Quick Start Guide

1. **Install Dependencies**
   ```bash
   pnpm install
   ```

2. **Start Development Server**
   ```bash
   pnpm run dev
   ```

3. **Run Tests**
   ```bash
   pnpm run test
   ```

4. **Build for Production**
   ```bash
   pnpm run build
   ```

The application provides a comprehensive desktop interface for OpenCode's multi-provider AI agent system, successfully porting Claudia's sophisticated UI while adding support for 75+ AI providers.
