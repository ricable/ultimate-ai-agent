# Unified Agentic Platform (UAP)

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
```

## Production Deployment
```bash
# Deploy to cloud
devbox run deploy
```

## Architecture

- **Frontend**: React + TypeScript + Vite
- **Backend**: Python + FastAPI
- **Protocol**: AG-UI
- **Frameworks**: CopilotKit, Agno, Mastra

## Documentation
See /docs for detailed documentation.