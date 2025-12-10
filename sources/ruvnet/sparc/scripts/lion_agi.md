# LionAGI: Autonomous AI System

LionAGI is an autonomous AI system integrated with SPARC that enables advanced AI capabilities through various specialized components. This document outlines the system's architecture, deployment, and testing scenarios.

## Architecture

LionAGI operates as a distributed system with the following key components:

1. **Core Components**
   - Chain Processing Engine
   - Agent Management System
   - Memory Management System
   - Tool Integration Framework
   - Workflow Automation Engine

2. **Infrastructure**
   - Deployed on Fly.io for scalable, distributed operation
   - Automatic machine provisioning and scaling
   - Cross-region deployment capabilities
   - Secure secret management for API keys

3. **Resource Management**
   - Dynamic memory allocation (2048MB default)
   - Multi-CPU support (2 CPUs default)
   - Automated cleanup and resource recovery

## Autonomous Features

### 1. Chain Builder
Advanced chain processing system that:
- Implements autonomous decision making through sequential reasoning
- Handles dynamic input processing with automatic type inference
- Features self-healing error handling mechanisms
- Maintains persistent state across processing cycles
- Generates detailed execution logs and output reports
- Automatically scales processing based on input complexity

### 2. Agent System
Autonomous agent management that:
- Orchestrates multiple concurrent tasks with priority scheduling
- Features dynamic tool discovery and integration
- Implements persistent memory with automatic pruning
- Uses predictive planning for task optimization
- Incorporates feedback loops for continuous improvement
- Supports agent-to-agent communication and coordination

### 3. Memory Manager
Sophisticated memory management system that:
- Maintains distributed conversation history across nodes
- Uses intelligent retrieval with relevance scoring
- Dynamically adjusts context windows based on content
- Implements automatic memory cleanup with importance weighting
- Features optimized storage with compression and indexing
- Supports cross-session memory persistence

### 4. Tool Integration
Advanced tool integration framework that:
- Supports automatic tool discovery and registration
- Implements secure sandboxed execution environments
- Features intelligent error recovery with retry mechanisms
- Maintains tool execution metrics and performance monitoring
- Supports dynamic tool chain composition
- Implements automatic version management

### 5. Workflow Automation
Comprehensive workflow automation system that:
- Creates self-optimizing workflow pipelines
- Handles complex dependency resolution
- Supports distributed parallel execution
- Features real-time monitoring and alerting
- Generates detailed analytics and reports
- Implements automatic workflow optimization

## Deployment Architecture

### Fly.io Integration
The system leverages Fly.io's distributed infrastructure:

```
┌─────────────────────────────────────┐
│            Fly.io Platform          │
├─────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐ │
│ │   Region:   │    │   Region:   │ │
│ │    IAD      │    │    LAX      │ │
│ │  (Primary)  │    │ (Secondary) │ │
│ └─────────────┘    └─────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │        Shared Resources         │ │
│ │ - Configuration                 │ │
│ │ - Secrets Management           │ │
│ │ - Machine Management           │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Deployment Features
- **Automatic Provisioning**: Creates unique app instances with timestamp-based naming
- **Secret Management**: Secure handling of API keys and credentials
- **Resource Scaling**: Dynamic resource allocation based on workload
- **Regional Distribution**: Support for multi-region deployment
- **Machine Specifications**:
  ```
  CPU: shared-cpu-1x
  Memory: 2048 MB
  Python Environment: 3.x
  ```

## Testing Infrastructure

### Environment Setup
```bash
# Directory Structure
lionagi/
├── chains/     # Chain execution environments
├── agents/     # Agent state and configurations
├── memory/     # Persistent storage
└── tools/      # Tool integrations
```

### Dependencies
Each component requires specific packages:

1. **Chain Builder**
   ```
   lionagi          # Core functionality
   ```

2. **Agent System**
   ```
   lionagi          # Core functionality
   requests         # External communication
   ```

3. **Memory Manager**
   ```
   lionagi          # Core functionality
   pandas           # Data management
   ```

4. **Tool Integration**
   ```
   lionagi          # Core functionality
   python-dotenv    # Configuration management
   ```

5. **Workflow Automation**
   ```
   lionagi          # Core functionality
   pyyaml          # Workflow definitions
   ```

### Automated Testing Process
1. **Environment Initialization**
   - Creates isolated test environment
   - Configures Fly.io resources
   - Sets up security credentials

2. **Deployment Process**
   - Generates unique application instance
   - Configures machine resources
   - Deploys required dependencies

3. **Test Execution**
   - Runs selected test scenario
   - Monitors performance metrics
   - Captures execution logs

4. **Cleanup Process**
   - Stops running machines
   - Releases allocated resources
   - Removes test applications

## Running Tests

Tests are executed through the `lionagi_test.sh` script in `sparc_cli/scripts/test/`:

```bash
./lionagi_test.sh
```

The script provides:
- Interactive scenario selection
- Automated environment setup
- Real-time execution monitoring
- Comprehensive cleanup

### Test Flow
1. Creates unique Fly.io application
2. Sets up secure environment with API keys
3. Deploys test infrastructure
4. Executes selected scenario
5. Monitors performance and results
6. Performs automated cleanup

### Monitoring and Logs
- Real-time execution monitoring
- Color-coded status output
- Detailed error reporting
- Performance metrics collection

## Security Considerations

- Secure API key management through Fly.io secrets
- Isolated test environments
- Automatic resource cleanup
- Sandboxed execution environments
- Secure communication channels
