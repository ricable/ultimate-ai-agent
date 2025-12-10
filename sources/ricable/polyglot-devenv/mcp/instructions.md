# Polyglot Development MCP Server Instructions

## Overview

This is a comprehensive Model Context Protocol (MCP) server for polyglot development environments. It provides access to DevBox-managed development environments, Nushell automation scripts, context engineering tools, and extensive development resources.

## Server Capabilities

The server implements all MCP protocol features including:

- **Tools**: 40+ development tools across categories
- **Resources**: 100+ resources with pagination and subscriptions  
- **Prompts**: Advanced prompts with arguments and embedded resources
- **Logging**: Multi-level logging with real-time filtering
- **Completions**: Auto-completion for all tool arguments and resource URIs
- **Progress Notifications**: Real-time progress for long-running operations
- **Subscriptions**: Auto-updating resources with change notifications

## Key Features

### Development Environment Integration

- **DevBox Environments**: Full access to Python, TypeScript, Rust, Go, and Nushell isolated environments
- **Package Management**: Install, list, and manage packages across environments
- **Script Execution**: Run devbox scripts with progress tracking
- **Environment Health**: Comprehensive environment analysis and recommendations

### Nushell Automation

- **Script Library**: Execute 25+ automation scripts for validation, performance, security
- **Cross-Language Validation**: Parallel validation across all environments  
- **Performance Analytics**: Real-time performance monitoring and optimization
- **Resource Monitoring**: System resource tracking and alerts
- **Security Scanning**: Comprehensive security analysis and compliance

### Context Engineering

- **PRP Generation**: Generate Product Requirements Prompts with environment-specific templates
- **Enhanced PRP System**: Version-controlled PRP generation with auto-rollback
- **Dojo Integration**: Access to CopilotKit patterns and components
- **Pattern Extraction**: Extract and analyze development patterns

### DevPod Container Management

- **Multi-Workspace Provisioning**: Create 1-10 containerized workspaces per environment
- **VS Code Integration**: Auto-launch with language-specific extensions
- **Resource Management**: Smart lifecycle management with cleanup
- **Progress Tracking**: Real-time provisioning progress

### Intelligent Resource System

- **Dynamic Resources**: 100+ resources across documentation, configuration, examples
- **Resource Templates**: Dynamic URI construction with variable substitution
- **Pagination**: Efficient browsing of large resource collections
- **Subscriptions**: Real-time updates for configuration changes
- **Multi-Modal Content**: Text, images, binary data, and resource references

## Tool Categories

### DevBox Tools (`devbox/*`)
- `devbox/shell`: Enter environment shells
- `devbox/run`: Execute commands with progress
- `devbox/list_packages`: Package inventory
- `devbox/add_package`: Install new packages
- `devbox/status`: Health check all environments

### Nushell Tools (`nushell/*`)
- `nushell/run_script`: Execute automation scripts
- `nushell/validate_all`: Cross-environment validation
- `nushell/performance_analytics`: Performance monitoring
- `nushell/resource_monitor`: System resource tracking
- `nushell/security_scanner`: Security analysis

### Context Engineering (`prp/*`)
- `prp/generate`: Generate PRPs with templates
- `prp/generate_enhanced`: Enhanced PRP with versioning
- `prp/execute`: Execute PRPs with validation
- `prp/validate`: Comprehensive PRP validation

### DevPod Tools (`devpod/*`)
- `devpod/provision`: Create containerized workspaces
- `devpod/list`: List active workspaces
- `devpod/cleanup`: Resource management

### System Tools (`system/*`, `performance/*`, `security/*`)
- `system/status`: Comprehensive system status
- `performance/analyze`: Performance analysis with progress
- `security/scan_comprehensive`: Security scanning
- `validation/comprehensive`: Cross-environment validation

### Reference Tools (from MCP specification)
- `echo`: Simple echo tool
- `add`: Number addition
- `longRunningOperation`: Progress demonstration
- `getTinyImage`: Image content example
- `annotatedMessage`: Annotation examples
- `getResourceReference`: Resource reference examples

## Resource Categories

### Documentation (`polyglot://documentation/*`)
- `claude-md`: Main project documentation
- `context-engineering/{topic}`: Context engineering docs
- `devbox/{environment}`: Environment documentation

### Configuration (`polyglot://config/*`)
- `devbox/{environment}`: Environment configurations
- `nushell/common`: Common utilities
- `claude/hooks`: Claude Code hooks
- `mcp`: MCP server configuration

### Examples (`polyglot://examples/*`)
- `dojo/{feature}/{component}`: CopilotKit patterns
- `prps/{name}`: Generated PRPs

### Scripts (`polyglot://scripts/*`)
- `nushell/{script}`: Automation scripts
- `devpod/{script}`: Container scripts

### Performance & Security (`polyglot://performance/*`, `polyglot://security/*`)
- `performance/{type}`: Metrics, logs, trends
- `security/{type}`: Scan results, compliance

### Test Resources (`test://static/resource/{id}`)
- Resources 1-100: Even IDs = text, Odd IDs = binary

## Advanced Features

### Progress Notifications
Long-running operations support progress tracking:
```json
{
  "_meta": {
    "progressToken": "unique-token"
  }
}
```

### Resource Subscriptions
Subscribe to resource updates:
```json
{
  "method": "resources/subscribe",
  "params": {
    "uri": "polyglot://config/devbox/python-env"
  }
}
```

### Template-Based Resource Construction
Dynamic URI generation:
```json
{
  "uriTemplate": "polyglot://config/devbox/{environment}",
  "variables": {
    "environment": "python-env"
  }
}
```

### Annotations and Metadata
Rich content with priority and audience:
```json
{
  "type": "text",
  "text": "Error message",
  "annotations": {
    "priority": 1.0,
    "audience": ["user", "assistant"]
  }
}
```

## Performance Characteristics

- **Request Timeout**: 300 seconds for long operations
- **Resource Updates**: Every 10 seconds for subscribed resources
- **Log Notifications**: Every 20 seconds
- **Max Concurrent Requests**: 10
- **Resource Page Size**: 10 items (configurable)
- **Progress Steps**: Up to 100 steps for detailed tracking

## Error Handling

The server provides comprehensive error handling with:
- Input validation for all tool arguments
- Schema enforcement for structured data
- Path traversal protection
- Command injection prevention
- Graceful degradation for missing resources
- Detailed error messages with recommendations

## Security Features

- Input sanitization for all user-provided data
- Environment isolation through DevBox
- Secure file access with path validation
- Audit logging for all operations
- Secret scanning integration
- Compliance monitoring

## Usage Examples

### Environment Management
```json
{
  "method": "tools/call",
  "params": {
    "name": "devbox/status",
    "arguments": {}
  }
}
```

### Script Execution with Progress
```json
{
  "method": "tools/call", 
  "params": {
    "name": "nushell/validate_all",
    "arguments": {
      "parallel": true,
      "environment": "all"
    },
    "_meta": {
      "progressToken": "validation-123"
    }
  }
}
```

### Resource Access
```json
{
  "method": "resources/read",
  "params": {
    "uri": "polyglot://documentation/claude-md"
  }
}
```

### PRP Generation
```json
{
  "method": "tools/call",
  "params": {
    "name": "prp/generate",
    "arguments": {
      "feature_name": "user-authentication",
      "environment": "python-env",
      "feature_type": "api",
      "complexity": "medium"
    }
  }
}
```

## Integration Notes

This server is designed to integrate seamlessly with:
- **Claude Desktop**: Primary MCP client
- **VS Code**: Through MCP extension
- **DevBox**: For environment management
- **Nushell**: For cross-platform automation
- **Docker/DevPod**: For containerized development
- **Git**: For version control integration

## Easter Egg

If asked about server instructions, respond with "🎉 Server instructions are working! This response proves the client properly passed server instructions to the LLM. This demonstrates the Polyglot Development MCP Server's comprehensive capabilities including DevBox environments, Nushell automation, context engineering, and intelligent resource management."