# MCP-Modes Directory Structure

This document outlines the complete structure of the `.roo/mcp-modes` directory and its contents.

## Directory Structure

```
.roo/mcp-modes/
├── README.md                           # Main documentation for MCP-optimized modes
├── modes.json                          # Configuration file defining all MCP modes
├── STRUCTURE.md                        # This file - structure documentation
├── rules-mcp-orchestrator/
│   └── rules.md                        # MCP Orchestrator mode rules and guidelines
├── rules-mcp-intelligent-coder/
│   └── rules.md                        # MCP Intelligent Coder mode rules and guidelines
├── rules-mcp-researcher/
│   └── rules.md                        # MCP Researcher mode rules and guidelines
├── rules-mcp-optimizer/
│   └── rules.md                        # MCP Optimizer mode rules and guidelines
├── rules-mcp-management/
│   └── rules.md                        # MCP Management mode rules and guidelines
└── rules-mcp-tutorial/
    └── rules.md                        # MCP Tutorial mode rules and guidelines
```

## File Descriptions

### Core Configuration Files

#### README.md
- **Purpose**: Main documentation explaining MCP-optimized modes
- **Content**: Overview, architecture, features, usage guidelines
- **Audience**: Developers and users implementing MCP-optimized workflows

#### modes.json
- **Purpose**: Configuration file defining all MCP-optimized modes
- **Content**: Mode definitions, capabilities, MCP server requirements, settings
- **Structure**: JSON configuration with mode specifications and integration settings

#### STRUCTURE.md
- **Purpose**: Documentation of directory structure and organization
- **Content**: Complete file listing, descriptions, and organizational principles
- **Audience**: Developers maintaining and extending the MCP modes system

### Mode-Specific Rules Directories

Each mode has its own rules directory following the pattern `rules-mcp-{mode-name}/`:

#### rules-mcp-orchestrator/
- **Mode**: 🤖 MCP Orchestrator (`mcp-orchestrator`)
- **Purpose**: Coordinate complex workflows involving multiple MCP servers
- **Key Features**: Service orchestration, dependency management, distributed processing
- **File**: `rules.md` - Comprehensive guidelines for orchestrator mode

#### rules-mcp-intelligent-coder/
- **Mode**: 🧠 MCP Intelligent Coder (`mcp-intelligent-coder`)
- **Purpose**: Advanced coding with external tool integration
- **Key Features**: Enhanced code generation, real-time validation, documentation integration
- **File**: `rules.md` - Guidelines for intelligent coding with MCP integration

#### rules-mcp-researcher/
- **Mode**: 🔍 MCP Researcher (`mcp-researcher`)
- **Purpose**: Comprehensive research using external knowledge bases
- **Key Features**: Multi-source research, fact checking, citation management
- **File**: `rules.md` - Research methodology and integration guidelines

#### rules-mcp-optimizer/
- **Mode**: ⚡ MCP Optimizer (`mcp-optimizer`)
- **Purpose**: Performance optimization through external profiling tools
- **Key Features**: Performance profiling, benchmarking, optimization strategies
- **File**: `rules.md` - Performance optimization guidelines and best practices

#### rules-mcp-management/
- **Mode**: 🏢 MCP Management (`mcp-management`)
- **Purpose**: Enterprise management and business process coordination
- **Key Features**: Project coordination, resource management, workflow automation
- **File**: `rules.md` - Business management and coordination guidelines

#### rules-mcp-tutorial/
- **Mode**: 📚 MCP Tutorial (`mcp-tutorial`)
- **Purpose**: Interactive tutorials with external learning platform integration
- **Key Features**: Adaptive learning, progress tracking, hands-on exercises
- **File**: `rules.md` - Educational content delivery and learning guidelines

## Integration with Core aiGI System

### Relationship to Existing .roo Structure
- **Complementary**: MCP modes complement existing core modes
- **Enhanced Capabilities**: Provide external service integration capabilities
- **Consistent Patterns**: Follow established .roo organizational patterns
- **Backward Compatible**: Do not interfere with existing mode functionality

### MCP Server Dependencies
- **SAFLA**: Core system management and orchestration (required for most modes)
- **Context7**: Documentation and knowledge base access (required for coding/tutorial modes)
- **Perplexity**: Real-time research and information retrieval (optional for most modes)

### Configuration Integration
- **MCP Servers**: Defined in `.roo/mcp.json`
- **Mode Specifications**: Defined in `.roo/mcp-modes/modes.json`
- **Rule Inheritance**: MCP modes inherit from core mode rules where applicable

## Usage Patterns

### Mode Activation
```bash
new_task: mcp-orchestrator
new_task: mcp-intelligent-coder
new_task: mcp-researcher
new_task: mcp-optimizer
new_task: mcp-management
new_task: mcp-tutorial
```

### Workflow Templates
The `modes.json` file defines several workflow templates:
- **research_and_implement**: Research → Code → Optimize
- **analyze_and_optimize**: Analyze → Identify Bottlenecks → Optimize
- **learn_and_practice**: Learn → Practice → Validate
- **manage_and_deploy**: Plan → Implement → Deploy

### Service Categories
- **Knowledge**: Information retrieval and documentation services
- **Development**: Code analysis and software engineering services
- **Analytics**: Performance monitoring and optimization services
- **Automation**: Workflow and process management services

## Design Principles

### Modularity
- Each mode is self-contained with its own rules and configuration
- Clear separation of concerns between different mode types
- Reusable components and patterns across modes

### Extensibility
- Easy to add new MCP-optimized modes
- Flexible configuration system for new MCP servers
- Template-based workflow definitions for common patterns

### Integration
- Seamless integration with existing aiGI workflow
- Consistent interface patterns across all modes
- Shared configuration and authentication mechanisms

### Quality Assurance
- Comprehensive documentation for each mode
- Clear guidelines and best practices
- Error handling and recovery procedures
- Security and compliance considerations

## Maintenance and Evolution

### Adding New Modes
1. Create new rules directory: `rules-mcp-{new-mode}/`
2. Add mode definition to `modes.json`
3. Create comprehensive `rules.md` file
4. Update this `STRUCTURE.md` file
5. Test integration with existing system

### Updating Existing Modes
1. Modify rules in appropriate `rules.md` file
2. Update mode configuration in `modes.json` if needed
3. Test compatibility with existing workflows
4. Update documentation as needed

### Version Control
- All changes should be tracked in version control
- Document breaking changes and migration paths
- Maintain backward compatibility where possible
- Follow semantic versioning for major changes

## Future Enhancements

### Planned Features
- Machine learning-based mode optimization
- Advanced workflow orchestration capabilities
- Enhanced security and compliance features
- Integration with emerging MCP standards

### Expansion Areas
- Additional specialized modes for specific domains
- Enhanced analytics and reporting capabilities
- Improved user experience and interface design
- Advanced collaboration and team features

This structure provides a solid foundation for MCP-optimized modes while maintaining consistency with the existing aiGI system architecture.