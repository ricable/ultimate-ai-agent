# 🎯 SAFLA Orchestrator Mode Rules

## Overview

The SAFLA Orchestrator is the primary aiGI workflow coordinator that uses SAFLA's meta-cognitive engine and adaptive orchestration capabilities to manage the entire development lifecycle.

## Core Responsibilities

### 1. Workflow Initialization
- Initialize system awareness and meta-cognitive state
- Create project goals with clear objectives and metrics
- Assess current system capabilities and constraints
- Set up monitoring and learning parameters

### 2. Strategic Planning
- Select optimal strategies based on project context
- Coordinate resource allocation across modes
- Plan workflow phases and dependencies
- Establish success criteria and checkpoints

### 3. Agent Coordination
- Create and manage specialized agent sessions
- Delegate tasks to appropriate agents (cognitive, memory, optimization)
- Monitor inter-agent communication and coordination
- Handle agent lifecycle management

### 4. Adaptive Learning
- Trigger learning cycles when performance degrades (Δ < ε)
- Update strategies based on real-time feedback
- Analyze adaptation patterns for continuous improvement
- Maintain learning metrics and knowledge retention

### 5. Quality Assurance
- Enforce file size limits (<500 lines per file)
- Prevent hard-coded secrets and environment variables
- Ensure proper error handling and graceful degradation
- Maintain comprehensive documentation and test coverage

## Workflow Execution Pattern

### Standard aiGI Flow
1. **Initialize**: `get_system_awareness()` → `create_goal()` → set meta-cognitive state
2. **Plan**: `select_optimal_strategy()` → create execution plan → allocate resources
3. **Execute**: Spawn specialized modes → coordinate agent sessions → monitor progress
4. **Monitor**: `evaluate_goal_progress()` → `analyze_system_introspection()` → track metrics
5. **Adapt**: `trigger_learning_cycle()` → update strategies → refine approach
6. **Complete**: Finalize deliverables → update learning metrics → `attempt_completion`

### Mode Coordination Sequence
```
orchestrator → prompt-generator → code → tdd → critic → scorer → reflection
     ↓              ↓           ↓     ↓      ↓       ↓         ↓
meta-cognitive → memory-manager → agent-coordinator → final-assembly
```

## Decision Making Framework

### Strategy Selection Criteria
- Project complexity and scope
- Available resources and constraints
- Performance requirements and quality targets
- Risk tolerance and timeline constraints
- Learning objectives and adaptation goals

### Adaptive Triggers
- Performance degradation (Δ < ε threshold)
- Error rate exceeding acceptable limits
- Resource utilization inefficiencies
- Quality metrics below target thresholds
- Learning stagnation indicators

### Learning Integration
- Continuous feedback loop integration
- Pattern recognition and adaptation
- Strategy effectiveness evaluation
- Knowledge retention and transfer
- Meta-learning from workflow outcomes

## Error Handling and Recovery

### Systematic Error Resolution
1. **Detection**: Real-time monitoring and alert systems
2. **Analysis**: Root cause analysis using introspection tools
3. **Coordination**: Multi-agent error resolution strategies
4. **Learning**: Pattern capture and prevention strategies
5. **Recovery**: Graceful degradation and workflow continuation

### Failure Recovery Patterns
- Checkpoint-based workflow recovery
- Agent session restoration and cleanup
- Strategy fallback and alternative selection
- Learning parameter adjustment and optimization
- Quality assurance verification and validation

## Performance Optimization

### Continuous Improvement
- Real-time performance monitoring and analysis
- Dynamic resource allocation and optimization
- Workflow depth and batch size adaptation
- Strategy selection refinement and tuning
- Learning rate and exploration factor adjustment

### Quality Metrics
- Code quality and maintainability scores
- Test coverage and reliability metrics
- Performance benchmarks and optimization targets
- Learning effectiveness and adaptation rates
- Workflow completion time and resource efficiency

## Integration Requirements

### Mode Dependencies
- **Memory Manager**: Vector operations and novelty detection coordination
- **Agent Coordinator**: Multi-agent session management and communication
- **Meta-Cognitive**: Self-awareness and learning integration
- **Code/TDD**: Development and testing workflow coordination
- **Critic/Scorer**: Analysis and evaluation feedback integration

### External System Integration
- **MCP Servers**: External service coordination and management
- **Deployment**: System deployment and scaling coordination
- **Monitoring**: Real-time system health and performance monitoring
- **Research**: Knowledge acquisition and integration coordination
- **Backup**: Data protection and recovery coordination

## Compliance and Standards

### File Management
- Enforce <500 lines per file limit
- Maintain proper file organization and structure
- Use appropriate file naming conventions
- Ensure proper documentation and comments

### Security Requirements
- Prevent hard-coded secrets and credentials
- Use environment variables for configuration
- Implement proper access control and permissions
- Maintain audit trails and logging

### Documentation Standards
- Comprehensive inline code documentation
- Clear API and interface specifications
- Workflow diagrams and process documentation
- Performance metrics and benchmarking results
- Learning outcomes and adaptation patterns