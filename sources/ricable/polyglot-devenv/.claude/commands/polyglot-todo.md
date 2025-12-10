# /polyglot-todo

Enhanced todo management with language awareness, cross-environment task tracking, and integration with existing TodoWrite/TodoRead systems.

## Usage
```
/polyglot-todo [action] [--env <environment>] [--priority <level>] [--integrate]
```

## Actions
- `add <task>` - Add new task with language context
- `list` - Show all tasks with environment mapping
- `complete <id>` - Mark task as completed
- `update <id> <status>` - Update task status
- `sync` - Synchronize with existing TodoWrite system
- `report` - Generate cross-environment task report

## Features
- **Language-aware tasks** with environment context detection
- **Cross-environment tracking** for polyglot feature development
- **Priority intelligence** based on dependencies and impact
- **Integration workflows** with existing automation systems
- **Performance tracking** for task completion analytics  
- **GitHub integration** for issue and PR correlation
- **Dependency mapping** between tasks across environments

## Task Categories by Environment

### Python Tasks (`python-env/`)
- Package updates and dependency management
- FastAPI endpoint development
- Database model changes
- Test coverage improvements
- Performance optimization
- Security vulnerability fixes

### TypeScript Tasks (`typescript-env/`)
- Component development and updates
- API client implementation
- Build configuration changes
- Test suite expansion
- Type definition updates
- Bundle optimization

### Rust Tasks (`rust-env/`)
- Service implementation
- Performance-critical modules
- Memory safety improvements
- Concurrency implementations
- Integration testing
- Cargo workspace management

### Go Tasks (`go-env/`)
- Microservice development
- API gateway implementation
- Database connection management
- Logging and monitoring
- Docker containerization
- Module dependency updates

### Nushell Tasks (`nushell-env/`)
- Automation script development
- Intelligence monitoring enhancements
- DevOps workflow improvements
- Configuration management
- Cross-environment orchestration
- Performance analytics expansion

## Smart Task Management

### Context-Aware Task Creation
- **File-based context**: Tasks inherit environment from current working directory
- **Language detection**: Automatically tag tasks with relevant language/environment
- **Dependency inference**: Detect cross-environment dependencies
- **Priority assignment**: Intelligent priority based on impact and urgency

### Cross-Environment Workflows
- **Feature development**: Track tasks across multiple environments for polyglot features
- **Migration tasks**: Coordinate updates across language boundaries
- **Integration testing**: Ensure comprehensive testing across environments
- **Deployment coordination**: Synchronize deployment-related tasks

## Instructions
1. **Environment Detection**:
   - Analyze current working directory and file context
   - Map tasks to appropriate environments
   - Identify cross-environment dependencies
   - Set intelligent priorities based on impact

2. **Task Creation with Context**:
   ```bash
   # Single environment task
   /polyglot-todo add "Implement user authentication endpoint" --env python-env --priority high
   
   # Cross-environment feature
   /polyglot-todo add "Add real-time notifications" --env python-env,typescript-env --priority medium
   
   # Infrastructure task
   /polyglot-todo add "Update CI/CD pipeline" --env nushell-env --priority low
   ```

3. **Integration with Existing System**:
   - Sync with TodoWrite/TodoRead for consistency
   - Maintain existing task IDs and structure
   - Enhance with environment and language context
   - Preserve priority and status information

4. **Intelligent Task Management**:
   - Detect task dependencies across environments
   - Suggest optimal task ordering for efficiency
   - Identify blocking tasks and critical path
   - Provide completion time estimates

5. **Progress Tracking**:
   - Monitor task completion across environments
   - Track performance metrics for task types
   - Generate productivity reports
   - Identify bottlenecks and improvement opportunities

6. **GitHub Integration**:
   - Link tasks to GitHub issues and PRs
   - Create issues for high-priority tasks
   - Track commit progress against tasks
   - Generate release notes from completed tasks

## Enhanced Task Structure
```json
{
  "id": "polyglot-001",
  "content": "Implement user authentication system",
  "status": "in_progress",
  "priority": "high",
  "environments": ["python-env", "typescript-env"],
  "languages": ["python", "typescript"],
  "dependencies": ["polyglot-002", "polyglot-003"],
  "created": "2025-01-15T10:00:00Z",
  "updated": "2025-01-15T14:30:00Z",
  "estimated_hours": 8,
  "actual_hours": 6.5,
  "github_issue": "#123",
  "commits": ["abc123", "def456"],
  "tags": ["feature", "security", "api"],
  "context": {
    "files": ["python-env/src/auth.py", "typescript-env/src/auth.ts"],
    "related_tasks": ["polyglot-004", "polyglot-005"],
    "performance_impact": "medium",
    "security_implications": "high"
  }
}
```

## Task Report Generation
```
🎯 POLYGLOT TODO REPORT

📊 SUMMARY
✅ Completed: 12 tasks
🔄 In Progress: 5 tasks  
📋 Pending: 18 tasks
⚡ Total: 35 tasks across 5 environments

📈 PROGRESS BY ENVIRONMENT
🐍 Python: 8/12 tasks (67% complete)
📘 TypeScript: 6/10 tasks (60% complete)
🦀 Rust: 4/5 tasks (80% complete)  
🐹 Go: 3/4 tasks (75% complete)
🐚 Nushell: 2/4 tasks (50% complete)

🔥 HIGH PRIORITY TASKS
1. [python-env] Fix authentication security vulnerability (#123)
2. [polyglot] Implement real-time notification system (#124)
3. [typescript-env] Optimize bundle size for production (#125)

🔄 IN PROGRESS TASKS
1. [rust-env] Implement payment processing service (6h remaining)
2. [go-env] Add database connection pooling (2h remaining)
3. [nushell-env] Enhance performance monitoring (4h remaining)

📊 PRODUCTIVITY METRICS
• Average completion time: 4.2 hours/task
• Cross-environment tasks: 40% longer completion time
• Python tasks: Fastest average completion (3.1h)
• Security tasks: Require 60% more time than estimated

🎯 OPTIMIZATION OPPORTUNITIES
• Batch similar tasks within environments for efficiency
• Address blocking dependencies in Go environment
• Consider parallel development for cross-environment features
• Prioritize security tasks for earlier completion

📋 UPCOMING DEPENDENCIES
• Task #456 (TypeScript) blocked by #123 (Python)
• Task #789 (Go) waiting for #456 (TypeScript) completion
• Infrastructure tasks (#234, #567) can be parallelized
```

## Integration Benefits
- **Seamless workflow**: Works with existing TodoWrite/TodoRead system
- **Environment awareness**: Context-sensitive task management
- **Cross-language coordination**: Manage polyglot feature development
- **Performance tracking**: Monitor and optimize task completion
- **GitHub synchronization**: Automatic issue and PR integration
- **Intelligence integration**: Leverage existing monitoring and analytics

## Advanced Features
- **Dependency visualization**: Show task dependency graphs
- **Automated scheduling**: Suggest optimal task ordering
- **Performance prediction**: Estimate completion times based on history
- **Risk assessment**: Identify high-risk tasks requiring careful attention
- **Resource allocation**: Balance workload across environments
- **Template tasks**: Create reusable task templates for common activities

## Error Handling
- Graceful degradation if TodoWrite/TodoRead unavailable
- Validation of environment names and dependencies
- Clear error messages for invalid task operations
- Backup and recovery for task data integrity
- Conflict resolution for concurrent task updates