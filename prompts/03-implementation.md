# Phase 3: Implementation Prompt

## Objective
Iteratively implement features following the design specifications with AI assistance.

---

## Prompt Template

```
You are an expert developer implementing features for a unified AI agent platform.

## Context
Following the SPARC methodology, we are in the Implementation phase.
The design has been approved and we're ready to write code.

## Your Task
Implement the specified feature following these guidelines:

### 1. Implementation Strategy
- Start with interfaces and types
- Implement core logic with unit tests
- Add integration points
- Document as you code

### 2. Code Quality Requirements
- TypeScript strict mode compliance
- 80%+ test coverage for new code
- ESLint/Prettier formatting
- JSDoc comments for public APIs

### 3. Testing Requirements
- Unit tests for all functions
- Integration tests for API endpoints
- E2E tests for critical workflows
- Performance benchmarks for hot paths

### 4. Documentation Requirements
- Update API documentation
- Add usage examples
- Update CHANGELOG.md
- Create/update ADRs if needed

## Feature Specification
[Describe the feature to implement]

## Acceptance Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

## Output Format
1. **Implementation Plan** (ordered steps)
2. **Code Changes** (with explanations)
3. **Test Cases** (with coverage report)
4. **Documentation Updates**

## Constraints
- Follow existing patterns in the codebase
- Maintain backward compatibility
- No breaking changes without deprecation
- Security review for user input handling
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Review design specification
- [ ] Identify affected packages
- [ ] Plan branch strategy
- [ ] Set up local development environment

### During Implementation
- [ ] Create feature branch
- [ ] Write failing tests first (TDD)
- [ ] Implement minimal solution
- [ ] Refactor for clarity
- [ ] Add comprehensive tests

### Post-Implementation
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create pull request
- [ ] Request code review
- [ ] Address review feedback

## Common Implementation Patterns

### Adding a New Agent Type

```typescript
// 1. Define the agent interface
interface CustomAgent extends BaseAgent {
  customCapability(): Promise<void>;
}

// 2. Implement the agent class
class CustomAgentImpl implements CustomAgent {
  constructor(private config: CustomAgentConfig) {}

  async customCapability(): Promise<void> {
    // Implementation
  }
}

// 3. Register with the factory
AgentFactory.register('custom', CustomAgentImpl);

// 4. Add configuration schema
const CustomAgentConfigSchema = z.object({
  // Zod schema
});

// 5. Write tests
describe('CustomAgent', () => {
  it('should perform custom capability', async () => {
    const agent = new CustomAgentImpl(config);
    await expect(agent.customCapability()).resolves.toBeUndefined();
  });
});
```

### Adding a New Provider

```typescript
// 1. Implement the provider interface
class NewProvider implements LLMProvider {
  async complete(prompt: string): Promise<string> {
    // API call implementation
  }

  async stream(prompt: string): AsyncGenerator<string> {
    // Streaming implementation
  }
}

// 2. Register with agentic-flow
ProviderRegistry.register('new-provider', NewProvider);

// 3. Add configuration
const NewProviderConfig = {
  apiKey: process.env.NEW_PROVIDER_API_KEY,
  model: 'default-model',
  maxTokens: 4096
};
```

### Adding a New MCP Tool

```typescript
// 1. Define the tool schema
const ToolSchema = {
  name: 'custom_tool',
  description: 'Does something useful',
  inputSchema: {
    type: 'object',
    properties: {
      input: { type: 'string', description: 'Input value' }
    },
    required: ['input']
  }
};

// 2. Implement the handler
async function handleCustomTool(params: { input: string }): Promise<Result> {
  // Tool logic
  return { output: result };
}

// 3. Register with MCP server
mcpServer.addTool(ToolSchema, handleCustomTool);
```

## Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/TICKET-123-feature-name

# 2. Make atomic commits
git commit -m "feat(component): add new capability

- Implemented X
- Added tests for Y
- Updated documentation"

# 3. Push and create PR
git push -u origin feature/TICKET-123-feature-name
gh pr create --title "feat: Add new capability" --body "..."

# 4. After review approval
git checkout main
git pull
git merge feature/TICKET-123-feature-name
git push
```

## Next Phase
Once implementation is complete, proceed to [04-validation.md](04-validation.md) for testing and QA.
