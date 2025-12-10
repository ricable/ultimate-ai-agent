# Phase 2: Design Prompt

## Objective
Design the unified architecture that integrates all components into a cohesive AI agent platform.

---

## Prompt Template

```
You are a senior software architect designing a unified AI agent development platform.

## Context
Based on the discovery phase, we have identified:
- 35+ projects spanning agent orchestration, vector memory, and self-learning
- Key packages: claude-flow, agentic-flow, ruvector, agentdb, flow-nexus
- Technologies: TypeScript, Python, Rust, MCP Protocol

## Your Task
Design a unified architecture following SPARC methodology:

### 1. System Architecture
Design a layered architecture that:
- Separates concerns (orchestration, memory, learning, infrastructure)
- Enables loose coupling between components
- Supports horizontal scaling
- Provides clear integration points

### 2. API Design
Define the unified API surface:
- RESTful endpoints for management operations
- WebSocket connections for real-time agent communication
- MCP protocol integration for tool access
- GraphQL for flexible querying (optional)

### 3. Data Architecture
Design the data layer:
- Vector storage schema (ruvector)
- Relational schema (agent metadata, configurations)
- Event sourcing for agent actions
- Caching strategy

### 4. Integration Patterns
Define how components interact:
- Event-driven communication
- Dependency injection
- Plugin architecture for extensions
- Middleware pipeline for cross-cutting concerns

## Output Format

1. **Architecture Diagram** (ASCII or description)
2. **Component Specifications** (for each major component)
3. **API Contracts** (OpenAPI-style definitions)
4. **Data Models** (TypeScript interfaces)
5. **Integration Flows** (sequence descriptions)

## Design Principles
- Prefer composition over inheritance
- Design for testability
- Minimize coupling, maximize cohesion
- Follow 12-factor app principles
- Security by design
```

---

## Design Templates

### Component Specification Template

```markdown
## Component: [Name]

### Purpose
[One sentence description]

### Responsibilities
- [Responsibility 1]
- [Responsibility 2]

### Dependencies
- [Package/Service 1]
- [Package/Service 2]

### API
\`\`\`typescript
interface [Name]API {
  method1(params: Params): Promise<Result>;
  method2(params: Params): Promise<Result>;
}
\`\`\`

### Configuration
\`\`\`typescript
interface [Name]Config {
  option1: string;
  option2: number;
}
\`\`\`

### Events Emitted
- `[name]:event1` - When [condition]
- `[name]:event2` - When [condition]

### Events Consumed
- `other:event1` - To [action]
```

### API Contract Template

```yaml
openapi: 3.0.0
info:
  title: [Component] API
  version: 1.0.0

paths:
  /resource:
    get:
      summary: Get resource
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Resource'

components:
  schemas:
    Resource:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
```

## Expected Outputs

After running this prompt, you should have:
- [ ] High-level architecture diagram
- [ ] Detailed component specifications
- [ ] API contracts for all public interfaces
- [ ] Data models and schemas
- [ ] Integration flow documentation

## Next Phase
Once design is complete, proceed to [03-implementation.md](03-implementation.md) for iterative development.
