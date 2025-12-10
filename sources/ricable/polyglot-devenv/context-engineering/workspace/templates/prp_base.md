name: "Polyglot PRP Template v3 - Context-Rich with Multi-Environment Validation"
description: |

## Purpose
Template optimized for AI agents to implement features across polyglot development environments with sufficient context and self-validation capabilities to achieve working code through iterative refinement.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the polyglot codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Environment Awareness**: Consider devbox, cross-language integration
6. **Global rules**: Follow all rules in CLAUDE.md and environment-specific conventions

---

## Goal
[What needs to be built - be specific about the end state and desires]

## Why
- [Business value and user impact]
- [Integration with existing features]
- [Problems this solves and for whom]

## What
[User-visible behavior and technical requirements]

### Success Criteria
- [ ] [Specific measurable outcomes]

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: [Official API docs URL]
  why: [Specific sections/methods you'll need]
  
- file: [path/to/example.py]
  why: [Pattern to follow, gotchas to avoid]
  
- doc: [Library documentation URL] 
  section: [Specific section about common pitfalls]
  critical: [Key insight that prevents common errors]

- docfile: [PRPs/ai_docs/file.md]
  why: [docs that the user has pasted in to the project]

```

### Target Environment(s)
```yaml
Environment: [python-env|typescript-env|rust-env|go-env|nushell-env|multi]
Devbox_Config: [path/to/devbox.json or multiple if multi-environment]
Dependencies: [list of packages and tools required]
Integration_Points: [list of environments this feature interacts with]
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
# Include relevant environment directories and files
python-env/
├── devbox.json
├── src/
├── tests/
└── pyproject.toml

typescript-env/
├── devbox.json
├── src/
├── tests/
└── package.json

# ... other environments as relevant
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
# Show exactly where new files will be created
[target-env]/
├── src/
│   └── new_feature.py  # Main feature implementation
├── tests/
│   └── test_new_feature.py  # Comprehensive test suite
└── [other new files and their purposes]
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Polyglot environment-specific gotchas
# Devbox: All commands must be run within devbox shell context
# Environment switching: Use 'cd [env] && devbox shell' pattern
# Cross-env: Use nushell scripts for cross-environment automation

# Environment-specific quirks:
# Python: Use uv exclusively, not pip/poetry/pipenv
# TypeScript: Node.js 20 strict mode, ESLint + Prettier required
# Rust: async with tokio, use clippy for linting
# Go: golangci-lint required, use context.Context for timeouts
# Nushell: Structured data preferred, type hints mandatory
```

## Implementation Blueprint

### Environment Setup
```bash
# Commands to set up the development environment
cd [target-env] && devbox shell

# Verify environment is ready
devbox run --quiet health-check 2>/dev/null || echo "Environment ready"

# Install/update dependencies if needed
devbox run install  # or build for compiled languages
```

### Data models and structure
Create the core data models ensuring type safety and consistency per language:

**Python:**
```python
# pydantic models, SQLAlchemy ORM, type hints
from pydantic import BaseModel
from typing import Optional

class FeatureModel(BaseModel):
    id: Optional[int] = None
    name: str
```

**TypeScript:**
```typescript
// interfaces, types, zod schemas
interface FeatureModel {
  id?: number;
  name: string;
}
```

**Rust:**
```rust
// structs with derives, serde integration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureModel {
    id: Option<u32>,
    name: String,
}
```

**Go:**
```go
// structs with json tags, validation
type FeatureModel struct {
    ID   *int   `json:"id,omitempty"`
    Name string `json:"name" validate:"required"`
}
```

**Nushell:**
```nushell
# record types with validation
def create-feature-record [name: string, id?: int] {
    {name: $name, id: $id}
}
```

### list of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Environment Setup
  COMMAND: cd [target-env] && devbox shell
  VERIFY: devbox run --version
  INSTALL: devbox run install

Task 2: Data Models
  CREATE: [target-env]/src/models.[ext]
  PATTERN: Follow existing model patterns in src/
  VALIDATE: Run type checking

Task 3: Core Implementation
  CREATE: [target-env]/src/[feature].[ext]
  MIRROR: Pattern from similar feature
  PRESERVE: Existing error handling patterns

Task 4: Testing
  CREATE: [target-env]/tests/test_[feature].[ext]
  PATTERN: Follow existing test structure
  COVERAGE: Ensure 80%+ test coverage

Task 5: Integration
  MODIFY: Integration points as specified
  VALIDATE: Cross-environment if applicable

Task 6: Documentation
  UPDATE: README.md if needed
  DOCSTRING: Add comprehensive documentation
```


### Per task pseudocode as needed added to each task
```python

# Task 1
# Pseudocode with CRITICAL details dont write entire code
async def new_feature(param: str) -> Result:
    # PATTERN: Always validate input first (see src/validators.py)
    validated = validate_input(param)  # raises ValidationError
    
    # GOTCHA: This library requires connection pooling
    async with get_connection() as conn:  # see src/db/pool.py
        # PATTERN: Use existing retry decorator
        @retry(attempts=3, backoff=exponential)
        async def _inner():
            # CRITICAL: API returns 429 if >10 req/sec
            await rate_limiter.acquire()
            return await external_api.call(validated)
        
        result = await _inner()
    
    # PATTERN: Standardized response format
    return format_response(result)  # see src/utils/responses.py
```

### Integration Points
```yaml
DATABASE:
  - migration: "Add column 'feature_enabled' to users table"
  - index: "CREATE INDEX idx_feature_lookup ON users(feature_id)"
  
CONFIG:
  - add to: config/settings.py
  - pattern: "FEATURE_TIMEOUT = int(os.getenv('FEATURE_TIMEOUT', '30'))"
  
ROUTES:
  - add to: src/api/routes.py  
  - pattern: "router.include_router(feature_router, prefix='/feature')"
```

## Validation Loop

### Level 1: Environment-Specific Syntax & Style
**Python:**
```bash
cd python-env && devbox shell
devbox run format  # ruff format
devbox run lint    # ruff check && mypy
```

**TypeScript:**
```bash
cd typescript-env && devbox shell
devbox run format  # prettier
devbox run lint    # eslint
```

**Rust:**
```bash
cd rust-env && devbox shell
devbox run format  # rustfmt
devbox run lint    # clippy
```

**Go:**
```bash
cd go-env && devbox shell
devbox run format  # goimports
devbox run lint    # golangci-lint
```

**Nushell:**
```bash
cd nushell-env && devbox shell
devbox run format  # nu fmt
devbox run check   # syntax validation
```

### Level 2: Unit Tests using existing test patterns
```python
# Language-agnostic test structure:
# - test_happy_path: Basic functionality works
# - test_validation_error: Invalid input handling
# - test_edge_cases: Boundary conditions
# - test_error_handling: Graceful failure modes

# Run tests with coverage:
devbox run test  # Uses environment-specific test runner
```

### Level 3: Integration & Cross-Environment Tests
```bash
# Single environment integration
cd [target-env] && devbox shell
devbox run integration-test  # If available

# Cross-environment validation (if applicable)
nu nushell-env/scripts/validate-all.nu parallel

# Security and performance checks
nu nushell-env/scripts/security-scanner.nu scan-all --quiet
nu nushell-env/scripts/performance-analytics.nu measure "feature-validation"
```

## Final validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] All tests pass: `devbox run test`
- [ ] No linting errors: `devbox run lint`
- [ ] No format issues: `devbox run format`
- [ ] Manual test successful: [environment-specific command]
- [ ] Cross-environment integration (if applicable)
- [ ] Security scan clean
- [ ] Performance within limits
- [ ] Error cases handled gracefully
- [ ] Documentation updated if needed

---

## Anti-Patterns to Avoid
- ❌ Don't create new patterns when existing ones work
- ❌ Don't skip validation because "it should work"  
- ❌ Don't ignore failing tests - fix them
- ❌ Don't bypass devbox environment activation
- ❌ Don't hardcode values that should be config
- ❌ Don't ignore environment-specific conventions
- ❌ Don't skip cross-environment testing when applicable
- ❌ Don't commit without running validation gates
- ❌ Don't use language-inappropriate patterns (e.g., sync in async contexts)

## Polyglot-Specific Guidelines
- ✅ Always activate devbox environment before running commands
- ✅ Use environment-specific package managers (uv for Python, npm for TypeScript, etc.)
- ✅ Follow language conventions for naming, structure, and testing
- ✅ Leverage existing intelligence scripts for monitoring and validation
- ✅ Consider cross-environment integration points
- ✅ Use nushell scripts for cross-environment automation
- ✅ Validate environment consistency with drift detection