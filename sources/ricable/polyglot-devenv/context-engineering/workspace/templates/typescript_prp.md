name: "TypeScript PRP Template - Node.js 20 with Strict Mode"
description: |

## Purpose
Template optimized for AI agents to implement TypeScript features in the typescript-env using Node.js 20, strict mode TypeScript, and modern development practices.

## Core Principles
1. **Strict TypeScript**: No any types, comprehensive type safety
2. **Modern Node.js**: ES modules, async/await patterns
3. **Quality Tools**: ESLint, Prettier, Jest for testing
4. **Type-First**: Interfaces and types defined before implementation
5. **Error Handling**: Result pattern or proper error types

---

## Goal
[What needs to be built - be specific about the TypeScript feature and Node.js integration]

## Why
- [Business value and user impact]
- [Integration with existing TypeScript services]
- [Problems this solves and for whom]

## What
[User-visible behavior and API endpoints, technical requirements]

### Success Criteria
- [ ] [Specific measurable outcomes for TypeScript implementation]
- [ ] All TypeScript strict mode checks passing
- [ ] Comprehensive test coverage with Jest (80%+)
- [ ] ESLint and Prettier formatting clean

## All Needed Context

### Target Environment
```yaml
Environment: typescript-env
Devbox_Config: typescript-env/devbox.json
Dependencies: [List required npm packages]
Node_Version: 20+ (as specified in devbox.json)
TypeScript_Version: Latest stable
Package_Manager: npm (with package-lock.json)
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://nodejs.org/docs/latest-v20.x/api/
  why: Node.js 20 API reference and patterns
  
- file: typescript-env/src/[existing_similar_module].ts
  why: Existing patterns to follow
  
- file: typescript-env/package.json
  why: Dependency management and scripts
  
- file: typescript-env/tsconfig.json
  why: TypeScript configuration and strict mode settings
  
- doc: https://typescript-eslint.io/
  section: ESLint TypeScript rules and best practices
  critical: Follow strict mode and no-any rules
```

### Current Codebase tree
```bash
typescript-env/
├── devbox.json         # Node.js 20, TypeScript, ESLint, Prettier, Jest
├── src/
│   ├── index.ts        # Main entry point
│   ├── types/          # Type definitions and interfaces
│   ├── utils/          # Utility functions
│   ├── services/       # Business logic layer
│   └── config/         # Configuration management
├── tests/
│   ├── __tests__/      # Jest test files
│   ├── fixtures/       # Test data and mocks
│   └── setup.ts        # Test configuration
├── package.json        # npm dependencies and scripts
├── tsconfig.json       # TypeScript strict configuration
├── .eslintrc.json      # ESLint rules
├── .prettierrc         # Prettier configuration
└── README.md
```

### Desired Codebase tree with files to be added
```bash
typescript-env/
├── src/
│   ├── types/
│   │   └── feature.types.ts     # TypeScript interfaces and types
│   ├── services/
│   │   └── feature.service.ts   # Business logic implementation
│   ├── utils/
│   │   └── feature.utils.ts     # Helper functions (if needed)
│   └── api/
│       └── feature.api.ts       # API layer (if needed)
├── tests/
│   └── __tests__/
│       ├── feature.service.test.ts
│       ├── feature.utils.test.ts
│       └── feature.api.test.ts
```

### Known Gotchas of TypeScript Environment
```typescript
// CRITICAL: TypeScript environment-specific gotchas
// Strict mode: No 'any' types allowed - use 'unknown' instead
// ES modules: Use import/export, not require/module.exports
// Node.js 20: Use built-in test runner or Jest for testing
// ESLint: @typescript-eslint/no-explicit-any rule enforced
// Prettier: Automatic formatting on save required
// Package.json: Use npm scripts for all commands

// Example patterns:
// ✅ const result: Result<Data, Error> = await operation();
// ✅ import { function } from './module';
// ✅ interface UserData { id: number; name: string; }
// ✅ unknown instead of any for untyped data
// ❌ const data: any = response.json();
// ❌ const util = require('./util');  // Use import instead
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate TypeScript environment
cd typescript-env && devbox shell

# Verify environment
node --version    # Should be 20+
npm --version
npx tsc --version

# Install dependencies if needed
npm install [new-package-name]
```

### Type Definitions First
```typescript
// src/types/feature.types.ts
export enum FeatureStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  PENDING = 'pending',
}

export interface FeatureBase {
  readonly id: string;
  name: string;
  description?: string;
  status: FeatureStatus;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

export interface CreateFeatureRequest {
  name: string;
  description?: string;
  status?: FeatureStatus;
}

export interface UpdateFeatureRequest {
  name?: string;
  description?: string;
  status?: FeatureStatus;
}

export interface FeatureFilters {
  status?: FeatureStatus;
  nameContains?: string;
  limit?: number;
  offset?: number;
}

// Result pattern for error handling
export type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

// Service interface
export interface IFeatureService {
  create(request: CreateFeatureRequest): Promise<Result<FeatureBase>>;
  findById(id: string): Promise<Result<FeatureBase>>;
  findMany(filters: FeatureFilters): Promise<Result<FeatureBase[]>>;
  update(id: string, request: UpdateFeatureRequest): Promise<Result<FeatureBase>>;
  delete(id: string): Promise<Result<void>>;
}
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup
  COMMAND: cd typescript-env && devbox shell
  VERIFY: node --version && npm --version
  INSTALL: npm install [required-packages]

Task 2: Type Definitions
  CREATE: typescript-env/src/types/feature.types.ts
  PATTERN: Follow existing type patterns in src/types/
  VALIDATE: npx tsc --noEmit (type checking only)

Task 3: Service Implementation
  CREATE: typescript-env/src/services/feature.service.ts
  PATTERN: Follow dependency injection and Result patterns
  ASYNC: All service methods return Promise<Result<T>>
  ERRORS: Use specific error types, no throwing

Task 4: Utility Functions
  CREATE: typescript-env/src/utils/feature.utils.ts (if needed)
  PATTERN: Pure functions with comprehensive types
  VALIDATION: Input validation and sanitization

Task 5: API Layer (if needed)
  CREATE: typescript-env/src/api/feature.api.ts
  PATTERN: RESTful patterns with proper error handling
  TYPES: Use defined interfaces for all I/O

Task 6: Comprehensive Testing
  CREATE: Jest tests for all modules
  PATTERN: Follow AAA pattern (Arrange, Act, Assert)
  COVERAGE: Ensure 80%+ test coverage
  MOCKING: Use Jest mocks for external dependencies

Task 7: Integration and Documentation
  MODIFY: typescript-env/src/index.ts for exports
  VALIDATE: Full TypeScript compilation successful
  DOCS: Update README.md with new functionality
```

### Per task pseudocode

**Task 3: Service Implementation**
```typescript
// src/services/feature.service.ts
import { 
  IFeatureService, 
  FeatureBase, 
  CreateFeatureRequest, 
  UpdateFeatureRequest, 
  FeatureFilters, 
  Result,
  FeatureStatus 
} from '../types/feature.types';
import { generateId } from '../utils/id.utils';
import { validateFeatureName } from '../utils/validation.utils';

export class FeatureService implements IFeatureService {
  private readonly features = new Map<string, FeatureBase>();

  async create(request: CreateFeatureRequest): Promise<Result<FeatureBase>> {
    try {
      // Validate input
      const nameValidation = validateFeatureName(request.name);
      if (!nameValidation.isValid) {
        return { 
          success: false, 
          error: new Error(`Invalid name: ${nameValidation.error}`) 
        };
      }

      // Create feature
      const feature: FeatureBase = {
        id: generateId(),
        name: request.name.trim(),
        description: request.description?.trim(),
        status: request.status ?? FeatureStatus.PENDING,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      this.features.set(feature.id, feature);

      return { success: true, data: feature };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error : new Error('Unknown error') 
      };
    }
  }

  async findById(id: string): Promise<Result<FeatureBase>> {
    const feature = this.features.get(id);
    
    if (!feature) {
      return { 
        success: false, 
        error: new Error(`Feature with id ${id} not found`) 
      };
    }

    return { success: true, data: feature };
  }

  async findMany(filters: FeatureFilters): Promise<Result<FeatureBase[]>> {
    try {
      let results = Array.from(this.features.values());

      // Apply filters
      if (filters.status) {
        results = results.filter(f => f.status === filters.status);
      }

      if (filters.nameContains) {
        const searchTerm = filters.nameContains.toLowerCase();
        results = results.filter(f => 
          f.name.toLowerCase().includes(searchTerm)
        );
      }

      // Apply pagination
      const offset = filters.offset ?? 0;
      const limit = filters.limit ?? 100;
      results = results.slice(offset, offset + limit);

      return { success: true, data: results };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error : new Error('Search failed') 
      };
    }
  }
}
```

**Task 6: Comprehensive Testing**
```typescript
// tests/__tests__/feature.service.test.ts
import { FeatureService } from '../../src/services/feature.service';
import { FeatureStatus, CreateFeatureRequest } from '../../src/types/feature.types';

describe('FeatureService', () => {
  let service: FeatureService;

  beforeEach(() => {
    service = new FeatureService();
  });

  describe('create', () => {
    it('should create a feature successfully', async () => {
      // Arrange
      const request: CreateFeatureRequest = {
        name: 'Test Feature',
        description: 'Test description',
        status: FeatureStatus.ACTIVE,
      };

      // Act
      const result = await service.create(request);

      // Assert
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.name).toBe('Test Feature');
        expect(result.data.status).toBe(FeatureStatus.ACTIVE);
        expect(result.data.id).toBeDefined();
        expect(result.data.createdAt).toBeInstanceOf(Date);
      }
    });

    it('should handle invalid name', async () => {
      // Arrange
      const request: CreateFeatureRequest = {
        name: '', // Invalid empty name
      };

      // Act
      const result = await service.create(request);

      // Assert
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.message).toContain('Invalid name');
      }
    });
  });

  describe('findById', () => {
    it('should find existing feature', async () => {
      // Arrange
      const createResult = await service.create({ name: 'Test Feature' });
      expect(createResult.success).toBe(true);
      
      if (!createResult.success) return;
      const featureId = createResult.data.id;

      // Act
      const result = await service.findById(featureId);

      // Assert
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.id).toBe(featureId);
        expect(result.data.name).toBe('Test Feature');
      }
    });

    it('should return error for non-existent feature', async () => {
      // Act
      const result = await service.findById('non-existent-id');

      // Assert
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.message).toContain('not found');
      }
    });
  });
});
```

### Integration Points
```yaml
MAIN_MODULE:
  - modify: typescript-env/src/index.ts
  - pattern: "export { FeatureService } from './services/feature.service';"
  
DEPENDENCIES:
  - add to: typescript-env/package.json
  - pattern: Use npm install for new dependencies
  
CONFIG:
  - add to: typescript-env/src/config/
  - pattern: Environment variable configuration with types
  
TYPES:
  - export from: typescript-env/src/types/index.ts
  - pattern: Central type exports for consumers
```

## Validation Loop

### Level 1: TypeScript Syntax & Style
```bash
cd typescript-env && devbox shell

# Format code with Prettier
npm run format  # or npx prettier --write src/ tests/

# Lint with ESLint
npm run lint    # or npx eslint src/ tests/ --fix

# Type checking with TypeScript
npm run typecheck  # or npx tsc --noEmit

# Expected: No errors. If errors, READ and fix before proceeding.
```

### Level 2: Unit Tests with Jest
```bash
# Run all tests
npm test        # or npx jest

# Run tests with coverage
npm run test:coverage  # or npx jest --coverage

# Run tests in watch mode during development
npm run test:watch     # or npx jest --watch

# Expected: All tests pass, 80%+ coverage
```

### Level 3: Build and Integration Tests
```bash
# Build the project
npm run build   # or npx tsc

# Run built JavaScript (if applicable)
node dist/index.js

# Integration test examples
npm run test:integration  # Custom integration tests

# Expected: Clean build, integration tests pass
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] Dependencies installed: `npm install` successful
- [ ] Code formatted: `npm run format` clean
- [ ] Linting passed: `npm run lint` clean
- [ ] Type checking passed: `npm run typecheck` clean
- [ ] All tests pass: `npm test`
- [ ] Test coverage 80%+: `npm run test:coverage`
- [ ] Build successful: `npm run build`
- [ ] No any types used: ESLint @typescript-eslint/no-explicit-any passes
- [ ] Error cases handled gracefully with Result pattern
- [ ] All exports properly typed
- [ ] Documentation updated if needed

---

## TypeScript-Specific Anti-Patterns to Avoid
- ❌ Don't use `any` type - use `unknown` or proper types
- ❌ Don't use `require()` - use ES module `import`
- ❌ Don't ignore TypeScript errors - fix them
- ❌ Don't skip interface definitions - define types first
- ❌ Don't use `function` declarations - use `const` with arrow functions for consistency
- ❌ Don't throw errors directly - use Result pattern
- ❌ Don't skip JSDoc comments for public APIs
- ❌ Don't use non-null assertion (`!`) without good reason

## TypeScript Best Practices
- ✅ Define interfaces and types before implementation
- ✅ Use strict TypeScript configuration
- ✅ Leverage union types and literal types for better type safety
- ✅ Use Result pattern for error handling instead of throwing
- ✅ Write comprehensive JSDoc comments for public APIs
- ✅ Use readonly properties where appropriate
- ✅ Prefer immutable data patterns
- ✅ Use proper generic constraints for reusable functions
- ✅ Structure code with clear separation of concerns