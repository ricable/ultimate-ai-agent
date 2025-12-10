name: "Go PRP Template - Modern Go with Context and Error Handling"
description: |

## Purpose
Template optimized for AI agents to implement Go features in the go-env using modern Go patterns, context for cancellation, and comprehensive error handling.

## Core Principles
1. **Simplicity**: Embrace Go's philosophy of simple, readable code
2. **Error Handling**: Explicit error checking and proper error types
3. **Context Usage**: Use context.Context for cancellation and timeouts
4. **Interface Design**: Small, focused interfaces
5. **Testing**: Table-driven tests and comprehensive coverage

---

## Goal
[What needs to be built - be specific about the Go feature and concurrent requirements]

## Why
- [Business value and performance requirements]
- [Integration with existing Go services]
- [Problems this solves and for whom]

## What
[User-visible behavior and system requirements, concurrent patterns needed]

### Success Criteria
- [ ] [Specific measurable outcomes for Go implementation]
- [ ] All golangci-lint checks passing
- [ ] Comprehensive test coverage with go test (80%+)
- [ ] Proper context usage for cancellation/timeouts

## All Needed Context

### Target Environment
```yaml
Environment: go-env
Devbox_Config: go-env/devbox.json
Dependencies: [List required modules in go.mod]
Go_Version: 1.22+ (as specified in devbox.json)
Modules: Use Go modules for dependency management
Linting: golangci-lint with comprehensive rules
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://golang.org/doc/effective_go.html
  why: Go idioms and best practices
  
- file: go-env/[existing_similar_module].go
  why: Existing patterns to follow
  
- file: go-env/go.mod
  why: Module dependencies and Go version
  
- doc: https://pkg.go.dev/context
  section: Context usage patterns for cancellation
  critical: Always accept context as first parameter
  
- doc: https://golang.org/doc/tutorial/
  section: Error handling and interface design
```

### Current Codebase tree
```bash
go-env/
├── go.mod              # Module definition and dependencies
├── go.sum              # Dependency checksums
├── cmd/
│   └── main.go         # Main application entry point
├── internal/
│   ├── config/         # Configuration management
│   ├── handlers/       # HTTP handlers or CLI commands
│   ├── services/       # Business logic layer
│   └── types/          # Type definitions
├── pkg/                # Public API packages
├── tests/
│   ├── integration/    # Integration tests
│   └── fixtures/       # Test data
├── scripts/            # Build and deployment scripts
└── README.md
```

### Desired Codebase tree with files to be added
```bash
go-env/
├── internal/
│   ├── types/
│   │   └── feature.go       # Feature-related types and structs
│   ├── services/
│   │   └── feature.go       # Business logic implementation
│   ├── repository/
│   │   └── feature.go       # Data access layer (if needed)
│   └── handlers/
│       └── feature.go       # HTTP handlers (if needed)
├── pkg/
│   └── feature/             # Public API (if needed)
│       └── client.go
├── tests/
│   ├── feature_test.go      # Unit tests
│   └── integration/
│       └── feature_integration_test.go
```

### Known Gotchas of Go Environment
```go
// CRITICAL: Go environment-specific gotchas
// Context: Always pass context.Context as first parameter
// Errors: Don't ignore errors - handle them explicitly
// Interfaces: Keep interfaces small and focused
// Goroutines: Always consider how to stop them gracefully
// Testing: Use table-driven tests for multiple test cases
// Imports: Use goimports for automatic import management

// Example patterns:
// ✅ func ProcessData(ctx context.Context, data Data) error
// ✅ if err != nil { return fmt.Errorf("processing failed: %w", err) }
// ✅ type Repository interface { Get(ctx context.Context, id string) (*Item, error) }
// ❌ func ProcessData(data Data) // Missing context
// ❌ err := operation(); _ = err // Ignoring errors
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate Go environment
cd go-env && devbox shell

# Verify environment
go version  # Should be 1.22+
go mod tidy

# Install dependencies if needed
go get [new-module-name]
```

### Type Definitions
```go
// internal/types/feature.go
package types

import (
    "time"
    "errors"
)

// FeatureStatus represents the current state of a feature
type FeatureStatus string

const (
    FeatureStatusActive   FeatureStatus = "active"
    FeatureStatusInactive FeatureStatus = "inactive"
    FeatureStatusPending  FeatureStatus = "pending"
)

// Validate ensures the status is valid
func (fs FeatureStatus) Validate() error {
    switch fs {
    case FeatureStatusActive, FeatureStatusInactive, FeatureStatusPending:
        return nil
    default:
        return ErrInvalidFeatureStatus
    }
}

// Feature represents a feature in the system
type Feature struct {
    ID          string        `json:"id" validate:"required"`
    Name        string        `json:"name" validate:"required,min=1,max=255"`
    Description *string       `json:"description,omitempty" validate:"omitempty,max=1000"`
    Status      FeatureStatus `json:"status" validate:"required"`
    CreatedAt   time.Time     `json:"created_at"`
    UpdatedAt   time.Time     `json:"updated_at"`
}

// CreateFeatureRequest represents a request to create a new feature
type CreateFeatureRequest struct {
    Name        string         `json:"name" validate:"required,min=1,max=255"`
    Description *string        `json:"description,omitempty" validate:"omitempty,max=1000"`
    Status      *FeatureStatus `json:"status,omitempty"`
}

// UpdateFeatureRequest represents a request to update an existing feature
type UpdateFeatureRequest struct {
    Name        *string        `json:"name,omitempty" validate:"omitempty,min=1,max=255"`
    Description *string        `json:"description,omitempty" validate:"omitempty,max=1000"`
    Status      *FeatureStatus `json:"status,omitempty"`
}

// FeatureFilters represents filters for querying features
type FeatureFilters struct {
    Status       *FeatureStatus `json:"status,omitempty"`
    NameContains *string        `json:"name_contains,omitempty"`
    Limit        *int           `json:"limit,omitempty" validate:"omitempty,min=1,max=1000"`
    Offset       *int           `json:"offset,omitempty" validate:"omitempty,min=0"`
}

// Custom errors
var (
    ErrFeatureNotFound        = errors.New("feature not found")
    ErrInvalidFeatureStatus   = errors.New("invalid feature status")
    ErrFeatureNameEmpty       = errors.New("feature name cannot be empty")
    ErrFeatureAlreadyExists   = errors.New("feature already exists")
)

// FeatureError wraps errors with additional context
type FeatureError struct {
    Op  string // Operation that failed
    Err error  // Underlying error
}

func (e *FeatureError) Error() string {
    return fmt.Sprintf("feature %s: %v", e.Op, e.Err)
}

func (e *FeatureError) Unwrap() error {
    return e.Err
}
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup
  COMMAND: cd go-env && devbox shell
  VERIFY: go version && go mod tidy
  BUILD: go build ./...

Task 2: Type Definitions
  CREATE: go-env/internal/types/feature.go
  PATTERN: Follow existing type patterns with validation tags
  VALIDATE: go build and go vet

Task 3: Interface Definition
  CREATE: Repository and Service interfaces
  PATTERN: Small, focused interfaces with context
  METHODS: All methods accept context as first parameter

Task 4: Service Implementation
  CREATE: go-env/internal/services/feature.go
  PATTERN: Struct implementing interface with dependencies
  CONTEXT: All methods use context for cancellation
  ERRORS: Proper error wrapping and handling

Task 5: Repository Layer (if needed)
  CREATE: go-env/internal/repository/feature.go
  PATTERN: Interface for data access abstraction
  CONTEXT: Database operations with context
  ERRORS: Wrapped database errors

Task 6: HTTP Handlers (if needed)
  CREATE: go-env/internal/handlers/feature.go
  PATTERN: HTTP handler functions with proper error responses
  CONTEXT: Request context propagation
  VALIDATION: Input validation and sanitization

Task 7: Comprehensive Testing
  CREATE: Table-driven tests for all components
  PATTERN: TestMain, setup/teardown, subtests
  COVERAGE: Use go test -cover for coverage analysis
  CONTEXT: Test context cancellation scenarios

Task 8: Integration and Documentation
  MODIFY: go-env/cmd/main.go for wiring
  PATTERN: Dependency injection and graceful shutdown
  DOCS: Comprehensive package documentation
```

### Per task pseudocode

**Task 4: Service Implementation**
```go
// internal/services/feature.go
package services

import (
    "context"
    "fmt"
    "time"
    "sync"
    
    "github.com/google/uuid"
    "your-module/internal/types"
)

// FeatureRepository defines the interface for feature data access
type FeatureRepository interface {
    Create(ctx context.Context, feature *types.Feature) error
    GetByID(ctx context.Context, id string) (*types.Feature, error)
    List(ctx context.Context, filters types.FeatureFilters) ([]*types.Feature, error)
    Update(ctx context.Context, feature *types.Feature) error
    Delete(ctx context.Context, id string) error
}

// FeatureService defines the interface for feature business logic
type FeatureService interface {
    Create(ctx context.Context, req types.CreateFeatureRequest) (*types.Feature, error)
    GetByID(ctx context.Context, id string) (*types.Feature, error)
    List(ctx context.Context, filters types.FeatureFilters) ([]*types.Feature, error)
    Update(ctx context.Context, id string, req types.UpdateFeatureRequest) (*types.Feature, error)
    Delete(ctx context.Context, id string) error
}

// featureService implements FeatureService
type featureService struct {
    repo FeatureRepository
    mu   sync.RWMutex // For in-memory implementation
    data map[string]*types.Feature
}

// NewFeatureService creates a new feature service
func NewFeatureService(repo FeatureRepository) FeatureService {
    return &featureService{
        repo: repo,
        data: make(map[string]*types.Feature),
    }
}

// Create creates a new feature
func (s *featureService) Create(ctx context.Context, req types.CreateFeatureRequest) (*types.Feature, error) {
    // Check context cancellation
    if err := ctx.Err(); err != nil {
        return nil, fmt.Errorf("context cancelled: %w", err)
    }

    // Validate input
    if req.Name == "" {
        return nil, &types.FeatureError{
            Op:  "create",
            Err: types.ErrFeatureNameEmpty,
        }
    }

    // Set defaults
    status := types.FeatureStatusPending
    if req.Status != nil {
        if err := req.Status.Validate(); err != nil {
            return nil, &types.FeatureError{
                Op:  "create",
                Err: err,
            }
        }
        status = *req.Status
    }

    now := time.Now()
    feature := &types.Feature{
        ID:          uuid.New().String(),
        Name:        req.Name,
        Description: req.Description,
        Status:      status,
        CreatedAt:   now,
        UpdatedAt:   now,
    }

    // Store feature
    if s.repo != nil {
        if err := s.repo.Create(ctx, feature); err != nil {
            return nil, &types.FeatureError{
                Op:  "create",
                Err: fmt.Errorf("repository create failed: %w", err),
            }
        }
    } else {
        // In-memory storage
        s.mu.Lock()
        s.data[feature.ID] = feature
        s.mu.Unlock()
    }

    return feature, nil
}

// GetByID retrieves a feature by ID
func (s *featureService) GetByID(ctx context.Context, id string) (*types.Feature, error) {
    if err := ctx.Err(); err != nil {
        return nil, fmt.Errorf("context cancelled: %w", err)
    }

    if id == "" {
        return nil, &types.FeatureError{
            Op:  "get",
            Err: errors.New("id cannot be empty"),
        }
    }

    if s.repo != nil {
        feature, err := s.repo.GetByID(ctx, id)
        if err != nil {
            return nil, &types.FeatureError{
                Op:  "get",
                Err: fmt.Errorf("repository get failed: %w", err),
            }
        }
        return feature, nil
    }

    // In-memory storage
    s.mu.RLock()
    feature, exists := s.data[id]
    s.mu.RUnlock()

    if !exists {
        return nil, &types.FeatureError{
            Op:  "get",
            Err: types.ErrFeatureNotFound,
        }
    }

    return feature, nil
}

// List retrieves features with filters
func (s *featureService) List(ctx context.Context, filters types.FeatureFilters) ([]*types.Feature, error) {
    if err := ctx.Err(); err != nil {
        return nil, fmt.Errorf("context cancelled: %w", err)
    }

    if s.repo != nil {
        features, err := s.repo.List(ctx, filters)
        if err != nil {
            return nil, &types.FeatureError{
                Op:  "list",
                Err: fmt.Errorf("repository list failed: %w", err),
            }
        }
        return features, nil
    }

    // In-memory storage with filtering
    s.mu.RLock()
    var features []*types.Feature
    for _, feature := range s.data {
        // Apply status filter
        if filters.Status != nil && feature.Status != *filters.Status {
            continue
        }

        // Apply name filter
        if filters.NameContains != nil && 
           !strings.Contains(strings.ToLower(feature.Name), 
                           strings.ToLower(*filters.NameContains)) {
            continue
        }

        features = append(features, feature)
    }
    s.mu.RUnlock()

    // Apply pagination
    offset := 0
    if filters.Offset != nil {
        offset = *filters.Offset
    }

    limit := 100
    if filters.Limit != nil {
        limit = *filters.Limit
    }

    if offset >= len(features) {
        return []*types.Feature{}, nil
    }

    end := offset + limit
    if end > len(features) {
        end = len(features)
    }

    return features[offset:end], nil
}
```

**Task 7: Comprehensive Testing**
```go
// tests/feature_test.go
package tests

import (
    "context"
    "testing"
    "time"
    
    "your-module/internal/services"
    "your-module/internal/types"
)

func TestFeatureService_Create(t *testing.T) {
    tests := []struct {
        name    string
        req     types.CreateFeatureRequest
        wantErr bool
        errType error
    }{
        {
            name: "successful creation",
            req: types.CreateFeatureRequest{
                Name:        "Test Feature",
                Description: stringPtr("Test description"),
                Status:      &types.FeatureStatusActive,
            },
            wantErr: false,
        },
        {
            name: "empty name",
            req: types.CreateFeatureRequest{
                Name: "",
            },
            wantErr: true,
            errType: types.ErrFeatureNameEmpty,
        },
        {
            name: "invalid status",
            req: types.CreateFeatureRequest{
                Name:   "Test Feature",
                Status: &types.FeatureStatus("invalid"),
            },
            wantErr: true,
            errType: types.ErrInvalidFeatureStatus,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            service := services.NewFeatureService(nil)
            ctx := context.Background()

            feature, err := service.Create(ctx, tt.req)

            if tt.wantErr {
                if err == nil {
                    t.Errorf("expected error but got none")
                    return
                }

                if tt.errType != nil {
                    var featureErr *types.FeatureError
                    if errors.As(err, &featureErr) {
                        if !errors.Is(featureErr.Err, tt.errType) {
                            t.Errorf("expected error type %v, got %v", tt.errType, featureErr.Err)
                        }
                    } else {
                        t.Errorf("expected FeatureError, got %T", err)
                    }
                }
                return
            }

            if err != nil {
                t.Errorf("unexpected error: %v", err)
                return
            }

            if feature == nil {
                t.Error("expected feature but got nil")
                return
            }

            if feature.Name != tt.req.Name {
                t.Errorf("expected name %s, got %s", tt.req.Name, feature.Name)
            }

            if feature.ID == "" {
                t.Error("expected non-empty ID")
            }

            if feature.CreatedAt.IsZero() {
                t.Error("expected non-zero CreatedAt")
            }
        })
    }
}

func TestFeatureService_ContextCancellation(t *testing.T) {
    service := services.NewFeatureService(nil)
    
    // Create a context that's already cancelled
    ctx, cancel := context.WithCancel(context.Background())
    cancel()

    req := types.CreateFeatureRequest{
        Name: "Test Feature",
    }

    _, err := service.Create(ctx, req)
    if err == nil {
        t.Error("expected error due to cancelled context")
    }

    if !errors.Is(err, context.Canceled) {
        t.Errorf("expected context.Canceled error, got %v", err)
    }
}

func TestFeatureService_Integration(t *testing.T) {
    service := services.NewFeatureService(nil)
    ctx := context.Background()

    // Create feature
    createReq := types.CreateFeatureRequest{
        Name:        "Integration Test",
        Description: stringPtr("Test feature for integration testing"),
        Status:      &types.FeatureStatusActive,
    }

    created, err := service.Create(ctx, createReq)
    if err != nil {
        t.Fatalf("failed to create feature: %v", err)
    }

    // Get feature
    retrieved, err := service.GetByID(ctx, created.ID)
    if err != nil {
        t.Fatalf("failed to get feature: %v", err)
    }

    if retrieved.Name != createReq.Name {
        t.Errorf("expected name %s, got %s", createReq.Name, retrieved.Name)
    }

    // Update feature
    updateReq := types.UpdateFeatureRequest{
        Name:   stringPtr("Updated Integration Test"),
        Status: &types.FeatureStatusInactive,
    }

    updated, err := service.Update(ctx, created.ID, updateReq)
    if err != nil {
        t.Fatalf("failed to update feature: %v", err)
    }

    if updated.Name != *updateReq.Name {
        t.Errorf("expected updated name %s, got %s", *updateReq.Name, updated.Name)
    }

    // List features
    filters := types.FeatureFilters{
        Status: &types.FeatureStatusInactive,
    }

    features, err := service.List(ctx, filters)
    if err != nil {
        t.Fatalf("failed to list features: %v", err)
    }

    if len(features) == 0 {
        t.Error("expected at least one feature in list")
    }

    // Delete feature
    err = service.Delete(ctx, created.ID)
    if err != nil {
        t.Fatalf("failed to delete feature: %v", err)
    }

    // Verify deletion
    _, err = service.GetByID(ctx, created.ID)
    if err == nil {
        t.Error("expected error when getting deleted feature")
    }
}

// Helper function
func stringPtr(s string) *string {
    return &s
}
```

### Integration Points
```yaml
MAIN_APPLICATION:
  - modify: go-env/cmd/main.go
  - pattern: Dependency injection and graceful shutdown
  
DEPENDENCIES:
  - add to: go-env/go.mod
  - pattern: Use go get for new dependencies
  
CONFIG:
  - add to: go-env/internal/config/
  - pattern: Environment configuration with validation
  
HTTP_SERVER:
  - create: go-env/internal/handlers/ (if needed)
  - pattern: HTTP handlers with proper error responses
```

## Validation Loop

### Level 1: Go Syntax & Style
```bash
cd go-env && devbox shell

# Format code with gofmt
go fmt ./...

# Imports with goimports
goimports -w .

# Lint with golangci-lint
golangci-lint run

# Vet for common issues
go vet ./...

# Expected: No errors or warnings. Fix all issues.
```

### Level 2: Unit Tests
```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests with race detection
go test -race ./...

# Verbose test output
go test -v ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Expected: All tests pass, 80%+ coverage, no race conditions
```

### Level 3: Build and Integration
```bash
# Build all packages
go build ./...

# Build specific binary
go build -o bin/app ./cmd/main.go

# Run integration tests
go test -tags=integration ./tests/integration/...

# Check module dependencies
go mod tidy
go mod verify

# Expected: Clean build, integration tests pass, modules verified
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] Dependencies resolved: `go mod tidy` clean
- [ ] Code formatted: `go fmt` and `goimports` clean
- [ ] Linting passed: `golangci-lint run` clean
- [ ] Vet checks passed: `go vet` clean
- [ ] All tests pass: `go test ./...`
- [ ] Test coverage 80%+: `go test -cover`
- [ ] Race conditions checked: `go test -race`
- [ ] Build successful: `go build ./...`
- [ ] Context usage proper: All functions accept context
- [ ] Error handling comprehensive: Errors properly wrapped
- [ ] Interfaces small and focused
- [ ] Documentation updated if needed

---

## Go-Specific Anti-Patterns to Avoid
- ❌ Don't ignore errors - handle them explicitly
- ❌ Don't use `panic` for recoverable errors - return errors
- ❌ Don't create large interfaces - keep them focused
- ❌ Don't skip context parameters - always accept context first
- ❌ Don't use `init()` functions unnecessarily - prefer explicit initialization
- ❌ Don't use global variables - use dependency injection
- ❌ Don't start goroutines without knowing how to stop them
- ❌ Don't use channels for simple data passing - use return values

## Go Best Practices
- ✅ Accept interfaces, return concrete types
- ✅ Use context.Context for cancellation and timeouts
- ✅ Handle errors explicitly and wrap them with context
- ✅ Write table-driven tests for comprehensive coverage
- ✅ Use small, focused interfaces
- ✅ Follow the single responsibility principle
- ✅ Use dependency injection for testability
- ✅ Write clear, descriptive variable and function names
- ✅ Document public APIs with clear comments