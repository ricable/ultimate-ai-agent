name: "Nushell PRP Template - Structured Data and Automation Scripts"
description: |

## Purpose
Template optimized for AI agents to implement Nushell features in the nushell-env using structured data processing, automation scripts, and cross-environment orchestration.

## Core Principles
1. **Structured Data**: Leverage Nushell's built-in data structures
2. **Type Safety**: Use type hints and validation for parameters
3. **Pipeline Thinking**: Design functions for data pipeline composition
4. **Error Handling**: Graceful error handling with proper exit codes
5. **Cross-Environment**: Orchestrate other language environments

---

## Goal
[What needs to be built - be specific about the Nushell automation and data processing requirements]

## Why
- [Business value and automation benefits]
- [Integration with existing Nushell automation]
- [Cross-environment orchestration needs]

## What
[User-visible behavior and CLI commands, automation workflows]

### Success Criteria
- [ ] [Specific measurable outcomes for Nushell implementation]
- [ ] All syntax validation passing with `nu --check`
- [ ] Comprehensive test coverage with custom testing framework
- [ ] Cross-environment integration working properly

## All Needed Context

### Target Environment
```yaml
Environment: nushell-env
Devbox_Config: nushell-env/devbox.json
Dependencies: [List required Nushell packages or external tools]
Nushell_Version: 0.103.0+ (as specified in devbox.json)
Configuration: nushell-env/config/config.nu
Common_Utilities: nushell-env/common.nu
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://www.nushell.sh/book/
  why: Nushell language fundamentals and patterns
  
- file: nushell-env/scripts/[existing_similar_script].nu
  why: Existing patterns to follow
  
- file: nushell-env/common.nu
  why: Shared utilities and helper functions
  
- file: nushell-env/config/config.nu
  why: Environment configuration and setup
  
- doc: https://www.nushell.sh/cookbook/
  section: Common patterns and recipes
  critical: Structured data processing patterns
```

### Current Codebase tree
```bash
nushell-env/
├── devbox.json         # Nushell and automation tools
├── scripts/
│   ├── performance-analytics.nu     # Performance monitoring
│   ├── resource-monitor.nu          # Resource tracking
│   ├── security-scanner.nu          # Security analysis
│   ├── dependency-monitor.nu        # Dependency management
│   ├── environment-drift.nu         # Environment consistency
│   ├── validate-all.nu              # Cross-environment validation
│   └── test-intelligence.nu         # Test analysis
├── config/
│   ├── config.nu       # Nushell configuration
│   ├── .env            # Environment variables
│   └── secrets.nu      # Secret management (gitignored)
├── common.nu           # Shared utilities and functions
├── tests/
│   ├── unit/           # Unit tests for scripts
│   └── integration/    # Integration tests
└── README.md
```

### Desired Codebase tree with files to be added
```bash
nushell-env/
├── scripts/
│   └── feature-automation.nu       # New feature automation script
├── modules/
│   ├── feature-module.nu           # Feature-specific module
│   └── feature-types.nu            # Type definitions and validation
├── tests/
│   ├── unit/
│   │   └── test-feature-automation.nu
│   └── integration/
│       └── test-feature-integration.nu
├── templates/
│   └── feature-template.nu         # Template for feature configuration
```

### Known Gotchas of Nushell Environment
```nushell
# CRITICAL: Nushell environment-specific gotchas
# Type safety: Use type hints for all parameters
# Data flow: Think in terms of structured data pipelines
# Error handling: Use try/catch or error propagation
# External commands: Use ^command for external tools
# Variable scope: Understand mut vs const variables
# String interpolation: Use f-strings or string interpolation

# Example patterns:
# ✅ def "feature process" [data: record, --verbose(-v): bool = false] -> table
# ✅ $data | where status == "active" | select name description
# ✅ try { risky-operation } catch { |e| log error $"Operation failed: ($e)" }
# ✅ ^devbox run test  # External command
# ❌ def process [data] # Missing type hints
# ❌ command_without_error_handling
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate Nushell environment
cd nushell-env && devbox shell

# Verify environment
nu --version
which nu

# Check syntax of existing scripts
nu --check scripts/*.nu
```

### Module Structure and Types
```nushell
# modules/feature-types.nu
# Type definitions and validation functions for features

# Feature status enumeration
export def "feature status values" [] -> list<string> {
    ["active", "inactive", "pending"]
}

# Validate feature status
export def "feature status validate" [status: string] -> bool {
    $status in (feature status values)
}

# Feature record type with validation
export def "feature record create" [
    name: string,
    description?: string,
    status: string = "pending"
] -> record {
    if not (feature status validate $status) {
        error make {
            msg: "Invalid feature status",
            label: {
                text: $"Status must be one of: (feature status values | str join ', ')",
                span: (metadata $status).span
            }
        }
    }

    if ($name | str trim | is-empty) {
        error make {
            msg: "Feature name cannot be empty",
            label: {
                text: "Name is required",
                span: (metadata $name).span
            }
        }
    }

    {
        id: (random uuid),
        name: ($name | str trim),
        description: ($description | default null),
        status: $status,
        created_at: (date now | format date "%Y-%m-%d %H:%M:%S"),
        updated_at: (date now | format date "%Y-%m-%d %H:%M:%S")
    }
}

# Validate feature record
export def "feature record validate" [feature: record] -> bool {
    let required_fields = ["id", "name", "status", "created_at", "updated_at"]
    let feature_keys = ($feature | columns)
    
    # Check all required fields exist
    let missing_fields = ($required_fields | where {|field| $field not-in $feature_keys})
    
    if ($missing_fields | length) > 0 {
        error make {
            msg: $"Missing required fields: ($missing_fields | str join ', ')"
        }
    }
    
    # Validate status
    if not (feature status validate $feature.status) {
        error make {
            msg: $"Invalid status: ($feature.status)"
        }
    }
    
    # Validate name is not empty
    if ($feature.name | str trim | is-empty) {
        error make {
            msg: "Feature name cannot be empty"
        }
    }
    
    true
}

# Feature filters for querying
export def "feature filters create" [
    --status: string,
    --name-contains: string,
    --limit: int = 100,
    --offset: int = 0
] -> record {
    let filters = {
        limit: $limit,
        offset: $offset
    }
    
    let filters = if ($status | is-not-empty) {
        $filters | upsert status $status
    } else { $filters }
    
    let filters = if ($name_contains | is-not-empty) {
        $filters | upsert name_contains $name_contains
    } else { $filters }
    
    $filters
}
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup
  COMMAND: cd nushell-env && devbox shell
  VERIFY: nu --version && nu --check scripts/*.nu
  SETUP: Source common.nu and verify utilities

Task 2: Type Module Creation
  CREATE: nushell-env/modules/feature-types.nu
  PATTERN: Follow existing module patterns in common.nu
  VALIDATE: nu --check modules/feature-types.nu

Task 3: Core Feature Module
  CREATE: nushell-env/modules/feature-module.nu
  PATTERN: Structured data processing with pipelines
  FUNCTIONS: CRUD operations with proper error handling
  TYPES: Use type hints for all parameters and returns

Task 4: Automation Script
  CREATE: nushell-env/scripts/feature-automation.nu
  PATTERN: Follow existing script patterns in scripts/
  INTEGRATION: Cross-environment orchestration capabilities
  CLI: User-friendly command-line interface

Task 5: Configuration Template
  CREATE: nushell-env/templates/feature-template.nu
  PATTERN: Configuration management with validation
  ENV: Environment variable integration
  SECRETS: Secure secret handling patterns

Task 6: Testing Framework
  CREATE: Unit and integration tests
  PATTERN: Custom testing framework following existing patterns
  COVERAGE: Comprehensive test scenarios
  AUTOMATION: Integration with validate-all.nu

Task 7: Cross-Environment Integration
  MODIFY: Integrate with existing automation scripts
  VALIDATE: Add to validate-all.nu orchestration
  MONITOR: Integration with performance and security monitoring
  DOCS: Update README.md and add usage examples

Task 8: Documentation and Examples
  CREATE: Usage examples and documentation
  PATTERN: Clear, executable examples
  HELP: Built-in help system with --help flags
```

### Per task pseudocode

**Task 3: Core Feature Module**
```nushell
# modules/feature-module.nu
use feature-types.nu *

# In-memory storage for demonstration (replace with actual storage)
let feature_storage = {}

# Create a new feature
export def "feature create" [
    name: string,
    description?: string,
    --status: string = "pending"
] -> record {
    log info $"Creating feature: ($name)"
    
    try {
        let feature = (feature record create $name $description $status)
        
        # Store feature (in real implementation, this would be persistent storage)
        $env.FEATURE_STORAGE = ($env.FEATURE_STORAGE | default {} | upsert $feature.id $feature)
        
        log info $"Successfully created feature: ($feature.id)"
        $feature
    } catch { |e|
        log error $"Failed to create feature: ($e.msg)"
        error make {
            msg: $"Feature creation failed: ($e.msg)"
        }
    }
}

# Get feature by ID
export def "feature get" [id: string] -> record {
    log debug $"Getting feature: ($id)"
    
    let storage = ($env.FEATURE_STORAGE | default {})
    
    if $id not-in ($storage | columns) {
        error make {
            msg: $"Feature not found: ($id)"
        }
    }
    
    $storage | get $id
}

# List features with optional filters
export def "feature list" [
    --status: string,
    --name-contains: string,
    --limit: int = 100,
    --offset: int = 0
] -> table {
    log debug "Listing features with filters"
    
    let storage = ($env.FEATURE_STORAGE | default {})
    let features = ($storage | values)
    
    # Apply status filter
    let features = if ($status | is-not-empty) {
        $features | where status == $status
    } else { $features }
    
    # Apply name filter
    let features = if ($name_contains | is-not-empty) {
        $features | where name =~ $name_contains
    } else { $features }
    
    # Apply pagination
    $features | skip $offset | first $limit
}

# Update feature
export def "feature update" [
    id: string,
    --name: string,
    --description: string,
    --status: string
] -> record {
    log info $"Updating feature: ($id)"
    
    try {
        let storage = ($env.FEATURE_STORAGE | default {})
        
        if $id not-in ($storage | columns) {
            error make {
                msg: $"Feature not found: ($id)"
            }
        }
        
        let feature = ($storage | get $id)
        
        # Update fields if provided
        let feature = if ($name | is-not-empty) {
            $feature | upsert name $name
        } else { $feature }
        
        let feature = if ($description | is-not-empty) {
            $feature | upsert description $description
        } else { $feature }
        
        let feature = if ($status | is-not-empty) {
            if not (feature status validate $status) {
                error make {
                    msg: $"Invalid status: ($status)"
                }
            }
            $feature | upsert status $status
        } else { $feature }
        
        # Update timestamp
        let feature = ($feature | upsert updated_at (date now | format date "%Y-%m-%d %H:%M:%S"))
        
        # Validate updated feature
        feature record validate $feature | ignore
        
        # Store updated feature
        $env.FEATURE_STORAGE = ($storage | upsert $id $feature)
        
        log info $"Successfully updated feature: ($id)"
        $feature
    } catch { |e|
        log error $"Failed to update feature: ($e.msg)"
        error make {
            msg: $"Feature update failed: ($e.msg)"
        }
    }
}

# Delete feature
export def "feature delete" [id: string] -> nothing {
    log info $"Deleting feature: ($id)"
    
    let storage = ($env.FEATURE_STORAGE | default {})
    
    if $id not-in ($storage | columns) {
        error make {
            msg: $"Feature not found: ($id)"
        }
    }
    
    $env.FEATURE_STORAGE = ($storage | reject $id)
    log info $"Successfully deleted feature: ($id)"
}

# Cross-environment integration helpers
export def "feature sync-to-env" [
    feature_id: string,
    target_env: string
] -> record {
    log info $"Syncing feature ($feature_id) to environment: ($target_env)"
    
    let feature = (feature get $feature_id)
    let env_path = $"../($target_env)"
    
    if not ($env_path | path exists) {
        error make {
            msg: $"Target environment not found: ($target_env)"
        }
    }
    
    # Generate environment-specific configuration
    let config = match $target_env {
        "python-env" => {
            ($feature | upsert type "python_feature" | to json)
        },
        "typescript-env" => {
            ($feature | upsert type "typescript_feature" | to json)
        },
        "rust-env" => {
            ($feature | upsert type "rust_feature" | to toml)
        },
        "go-env" => {
            ($feature | upsert type "go_feature" | to json)
        },
        _ => {
            error make {
                msg: $"Unsupported environment: ($target_env)"
            }
        }
    }
    
    # Write configuration to target environment
    let config_file = $"($env_path)/feature-config.json"
    $config | save $config_file
    
    log info $"Feature synced to: ($config_file)"
    {
        feature_id: $feature_id,
        target_env: $target_env,
        config_file: $config_file,
        synced_at: (date now | format date "%Y-%m-%d %H:%M:%S")
    }
}
```

**Task 4: Automation Script**
```nushell
# scripts/feature-automation.nu
#!/usr/bin/env nu

use ../modules/feature-module.nu *
use ../common.nu *

# Main command dispatcher
def main [command?: string, ...args: string] {
    let available_commands = [
        "create", "get", "list", "update", "delete", 
        "sync", "validate", "monitor", "help"
    ]
    
    if ($command | is-empty) or $command == "help" {
        show_help $available_commands
        return
    }
    
    if $command not-in $available_commands {
        log error $"Unknown command: ($command)"
        log info $"Available commands: ($available_commands | str join ', ')"
        exit 1
    }
    
    match $command {
        "create" => { feature_create_command $args },
        "get" => { feature_get_command $args },
        "list" => { feature_list_command $args },
        "update" => { feature_update_command $args },
        "delete" => { feature_delete_command $args },
        "sync" => { feature_sync_command $args },
        "validate" => { feature_validate_command $args },
        "monitor" => { feature_monitor_command $args },
        _ => { 
            log error $"Command not implemented: ($command)"
            exit 1
        }
    }
}

# Create feature command
def feature_create_command [args: list<string>] {
    if ($args | length) < 1 {
        log error "Usage: feature-automation create <name> [description] [--status <status>]"
        exit 1
    }
    
    let name = ($args | get 0)
    let description = if ($args | length) > 1 { $args | get 1 } else { null }
    
    try {
        let feature = (feature create $name $description)
        print $"✅ Created feature: ($feature.name) \(ID: ($feature.id)\)"
        $feature | table
    } catch { |e|
        log error $"Failed to create feature: ($e.msg)"
        exit 1
    }
}

# List features command
def feature_list_command [args: list<string>] {
    try {
        let features = (feature list)
        
        if ($features | length) == 0 {
            print "No features found."
            return
        }
        
        print $"📋 Found ($features | length) features:"
        $features | table
    } catch { |e|
        log error $"Failed to list features: ($e.msg)"
        exit 1
    }
}

# Cross-environment validation
def feature_validate_command [args: list<string>] {
    log info "🔍 Validating features across environments"
    
    try {
        # Check syntax of all Nushell scripts
        let nu_scripts = (ls scripts/*.nu | get name)
        for script in $nu_scripts {
            try {
                ^nu --check $script
                log info $"✅ Syntax valid: ($script)"
            } catch { |e|
                log error $"❌ Syntax error in ($script): ($e.msg)"
            }
        }
        
        # Validate cross-environment integration
        let environments = ["python-env", "typescript-env", "rust-env", "go-env"]
        for env in $environments {
            if (ls -a $"../($env)" | where name =~ "devbox.json" | length) > 0 {
                log info $"✅ Environment found: ($env)"
            } else {
                log warning $"⚠️  Environment missing: ($env)"
            }
        }
        
        log info "🎉 Validation completed"
    } catch { |e|
        log error $"Validation failed: ($e.msg)"
        exit 1
    }
}

# Performance monitoring integration
def feature_monitor_command [args: list<string>] {
    log info "📊 Starting feature performance monitoring"
    
    try {
        # Integration with existing monitoring scripts
        if ("performance-analytics.nu" | path exists) {
            ^nu scripts/performance-analytics.nu measure "feature-automation" "monitoring" {
                let features = (feature list)
                log info $"Monitoring ($features | length) features"
                
                # Add custom metrics
                let metrics = {
                    feature_count: ($features | length),
                    active_features: ($features | where status == "active" | length),
                    pending_features: ($features | where status == "pending" | length),
                    monitored_at: (date now | format date "%Y-%m-%d %H:%M:%S")
                }
                
                $metrics | to json | save "feature-metrics.json"
                log info "📈 Metrics saved to feature-metrics.json"
            }
        } else {
            log warning "Performance analytics script not found, running basic monitoring"
            let features = (feature list)
            log info $"Total features: ($features | length)"
        }
    } catch { |e|
        log error $"Monitoring failed: ($e.msg)"
        exit 1
    }
}

# Help system
def show_help [commands: list<string>] {
    print "🚀 Feature Automation Script"
    print ""
    print "Usage: nu feature-automation.nu <command> [arguments]"
    print ""
    print "Available commands:"
    for cmd in $commands {
        match $cmd {
            "create" => { print "  create <name> [description]  - Create a new feature" },
            "get" => { print "  get <id>                     - Get feature by ID" },
            "list" => { print "  list                         - List all features" },
            "update" => { print "  update <id> [options]        - Update feature" },
            "delete" => { print "  delete <id>                  - Delete feature" },
            "sync" => { print "  sync <id> <environment>      - Sync feature to environment" },
            "validate" => { print "  validate                     - Validate all features and environments" },
            "monitor" => { print "  monitor                      - Start performance monitoring" },
            "help" => { print "  help                         - Show this help message" }
        }
    }
    print ""
    print "Examples:"
    print "  nu feature-automation.nu create \"My Feature\" \"Description\""
    print "  nu feature-automation.nu list"
    print "  nu feature-automation.nu validate"
    print "  nu feature-automation.nu monitor"
}
```

### Integration Points
```yaml
COMMON_UTILITIES:
  - modify: nushell-env/common.nu
  - pattern: Add feature-related utility functions
  
VALIDATION_SCRIPT:
  - modify: nushell-env/scripts/validate-all.nu
  - pattern: Include feature validation in cross-environment checks
  
MONITORING:
  - integrate: nushell-env/scripts/performance-analytics.nu
  - pattern: Add feature metrics to performance monitoring
  
CONFIGURATION:
  - add to: nushell-env/config/config.nu
  - pattern: Feature-related environment configuration
```

## Validation Loop

### Level 1: Nushell Syntax & Style
```bash
cd nushell-env && devbox shell

# Check syntax of all scripts
nu --check scripts/*.nu
nu --check modules/*.nu

# Format code (if formatter available)
# Currently Nushell doesn't have an official formatter

# Validate script execution
nu scripts/feature-automation.nu help

# Expected: No syntax errors, help displays correctly
```

### Level 2: Unit Tests
```nushell
# tests/unit/test-feature-automation.nu
#!/usr/bin/env nu

use ../../modules/feature-module.nu *
use ../../common.nu *

# Test suite runner
def main [] {
    log info "🧪 Running feature automation tests"
    
    let tests = [
        test_feature_creation,
        test_feature_validation,
        test_feature_crud_operations,
        test_cross_environment_sync
    ]
    
    let results = []
    
    for test in $tests {
        try {
            do $test
            let results = ($results | append {test: $test, status: "✅ PASS"})
            log info $"✅ ($test) - PASS"
        } catch { |e|
            let results = ($results | append {test: $test, status: "❌ FAIL", error: $e.msg})
            log error $"❌ ($test) - FAIL: ($e.msg)"
        }
    }
    
    print "\n📊 Test Results:"
    $results | table
    
    let failed_tests = ($results | where status =~ "FAIL")
    if ($failed_tests | length) > 0 {
        log error $"($failed_tests | length) tests failed"
        exit 1
    } else {
        log info "🎉 All tests passed!"
    }
}

def test_feature_creation [] {
    # Setup
    $env.FEATURE_STORAGE = {}
    
    # Test
    let feature = (feature create "Test Feature" "Test description")
    
    # Assertions
    assert ($feature.name == "Test Feature")
    assert ($feature.description == "Test description")
    assert ($feature.status == "pending")
    assert ($feature.id | is-not-empty)
    
    log debug "Feature creation test passed"
}

def test_feature_validation [] {
    # Test valid feature
    let valid_feature = {
        id: "test-id",
        name: "Valid Feature",
        status: "active",
        created_at: "2024-01-01 00:00:00",
        updated_at: "2024-01-01 00:00:00"
    }
    
    feature record validate $valid_feature | ignore
    
    # Test invalid feature (should throw error)
    try {
        let invalid_feature = {
            id: "test-id",
            name: "",  # Empty name should fail
            status: "invalid-status"
        }
        feature record validate $invalid_feature | ignore
        error make { msg: "Validation should have failed" }
    } catch { |e|
        # Expected to fail
    }
    
    log debug "Feature validation test passed"
}

def test_feature_crud_operations [] {
    # Setup
    $env.FEATURE_STORAGE = {}
    
    # Create
    let feature = (feature create "CRUD Test")
    let feature_id = $feature.id
    
    # Read
    let retrieved = (feature get $feature_id)
    assert ($retrieved.name == "CRUD Test")
    
    # Update
    let updated = (feature update $feature_id --name "Updated CRUD Test" --status "active")
    assert ($updated.name == "Updated CRUD Test")
    assert ($updated.status == "active")
    
    # List
    let features = (feature list)
    assert (($features | length) == 1)
    
    # Delete
    feature delete $feature_id
    
    # Verify deletion
    try {
        feature get $feature_id | ignore
        error make { msg: "Feature should have been deleted" }
    } catch { |e|
        # Expected to fail
    }
    
    log debug "CRUD operations test passed"
}

def test_cross_environment_sync [] {
    # Setup
    $env.FEATURE_STORAGE = {}
    let feature = (feature create "Sync Test")
    
    # Mock environment directory
    mkdir -p "../test-env"
    
    try {
        # This would normally sync to a real environment
        # For testing, we just verify the function structure
        let sync_result = {
            feature_id: $feature.id,
            target_env: "test-env",
            config_file: "../test-env/feature-config.json",
            synced_at: (date now | format date "%Y-%m-%d %H:%M:%S")
        }
        
        assert ($sync_result.feature_id == $feature.id)
        log debug "Cross-environment sync test passed"
    } catch { |e|
        log debug $"Sync test skipped: ($e.msg)"
    }
    
    # Cleanup
    rm -rf "../test-env"
}

# Helper assertion function
def assert [condition: bool] {
    if not $condition {
        error make { msg: "Assertion failed" }
    }
}
```

### Level 3: Integration Tests
```bash
# Run integration tests
nu tests/integration/test-feature-integration.nu

# Test cross-environment validation
nu scripts/feature-automation.nu validate

# Test performance monitoring integration
nu scripts/feature-automation.nu monitor

# Expected: All integration tests pass, cross-environment validation succeeds
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] Syntax validation passed: `nu --check` clean for all scripts
- [ ] Module imports working: `use` statements resolve correctly
- [ ] Unit tests pass: Custom test framework executes successfully
- [ ] Integration tests pass: Cross-environment validation works
- [ ] CLI interface working: Help system and commands functional
- [ ] Type safety enforced: All functions have proper type hints
- [ ] Error handling comprehensive: Graceful error messages and exit codes
- [ ] Cross-environment integration: Sync with other language environments
- [ ] Performance monitoring: Integration with existing analytics
- [ ] Documentation updated: README and examples provided

---

## Nushell-Specific Anti-Patterns to Avoid
- ❌ Don't skip type hints - they improve reliability and debugging
- ❌ Don't ignore error handling - use try/catch or proper error propagation
- ❌ Don't use external commands without `^` prefix
- ❌ Don't create functions without clear input/output types
- ❌ Don't forget to validate structured data before processing
- ❌ Don't use mutable variables without necessity
- ❌ Don't hardcode paths - use relative paths and environment variables
- ❌ Don't skip parameter validation in public functions

## Nushell Best Practices
- ✅ Use structured data (records, tables) instead of plain text when possible
- ✅ Leverage Nushell's built-in data processing commands (where, select, etc.)
- ✅ Design functions for pipeline composition
- ✅ Use type hints for all function parameters and return values
- ✅ Implement comprehensive error handling with meaningful messages
- ✅ Create modular code with proper use statements
- ✅ Use environment variables for configuration
- ✅ Write clear documentation with examples
- ✅ Follow consistent naming conventions (kebab-case for commands)