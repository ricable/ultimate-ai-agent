# Base PRP Command Template

This is the base template for all PRP-related commands using the Template Method pattern.
This file should not be called directly - it's used by specific command implementations.

## Command Structure

### Arguments Processing
- Parse command line arguments
- Validate input parameters
- Determine target environment(s)

### Validation Phase
- Validate feature file exists and is readable
- Check environment compatibility
- Verify required dependencies

### Processing Phase
- Execute command-specific logic (implemented by subclasses)
- Handle environment-specific requirements
- Apply appropriate templates

### Output Phase
- Generate final output
- Save to appropriate location
- Provide user feedback

## Template Method Implementation

```bash
# Step 1: Parse and validate arguments
parse_arguments() {
    FEATURE_FILE="$1"
    ENV_ARG="$2"
    
    if [[ -z "$FEATURE_FILE" ]]; then
        echo "Error: Feature file required"
        echo "Usage: /command-name <feature-file> [--env <environment>]"
        exit 1
    fi
    
    if [[ ! -f "$FEATURE_FILE" ]]; then
        echo "Error: Feature file '$FEATURE_FILE' not found"
        exit 1
    fi
}

# Step 2: Environment validation
validate_environment() {
    if [[ -n "$ENV_ARG" ]]; then
        case "$ENV_ARG" in
            python-env|typescript-env|rust-env|go-env|nushell-env|multi)
                echo "✓ Valid environment: $ENV_ARG"
                ;;
            *)
                echo "Error: Invalid environment '$ENV_ARG'"
                echo "Valid options: python-env, typescript-env, rust-env, go-env, nushell-env, multi"
                exit 1
                ;;
        esac
    fi
}

# Step 3: Command-specific processing (to be implemented by subclasses)
execute_command_logic() {
    echo "Error: execute_command_logic() must be implemented by subclass"
    exit 1
}

# Step 4: Output generation (to be implemented by subclasses)
generate_output() {
    echo "Error: generate_output() must be implemented by subclass"
    exit 1
}

# Main template method
main() {
    parse_arguments "$@"
    validate_environment
    execute_command_logic
    generate_output
}
```

## Shared Utilities

### Environment Detection
Automatically detect target environment from feature file if not specified:

```bash
detect_environment() {
    if [[ -z "$ENV_ARG" ]]; then
        # Analyze feature file content to determine likely environment
        if grep -qi "fastapi\|python\|django\|flask" "$FEATURE_FILE"; then
            ENV_ARG="python-env"
        elif grep -qi "react\|node\|typescript\|javascript" "$FEATURE_FILE"; then
            ENV_ARG="typescript-env"
        elif grep -qi "cargo\|rust\|tokio" "$FEATURE_FILE"; then
            ENV_ARG="rust-env"
        elif grep -qi "golang\|go\|gin" "$FEATURE_FILE"; then
            ENV_ARG="go-env"
        elif grep -qi "nushell\|nu\|pipeline" "$FEATURE_FILE"; then
            ENV_ARG="nushell-env"
        else
            ENV_ARG="multi"
        fi
        echo "Auto-detected environment: $ENV_ARG"
    fi
}
```

### Template Resolution
Resolve appropriate template based on environment:

```bash
resolve_template() {
    local env="$1"
    case "$env" in
        python-env)
            echo "context-engineering/templates/composed/python_composed.md"
            ;;
        typescript-env)
            echo "context-engineering/templates/composed/typescript_composed.md"
            ;;
        rust-env)
            echo "context-engineering/templates/composed/rust_composed.md"
            ;;
        go-env)
            echo "context-engineering/templates/composed/go_composed.md"
            ;;
        nushell-env)
            echo "context-engineering/templates/composed/nushell_composed.md"
            ;;
        multi)
            echo "context-engineering/templates/composed/multi_composed.md"
            ;;
        *)
            echo "context-engineering/templates/composed/base_composed.md"
            ;;
    esac
}
```

## Error Handling

### Graceful Degradation
```bash
handle_error() {
    local error_code="$1"
    local error_message="$2"
    
    echo "Error ($error_code): $error_message"
    
    case "$error_code" in
        "ENV_NOT_FOUND")
            echo "Suggestion: Ensure the target environment directory exists"
            echo "Available environments: python-env, typescript-env, rust-env, go-env, nushell-env"
            ;;
        "TEMPLATE_NOT_FOUND")
            echo "Suggestion: Check that template files exist in context-engineering/templates/"
            echo "Run: ls context-engineering/templates/composed/"
            ;;
        "VALIDATION_FAILED")
            echo "Suggestion: Review feature file format and requirements"
            echo "See: context-engineering/docs/feature-file-format.md"
            ;;
        *)
            echo "Suggestion: Check logs for more details or run with --debug flag"
            ;;
    esac
    
    exit "$error_code"
}
```

## Debug Support

### Debug Mode
```bash
debug_log() {
    if [[ "$DEBUG" == "true" ]]; then
        echo "[DEBUG] $*" >&2
    fi
}

enable_debug() {
    if [[ "$1" == "--debug" ]]; then
        DEBUG="true"
        shift
    fi
}
```

This base template provides the common structure and utilities that all PRP commands will inherit from, ensuring consistency and reducing duplication.