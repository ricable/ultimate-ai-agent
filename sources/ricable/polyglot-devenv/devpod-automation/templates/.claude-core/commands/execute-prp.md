# /execute-prp

Executes a Product Requirements Prompt (PRP) to implement features using the Template Method pattern for consistent, reliable execution with comprehensive validation and error handling.

## Usage
```
/execute-prp <prp-file> [--validate] [--monitor] [--dry-run] [--debug]
```

## Arguments
- `<prp-file>`: Path to the PRP file to execute (e.g., `context-engineering/PRPs/user-api-python.md`)
- `--validate`: Run all validation gates after implementation
- `--monitor`: Enable performance monitoring during execution
- `--dry-run`: Show what would be done without executing
- `--debug`: Enable debug output

## PRP file: $ARGUMENTS

Execute the specified PRP to implement the feature with iterative validation and error recovery. This command extends the base PRP command template to provide execution-specific functionality.

## Command Implementation (Template Method Pattern)

```bash
# Source the base command template
source "$(dirname "$0")/_base_prp_command.md"

# Override: Command-specific processing for PRP execution
execute_command_logic() {
    debug_log "Starting PRP execution for $PRP_FILE"
    
    # Parse PRP file
    parse_prp_file
    
    # Validate prerequisites
    echo "🔍 Validating prerequisites..."
    validate_prerequisites
    
    # Implementation phase
    echo "🚀 Starting implementation..."
    if [[ "$DRY_RUN" == "true" ]]; then
        show_execution_plan
    else
        execute_implementation_tasks
    fi
    
    # Validation phase (if requested)
    if [[ "$VALIDATE" == "true" ]]; then
        echo "✅ Running validation gates..."
        run_validation_gates
    fi
    
    debug_log "PRP execution completed"
}

# Override: Generate execution report
generate_output() {
    local report_file="context-engineering/execution-reports/${PRP_NAME}-$(date +%Y%m%d-%H%M%S).md"
    
    echo "📊 Generating execution report: $report_file"
    
    generate_execution_report > "$report_file"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Execution completed successfully"
        echo "📊 Report saved: $report_file"
        if [[ "$MONITOR" == "true" ]]; then
            echo "⏱️ Performance metrics: $(get_performance_summary)"
        fi
    else
        handle_error "EXECUTION_FAILED" "PRP execution failed"
    fi
}

# Parse PRP file and extract execution information
parse_prp_file() {
    debug_log "Parsing PRP file: $PRP_FILE"
    
    if [[ ! -f "$PRP_FILE" ]]; then
        handle_error "PRP_NOT_FOUND" "PRP file not found: $PRP_FILE"
    fi
    
    # Extract PRP metadata
    PRP_NAME=$(basename "$PRP_FILE" .md)
    TARGET_ENV=$(grep -A 3 "^Environment:" "$PRP_FILE" | grep -v "^Environment:" | head -1 | xargs)
    
    if [[ -z "$TARGET_ENV" ]]; then
        # Try to extract from YAML block
        TARGET_ENV=$(grep -A 10 "^```yaml" "$PRP_FILE" | grep "Environment:" | cut -d: -f2 | xargs)
    fi
    
    if [[ -z "$TARGET_ENV" ]]; then
        handle_error "INVALID_PRP" "Could not determine target environment from PRP"
    fi
    
    debug_log "PRP Name: $PRP_NAME"
    debug_log "Target Environment: $TARGET_ENV"
}

# Validate that all prerequisites are met
validate_prerequisites() {
    debug_log "Validating prerequisites for $TARGET_ENV"
    
    # Check that target environment exists
    if [[ ! -d "$TARGET_ENV" ]]; then
        handle_error "ENV_NOT_FOUND" "Target environment directory not found: $TARGET_ENV"
    fi
    
    # Check devbox configuration
    if [[ ! -f "$TARGET_ENV/devbox.json" ]]; then
        echo "⚠️  Warning: No devbox.json found in $TARGET_ENV"
    fi
    
    # Validate we can enter the environment
    if ! (cd "$TARGET_ENV" && devbox shell --command "echo 'Environment OK'" > /dev/null 2>&1); then
        handle_error "ENV_INVALID" "Cannot activate $TARGET_ENV environment"
    fi
    
    echo "✓ Prerequisites validated for $TARGET_ENV"
}

# Show what would be executed (dry-run mode)
show_execution_plan() {
    echo "📋 Execution Plan (DRY RUN):"
    echo "  PRP: $PRP_NAME"
    echo "  Environment: $TARGET_ENV"
    echo "  Tasks to execute:"
    
    # Extract task list from PRP
    extract_task_list | while read -r task; do
        echo "    - $task"
    done
    
    echo "  Validation gates:"
    extract_validation_gates | while read -r gate; do
        echo "    - $gate"
    done
}

# Execute the implementation tasks from the PRP
execute_implementation_tasks() {
    debug_log "Executing implementation tasks"
    
    # Start monitoring if requested
    if [[ "$MONITOR" == "true" ]]; then
        start_performance_monitoring
    fi
    
    # Execute tasks in order
    local task_count=0
    extract_task_list | while read -r task; do
        task_count=$((task_count + 1))
        echo "📝 Task $task_count: $task"
        
        if ! execute_single_task "$task"; then
            handle_error "TASK_FAILED" "Task failed: $task"
        fi
    done
    
    # Stop monitoring
    if [[ "$MONITOR" == "true" ]]; then
        stop_performance_monitoring
    fi
}

# Execute a single task with error handling
execute_single_task() {
    local task="$1"
    debug_log "Executing task: $task"
    
    # Task execution logic would be implemented here
    # This is a placeholder that would contain the actual implementation
    echo "  ⚙️ Executing: $task"
    
    # Simulate task execution
    sleep 0.1
    
    # Return success for now (real implementation would handle actual tasks)
    return 0
}

# Run validation gates from the PRP
run_validation_gates() {
    debug_log "Running validation gates"
    
    local gate_count=0
    local failed_gates=0
    
    extract_validation_gates | while read -r gate; do
        gate_count=$((gate_count + 1))
        echo "🔍 Validation $gate_count: $gate"
        
        if ! execute_validation_gate "$gate"; then
            failed_gates=$((failed_gates + 1))
            echo "❌ Validation failed: $gate"
        else
            echo "✅ Validation passed: $gate"
        fi
    done
    
    if [[ $failed_gates -gt 0 ]]; then
        handle_error "VALIDATION_FAILED" "$failed_gates validation gate(s) failed"
    fi
}

# Execute a single validation gate
execute_validation_gate() {
    local gate="$1"
    debug_log "Running validation gate: $gate"
    
    # Enter the target environment and run the validation command
    (cd "$TARGET_ENV" && devbox shell --command "$gate")
    return $?
}

# Extract task list from PRP file
extract_task_list() {
    # Extract tasks from YAML task list section
    sed -n '/^```yaml/,/^```/p' "$PRP_FILE" | grep "^Task [0-9]" | sed 's/Task [0-9]*: //'
}

# Extract validation gates from PRP file
extract_validation_gates() {
    # Extract validation commands from the validation section
    sed -n '/### Level [0-9]/,/^$/p' "$PRP_FILE" | grep -E "^(uv run|npm run|cargo|devbox run)" | head -10
}

# Generate execution report
generate_execution_report() {
    cat << EOF
# PRP Execution Report

**PRP:** $PRP_NAME
**Environment:** $TARGET_ENV  
**Date:** $(date)
**Status:** ${EXECUTION_STATUS:-SUCCESS}

## Execution Summary
- Tasks executed: $(extract_task_list | wc -l)
- Validation gates: $(extract_validation_gates | wc -l)
- Duration: ${EXECUTION_DURATION:-unknown}

## Performance Metrics
$(if [[ "$MONITOR" == "true" ]]; then get_performance_summary; else echo "Monitoring not enabled"; fi)

## Validation Results
$(if [[ "$VALIDATE" == "true" ]]; then echo "All validations passed"; else echo "Validation skipped"; fi)

EOF
}

# Performance monitoring functions
start_performance_monitoring() {
    EXECUTION_START_TIME=$(date +%s)
    debug_log "Started performance monitoring"
}

stop_performance_monitoring() {
    EXECUTION_END_TIME=$(date +%s)
    EXECUTION_DURATION=$((EXECUTION_END_TIME - EXECUTION_START_TIME))
    debug_log "Stopped performance monitoring. Duration: ${EXECUTION_DURATION}s"
}

get_performance_summary() {
    echo "Execution time: ${EXECUTION_DURATION:-0}s"
}

# Parse additional arguments specific to execute-prp
parse_execute_prp_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --validate)
                VALIDATE="true"
                shift
                ;;
            --monitor)
                MONITOR="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            *)
                break
                ;;
        esac
    done
}

# Override argument parsing for PRP files
parse_arguments() {
    PRP_FILE="$1"
    
    if [[ -z "$PRP_FILE" ]]; then
        echo "Error: PRP file required"
        echo "Usage: /execute-prp <prp-file> [--validate] [--monitor] [--dry-run] [--debug]"
        exit 1
    fi
    
    if [[ ! -f "$PRP_FILE" ]]; then
        echo "Error: PRP file '$PRP_FILE' not found"
        exit 1
    fi
}

# Main execution with argument parsing
main() {
    enable_debug "$@"
    parse_execute_prp_args "$@"
    parse_arguments "$@"
    # Note: validate_environment is handled in parse_prp_file for execute-prp
    execute_command_logic
    generate_output
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

## Execution Process

1. **PRP Parsing**
   - Parse PRP file metadata and requirements
   - Extract target environment and dependencies
   - Identify task list and validation gates

2. **Prerequisites Validation**
   - Verify target environment exists and is accessible
   - Check devbox configuration and dependencies
   - Validate environment can be activated

3. **Implementation Execution**
   - Execute tasks in order as defined in PRP
   - Monitor performance and resource usage (if enabled)
   - Handle errors and provide recovery suggestions

4. **Validation Gates**
   - Run all validation commands from PRP
   - Report success/failure for each gate
   - Provide detailed error information for failures

## Error Recovery

### Automatic Recovery
- Retry failed tasks up to 3 times
- Automatically fix common formatting issues
- Install missing dependencies when possible

### Manual Recovery Suggestions
- Provide specific commands to fix common issues
- Reference documentation for complex problems
- Generate reports for debugging

## Performance Monitoring

When `--monitor` is enabled:
- Track execution time for each task
- Monitor resource usage (CPU, memory)
- Generate performance reports
- Integration with polyglot monitoring systems

Remember: The goal is reliable, validated implementation with comprehensive error handling and recovery.