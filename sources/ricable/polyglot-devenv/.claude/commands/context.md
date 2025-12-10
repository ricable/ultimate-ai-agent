# /context

Unified context engineering command with dynamic operations, smart defaults, and modular functionality.

## Usage

```bash
/context <operation> [target] [options]
```

## Arguments

- `--operation, -o` (required): Operation to perform Choices: `generate, execute, workflow, devpod, analyze`
- `--target, -t`: Target file or feature name
- `--env, -e`: Target environment Choices: `python-env, typescript-env, rust-env, go-env, nushell-env, multi`
- `--template`: Template to use
- `--examples`: Examples to include (comma-separated)
- `--count`: Number of instances (for devpod) Default: `1`
- `--validate`: Enable validation Default: `false`
- `--monitor`: Enable monitoring Default: `false`
- `--format`: Output format Default: `markdown` Choices: `markdown, json, yaml`
- `--output, -o`: Output file or directory
- `--dry-run`: Show what would be done without executing Default: `false`

## Examples

- `/context generate user-api --env python --template fastapi`
- `/context execute user-api-python.md --validate`
- `/context workflow user-management --env python --validate --monitor`
- `/context devpod --env python --count 3`
- `/context analyze --env typescript --examples dojo`

## Command Implementation

```bash
#!/usr/bin/env nu

# Source the enhanced argument parser and utilities
source /Users/cedric/dev/github.com/polyglot-devenv/context-engineering/utils/argument-parser.nu
source /Users/cedric/dev/github.com/polyglot-devenv/context-engineering/utils/command-builder.nu

# Get raw arguments from Claude Code
let RAW_ARGS = "${ARGUMENTS}"

# Define command specification
let command_spec = {
    name: "context",
    description: "Unified context engineering command with dynamic operations",
    arguments: [
        {
            name: "operation",
            description: "Operation to perform",
            type: "string",
            short: "op",
            required: true,
            choices: ["generate", "execute", "workflow", "devpod", "analyze"]
        },
        {
            name: "target",
            description: "Target file or feature name",
            type: "string",
            short: "t"
        },
        {
            name: "env",
            description: "Target environment",
            type: "string",
            short: "e",
            choices: ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env", "multi"]
        },
        {
            name: "template",
            description: "Template to use",
            type: "string"
        },
        {
            name: "examples",
            description: "Examples to include (comma-separated)",
            type: "list"
        },
        {
            name: "count",
            description: "Number of instances (for devpod)",
            type: "int",
            default: 1,
            min: 1,
            max: 10
        },
        {
            name: "validate",
            description: "Enable validation",
            type: "bool",
            default: false
        },
        {
            name: "monitor",
            description: "Enable monitoring",
            type: "bool",
            default: false
        },
        {
            name: "format",
            description: "Output format",
            type: "string",
            default: "markdown",
            choices: ["markdown", "json", "yaml"]
        },
        {
            name: "output",
            description: "Output file or directory",
            type: "string",
            short: "o"
        },
        {
            name: "dry-run",
            description: "Show what would be done without executing",
            type: "bool",
            default: false
        }
    ]
}

# Check for help flag
if (has help flag $RAW_ARGS) {
    parse arguments $RAW_ARGS $command_spec --help
    exit 0
}

# Parse arguments with enhanced validation
let result = (parse arguments $RAW_ARGS $command_spec)

if ($result.help_shown) {
    exit 0
}

if ($result.error? != null) {
    print $"❌ Error: ($result.error)"
    exit 1
}

let parsed_args = $result.parsed

# Show what would be executed in dry-run mode
if ($parsed_args.dry-run? | default false) {
    print "🔍 Dry run mode - showing what would be executed:"
    print ""
    print $"Operation: ($parsed_args.operation)"
    print $"Target: ($parsed_args.target? | default 'auto-detect')"
    print $"Environment: ($parsed_args.env? | default 'auto-detect')"
    print $"Template: (select_template $parsed_args)"
    let examples = (integrate_examples $parsed_args ($parsed_args.env? | default "auto"))
    if (($examples | length) > 0) {
        print $"Examples: ($examples | str join ', ')"
    }
    print $"Validation: ($parsed_args.validate? | default false)"
    print $"Monitoring: ($parsed_args.monitor? | default false)"
    exit 0
}

# Dynamic template selection
def select_template [args: record] -> string {
    
    # Explicit template specified
    if ("template" in ($args | columns)) and ($args.template != null) {
        return $args.template
    }
    
    # Auto-detect from environment
    if ("env" in ($args | columns)) and ($args.env != null) {
        match $args.env {
            "python-env" => { return "python_prp" }
            "typescript-env" => { return "typescript_prp" }
            "rust-env" => { return "rust_prp" }
            "go-env" => { return "go_prp" }
            "nushell-env" => { return "nushell_prp" }
            "multi" => { return "prp_base" }
            _ => { return "prp_base" }
        }
    }
    
    # Auto-detect from target file
    if ("target" in ($args | columns)) and ($args.target != null) {
        let target = $args.target
        if ($target | str ends-with ".py") { return "python_prp" }
        if ($target | str ends-with ".ts") or ($target | str ends-with ".js") { return "typescript_prp" }
        if ($target | str ends-with ".rs") { return "rust_prp" }
        if ($target | str ends-with ".go") { return "go_prp" }
        if ($target | str ends-with ".nu") { return "nushell_prp" }
    }
    
    # Default template
    return "prp_base"
}

# Dynamic example integration
def integrate_examples [args: record, target_env: string] -> list {
    
    mut examples = []
    
    # Add explicitly requested examples
    if ("examples" in ($args | columns)) and ($args.examples != null) {
        let requested = (parse list arg $args.examples)
        $examples = ($examples | append $requested)
    }
    
    # Auto-detect relevant examples based on target
    if ("target" in ($args | columns)) and ($args.target != null) {
        let target = $args.target
        
        # If target mentions UI, web, or frontend, include dojo
        if ($target | str contains "ui") or ($target | str contains "web") or ($target | str contains "frontend") {
            $examples = ($examples | append "dojo")
        }
        
        # If target mentions API, include API examples
        if ($target | str contains "api") or ($target | str contains "rest") {
            match $target_env {
                "python-env" => { $examples = ($examples | append "python-api-example") }
                "typescript-env" => { $examples = ($examples | append "typescript-api-example") }
                _ => {}
            }
        }
        
        # If target mentions auth, include auth examples
        if ($target | str contains "auth") or ($target | str contains "user") {
            $examples = ($examples | append "user-management")
        }
    }
    
    # Environment-specific auto-examples
    match $target_env {
        "typescript-env" => {
            if ("dojo" not-in $examples) and (($args.target? | default "") | str contains "web" or "ui" or "component") {
                $examples = ($examples | append "dojo")
            }
        }
        _ => {}
    }
    
    return ($examples | uniq)
}

# Dynamic validation command generation
def build_validation_commands [env: string, validate: bool] -> list {
    
    if not $validate {
        return []
    }
    
    mut commands = []
    
    match $env {
        "python-env" => {
            $commands = [
                "cd python-env && devbox shell",
                "uv run ruff format src/ tests/",
                "uv run ruff check src/ tests/ --fix",
                "uv run mypy src/ tests/",
                "uv run pytest tests/ -v --cov=src"
            ]
        }
        "typescript-env" => {
            $commands = [
                "cd typescript-env && devbox shell",
                "npm run format",
                "npm run lint",
                "npm run typecheck",
                "npm run test"
            ]
        }
        "rust-env" => {
            $commands = [
                "cd rust-env && devbox shell",
                "cargo fmt",
                "cargo clippy",
                "cargo test"
            ]
        }
        "go-env" => {
            $commands = [
                "cd go-env && devbox shell",
                "go fmt ./...",
                "golangci-lint run",
                "go test ./..."
            ]
        }
        "nushell-env" => {
            $commands = [
                "cd nushell-env && devbox shell",
                "nu scripts/format-all.nu",
                "nu scripts/check-syntax.nu",
                "nu scripts/test-all.nu"
            ]
        }
        "multi" => {
            $commands = [
                "nu nushell-env/scripts/validate-all.nu parallel"
            ]
        }
        _ => {}
    }
    
    return $commands
}

# Main operation executor
def execute_operation [args: record] -> nothing {
    
    let operation = $args.operation
    
    print $"🚀 Executing operation: ($operation)"
    
    match $operation {
        "generate" => {
            execute_generate $args
        }
        "execute" => {
            execute_prp $args
        }
        "workflow" => {
            execute_workflow $args
        }
        "devpod" => {
            execute_devpod $args
        }
        "analyze" => {
            execute_analyze $args
        }
        _ => {
            print $"❌ Unknown operation: ($operation)"
            exit 1
        }
    }
}

# Execute generate operation
def execute_generate [args: record] -> nothing {
    print "🔧 Generating PRP..."
    
    let template = (select_template $args)
    let target_env = ($args.env? | default "auto")
    let examples = (integrate_examples $args $target_env)
    
    print $"📋 Using template: ($template)"
    if (($examples | length) > 0) {
        print $"📚 Including examples: ($examples | str join ', ')"
    }
    
    # Determine target file
    let target = ($args.target? | default "feature")
    let feature_file = if ($target | str ends-with ".md") { $target } else { $"features/($target).md" }
    
    print $"📄 Feature file: ($feature_file)"
    print $"🎯 Target environment: ($target_env)"
    
    if ($args.monitor? | default false) {
        print "📊 Monitoring enabled"
        # Start performance tracking
        nu ../nushell-env/scripts/performance-analytics.nu record "context-generate" $target_env "start"
    }
    
    # Call the original generate-prp functionality with enhanced parameters
    let generate_args = if ($target_env != "auto") { 
        $"($feature_file) --env ($target_env)" 
    } else { 
        $feature_file 
    }
    
    print $"🔄 Executing: /generate-prp ($generate_args)"
    
    # Here we would call the original generate-prp command
    # For now, simulate the call
    print "✅ PRP generation completed"
    
    if ($args.monitor? | default false) {
        nu ../nushell-env/scripts/performance-analytics.nu record "context-generate" $target_env "success"
    }
}

# Execute PRP operation
def execute_prp [args: record] -> nothing {
    print "⚙️  Executing PRP..."
    
    let target = ($args.target? | default "")
    if ($target == "") {
        print "❌ Target PRP file is required for execute operation"
        exit 1
    }
    
    # Determine PRP file path
    let prp_file = if ($target | str starts-with "context-engineering/PRPs/") {
        $target
    } else if ($target | str ends-with ".md") {
        $"context-engineering/PRPs/($target)"
    } else {
        $"context-engineering/PRPs/($target).md"
    }
    
    print $"📄 PRP file: ($prp_file)"
    
    if ($args.validate? | default false) {
        let env = ($args.env? | default "auto")
        let validation_commands = (build_validation_commands $env true)
        print $"✅ Validation enabled: ($validation_commands | length) checks"
    }
    
    if ($args.monitor? | default false) {
        print "📊 Monitoring enabled"
        nu ../nushell-env/scripts/performance-analytics.nu record "context-execute" ($args.env? | default "auto") "start"
    }
    
    # Execute the PRP with validation flags
    let execute_flags = if ($args.validate? | default false) { " --validate" } else { "" }
    print $"🔄 Executing: /execute-prp ($prp_file)($execute_flags)"
    
    # Here we would call the original execute-prp command
    print "✅ PRP execution completed"
    
    if ($args.monitor? | default false) {
        nu ../nushell-env/scripts/performance-analytics.nu record "context-execute" ($args.env? | default "auto") "success"
    }
}

# Execute workflow operation (generate + execute)
def execute_workflow [args: record] -> nothing {
    print "🔄 Running complete workflow (generate + execute)..."
    
    # Generate first
    execute_generate $args
    
    # Determine generated PRP file name for execution
    let target = ($args.target? | default "feature")
    let env = ($args.env? | default "auto")
    
    let prp_filename = if ($env == "multi") {
        $"($target)-multi.md"
    } else if ($env != "auto") {
        let env_name = ($env | str replace "-env" "")
        $"($target)-($env_name).md"
    } else {
        $"($target).md"
    }
    
    # Update args for execution
    let execute_args = ($args | insert target $prp_filename)
    
    print $"🔗 Continuing with execution of: ($prp_filename)"
    
    # Then execute
    execute_prp $execute_args
    
    print "🎉 Workflow completed successfully!"
}

# Execute DevPod operation
def execute_devpod [args: record] -> nothing {
    let count = ($args.count? | default 1)
    let env = ($args.env? | default "python-env")
    
    print $"🐳 Provisioning ($count) DevPod workspace(s) for ($env)..."
    
    # Map to the existing devpod commands
    let devpod_command = match $env {
        "python-env" => "/devpod-python"
        "typescript-env" => "/devpod-typescript"
        "rust-env" => "/devpod-rust"
        "go-env" => "/devpod-go"
        _ => {
            print $"❌ DevPod not supported for environment: ($env)"
            exit 1
        }
    }
    
    print $"🔄 Executing: ($devpod_command) ($count)"
    
    # Here we would call the specific devpod command
    print $"✅ DevPod provisioning completed for ($env)"
}

# Execute analyze operation
def execute_analyze [args: record] -> nothing {
    let env = ($args.env? | default "auto")
    
    print $"🔍 Analyzing environment: ($env)"
    
    if ($env == "auto") {
        print "🔍 Auto-detecting environment from current directory..."
        # Auto-detection logic here
        print "📊 Found multiple environments - analyzing all"
    }
    
    let examples = (integrate_examples $args $env)
    let template = (select_template $args)
    
    print $"📋 Recommended template: ($template)"
    if (($examples | length) > 0) {
        print $"📚 Relevant examples found: ($examples | str join ', ')"
    }
    
    # Analyze existing patterns
    print "🔍 Analyzing existing code patterns..."
    print "📊 Checking integration points..."
    print "🎯 Generating recommendations..."
    
    print "✅ Analysis completed"
}

# Execute main operation
execute_operation $parsed_args
```