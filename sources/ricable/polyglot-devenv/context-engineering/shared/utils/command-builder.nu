#!/usr/bin/env nu
# Command builder for dynamic context engineering commands
# Provides utilities for creating composable, parameterized commands

use argument-parser.nu *

# Build a context engineering command with enhanced argument support
def "build context command" [
    command_name: string,         # Name of the command
    command_spec: record,         # Command specification
    executor_script: string       # Path to executor script or inline script
] -> string {
    
    let command_template = $"
# ($command_name)

($command_spec.description)

## Usage

```bash
/($command_name) [arguments]
```

## Arguments

(generate_args_documentation $command_spec)

## Command Implementation

```bash
#!/usr/bin/env nu

# Source the argument parser
source ($nu.current-exe | path dirname | path join \"utils\" \"argument-parser.nu\")

# Get raw arguments
RAW_ARGS = \"($ARGUMENTS)\"

# Check for help flag
if (has help flag $RAW_ARGS) {{
    parse arguments $RAW_ARGS ($command_spec | to json | from json) --help
    exit 0
}}

# Parse arguments
let result = (parse arguments $RAW_ARGS ($command_spec | to json | from json))

if ($result.help_shown) {{
    exit 0
}}

if ($result.error? != null) {{
    print $\"❌ Error: ($result.error)\"
    exit 1
}}

let parsed_args = $result.parsed

# Execute the main command logic
($executor_script)
```"
    
    return $command_template
}

# Generate arguments documentation
def generate_args_documentation [
    command_spec: record          # Command specification
] -> string {
    
    if (($command_spec.arguments? | default [] | length) == 0) {
        return "No arguments required."
    }
    
    mut doc = ""
    
    for spec in $command_spec.arguments {
        let required_mark = if ($spec.required? == true) { " (required)" } else { "" }
        let short_flag = if ($spec.short? != null) { $", -($spec.short)" } else { "" }
        let default_text = if ($spec.default? != null) { $" Default: `($spec.default)`" } else { "" }
        let choices_text = if ($spec.choices? != null) { $" Choices: `($spec.choices | str join ', ')`" } else { "" }
        
        $doc = $doc + $"- `--($spec.name)`($short_flag)($required_mark): ($spec.description)($default_text)($choices_text)\n"
    }
    
    return $doc
}

# Create a unified context command that can handle multiple operations
def "create unified context command" [] -> record {
    
    let command_spec = (create command spec "context" "Unified context engineering command with dynamic operations" [
        (create arg spec "operation" "Operation to perform" --type "string" --short "o" --required true 
         --choices ["generate", "execute", "workflow", "devpod", "analyze"])
        (create arg spec "target" "Target file or feature name" --type "string" --short "t")
        (create arg spec "env" "Target environment" --type "string" --short "e" 
         --choices ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env", "multi"])
        (create arg spec "template" "Template to use" --type "string")
        (create arg spec "examples" "Examples to include (comma-separated)" --type "list")
        (create arg spec "count" "Number of instances (for devpod)" --type "int" --default 1 --min 1 --max 10)
        (create arg spec "validate" "Enable validation" --type "bool" --default false)
        (create arg spec "monitor" "Enable monitoring" --type "bool" --default false)
        (create arg spec "format" "Output format" --type "string" --default "markdown" 
         --choices ["markdown", "json", "yaml"])
        (create arg spec "output" "Output file or directory" --type "string" --short "o")
        (create arg spec "dry-run" "Show what would be done without executing" --type "bool" --default false)
    ] --usage "/context <operation> [target] [options]" --examples [
        "/context generate user-api --env python --template fastapi"
        "/context execute user-api-python.md --validate"
        "/context workflow user-management --env python --validate --monitor"
        "/context devpod --env python --count 3"
        "/context analyze --env typescript --examples dojo"
    ])
    
    return $command_spec
}

# Create environment-specific command specs
def "create env command specs" [] -> record {
    
    let envs = ["python", "typescript", "rust", "go", "nushell"]
    mut env_specs = {}
    
    for env in $envs {
        let spec = (create command spec $"($env)-context" $"Context engineering for ($env) environment" [
            (create arg spec "operation" "Operation to perform" --type "string" --short "o" --required true 
             --choices ["generate", "execute", "devpod", "analyze"])
            (create arg spec "target" "Target file or feature name" --type "string" --short "t")
            (create arg spec "template" "Template to use" --type "string")
            (create arg spec "examples" "Examples to include" --type "list")
            (create arg spec "count" "Number of DevPod instances" --type "int" --default 1 --min 1 --max 10)
            (create arg spec "validate" "Enable validation" --type "bool" --default true)
        ] --usage $"/($env)-context <operation> [target] [options]")
        
        $env_specs = ($env_specs | insert $env $spec)
    }
    
    return $env_specs
}

# Generate template selection logic
def "generate template selector" [
    parsed_args: record           # Parsed arguments
] -> string {
    
    let template_logic = $"
# Dynamic template selection
def select_template [args: record] -> string {{
    
    # Explicit template specified
    if (\"template\" in ($args | columns)) and ($args.template != null) {{
        return $args.template
    }}
    
    # Auto-detect from environment
    if (\"env\" in ($args | columns)) and ($args.env != null) {{
        match $args.env {{
            \"python-env\" => {{ return \"python_prp\" }}
            \"typescript-env\" => {{ return \"typescript_prp\" }}
            \"rust-env\" => {{ return \"rust_prp\" }}
            \"go-env\" => {{ return \"go_prp\" }}
            \"nushell-env\" => {{ return \"nushell_prp\" }}
            \"multi\" => {{ return \"prp_base\" }}
            _ => {{ return \"prp_base\" }}
        }}
    }}
    
    # Auto-detect from target file
    if (\"target\" in ($args | columns)) and ($args.target != null) {{
        let target = $args.target
        if ($target | str ends-with \".py\") {{ return \"python_prp\" }}
        if ($target | str ends-with \".ts\") or ($target | str ends-with \".js\") {{ return \"typescript_prp\" }}
        if ($target | str ends-with \".rs\") {{ return \"rust_prp\" }}
        if ($target | str ends-with \".go\") {{ return \"go_prp\" }}
        if ($target | str ends-with \".nu\") {{ return \"nushell_prp\" }}
    }}
    
    # Default template
    return \"prp_base\"
}}"
    
    return $template_logic
}

# Generate example integration logic
def "generate example integrator" [] -> string {
    
    let integration_logic = $"
# Dynamic example integration
def integrate_examples [args: record, target_env: string] -> list {{
    
    mut examples = []
    
    # Add explicitly requested examples
    if (\"examples\" in ($args | columns)) and ($args.examples != null) {{
        let requested = (parse list arg $args.examples)
        $examples = ($examples | append $requested)
    }}
    
    # Auto-detect relevant examples based on target
    if (\"target\" in ($args | columns)) and ($args.target != null) {{
        let target = $args.target
        
        # If target mentions UI, web, or frontend, include dojo
        if ($target | str contains \"ui\") or ($target | str contains \"web\") or ($target | str contains \"frontend\") {{
            $examples = ($examples | append \"dojo\")
        }}
        
        # If target mentions API, include API examples
        if ($target | str contains \"api\") or ($target | str contains \"rest\") {{
            match $target_env {{
                \"python-env\" => {{ $examples = ($examples | append \"python-api-example\") }}
                \"typescript-env\" => {{ $examples = ($examples | append \"typescript-api-example\") }}
                _ => {{}}
            }}
        }}
        
        # If target mentions auth, include auth examples
        if ($target | str contains \"auth\") or ($target | str contains \"user\") {{
            $examples = ($examples | append \"user-management\")
        }}
    }}
    
    # Environment-specific auto-examples
    match $target_env {{
        \"typescript-env\" => {{
            if (\"dojo\" not-in $examples) {{
                $examples = ($examples | append \"dojo\")
            }}
        }}
        _ => {{}}
    }}
    
    return ($examples | uniq)
}}"
    
    return $integration_logic
}

# Generate validation command builder
def "generate validation builder" [] -> string {
    
    let validation_logic = $"
# Dynamic validation command generation
def build_validation_commands [env: string, validate: bool] -> list {{
    
    if not $validate {{
        return []
    }}
    
    mut commands = []
    
    match $env {{
        \"python-env\" => {{
            $commands = [
                \"cd python-env && devbox shell\",
                \"uv run ruff format src/ tests/\",
                \"uv run ruff check src/ tests/ --fix\",
                \"uv run mypy src/ tests/\",
                \"uv run pytest tests/ -v --cov=src\"
            ]
        }}
        \"typescript-env\" => {{
            $commands = [
                \"cd typescript-env && devbox shell\",
                \"npm run format\",
                \"npm run lint\",
                \"npm run typecheck\",
                \"npm run test\"
            ]
        }}
        \"rust-env\" => {{
            $commands = [
                \"cd rust-env && devbox shell\",
                \"cargo fmt\",
                \"cargo clippy\",
                \"cargo test\"
            ]
        }}
        \"go-env\" => {{
            $commands = [
                \"cd go-env && devbox shell\",
                \"go fmt ./...\",
                \"golangci-lint run\",
                \"go test ./...\"
            ]
        }}
        \"nushell-env\" => {{
            $commands = [
                \"cd nushell-env && devbox shell\",
                \"nu scripts/format-all.nu\",
                \"nu scripts/check-syntax.nu\",
                \"nu scripts/test-all.nu\"
            ]
        }}
        \"multi\" => {{
            $commands = [
                \"nu nushell-env/scripts/validate-all.nu parallel\"
            ]
        }}
        _ => {{}}
    }}
    
    return $commands
}}"
    
    return $validation_logic
}

# Generate operation executor
def "generate operation executor" [] -> string {
    
    let executor_logic = $"
# Main operation executor
def execute_operation [args: record] -> nothing {{
    
    let operation = $args.operation
    
    match $operation {{
        \"generate\" => {{
            execute_generate $args
        }}
        \"execute\" => {{
            execute_prp $args
        }}
        \"workflow\" => {{
            execute_workflow $args
        }}
        \"devpod\" => {{
            execute_devpod $args
        }}
        \"analyze\" => {{
            execute_analyze $args
        }}
        _ => {{
            print $\"❌ Unknown operation: ($operation)\"
            exit 1
        }}
    }}
}}

# Execute generate operation
def execute_generate [args: record] -> nothing {{
    print \"🔧 Generating PRP...\"
    
    let template = (select_template $args)
    let examples = (integrate_examples $args ($args.env? | default \"auto\"))
    
    print $\"📋 Using template: ($template)\"
    if (($examples | length) > 0) {{
        print $\"📚 Including examples: ($examples | str join ', ')\"
    }}
    
    # Implementation continues...
}}

# Execute PRP operation
def execute_prp [args: record] -> nothing {{
    print \"⚙️  Executing PRP...\"
    
    if ($args.validate? | default false) {{
        let validation_commands = (build_validation_commands ($args.env? | default \"auto\") true)
        print $\"✅ Validation enabled: ($validation_commands | length) checks\"
    }}
    
    # Implementation continues...
}}

# Execute workflow operation (generate + execute)
def execute_workflow [args: record] -> nothing {{
    print \"🔄 Running complete workflow...\"
    
    # Generate first
    execute_generate $args
    
    # Then execute
    execute_prp $args
}}

# Execute DevPod operation
def execute_devpod [args: record] -> nothing {{
    let count = ($args.count? | default 1)
    let env = ($args.env? | default \"python-env\")
    
    print $\"🐳 Provisioning ($count) DevPod workspace(s) for ($env)...\"
    
    # Implementation continues...
}}

# Execute analyze operation
def execute_analyze [args: record] -> nothing {{
    let env = ($args.env? | default \"auto\")
    
    print $\"🔍 Analyzing environment: ($env)\"
    
    # Implementation continues...
}}"
    
    return $executor_logic
}

# Create complete enhanced command
def "create enhanced command" [
    command_name: string,         # Command name
    operation_type: string        # Type of operation (unified, env-specific, etc.)
] -> string {
    
    let command_spec = match $operation_type {
        "unified" => (create unified context command)
        _ => (create unified context command)  # Default to unified for now
    }
    
    let template_selector = (generate template selector {})
    let example_integrator = (generate example integrator)
    let validation_builder = (generate validation builder)
    let operation_executor = (generate operation executor)
    
    let full_script = $"
($template_selector)

($example_integrator)

($validation_builder)

($operation_executor)

# Execute main operation
execute_operation $parsed_args
"
    
    return (build context command $command_name $command_spec $full_script)
}

# Export main function
export def main [] {
    print "Context Engineering Command Builder"
    print "Usage: Use functions to create enhanced commands"
}