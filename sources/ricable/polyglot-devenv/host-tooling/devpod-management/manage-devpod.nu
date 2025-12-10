#!/usr/bin/env nu

# Centralized DevPod Management Script
# Handles all DevPod operations for any environment (python, typescript, rust, go, nushell, agentic variants)
# Usage: nu manage-devpod.nu <command> <environment>
# Supports: Standard environments (python, typescript, rust, go, nushell)
#          Agentic environments (agentic-python, agentic-typescript, agentic-rust, agentic-go, agentic-nushell)
#          Evaluation environments (agentic-eval-*)

def main [command?: string, environment?: string] {
    # Show usage if no arguments provided
    if ($command | is-empty) {
        show usage
        return
    }
    
    if ($environment | is-empty) {
        error make {
            msg: "Environment parameter is required"
            help: "Usage: nu manage-devpod.nu <command> <environment>"
        }
    }
    let valid_environments = ["python", "typescript", "rust", "go", "nushell", "agentic-eval-unified", "agentic-eval-claude", "agentic-eval-gemini", "agentic-eval-results", "agentic-python", "agentic-typescript", "agentic-rust", "agentic-go", "agentic-nushell"]
    let valid_commands = ["provision", "connect", "start", "stop", "delete", "sync", "status", "help", "provision-eval"]
    
    if $environment not-in $valid_environments {
        error make {
            msg: $"Invalid environment: ($environment)"
            help: $"Valid environments: ($valid_environments | str join ', ')"
        }
    }
    
    if $command not-in $valid_commands {
        error make {
            msg: $"Invalid command: ($command)"
            help: $"Valid commands: ($valid_commands | str join ', ')"
        }
    }
    
    match $command {
        "provision" => { provision $environment }
        "provision-eval" => { provision_eval $environment }
        "connect" => { connect $environment }
        "start" => { start $environment }
        "stop" => { stop $environment }
        "delete" => { delete $environment }
        "sync" => { sync $environment }
        "status" => { status $environment }
        "help" => { help_command $environment }
        _ => { error make { msg: $"Unknown command: ($command)" } }
    }
}

def provision [environment: string] {
    print $"🚀 Provisioning ($environment) DevPod workspace..."
    
    # Handle agentic evaluation environments
    if ($environment | str starts-with "agentic-eval-") {
        provision_agentic_eval $environment
        return
    }
    
    # Handle agentic environments (agentic-python, agentic-typescript, etc.)
    if ($environment | str starts-with "agentic-") and not ($environment | str starts-with "agentic-eval-") {
        provision_agentic $environment
        return
    }
    
    let script_path = $"../../devpod-automation/scripts/provision-($environment).sh"
    
    if not ($script_path | path exists) {
        error make {
            msg: $"Provisioning script not found: ($script_path)"
            help: "Make sure the devpod-automation scripts are properly set up"
        }
    }
    
    bash $script_path
}

def provision_eval [environment: string, count?: int] {
    let workspace_count = if ($count | is-empty) { 1 } else { $count }
    print $"🧪 Provisioning ($workspace_count) ($environment) agentic evaluation workspace(s)..."
    provision_agentic_eval $environment $workspace_count
}

def provision_agentic_eval [environment: string, count?: int] {
    let workspace_count = if ($count | is-empty) { 1 } else { $count }
    print $"🤖 Setting up ($workspace_count) agentic evaluation workspace(s) for ($environment)..."
    
    # Validate evaluation environment types
    let valid_eval_environments = ["agentic-eval-unified", "agentic-eval-claude", "agentic-eval-gemini", "agentic-eval-results"]
    if $environment not-in $valid_eval_environments {
        error make {
            msg: $"Invalid agentic evaluation environment: ($environment)"
            help: $"Valid evaluation environments: ($valid_eval_environments | str join ', ')"
        }
    }
    
    # Check if DevPod is available
    let devpod_available = try {
        bash -c "command -v devpod"
        true
    } catch {
        false
    }
    
    if not $devpod_available {
        error make {
            msg: "DevPod is not installed or not available in PATH"
            help: "Please install DevPod first: https://devpod.sh/docs/getting-started/install"
        }
    }
    
    # Get the appropriate template path
    let template_path = $"../devpod-automation/templates/($environment)"
    
    if not ($template_path | path exists) {
        error make {
            msg: $"Template not found: ($template_path)"
            help: "Make sure the agentic evaluation templates are properly set up"
        }
    }
    
    # Create workspaces
    for i in 1..$workspace_count {
        let timestamp = (date now | format date "%Y%m%d-%H%M%S")
        let random_suffix = (random uuid | str substring 0..7)
        let workspace_name = $"polyglot-($environment)-($timestamp)-($random_suffix)"
        print $"🔧 Creating workspace ($i)/($workspace_count): ($workspace_name)"
        
        try {
            bash -c $"devpod up ($workspace_name) --ide vscode --devcontainer-path ($template_path)/devcontainer.json"
            print $"✅ Successfully created workspace: ($workspace_name)"
        } catch {
            print $"❌ Failed to create workspace: ($workspace_name)"
        }
    }
    
    print $"🎉 Agentic evaluation setup complete!"
    print $"📊 Environment: ($environment)"
    print $"🔢 Workspaces created: ($workspace_count)"
    print ""
    print "🚀 Next steps:"
    match $environment {
        "agentic-eval-unified" => {
            print "  - Both Claude Code CLI and Gemini CLI are available"
            print "  - Start comparative evaluations across all languages"
            print "  - Access evaluation framework at /workspace/agentic-eval/"
        }
        "agentic-eval-claude" => {
            print "  - Claude Code CLI focused environment"
            print "  - Evaluate Claude's performance on coding tasks"
            print "  - Results stored in /workspace/agentic-eval/results/claude/"
        }
        "agentic-eval-gemini" => {
            print "  - Gemini CLI focused environment"
            print "  - Evaluate Gemini's performance on coding tasks"
            print "  - Results stored in /workspace/agentic-eval/results/gemini/"
        }
        "agentic-eval-results" => {
            print "  - Results analysis and visualization environment"
            print "  - Jupyter Lab available for data analysis"
            print "  - Compare Claude vs Gemini performance metrics"
        }
    }
}

def provision_agentic [environment: string, count?: int] {
    let workspace_count = if ($count | is-empty) { 1 } else { $count }
    print $"🤖 Setting up ($workspace_count) agentic workspace(s) for ($environment)..."
    
    # Validate agentic environment types
    let valid_agentic_environments = ["agentic-python", "agentic-typescript", "agentic-rust", "agentic-go", "agentic-nushell"]
    if $environment not-in $valid_agentic_environments {
        error make {
            msg: $"Invalid agentic environment: ($environment)"
            help: $"Valid agentic environments: ($valid_agentic_environments | str join ', ')"
        }
    }
    
    # Check if DevPod is available
    let devpod_available = try {
        bash -c "command -v devpod"
        true
    } catch {
        false
    }
    
    if not $devpod_available {
        error make {
            msg: "DevPod is not installed or not in PATH"
            help: "Please install DevPod first: https://devpod.sh/docs/getting-started/install"
        }
    }
    
    # Check if template exists
    let template_path = $"../../devpod-automation/templates/($environment)"
    if not ($template_path | path exists) {
        error make {
            msg: $"Template not found: ($template_path)"
            help: "Make sure the agentic AG-UI templates are properly set up"
        }
    }
    
    # Create workspaces
    for i in 1..$workspace_count {
        let timestamp = (date now | format date "%Y%m%d-%H%M%S")
        let random_suffix = (random uuid | str substring 0..7)
        let workspace_name = $"polyglot-($environment)-($timestamp)-($random_suffix)"
        print $"🔧 Creating workspace ($i)/($workspace_count): ($workspace_name)"
        
        try {
            bash -c $"devpod up ($workspace_name) --ide vscode --devcontainer-path ($template_path)/devcontainer.json"
            print $"✅ Successfully created workspace: ($workspace_name)"
        } catch {
            print $"❌ Failed to create workspace: ($workspace_name)"
        }
    }
    
    print $"🎉 Agentic environment setup complete!"
    print $"📊 Environment: ($environment)"
    print $"🔢 Workspaces created: ($workspace_count)"
    print ""
    print "🚀 Next steps:"
    match $environment {
        "agentic-python" => {
            print "  - FastAPI agent server with AG-UI protocol support"
            print "  - Access agent development at /workspace/src/agents/"
            print "  - Start server: uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
        }
        "agentic-typescript" => {
            print "  - Next.js with CopilotKit integration"
            print "  - Access agent development at /workspace/src/agents/"
            print "  - Start development: npm run dev"
        }
        "agentic-rust" => {
            print "  - Tokio-based async agent server"
            print "  - Access agent development at /workspace/src/agents/"
            print "  - Start server: cargo run --bin agent-server"
        }
        "agentic-go" => {
            print "  - Go HTTP server with agent middleware"
            print "  - Access agent development at /workspace/internal/agents/"
            print "  - Start server: go run cmd/agent-server/main.go"
        }
        "agentic-nushell" => {
            print "  - Pipeline-based agent orchestration"
            print "  - Access agent development at /workspace/scripts/agents/"
            print "  - Start server: nu scripts/agents/start-agent.nu"
        }
    }
    print ""
    print "🎛️ AG-UI Features available:"
    print "  • Agentic chat with CopilotKit"
    print "  • Generative UI components"
    print "  • Human-in-the-loop workflows"
    print "  • Shared state management"
    print "  • Tool-based generative UI"
    print "  • Predictive state updates"
}

def connect [environment: string] {
    print $"ℹ️  Connect to ($environment) DevPod workspace:"
    print "Each provision creates a new workspace. Use the provision command to create and connect."
    print $"Or use 'devpod list' to see existing workspaces and connect manually."
}

def start [environment: string] {
    print $"ℹ️  Start ($environment) DevPod workspace:"
    print "Use the provision command to create a new workspace or 'devpod list' to see existing ones."
    print $"Then use 'devpod start <workspace-name>' to start a specific workspace."
}

def stop [environment: string] {
    print $"🛑 Available ($environment) workspaces:"
    
    # Determine search pattern based on environment type
    let search_pattern = $"polyglot-($environment)-"
    
    let workspaces = try {
        bash -c $"devpod list | grep ($search_pattern)"
    } catch {
        ""
    }
    
    if ($workspaces | is-empty) {
        print $"No ($environment) workspaces found"
        print $"Run 'nu manage-devpod.nu provision ($environment)' to create one"
    } else {
        print $workspaces
        print $"Use 'devpod stop <workspace-name>' to stop a specific workspace"
    }
}

def delete [environment: string] {
    print $"🗑️  ($environment) workspaces to delete:"
    
    # Determine search pattern based on environment type
    let search_pattern = $"polyglot-($environment)-"
    
    let workspaces = try {
        bash -c $"devpod list | grep ($search_pattern)"
    } catch {
        ""
    }
    
    if ($workspaces | is-empty) {
        print $"No ($environment) workspaces found"
    } else {
        print $workspaces
        print $"Use 'devpod delete <workspace-name>' to delete a specific workspace"
    }
}

def sync [environment: string] {
    print $"🔄 Sync ($environment) DevPod workspace:"
    print "To sync configuration changes:"
    print "1. Update devbox.json in the dev-env directory"
    print "2. Run the provision command to rebuild the workspace with new configuration"
}

def status [environment: string] {
    print $"📊 ($environment) DevPod workspaces:"
    
    # Determine search pattern based on environment type
    let search_pattern = $"polyglot-($environment)-"
    
    let workspaces = try {
        bash -c $"devpod list | grep ($search_pattern)"
    } catch {
        ""
    }
    
    if ($workspaces | is-empty) {
        print $"No ($environment) workspaces found."
        print $"Run 'nu manage-devpod.nu provision ($environment)' to create one"
    } else {
        print $workspaces
        
        # Show additional info for agentic evaluation environments
        if ($environment | str starts-with "agentic-eval-") {
            print ""
            print "🤖 Agentic Evaluation Environment Info:"
            match $environment {
                "agentic-eval-unified" => {
                    print "  - Comparative evaluation (Claude + Gemini)"
                    print "  - Multi-language support (Python, TypeScript, Rust, Go, Nushell)"
                }
                "agentic-eval-claude" => {
                    print "  - Claude Code CLI focused"
                    print "  - Performance analysis and benchmarking"
                }
                "agentic-eval-gemini" => {
                    print "  - Gemini CLI focused"
                    print "  - Performance analysis and benchmarking"
                }
                "agentic-eval-results" => {
                    print "  - Results analysis and visualization"
                    print "  - Jupyter Lab for data science workflows"
                }
            }
        }
    }
}

def help_command [environment: string] {
    print $"🔧 DevPod Management for ($environment)"
    print ""
    print "Available commands:"
    print "  provision      - Create and provision a new DevPod workspace"
    print "  provision-eval - Create agentic evaluation workspace(s) with count"
    print "  connect        - Show connection instructions"
    print "  start          - Show workspace start instructions"
    print "  stop           - List and stop workspaces"
    print "  delete         - List and delete workspaces"
    print "  sync           - Sync configuration changes"
    print "  status         - Show workspace status"
    print "  help           - Show this help message"
    print ""
    
    if ($environment | str starts-with "agentic-eval-") {
        print "🤖 Agentic Evaluation Commands:"
        print $"  nu manage-devpod.nu provision ($environment)     - Create single evaluation workspace"
        print $"  nu manage-devpod.nu provision-eval ($environment) 3 - Create 3 evaluation workspaces"
        print $"  nu manage-devpod.nu status ($environment)        - Show evaluation workspace status"
        print ""
        print "🔬 Evaluation Features:"
        match $environment {
            "agentic-eval-unified" => {
                print "  - Both Claude Code CLI and Gemini CLI available"
                print "  - Comparative evaluation across Python, TypeScript, Rust, Go, Nushell"
                print "  - Comprehensive evaluation framework at /workspace/agentic-eval/"
            }
            "agentic-eval-claude" => {
                print "  - Claude Code CLI focused environment"
                print "  - Claude performance analysis and benchmarking"
                print "  - Results at /workspace/agentic-eval/results/claude/"
            }
            "agentic-eval-gemini" => {
                print "  - Gemini CLI focused environment"
                print "  - Gemini performance analysis and benchmarking"
                print "  - Results at /workspace/agentic-eval/results/gemini/"
            }
            "agentic-eval-results" => {
                print "  - Results analysis and visualization environment"
                print "  - Jupyter Lab for data science and analytics"
                print "  - Compare Claude vs Gemini performance metrics"
            }
        }
    } else {
        print "Standard Environment Commands:"
        print $"  nu manage-devpod.nu provision ($environment)"
        print $"  nu manage-devpod.nu status ($environment)"
        print $"  nu manage-devpod.nu stop ($environment)"
    }
    
    print ""
    print "Direct devpod commands:"
    print "  devpod list                    - List all workspaces"
    print "  devpod start <workspace-name>  - Start a workspace"
    print "  devpod stop <workspace-name>   - Stop a workspace"
    print "  devpod delete <workspace-name> - Delete a workspace"
}

# Show help if no arguments provided  
def "show usage" [] {
    print "DevPod Management Script with Agentic Evaluation Support"
    print "Usage: nu manage-devpod.nu <command> <environment> [count]"
    print ""
    print "Commands: provision, provision-eval, connect, start, stop, delete, sync, status, help"
    print ""
    print "Standard Environments:"
    print "  python, typescript, rust, go, nushell"
    print ""
    print "🤖 Agentic Evaluation Environments:"
    print "  agentic-eval-unified  - Comparative Claude + Gemini evaluation"
    print "  agentic-eval-claude   - Claude Code CLI focused"
    print "  agentic-eval-gemini   - Gemini CLI focused"
    print "  agentic-eval-results  - Results analysis & visualization"
    print ""
    print "Standard Examples:"
    print "  nu manage-devpod.nu provision python"
    print "  nu manage-devpod.nu status typescript"
    print "  nu manage-devpod.nu help rust"
    print ""
    print "🧪 Agentic Evaluation Examples:"
    print "  nu manage-devpod.nu provision agentic-eval-unified"
    print "  nu manage-devpod.nu provision-eval agentic-eval-claude 3"
    print "  nu manage-devpod.nu status agentic-eval-results"
    print "  nu manage-devpod.nu help agentic-eval-unified"
}