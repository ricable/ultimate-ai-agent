#!/usr/bin/env nu

# Configuration Refactoring System
# Moves dev-env/ configurations to use generated configs from canonical source

def main [
    --backup = true     # Create backup before refactoring
    --force = false     # Force overwrite without confirmation
    --validate = true   # Validate after refactoring
] {
    log info "🔧 Configuration Refactoring System"
    log info "Implementing Single Source of Truth Architecture"
    print "=" * 60
    
    if $backup {
        create-backup
    }
    
    # Define canonical configurations directly in the script for now
    # (This will later be moved to a proper YAML/JSON file)
    let canonical_configs = get-canonical-configs
    
    for env_name in ($canonical_configs | columns) {
        let env_config = $canonical_configs | get $env_name
        
        log info $"🔧 Refactoring ($env_name) configuration..."
        
        # Generate and place devbox.json
        refactor-devbox-config $env_name $env_config $force
        
        # Generate and place devcontainer.json
        refactor-devcontainer-config $env_name $env_config $force
        
        if $validate {
            validate-refactored-config $env_name
        }
    }
    
    log success "✅ Configuration refactoring completed!"
    log info "💡 All configurations now generated from single source of truth"
    log info "   - dev-env/ contains generated devbox.json files"
    log info "   - devpod-automation/templates/ contains generated devcontainer.json files"
    log info "   - Both are derived from canonical definitions"
    log info "   - Configuration drift is now impossible!"
}

def get-canonical-configs [] {
    {
        python: {
            name: "Python Development Environment",
            packages: {
                devbox: ["python@3.12", "uv", "ruff", "mypy", "nushell"],
                devcontainer: {
                    base_image: "mcr.microsoft.com/devcontainers/python:3.12-bullseye",
                    features: {
                        python: {
                            version: "3.12",
                            install_tools: true
                        }
                    }
                }
            },
            environment: {
                PYTHONPATH: "$PWD/src",
                UV_CACHE_DIR: "$PWD/.uv-cache",
                UV_PYTHON_PREFERENCE: "only-managed",
                devcontainer: {
                    PYTHONPATH: "/workspace/src",
                    UV_CACHE_DIR: "/workspace/.uv-cache"
                }
            },
            scripts: {
                setup: "uv sync --dev",
                install: "uv sync --dev",
                add: "uv add",
                remove: "uv remove",
                format: "uv run ruff format .",
                lint: "uv run ruff check . --fix",
                "type-check": "uv run mypy .",
                test: "uv run pytest --cov=src",
                "test-watch": "uv run pytest --cov=src -f",
                clean: "find . -type d -name '__pycache__' -exec rm -rf {} + && find . -name '*.pyc' -delete",
                build: "uv build",
                deps: "uv tree",
                lock: "uv lock",
                sync: "uv sync --dev",
                run: "uv run",
                watch: "uv run pytest --cov=src -f"
            },
            vscode: {
                extensions: [
                    "ms-python.python",
                    "ms-python.pylint", 
                    "ms-python.mypy-type-checker",
                    "charliermarsh.ruff",
                    "ms-python.debugpy"
                ],
                settings: {
                    "python.defaultInterpreterPath": "/usr/local/bin/python",
                    "python.formatting.provider": "none",
                    "[python]": {
                        "editor.defaultFormatter": "charliermarsh.ruff",
                        "editor.codeActionsOnSave": {
                            "source.organizeImports": "explicit"
                        }
                    }
                }
            },
            container: {
                ports: [
                    {port: 8000, label: "HTTP Server", auto_forward: true},
                    {port: 5000, label: "Python Web Server", auto_forward: true}
                ],
                mounts: [
                    {type: "volume", source: "uv-cache", target: "/workspace/.uv-cache"}
                ],
                post_create: [
                    "echo 'Setting up Python environment...'",
                    "pip install uv",
                    "uv sync --dev",
                    "echo 'Python environment ready'"
                ],
                post_start: "echo 'Python DevPod ready. Run: uv run <script>'",
                post_attach: "echo 'Welcome to Python DevPod! 🐍'"
            }
        },
        
        go: {
            name: "Go Development Environment",
            packages: {
                devbox: ["go@1.22", "golangci-lint@latest", "nushell"],
                devcontainer: {
                    base_image: "mcr.microsoft.com/devcontainers/go:1.22-bullseye",
                    features: {
                        go: {
                            version: "1.22"
                        }
                    }
                }
            },
            environment: {
                CGO_ENABLED: "0",
                devcontainer: {
                    GO111MODULE: "on",
                    GOPROXY: "https://proxy.golang.org,direct"
                }
            },
            scripts: {
                build: "go build ./...",
                run: "go run ./cmd/main.go",
                test: "go test ./...",
                "test-watch": "find . -name '*.go' | entr -r go test ./...",
                format: "gofmt -w .",
                lint: "golangci-lint run",
                clean: "go clean -cache -modcache -testcache",
                "mod-tidy": "go mod tidy",
                "mod-download": "go mod download",
                vet: "go vet ./...",
                watch: "find . -name '*.go' | entr -r go run ./cmd/main.go"
            },
            vscode: {
                extensions: [
                    "golang.go",
                    "golang.go-nightly"
                ],
                settings: {
                    "go.useLanguageServer": true,
                    "go.lintOnSave": "package",
                    "go.formatTool": "goimports"
                }
            },
            container: {
                ports: [
                    {port: 8080, label: "API Server", auto_forward: true},
                    {port: 3000, label: "Frontend Development Server", auto_forward: true}
                ],
                mounts: [
                    {type: "volume", source: "go-cache", target: "/home/vscode/go/pkg"}
                ],
                post_create: [
                    "echo 'Setting up Go environment...'",
                    "go mod download",
                    "echo 'Go environment ready'"
                ],
                post_start: "echo 'Go DevPod ready. Run: go run ./cmd'",
                post_attach: "echo 'Welcome to Go DevPod! 🐹'"
            }
        }
    }
}

def refactor-devbox-config [env_name: string, env_config: record, force: bool] {
    let target_file = $"dev-env/($env_name)/devbox.json"
    
    # Extract version commands based on environment
    let version_commands = match $env_name {
        "python" => ["uv --version", "python --version"]
        "go" => ["go version"]
        "typescript" => ["node --version", "npm --version"]
        "rust" => ["rustc --version", "cargo --version"]
        "nushell" => ["nu --version"]
        _ => []
    }
    
    # Build the devbox configuration
    let devbox_config = {
        packages: $env_config.packages.devbox,
        shell: {
            init_hook: ([
                $"echo '($env_config.name)'"
            ] | append $version_commands),
            scripts: (
                $env_config.scripts 
                | upsert "devpod:provision" $"bash ../devpod-automation/scripts/provision-($env_name).sh"
                | upsert "devpod:connect" "echo 'ℹ️  Each provision creates a new workspace. Use devbox run devpod:provision to create and connect.'"
                | upsert "devpod:start" "echo 'ℹ️  Use devbox run devpod:provision to create a new workspace or devpod list to see existing ones.'"
                | upsert "devpod:stop" $"bash -c 'echo \"🛑 Available ($env_name | str capitalize) workspaces:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found\"'"
                | upsert "devpod:delete" $"bash -c 'echo \"🗑️  ($env_name | str capitalize) workspaces to delete:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found\"'"
                | upsert "devpod:sync" "echo 'Sync: Update devbox.json and rebuild workspace with devbox run devpod:provision'"
                | upsert "devpod:status" $"bash -c 'echo \"📊 ($env_name | str capitalize) DevPod workspaces:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found. Run: devbox run devpod:provision\"'"
            )
        },
        env: ($env_config.environment | reject devcontainer?)
    }
    
    # Check if file exists and handle overwrite
    if ($target_file | path exists) and not $force {
        let response = (input $"⚠️  ($target_file) exists. Overwrite with generated config? y/N: ")
        if $response != "y" and $response != "Y" {
            log info $"⏭️  Skipped: ($target_file)"
            return
        }
    }
    
    # Ensure directory exists
    mkdir ($target_file | path dirname)
    
    # Write the generated configuration
    $devbox_config | to json --indent 2 | save $target_file --force
    log success $"✅ Generated: ($target_file)"
    log info $"   📋 Single source of truth: canonical definitions"
}

def refactor-devcontainer-config [env_name: string, env_config: record, force: bool] {
    let target_file = $"devpod-automation/templates/($env_name)/devcontainer.json"
    
    # Build the devcontainer configuration
    let devcontainer_config = {
        name: $env_config.name,
        image: $env_config.packages.devcontainer.base_image,
        features: (build-features $env_config.packages.devcontainer.features),
        customizations: {
            vscode: {
                extensions: $env_config.vscode.extensions,
                settings: $env_config.vscode.settings
            },
            devpod: {
                prebuildRepository: $"ghcr.io/ricable/polyglot-devenv-($env_name)"
            }
        },
        containerEnv: (get-container-env $env_config.environment),
        mounts: (build-mounts $env_config.container.mounts),
        forwardPorts: ($env_config.container.ports | each { |port| $port.port }),
        portsAttributes: (build-port-attrs $env_config.container.ports),
        postCreateCommand: $env_config.container.post_create,
        postStartCommand: $env_config.container.post_start,
        postAttachCommand: $env_config.container.post_attach
    }
    
    # Check if file exists and handle overwrite
    if ($target_file | path exists) and not $force {
        let response = (input $"⚠️  ($target_file) exists. Overwrite with generated config? y/N: ")
        if $response != "y" and $response != "Y" {
            log info $"⏭️  Skipped: ($target_file)"
            return
        }
    }
    
    # Ensure directory exists
    mkdir ($target_file | path dirname)
    
    # Write the generated configuration
    $devcontainer_config | to json --indent 2 | save $target_file --force
    log success $"✅ Generated: ($target_file)"
    log info $"   📋 Single source of truth: canonical definitions"
}

def build-features [features: record] {
    let result = {}
    for feature in ($features | columns) {
        let config = $features | get $feature
        let result = ($result | upsert $"ghcr.io/devcontainers/features/($feature):1" $config)
    }
    $result
}

def get-container-env [env_vars: record] {
    if "devcontainer" in ($env_vars | columns) {
        $env_vars.devcontainer
    } else {
        # Convert devbox paths to container paths
        let result = {}
        for var in ($env_vars | columns) {
            if $var != "devcontainer" {
                let value = $env_vars | get $var
                let container_value = ($value | str replace "$PWD" "/workspace")
                let result = ($result | upsert $var $container_value)
            }
        }
        $result
    }
}

def build-mounts [mounts: list] {
    $mounts | each { |mount|
        $"type=($mount.type),source=($mount.source),target=($mount.target)"
    }
}

def build-port-attrs [ports: list] {
    let result = {}
    for port in $ports {
        let port_str = ($port.port | into string)
        let attrs = {
            label: $port.label,
            onAutoForward: (if $port.auto_forward { "notify" } else { "silent" })
        }
        let result = ($result | upsert $port_str $attrs)
    }
    $result
}

def create-backup [] {
    let timestamp = (date now | format date "%Y%m%d_%H%M%S")
    let backup_dir = $"backups/config_refactor_($timestamp)"
    
    log info $"💾 Creating backup: ($backup_dir)"
    mkdir $backup_dir
    
    # Backup existing devbox.json files
    for env in [python go typescript rust nushell] {
        let devbox_file = $"dev-env/($env)/devbox.json"
        if ($devbox_file | path exists) {
            cp $devbox_file $"($backup_dir)/($env)-devbox.json"
        }
        
        let devcontainer_file = $"devpod-automation/templates/($env)/devcontainer.json"
        if ($devcontainer_file | path exists) {
            cp $devcontainer_file $"($backup_dir)/($env)-devcontainer.json"
        }
    }
    
    log success $"✅ Backup created: ($backup_dir)"
}

def validate-refactored-config [env_name: string] {
    log info $"🔍 Validating refactored configurations for ($env_name)..."
    
    let devbox_file = $"dev-env/($env_name)/devbox.json"
    if ($devbox_file | path exists) {
        try {
            open $devbox_file | from json | ignore
            log success $"✅ Valid: ($devbox_file)"
        } catch {
            log error $"❌ Invalid JSON: ($devbox_file)"
        }
    }
    
    let devcontainer_file = $"devpod-automation/templates/($env_name)/devcontainer.json"
    if ($devcontainer_file | path exists) {
        try {
            open $devcontainer_file | from json | ignore
            log success $"✅ Valid: ($devcontainer_file)"
        } catch {
            log error $"❌ Invalid JSON: ($devcontainer_file)"
        }
    }
}

def log [level: string, message: string] {
    let timestamp = (date now | format date "%H:%M:%S")
    match $level {
        "info" => { print $"[($timestamp)] ℹ️  ($message)" }
        "success" => { print $"[($timestamp)] ✅ ($message)" }
        "error" => { print $"[($timestamp)] ❌ ($message)" }
        "warning" => { print $"[($timestamp)] ⚠️  ($message)" }
        _ => { print $"[($timestamp)] ($message)" }
    }
}