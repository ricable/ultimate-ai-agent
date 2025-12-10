#!/usr/bin/env nu

# DevPod Provisioning Orchestrator for Polyglot Development Environment
# Automates the creation and management of language-specific DevPod workspaces

use ../../nushell-env/common.nu *

# Main provisioning command
def main [
    --language(-l): string = "python"  # Language environment to provision (python, typescript, rust, go, nushell, full-stack)
    --ide: string = "vscode"           # IDE to use (vscode, openvscode, goland, none)
    --reset                            # Reset existing workspace
    --no-start                         # Don't start workspace after creation
    --prebuild                         # Build and cache the workspace
    --help(-h)                         # Show help
] {
    if $help {
        show_help
        return
    }

    log info $"Starting DevPod provisioning for ($language) environment..."
    
    # Validate prerequisites
    check_prerequisites
    
    # Setup Docker provider if needed
    ensure_docker_provider
    
    # Get workspace configuration
    let workspace_config = get_workspace_config $language
    
    # Generate devcontainer.json from devbox environment
    generate_devcontainer $language
    
    # Create or update workspace
    if $reset {
        reset_workspace $workspace_config.name
    }
    
    provision_workspace $workspace_config $ide
    
    if not $no_start {
        start_workspace $workspace_config.name $ide
    }
    
    if $prebuild {
        prebuild_workspace $workspace_config.name
    }
    
    show_workspace_info $workspace_config.name
    
    log success $"DevPod workspace ($workspace_config.name) provisioned successfully!"
}

# Check all prerequisites
def check_prerequisites [] {
    log info "Checking prerequisites..."
    
    # Check Docker
    try {
        docker --version | ignore
        log success "✓ Docker is available"
    } catch {
        log error "✗ Docker not found. Please install Docker and ensure it's running."
        exit 1
    }
    
    # Check DevPod
    try {
        devpod --version | ignore
        log success "✓ DevPod is available"
    } catch {
        log warning "DevPod not found. Installing..."
        install_devpod
    }
    
    # Check Devbox
    try {
        devbox --version | ignore
        log success "✓ Devbox is available"
    } catch {
        log error "✗ Devbox not found. This system requires Devbox for devcontainer generation."
        exit 1
    }
}

# Install DevPod automatically
def install_devpod [] {
    log info "Installing DevPod..."
    
    let os = (sys).host.name
    
    match $os {
        "Darwin" => {
            if (which brew | is-empty) {
                log error "Homebrew required for macOS installation"
                exit 1
            }
            brew install --cask devpod
        }
        "Linux" => {
            let arch = (uname -m)
            let url = match $arch {
                "x86_64" => "https://github.com/loft-sh/devpod/releases/latest/download/devpod-linux-amd64"
                "aarch64" => "https://github.com/loft-sh/devpod/releases/latest/download/devpod-linux-arm64"
                _ => {
                    log error $"Unsupported architecture: ($arch)"
                    exit 1
                }
            }
            
            curl -L -o devpod $url
            sudo install -c -m 0755 devpod /usr/local/bin
            rm -f devpod
        }
        _ => {
            log error $"Unsupported operating system: ($os)"
            exit 1
        }
    }
    
    log success "DevPod installed successfully"
}

# Ensure Docker provider is configured
def ensure_docker_provider [] {
    log info "Configuring Docker provider..."
    
    # Check if Docker provider exists
    let providers = (devpod provider list | lines)
    
    if not ($providers | any {|line| $line | str contains "docker"}) {
        log info "Adding Docker provider..."
        devpod provider add docker
    }
    
    # Set as default if not already set
    try {
        devpod provider use docker
        log success "✓ Docker provider configured"
    } catch {
        log warning "Could not set Docker as default provider, but it's available"
    }
}

# Get workspace configuration for language
def get_workspace_config [language: string] {
    let configs = {
        python: {
            name: "polyglot-python-devpod"
            path: "../python-env"
            description: "Python development environment with uv and testing tools"
            ports: [8000, 5000]
        }
        typescript: {
            name: "polyglot-typescript-devpod"
            path: "../typescript-env"
            description: "TypeScript/Node.js development environment"
            ports: [3000, 8080, 5173]
        }
        rust: {
            name: "polyglot-rust-devpod"
            path: "../rust-env"
            description: "Rust development environment with cargo and clippy"
            ports: [8000, 3030]
        }
        go: {
            name: "polyglot-go-devpod"
            path: "../go-env"
            description: "Go development environment with toolchain"
            ports: [8080, 3000]
        }
        nushell: {
            name: "polyglot-nushell-devpod"
            path: "../nushell-env"
            description: "Nushell scripting and automation environment"
            ports: []
        }
        "full-stack": {
            name: "polyglot-full-devpod"
            path: ".."
            description: "Complete polyglot development environment"
            ports: [3000, 5000, 8000, 8080, 5173]
        }
    }
    
    if $language not-in ($configs | columns) {
        log error $"Unknown language: ($language). Available: (($configs | columns) | str join ', ')"
        exit 1
    }
    
    $configs | get $language
}

# Generate devcontainer.json from devbox environment
def generate_devcontainer [language: string] {
    log info $"Generating devcontainer.json for ($language)..."
    
    let workspace_config = get_workspace_config $language
    let env_path = $workspace_config.path
    
    # Navigate to environment directory
    cd $env_path
    
    # Generate base devcontainer.json using devbox
    try {
        devbox generate devcontainer --force
        log success "✓ Base devcontainer.json generated from devbox"
    } catch {
        log warning "Could not generate from devbox, using template"
        # Fallback to template
        nu ($"($env.PWD)/devpod-automation/scripts/devpod-generate.nu" | path expand) --language $language --output ".devcontainer/devcontainer.json"
    }
    
    # Enhance with DevPod-specific customizations
    enhance_devcontainer $language
    
    cd ..
}

# Enhance devcontainer.json with DevPod customizations
def enhance_devcontainer [language: string] {
    let devcontainer_path = ".devcontainer/devcontainer.json"
    
    if not ($devcontainer_path | path exists) {
        log error "devcontainer.json not found"
        return
    }
    
    log info "Enhancing devcontainer.json with DevPod customizations..."
    
    # Read existing devcontainer.json
    let devcontainer = (open $devcontainer_path)
    
    # Get workspace config for ports and settings
    let workspace_config = get_workspace_config $language
    
    # Add DevPod-specific enhancements
    let enhanced = ($devcontainer | merge {
        customizations: {
            devpod: {
                prebuildRepository: $"ghcr.io/ricable/polyglot-devenv-($language)"
            }
            vscode: {
                extensions: (get_vscode_extensions $language)
                settings: (get_vscode_settings $language)
            }
        }
        forwardPorts: $workspace_config.ports
        portsAttributes: (get_port_attributes $workspace_config.ports)
        postCreateCommand: (get_post_create_command $language)
        postStartCommand: (get_post_start_command $language)
    })
    
    # Save enhanced devcontainer.json
    $enhanced | to json | save $devcontainer_path --force
    
    log success "✓ devcontainer.json enhanced with DevPod customizations"
}

# Get VS Code extensions for language
def get_vscode_extensions [language: string] {
    let base_extensions = [
        "ms-vscode.vscode-json"
        "github.vscode-github-actions"
        "github.vscode-pull-request-github"
        "ms-vscode.vscode-typescript-next"
    ]
    
    let language_extensions = match $language {
        "python" => [
            "ms-python.python"
            "ms-python.pylint"
            "ms-python.mypy-type-checker"
            "charliermarsh.ruff"
        ]
        "typescript" => [
            "bradlc.vscode-tailwindcss"
            "esbenp.prettier-vscode"
            "ms-vscode.vscode-typescript-next"
        ]
        "rust" => [
            "rust-lang.rust-analyzer"
            "vadimcn.vscode-lldb"
            "serayuzgur.crates"
        ]
        "go" => [
            "golang.go"
            "golang.go-nightly"
        ]
        "nushell" => [
            "thenuprojectcontributors.vscode-nushell-lang"
        ]
        "full-stack" => [
            "ms-python.python"
            "rust-lang.rust-analyzer" 
            "golang.go"
            "thenuprojectcontributors.vscode-nushell-lang"
            "charliermarsh.ruff"
            "esbenp.prettier-vscode"
        ]
        _ => []
    }
    
    $base_extensions | append $language_extensions
}

# Get VS Code settings for language
def get_vscode_settings [language: string] {
    let base_settings = {
        "editor.formatOnSave": true
        "editor.rulers": [88, 100, 120]
        "files.trimTrailingWhitespace": true
        "git.autofetch": true
    }
    
    let language_settings = match $language {
        "python" => {
            "python.defaultInterpreterPath": "/opt/venv/bin/python"
            "python.formatting.provider": "none"
            "[python]": {
                "editor.defaultFormatter": "charliermarsh.ruff"
                "editor.codeActionsOnSave": {
                    "source.organizeImports": true
                }
            }
        }
        "typescript" => {
            "typescript.preferences.importModuleSpecifier": "relative"
            "[typescript]": {
                "editor.defaultFormatter": "esbenp.prettier-vscode"
            }
            "[javascript]": {
                "editor.defaultFormatter": "esbenp.prettier-vscode"
            }
        }
        "rust" => {
            "rust-analyzer.cargo.autoreload": true
            "rust-analyzer.check.command": "clippy"
        }
        "go" => {
            "go.useLanguageServer": true
            "go.lintOnSave": "package"
            "go.formatTool": "goimports"
        }
        _ => {}
    }
    
    $base_settings | merge $language_settings
}

# Get port attributes for forwarded ports
def get_port_attributes [ports: list] {
    let attributes = {}
    
    for port in $ports {
        $attributes = ($attributes | merge {
            ($port | into string): {
                label: (get_port_label $port)
                onAutoForward: "notify"
            }
        })
    }
    
    $attributes
}

# Get label for port
def get_port_label [port: int] {
    match $port {
        3000 => "Frontend Development Server"
        5000 => "Python Web Server"
        8000 => "HTTP Server"
        8080 => "API Server"
        5173 => "Vite Development Server"
        _ => $"Service Port ($port)"
    }
}

# Get post-create command for language
def get_post_create_command [language: string] {
    match $language {
        "python" => "uv sync --dev && echo 'Python environment ready'"
        "typescript" => "npm install && echo 'TypeScript environment ready'"
        "rust" => "cargo fetch && echo 'Rust environment ready'"
        "go" => "go mod download && echo 'Go environment ready'"
        "nushell" => "nu scripts/setup.nu && echo 'Nushell environment ready'"
        "full-stack" => "echo 'Setting up polyglot environment...' && (cd python-env && uv sync --dev) && (cd typescript-env && npm install) && (cd rust-env && cargo fetch) && (cd go-env && go mod download) && echo 'Polyglot environment ready'"
        _ => "echo 'Environment ready'"
    }
}

# Get post-start command for language
def get_post_start_command [language: string] {
    match $language {
        "python" => "echo 'Python DevPod ready. Run: uv run <script>'"
        "typescript" => "echo 'TypeScript DevPod ready. Run: npm run dev'"
        "rust" => "echo 'Rust DevPod ready. Run: cargo run'"
        "go" => "echo 'Go DevPod ready. Run: go run ./cmd'"
        "nushell" => "echo 'Nushell DevPod ready. Scripts available in: scripts/'"
        "full-stack" => "echo 'Polyglot DevPod ready. All languages available.'"
        _ => "echo 'DevPod ready'"
    }
}

# Reset existing workspace
def reset_workspace [workspace_name: string] {
    log info $"Resetting workspace ($workspace_name)..."
    
    try {
        devpod delete $workspace_name --force
        log success $"✓ Workspace ($workspace_name) reset"
    } catch {
        log info $"Workspace ($workspace_name) didn't exist or already deleted"
    }
}

# Provision the workspace
def provision_workspace [workspace_config: record, ide: string] {
    let workspace_name = $workspace_config.name
    let workspace_path = $workspace_config.path
    
    log info $"Provisioning workspace ($workspace_name)..."
    
    # Create workspace from local path
    try {
        devpod up $workspace_path --id $workspace_name --ide none
        log success $"✓ Workspace ($workspace_name) created successfully"
    } catch {
        log error $"Failed to create workspace ($workspace_name)"
        exit 1
    }
}

# Start workspace with specified IDE
def start_workspace [workspace_name: string, ide: string] {
    log info $"Starting workspace ($workspace_name) with IDE ($ide)..."
    
    try {
        devpod up $workspace_name --ide $ide
        log success $"✓ Workspace ($workspace_name) started with ($ide)"
    } catch {
        log error $"Failed to start workspace ($workspace_name) with IDE ($ide)"
        exit 1
    }
}

# Prebuild workspace
def prebuild_workspace [workspace_name: string] {
    log info $"Prebuilding workspace ($workspace_name)..."
    
    try {
        devpod build $workspace_name
        log success $"✓ Workspace ($workspace_name) prebuilt successfully"
    } catch {
        log warning $"Could not prebuild workspace ($workspace_name)"
    }
}

# Show workspace information
def show_workspace_info [workspace_name: string] {
    log info $"Workspace ($workspace_name) Information:"
    
    try {
        let status = (devpod list | lines | where {|line| $line | str contains $workspace_name})
        
        if ($status | length) > 0 {
            print $"Status: ($status | first)"
        }
        
        print $"SSH Access: ssh ($workspace_name).devpod"
        print $"Direct SSH: devpod ssh ($workspace_name)"
        print $"Stop: devpod stop ($workspace_name)"
        print $"Delete: devpod delete ($workspace_name)"
        
    } catch {
        log warning "Could not retrieve workspace status"
    }
}

# Show help information
def show_help [] {
    print "DevPod Provisioning Orchestrator for Polyglot Development Environment"
    print ""
    print "USAGE:"
    print "    nu devpod-provision.nu [OPTIONS]"
    print ""
    print "OPTIONS:"
    print "    -l, --language <LANGUAGE>    Language environment (python, typescript, rust, go, nushell, full-stack) [default: python]"
    print "    --ide <IDE>                  IDE to use (vscode, openvscode, goland, none) [default: vscode]"
    print "    --reset                      Reset existing workspace"
    print "    --no-start                   Don't start workspace after creation"
    print "    --prebuild                   Build and cache the workspace"
    print "    -h, --help                   Show this help"
    print ""
    print "EXAMPLES:"
    print "    nu devpod-provision.nu --language python --ide vscode"
    print "    nu devpod-provision.nu --language full-stack --reset --prebuild"
    print "    nu devpod-provision.nu --language typescript --ide openvscode --no-start"
    print ""
    print "AVAILABLE LANGUAGES:"
    print "    python      - Python 3.12 with uv and testing tools"
    print "    typescript  - Node.js 20 with TypeScript and frontend tools"
    print "    rust        - Rust toolchain with cargo and clippy"
    print "    go          - Go 1.22 with standard toolchain"
    print "    nushell     - Nushell with automation and scripting tools"
    print "    full-stack  - All languages combined in one workspace"
}