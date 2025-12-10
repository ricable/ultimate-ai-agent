#!/usr/bin/env nu

# DevContainer Generation Script for Polyglot Development Environment
# Generates devcontainer.json files from devbox environments or templates

use ../../nushell-env/common.nu *
source ../config.nu

# Main generation command
def main [
    --language(-l): string = "python"      # Language environment to generate for
    --output(-o): string = "devcontainer.json"  # Output file path
    --template(-t)                         # Use template instead of devbox generation
    --merge(-m): string                    # Merge with existing devcontainer.json file
    --help(-h)                            # Show help
] {
    if $help {
        show_help
        return
    }

    log info $"Generating devcontainer.json for ($language)..."
    
    let devcontainer = if $template {
        generate_from_template $language
    } else {
        generate_from_devbox $language
    }
    
    let final_devcontainer = if $merge != "" {
        merge_with_existing $devcontainer $merge
    } else {
        $devcontainer
    }
    
    # Ensure output directory exists
    let output_dir = ($output | path dirname)
    if $output_dir != "." and $output_dir != "" {
        mkdir $output_dir
    }
    
    # Save the devcontainer.json
    $final_devcontainer | to json --indent 4 | save $output --force
    
    log success $"devcontainer.json generated successfully: ($output)"
    
    # Validate the generated file
    validate_devcontainer $output
}

# Generate devcontainer.json from devbox environment
def generate_from_devbox [language: string] {
    log info "Generating from devbox environment..."
    
    let env_path = get_env_path $language
    let devbox_config_path = $"($env_path)/devbox.json"
    
    if not ($devbox_config_path | path exists) {
        log error $"devbox.json not found at ($devbox_config_path)"
        exit 1
    }
    
    # Read devbox configuration
    let devbox_config = (open $devbox_config_path)
    
    # Generate base devcontainer from devbox packages
    let base_image = get_base_image_from_devbox $devbox_config $language
    let features = get_features_from_devbox $devbox_config $language
    
    # Create devcontainer configuration
    {
        name: (get_container_name $language)
        image: $base_image
        features: $features
        customizations: (get_customizations $language)
        containerEnv: (get_container_env $language)
        mounts: (get_mounts $language)
        forwardPorts: (get_forward_ports $language)
        portsAttributes: (get_port_attributes_for_ports (get_forward_ports $language))
        postCreateCommand: (get_post_create_command $language)
        postStartCommand: (get_post_start_command $language)
        postAttachCommand: (get_post_attach_command $language)
        workspaceFolder: "/workspace"
        workspaceMount: "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
        remoteUser: "vscode"
        updateRemoteUserUID: true
    }
}

# Generate devcontainer.json from template
def generate_from_template [language: string] {
    log info "Generating from template..."
    
    let template_path = $"($config.templates)/($language)/devcontainer.json"
    
    if not ($template_path | path exists) {
        log warning $"Template not found: ($template_path), using base template"
        let template_path = $"($config.templates)/base/devcontainer.json"
        
        if not ($template_path | path exists) {
            # Generate base template if it doesn't exist
            generate_base_template $language
        } else {
            open $template_path
        }
    } else {
        open $template_path
    }
}

# Get environment path for language
def get_env_path [language: string] {
    let base_path = $"($config.output_dir)/($language)-env"
    if not ($base_path | path exists) {
        mkdir $base_path
    }
    $base_path
}

# Get base image from devbox packages
def get_base_image_from_devbox [devbox_config: record, language: string] {
    let packages = $devbox_config.packages
    
    match $language {
        "python" => {
            let python_version = ($packages | where {|pkg| $pkg | str contains "python@"} | first | default "python@3.12")
            let version = ($python_version | str replace "python@" "")
            $"mcr.microsoft.com/devcontainers/python:($version)-bullseye"
        }
        "typescript" => {
            let node_version = ($packages | where {|pkg| $pkg | str contains "nodejs@"} | first | default "nodejs@20")
            let version = ($node_version | str replace "nodejs@" "")
            $"mcr.microsoft.com/devcontainers/typescript-node:($version)-bullseye"
        }
        "rust" => "mcr.microsoft.com/devcontainers/rust:1-bullseye"
        "go" => {
            let go_version = ($packages | where {|pkg| $pkg | str contains "go@"} | first | default "go@1.22")
            let version = ($go_version | str replace "go@" "")
            $"mcr.microsoft.com/devcontainers/go:($version)-bullseye"
        }
        "nushell" => "mcr.microsoft.com/devcontainers/base:ubuntu-22.04"
        "full-stack" => "mcr.microsoft.com/devcontainers/universal:2-linux"
        _ => "mcr.microsoft.com/devcontainers/base:ubuntu-22.04"
    }
}

# Get DevContainer features from devbox packages
def get_features_from_devbox [devbox_config: record, language: string] {
    let base_features = {
        "ghcr.io/devcontainers/features/common-utils:2": {
            "installZsh": true
            "username": "vscode"
            "userUid": "1000"
            "userGid": "1000"
        }
        "ghcr.io/devcontainers/features/git:1": {
            "version": "latest"
            "ppa": true
        }
        "ghcr.io/devcontainers/features/github-cli:1": {
            "version": "latest"
        }
        "ghcr.io/devcontainers/features/docker-in-docker:2": {
            "version": "latest"
            "enableNonRootDocker": true
        }
    }
    
    let language_features = match $language {
        "python" => {
            "ghcr.io/devcontainers/features/python:1": {
                "version": "3.12"
                "installTools": true
            }
        }
        "typescript" => {
            "ghcr.io/devcontainers/features/node:1": {
                "version": "20"
                "nodeGypDependencies": true
            }
        }
        "rust" => {
            "ghcr.io/devcontainers/features/rust:1": {
                "version": "latest"
                "profile": "default"
            }
        }
        "go" => {
            "ghcr.io/devcontainers/features/go:1": {
                "version": "1.22"
            }
        }
        "nushell" => {}
        "full-stack" => {
            "ghcr.io/devcontainers/features/python:1": {
                "version": "3.12"
                "installTools": true
            }
            "ghcr.io/devcontainers/features/node:1": {
                "version": "20"
                "nodeGypDependencies": true
            }
            "ghcr.io/devcontainers/features/rust:1": {
                "version": "latest"
                "profile": "default"
            }
            "ghcr.io/devcontainers/features/go:1": {
                "version": "1.22"
            }
        }
        _ => {}
    }
    
    $base_features | merge $language_features
}

# Get container name
def get_container_name [language: string] {
    match $language {
        "python" => "Polyglot Python Development"
        "typescript" => "Polyglot TypeScript Development"
        "rust" => "Polyglot Rust Development"
        "go" => "Polyglot Go Development"
        "nushell" => "Polyglot Nushell Development"
        "full-stack" => "Polyglot Full-Stack Development"
        _ => $"Polyglot ($language | str title-case) Development"
    }
}

# Get customizations for language
def get_customizations [language: string] {
    {
        vscode: {
            extensions: (get_vscode_extensions $language)
            settings: (get_vscode_settings $language)
        }
        devpod: {
            prebuildRepository: $"ghcr.io/ricable/polyglot-devenv-($language)"
        }
    }
}

# Get VS Code extensions for language (imported from main script)
def get_vscode_extensions [language: string] {
    let base_extensions = [
        "ms-vscode.vscode-json"
        "github.vscode-github-actions"
        "github.vscode-pull-request-github"
        "eamodio.gitlens"
        "ms-vscode.vscode-typescript-next"
    ]
    
    let language_extensions = match $language {
        "python" => [
            "ms-python.python"
            "ms-python.pylint"
            "ms-python.mypy-type-checker"
            "charliermarsh.ruff"
            "ms-python.debugpy"
        ]
        "typescript" => [
            "bradlc.vscode-tailwindcss"
            "esbenp.prettier-vscode"
            "ms-vscode.vscode-typescript-next"
            "dbaeumer.vscode-eslint"
        ]
        "rust" => [
            "rust-lang.rust-analyzer"
            "vadimcn.vscode-lldb"
            "serayuzgur.crates"
            "tamasfe.even-better-toml"
        ]
        "go" => [
            "golang.go"
            "golang.go-nightly"
        ]
        "nushell" => [
            "thenuprojectcontributors.vscode-nushell-lang"
            "mkhl.direnv"
        ]
        "full-stack" => [
            "ms-python.python"
            "rust-lang.rust-analyzer" 
            "golang.go"
            "thenuprojectcontributors.vscode-nushell-lang"
            "charliermarsh.ruff"
            "esbenp.prettier-vscode"
            "vadimcn.vscode-lldb"
            "serayuzgur.crates"
        ]
        _ => []
    }
    
    $base_extensions | append $language_extensions
}

# Get VS Code settings for language (imported from main script)  
def get_vscode_settings [language: string] {
    let base_settings = {
        "editor.formatOnSave": true
        "editor.rulers": [88, 100, 120]
        "files.trimTrailingWhitespace": true
        "git.autofetch": true
        "terminal.integrated.defaultProfile.linux": "zsh"
    }
    
    let language_settings = match $language {
        "python" => {
            "python.defaultInterpreterPath": "/usr/local/bin/python"
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
            "rust-analyzer.cargo.features": "all"
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

# Get container environment variables
def get_container_env [language: string] {
    let base_env = {
        "TERM": "xterm-256color"
        "COLORTERM": "truecolor"
    }
    
    let language_env = match $language {
        "python" => {
            "PYTHONPATH": "/workspace/src"
            "UV_CACHE_DIR": "/workspace/.uv-cache"
            "UV_PYTHON_PREFERENCE": "only-managed"
        }
        "typescript" => {
            "NODE_ENV": "development"
            "NPM_CONFIG_UPDATE_NOTIFIER": "false"
        }
        "rust" => {
            "RUST_BACKTRACE": "1"
            "CARGO_TARGET_DIR": "/workspace/target"
        }
        "go" => {
            "GO111MODULE": "on"
            "GOPROXY": "https://proxy.golang.org,direct"
        }
        "nushell" => {}
        "full-stack" => {
            "PYTHONPATH": "/workspace/python-env/src"
            "NODE_ENV": "development"
            "RUST_BACKTRACE": "1"
            "GO111MODULE": "on"
        }
        _ => {}
    }
    
    $base_env | merge $language_env
}

# Get mounts for the container
def get_mounts [language: string] {
    let base_mounts = [
        "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"
        "type=volume,source=vscode-server-extensions,target=/home/vscode/.vscode-server/extensions"
    ]
    
    let language_mounts = match $language {
        "python" => [
            "type=volume,source=uv-cache,target=/workspace/.uv-cache"
        ]
        "typescript" => [
            "type=volume,source=npm-cache,target=/home/vscode/.npm"
            "type=volume,source=node-modules,target=/workspace/node_modules"
        ]
        "rust" => [
            "type=volume,source=cargo-cache,target=/home/vscode/.cargo"
            "type=volume,source=cargo-target,target=/workspace/target"
        ]
        "go" => [
            "type=volume,source=go-cache,target=/home/vscode/go/pkg"
        ]
        _ => []
    }
    
    $base_mounts | append $language_mounts
}

# Get ports to forward
def get_forward_ports [language: string] {
    match $language {
        "python" => [8000, 5000]
        "typescript" => [3000, 8080, 5173]
        "rust" => [8000, 3030]
        "go" => [8080, 3000]
        "nushell" => []
        "full-stack" => [3000, 5000, 8000, 8080, 5173]
        _ => []
    }
}

# Get port attributes for ports
def get_port_attributes_for_ports [ports: list] {
    let attributes = {}
    
    for port in $ports {
        let attributes = ($attributes | merge {
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
        3030 => "Rust Web Server"
        _ => $"Service Port ($port)"
    }
}

# Get post-create command
def get_post_create_command [language: string] {
    match $language {
        "python" => [
            "echo 'Setting up Python environment...'",
            "pip install uv",
            "uv sync --dev",
            "echo 'Python environment ready'"
        ]
        "typescript" => [
            "echo 'Setting up TypeScript environment...'",
            "npm install",
            "echo 'TypeScript environment ready'"
        ]
        "rust" => [
            "echo 'Setting up Rust environment...'",
            "cargo fetch",
            "echo 'Rust environment ready'"
        ]
        "go" => [
            "echo 'Setting up Go environment...'",
            "go mod download",
            "echo 'Go environment ready'"
        ]
        "nushell" => [
            "echo 'Setting up Nushell environment...'",
            "curl -fsSL https://get.jetify.com/devbox | bash",
            "echo 'Nushell environment ready'"
        ]
        "full-stack" => [
            "echo 'Setting up polyglot environment...'",
            "(cd python-env && pip install uv && uv sync --dev)",
            "(cd typescript-env && npm install)",
            "(cd rust-env && cargo fetch)",
            "(cd go-env && go mod download)",
            "echo 'Polyglot environment ready'"
        ]
        _ => ["echo 'Environment ready'"]
    }
}

# Get post-start command
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

# Get post-attach command
def get_post_attach_command [language: string] {
    match $language {
        "python" => "echo 'Welcome to Python DevPod! 🐍'"
        "typescript" => "echo 'Welcome to TypeScript DevPod! 📘'"
        "rust" => "echo 'Welcome to Rust DevPod! 🦀'"
        "go" => "echo 'Welcome to Go DevPod! 🐹'"
        "nushell" => "echo 'Welcome to Nushell DevPod! 🐚'"
        "full-stack" => "echo 'Welcome to Polyglot DevPod! 🚀'"
        _ => "echo 'Welcome to DevPod!'"
    }
}

# Generate base template if none exists
def generate_base_template [language: string] {
    {
        name: (get_container_name $language)
        image: "mcr.microsoft.com/devcontainers/base:ubuntu-22.04"
        features: (get_features_from_devbox {} $language)
        customizations: (get_customizations $language)
        containerEnv: (get_container_env $language)
        mounts: (get_mounts $language)
        forwardPorts: (get_forward_ports $language)
        portsAttributes: (get_port_attributes_for_ports (get_forward_ports $language))
        postCreateCommand: (get_post_create_command $language)
        postStartCommand: (get_post_start_command $language)
        postAttachCommand: (get_post_attach_command $language)
        workspaceFolder: "/workspace"
        workspaceMount: "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
        remoteUser: "vscode"
        updateRemoteUserUID: true
    }
}

# Merge with existing devcontainer.json
def merge_with_existing [new_config: record, existing_file: string] {
    if not ($existing_file | path exists) {
        log warning $"Existing file not found: ($existing_file), using new configuration"
        return $new_config
    }
    
    log info $"Merging with existing configuration: ($existing_file)"
    
    let existing_config = (open $existing_file)
    $existing_config | merge $new_config
}

# Validate generated devcontainer.json
def validate_devcontainer [file_path: string] {
    log info "Validating generated devcontainer.json..."
    
    try {
        let config = (open $file_path)
        
        # Check required fields
        let required_fields = ["name"]
        for field in $required_fields {
            if $field not-in ($config | columns) {
                log warning $"Missing required field: ($field)"
            }
        }
        
        # Check for either image or build
        if "image" not-in ($config | columns) and "build" not-in ($config | columns) {
            log warning "Neither 'image' nor 'build' specified in devcontainer.json"
        }
        
        log success "✓ devcontainer.json validation completed"
        
    } catch {
        log error "Failed to validate devcontainer.json - invalid JSON format"
        exit 1
    }
}

# Show help information
def show_help [] {
    print "DevContainer Generation Script for Polyglot Development Environment"
    print ""
    print "USAGE:"
    print "    nu devpod-generate.nu [OPTIONS]"
    print ""
    print "OPTIONS:"
    print "    -l, --language <LANGUAGE>    Language environment (python, typescript, rust, go, nushell, full-stack) [default: python]"
    print "    -o, --output <FILE>          Output file path [default: devcontainer.json]"
    print "    -t, --template               Use template instead of devbox generation"
    print "    -m, --merge <FILE>           Merge with existing devcontainer.json file"
    print "    -h, --help                   Show this help"
    print ""
    print "EXAMPLES:"
    print "    nu devpod-generate.nu --language python --output .devcontainer/devcontainer.json"
    print "    nu devpod-generate.nu --language typescript --template"
    print "    nu devpod-generate.nu --language full-stack --merge existing-devcontainer.json"
    print ""
    print "DESCRIPTION:"
    print "    Generates devcontainer.json files for DevPod workspaces either from existing"
    print "    devbox.json configurations or from predefined templates. The generated files"
    print "    include language-specific features, extensions, and optimization settings."
}