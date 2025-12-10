#!/usr/bin/env nu

# Configuration Generation System - Single Source of Truth
# Generates all devbox.json, devcontainer.json, and other configs from canonical-environments.yaml

def main [
    --environment: string = "all"  # Generate for specific environment or "all"
    --target: string = "all"       # Generate specific target: devbox, devcontainer, or "all"
    --validate = false             # Validate generated configs
    --force = false                # Overwrite existing files without confirmation
    --dry-run = false              # Show what would be generated without writing files
] {
    log info "🔧 Configuration Generation System"
    log info "Single Source of Truth: canonical-environments.yaml"
    print "=" * 60
    
    # Load canonical definitions
    let canonical_file = "context-engineering/devpod/environments/canonical-environments.yaml"
    
    if not ($canonical_file | path exists) {
        log error $"❌ Canonical definitions not found: ($canonical_file)"
        exit 1
    }
    
    let config = (open $canonical_file | from yaml)
    let environments = $config.environments
    
    # Determine which environments to process
    let env_list = if $environment == "all" {
        $environments | columns
    } else {
        [$environment]
    }
    
    log info $"Processing environments: ($env_list | str join ', ')"
    
    # Generate configurations for each environment
    for env_name in $env_list {
        if not ($env_name in ($environments | columns)) {
            log error $"❌ Environment '($env_name)' not found in canonical definitions"
            continue
        }
        
        let env_config = $environments | get $env_name
        
        log info $"🔧 Generating configurations for ($env_name)..."
        
        if $target == "all" or $target == "devbox" {
            generate-devbox-config $env_name $env_config $dry_run $force
        }
        
        if $target == "all" or $target == "devcontainer" {
            generate-devcontainer-config $env_name $env_config $dry_run $force
        }
        
        if $validate {
            validate-generated-config $env_name $target
        }
    }
    
    if not $dry_run {
        log success "✅ Configuration generation completed!"
        log info "💡 Generated configs are now the single source of truth"
        log info "   Do not edit generated files directly - modify canonical-environments.yaml instead"
    } else {
        log info "🔍 Dry run completed - no files were modified"
    }
}

def generate-devbox-config [env_name: string, env_config: record, dry_run: bool, force: bool] {
    let target_file = $"dev-env/($env_name)/devbox.json"
    
    # Build devbox configuration
    let devbox_config = {
        "packages": $env_config.packages.devbox,
        "shell": {
            "init_hook": [
                $"echo '($env_config.name)'",
                # Add language-specific version commands
                ...(get-version-commands $env_name $env_config)
            ],
            "scripts": (
                $env_config.scripts 
                | upsert devpod:provision $"bash ../devpod-automation/scripts/provision-($env_name).sh"
                | upsert devpod:connect "echo 'ℹ️  Each provision creates a new workspace. Use devbox run devpod:provision to create and connect.'"
                | upsert devpod:start "echo 'ℹ️  Use devbox run devpod:provision to create a new workspace or devpod list to see existing ones.'"
                | upsert devpod:stop $"bash -c 'echo \"🛑 Available ($env_name | str capitalize) workspaces:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found\"'"
                | upsert devpod:delete $"bash -c 'echo \"🗑️  ($env_name | str capitalize) workspaces to delete:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found\"'"
                | upsert devpod:sync $"echo 'Sync: Update devbox.json and rebuild workspace with devbox run devpod:provision'"
                | upsert devpod:status $"bash -c 'echo \"📊 ($env_name | str capitalize) DevPod workspaces:\"; devpod list | grep polyglot-($env_name)-devpod || echo \"No ($env_name | str capitalize) workspaces found. Run: devbox run devpod:provision\"'"
            )
        },
        "env": (filter-environment-vars $env_config.environment false)
    }
    
    if $dry_run {
        log info $"📄 Would generate: ($target_file)"
        print ($devbox_config | to json --indent 2)
        print ""
        return
    }
    
    # Check if file exists and handle overwrite
    if ($target_file | path exists) and not $force {
        let response = (input $"⚠️  ($target_file) exists. Overwrite? (y/N): ")
        if $response != "y" and $response != "Y" {
            log info $"⏭️  Skipped: ($target_file)"
            return
        }
    }
    
    # Ensure directory exists
    mkdir ($target_file | path dirname)
    
    # Write configuration
    $devbox_config | to json --indent 2 | save $target_file --force
    log success $"✅ Generated: ($target_file)"
}

def generate-devcontainer-config [env_name: string, env_config: record, dry_run: bool, force: bool] {
    let target_file = $"devpod-automation/templates/($env_name)/devcontainer.json"
    
    # Build devcontainer configuration
    let devcontainer_config = {
        "name": $env_config.name,
        "image": $env_config.packages.devcontainer.base_image,
        "features": (build-devcontainer-features $env_config.packages.devcontainer.features),
        "customizations": {
            "vscode": {
                "extensions": $env_config.vscode.extensions,
                "settings": $env_config.vscode.settings
            },
            "devpod": {
                "prebuildRepository": $"ghcr.io/ricable/polyglot-devenv-($env_name)"
            }
        },
        "containerEnv": (filter-environment-vars $env_config.environment true),
        "mounts": (build-container-mounts $env_config.container.mounts),
        "forwardPorts": ($env_config.container.ports | each { |port| $port.port }),
        "portsAttributes": (build-port-attributes $env_config.container.ports),
        "postCreateCommand": $env_config.container.post_create,
        "postStartCommand": $env_config.container.post_start,
        "postAttachCommand": $env_config.container.post_attach
    }
    
    if $dry_run {
        log info $"📄 Would generate: ($target_file)"
        print ($devcontainer_config | to json --indent 2)
        print ""
        return
    }
    
    # Check if file exists and handle overwrite
    if ($target_file | path exists) and not $force {
        let response = (input $"⚠️  ($target_file) exists. Overwrite? (y/N): ")
        if $response != "y" and $response != "Y" {
            log info $"⏭️  Skipped: ($target_file)"
            return
        }
    }
    
    # Ensure directory exists
    mkdir ($target_file | path dirname)
    
    # Write configuration
    $devcontainer_config | to json --indent 2 | save $target_file --force
    log success $"✅ Generated: ($target_file)"
}

def get-version-commands [env_name: string, env_config: record] {
    match $env_name {
        "python" => ["uv --version", "python --version"]
        "typescript" => ["node --version", "npm --version"]
        "rust" => ["rustc --version", "cargo --version"]
        "go" => ["go version"]
        "nushell" => ["nu --version"]
        _ => []
    }
}

def filter-environment-vars [env_vars: record, devcontainer: bool] {
    if $devcontainer {
        if "devcontainer" in ($env_vars | columns) {
            $env_vars.devcontainer
        } else {
            # Filter out devbox-specific paths and convert to devcontainer format
            $env_vars | reject devcontainer? | items { |key, value|
                {
                    key: $key,
                    value: (if ($value | str contains "$PWD") {
                        $value | str replace "$PWD" "/workspace"
                    } else {
                        $value
                    })
                }
            } | reduce -f {} { |item, acc| $acc | upsert $item.key $item.value }
        }
    } else {
        $env_vars | reject devcontainer?
    }
}

def build-devcontainer-features [features: record] {
    $features | items { |name, config|
        {$"ghcr.io/devcontainers/features/($name):1": $config}
    } | reduce -f {} { |item, acc| $acc | merge $item }
}

def build-container-mounts [mounts: list] {
    $mounts | each { |mount|
        $"type=($mount.type),source=($mount.source),target=($mount.target)"
    }
}

def build-port-attributes [ports: list] {
    $ports | each { |port|
        {
            ($port.port | into string): {
                "label": $port.label,
                "onAutoForward": (if $port.auto_forward { "notify" } else { "silent" })
            }
        }
    } | reduce -f {} { |item, acc| $acc | merge $item }
}

def validate-generated-config [env_name: string, target: string] {
    log info $"🔍 Validating generated configurations for ($env_name)..."
    
    if $target == "all" or $target == "devbox" {
        let devbox_file = $"dev-env/($env_name)/devbox.json"
        if ($devbox_file | path exists) {
            try {
                open $devbox_file | from json | ignore
                log success $"✅ Valid JSON: ($devbox_file)"
            } catch {
                log error $"❌ Invalid JSON: ($devbox_file)"
            }
        }
    }
    
    if $target == "all" or $target == "devcontainer" {
        let devcontainer_file = $"devpod-automation/templates/($env_name)/devcontainer.json"
        if ($devcontainer_file | path exists) {
            try {
                open $devcontainer_file | from json | ignore
                log success $"✅ Valid JSON: ($devcontainer_file)"
            } catch {
                log error $"❌ Invalid JSON: ($devcontainer_file)"
            }
        }
    }
}

# Utility functions for logging
def log [level: string, message: string] {
    let timestamp = (date now | format date "%Y-%m-%d %H:%M:%S")
    match $level {
        "info" => { print $"[($timestamp)] ℹ️  ($message)" }
        "success" => { print $"[($timestamp)] ✅ ($message)" }
        "error" => { print $"[($timestamp)] ❌ ($message)" }
        "warning" => { print $"[($timestamp)] ⚠️  ($message)" }
        _ => { print $"[($timestamp)] ($message)" }
    }
}

# Additional utility commands
def "main validate-all" [] {
    log info "🔍 Validating all generated configurations..."
    
    let canonical_file = "context-engineering/devpod/environments/canonical-environments.yaml"
    let config = (open $canonical_file | from yaml)
    let environments = $config.environments | columns
    
    for env_name in $environments {
        validate-generated-config $env_name "all"
    }
    
    log success "✅ Validation completed!"
}

def "main diff" [environment: string = "all"] {
    log info "📊 Comparing canonical definitions with existing configurations..."
    
    # This would show differences between generated and existing configs
    # Implementation depends on specific comparison needs
    log info "Diff functionality - to be implemented based on specific needs"
}

def "main backup" [] {
    let timestamp = (date now | format date "%Y%m%d_%H%M%S")
    let backup_dir = $"backups/config_backup_($timestamp)"
    
    log info $"💾 Creating backup of existing configurations: ($backup_dir)"
    
    mkdir $backup_dir
    
    # Backup existing configurations
    cp -r dev-env/*/devbox.json $"($backup_dir)/devbox/" | ignore
    cp -r devpod-automation/templates/*/devcontainer.json $"($backup_dir)/devcontainer/" | ignore
    
    log success $"✅ Backup created: ($backup_dir)"
}