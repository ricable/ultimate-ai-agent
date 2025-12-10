#!/usr/bin/env nu

# DevPod Synchronization Script for Polyglot Development Environment
# Keeps Devbox and DevPod environments synchronized

use ../../nushell-env/common.nu *

# Main sync command
def main [
    --language(-l): string           # Language environment to sync (python, typescript, rust, go, nushell, full-stack, all)
    --direction(-d): string = "both" # Sync direction (devbox-to-devpod, devpod-to-devbox, both)
    --auto(-a)                       # Auto-sync mode (watch for changes)
    --dry-run(-n)                    # Show what would be done without making changes
    --force(-f)                      # Force sync without confirmation
    --recreate(-r)                   # Recreate DevPod workspaces after sync
    --help(-h)                       # Show help
] {
    if $help {
        show_help
        return
    }

    if $language == null {
        log error "Language parameter is required. Use --language all for all environments."
        exit 1
    }

    log info $"Starting synchronization for ($language) environment(s)..."

    if $auto {
        start_auto_sync $language $direction
    } else {
        if $language == "all" {
            sync_all_environments $direction $dry_run $force $recreate
        } else {
            sync_environment $language $direction $dry_run $force $recreate
        }
    }
}

# Sync all environments
def sync_all_environments [direction: string, dry_run: bool, force: bool, recreate: bool] {
    let languages = ["python", "typescript", "rust", "go", "nushell"]
    
    for lang in $languages {
        log info $"Syncing ($lang) environment..."
        sync_environment $lang $direction $dry_run $force $recreate
    }
    
    # Also sync full-stack environment
    log info "Syncing full-stack environment..."
    sync_environment "full-stack" $direction $dry_run $force $recreate
}

# Sync a specific environment
def sync_environment [language: string, direction: string, dry_run: bool, force: bool, recreate: bool] {
    let env_config = get_environment_config $language
    
    log info $"Synchronizing ($language) environment..."
    
    match $direction {
        "devbox-to-devpod" => sync_devbox_to_devpod $env_config $dry_run $force $recreate
        "devpod-to-devbox" => sync_devpod_to_devbox $env_config $dry_run $force
        "both" => {
            sync_devbox_to_devpod $env_config $dry_run $force false
            sync_devpod_to_devbox $env_config $dry_run $force
            if $recreate {
                recreate_devpod_workspace $env_config.workspace_name $force
            }
        }
        _ => {
            log error $"Invalid direction: ($direction). Use 'devbox-to-devpod', 'devpod-to-devbox', or 'both'"
            exit 1
        }
    }
}

# Sync from devbox.json to devcontainer.json
def sync_devbox_to_devpod [env_config: record, dry_run: bool, force: bool, recreate: bool] {
    let devbox_path = $"($env_config.env_path)/devbox.json"
    let devcontainer_path = $"($env_config.env_path)/.devcontainer/devcontainer.json"
    
    if not ($devbox_path | path exists) {
        log error $"devbox.json not found: ($devbox_path)"
        return
    }
    
    log info "Syncing devbox.json → devcontainer.json..."
    
    # Check if devbox.json has changed
    let devbox_modified = (ls $devbox_path | get modified | first)
    let devcontainer_exists = ($devcontainer_path | path exists)
    let devcontainer_modified = if $devcontainer_exists {
        (ls $devcontainer_path | get modified | first)
    } else {
        (date now | date to-record | date to-timezone utc)
    }
    
    if $devcontainer_exists and $devbox_modified < $devcontainer_modified and not $force {
        log info "devcontainer.json is newer than devbox.json, skipping sync"
        return
    }
    
    if $dry_run {
        print $"Would regenerate: ($devcontainer_path)"
        return
    }
    
    # Generate new devcontainer.json from devbox.json
    try {
        nu ($"($env.PWD)/devpod-automation/scripts/devpod-generate.nu" | path expand) --language $env_config.language --output $devcontainer_path
        log success $"✓ devcontainer.json updated from devbox.json"
        
        # Update package synchronization
        sync_packages_devbox_to_devpod $env_config
        
        if $recreate {
            recreate_devpod_workspace $env_config.workspace_name $force
        }
        
    } catch {
        log error "Failed to generate devcontainer.json from devbox.json"
        exit 1
    }
}

# Sync package configurations from devbox to devcontainer
def sync_packages_devbox_to_devpod [env_config: record] {
    let devbox_config = (open $"($env_config.env_path)/devbox.json")
    let devcontainer_path = $"($env_config.env_path)/.devcontainer/devcontainer.json"
    
    if not ($devcontainer_path | path exists) {
        return
    }
    
    let devcontainer_config = (open $devcontainer_path)
    
    # Update features based on devbox packages
    let updated_features = update_features_from_devbox $devbox_config.packages $env_config.language
    
    let updated_devcontainer = ($devcontainer_config | merge {
        features: $updated_features
        containerEnv: (merge_environment_variables $devcontainer_config.containerEnv $devbox_config.env)
    })
    
    $updated_devcontainer | to json --indent 4 | save $devcontainer_path --force
    
    log success "✓ DevContainer features updated from devbox packages"
}

# Update DevContainer features based on devbox packages
def update_features_from_devbox [packages: list, language: string] {
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
    
    # Add language-specific features based on packages
    let language_features = {}
    
    for package in $packages {
        if $package | str contains "python@" {
            let version = ($package | str replace "python@" "")
            $language_features = ($language_features | merge {
                "ghcr.io/devcontainers/features/python:1": {
                    "version": $version
                    "installTools": true
                }
            })
        } else if $package | str contains "nodejs@" {
            let version = ($package | str replace "nodejs@" "")
            $language_features = ($language_features | merge {
                "ghcr.io/devcontainers/features/node:1": {
                    "version": $version
                    "nodeGypDependencies": true
                }
            })
        } else if $package | str contains "go@" {
            let version = ($package | str replace "go@" "")
            $language_features = ($language_features | merge {
                "ghcr.io/devcontainers/features/go:1": {
                    "version": $version
                }
            })
        } else if $package == "rust" or ($package | str contains "rust") {
            $language_features = ($language_features | merge {
                "ghcr.io/devcontainers/features/rust:1": {
                    "version": "latest"
                    "profile": "default"
                }
            })
        }
    }
    
    $base_features | merge $language_features
}

# Merge environment variables
def merge_environment_variables [devcontainer_env: record, devbox_env: record] {
    if $devbox_env == null {
        $devcontainer_env
    } else {
        $devcontainer_env | merge $devbox_env
    }
}

# Sync from devcontainer.json to devbox.json (limited)
def sync_devpod_to_devbox [env_config: record, dry_run: bool, force: bool] {
    let devcontainer_path = $"($env_config.env_path)/.devcontainer/devcontainer.json"
    let devbox_path = $"($env_config.env_path)/devbox.json"
    
    if not ($devcontainer_path | path exists) {
        log warning $"devcontainer.json not found: ($devcontainer_path)"
        return
    }
    
    if not ($devbox_path | path exists) {
        log warning $"devbox.json not found: ($devbox_path)"
        return
    }
    
    log info "Syncing devcontainer.json → devbox.json (environment variables only)..."
    
    if $dry_run {
        print $"Would update environment variables in: ($devbox_path)"
        return
    }
    
    # Read configurations
    let devcontainer_config = (open $devcontainer_path)
    let devbox_config = (open $devbox_path)
    
    # Extract environment variables from devcontainer
    let container_env = ($devcontainer_config.containerEnv | default {})
    
    # Filter out DevPod-specific environment variables
    let filtered_env = ($container_env | reject TERM COLORTERM)
    
    # Update devbox.json environment
    let updated_devbox = ($devbox_config | merge {
        env: (($devbox_config.env | default {}) | merge $filtered_env)
    })
    
    $updated_devbox | to json --indent 2 | save $devbox_path --force
    
    log success "✓ devbox.json environment updated from devcontainer.json"
}

# Recreate DevPod workspace after sync
def recreate_devpod_workspace [workspace_name: string, force: bool] {
    if not $force {
        let confirm = (input $"Recreate DevPod workspace '($workspace_name)' to apply changes? (y/N): ")
        if $confirm != "y" and $confirm != "Y" {
            log info "Workspace recreation skipped"
            return
        }
    }
    
    log info $"Recreating DevPod workspace: ($workspace_name)"
    
    try {
        devpod up $workspace_name --recreate --ide none
        log success $"✓ Workspace ($workspace_name) recreated successfully"
    } catch {
        log error $"Failed to recreate workspace: ($workspace_name)"
    }
}

# Start auto-sync mode (watch for changes)
def start_auto_sync [language: string, direction: string] {
    log info $"Starting auto-sync mode for ($language) environment(s)..."
    log info "Watching for changes in devbox.json files..."
    log info "Press Ctrl+C to stop auto-sync"
    
    let environments = if $language == "all" {
        ["python", "typescript", "rust", "go", "nushell", "full-stack"]
    } else {
        [$language]
    }
    
    # Create a map of file paths to watch
    let watch_files = ($environments | each { |env|
        let env_config = get_environment_config $env
        {
            file: $"($env_config.env_path)/devbox.json"
            language: $env
        }
    })
    
    # Store last modification times
    let mut last_modified = {}
    
    for file_info in $watch_files {
        if ($file_info.file | path exists) {
            let modified = (ls $file_info.file | get modified | first)
            $last_modified = ($last_modified | merge {
                ($file_info.file): $modified
            })
        }
    }
    
    # Watch loop
    loop {
        for file_info in $watch_files {
            if ($file_info.file | path exists) {
                let current_modified = (ls $file_info.file | get modified | first)
                let last_mod = ($last_modified | get $file_info.file | default (date now | date to-record | date to-timezone utc))
                
                if $current_modified > $last_mod {
                    log info $"Change detected in ($file_info.file)"
                    sync_environment $file_info.language $direction false true false
                    
                    $last_modified = ($last_modified | merge {
                        ($file_info.file): $current_modified
                    })
                }
            }
        }
        
        sleep 2sec
    }
}

# Get environment configuration
def get_environment_config [language: string] {
    let configs = {
        python: {
            language: "python"
            env_path: "../python-env"
            workspace_name: "polyglot-python-devpod"
        }
        typescript: {
            language: "typescript"
            env_path: "../typescript-env"
            workspace_name: "polyglot-typescript-devpod"
        }
        rust: {
            language: "rust"
            env_path: "../rust-env"
            workspace_name: "polyglot-rust-devpod"
        }
        go: {
            language: "go"
            env_path: "../go-env"
            workspace_name: "polyglot-go-devpod"
        }
        nushell: {
            language: "nushell"
            env_path: "../nushell-env"
            workspace_name: "polyglot-nushell-devpod"
        }
        "full-stack": {
            language: "full-stack"
            env_path: ".."
            workspace_name: "polyglot-full-devpod"
        }
    }
    
    if $language not-in ($configs | columns) {
        log error $"Unknown language: ($language)"
        exit 1
    }
    
    $configs | get $language
}

# Check sync status for all environments
def check_sync_status [] {
    let languages = ["python", "typescript", "rust", "go", "nushell", "full-stack"]
    
    print "Synchronization Status Report"
    print "============================"
    print ""
    
    for lang in $languages {
        let env_config = get_environment_config $lang
        let devbox_path = $"($env_config.env_path)/devbox.json"
        let devcontainer_path = $"($env_config.env_path)/.devcontainer/devcontainer.json"
        
        print $"Environment: ($lang)"
        
        if ($devbox_path | path exists) {
            let devbox_modified = (ls $devbox_path | get modified | first)
            print $"  devbox.json: ✓ (modified: ($devbox_modified))"
        } else {
            print $"  devbox.json: ✗ (not found)"
        }
        
        if ($devcontainer_path | path exists) {
            let devcontainer_modified = (ls $devcontainer_path | get modified | first)
            print $"  devcontainer.json: ✓ (modified: ($devcontainer_modified))"
            
            # Check if sync is needed
            if ($devbox_path | path exists) {
                let devbox_modified = (ls $devbox_path | get modified | first)
                if $devbox_modified > $devcontainer_modified {
                    print $"  Status: ⚠️  Sync needed (devbox.json is newer)"
                } else {
                    print $"  Status: ✅ In sync"
                }
            }
        } else {
            print $"  devcontainer.json: ✗ (not found)"
            if ($devbox_path | path exists) {
                print $"  Status: ⚠️  DevContainer needs generation"
            }
        }
        
        print ""
    }
}

# Show help information
def show_help [] {
    print "DevPod Synchronization Script for Polyglot Development Environment"
    print ""
    print "USAGE:"
    print "    nu devpod-sync.nu [OPTIONS]"
    print ""
    print "OPTIONS:"
    print "    -l, --language <LANGUAGE>    Language environment (python, typescript, rust, go, nushell, full-stack, all)"
    print "    -d, --direction <DIRECTION>  Sync direction (devbox-to-devpod, devpod-to-devbox, both) [default: both]"
    print "    -a, --auto                   Auto-sync mode (watch for changes)"
    print "    -n, --dry-run                Show what would be done without making changes"
    print "    -f, --force                  Force sync without confirmation"
    print "    -r, --recreate               Recreate DevPod workspaces after sync"
    print "    -h, --help                   Show this help"
    print ""
    print "EXAMPLES:"
    print "    nu devpod-sync.nu --language python"
    print "    nu devpod-sync.nu --language all --direction devbox-to-devpod"
    print "    nu devpod-sync.nu --language typescript --auto"
    print "    nu devpod-sync.nu --language rust --dry-run"
    print "    nu devpod-sync.nu --language go --force --recreate"
    print ""
    print "SYNC DIRECTIONS:"
    print "    devbox-to-devpod     Sync from devbox.json to devcontainer.json"
    print "    devpod-to-devbox     Sync environment variables back to devbox.json"
    print "    both                 Sync in both directions (default)"
    print ""
    print "AUTO-SYNC MODE:"
    print "    Watches for changes in devbox.json files and automatically syncs"
    print "    devcontainer.json when changes are detected. Use Ctrl+C to stop."
    print ""
    print "DESCRIPTION:"
    print "    This script keeps Devbox and DevPod environments synchronized by:"
    print "    - Regenerating devcontainer.json from devbox.json changes"
    print "    - Updating package configurations and features"
    print "    - Syncing environment variables between configurations"
    print "    - Optionally recreating DevPod workspaces to apply changes"
}