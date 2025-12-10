#!/usr/bin/env nu

# DevPod Workspace Management Script for Polyglot Development Environment
# Provides lifecycle management for DevPod workspaces

use ../../nushell-env/common.nu *

# Main management command
def main [
    action: string = "list"  # Action to perform (list, status, connect, start, stop, delete, recreate, prebuild)
    --workspace(-w): string  # Workspace name
    --language(-l): string   # Language environment for new workspaces
    --ide: string = "vscode" # IDE to use for connection
    --force(-f)              # Force operation without confirmation
    --all(-a)                # Apply to all polyglot workspaces
    --help(-h)               # Show help
] {
    if $help {
        show_help
        return
    }

    match $action {
        "list" => { list_workspaces }
        "status" => { show_workspace_status $workspace }
        "connect" => { connect_to_workspace $workspace $ide }
        "start" => { start_workspace $workspace $ide }
        "stop" => { stop_workspace $workspace $force }
        "delete" => { delete_workspace $workspace $force }
        "recreate" => { recreate_workspace $workspace $ide }
        "prebuild" => { prebuild_workspace $workspace }
        "cleanup" => { cleanup_workspaces $force }
        "export" => { export_workspace_config $workspace }
        "import" => { import_workspace_config $workspace }
        "sync" => { sync_workspace $workspace }
        _ => {
            log error $"Unknown action: ($action)"
            show_help
            exit 1
        }
    }
}

# List all DevPod workspaces
def list_workspaces [] {
    log info "Listing DevPod workspaces..."
    
    try {
        let workspaces = (devpod list --output json | from json)
        
        if ($workspaces | length) == 0 {
            log info "No DevPod workspaces found"
            return
        }
        
        # Filter polyglot workspaces
        let polyglot_workspaces = ($workspaces | where name =~ "polyglot-.*-devpod")
        
        if ($polyglot_workspaces | length) == 0 {
            log info "No polyglot DevPod workspaces found"
            print "Available workspaces (non-polyglot):"
            $workspaces | table
            return
        }
        
        print "Polyglot DevPod Workspaces:"
        print "=========================="
        
        $polyglot_workspaces | each { |ws|
            let language = get_language_from_workspace_name $ws.name
            let status_emoji = match $ws.status {
                "Running" => "🟢"
                "Stopped" => "🔴"
                "Creating" => "🟡"
                _ => "⚪"
            }
            
            print $"($status_emoji) ($ws.name)"
            print $"   Language: ($language)"
            print $"   Status: ($ws.status)"
            print $"   Provider: ($ws.provider)"
            print $"   Created: ($ws.created)"
            print $"   SSH: ssh ($ws.name).devpod"
            print ""
        }
        
        if ($workspaces | length) > ($polyglot_workspaces | length) {
            print "Other workspaces:"
            ($workspaces | where name !~ "polyglot-.*-devpod") | table
        }
        
    } catch {
        log error "Failed to list workspaces. Is DevPod installed and configured?"
        exit 1
    }
}

# Show detailed status for a workspace
def show_workspace_status [workspace_name?: string] {
    if $workspace_name == null {
        log error "Workspace name required for status command"
        exit 1
    }
    
    log info $"Getting status for workspace: ($workspace_name)"
    
    try {
        let workspace_info = (devpod list --output json | from json | where name == $workspace_name | first)
        
        if $workspace_info == null {
            log error $"Workspace not found: ($workspace_name)"
            exit 1
        }
        
        print $"Workspace Status: ($workspace_name)"
        print "================================"
        print $"Status: ($workspace_info.status)"
        print $"Provider: ($workspace_info.provider)"
        print $"Created: ($workspace_info.created)"
        print $"Language: (get_language_from_workspace_name $workspace_name)"
        print ""
        
        # Get machine info if available
        try {
            let machine_info = (devpod machine list --output json | from json | where workspace == $workspace_name)
            if ($machine_info | length) > 0 {
                print "Machine Information:"
                $machine_info | table
            }
        } catch {
            log info "No machine information available"
        }
        
        # Show connection options
        print "Connection Options:"
        print $"  SSH:           ssh ($workspace_name).devpod"
        print $"  Direct SSH:    devpod ssh ($workspace_name)"
        print $"  VS Code:       devpod up ($workspace_name) --ide vscode"
        print $"  Browser:       devpod up ($workspace_name) --ide openvscode"
        print $"  Terminal:      devpod up ($workspace_name) --ide none"
        
    } catch {
        log error $"Failed to get status for workspace: ($workspace_name)"
        exit 1
    }
}

# Connect to a workspace with specified IDE
def connect_to_workspace [workspace_name?: string, ide: string] {
    if $workspace_name == null {
        log error "Workspace name required for connect command"
        exit 1
    }
    
    log info $"Connecting to workspace ($workspace_name) with IDE ($ide)..."
    
    try {
        devpod up $workspace_name --ide $ide
        log success $"Connected to workspace ($workspace_name) with ($ide)"
    } catch {
        log error $"Failed to connect to workspace: ($workspace_name)"
        exit 1
    }
}

# Start a workspace
def start_workspace [workspace_name?: string, ide: string] {
    if $workspace_name == null {
        log error "Workspace name required for start command"
        exit 1
    }
    
    log info $"Starting workspace: ($workspace_name)"
    
    try {
        if $ide == "none" {
            devpod up $workspace_name --ide none
        } else {
            devpod up $workspace_name --ide $ide
        }
        log success $"Workspace ($workspace_name) started successfully"
    } catch {
        log error $"Failed to start workspace: ($workspace_name)"
        exit 1
    }
}

# Stop a workspace
def stop_workspace [workspace_name?: string, force: bool] {
    if $workspace_name == null {
        log error "Workspace name required for stop command"
        exit 1
    }
    
    if not $force {
        let confirm = (input $"Are you sure you want to stop workspace '($workspace_name)'? (y/N): ")
        if $confirm != "y" and $confirm != "Y" {
            log info "Operation cancelled"
            return
        }
    }
    
    log info $"Stopping workspace: ($workspace_name)"
    
    try {
        devpod stop $workspace_name
        log success $"Workspace ($workspace_name) stopped successfully"
    } catch {
        log error $"Failed to stop workspace: ($workspace_name)"
        exit 1
    }
}

# Delete a workspace
def delete_workspace [workspace_name?: string, force: bool] {
    if $workspace_name == null {
        log error "Workspace name required for delete command"
        exit 1
    }
    
    if not $force {
        let confirm = (input $"Are you sure you want to DELETE workspace '($workspace_name)'? This cannot be undone. (y/N): ")
        if $confirm != "y" and $confirm != "Y" {
            log info "Operation cancelled"
            return
        }
    }
    
    log info $"Deleting workspace: ($workspace_name)"
    
    try {
        devpod delete $workspace_name --force
        log success $"Workspace ($workspace_name) deleted successfully"
    } catch {
        log error $"Failed to delete workspace: ($workspace_name)"
        exit 1
    }
}

# Recreate a workspace
def recreate_workspace [workspace_name?: string, ide: string] {
    if $workspace_name == null {
        log error "Workspace name required for recreate command"
        exit 1
    }
    
    log info $"Recreating workspace: ($workspace_name)"
    
    try {
        devpod up $workspace_name --recreate --ide $ide
        log success $"Workspace ($workspace_name) recreated successfully"
    } catch {
        log error $"Failed to recreate workspace: ($workspace_name)"
        exit 1
    }
}

# Prebuild a workspace
def prebuild_workspace [workspace_name?: string] {
    if $workspace_name == null {
        log error "Workspace name required for prebuild command"
        exit 1
    }
    
    log info $"Prebuilding workspace: ($workspace_name)"
    
    try {
        devpod build $workspace_name
        log success $"Workspace ($workspace_name) prebuilt successfully"
    } catch {
        log warning $"Failed to prebuild workspace: ($workspace_name) - this may be normal if prebuild is not supported"
    }
}

# Cleanup unused workspaces and resources
def cleanup_workspaces [force: bool] {
    log info "Cleaning up DevPod workspaces and resources..."
    
    if not $force {
        let confirm = (input "This will remove stopped workspaces and unused resources. Continue? (y/N): ")
        if $confirm != "y" and $confirm != "Y" {
            log info "Operation cancelled"
            return
        }
    }
    
    try {
        # Get list of workspaces
        let workspaces = (devpod list --output json | from json)
        let stopped_workspaces = ($workspaces | where status == "Stopped")
        
        if ($stopped_workspaces | length) > 0 {
            print "Stopped workspaces that can be cleaned up:"
            $stopped_workspaces | each { |ws|
                print $"  - ($ws.name) (stopped since ($ws.created))"
            }
            
            let cleanup_confirm = (input "Delete these stopped workspaces? (y/N): ")
            if $cleanup_confirm == "y" or $cleanup_confirm == "Y" {
                $stopped_workspaces | each { |ws|
                    try {
                        devpod delete $ws.name --force
                        log success $"Deleted workspace: ($ws.name)"
                    } catch {
                        log error $"Failed to delete workspace: ($ws.name)"
                    }
                }
            }
        } else {
            log info "No stopped workspaces found for cleanup"
        }
        
        # Cleanup Docker resources if available
        try {
            docker system prune -f
            log success "Docker system cleanup completed"
        } catch {
            log info "Docker cleanup skipped (Docker not available or no resources to clean)"
        }
        
    } catch {
        log error "Failed to cleanup workspaces"
        exit 1
    }
}

# Export workspace configuration
def export_workspace_config [workspace_name?: string] {
    if $workspace_name == null {
        log error "Workspace name required for export command"
        exit 1
    }
    
    log info $"Exporting configuration for workspace: ($workspace_name)"
    
    let language = get_language_from_workspace_name $workspace_name
    let env_path = get_env_path_for_language $language
    let devcontainer_path = $"($env_path)/.devcontainer/devcontainer.json"
    
    if not ($devcontainer_path | path exists) {
        log error $"devcontainer.json not found at: ($devcontainer_path)"
        exit 1
    }
    
    let export_file = $"devpod-export-($workspace_name).json"
    
    let export_config = {
        workspace_name: $workspace_name
        language: $language
        export_date: (date now | format date "%Y-%m-%d %H:%M:%S")
        devcontainer: (open $devcontainer_path)
    }
    
    $export_config | to json | save $export_file --force
    
    log success $"Workspace configuration exported to: ($export_file)"
}

# Import workspace configuration
def import_workspace_config [config_file?: string] {
    if $config_file == null {
        log error "Configuration file required for import command"
        exit 1
    }
    
    if not ($config_file | path exists) {
        log error $"Configuration file not found: ($config_file)"
        exit 1
    }
    
    log info $"Importing workspace configuration from: ($config_file)"
    
    try {
        let import_config = (open $config_file)
        let workspace_name = $import_config.workspace_name
        let language = $import_config.language
        let devcontainer = $import_config.devcontainer
        
        let env_path = get_env_path_for_language $language
        let devcontainer_path = $"($env_path)/.devcontainer"
        
        # Create .devcontainer directory if it doesn't exist
        mkdir $devcontainer_path
        
        # Save the devcontainer.json
        $devcontainer | to json --indent 4 | save $"($devcontainer_path)/devcontainer.json" --force
        
        log success $"Configuration imported for workspace: ($workspace_name)"
        log info $"devcontainer.json saved to: ($devcontainer_path)/devcontainer.json"
        log info $"You can now provision this workspace with: nu scripts/devpod-provision.nu --language ($language)"
        
    } catch {
        log error $"Failed to import configuration from: ($config_file)"
        exit 1
    }
}

# Sync workspace with latest devbox configuration
def sync_workspace [workspace_name?: string] {
    if $workspace_name == null {
        log error "Workspace name required for sync command"
        exit 1
    }
    
    log info $"Syncing workspace ($workspace_name) with latest devbox configuration..."
    
    let language = get_language_from_workspace_name $workspace_name
    
    # Regenerate devcontainer.json
    nu ../scripts/devpod-generate.nu --language $language --output $"../($language)-env/.devcontainer/devcontainer.json"
    
    # Recreate the workspace to apply changes
    recreate_workspace $workspace_name "none"
    
    log success $"Workspace ($workspace_name) synced successfully"
}

# Helper function to get language from workspace name
def get_language_from_workspace_name [workspace_name: string] {
    if $workspace_name =~ "polyglot-(.+)-devpod" {
        let matches = ($workspace_name | parse "polyglot-{language}-devpod")
        if ($matches | length) > 0 {
            $matches.0.language
        } else {
            "unknown"
        }
    } else {
        "unknown"
    }
}

# Helper function to get environment path for language
def get_env_path_for_language [language: string] {
    match $language {
        "python" => "../python-env"
        "typescript" => "../typescript-env"
        "rust" => "../rust-env"
        "go" => "../go-env"
        "nushell" => "../nushell-env"
        "full-stack" => ".."
        _ => ".."
    }
}

# Show help information
def show_help [] {
    print "DevPod Workspace Management Script for Polyglot Development Environment"
    print ""
    print "USAGE:"
    print "    nu devpod-manage.nu <ACTION> [OPTIONS]"
    print ""
    print "ACTIONS:"
    print "    list                List all DevPod workspaces"
    print "    status              Show detailed status for a workspace"
    print "    connect             Connect to a workspace with specified IDE"
    print "    start               Start a workspace"
    print "    stop                Stop a workspace"
    print "    delete              Delete a workspace"
    print "    recreate            Recreate a workspace (apply devcontainer changes)"
    print "    prebuild            Prebuild a workspace for faster startup"
    print "    cleanup             Cleanup stopped workspaces and unused resources"
    print "    export              Export workspace configuration to file"
    print "    import              Import workspace configuration from file"
    print "    sync                Sync workspace with latest devbox configuration"
    print ""
    print "OPTIONS:"
    print "    -w, --workspace <NAME>    Workspace name"
    print "    -l, --language <LANG>     Language environment for new workspaces"
    print "    --ide <IDE>               IDE to use (vscode, openvscode, goland, none) [default: vscode]"
    print "    -f, --force               Force operation without confirmation"
    print "    -a, --all                 Apply to all polyglot workspaces"
    print "    -h, --help                Show this help"
    print ""
    print "EXAMPLES:"
    print "    nu devpod-manage.nu list"
    print "    nu devpod-manage.nu status --workspace polyglot-python-devpod"
    print "    nu devpod-manage.nu connect --workspace polyglot-typescript-devpod --ide vscode"
    print "    nu devpod-manage.nu stop --workspace polyglot-rust-devpod --force"
    print "    nu devpod-manage.nu delete --workspace polyglot-go-devpod"
    print "    nu devpod-manage.nu recreate --workspace polyglot-nushell-devpod --ide openvscode"
    print "    nu devpod-manage.nu cleanup --force"
    print "    nu devpod-manage.nu export --workspace polyglot-python-devpod"
    print "    nu devpod-manage.nu sync --workspace polyglot-typescript-devpod"
    print ""
    print "WORKSPACE MANAGEMENT WORKFLOW:"
    print "    1. Provision:  nu scripts/devpod-provision.nu --language python"
    print "    2. Connect:    nu devpod-manage.nu connect --workspace polyglot-python-devpod"
    print "    3. Develop:    (use your IDE or SSH)"
    print "    4. Sync:       nu devpod-manage.nu sync --workspace polyglot-python-devpod"
    print "    5. Stop:       nu devpod-manage.nu stop --workspace polyglot-python-devpod"
}