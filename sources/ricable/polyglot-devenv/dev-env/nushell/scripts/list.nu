#!/usr/bin/env nu

# Lists all available Nushell scripts and their descriptions
# Usage: nu scripts/list.nu

use ../common.nu *

def main [] {
    log info "Available Nushell Scripts:"
    print ""
    
    let scripts = [
        {
            file: "setup.nu"
            description: "Setup the Nushell development environment"
            usage: "nu scripts/setup.nu"
        }
        {
            file: "list.nu"
            description: "List all available scripts (this script)"
            usage: "nu scripts/list.nu"
        }
        {
            file: "test.nu"
            description: "Run Nushell script tests and validations"
            usage: "nu scripts/test.nu"
        }
        {
            file: "check.nu"
            description: "Check Nushell scripts for syntax and best practices"
            usage: "nu scripts/check.nu"
        }
        {
            file: "format.nu"
            description: "Format Nushell scripts (basic formatting)"
            usage: "nu scripts/format.nu"
        }
        {
            file: "watch.nu"
            description: "Watch for changes and run validations"
            usage: "nu scripts/watch.nu"
        }
        {
            file: "validate.nu"
            description: "Validate all environments and configurations"
            usage: "nu scripts/validate.nu"
        }
        {
            file: "deploy.nu"
            description: "Deploy/sync configurations across environments"
            usage: "nu scripts/deploy.nu"
        }
        {
            file: "env-sync.nu"
            description: "Sync environment variables across environments"
            usage: "nu scripts/env-sync.nu"
        }
    ]
    
    $scripts | each { |script|
        let exists = $"scripts/($script.file)" | path exists
        let status = if $exists { 
            $"(ansi green_bold)✓(ansi reset)" 
        } else { 
            $"(ansi red_bold)✗(ansi reset)" 
        }
        
        print $"($status) (ansi blue_bold)($script.file)(ansi reset)"
        print $"    ($script.description)"
        print $"    Usage: (ansi yellow)($script.usage)(ansi reset)"
        print ""
    }
    
    # Show devbox scripts
    print $"(ansi cyan_bold)Devbox Scripts:(ansi reset)"
    if ("devbox.json" | path exists) {
        let devbox_config = open devbox.json
        if ("shell" in $devbox_config and "scripts" in $devbox_config.shell) {
            $devbox_config.shell.scripts | transpose key value | each { |script|
                print $"  (ansi green_bold)devbox run ($script.key)(ansi reset) - ($script.value)"
            }
        }
    }
    
    print ""
    log info "To run any script: nu scripts/<script-name>"
    log info "To run devbox scripts: devbox run <script-name>"
}