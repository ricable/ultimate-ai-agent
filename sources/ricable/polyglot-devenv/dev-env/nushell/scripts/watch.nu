#!/usr/bin/env nu

# Watch for changes and run validations automatically
# Usage: nu scripts/watch.nu [--command command] [--interval seconds]

use ../common.nu *

def main [
    --command: string = "check"
    --interval: int = 5
] {
    log info $"Starting file watcher (checking every ($interval) seconds)..."
    log info $"Running command: ($command)"
    log info "Press Ctrl+C to stop"
    
    mut last_check = (date now)
    
    loop {
        let current_time = (date now)
        let changed_files = get-changed-files $last_check
        
        if ($changed_files | length) > 0 {
            log info $"Changes detected in ($changed_files | length) files:"
            for file in $changed_files {
                log info $"  - ($file)"
            }
            
            run-watch-command $command
            $last_check = $current_time
        }
        
        sleep ($interval * 1sec)
    }
}

def get-changed-files [since: datetime] {
    let all_files = []
    
    # Check Nushell scripts
    let nu_files = try {
        ls scripts/*.nu | where modified > $since | get name
    } catch {
        []
    }
    
    # Check common.nu
    let common_file = try {
        if ("common.nu" | path exists) and ((ls common.nu | get 0.modified) > $since) {
            ["common.nu"]
        } else {
            []
        }
    } catch {
        []
    }
    
    # Check configuration files
    let config_files = try {
        ["devbox.json", ".teller.yml", ".env"] 
        | where ($it | path exists) 
        | where (ls $it | get 0.modified) > $since
    } catch {
        []
    }
    
    $nu_files | append $common_file | append $config_files
}

def run-watch-command [cmd: string] {
    log info $"Running: ($cmd)"
    
    match $cmd {
        "check" => {
            try {
                nu scripts/check.nu
                log success "✅ Check passed"
            } catch { |e|
                log error $"❌ Check failed: ($e.msg)"
            }
        }
        "test" => {
            try {
                nu scripts/test.nu
                log success "✅ Tests passed"
            } catch { |e|
                log error $"❌ Tests failed: ($e.msg)"
            }
        }
        "format" => {
            try {
                nu scripts/format.nu --fix
                log success "✅ Format applied"
            } catch { |e|
                log error $"❌ Format failed: ($e.msg)"
            }
        }
        "validate" => {
            try {
                nu scripts/validate.nu
                log success "✅ Validation passed"
            } catch { |e|
                log error $"❌ Validation failed: ($e.msg)"
            }
        }
        _ => {
            log warn $"Unknown command: ($cmd). Available: check, test, format, validate"
        }
    }
}

# Watch specific directories or files
def "main watch-directory" [
    directory: string
    --command: string = "echo 'File changed'"
    --interval: int = 2
] {
    if not ($directory | path exists) {
        log error $"Directory not found: ($directory)"
        exit 1
    }
    
    log info $"Watching directory: ($directory)"
    mut last_check = (date now)
    
    loop {
        let changed = try {
            ls $directory | where modified > $last_check | length
        } catch {
            0
        }
        
        if $changed > 0 {
            log info $"Changes detected in ($directory)"
            bash -c $command
            $last_check = (date now)
        }
        
        sleep ($interval * 1sec)
    }
}

# Interactive watch mode
def "main watch-interactive" [] {
    log info "Interactive watch mode"
    log info "Available commands:"
    log info "  c - run check"
    log info "  t - run test"
    log info "  f - run format"
    log info "  v - run validate"
    log info "  q - quit"
    log info ""
    
    loop {
        let input = input "Enter command (c/t/f/v/q): "
        
        match $input {
            "c" => { run-watch-command "check" }
            "t" => { run-watch-command "test" }
            "f" => { run-watch-command "format" }
            "v" => { run-watch-command "validate" }
            "q" => { 
                log info "Exiting watch mode"
                break
            }
            _ => { log warn "Invalid command. Use c/t/f/v/q" }
        }
        
        print ""
    }
}