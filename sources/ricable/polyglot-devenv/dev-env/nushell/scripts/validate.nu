#!/usr/bin/env nu

# Cross-language validation script for the polyglot development environment
# This is the Nushell implementation of the validation script from CLAUDE.md
# Usage: nu scripts/validate.nu [--parallel] [--env environment]

use ../common.nu *

def main [
    --parallel = false
    --environment: string = "all"
    --skip: list<string> = []
] {
    log info "Starting polyglot environment validation..."
    
    if $parallel {
        validate-parallel $environment $skip
    } else {
        validate-sequential $environment $skip
    }
}

def validate-sequential [target_env: string, skip: list<string>] {
    let environments = get-environments $target_env $skip
    mut success_count = 0
    mut total_count = ($environments | length)
    
    for environment in $environments {
        log info $"Validating ($environment.name)..."
        
        if ($environment.dir | path exists) {
            cd $environment.dir
            
            let result = validate-environment $environment
            
            if $result {
                log success $"✅ ($environment.name) validation passed"
                $success_count = $success_count + 1
            } else {
                log error $"❌ ($environment.name) validation failed"
            }
            
            cd ..
        } else {
            log warn $"⚠️  ($environment.name) directory not found: ($environment.dir)"
            $total_count = $total_count - 1
        }
    }
    
    print ""
    log info $"Validation Results: ($success_count)/($total_count) environments passed"
    
    if $success_count != $total_count {
        exit 1
    } else {
        log success "🎉 All validations completed successfully!"
    }
}

def validate-parallel [target_env: string, skip: list<string>] {
    log info "Running validations in parallel..."
    
    let environments = get-environments $target_env $skip
    
    let results = $environments | par-each { |env|
        if ($env.dir | path exists) {
            let original_dir = pwd
            cd $env.dir
            
            let result = try {
                validate-environment $env
                {name: $env.name, status: "passed", error: null}
            } catch { |e|
                {name: $env.name, status: "failed", error: $e.msg}
            }
            
            cd $original_dir
            $result
        } else {
            {name: $env.name, status: "skipped", error: "Directory not found"}
        }
    }
    
    # Report results
    for result in $results {
        match $result.status {
            "passed" => { log success $"✅ ($result.name)" }
            "failed" => { 
                log error $"❌ ($result.name)"
                if $result.error != null {
                    log error $"  Error: ($result.error)"
                }
            }
            "skipped" => { log warn $"⚠️  ($result.name) - ($result.error)" }
        }
    }
    
    let passed = $results | where status == "passed" | length
    let failed = $results | where status == "failed" | length
    let skipped = $results | where status == "skipped" | length
    
    print ""
    log info $"Parallel Validation Results: ($passed) passed, ($failed) failed, ($skipped) skipped"
    
    if $failed > 0 {
        exit 1
    } else {
        log success "🎉 All parallel validations completed successfully!"
    }
}

def get-environments [target_env: string, skip: list<string>] {
    let all_environments = [
        {name: "🐍 Python", dir: "python-env", commands: ["lint", "test"]},
        {name: "📘 TypeScript", dir: "typescript-env", commands: ["lint", "test"]}, 
        {name: "🦀 Rust", dir: "rust-env", commands: ["lint", "test"]},
        {name: "🐹 Go", dir: "go-env", commands: ["lint", "test"]},
        {name: "🐚 Nushell", dir: "nushell-env", commands: ["check", "test"]}
    ]
    
    let filtered = if $target_env == "all" {
        $all_environments
    } else {
        $all_environments | where name =~ $target_env
    }
    
    $filtered | where not ($it.dir in $skip)
}

def validate-environment [environment: record] {
    log info $"  Running commands for ($environment.name)..."
    
    # Check if devbox.json exists
    if not ("devbox.json" | path exists) {
        log warn $"    No devbox.json found in ($environment.dir)"
        return false
    }
    
    # Check if devbox is available
    if not (cmd exists "devbox") {
        log error $"    devbox command not found"
        return false
    }
    
    # Run each command for the environment
    for cmd in $environment.commands {
        log info $"    Running: devbox run ($cmd)"
        
        let result = try {
            run safe $"devbox run ($cmd)"
            true
        } catch { |e|
            log error $"    Command failed: ($e.msg)"
            false
        }
        
        if not $result {
            return false
        }
    }
    
    true
}

# Specific validation functions
def "main validate-configs" [] {
    log info "Validating configuration files across all environments..."
    
    let environments = ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]
    
    for environment in $environments {
        if ($environment | path exists) {
            log info $"Checking ($environment) configuration..."
            cd $environment
            
            # Check devbox.json
            if ("devbox.json" | path exists) {
                try {
                    open devbox.json | ignore
                    log success $"  ✅ devbox.json readable"
                } catch {
                    log error $"  ❌ devbox.json invalid"
                }
            } else {
                log warn $"  ⚠️  No devbox.json found"
            }
            
            cd ..
        }
    }
}

def "main validate-dependencies" [] {
    log info "Validating external dependencies..."
    
    let required_tools = [
        {name: "devbox", description: "Package manager"},
        {name: "git", description: "Version control"},
        {name: "nu", description: "Nushell"}
    ]
    
    let optional_tools = [
        {name: "teller", description: "Secret management"},
        {name: "direnv", description: "Environment management"}
    ]
    
    log info "Required tools:"
    mut all_required_ok = true
    for tool in $required_tools {
        if (cmd exists $tool.name) {
            log success $"  ✅ ($tool.name) - ($tool.description)"
        } else {
            log error $"  ❌ ($tool.name) - ($tool.description) - MISSING"
            $all_required_ok = false
        }
    }
    
    log info "Optional tools:"
    for tool in $optional_tools {
        if (cmd exists $tool.name) {
            log success $"  ✅ ($tool.name) - ($tool.description)"
        } else {
            log warn $"  ⚠️  ($tool.name) - ($tool.description) - not installed"
        }
    }
    
    if not $all_required_ok {
        log error "Some required tools are missing!"
        exit 1
    }
}

def "main validate-structure" [] {
    log info "Validating project structure..."
    
    let expected_structure = [
        {path: "CLAUDE.md", type: "file", required: true},
        {path: "python-env", type: "dir", required: false},
        {path: "typescript-env", type: "dir", required: false},
        {path: "rust-env", type: "dir", required: false},
        {path: "go-env", type: "dir", required: false},
        {path: "nushell-env", type: "dir", required: true},
        {path: "nushell-env/devbox.json", type: "file", required: true},
        {path: "nushell-env/common.nu", type: "file", required: true},
        {path: "nushell-env/scripts", type: "dir", required: true}
    ]
    
    mut structure_ok = true
    
    for item in $expected_structure {
        let exists = $item.path | path exists
        
        if $exists {
            log success $"  ✅ ($item.path)"
        } else if $item.required {
            log error $"  ❌ ($item.path) - REQUIRED"
            $structure_ok = false
        } else {
            log warn $"  ⚠️  ($item.path) - optional"
        }
    }
    
    if not $structure_ok {
        log error "Project structure validation failed!"
        exit 1
    } else {
        log success "Project structure validation passed!"
    }
}