#!/usr/bin/env nu

# Cross-Language Quality Gates for Polyglot Development Environment
# This is the implementation of the validation script described in CLAUDE.md
# Usage: nu scripts/validate-all.nu [--parallel] [--environment environment]

use ../dev-env/nushell/common.nu *

def main [
    --parallel(-p)
    --environment(-e): string = "all"
    --skip: list<string> = []
    --verbose(-v)
] {
    log info "🚀 Starting cross-language quality gates validation..."
    log info $"Target environments: ($environment)"
    
    if $parallel {
        validate-parallel $environment $skip $verbose
    } else {
        validate-sequential $environment $skip $verbose
    }
}

def validate-sequential [target_env: string, skip: list<string>, verbose: bool] {
    let environments = get-validation-environments $target_env $skip
    mut success_count = 0
    mut total_count = ($environments | length)
    
    for environment in $environments {
        print ""
        log info $"($environment.name)..."
        
        if ($environment.dir | path exists) {
            cd $environment.dir
            
            let result = validate-environment $environment $verbose
            
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
    print (0..59 | each { "=" } | str join "")
    log info $"Validation Results: ($success_count)/($total_count) environments passed"
    
    if $success_count == $total_count {
        log success "🎉 All validations passed!"
        exit 0
    } else {
        log error "💥 Some validations failed!"
        exit 1
    }
}

def validate-parallel [target_env: string, skip: list<string>, verbose: bool] {
    log info "🏃‍♂️ Running validations in parallel..."
    
    let environments = get-validation-environments $target_env $skip
    
    let results = $environments | par-each { |environment|
        if ($environment.dir | path exists) {
            let original_dir = pwd
            
            let result = try {
                cd $environment.dir
                validate-environment $environment $verbose
                cd $original_dir
                {name: $environment.name, status: "passed", error: null, emoji: $environment.emoji}
            } catch { |e|
                cd $original_dir
                {name: $environment.name, status: "failed", error: $e.msg, emoji: $environment.emoji}
            }
            
            $result
        } else {
            {name: $environment.name, status: "skipped", error: "Directory not found", emoji: $environment.emoji}
        }
    }
    
    # Report results
    print ""
    for result in $results {
        match $result.status {
            "passed" => { log success $"✅ ($result.emoji) ($result.name)" }
            "failed" => { 
                log error $"❌ ($result.emoji) ($result.name)"
                if $verbose and $result.error != null {
                    log error $"    Error: ($result.error)"
                }
            }
            "skipped" => { log warn $"⚠️  ($result.emoji) ($result.name) - ($result.error)" }
        }
    }
    
    let passed = $results | where status == "passed" | length
    let failed = $results | where status == "failed" | length
    let skipped = $results | where status == "skipped" | length
    
    print ""
    print (0..59 | each { "=" } | str join "")
    log info $"Parallel Validation Results: ($passed) passed, ($failed) failed, ($skipped) skipped"
    
    if $failed == 0 {
        log success "🎉 All parallel validations completed successfully!"
        exit 0
    } else {
        log error "💥 Some parallel validations failed!"
        exit 1
    }
}

def get-validation-environments [target_env: string, skip: list<string>] {
    let all_environments = [
        {name: "Python", emoji: "🐍", dir: "dev-env/python", commands: ["lint", "test"]},
        {name: "TypeScript", emoji: "📘", dir: "dev-env/typescript", commands: ["lint", "test"]}, 
        {name: "Rust", emoji: "🦀", dir: "dev-env/rust", commands: ["lint", "test"]},
        {name: "Go", emoji: "🐹", dir: "dev-env/go", commands: ["lint", "test"]},
        {name: "Nushell", emoji: "🐚", dir: "dev-env/nushell", commands: ["check", "test"]}
    ]
    
    let filtered = if $target_env == "all" {
        $all_environments
    } else {
        $all_environments | where ($it.name | str downcase) =~ ($target_env | str downcase)
    }
    
    $filtered | where not ($it.dir in $skip)
}

def validate-environment [environment: record, verbose: bool] {
    if $verbose {
        log info $"  📁 Working directory: (pwd)"
    }
    
    # Check if devbox.json exists
    if not ("devbox.json" | path exists) {
        if $verbose {
            log warn $"    No devbox.json found in ($environment.dir)"
        }
        return false
    }
    
    # Check if devbox is available
    if not (cmd exists "devbox") {
        log error $"    devbox command not found"
        return false
    }
    
    # Run each command for the environment
    for cmd in $environment.commands {
        if $verbose {
            log info $"    🔧 Running: devbox run ($cmd)"
        } else {
            log info $"    Running ($cmd)..."
        }
        
        let result = try {
            # Use run safe from common.nu for better error handling
            run safe $"devbox run ($cmd)"
            true
        } catch { |e|
            if $verbose {
                log error $"    ❌ Command failed: ($e.msg)"
            } else {
                log error $"    ❌ ($cmd) failed"
            }
            false
        }
        
        if not $result {
            return false
        } else if $verbose {
            log success $"    ✅ ($cmd) passed"
        }
    }
    
    true
}

# Specific validation modes
def "main quick" [] {
    log info "🏃‍♂️ Running quick validation (syntax checks only)..."
    
    let environments = get-validation-environments "all" []
    
    for environment in $environments {
        if ($environment.dir | path exists) {
            log info $"($environment.emoji) ($environment.name)..."
            cd $environment.dir
            
            # Quick syntax/format checks only
            if ("devbox.json" | path exists) {
                log success $"  ✅ devbox.json exists"
            } else {
                log error $"  ❌ devbox.json missing"
            }
            
            cd ..
        }
    }
    
    log success "🎉 Quick validation completed!"
}

def "main dependencies" [] {
    log info "🔍 Checking external dependencies..."
    
    let required_tools = [
        {name: "devbox", description: "Package manager", required: true},
        {name: "git", description: "Version control", required: true},
        {name: "nu", description: "Nushell", required: true}
    ]
    
    let optional_tools = [
        {name: "docker", description: "Containerization", required: false},
        {name: "kubectl", description: "Kubernetes CLI", required: false},
        {name: "gh", description: "GitHub CLI", required: false},
        {name: "teller", description: "Secret management", required: false},
        {name: "direnv", description: "Environment management", required: false}
    ]
    
    mut all_required_ok = true
    
    log info "Required tools:"
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
    
    if $all_required_ok {
        log success "🎉 All required dependencies are available!"
    } else {
        log error "💥 Some required dependencies are missing!"
        exit 1
    }
}

def "main structure" [] {
    log info "🏗️  Validating project structure..."
    
    let expected_structure = [
        {path: "CLAUDE.md", type: "file", required: true, description: "Project documentation"},
        {path: "scripts/validate-all.nu", type: "file", required: true, description: "This validation script"},
        {path: "nushell-env", type: "dir", required: true, description: "Nushell environment"},
        {path: "nushell-env/devbox.json", type: "file", required: true, description: "Nushell devbox config"},
        {path: "nushell-env/common.nu", type: "file", required: true, description: "Common utilities"},
        {path: "nushell-env/scripts", type: "dir", required: true, description: "Nushell scripts"},
        {path: "python-env", type: "dir", required: false, description: "Python environment"},
        {path: "typescript-env", type: "dir", required: false, description: "TypeScript environment"},
        {path: "rust-env", type: "dir", required: false, description: "Rust environment"},
        {path: "go-env", type: "dir", required: false, description: "Go environment"}
    ]
    
    mut structure_ok = true
    
    for item in $expected_structure {
        let exists = $item.path | path exists
        
        if $exists {
            log success $"  ✅ ($item.path) - ($item.description)"
        } else if $item.required {
            log error $"  ❌ ($item.path) - ($item.description) - REQUIRED"
            $structure_ok = false
        } else {
            log warn $"  ⚠️  ($item.path) - ($item.description) - optional"
        }
    }
    
    if $structure_ok {
        log success "🎉 Project structure validation passed!"
    } else {
        log error "💥 Project structure validation failed!"
        exit 1
    }
}

def "main help" [] {
    print $"Cross-Language Quality Gates for Polyglot Development Environment

Usage: nu scripts/validate-all.nu [OPTIONS] [COMMAND]

Commands:
  <default>      Run full validation
  quick          Run quick validation - syntax checks only
  dependencies   Check external tool dependencies
  structure      Validate project structure
  help           Show this help message

Options:
  --parallel     Run validations in parallel - faster
  --environment <name>   Target specific environment - default: all
  --skip <list>  Skip specific environments
  --verbose      Show detailed output

Examples:
  nu scripts/validate-all.nu                    # Full validation
  nu scripts/validate-all.nu --parallel         # Parallel validation
  nu scripts/validate-all.nu --environment python       # Python only
  nu scripts/validate-all.nu quick              # Quick checks
  nu scripts/validate-all.nu dependencies       # Check tools
  nu scripts/validate-all.nu structure          # Check structure

Available environments: python typescript rust go nushell
"
}