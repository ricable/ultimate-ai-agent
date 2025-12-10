#!/usr/bin/env nu

# Checker for Nushell scripts - validates syntax and best practices
# Usage: nu scripts/check.nu [--fix] [--verbose]

use ../common.nu *

def main [
    --fix = false
    --verbose = false
] {
    log info "Checking Nushell scripts for syntax and best practices..."
    
    let scripts = ls scripts/*.nu | get name
    mut issues_found = false
    
    for script in $scripts {
        log info $"Checking ($script)..."
        
        # Syntax check
        let syntax_ok = check-syntax $script $verbose
        
        # Best practices check
        let practices_ok = check-best-practices $script $verbose $fix
        
        if not ($syntax_ok and $practices_ok) {
            $issues_found = true
        }
    }
    
    # Check common.nu
    log info "Checking common.nu..."
    let common_ok = check-syntax "common.nu" $verbose
    if not $common_ok {
        $issues_found = true
    }
    
    # Check configuration files
    check-config-files $verbose
    
    if $issues_found {
        log error "Issues found in Nushell scripts"
        exit 1
    } else {
        log success "All checks passed!"
    }
}

def check-syntax [script_path: string, verbose: bool] {
    log info $"  Syntax check: ($script_path)"
    
    try {
        nu --ide-check 10 $script_path | ignore
        log success $"    ✅ Syntax OK"
        true
    } catch { |e|
        log error $"    ❌ Syntax Error: ($e.msg)"
        if $verbose {
            log error $"      In file: ($script_path)"
        }
        false
    }
}

def check-best-practices [script_path: string, verbose: bool, fix: bool] {
    log info $"  Best practices check: ($script_path)"
    
    let content = open $script_path
    mut issues = []
    mut fixed_content = $content
    
    # Check for shebang
    if not ($content | str starts-with "#!/usr/bin/env nu") {
        $issues = $issues | append "Missing shebang line"
        if $fix {
            $fixed_content = $"#!/usr/bin/env nu\n\n($content)"
        }
    }
    
    # Check for proper function documentation
    let functions = $content | lines | where ($it | str contains "def ") and not ($it | str contains "#")
    if ($functions | length) > 0 {
        for func in $functions {
            if not ($content | str contains $"# ($func | str replace 'def ' '' | str replace ' []' '' | str replace ' [' '')") {
                $issues = $issues | append $"Function ($func) missing documentation"
            }
        }
    }
    
    # Check for proper error handling
    let has_error_handling = ($content | str contains "try") or ($content | str contains "do --ignore-errors")
    if not $has_error_handling and ($content | str contains "bash -c") {
        $issues = $issues | append "External commands should use proper error handling"
    }
    
    # Check for hardcoded paths
    let hardcoded_paths = $content | lines | where ($it | str contains "/home/") or ($it | str contains "/Users/")
    if ($hardcoded_paths | length) > 0 {
        $issues = $issues | append "Hardcoded paths found - use environment variables instead"
    }
    
    # Check for secret handling
    let potential_secrets = $content | lines | where ($it | str contains "password") or ($it | str contains "token") or ($it | str contains "key")
    if ($potential_secrets | length) > 0 {
        let secure_input = $content | str contains "--suppress-output"
        if not $secure_input {
            $issues = $issues | append "Potential secrets found - use --suppress-output for sensitive input"
        }
    }
    
    # Apply fixes if requested
    if $fix and ($fixed_content != $content) {
        $fixed_content | save $script_path --force
        log info $"    🔧 Applied automatic fixes"
    }
    
    # Report issues
    if ($issues | length) > 0 {
        log warn $"    ⚠️  Issues found:"
        for issue in $issues {
            log warn $"      - ($issue)"
        }
        if $verbose {
            log info $"      Use --fix to automatically fix some issues"
        }
        false
    } else {
        log success $"    ✅ Best practices OK"
        true
    }
}

def check-config-files [verbose: bool] {
    log info "Checking configuration files..."
    
    # Check devbox.json
    if ("devbox.json" | path exists) {
        try {
            open devbox.json | ignore
            log success "  ✅ devbox.json is readable"
        } catch { |e|
            log error $"  ❌ devbox.json is invalid: ($e.msg)"
        }
    } else {
        log warn "  ⚠️  devbox.json not found"
    }
    
    # Check .teller.yml
    if (".teller.yml" | path exists) {
        try {
            open .teller.yml | ignore
            log success "  ✅ .teller.yml is readable"
        } catch { |e|
            log error $"  ❌ .teller.yml is invalid: ($e.msg)"
        }
    } else {
        log warn "  ⚠️  .teller.yml not found"
    }
    
    # Check .env file format
    if (".env" | path exists) {
        let env_content = open .env
        let invalid_lines = $env_content | lines | where not ($it | str starts-with "#") and not ($it | str starts-with "export ") and not ($it | str trim | is-empty)
        
        if ($invalid_lines | length) > 0 {
            log warn "  ⚠️  .env file has non-standard format lines:"
            for line in $invalid_lines {
                log warn $"    ($line)"
            }
        } else {
            log success "  ✅ .env file format OK"
        }
    }
}

# Specific checks for common patterns
def "main check-imports" [] {
    log info "Checking import statements..."
    
    let scripts = ls scripts/*.nu | get name
    
    for script in $scripts {
        let content = open $script
        let imports = $content | lines | where ($it | str contains "use ")
        
        for import in $imports {
            let module = $import | str replace "use " "" | str replace " *" "" | str trim
            if ($module | str starts-with "../") {
                let module_path = $module
                if not ($module_path | path exists) {
                    log error $"  ❌ ($script): Import not found: ($module_path)"
                } else {
                    log success $"  ✅ ($script): Import OK: ($module_path)"
                }
            }
        }
    }
}

def "main check-permissions" [] {
    log info "Checking file permissions..."
    
    let scripts = ls scripts/*.nu | get name
    
    for script in $scripts {
        let perms = ls -l $script | get permissions.0
        if ($perms | str contains "x") {
            log success $"  ✅ ($script): Executable"
        } else {
            log warn $"  ⚠️  ($script): Not executable (consider chmod +x)"
        }
    }
}

def "main check-dependencies" [] {
    log info "Checking external dependencies..."
    
    let required_commands = ["git", "nu"]
    let optional_commands = ["teller", "devbox"]
    
    for cmd in $required_commands {
        if (cmd exists $cmd) {
            log success $"  ✅ Required: ($cmd)"
        } else {
            log error $"  ❌ Required: ($cmd) not found"
        }
    }
    
    for cmd in $optional_commands {
        if (cmd exists $cmd) {
            log success $"  ✅ Optional: ($cmd)"
        } else {
            log warn $"  ⚠️  Optional: ($cmd) not found"
        }
    }
}