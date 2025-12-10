#!/usr/bin/env nu

# Test runner for Nushell scripts and environment validation
# Usage: nu scripts/test.nu [--verbose] [--parallel]

use ../common.nu *

def main [
    --verbose = false
    --parallel = false
] {
    log info "Running Nushell environment tests..."
    
    if $parallel {
        run-tests-parallel $verbose
    } else {
        run-tests-sequential $verbose
    }
}

def run-tests-sequential [verbose: bool] {
    let tests = get-test-suite
    mut passed = 0
    mut failed = 0
    
    for test in $tests {
        log info $"Running test: ($test.name)"
        
        let result = try {
            do $test.test
            true
        } catch { |e|
            if $verbose {
                log error $"Test failed: ($e.msg)"
            }
            false
        }
        
        if $result {
            log success $"✅ ($test.name)"
            $passed = $passed + 1
        } else {
            log error $"❌ ($test.name)"
            $failed = $failed + 1
        }
    }
    
    print ""
    log info $"Test Results: ($passed) passed, ($failed) failed"
    
    if $failed > 0 {
        exit 1
    }
}

def run-tests-parallel [verbose: bool] {
    log info "Running tests in parallel..."
    
    let tests = get-test-suite
    
    let results = $tests | par-each { |test|
        let result = try {
            do $test.test
            {name: $test.name, status: "passed", error: null}
        } catch { |e|
            {name: $test.name, status: "failed", error: $e.msg}
        }
    }
    
    let passed = $results | where status == "passed" | length
    let failed = $results | where status == "failed" | length
    
    for result in $results {
        if $result.status == "passed" {
            log success $"✅ ($result.name)"
        } else {
            log error $"❌ ($result.name)"
            if $verbose and $result.error != null {
                log error $"  Error: ($result.error)"
            }
        }
    }
    
    print ""
    log info $"Parallel Test Results: ($passed) passed, ($failed) failed"
    
    if $failed > 0 {
        exit 1
    }
}

def get-test-suite [] {
    [
        {
            name: "Nushell version check"
            test: { nu --version | str contains "0.10" }
        }
        {
            name: "Common module import"
            test: { 
                try { use ../common.nu *; true } catch { false }
            }
        }
        {
            name: "Environment variables access"
            test: { $env.HOME | is-not-empty }
        }
        {
            name: "Git availability"
            test: { cmd exists "git" }
        }
        {
            name: "Devbox configuration"
            test: { "devbox.json" | path exists }
        }
        {
            name: "Scripts directory structure"
            test: { 
                (("scripts" | path exists) and 
                ("scripts/setup.nu" | path exists) and
                ("scripts/list.nu" | path exists))
            }
        }
        {
            name: "Common utilities"
            test: { 
                use ../common.nu *
                ((log info "test" | is-not-empty) and
                (env get-or-prompt "HOME" "test" | is-not-empty))
            }
        }
        {
            name: "Configuration directory"
            test: { "config" | path exists }
        }
        {
            name: "Teller configuration"
            test: { ".teller.yml" | path exists }
        }
        {
            name: "JSON handling"
            test: { 
                let test_data = {name: "test", value: 42}
                let json_str = $test_data | to json
                let parsed = $json_str | from json
                $parsed.name == "test" and $parsed.value == 42
            }
        }
        {
            name: "Pipeline operations"
            test: { 
                ([1, 2, 3, 4, 5] | where $it > 3 | length) == 2
            }
        }
        {
            name: "File operations"
            test: { 
                let temp_file = "tmp/test.txt"
                if not ("tmp" | path exists) { mkdir tmp }
                "test content" | save $temp_file --force
                let content = open $temp_file
                rm $temp_file
                $content == "test content"
            }
        }
    ]
}

# Helper function to test external commands
def test-external-command [cmd: string] {
    try {
        (which $cmd | length) > 0
    } catch {
        false
    }
}

# Test specific script functionality
def "main test-script" [script_name: string] {
    let script_path = $"scripts/($script_name)"
    
    if not ($script_path | path exists) {
        log error $"Script not found: ($script_path)"
        exit 1
    }
    
    log info $"Testing script: ($script_name)"
    
    try {
        nu --check $script_path
        log success $"✅ Syntax check passed for ($script_name)"
    } catch { |e|
        log error $"❌ Syntax check failed for ($script_name): ($e.msg)"
        exit 1
    }
}

# Test all scripts syntax
def "main test-all-scripts" [] {
    log info "Testing all script syntax..."
    
    let scripts = ls scripts/*.nu | get name
    let results = $scripts | each { |script|
        try {
            nu --ide-check 10 $script | ignore
            {script: $script, status: "passed"}
        } catch { |e|
            {script: $script, status: "failed", error: $e.msg}
        }
    }
    
    let failed = $results | where status == "failed"
    
    for result in $results {
        if $result.status == "passed" {
            log success $"✅ ($result.script)"
        } else {
            log error $"❌ ($result.script): ($result.error)"
        }
    }
    
    if ($failed | length) > 0 {
        exit 1
    }
    
    log success "All scripts passed syntax validation!"
}