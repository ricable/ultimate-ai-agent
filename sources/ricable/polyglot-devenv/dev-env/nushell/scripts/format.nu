#!/usr/bin/env nu

# Format Nushell scripts with basic formatting rules
# Usage: nu scripts/format.nu [--check] [--fix]

use ../common.nu *

def main [
    --check = false
    --fix = false
] {
    if $check {
        log info "Checking format of Nushell scripts..."
        check-format
    } else if $fix {
        log info "Formatting Nushell scripts..."
        format-scripts
    } else {
        log info "Analyzing format of Nushell scripts..."
        analyze-format
    }
}

def format-scripts [] {
    let scripts = get-all-scripts
    mut formatted_count = 0
    
    for script in $scripts {
        log info $"Formatting ($script)..."
        
        let original_content = open $script
        let formatted_content = format-content $original_content
        
        if $formatted_content != $original_content {
            $formatted_content | save $script --force
            log success $"  ✅ Formatted ($script)"
            $formatted_count = $formatted_count + 1
        } else {
            log info $"  ℹ️  No changes needed for ($script)"
        }
    }
    
    log success $"Formatting complete: ($formatted_count) files updated"
}

def check-format [] {
    let scripts = get-all-scripts
    mut issues_found = false
    
    for script in $scripts {
        let original_content = open $script
        let formatted_content = format-content $original_content
        
        if $formatted_content != $original_content {
            log warn $"  ⚠️  Format issues in ($script)"
            $issues_found = true
        } else {
            log success $"  ✅ ($script) is properly formatted"
        }
    }
    
    if $issues_found {
        log warn "Format check failed - run with --fix to apply formatting"
        exit 1
    } else {
        log success "All scripts are properly formatted!"
    }
}

def analyze-format [] {
    let scripts = get-all-scripts
    
    for script in $scripts {
        log info $"Analyzing ($script)..."
        
        let content = open $script
        let lines = $content | lines
        
        # Basic stats
        let total_lines = $lines | length
        let blank_lines = $lines | where ($it | str trim | is-empty) | length
        let comment_lines = $lines | where ($it | str trim | str starts-with "#") | length
        let code_lines = $total_lines - $blank_lines - $comment_lines
        
        log info $"  Lines: ($code_lines) code, ($comment_lines) comments, ($blank_lines) blank"
        
        # Check for potential issues
        let long_lines = $lines | where ($it | str length) > 100
        if ($long_lines | length) > 0 {
            log warn $"  ⚠️  ($long_lines | length) lines longer than 100 characters"
        }
        
        let trailing_spaces = $lines | where ($it | str ends-with " ")
        if ($trailing_spaces | length) > 0 {
            log warn $"  ⚠️  ($trailing_spaces | length) lines with trailing spaces"
        }
        
        print ""
    }
}

def format-content [content: string] {
    let lines = $content | lines
    
    # Apply formatting rules
    let formatted_lines = $lines | each { |line|
        # Remove trailing whitespace
        let trimmed = $line | str trim --right
        
        # Normalize indentation (basic)
        if ($trimmed | str starts-with "  ") {
            $trimmed
        } else if ($trimmed | str starts-with "\t") {
            $trimmed | str replace "\t" "    "
        } else {
            $trimmed
        }
    }
    
    # Join lines back together
    $formatted_lines | str join "\n"
}

def get-all-scripts [] {
    let script_files = ls scripts/*.nu | get name
    let common_file = if ("common.nu" | path exists) { ["common.nu"] } else { [] }
    
    $script_files | append $common_file
}

# Specific formatting functions
def "main format-json" [] {
    log info "Formatting JSON configuration files..."
    
    let json_files = ["devbox.json"]
    
    for file in $json_files {
        if ($file | path exists) {
            log info $"Formatting ($file)..."
            
            try {
                let content = open $file | from json
                $content | to json --indent 2 | save $file --force
                log success $"  ✅ Formatted ($file)"
            } catch { |e|
                log error $"  ❌ Failed to format ($file): ($e.msg)"
            }
        }
    }
}

def "main format-yaml" [] {
    log info "Formatting YAML configuration files..."
    
    let yaml_files = [".teller.yml"]
    
    for file in $yaml_files {
        if ($file | path exists) {
            log info $"Formatting ($file)..."
            
            try {
                let content = open $file | from yaml
                $content | to yaml | save $file --force
                log success $"  ✅ Formatted ($file)"
            } catch { |e|
                log error $"  ❌ Failed to format ($file): ($e.msg)"
            }
        }
    }
}

def "main format-all" [] {
    log info "Formatting all configuration and script files..."
    
    format-scripts
    main format-json
    main format-yaml
    
    log success "All formatting complete!"
}