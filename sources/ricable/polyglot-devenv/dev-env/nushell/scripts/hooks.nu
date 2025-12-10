#!/usr/bin/env nu
# hooks.nu - Claude Code hooks management for polyglot environment

# Main command dispatcher
def main [] {
    print "Claude Code Hooks Management for Polyglot Environment"
    print "===================================================="
    print ""
    print "Commands:"
    print "  nu hooks.nu generate      - Generate hooks from CLAUDE.md rules"
    print "  nu hooks.nu validate      - Validate existing hook configuration"
    print "  nu hooks.nu install       - Install hooks to project or user settings"
    print "  nu hooks.nu backup        - Backup current hook configuration"
    print "  nu hooks.nu restore       - Restore hook configuration from backup"
    print "  nu hooks.nu test          - Test hook functionality"
    print "  nu hooks.nu status        - Show current hooks status"
}

# Generate hooks from CLAUDE.md rules
def "main generate" [
    --project = true    # Generate project-specific hooks
    --user = false      # Generate user-global hooks
    --dry-run = false   # Show what would be generated without applying
] {
    print "🔧 Generating Claude Code hooks from CLAUDE.md rules..."
    
    let rules = extract_rules_from_claude_md
    
    if ($rules | length) == 0 {
        print "❌ No automation rules found in CLAUDE.md"
        return
    }
    
    print $"📋 Found ($rules | length) automation rules:"
    $rules | each { |rule| print $"  - ($rule)" }
    print ""
    
    let hook_config = generate_polyglot_hooks $rules
    
    if $dry_run {
        print "🔍 Dry-run mode - Generated configuration:"
        print ($hook_config | to json --indent 2)
        return
    }
    
    let settings_file = if $project { ".claude/settings.json" } else { $"($env.HOME)/.claude/settings.json" }
    
    install_hooks $hook_config $settings_file
    print $"✅ Hooks installed to ($settings_file)"
}

# Extract automation rules from CLAUDE.md
def extract_rules_from_claude_md [] {
    let claude_files = [
        "./CLAUDE.md"
        "./CLAUDE.local.md"
        $"($env.HOME)/.claude/CLAUDE.md"
    ]
    
    let rules = ($claude_files | each { |file|
        if ($file | path exists) {
            print $"📖 Reading rules from ($file)"
            let content = open $file
            
            # Extract automation-related patterns
            let automation_patterns = [
                "format.*after.*edit"
                "run.*test.*after.*modifying"
                "lint.*before.*commit"
                "validate.*when.*finish"
                "check.*before.*save"
                "execute.*when.*done"
            ]
            
            $automation_patterns | each { |pattern|
                $content | lines | where $it =~ $pattern
            } | flatten
        } else {
            []
        }
    } | flatten)
    
    $rules | uniq
}

# Generate polyglot-aware hook configuration
def generate_polyglot_hooks [rules: list] {
    let base_config = {
        hooks: {
            PreToolUse: []
            PostToolUse: []
            Stop: []
        }
    }
    
    let hooks_list = ($rules | each { |rule|
        analyze_rule_for_hooks $rule
    } | where $it != null)
    
    let final_config = ($hooks_list | reduce -f $base_config { |hook, acc|
        $acc | upsert hooks ($acc.hooks | upsert $hook.event ($acc.hooks | get $hook.event | append $hook.config))
    })
    
    $final_config
}

# Analyze a rule and generate appropriate hook configuration
def analyze_rule_for_hooks [rule: string] {
    let rule_lower = ($rule | str downcase)
    
    # Determine hook event
    let event = if ($rule_lower =~ "before|check|validate|prevent|scan") {
        "PreToolUse"
    } else if ($rule_lower =~ "after|following|once.*done|when.*finished") {
        "PostToolUse"
    } else if ($rule_lower =~ "finish|complete|end.*task|done|wrap.*up") {
        "Stop"
    } else {
        "PostToolUse"  # Default
    }
    
    # Generate environment-aware command
    let command = if ($rule_lower =~ "format") {
        generate_format_command
    } else if ($rule_lower =~ "test") {
        generate_test_command
    } else if ($rule_lower =~ "lint") {
        generate_lint_command
    } else if ($rule_lower =~ "validate.*all|cross.*language") {
        "nu nushell-env/scripts/validate-all.nu 2>/dev/null || true"
    } else {
        $"echo 'Hook triggered for rule: ($rule)'"
    }
    
    # Determine matcher
    let matcher = if $event == "Stop" {
        null
    } else {
        "Edit|MultiEdit|Write"
    }
    
    {
        event: $event
        config: {
            matcher: $matcher
            hooks: [{
                type: "command"
                command: $command
            }]
        }
    }
}

# Generate environment-aware format command
def generate_format_command [] {
    "bash -c 'file_path=$(echo \"$0\" | jq -r \".tool_input.file_path // \\\"\\\"\" 2>/dev/null || echo \"\"); if [[ \"$file_path\" =~ \\.py$ ]] || [[ \"$PWD\" =~ python-env ]]; then cd python-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.(ts|js)$ ]] || [[ \"$PWD\" =~ typescript-env ]]; then cd typescript-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.rs$ ]] || [[ \"$PWD\" =~ rust-env ]]; then cd rust-env && devbox run format --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.go$ ]] || [[ \"$PWD\" =~ go-env ]]; then cd go-env && devbox run format --quiet 2>/dev/null || true; fi'"
}

# Generate environment-aware test command
def generate_test_command [] {
    "bash -c 'file_path=$(echo \"$0\" | jq -r \".tool_input.file_path // \\\"\\\"\" 2>/dev/null || echo \"\"); if [[ \"$file_path\" =~ test.*\\.py$ ]]; then cd python-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*\\.(test|spec)\\.(ts|js)$ ]]; then cd typescript-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*_test\\.rs$ ]]; then cd rust-env && devbox run test --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ .*_test\\.go$ ]]; then cd go-env && devbox run test --quiet 2>/dev/null || true; fi'"
}

# Generate environment-aware lint command
def generate_lint_command [] {
    "bash -c 'file_path=$(echo \"$0\" | jq -r \".tool_input.file_path // \\\"\\\"\" 2>/dev/null || echo \"\"); if [[ \"$file_path\" =~ \\.py$ ]] || [[ \"$PWD\" =~ python-env ]]; then cd python-env && devbox run lint --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.(ts|js)$ ]] || [[ \"$PWD\" =~ typescript-env ]]; then cd typescript-env && devbox run lint --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.rs$ ]] || [[ \"$PWD\" =~ rust-env ]]; then cd rust-env && devbox run lint --quiet 2>/dev/null || true; elif [[ \"$file_path\" =~ \\.go$ ]] || [[ \"$PWD\" =~ go-env ]]; then cd go-env && devbox run lint --quiet 2>/dev/null || true; fi'"
}

# Install hooks to settings file
def install_hooks [config: record, settings_file: string] {
    # Create directory if it doesn't exist
    let settings_dir = ($settings_file | path dirname)
    mkdir $settings_dir
    
    # Backup existing settings
    if ($settings_file | path exists) {
        let backup_file = $"($settings_file).backup-(date now | format date '%Y%m%d_%H%M%S')"
        cp $settings_file $backup_file
        print $"💾 Backed up existing settings to ($backup_file)"
    }
    
    # Merge with existing settings
    let existing_settings = if ($settings_file | path exists) {
        open $settings_file
    } else {
        {}
    }
    
    let merged_settings = $existing_settings | upsert hooks $config.hooks
    $merged_settings | to json --indent 2 | save --force $settings_file
}

# Validate hook configuration
def "main validate" [
    --file: string = ""  # Specific file to validate (default: project settings)
] {
    let settings_file = if $file != "" {
        $file
    } else if (".claude/settings.json" | path exists) {
        ".claude/settings.json"
    } else {
        $"($env.HOME)/.claude/settings.json"
    }
    
    if not ($settings_file | path exists) {
        print $"❌ Settings file not found: ($settings_file)"
        return
    }
    
    print $"🔍 Validating hooks in ($settings_file)"
    
    try {
        let settings = open $settings_file
        
        if not ("hooks" in $settings) {
            print "⚠️  No hooks configuration found"
            return
        }
        
        let hooks = $settings.hooks
        
        let events_data = (["PreToolUse", "PostToolUse", "Stop", "Notification"] | each { |event|
            if $event in $hooks {
                let event_hooks = $hooks | get $event
                print $"📌 ($event): ($event_hooks | length) configurations"
                
                let hook_count = ($event_hooks | each { |config|
                    let matcher = $config | get matcher? | default "All tools"
                    let hook_list = $config | get hooks
                    print $"  Matcher: ($matcher)"
                    
                    $hook_list | each { |hook|
                        let command = $hook | get command
                        print $"    ✅ Command: ($command | str substring 0..60)..."
                    } | length
                } | math sum)
                
                $hook_count
            } else {
                0
            }
        })
        
        let total_hooks = ($events_data | math sum)
        print $"📊 Total: ($total_hooks) hooks configured"
        print "✅ Validation passed"
        
    } catch {
        print "❌ Failed to parse settings file - invalid JSON"
    }
}

# Show current hooks status
def "main status" [] {
    print "Claude Code Hooks Status"
    print "========================"
    print ""
    
    # Check project settings
    if (".claude/settings.json" | path exists) {
        print "📁 Project hooks: .claude/settings.json"
        main validate --file ".claude/settings.json"
    } else {
        print "📁 Project hooks: Not configured"
    }
    
    print ""
    
    # Check user settings
    let user_settings = $"($env.HOME)/.claude/settings.json"
    if ($user_settings | path exists) {
        print $"👤 User hooks: ($user_settings)"
        main validate --file $user_settings
    } else {
        print "👤 User hooks: Not configured"
    }
    
    print ""
    print "Environment status:"
    
    # Check each environment
    let environments = [
        {name: "Python", dir: "python-env", file: "devbox.json"}
        {name: "TypeScript", dir: "typescript-env", file: "devbox.json"}
        {name: "Rust", dir: "rust-env", file: "devbox.json"}
        {name: "Go", dir: "go-env", file: "devbox.json"}
    ]
    
    $environments | each { |environment|
        let env_path = $"($environment.dir)/($environment.file)"
        if ($env_path | path exists) {
            print $"  ✅ ($environment.name): Ready (($environment.dir))"
        } else {
            print $"  ❌ ($environment.name): Not found (($environment.dir))"
        }
    } | ignore
}

# Test hook functionality
def "main test" [
    --hook-type: string = "format"  # Type of hook to test (format, lint, test)
] {
    print $"🧪 Testing ($hook_type) hook functionality..."
    
    match $hook_type {
        "format" => {
            print "Testing format commands in each environment:"
            test_environment_command "python-env" "format"
            test_environment_command "typescript-env" "format"
            test_environment_command "rust-env" "format"
            test_environment_command "go-env" "format"
        }
        "lint" => {
            print "Testing lint commands in each environment:"
            test_environment_command "python-env" "lint"
            test_environment_command "typescript-env" "lint"
            test_environment_command "rust-env" "lint"
            test_environment_command "go-env" "lint"
        }
        "test" => {
            print "Testing test commands in each environment:"
            test_environment_command "python-env" "test"
            test_environment_command "typescript-env" "test"
            test_environment_command "rust-env" "test"
            test_environment_command "go-env" "test"
        }
        _ => {
            print $"❌ Unknown hook type: ($hook_type)"
            print "Available types: format, lint, test"
        }
    }
}

# Test command in specific environment
def test_environment_command [env_dir: string, command: string] {
    if ($env_dir | path exists) {
        print $"  Testing ($env_dir) ($command)..."
        let result = do -i { 
            cd $env_dir
            devbox run $command --help 2>/dev/null 
        }
        
        if $result != null {
            print $"    ✅ ($command) command available"
        } else {
            print $"    ⚠️  ($command) command not found or not working"
        }
    } else {
        print $"    ❌ Environment ($env_dir) not found"
    }
}

# Backup current hook configuration
def "main backup" [
    --name: string = ""  # Custom backup name
] {
    let timestamp = (date now | format date '%Y%m%d_%H%M%S')
    let backup_name = if $name != "" { $name } else { $"hooks_backup_($timestamp)" }
    let backup_dir = ".claude/backups"
    
    mkdir $backup_dir
    
    # Backup project settings
    if (".claude/settings.json" | path exists) {
        cp ".claude/settings.json" $"($backup_dir)/($backup_name)_project.json"
        print $"💾 Project hooks backed up to ($backup_dir)/($backup_name)_project.json"
    }
    
    # Backup user settings
    let user_settings = $"($env.HOME)/.claude/settings.json"
    if ($user_settings | path exists) {
        cp $user_settings $"($backup_dir)/($backup_name)_user.json"
        print $"💾 User hooks backed up to ($backup_dir)/($backup_name)_user.json"
    }
    
    print $"✅ Backup completed: ($backup_name)"
}

# Restore hook configuration from backup
def "main restore" [
    backup_name: string  # Name of backup to restore
    --scope: string = "project"  # Restore scope: project, user, or both
] {
    let backup_dir = ".claude/backups"
    
    match $scope {
        "project" => {
            let backup_file = $"($backup_dir)/($backup_name)_project.json"
            if ($backup_file | path exists) {
                cp $backup_file ".claude/settings.json"
                print $"✅ Project hooks restored from ($backup_file)"
            } else {
                print $"❌ Backup file not found: ($backup_file)"
            }
        }
        "user" => {
            let backup_file = $"($backup_dir)/($backup_name)_user.json"
            let user_settings = $"($env.HOME)/.claude/settings.json"
            if ($backup_file | path exists) {
                cp $backup_file $user_settings
                print $"✅ User hooks restored from ($backup_file)"
            } else {
                print $"❌ Backup file not found: ($backup_file)"
            }
        }
        "both" => {
            main restore $backup_name --scope "project"
            main restore $backup_name --scope "user"
        }
        _ => {
            print $"❌ Invalid scope: ($scope). Use 'project', 'user', or 'both'"
        }
    }
}