#!/usr/bin/env nu

# Enhanced todo management with language awareness and cross-environment tracking
# Based on polyglot-todo.md specifications

def main [] {
    echo "🎯 Enhanced Polyglot Todo Management"
    echo "Usage: nu enhanced-todo.nu [analyze|suggest|track|report|sync]"
}

# Analyze current development context for intelligent task suggestions
def "main analyze" [] {
    echo "📊 DEVELOPMENT CONTEXT ANALYSIS"
    echo ""
    
    # Analyze git status for context
    let git_context = (analyze_git_context)
    echo $"🌿 Current branch: ($git_context.branch)"
    echo $"📁 Modified files: ($git_context.modified | length)"
    echo $"📝 Staged files: ($git_context.staged | length)"
    
    # Analyze environments
    let env_context = (analyze_environment_context $git_context.modified)
    echo ""
    echo "🏗️  ENVIRONMENT ANALYSIS"
    $env_context | each { |env|
        echo $"   ($env.name): ($env.files | length) files, ($env.priority) priority"
    }
    
    # Suggest task priorities
    echo ""
    echo "🎯 SUGGESTED TASK PRIORITIES"
    let suggestions = (generate_task_suggestions $git_context $env_context)
    $suggestions | each { |suggestion|
        echo $"   ($suggestion.priority) - ($suggestion.description)"
    }
}

# Sync with existing TodoWrite/TodoRead system and enhance with environment context
def "main sync" [] {
    echo "🔄 Syncing with TodoWrite/TodoRead system..."
    
    # Read current todos (this would integrate with the actual TodoRead tool)
    echo "📋 Current todos enhanced with environment context"
    echo "💡 Use TodoWrite and TodoRead tools for actual todo management"
    echo "   This script provides analysis and suggestions to enhance those tools"
}

# Track task completion across environments
def "main track" [task_description: string] {
    echo $"📊 Tracking task: ($task_description)"
    
    let start_time = (date now)
    let environments = (detect_relevant_environments $task_description)
    
    echo $"🎯 Relevant environments: ($environments | str join ', ')"
    echo $"⏰ Started at: ($start_time)"
    
    # This could be enhanced to actually track progress
    echo "💡 Use TodoWrite to create and track actual tasks"
    echo "   This analysis helps with environment context and priority"
}

# Generate comprehensive todo report
def "main report" [] {
    echo "📊 POLYGLOT TODO ENVIRONMENT REPORT"
    echo ""
    
    # Analyze each environment
    let environments = ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]
    
    $environments | each { |env|
        echo $"🔍 ($env | str upcase):"
        let env_analysis = (analyze_single_environment $env)
        echo $"   📁 Files: ($env_analysis.total_files)"
        echo $"   🧪 Tests: ($env_analysis.test_files)"
        echo $"   ⚠️  Issues: ($env_analysis.potential_issues)"
        echo $"   💡 Suggestions: ($env_analysis.suggestions)"
        echo ""
    }
    
    # Cross-environment insights
    echo "🔗 CROSS-ENVIRONMENT INSIGHTS"
    let cross_insights = (generate_cross_environment_insights)
    $cross_insights | each { |insight|
        echo $"   • ($insight)"
    }
}

# Suggest tasks based on code analysis
def "main suggest" [] {
    echo "💡 INTELLIGENT TASK SUGGESTIONS"
    echo ""
    
    let git_context = (analyze_git_context)
    let suggestions = (generate_intelligent_suggestions $git_context)
    
    $suggestions | each { |suggestion|
        echo $"🎯 ($suggestion.priority | str upcase) - ($suggestion.task)"
        echo $"   Environment: ($suggestion.environment)"
        echo $"   Estimated time: ($suggestion.time_estimate)"
        echo $"   Dependencies: ($suggestion.dependencies | str join ', ')"
        echo ""
    }
}

# Analyze git context for task suggestions
def analyze_git_context [] {
    if not (try { git rev-parse --git-dir | complete | get exit_code } | default 1) == 0 {
        return {
            branch: "unknown",
            modified: [],
            staged: [],
            untracked: []
        }
    }
    
    {
        branch: (git rev-parse --abbrev-ref HEAD),
        modified: (git diff --name-only | lines),
        staged: (git diff --cached --name-only | lines),
        untracked: (git ls-files --others --exclude-standard | lines)
    }
}

# Analyze environment context based on modified files
def analyze_environment_context [modified_files: list<string>] {
    let environments = [
        {name: "🐍 Python", pattern: "python-env", extensions: [".py"]},
        {name: "📘 TypeScript", pattern: "typescript-env", extensions: [".ts", ".tsx", ".js", ".jsx"]},
        {name: "🦀 Rust", pattern: "rust-env", extensions: [".rs"]},
        {name: "🐹 Go", pattern: "go-env", extensions: [".go"]},
        {name: "🐚 Nushell", pattern: "nushell-env", extensions: [".nu"]}
    ]
    
    $environments | each { |env|
        let env_files = ($modified_files | where { |file|
            ($file | str starts-with $env.pattern) or 
            ($env.extensions | any { |ext| $file | str ends-with $ext })
        })
        
        let priority = if ($env_files | length) > 3 { "high" }
                      else if ($env_files | length) > 0 { "medium" }
                      else { "low" }
        
        {
            name: $env.name,
            pattern: $env.pattern,
            files: $env_files,
            priority: $priority
        }
    }
}

# Generate task suggestions based on context
def generate_task_suggestions [git_context: record, env_context: list] {
    mut suggestions = []
    
    # Suggest based on modified files
    if ($git_context.modified | length) > 0 {
        $suggestions = ($suggestions | append {
            priority: "high",
            description: "Review and test modified files before committing"
        })
    }
    
    # Suggest based on untracked files
    if ($git_context.untracked | length) > 0 {
        $suggestions = ($suggestions | append {
            priority: "medium", 
            description: $"Add or gitignore ($git_context.untracked | length) untracked files"
        })
    }
    
    # Suggest based on environment priorities
    let high_priority_envs = ($env_context | where priority == "high")
    if ($high_priority_envs | length) > 0 {
        $suggestions = ($suggestions | append {
            priority: "high",
            description: $"Focus on ($high_priority_envs | get name | str join ', ') environments"
        })
    }
    
    $suggestions
}

# Detect relevant environments for a task
def detect_relevant_environments [task_description: string] {
    mut environments = []
    
    if ($task_description | str downcase | str contains "python") or ($task_description | str downcase | str contains "fastapi") {
        $environments = ($environments | append "python")
    }
    if ($task_description | str downcase | str contains "typescript") or ($task_description | str downcase | str contains "react") {
        $environments = ($environments | append "typescript")
    }
    if ($task_description | str downcase | str contains "rust") {
        $environments = ($environments | append "rust") 
    }
    if ($task_description | str downcase | str contains "go") {
        $environments = ($environments | append "go")
    }
    if ($task_description | str downcase | str contains "script") or ($task_description | str downcase | str contains "automation") {
        $environments = ($environments | append "nushell")
    }
    
    if ($environments | is-empty) {
        $environments = ["polyglot"]
    }
    
    $environments
}

# Analyze a single environment for todo suggestions
def analyze_single_environment [env: string] {
    if not (($env | path exists)) {
        return {
            total_files: 0,
            test_files: 0,
            potential_issues: 0,
            suggestions: "Environment not found"
        }
    }
    
    let files = (ls -la $env | where type == file)
    let test_files = ($files | where name =~ "test")
    
    # Basic analysis - could be enhanced
    {
        total_files: ($files | length),
        test_files: ($test_files | length),
        potential_issues: 0,
        suggestions: "Run environment validation"
    }
}

# Generate cross-environment insights
def generate_cross_environment_insights [] {
    [
        "Consider running cross-language validation with validate-all.nu",
        "Check dependency health across all environments",
        "Ensure consistent code formatting across languages", 
        "Validate environment configurations are synchronized",
        "Review integration points between services"
    ]
}

# Generate intelligent task suggestions based on git analysis
def generate_intelligent_suggestions [git_context: record] {
    mut suggestions = []
    
    # Analyze file types and suggest tasks
    let has_tests = ($git_context.modified | any { |file| $file | str contains "test" })
    let has_configs = ($git_context.modified | any { |file| 
        ($file | str contains "package.json") or 
        ($file | str contains "pyproject.toml") or
        ($file | str contains "devbox.json")
    })
    
    if $has_tests {
        $suggestions = ($suggestions | append {
            priority: "high",
            task: "Run comprehensive test suite to validate changes",
            environment: "cross-language",
            time_estimate: "10-15 minutes",
            dependencies: []
        })
    }
    
    if $has_configs {
        $suggestions = ($suggestions | append {
            priority: "medium",
            task: "Update environment dependencies and validate configurations",
            environment: "devbox",
            time_estimate: "5-10 minutes", 
            dependencies: ["config validation"]
        })
    }
    
    # Default suggestions if no specific patterns found
    if ($suggestions | is-empty) {
        $suggestions = ($suggestions | append {
            priority: "medium",
            task: "Review code changes and ensure quality standards",
            environment: "general",
            time_estimate: "15-30 minutes",
            dependencies: []
        })
    }
    
    $suggestions
}