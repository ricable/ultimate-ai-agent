#!/usr/bin/env nu

# Intelligent commit message generation for polyglot development environments
# Based on polyglot-commit.md specifications

def main [] {
    echo "🎯 Intelligent Commit Message Generator"
    echo "Usage: nu intelligent-commit.nu [analyze|suggest|validate]"
}

# Analyze staged changes and suggest commit message
def "main suggest" [] {
    let changes = (git diff --cached --name-only | lines)
    
    if ($changes | is-empty) {
        echo "❌ No staged changes found. Stage your changes first with 'git add'"
        return
    }
    
    let analysis = (analyze_changes $changes)
    let commit_type = (detect_commit_type $changes)
    let scope = (detect_scope $analysis.languages $analysis.directories)
    let message = (generate_commit_message $commit_type $scope $analysis)
    
    echo "📊 STAGED CHANGES ANALYSIS"
    echo $"Languages: ($analysis.languages | str join ', ')"
    echo $"Directories: ($analysis.directories | str join ', ')"
    echo $"File types: ($analysis.file_types | str join ', ')"
    echo ""
    echo "🎯 SUGGESTED COMMIT MESSAGE:"
    echo $"($message)"
    echo ""
    echo "🔧 Git Command:"
    echo $"git commit -m \"($message)\""
}

# Analyze staged changes for language and type detection
def analyze_changes [changes: list<string>] {
    let languages = ($changes | each { |file|
        if ($file | str ends-with ".py") { "python" }
        else if (($file | str ends-with ".ts") or ($file | str ends-with ".tsx") or ($file | str ends-with ".js") or ($file | str ends-with ".jsx")) { "typescript" }
        else if ($file | str ends-with ".rs") { "rust" }
        else if ($file | str ends-with ".go") { "go" }
        else if ($file | str ends-with ".nu") { "nushell" }
        else { null }
    } | where $it != null | uniq)
    
    let directories = ($changes | each { |file|
        if ($file | str starts-with "python-env/") { "python-env" }
        else if ($file | str starts-with "typescript-env/") { "typescript-env" }
        else if ($file | str starts-with "rust-env/") { "rust-env" }
        else if ($file | str starts-with "go-env/") { "go-env" }
        else if ($file | str starts-with "nushell-env/") { "nushell-env" }
        else if ($file | str starts-with ".claude/") { "config" }
        else { "root" }
    } | uniq)
    
    let file_types = ($changes | each { |file|
        if ($file | str contains "test") { "test" }
        else if ($file | str ends-with ".md") { "docs" }
        else if ($file | str contains "package.json") or ($file | str contains "pyproject.toml") or ($file | str contains "Cargo.toml") or ($file | str contains "go.mod") { "deps" }
        else if ($file | str contains "devbox.json") or ($file | str contains ".claude") { "config" }
        else { "code" }
    } | uniq)
    
    {
        languages: $languages,
        directories: $directories,
        file_types: $file_types,
        total_files: ($changes | length)
    }
}

# Detect commit type based on file changes
def detect_commit_type [changes: list<string>] {
    let has_new_files = (git diff --cached --diff-filter=A --name-only | lines | length) > 0
    let has_test_files = ($changes | any { |file| $file | str contains "test" })
    let has_docs = ($changes | any { |file| $file | str ends-with ".md" })
    let has_config = ($changes | any { |file| 
        ($file | str contains "devbox.json") or 
        ($file | str contains "package.json") or 
        ($file | str contains "pyproject.toml") or 
        ($file | str contains "Cargo.toml") or 
        ($file | str contains "go.mod") or
        ($file | str contains ".claude")
    })
    let has_perf_changes = (git diff --cached | rg -i "(performance|optimization|benchmark)" | lines | length) > 0
    
    if $has_new_files and not $has_test_files and not $has_docs { "feat" }
    else if $has_test_files and not $has_new_files { "test" }
    else if $has_docs and not $has_new_files { "docs" }
    else if $has_config { "chore" }
    else if $has_perf_changes { "perf" }
    else if (git diff --cached | rg -i "(fix|bug|error)" | lines | length) > 0 { "fix" }
    else if (git diff --cached | rg -i "(refactor|cleanup|reorganize)" | lines | length) > 0 { "refactor" }
    else if (git diff --cached | rg -i "(format|style|lint)" | lines | length) > 0 { "style" }
    else { "feat" }
}

# Detect appropriate scope based on languages and directories
def detect_scope [languages: list<string>, directories: list<string>] {
    if ($languages | length) > 1 or ($directories | length) > 1 { "polyglot" }
    else if ($languages | length) == 1 { $languages.0 }
    else if ($directories | length) == 1 {
        let dir = $directories.0
        if $dir == "python-env" { "python" }
        else if $dir == "typescript-env" { "typescript" }
        else if $dir == "rust-env" { "rust" }
        else if $dir == "go-env" { "go" }
        else if $dir == "nushell-env" { "nushell" }
        else if $dir == "config" { "config" }
        else { "project" }
    }
    else { "project" }
}

# Generate the complete commit message
def generate_commit_message [commit_type: string, scope: string, analysis: record] {
    let short_description = (generate_short_description $commit_type $scope $analysis)
    $"($commit_type)\(($scope)\): ($short_description)"
}

# Generate short description based on analysis
def generate_short_description [commit_type: string, scope: string, analysis: record] {
    if $commit_type == "feat" {
        if "test" in $analysis.file_types { "add comprehensive test coverage" }
        else if "api" in ($analysis.file_types | str join " ") { "implement new API endpoints" }
        else { "add new functionality" }
    } else if $commit_type == "fix" {
        if "test" in $analysis.file_types { "resolve test failures" }
        else if "security" in ($analysis.file_types | str join " ") { "address security vulnerabilities" }
        else { "resolve issues and bugs" }
    } else if $commit_type == "refactor" {
        if ($analysis.total_files) > 5 { "restructure codebase for better maintainability" }
        else { "improve code structure and organization" }
    } else if $commit_type == "test" {
        "enhance test coverage and reliability"
    } else if $commit_type == "docs" {
        "update documentation and examples"
    } else if $commit_type == "chore" {
        if "deps" in $analysis.file_types { "update dependencies and build configuration" }
        else { "update configuration and tooling" }
    } else if $commit_type == "perf" {
        "optimize performance and resource usage"
    } else if $commit_type == "style" {
        "format code and fix linting issues"
    } else {
        "update project files"
    }
}

# Validate existing commit message for conventional commit format
def "main validate" [message: string] {
    let pattern = '^(feat|fix|docs|style|refactor|test|chore|perf)\([a-z]+\): .{10,}'
    
    if ($message | str length) < 10 {
        echo "❌ Commit message too short (minimum 10 characters)"
        return false
    }
    
    if not ($message | str contains ":") {
        echo "❌ Missing conventional commit format (type(scope): description)"
        return false
    }
    
    let parts = ($message | split column ":" | get column1.0 | split column "(" | get column1.0)
    let valid_types = ["feat", "fix", "docs", "style", "refactor", "test", "chore", "perf"]
    
    if not ($parts in $valid_types) {
        echo $"❌ Invalid commit type. Use one of: ($valid_types | str join ', ')"
        return false
    }
    
    echo "✅ Commit message follows conventional commit format"
    return true
}

# Analyze git repository state
def "main analyze" [] {
    echo "📊 REPOSITORY ANALYSIS"
    echo ""
    
    # Check if we're in a git repository
    if not (try { git rev-parse --git-dir | complete | get exit_code } | default 1) == 0 {
        echo "❌ Not in a git repository"
        return
    }
    
    # Show current branch
    let branch = (git rev-parse --abbrev-ref HEAD)
    echo $"🌿 Current branch: ($branch)"
    
    # Show staged changes
    let staged = (git diff --cached --name-only | lines)
    echo $"📁 Staged files: ($staged | length)"
    if not ($staged | is-empty) {
        $staged | each { |file| echo $"   • ($file)" }
    }
    
    # Show unstaged changes
    let unstaged = (git diff --name-only | lines)
    echo $"📝 Modified files: ($unstaged | length)"
    if not ($unstaged | is-empty) {
        $unstaged | each { |file| echo $"   • ($file)" }
    }
    
    # Show untracked files
    let untracked = (git ls-files --others --exclude-standard | lines)
    echo $"❓ Untracked files: ($untracked | length)"
    if not ($untracked | is-empty) {
        $untracked | each { |file| echo $"   • ($file)" }
    }
    
    echo ""
    if not ($staged | is-empty) {
        echo "💡 Run 'nu intelligent-commit.nu suggest' for commit message suggestions"
    } else {
        echo "💡 Stage your changes with 'git add' then run 'nu intelligent-commit.nu suggest'"
    }
}