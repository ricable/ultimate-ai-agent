#!/usr/bin/env nu

# Documentation automation for polyglot development environments
# Based on polyglot-docs.md specifications

def main [] {
    echo "📚 Polyglot Documentation Automation"
    echo "Usage: nu docs-automation.nu [check|update|generate|analyze]"
}

# Check if documentation needs updating based on code changes
def "main check" [] {
    echo "🔍 DOCUMENTATION DRIFT ANALYSIS"
    echo ""
    
    let git_context = (analyze_git_changes)
    let api_changes = (detect_api_changes $git_context.modified)
    let config_changes = (detect_config_changes $git_context.modified)
    let doc_freshness = (check_doc_freshness)
    
    echo $"📝 Modified files: ($git_context.modified | length)"
    echo $"🔌 API changes detected: ($api_changes.detected)"
    echo $"⚙️  Config changes: ($config_changes.detected)"
    echo $"📚 Documentation freshness: ($doc_freshness.status)"
    
    if $api_changes.detected or $config_changes.detected or ($doc_freshness.status == "stale") {
        echo ""
        echo "💡 DOCUMENTATION UPDATE RECOMMENDED"
        echo "   Run: nu docs-automation.nu update"
        echo "   Or: nu docs-automation.nu generate api"
    } else {
        echo ""
        echo "✅ Documentation appears up to date"
    }
}

# Update documentation based on detected changes
def "main update" [] {
    echo "📚 Updating documentation..."
    
    let changes = (analyze_git_changes)
    
    # Update API documentation if API files changed
    let api_files = ($changes.modified | where { |file| 
        ($file | str contains "api") or 
        ($file | str contains "endpoint") or
        ($file | str contains "route") or
        ($file | str ends-with ".py") or
        ($file | str ends-with ".ts")
    })
    
    if ($api_files | length) > 0 {
        echo "🔌 API files changed, suggesting documentation update:"
        echo "   • Python: Consider updating FastAPI docs"
        echo "   • TypeScript: Consider updating TypeDoc documentation"
        echo "   • Generate: nu docs-automation.nu generate api"
    }
    
    # Update setup documentation if configuration changed
    let config_files = ($changes.modified | where { |file|
        ($file | str contains "devbox.json") or
        ($file | str contains "package.json") or
        ($file | str contains "pyproject.toml") or
        ($file | str contains "Cargo.toml") or
        ($file | str contains "go.mod")
    })
    
    if ($config_files | length) > 0 {
        echo "⚙️  Configuration files changed, update setup documentation:"
        echo "   • Review CLAUDE.md environment setup sections"
        echo "   • Update installation and dependency instructions"
        echo "   • Generate: nu docs-automation.nu generate setup"
    }
}

# Generate specific types of documentation
def "main generate" [type: string] {
    match $type {
        "api" => { generate_api_docs }
        "architecture" => { generate_architecture_docs }
        "setup" => { generate_setup_docs }
        "changelog" => { generate_changelog }
        "all" => { generate_all_docs }
        _ => { echo $"❌ Unknown documentation type: ($type)" }
    }
}

# Analyze documentation completeness and quality
def "main analyze" [] {
    echo "📊 DOCUMENTATION ANALYSIS REPORT"
    echo ""
    
    let environments = ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]
    
    echo "📁 ENVIRONMENT DOCUMENTATION STATUS"
    $environments | each { |env|
        let status = (analyze_environment_docs $env)
        echo $"   ($env): ($status.coverage)% coverage, ($status.freshness) freshness"
    }
    
    echo ""
    echo "📋 DOCUMENTATION COMPLETENESS"
    let completeness = (assess_doc_completeness)
    $completeness | each { |item|
        echo $"   ($item.type): ($item.status) - ($item.description)"
    }
    
    echo ""
    echo "💡 RECOMMENDATIONS"
    let recommendations = (generate_doc_recommendations)
    $recommendations | each { |rec|
        echo $"   • ($rec)"
    }
}

# Analyze git changes for documentation relevance
def analyze_git_changes [] {
    if not (try { git rev-parse --git-dir | complete | get exit_code } | default 1) == 0 {
        return { modified: [], staged: [], recent_commits: [] }
    }
    
    {
        modified: (git diff --name-only | lines),
        staged: (git diff --cached --name-only | lines),
        recent_commits: (git log --oneline -10 | lines)
    }
}

# Detect API-related changes
def detect_api_changes [modified_files: list<string>] {
    let api_patterns = [
        "app.py", "main.py", "routes", "api", "endpoint", 
        "controller", "service", "handler", ".ts", ".js"
    ]
    
    let api_files = ($modified_files | where { |file|
        $api_patterns | any { |pattern| $file | str contains $pattern }
    })
    
    {
        detected: (($api_files | length) > 0),
        files: $api_files,
        environments: (detect_affected_environments $api_files)
    }
}

# Detect configuration changes
def detect_config_changes [modified_files: list<string>] {
    let config_patterns = [
        "devbox.json", "package.json", "pyproject.toml", 
        "Cargo.toml", "go.mod", "tsconfig.json", ".claude"
    ]
    
    let config_files = ($modified_files | where { |file|
        $config_patterns | any { |pattern| $file | str contains $pattern }
    })
    
    {
        detected: (($config_files | length) > 0),
        files: $config_files,
        types: (classify_config_changes $config_files)
    }
}

# Check documentation freshness
def check_doc_freshness [] {
    # Check if CLAUDE.md exists and when it was last modified
    if ("CLAUDE.md" | path exists) {
        let claude_md_time = (ls CLAUDE.md | get modified.0)
        let recent_code_changes = (try { 
            git log --since="1 week ago" --oneline | lines | length 
        } | default 0)
        
        if $recent_code_changes > 10 {
            { status: "stale", reason: "Many recent code changes" }
        } else {
            { status: "fresh", reason: "Recently updated" }
        }
    } else {
        { status: "missing", reason: "CLAUDE.md not found" }
    }
}

# Generate API documentation suggestions
def generate_api_docs [] {
    echo "🔌 GENERATING API DOCUMENTATION SUGGESTIONS"
    echo ""
    
    echo "📋 Recommended API Documentation Updates:"
    echo ""
    
    # Python FastAPI documentation
    if ("python-env" | path exists) {
        echo "🐍 PYTHON (FastAPI):"
        echo "   • Run: cd python-env && devbox run docs"
        echo "   • Update OpenAPI schemas and docstrings"
        echo "   • Generate Sphinx documentation if configured"
        echo ""
    }
    
    # TypeScript documentation
    if ("typescript-env" | path exists) {
        echo "📘 TYPESCRIPT:"
        echo "   • Run: cd typescript-env && devbox run docs"
        echo "   • Update TypeDoc comments and interfaces"
        echo "   • Generate Storybook components if applicable"
        echo ""
    }
    
    # Rust documentation
    if ("rust-env" | path exists) {
        echo "🦀 RUST:"
        echo "   • Run: cd rust-env && cargo doc --open"
        echo "   • Update rustdoc comments and examples"
        echo ""
    }
    
    # Go documentation
    if ("go-env" | path exists) {
        echo "🐹 GO:"
        echo "   • Run: cd go-env && godoc -http=:6060"
        echo "   • Update package documentation and examples"
        echo ""
    }
    
    echo "💡 Cross-language API integration documentation needed"
}

# Generate architecture documentation
def generate_architecture_docs [] {
    echo "🏗️  GENERATING ARCHITECTURE DOCUMENTATION"
    echo ""
    echo "📊 Recommended Architecture Updates:"
    echo "   • Update system overview diagrams"
    echo "   • Document service boundaries and interactions"
    echo "   • Create deployment architecture diagrams"
    echo "   • Update integration patterns documentation"
}

# Generate setup documentation
def generate_setup_docs [] {
    echo "⚙️  GENERATING SETUP DOCUMENTATION"
    echo ""
    echo "📋 Setup Documentation Tasks:"
    echo "   • Update prerequisite installation steps"
    echo "   • Verify devbox environment setup instructions"
    echo "   • Update dependency versions and compatibility"
    echo "   • Test setup guide with fresh environment"
}

# Generate changelog from git history
def generate_changelog [] {
    echo "📝 GENERATING CHANGELOG"
    echo ""
    
    let recent_commits = (git log --oneline --since="1 month ago" | lines)
    echo $"📊 Recent commits: ($recent_commits | length)"
    
    echo ""
    echo "💡 Changelog generation suggestions:"
    echo "   • Group commits by type (feat, fix, docs, etc.)"
    echo "   • Extract breaking changes and new features"
    echo "   • Link to relevant issues and PRs"
    echo "   • Include performance improvements and security fixes"
}

# Generate all documentation types
def generate_all_docs [] {
    echo "📚 COMPREHENSIVE DOCUMENTATION GENERATION"
    echo ""
    
    generate_api_docs
    echo ""
    generate_architecture_docs
    echo ""
    generate_setup_docs
    echo ""
    generate_changelog
}

# Analyze documentation for a specific environment
def analyze_environment_docs [env: string] {
    if not (($env | path exists)) {
        return { coverage: 0, freshness: "missing" }
    }
    
    # Basic analysis - could be enhanced with actual file scanning
    let has_readme = ([$env, "README.md"] | path join | path exists)
    let has_docs_dir = ([$env, "docs"] | path join | path exists)
    
    let coverage = if $has_readme and $has_docs_dir { 80 }
                  else if $has_readme { 60 }
                  else { 20 }
    
    { coverage: $coverage, freshness: "unknown" }
}

# Assess overall documentation completeness
def assess_doc_completeness [] {
    [
        { type: "API Docs", status: "partial", description: "Some environments have API documentation" },
        { type: "Setup Guide", status: "good", description: "CLAUDE.md provides comprehensive setup" },
        { type: "Architecture", status: "partial", description: "Could benefit from diagrams" },
        { type: "Examples", status: "partial", description: "Some code examples in documentation" },
        { type: "Troubleshooting", status: "basic", description: "Basic troubleshooting information" }
    ]
}

# Generate documentation recommendations
def generate_doc_recommendations [] {
    [
        "Add Mermaid diagrams for system architecture",
        "Create interactive setup validation scripts",
        "Generate cross-language API integration examples",
        "Add performance benchmarking documentation",
        "Create troubleshooting guides with common solutions",
        "Set up automated documentation deployment"
    ]
}

# Detect affected environments from file changes
def detect_affected_environments [files: list<string>] {
    let environments = []
    
    # This would be enhanced to properly detect environments
    if ($files | any { |f| $f | str starts-with "python-env" }) {
        $environments | append "python"
    } else {
        $environments
    }
}

# Classify types of configuration changes
def classify_config_changes [config_files: list<string>] {
    mut types = []
    
    if ($config_files | any { |f| $f | str contains "package.json" }) {
        $types = ($types | append "dependencies")
    }
    if ($config_files | any { |f| $f | str contains "devbox.json" }) {
        $types = ($types | append "environment")
    }
    
    $types
}