#!/usr/bin/env nu

# GitHub automation utilities for the polyglot development environment
# Adapted from DevOps automation patterns in nushell.md
# Usage: nu scripts/github.nu <command> [options]

use ../common.nu *

# Main GitHub command dispatcher
def main [command?: string] {
    if ($command | is-empty) {
        show-help
    } else {
        match $command {
            "setup" => { main setup }
            "status" => { main status }
            "workflow" => { main workflow }
            "release" => { main release }
            "clone-all" => { main clone-all }
            "sync-envs" => { main sync-envs }
            "create-workflow" => { main create-workflow }
            _ => {
                log error $"Unknown command: ($command)"
                show-help
                exit 1
            }
        }
    }
}

def show-help [] {
    log info "GitHub utilities for polyglot development environment"
    print ""
    print "Commands:"
    print "  setup          - Setup GitHub CLI and authentication"
    print "  status         - Show repository and workflow status"
    print "  workflow       - Manage GitHub Actions workflows"
    print "  release        - Create releases for all environments"
    print "  clone-all      - Clone all repositories for the organization"
    print "  sync-envs      - Sync environment configurations to GitHub"
    print "  create-workflow - Create CI/CD workflows for each environment"
    print ""
    print "Examples:"
    print "  nu scripts/github.nu setup"
    print "  nu scripts/github.nu status"
    print "  nu scripts/github.nu workflow --list"
    print "  nu scripts/github.nu create-workflow --language python"
}

# Setup GitHub CLI and authentication
def "main setup" [
    --token: string = ""
] {
    log info "Setting up GitHub integration..."
    
    # Check if gh CLI is available
    if not (cmd exists "gh") {
        log error "GitHub CLI (gh) not found. Install with: brew install gh"
        exit 1
    }
    
    # Check authentication
    try {
        gh auth status
        log success "✅ GitHub CLI is authenticated"
    } catch {
        log info "GitHub CLI not authenticated. Starting authentication..."
        
        if ($token | is-not-empty) {
            echo $token | gh auth login --with-token
        } else {
            gh auth login
        }
        
        log success "✅ GitHub CLI authentication completed"
    }
    
    # Verify current user
    let user = gh api user | from json | get login
    log success $"Authenticated as: ($user)"
}

# Show repository and workflow status
def "main status" [
    --repo: string = ""
] {
    log info "Checking GitHub repository status..."
    
    # Get current repository info
    let repo_info = if ($repo | is-not-empty) {
        gh repo view $repo --json name,owner,description,isPrivate,defaultBranch
    } else {
        gh repo view --json name,owner,description,isPrivate,defaultBranch
    } | from json
    
    log info $"Repository: ($repo_info.owner.login)/($repo_info.name)"
    log info $"Description: ($repo_info.description)"
    log info $"Default branch: ($repo_info.defaultBranch)"
    log info $"Private: ($repo_info.isPrivate)"
    
    # Check workflow status
    show-workflow-status $repo
    
    # Check recent commits
    show-recent-activity $repo
}

def show-workflow-status [repo: string] {
    log info "GitHub Actions workflow status:"
    
    try {
        let workflows = if ($repo | is-not-empty) {
            gh workflow list --repo $repo --json name,state,id
        } else {
            gh workflow list --json name,state,id
        } | from json
        
        if ($workflows | length) > 0 {
            for workflow in $workflows {
                let status_icon = if $workflow.state == "active" { "✅" } else { "⚠️" }
                log info $"  ($status_icon) ($workflow.name) - ($workflow.state)"
            }
        } else {
            log info "  No workflows found"
        }
    } catch {
        log warn "  Unable to fetch workflow information"
    }
}

def show-recent-activity [repo: string] {
    log info "Recent activity:"
    
    try {
        let commits = if ($repo | is-not-empty) {
            gh api $"repos/($repo)/commits" --paginate=false
        } else {
            gh api "repos/:owner/:repo/commits" --paginate=false
        } | from json | first 5
        
        for commit in $commits {
            let message = $commit.commit.message | lines | first
            let author = $commit.commit.author.name
            let date = $commit.commit.author.date | into datetime | format date "%Y-%m-%d %H:%M"
            log info $"  ($date) - ($message) by ($author)"
        }
    } catch {
        log warn "  Unable to fetch recent commits"
    }
}

# Manage GitHub Actions workflows
def "main workflow" [
    --list = false
    --run: string = ""
    --status: string = ""
    --logs: string = ""
] {
    if $list {
        list-workflows
    } else if ($run | is-not-empty) {
        run-workflow $run
    } else if ($status | is-not-empty) {
        workflow-status $status
    } else if ($logs | is-not-empty) {
        workflow-logs $logs
    } else {
        log error "Please specify an action: --list, --run, --status, or --logs"
        exit 1
    }
}

def list-workflows [] {
    log info "Available workflows:"
    
    let workflows = gh workflow list --json name,state,id,path | from json
    
    for workflow in $workflows {
        log info $"  ID: ($workflow.id)"
        log info $"  Name: ($workflow.name)"
        log info $"  State: ($workflow.state)"
        log info $"  Path: ($workflow.path)"
        print ""
    }
}

def run-workflow [workflow_name: string] {
    log info $"Running workflow: ($workflow_name)"
    
    try {
        gh workflow run $workflow_name
        log success $"✅ Triggered workflow: ($workflow_name)"
    } catch { |e|
        log error $"❌ Failed to run workflow: ($e.msg)"
        exit 1
    }
}

def workflow-status [workflow_name: string] {
    log info $"Checking status of workflow: ($workflow_name)"
    
    let runs = gh run list --workflow $workflow_name --json status,conclusion,createdAt,headBranch --limit 5 | from json
    
    for run in $runs {
        let status_icon = match $run.conclusion {
            "success" => "✅"
            "failure" => "❌"
            "cancelled" => "⚠️"
            _ => "🔄"
        }
        let date = $run.createdAt | into datetime | format date "%Y-%m-%d %H:%M"
        log info $"  ($status_icon) ($run.status) - ($run.headBranch) - ($date)"
    }
}

def workflow-logs [run_id: string] {
    log info $"Fetching logs for run: ($run_id)"
    gh run view $run_id --log
}

# Create releases for all environments
def "main release" [
    --version: string = ""
    --environments: list<string> = ["python", "typescript", "rust", "go"]
    --draft = false
] {
    if ($version | is-empty) {
        log error "Please specify a version with --version"
        exit 1
    }
    
    log info $"Creating release ($version) for environments: ($environments | str join ', ')"
    
    for env in $environments {
        create-environment-release $env $version $draft
    }
}

def create-environment-release [env: string, version: string, draft: bool] {
    let env_dir = $"($env)-env"
    
    if not ($env_dir | path exists) {
        log warn $"Environment directory not found: ($env_dir), skipping..."
        return
    }
    
    let tag = $"($env)-($version)"
    let title = $"($env | str title-case) Environment ($version)"
    
    log info $"Creating release for ($env)..."
    
    # Generate release notes
    let release_notes = generate-release-notes $env $version
    
    let draft_flag = if $draft { "--draft" } else { "" }
    
    try {
        gh release create $tag --title $title --notes $release_notes $draft_flag
        log success $"✅ Created release: ($tag)"
    } catch { |e|
        log error $"❌ Failed to create release for ($env): ($e.msg)"
    }
}

def generate-release-notes [env: string, version: string] {
    $"# ($env | str title-case) Environment Release ($version)

## Changes
- Updated dependencies to latest versions
- Improved development setup and configuration
- Enhanced CI/CD pipeline integration

## Environment Details
- Environment: ($env)-env
- Version: ($version)
- Generated: (date now | format date '%Y-%m-%d %H:%M:%S')

## Installation
```bash
cd ($env)-env
devbox shell
devbox run install
```

## Testing
```bash
devbox run test
```"
}

# Clone all repositories for an organization
def "main clone-all" [
    --org: string = ""
    --limit: int = 50
    --target-dir: string = "repos"
] {
    if ($org | is-empty) {
        log error "Please specify an organization with --org"
        exit 1
    }
    
    log info $"Cloning repositories from organization: ($org)"
    
    mkdir $target_dir
    cd $target_dir
    
    let repos = gh repo list $org --limit $limit --json name,cloneUrl | from json
    
    for repo in $repos {
        log info $"Cloning ($repo.name)..."
        
        try {
            git clone $repo.cloneUrl
            log success $"✅ Cloned ($repo.name)"
        } catch { |e|
            log error $"❌ Failed to clone ($repo.name): ($e.msg)"
        }
    }
    
    cd ..
    log success $"Completed cloning ($repos | length) repositories to ($target_dir)/"
}

# Sync environment configurations to GitHub
def "main sync-envs" [
    --repo: string = ""
    --commit-message: string = "Sync environment configurations"
] {
    log info "Syncing environment configurations to GitHub..."
    
    # Check if we're in a git repository
    if not (".git" | path exists) {
        log error "Not in a git repository"
        exit 1
    }
    
    # Add all environment changes
    git add *-env/
    git add nushell-env/
    git add CLAUDE.md
    
    # Check if there are changes to commit
    if (git is-clean) {
        log info "No changes to sync"
        return
    }
    
    # Commit changes
    git commit -m $commit_message
    
    # Push to remote
    try {
        git push
        log success "✅ Synced configurations to GitHub"
    } catch { |e|
        log error $"❌ Failed to push changes: ($e.msg)"
        exit 1
    }
}

# Create CI/CD workflows for each environment
def "main create-workflow" [
    --language: string = "all"
    --template: string = "standard"
] {
    log info "Creating GitHub Actions workflows..."
    
    let languages = if $language == "all" {
        ["python", "typescript", "rust", "go", "nushell"]
    } else {
        [$language]
    }
    
    mkdir .github/workflows
    
    for lang in $languages {
        create-language-workflow $lang $template
    }
    
    # Create a main workflow that runs all environment validations
    create-main-workflow
    
    log success "GitHub Actions workflows created!"
}

def create-language-workflow [language: string, template: string] {
    let workflow_content = match $language {
        "python" => generate-python-workflow
        "typescript" => generate-typescript-workflow
        "rust" => generate-rust-workflow
        "go" => generate-go-workflow
        "nushell" => generate-nushell-workflow
        _ => {
            log warn $"Unknown language: ($language)"
            return
        }
    }
    
    let workflow_file = $".github/workflows/($language).yml"
    $workflow_content | save $workflow_file --force
    log success $"✅ Created workflow: ($workflow_file)"
}

def generate-python-workflow [] {
    $"name: Python Environment

on:
  push:
    paths:
      - 'python-env/**'
      - '.github/workflows/python.yml'
  pull_request:
    paths:
      - 'python-env/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run Python environment tests
        run: |
          cd python-env
          devbox run lint
          devbox run test
"
}

def generate-typescript-workflow [] {
    $"name: TypeScript Environment

on:
  push:
    paths:
      - 'typescript-env/**'
      - '.github/workflows/typescript.yml'
  pull_request:
    paths:
      - 'typescript-env/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run TypeScript environment tests
        run: |
          cd typescript-env
          devbox run lint
          devbox run test
"
}

def generate-rust-workflow [] {
    $"name: Rust Environment

on:
  push:
    paths:
      - 'rust-env/**'
      - '.github/workflows/rust.yml'
  pull_request:
    paths:
      - 'rust-env/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run Rust environment tests
        run: |
          cd rust-env
          devbox run lint
          devbox run test
"
}

def generate-go-workflow [] {
    $"name: Go Environment

on:
  push:
    paths:
      - 'go-env/**'
      - '.github/workflows/go.yml'
  pull_request:
    paths:
      - 'go-env/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run Go environment tests
        run: |
          cd go-env
          devbox run lint
          devbox run test
"
}

def generate-nushell-workflow [] {
    $"name: Nushell Environment

on:
  push:
    paths:
      - 'nushell-env/**'
      - '.github/workflows/nushell.yml'
  pull_request:
    paths:
      - 'nushell-env/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run Nushell environment tests
        run: |
          cd nushell-env
          devbox run check
          devbox run test
"
}

def create-main-workflow [] {
    let workflow_content = $"name: Polyglot Environment Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate-all:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install devbox
        uses: jetify-com/devbox-action@v0.13.0
        
      - name: Run cross-environment validation
        run: |
          cd nushell-env
          devbox run validate
"
    
    $workflow_content | save .github/workflows/main.yml --force
    log success "✅ Created main validation workflow"
}