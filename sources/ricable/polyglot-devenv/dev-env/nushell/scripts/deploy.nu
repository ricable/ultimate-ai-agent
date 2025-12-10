#!/usr/bin/env nu

# Deploy/sync configurations across environments
# Usage: nu scripts/deploy.nu [--target environment] [--dry-run]

use ../common.nu *

def main [
    --target: string = "all"
    --dry-run = false
] {
    if $dry_run {
        log info "Dry run mode - no changes will be made"
    }
    
    log info $"Deploying configurations to: ($target)"
    
    match $target {
        "all" => { deploy-to-all $dry_run }
        "python" => { deploy-to-python $dry_run }
        "typescript" => { deploy-to-typescript $dry_run }
        "rust" => { deploy-to-rust $dry_run }
        "go" => { deploy-to-go $dry_run }
        _ => { 
            log error $"Unknown target: ($target)"
            log info "Available targets: all, python, typescript, rust, go"
            exit 1
        }
    }
}

def deploy-to-all [dry_run: bool] {
    let environments = ["python", "typescript", "rust", "go"]
    
    for env in $environments {
        log info $"Deploying to ($env)..."
        
        match $env {
            "python" => { deploy-to-python $dry_run }
            "typescript" => { deploy-to-typescript $dry_run }
            "rust" => { deploy-to-rust $dry_run }
            "go" => { deploy-to-go $dry_run }
        }
    }
    
    log success "Deployment to all environments completed!"
}

def deploy-to-python [dry_run: bool] {
    let target_dir = "python-env"
    
    if not ($target_dir | path exists) {
        log warn $"($target_dir) not found, skipping..."
        return
    }
    
    # Sync common configurations
    sync-common-configs $target_dir $dry_run
    
    # Python-specific configurations
    let python_configs = {
        ".gitignore": generate-python-gitignore
        ".env.example": generate-python-env-example
    }
    
    deploy-configs $python_configs $target_dir $dry_run
}

def deploy-to-typescript [dry_run: bool] {
    let target_dir = "typescript-env"
    
    if not ($target_dir | path exists) {
        log warn $"($target_dir) not found, skipping..."
        return
    }
    
    sync-common-configs $target_dir $dry_run
    
    let ts_configs = {
        ".gitignore": generate-typescript-gitignore
        ".env.example": generate-typescript-env-example
    }
    
    deploy-configs $ts_configs $target_dir $dry_run
}

def deploy-to-rust [dry_run: bool] {
    let target_dir = "rust-env"
    
    if not ($target_dir | path exists) {
        log warn $"($target_dir) not found, skipping..."
        return
    }
    
    sync-common-configs $target_dir $dry_run
    
    let rust_configs = {
        ".gitignore": generate-rust-gitignore
        ".env.example": generate-rust-env-example
    }
    
    deploy-configs $rust_configs $target_dir $dry_run
}

def deploy-to-go [dry_run: bool] {
    let target_dir = "go-env"
    
    if not ($target_dir | path exists) {
        log warn $"($target_dir) not found, skipping..."
        return
    }
    
    sync-common-configs $target_dir $dry_run
    
    let go_configs = {
        ".gitignore": generate-go-gitignore
        ".env.example": generate-go-env-example
    }
    
    deploy-configs $go_configs $target_dir $dry_run
}

def sync-common-configs [target_dir: string, dry_run: bool] {
    log info $"  Syncing common configs to ($target_dir)..."
    
    # Common environment variables
    if (".env" | path exists) {
        let env_content = open .env
        let target_env = $"($target_dir)/.env"
        
        if $dry_run {
            log info $"    Would sync .env to ($target_env)"
        } else {
            $env_content | save $target_env --force
            log success $"    ✅ Synced .env"
        }
    }
}

def deploy-configs [configs: record, target_dir: string, dry_run: bool] {
    for config in ($configs | transpose key value) {
        let target_file = $"($target_dir)/($config.key)"
        let content = do $config.value
        
        if $dry_run {
            log info $"    Would create/update ($target_file)"
        } else {
            $content | save $target_file --force
            log success $"    ✅ Created/updated ($config.key)"
        }
    }
}

# Generate language-specific .gitignore files
def generate-python-gitignore [] {
    "# Python specific
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
.pytest_cache/
.tox/
.nox/
coverage.xml
*.cover
*.py,cover
.hypothesis/

# Jupyter
.ipynb_checkpoints

# Logs
*.log"
}

def generate-typescript-gitignore [] {
    "# TypeScript/Node.js specific
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*
.pnpm-debug.log*

# Build outputs
dist/
build/
*.tsbuildinfo

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
coverage/
.nyc_output/

# Logs
*.log
logs/

# Runtime
*.pid
*.seed
*.pid.lock"
}

def generate-rust-gitignore [] {
    "# Rust specific
/target/
Cargo.lock

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env

# Logs
*.log

# Backup files
*~
*.bak
*.backup"
}

def generate-go-gitignore [] {
    "# Go specific
# Binaries
*.exe
*.exe~
*.dll
*.so
*.dylib

# Test binary
*.test

# Output of the go coverage tool
*.out

# Dependency directories
vendor/

# Go workspace file
go.work

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env

# Logs
*.log"
}

# Generate environment examples
def generate-python-env-example [] {
    "# Python Environment Variables
# Copy this file to .env and fill in your values

# Development settings
PYTHONPATH=./src
PYTHONDONTWRITEBYTECODE=1

# Database
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API Keys
# API_KEY=your_api_key_here

# AWS (if needed)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_DEFAULT_REGION=us-east-1"
}

def generate-typescript-env-example [] {
    "# TypeScript/Node.js Environment Variables
# Copy this file to .env and fill in your values

# Development settings
NODE_ENV=development
PORT=3000

# Database
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API Keys
# API_KEY=your_api_key_here

# JWT
# JWT_SECRET=your_jwt_secret_here"
}

def generate-rust-env-example [] {
    "# Rust Environment Variables
# Copy this file to .env and fill in your values

# Development settings
RUST_BACKTRACE=1
RUST_LOG=debug

# Database
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API Keys
# API_KEY=your_api_key_here"
}

def generate-go-env-example [] {
    "# Go Environment Variables
# Copy this file to .env and fill in your values

# Development settings
GO111MODULE=on
GOPROXY=https://proxy.golang.org

# Database
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API Keys
# API_KEY=your_api_key_here"
}