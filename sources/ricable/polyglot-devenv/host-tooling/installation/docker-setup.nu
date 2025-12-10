#!/usr/bin/env nu

# Docker Provider Setup and Optimization Script
# Configures DevPod with Docker provider for polyglot development

use ../common.nu *

def main [
    --install(-i)          # Install Docker and DevPod if not present
    --configure(-c)        # Configure Docker provider settings
    --optimize(-o)         # Apply performance optimizations
    --reset(-r)           # Reset Docker provider configuration
    --status(-s)          # Show Docker provider status
    --help(-h)            # Show help
] {
    log info "🐳 DevPod Docker Provider Setup"
    
    if $help {
        show_help
        return
    }
    
    if $status or (not $install and not $configure and not $optimize and not $reset) {
        show_status
        return
    }
    
    if $reset {
        reset_configuration
        return
    }
    
    if $install {
        install_dependencies
    }
    
    if $configure {
        configure_provider
    }
    
    if $optimize {
        apply_optimizations
    }
    
    log success "✅ Docker provider setup completed"
}

def show_help [] {
    print "DevPod Docker Provider Setup Script

USAGE:
    nu docker-setup.nu [OPTIONS]

OPTIONS:
    -i, --install      Install Docker and DevPod if not present
    -c, --configure    Configure Docker provider settings
    -o, --optimize     Apply performance optimizations
    -r, --reset        Reset Docker provider configuration
    -s, --status       Show Docker provider status
    -h, --help         Show this help message

EXAMPLES:
    # Complete setup (install + configure + optimize)
    nu docker-setup.nu --install --configure --optimize
    
    # Just configure the provider
    nu docker-setup.nu --configure
    
    # Check status
    nu docker-setup.nu --status
    
    # Reset and reconfigure
    nu docker-setup.nu --reset --configure"
}

def show_status [] {
    log info "📊 Docker Provider Status"
    print "=" * 50
    
    # Check Docker installation
    log info "🐳 Docker Installation:"
    try {
        let docker_version = (docker --version | complete)
        if $docker_version.exit_code == 0 {
            print $"   ✅ Docker: ($docker_version.stdout | str trim)"
        } else {
            print "   ❌ Docker: Not installed"
        }
    } catch {
        print "   ❌ Docker: Not installed"
    }
    
    # Check Docker daemon
    log info "🔄 Docker Daemon:"
    try {
        let docker_info = (docker info | complete)
        if $docker_info.exit_code == 0 {
            print "   ✅ Docker daemon: Running"
            
            # Parse Docker info for key metrics
            let info_lines = ($docker_info.stdout | lines)
            for line in $info_lines {
                if ($line | str contains "Server Version:") {
                    print $"   📦 ($line | str trim)"
                }
                if ($line | str contains "Total Memory:") {
                    print $"   💾 ($line | str trim)"
                }
                if ($line | str contains "CPUs:") {
                    print $"   🖥️  ($line | str trim)"
                }
            }
        } else {
            print "   ❌ Docker daemon: Not running"
        }
    } catch {
        print "   ❌ Docker daemon: Not accessible"
    }
    
    # Check DevPod installation
    log info "🚀 DevPod Installation:"
    try {
        let devpod_version = (devpod version | complete)
        if $devpod_version.exit_code == 0 {
            print $"   ✅ DevPod: ($devpod_version.stdout | str trim)"
        } else {
            print "   ❌ DevPod: Not installed"
        }
    } catch {
        print "   ❌ DevPod: Not installed"
    }
    
    # Check Docker provider status
    log info "🔧 Docker Provider:"
    try {
        let providers = (devpod provider list | complete)
        if $providers.exit_code == 0 {
            let provider_lines = ($providers.stdout | lines)
            let docker_provider = ($provider_lines | where $it =~ "docker")
            
            if ($docker_provider | length) > 0 {
                print "   ✅ Docker provider: Configured"
                for line in $docker_provider {
                    print $"   📋 ($line | str trim)"
                }
            } else {
                print "   ⚠️  Docker provider: Not configured"
            }
        } else {
            print "   ❌ Docker provider: Cannot check status"
        }
    } catch {
        print "   ❌ Docker provider: Cannot check status"
    }
    
    # Check existing workspaces
    log info "🏗️  DevPod Workspaces:"
    try {
        let workspaces = (devpod list | complete)
        if $workspaces.exit_code == 0 {
            let workspace_lines = ($workspaces.stdout | lines | where $it != "")
            let polyglot_workspaces = ($workspace_lines | where $it =~ "polyglot")
            
            if ($polyglot_workspaces | length) > 0 {
                print $"   ✅ Found ($polyglot_workspaces | length) polyglot workspaces:"
                for workspace in $polyglot_workspaces {
                    print $"   📁 ($workspace | str trim)"
                }
            } else {
                print "   📝 No polyglot workspaces found"
            }
        } else {
            print "   ❌ Cannot list workspaces"
        }
    } catch {
        print "   ❌ Cannot list workspaces"
    }
    
    # Check Docker resources
    log info "💻 Docker Resources:"
    try {
        let containers = (docker ps -a --format "table {{.Names}}\t{{.Status}}" | complete)
        if $containers.exit_code == 0 {
            let container_lines = ($containers.stdout | lines | where $it != "" and $it !~ "NAMES")
            let devpod_containers = ($container_lines | where $it =~ "devpod")
            
            if ($devpod_containers | length) > 0 {
                print $"   🔍 Active DevPod containers: ($devpod_containers | length)"
                for container in $devpod_containers {
                    print $"   📦 ($container | str trim)"
                }
            } else {
                print "   📝 No DevPod containers running"
            }
        }
    } catch {
        print "   ❌ Cannot check containers"
    }
}

def install_dependencies [] {
    log info "📦 Installing Docker and DevPod dependencies"
    
    # Check if Docker is installed
    try {
        docker --version | ignore
        log success "✅ Docker is already installed"
    } catch {
        log info "🔄 Installing Docker..."
        
        # Detect platform and install Docker
        let os = (uname -s | str downcase)
        
        if $os == "darwin" {
            log info "🍎 macOS detected - Installing Docker Desktop"
            if (which brew | is-empty) {
                log error "❌ Homebrew not found. Please install Homebrew first: https://brew.sh"
                return
            }
            
            try {
                brew install --cask docker
                log success "✅ Docker Desktop installed"
                log info "ℹ️  Please start Docker Desktop manually"
            } catch {
                log error "❌ Failed to install Docker Desktop"
                return
            }
        } else if $os == "linux" {
            log info "🐧 Linux detected - Installing Docker Engine"
            
            # Install Docker using the official script
            try {
                curl -fsSL https://get.docker.com -o get-docker.sh
                sudo sh get-docker.sh
                sudo usermod -aG docker $env.USER
                rm get-docker.sh
                log success "✅ Docker Engine installed"
                log info "ℹ️  Please log out and back in for group changes to take effect"
            } catch {
                log error "❌ Failed to install Docker Engine"
                return
            }
        } else {
            log error $"❌ Unsupported platform: ($os)"
            return
        }
    }
    
    # Check if DevPod is installed
    try {
        devpod version | ignore
        log success "✅ DevPod is already installed"
    } catch {
        log info "🔄 Installing DevPod..."
        
        # Install DevPod
        try {
            if (which brew | is-not-empty) {
                brew install devpod-io/devpod/devpod
            } else {
                # Use the official installer script
                curl -L -o devpod https://github.com/loft-sh/devpod/releases/latest/download/devpod-linux-amd64
                chmod +x devpod
                sudo mv devpod /usr/local/bin/
            }
            log success "✅ DevPod installed"
        } catch {
            log error "❌ Failed to install DevPod"
            return
        }
    }
    
    # Wait for Docker to be ready
    log info "⏳ Waiting for Docker to be ready..."
    mut attempts = 0
    mut docker_ready = false
    
    while $attempts < 30 and not $docker_ready {
        try {
            docker info | ignore
            set docker_ready = true
            log success "✅ Docker is ready"
        } catch {
            $attempts = $attempts + 1
            sleep 2sec
            print "."
        }
    }
    
    if not $docker_ready {
        log error "❌ Docker is not ready after 60 seconds. Please check Docker installation."
        return
    }
}

def configure_provider [] {
    log info "⚙️  Configuring Docker provider"
    
    # Add Docker provider if not already added
    try {
        let providers = (devpod provider list)
        if not ($providers | str contains "docker") {
            log info "➕ Adding Docker provider"
            devpod provider add docker
            log success "✅ Docker provider added"
        } else {
            log info "ℹ️  Docker provider already configured"
        }
    } catch {
        log error "❌ Failed to configure Docker provider"
        return
    }
    
    # Apply configuration from docker-provider.yaml
    let config_file = "../config/docker-provider.yaml"
    
    if ($config_file | path exists) {
        log info "📋 Applying configuration from docker-provider.yaml"
        
        # Set Docker provider options
        try {
            # Resource limits
            devpod provider set-options docker MEMORY=4g
            devpod provider set-options docker CPUS=2.0
            
            # Build optimizations
            devpod provider set-options docker DOCKER_BUILDKIT=1
            devpod provider set-options docker BUILDKIT_INLINE_CACHE=1
            
            # Security settings
            devpod provider set-options docker SECURITY_OPT="seccomp:unconfined,apparmor:unconfined"
            devpod provider set-options docker CAP_ADD="SYS_PTRACE,NET_ADMIN"
            
            log success "✅ Docker provider configured"
        } catch {
            log warning "⚠️  Some configuration options may not be available"
        }
    } else {
        log warning "⚠️  Configuration file not found: $config_file"
    }
    
    # Set Docker provider as default
    try {
        devpod provider use docker
        log success "✅ Docker provider set as default"
    } catch {
        log warning "⚠️  Could not set Docker provider as default"
    }
}

def apply_optimizations [] {
    log info "🚀 Applying performance optimizations"
    
    # Create Docker volumes for caching
    log info "📦 Creating cache volumes"
    
    let cache_volumes = [
        "devpod-npm-cache"
        "devpod-cargo-cache" 
        "devpod-pip-cache"
        "devpod-go-cache"
        "devpod-uv-cache"
    ]
    
    for volume in $cache_volumes {
        try {
            let existing = (docker volume ls --filter $"name=($volume)" --format "{{.Name}}")
            if ($existing | str trim) == "" {
                docker volume create $volume
                log info $"   ✅ Created volume: ($volume)"
            } else {
                log info $"   ℹ️  Volume already exists: ($volume)"
            }
        } catch {
            log warning $"   ⚠️  Could not create volume: ($volume)"
        }
    }
    
    # Configure Docker daemon optimizations
    log info "⚙️  Configuring Docker daemon optimizations"
    
    let docker_config_dir = if (uname -s | str downcase) == "darwin" {
        "~/.docker"
    } else {
        "/etc/docker"
    }
    
    let daemon_config = {
        "builder": {
            "gc": {
                "enabled": true,
                "defaultKeepStorage": "10GB"
            }
        },
        "experimental": false,
        "features": {
            "buildkit": true
        },
        "storage-driver": "overlay2",
        "log-driver": "json-file",
        "log-opts": {
            "max-size": "10m",
            "max-file": "3"
        }
    }
    
    try {
        mkdir $docker_config_dir
        ($daemon_config | to json) | save $"($docker_config_dir)/daemon.json"
        log info "   ✅ Docker daemon configuration updated"
        log warning "   ⚠️  Please restart Docker for changes to take effect"
    } catch {
        log warning "   ⚠️  Could not update Docker daemon configuration"
    }
    
    # Set environment variables for build optimization
    log info "🔧 Setting build optimization environment variables"
    
    let env_vars = [
        "DOCKER_BUILDKIT=1"
        "BUILDKIT_PROGRESS=plain"
        "BUILDKIT_INLINE_CACHE=1"
    ]
    
    for env_var in $env_vars {
        $env.($env_var | split column "=" | get column1.0) = ($env_var | split column "=" | get column2.0)
        log info $"   ✅ Set ($env_var)"
    }
    
    # Create optimization script for persistent settings
    let opt_script = "../scripts/docker-optimize.sh"
    
    $"#!/bin/bash
# Docker optimization settings for DevPod

# Enable BuildKit
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain
export BUILDKIT_INLINE_CACHE=1

# Build optimizations
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_CLI_EXPERIMENTAL=enabled

# Cache settings
export DOCKER_BUILDKIT_CACHE_MOUNT=1

echo \"🚀 Docker optimizations applied\"
" | save $opt_script
    
    chmod +x $opt_script
    log success "✅ Optimization script created: $opt_script"
}

def reset_configuration [] {
    log info "🔄 Resetting Docker provider configuration"
    
    print "⚠️  This will remove the Docker provider and all associated workspaces."
    let confirm = (input "Are you sure you want to continue? (y/N): ")
    
    if $confirm != "y" and $confirm != "Y" {
        log info "❌ Reset cancelled"
        return
    }
    
    # Stop all DevPod workspaces
    try {
        let workspaces = (devpod list --output json | from json)
        for workspace in $workspaces {
            if ($workspace.name | str contains "polyglot") {
                log info $"🛑 Stopping workspace: ($workspace.name)"
                devpod stop $workspace.name
            }
        }
    } catch {
        log warning "⚠️  Could not stop workspaces"
    }
    
    # Remove Docker provider
    try {
        devpod provider delete docker
        log success "✅ Docker provider removed"
    } catch {
        log warning "⚠️  Could not remove Docker provider"
    }
    
    # Clean up Docker resources
    try {
        # Remove DevPod containers
        let containers = (docker ps -a --filter "label=devpod" --format "{{.ID}}")
        if ($containers | str trim) != "" {
            docker rm -f ($containers | lines)
            log info "🗑️  Removed DevPod containers"
        }
        
        # Remove DevPod images
        let images = (docker images --filter "label=devpod" --format "{{.ID}}")
        if ($images | str trim) != "" {
            docker rmi -f ($images | lines)
            log info "🗑️  Removed DevPod images"
        }
        
        # Clean up build cache
        docker builder prune -af
        log info "🧹 Cleaned build cache"
        
    } catch {
        log warning "⚠️  Could not clean up all Docker resources"
    }
    
    log success "✅ Docker provider configuration reset"
    log info "ℹ️  Run with --configure to set up again"
}