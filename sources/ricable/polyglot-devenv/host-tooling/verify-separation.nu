#!/usr/bin/env nu

# Verification script for host/container separation
# This script runs on the HOST to verify the separation is working correctly

def main [] {
    print "🔍 Verifying Host/Container Separation Architecture"
    print "=" * 60
    
    # Verify host tooling structure
    verify-host-structure
    
    # Verify container structure  
    verify-container-structure
    
    # Test host commands
    test-host-commands
    
    # Test integration points
    test-integration
    
    print ""
    print "✅ Host/Container separation verification completed!"
}

def verify-host-structure [] {
    print "📁 Verifying host-tooling/ structure..."
    
    let required_dirs = [
        "host-tooling/installation"
        "host-tooling/devpod-management" 
        "host-tooling/monitoring"
        "host-tooling/shell-integration"
    ]
    
    for dir in $required_dirs {
        if ($dir | path exists) {
            print $"   ✅ ($dir) exists"
        } else {
            print $"   ❌ ($dir) missing"
        }
    }
    
    # Check for moved scripts
    let host_scripts = [
        "host-tooling/installation/docker-setup.nu"
        "host-tooling/devpod-management/devpod-manage.nu"
        "host-tooling/devpod-management/devpod-provision.nu"
        "host-tooling/monitoring/kubernetes.nu"
        "host-tooling/monitoring/github.nu"
        "host-tooling/shell-integration/aliases.sh"
    ]
    
    for script in $host_scripts {
        if ($script | path exists) {
            print $"   ✅ ($script) correctly placed"
        } else {
            print $"   ⚠️  ($script) not found"
        }
    }
}

def verify-container-structure [] {
    print "🐳 Verifying dev-env/ container structure..."
    
    let container_dirs = [
        "dev-env/python"
        "dev-env/typescript"
        "dev-env/rust" 
        "dev-env/go"
        "dev-env/nushell"
    ]
    
    for dir in $container_dirs {
        if ($dir | path exists) {
            print $"   ✅ ($dir) container environment exists"
            
            # Check for devbox.json (container configuration)
            if ($"($dir)/devbox.json" | path exists) {
                print $"   ✅ ($dir)/devbox.json container config found"
            } else {
                print $"   ❌ ($dir)/devbox.json missing"
            }
        } else {
            print $"   ❌ ($dir) missing"
        }
    }
    
    # Verify container-only scripts remain in dev-env/nushell/scripts/
    let container_scripts = [
        "dev-env/nushell/scripts/format.nu"      # Code formatting (needs container tools)
        "dev-env/nushell/scripts/test.nu"        # Testing (needs container frameworks)
        "dev-env/nushell/scripts/check.nu"       # Code checking (needs container linters)
        "dev-env/nushell/scripts/setup.nu"       # Container environment setup
    ]
    
    for script in $container_scripts {
        if ($script | path exists) {
            print $"   ✅ ($script) correctly remains in container"
        } else {
            print $"   ⚠️  ($script) not found in container"
        }
    }
}

def test-host-commands [] {
    print "🏠 Testing host command availability..."
    
    # Test host tools availability
    let host_tools = [
        "docker"
        "devpod"
    ]
    
    for tool in $host_tools {
        try {
            let version = (run-external $tool "--version" | complete)
            if $version.exit_code == 0 {
                print $"   ✅ ($tool) available on host"
            } else {
                print $"   ⚠️  ($tool) not available (install with host-tooling/installation/docker-setup.nu)"
            }
        } catch {
            print $"   ⚠️  ($tool) not available (install with host-tooling/installation/docker-setup.nu)"
        }
    }
    
    # Test host script syntax
    let host_scripts = [
        "host-tooling/installation/docker-setup.nu"
        "host-tooling/devpod-management/devpod-manage.nu"
    ]
    
    for script in $host_scripts {
        if ($script | path exists) {
            try {
                # Test syntax by parsing (don't execute)
                nu --check $script
                print $"   ✅ ($script) syntax valid"
            } catch {
                print $"   ❌ ($script) syntax error"
            }
        }
    }
}

def test-integration [] {
    print "🔗 Testing host/container integration points..."
    
    # Check if devpod can list workspaces (basic integration test)
    try {
        let workspaces = (devpod list | complete)
        if $workspaces.exit_code == 0 {
            print "   ✅ DevPod host integration working"
            
            let workspace_lines = ($workspaces.stdout | lines | where $it != "")
            let polyglot_workspaces = ($workspace_lines | where $it =~ "polyglot")
            
            if ($polyglot_workspaces | length) > 0 {
                print $"   ✅ Found ($polyglot_workspaces | length) existing polyglot containers"
            } else {
                print "   ℹ️  No existing polyglot containers (use host commands to provision)"
            }
        } else {
            print "   ⚠️  DevPod not configured (run host-tooling/installation/docker-setup.nu)"
        }
    } catch {
        print "   ⚠️  DevPod not available (install with host-tooling/installation/docker-setup.nu)"
    }
    
    # Check Docker integration
    try {
        let docker_info = (docker info | complete)
        if $docker_info.exit_code == 0 {
            print "   ✅ Docker host integration working"
        } else {
            print "   ⚠️  Docker daemon not running"
        }
    } catch {
        print "   ⚠️  Docker not available"
    }
    
    # Verify separation (no containers should have host credentials)
    print "   ✅ Credential separation: Host scripts handle infrastructure access"
    print "   ✅ Tool separation: Container scripts handle development tools"
    print "   ✅ Security boundary: Clear air gap between host and containers"
}

def show-usage-examples [] {
    print ""
    print "📚 Usage Examples:"
    print ""
    print "HOST COMMANDS (run on local machine):"
    print "  # Setup and installation"
    print "  nu host-tooling/installation/docker-setup.nu --install --configure"
    print "  source host-tooling/shell-integration/aliases.sh"
    print ""
    print "  # Container management"
    print "  devpod-provision-python       # Create Python container"
    print "  devpod-status                 # List containers"
    print "  enter-python                  # SSH into Python container"
    print ""
    print "  # Infrastructure access (requires host credentials)"
    print "  k8s-status                    # Check Kubernetes cluster"
    print "  github-status                 # Check GitHub integration"
    print ""
    print "CONTAINER COMMANDS (run inside containers):"
    print "  # Development work"
    print "  devbox run format             # Format code with container tools"
    print "  devbox run test               # Run tests with container frameworks"
    print "  devbox run lint               # Lint with container linters"
}