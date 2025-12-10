# Migration Guide: Host/Container Separation

This guide helps migrate from the previous mixed-responsibility structure to the new clear host/container separation.

## What Changed

### Scripts Moved to Host Tooling

**From `devpod-automation/scripts/` to `host-tooling/devpod-management/`**:
- `devpod-generate.nu` - Container generation (HOST)
- `devpod-manage.nu` - Container lifecycle management (HOST)
- `devpod-provision.nu` - Container provisioning (HOST)
- `devpod-sync.nu` - Host-container synchronization (HOST)

**From `devpod-automation/scripts/` to `host-tooling/installation/`**:
- `docker-setup.nu` - Docker & DevPod installation (HOST)

**From `dev-env/nushell/scripts/` to `host-tooling/monitoring/`**:
- `kubernetes.nu` - Kubernetes cluster management (HOST, requires credentials)
- `github.nu` - GitHub integration (HOST, requires credentials)

### Scripts Remaining in Containers

**Container-only scripts in `dev-env/nushell/scripts/`**:
- `format.nu` - Code formatting (needs container tools)
- `test.nu` - Testing (needs container frameworks)
- `check.nu` - Code validation (needs container linters)
- `setup.nu` - Container environment setup
- `performance-analytics.nu` - Container performance monitoring
- `resource-monitor.nu` - Container resource usage
- `security-scanner.nu` - Code security analysis
- All other development-focused automation

## Migration Steps

### 1. Update Shell Configuration

**Old way**:
```bash
# Mixed aliases that ran from anywhere
alias devpod-status="devpod list"
```

**New way**:
```bash
# Source host-specific aliases
source host-tooling/shell-integration/aliases.sh

# Now you have clear host commands:
devpod-status                 # Host: List containers
devpod-provision-python       # Host: Create Python container
enter-python                  # Host: SSH into Python container
```

### 2. Update Command Usage

**Old way** (ambiguous location):
```bash
# Could run from anywhere, unclear context
nu scripts/docker-setup.nu --install
nu scripts/devpod-provision.nu python
nu scripts/kubernetes.nu status
```

**New way** (clear host/container separation):
```bash
# HOST commands (require host credentials, manage containers)
nu host-tooling/installation/docker-setup.nu --install --configure
nu host-tooling/devpod-management/devpod-provision.nu python
nu host-tooling/monitoring/kubernetes.nu status

# CONTAINER commands (run inside containers with development tools)
# First enter container: enter-python
devbox run format             # Inside container
devbox run test              # Inside container
devbox run lint              # Inside container
```

### 3. Update Scripts and Automation

**Update import paths** in any custom scripts:
```bash
# Old import paths
source scripts/docker-setup.nu
nu scripts/devpod-manage.nu

# New import paths  
source host-tooling/installation/docker-setup.nu
nu host-tooling/devpod-management/devpod-manage.nu
```

**Update CI/CD and automation**:
```yaml
# OLD: Mixed responsibilities
- run: nu scripts/docker-setup.nu --install
- run: cd dev-env/python && nu ../nushell/scripts/kubernetes.nu deploy

# NEW: Clear separation
- name: "Host: Install Docker/DevPod"
  run: nu host-tooling/installation/docker-setup.nu --install
- name: "Host: Deploy to Kubernetes"  
  run: nu host-tooling/monitoring/kubernetes.nu deploy
- name: "Container: Run tests"
  run: devpod ssh python-container "devbox run test"
```

### 4. Update Documentation References

**Update any documentation** that references old script locations:
- `devpod-automation/scripts/` → `host-tooling/devpod-management/`
- `dev-env/nushell/scripts/kubernetes.nu` → `host-tooling/monitoring/kubernetes.nu`
- `dev-env/nushell/scripts/github.nu` → `host-tooling/monitoring/github.nu`

### 5. Credential Management

**Host credentials** (remain on host):
- Kubernetes config (`~/.kube/config`)
- GitHub tokens (`$GITHUB_TOKEN`)
- Docker registry credentials
- Cloud provider keys

**Container isolation** (no host credentials):
- Development happens in isolated containers
- Source code mounted from host
- No access to host credentials or infrastructure

## Benefits After Migration

### Security Improvements
- **Credential Isolation**: Host credentials never enter containers
- **Attack Surface Reduction**: Containers can't access host infrastructure
- **Principle of Least Privilege**: Clear separation of responsibilities

### Developer Experience
- **Clear Mental Model**: Host vs container responsibilities are obvious
- **Easier Onboarding**: New developers understand the architecture immediately
- **Reduced Confusion**: No ambiguity about where scripts should run

### Operational Benefits
- **Reproducible Environments**: Containers are identical across developers
- **Infrastructure Control**: Host maintains access to clusters and external services
- **Clean Separation**: Easy to backup, restore, and manage each layer independently

## Verification

Run the verification script to ensure migration was successful:
```bash
nu host-tooling/verify-separation.nu
```

This will check:
- ✅ Host tooling structure is correct
- ✅ Container environments are intact  
- ✅ Script locations are appropriate
- ✅ Integration points work correctly

## Rollback (If Needed)

If you need to rollback temporarily:
```bash
# Move scripts back (not recommended)
mv host-tooling/devpod-management/* devpod-automation/scripts/
mv host-tooling/monitoring/kubernetes.nu dev-env/nushell/scripts/
mv host-tooling/monitoring/github.nu dev-env/nushell/scripts/
mv host-tooling/installation/docker-setup.nu devpod-automation/scripts/

# Remove host-tooling (not recommended)
rm -rf host-tooling/
```

**However**, the new separation provides significant benefits and is the recommended approach going forward.

## Support

If you encounter issues with the migration:
1. Run `nu host-tooling/verify-separation.nu` to diagnose problems
2. Check that Docker and DevPod are properly installed on the host
3. Verify that container environments still have their devbox.json files
4. Ensure you're sourcing `host-tooling/shell-integration/aliases.sh` for host commands

The separation creates a more secure, maintainable, and understandable development environment.