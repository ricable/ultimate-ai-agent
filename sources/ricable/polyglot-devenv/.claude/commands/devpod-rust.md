# DevPod Rust Environment

Launch one or multiple DevPod workspaces for Rust development.

## Usage

```bash
/devpod-rust [count]
```

- `count`: Number of workspaces to provision (default: 1, max: 10)

## Description

Creates new DevPod workspace(s) for Rust development with:
- Rust 1.87 toolchain and Cargo
- VS Code with rust-analyzer, CodeLLDB, and TOML extensions
- Debugging support with LLDB
- Clippy and rustfmt for code quality
- Automatic language detection and container setup
- Access to your project files
- Unique workspace names with timestamps

## Command

```bash
cd /Users/cedric/dev/github.com/polyglot-devenv/dev-env/rust

# Get count from argument or default to 1
COUNT=${ARGUMENT:-1}

# Validate count (max 10 to prevent resource exhaustion)
if [[ "$COUNT" -gt 10 ]]; then
    echo "⚠️  Maximum 10 workspaces allowed. Setting count to 10."
    COUNT=10
fi

if [[ "$COUNT" -lt 1 ]]; then
    echo "⚠️  Count must be at least 1. Setting count to 1."
    COUNT=1
fi

echo "🦀 Provisioning $COUNT Rust DevPod workspace(s)..."

# Provision the requested number of workspaces
for i in $(seq 1 $COUNT); do
    echo ""
    echo "📦 Creating workspace $i of $COUNT..."
    
    # Generate unique workspace name with timestamp and sequence
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    WORKSPACE_NAME="polyglot-rust-devpod-$TIMESTAMP-$i"
    WORKSPACE_SOURCE="/Users/cedric/dev/github.com/polyglot-devenv/dev-env/rust"
    
    echo "🏷️  Workspace name: $WORKSPACE_NAME"
    
    # Create workspace
    devpod up "$WORKSPACE_SOURCE" \
        --id "$WORKSPACE_NAME" \
        --ide vscode \
        --provider docker
    
    echo "✅ Workspace $i completed: $WORKSPACE_NAME"
    
    # Small delay between provisions to avoid conflicts
    if [[ $i -lt $COUNT ]]; then
        sleep 2
    fi
done

echo ""
echo "🎉 All $COUNT Rust DevPod workspace(s) provisioned successfully!"
echo "💡 Use 'devpod list' to see all workspaces"
echo "💡 Use 'devpod stop <workspace-name>' to stop individual workspaces"
echo "🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces"
```

## Example Usage

```bash
/devpod-rust          # Creates 1 workspace
/devpod-rust 3         # Creates 3 workspaces  
/devpod-rust 5         # Creates 5 workspaces
```

## Example Output

```
🦀 Provisioning 3 Rust DevPod workspace(s)...

📦 Creating workspace 1 of 3...
🏷️  Workspace name: polyglot-rust-devpod-20241206-220830-1
✅ Workspace 1 completed: polyglot-rust-devpod-20241206-220830-1

📦 Creating workspace 2 of 3...
🏷️  Workspace name: polyglot-rust-devpod-20241206-220832-2
✅ Workspace 2 completed: polyglot-rust-devpod-20241206-220832-2

📦 Creating workspace 3 of 3...
🏷️  Workspace name: polyglot-rust-devpod-20241206-220834-3
✅ Workspace 3 completed: polyglot-rust-devpod-20241206-220834-3

🎉 All 3 Rust DevPod workspace(s) provisioned successfully!
💡 Use 'devpod list' to see all workspaces
💡 Use 'devpod stop <workspace-name>' to stop individual workspaces
🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces
```