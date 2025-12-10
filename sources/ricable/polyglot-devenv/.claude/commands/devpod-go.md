# DevPod Go Environment

Launch one or multiple DevPod workspaces for Go development.

## Usage

```bash
/devpod-go [count]
```

- `count`: Number of workspaces to provision (default: 1, max: 10)

## Description

Creates new DevPod workspace(s) for Go development with:
- Go 1.22 toolchain and modules
- VS Code with Go extension and ESLint
- golangci-lint for code quality
- Debugging support with Delve
- Automatic language detection and container setup
- Access to your project files
- Unique workspace names with timestamps

## Command

```bash
cd /Users/cedric/dev/github.com/polyglot-devenv/dev-env/go

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

echo "🐹 Provisioning $COUNT Go DevPod workspace(s)..."

# Provision the requested number of workspaces
for i in $(seq 1 $COUNT); do
    echo ""
    echo "📦 Creating workspace $i of $COUNT..."
    
    # Generate unique workspace name with timestamp and sequence
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    WORKSPACE_NAME="polyglot-go-devpod-$TIMESTAMP-$i"
    WORKSPACE_SOURCE="/Users/cedric/dev/github.com/polyglot-devenv/dev-env/go"
    
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
echo "🎉 All $COUNT Go DevPod workspace(s) provisioned successfully!"
echo "💡 Use 'devpod list' to see all workspaces"
echo "💡 Use 'devpod stop <workspace-name>' to stop individual workspaces"
echo "🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces"
```

## Example Usage

```bash
/devpod-go          # Creates 1 workspace
/devpod-go 3         # Creates 3 workspaces  
/devpod-go 5         # Creates 5 workspaces
```

## Example Output

```
🐹 Provisioning 3 Go DevPod workspace(s)...

📦 Creating workspace 1 of 3...
🏷️  Workspace name: polyglot-go-devpod-20241206-220830-1
✅ Workspace 1 completed: polyglot-go-devpod-20241206-220830-1

📦 Creating workspace 2 of 3...
🏷️  Workspace name: polyglot-go-devpod-20241206-220832-2
✅ Workspace 2 completed: polyglot-go-devpod-20241206-220832-2

📦 Creating workspace 3 of 3...
🏷️  Workspace name: polyglot-go-devpod-20241206-220834-3
✅ Workspace 3 completed: polyglot-go-devpod-20241206-220834-3

🎉 All 3 Go DevPod workspace(s) provisioned successfully!
💡 Use 'devpod list' to see all workspaces
💡 Use 'devpod stop <workspace-name>' to stop individual workspaces
🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces
```