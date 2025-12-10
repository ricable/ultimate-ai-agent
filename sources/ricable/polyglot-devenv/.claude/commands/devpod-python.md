# DevPod Python Environment

Launch one or multiple DevPod workspaces for Python development.

## Usage

```bash
/devpod-python [count]
```

- `count`: Number of workspaces to provision (default: 1, max: 10)

## Description

Creates new DevPod workspace(s) for Python development with:
- Python 3.12.11 and uv 0.7.19 package manager
- VS Code with Python, Pylance, ESLint, and autopep8 extensions
- Python devcontainer with automatic setup
- Debugging support
- Access to your project files
- Unique workspace names with timestamps

## Command

```bash
cd /Users/cedric/dev/github.com/polyglot-devenv/dev-env/python

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

echo "🐍 Provisioning $COUNT Python DevPod workspace(s)..."

# Provision the requested number of workspaces
for i in $(seq 1 $COUNT); do
    echo ""
    echo "📦 Creating workspace $i of $COUNT..."
    
    # Generate unique workspace name with timestamp and sequence
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    WORKSPACE_NAME="polyglot-python-devpod-$TIMESTAMP-$i"
    WORKSPACE_SOURCE="/Users/cedric/dev/github.com/polyglot-devenv/dev-env/python"
    
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
echo "🎉 All $COUNT Python DevPod workspace(s) provisioned successfully!"
echo "💡 Use 'devpod list' to see all workspaces"
echo "💡 Use 'devpod stop <workspace-name>' to stop individual workspaces"
echo "🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces"
```

## Example Usage

```bash
/devpod-python          # Creates 1 workspace
/devpod-python 3         # Creates 3 workspaces  
/devpod-python 5         # Creates 5 workspaces
```

## Example Output

```
🐍 Provisioning 3 Python DevPod workspace(s)...

📦 Creating workspace 1 of 3...
🏷️  Workspace name: polyglot-python-devpod-20241206-220830-1
✅ Workspace 1 completed: polyglot-python-devpod-20241206-220830-1

📦 Creating workspace 2 of 3...
🏷️  Workspace name: polyglot-python-devpod-20241206-220832-2
✅ Workspace 2 completed: polyglot-python-devpod-20241206-220832-2

📦 Creating workspace 3 of 3...
🏷️  Workspace name: polyglot-python-devpod-20241206-220834-3
✅ Workspace 3 completed: polyglot-python-devpod-20241206-220834-3

🎉 All 3 Python DevPod workspace(s) provisioned successfully!
💡 Use 'devpod list' to see all workspaces
💡 Use 'devpod stop <workspace-name>' to stop individual workspaces
🧹 Use 'bash devpod-automation/scripts/provision-all.sh clean-all' to clean up old workspaces
```