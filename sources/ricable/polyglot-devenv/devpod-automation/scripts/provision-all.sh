#!/bin/bash
# DevPod provisioning overview and bulk operations script
# NOTE: This script runs on the HOST machine and delegates to host-tooling/ scripts

set -e

echo "🐳 DevPod Polyglot Environment Manager (Host Script)"
echo "===================================================="
echo "⚠️  IMPORTANT: DevPod management scripts moved to host-tooling/"
echo "   Use: nu host-tooling/devpod-management/devpod-manage.nu"
echo "   Or source: host-tooling/shell-integration/aliases.sh"

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  status     Show all workspace status"
    echo "  list       List all available provision scripts"
    echo "  provision  Provision all environments (interactive)"
    echo "  stop-all   Stop all running workspaces"
    echo "  clean-all  Delete all workspaces"
    echo ""
    echo "EXAMPLES:"
    echo "  ./provision-all.sh status"
    echo "  ./provision-all.sh provision"
    echo "  ./provision-all.sh stop-all"
}

show_status() {
    echo "📊 Current DevPod Workspace Status:"
    echo "-----------------------------------"
    devpod list
    echo ""
    
    # Check individual workspace status
    workspaces=("polyglot-python-devpod" "polyglot-typescript-devpod" "polyglot-rust-devpod" "polyglot-go-devpod" "polyglot-nushell-devpod")
    
    echo "🔍 Detailed Status:"
    for workspace in "${workspaces[@]}"; do
        if devpod list | grep -q "$workspace"; then
            echo "  ✅ $workspace: Running"
        else
            echo "  ❌ $workspace: Not provisioned"
        fi
    done
}

list_scripts() {
    echo "📋 Available DevPod Provision Scripts:"
    echo "-------------------------------------"
    echo "  🐍 Python:     bash devpod-automation/scripts/provision-python.sh"
    echo "  📘 TypeScript: bash devpod-automation/scripts/provision-typescript.sh"
    echo "  🦀 Rust:       bash devpod-automation/scripts/provision-rust.sh"
    echo "  🐹 Go:         bash devpod-automation/scripts/provision-go.sh"
    echo "  🐚 Nushell:    bash devpod-automation/scripts/provision-nushell.sh"
    echo ""
    echo "Or use from within environments:"
    echo "  cd python-env && devbox run devpod:provision"
    echo "  cd typescript-env && devbox run devpod:provision"
    echo "  cd rust-env && devbox run devpod:provision"
    echo "  cd go-env && devbox run devpod:provision"
    echo "  cd nushell-env && devbox run devpod:provision"
}

provision_all() {
    echo "🚀 Interactive DevPod Provisioning"
    echo "=================================="
    
    languages=("python" "typescript" "rust" "go" "nushell")
    
    for lang in "${languages[@]}"; do
        echo ""
        read -p "🔹 Provision $lang environment? (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "📦 Provisioning $lang..."
            bash "$(dirname "$0")/provision-$lang.sh"
        else
            echo "⏭️  Skipping $lang"
        fi
    done
    
    echo ""
    echo "✅ Provisioning complete!"
    show_status
}

stop_all() {
    echo "🛑 Stopping all DevPod workspaces..."
    
    workspaces=($(devpod list --output json | jq -r '.[].name' 2>/dev/null || devpod list | tail -n +2 | awk '{print $1}'))
    
    if [ ${#workspaces[@]} -eq 0 ]; then
        echo "📝 No workspaces running"
        return
    fi
    
    for workspace in "${workspaces[@]}"; do
        if [[ "$workspace" == "polyglot-"* ]]; then
            echo "🛑 Stopping $workspace..."
            devpod stop "$workspace" || echo "⚠️  Could not stop $workspace"
        fi
    done
    
    echo "✅ All workspaces stopped"
}

clean_all() {
    echo "🗑️  Cleaning all DevPod workspaces..."
    echo "⚠️  This will delete all polyglot workspaces permanently!"
    
    read -p "Are you sure? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled"
        return
    fi
    
    workspaces=($(devpod list --output json | jq -r '.[].name' 2>/dev/null || devpod list | tail -n +2 | awk '{print $1}'))
    
    for workspace in "${workspaces[@]}"; do
        if [[ "$workspace" == "polyglot-"* ]]; then
            echo "🗑️  Deleting $workspace..."
            devpod delete "$workspace" --force || echo "⚠️  Could not delete $workspace"
        fi
    done
    
    echo "✅ All workspaces cleaned"
}

# Main command handling
case "${1:-status}" in
    "status"|"s")
        show_status
        ;;
    "list"|"l")
        list_scripts
        ;;
    "provision"|"p")
        provision_all
        ;;
    "stop-all"|"stop")
        stop_all
        ;;
    "clean-all"|"clean")
        clean_all
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac