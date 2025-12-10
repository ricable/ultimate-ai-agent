#!/bin/bash
# install-hooks.sh - Install polyglot environment hooks for Claude Code

set -e

echo "🪝 Installing Claude Code Hooks for Polyglot Environment"
echo "======================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [[ ! -f "CLAUDE.md" ]] || [[ ! -d "python-env" ]] || [[ ! -d "typescript-env" ]]; then
    echo -e "${RED}❌ Error: This doesn't appear to be the polyglot development environment.${NC}"
    echo "Please run this script from the root of the polyglot project."
    exit 1
fi

# Function to backup existing settings
backup_settings() {
    local settings_file="$1"
    if [[ -f "$settings_file" ]]; then
        local backup_file="${settings_file}.backup-$(date +%Y%m%d_%H%M%S)"
        cp "$settings_file" "$backup_file"
        echo -e "${BLUE}💾 Backed up existing settings to $(basename "$backup_file")${NC}"
        return 0
    fi
    return 1
}

# Function to install hooks
install_hooks() {
    local scope="$1"  # "project" or "user"
    local config_file=".claude/polyglot-hooks-config.json"
    
    if [[ "$scope" == "project" ]]; then
        local settings_file=".claude/settings.json"
        echo -e "${YELLOW}📁 Installing project-specific hooks...${NC}"
    else
        local settings_file="$HOME/.claude/settings.json"
        echo -e "${YELLOW}👤 Installing user-global hooks...${NC}"
        
        # Create user .claude directory if it doesn't exist
        mkdir -p "$(dirname "$settings_file")"
    fi
    
    # Backup existing settings
    backup_settings "$settings_file"
    
    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}❌ Configuration file not found: $config_file${NC}"
        return 1
    fi
    
    # Create settings directory if it doesn't exist
    mkdir -p "$(dirname "$settings_file")"
    
    # Merge with existing settings or create new
    if [[ -f "$settings_file" ]]; then
        echo -e "${BLUE}🔄 Merging with existing settings...${NC}"
        
        # Use jq to merge configurations if available
        if command -v jq >/dev/null 2>&1; then
            local temp_file
            temp_file=$(mktemp)
            
            # Merge the hooks configuration
            jq -s '.[0] * .[1]' "$settings_file" "$config_file" > "$temp_file"
            mv "$temp_file" "$settings_file"
        else
            echo -e "${YELLOW}⚠️  jq not found. Replacing entire configuration...${NC}"
            cp "$config_file" "$settings_file"
        fi
    else
        echo -e "${BLUE}📄 Creating new settings file...${NC}"
        cp "$config_file" "$settings_file"
    fi
    
    echo -e "${GREEN}✅ Hooks installed to $settings_file${NC}"
    return 0
}

# Function to validate installation
validate_installation() {
    local settings_file="$1"
    
    if [[ ! -f "$settings_file" ]]; then
        echo -e "${RED}❌ Settings file not found: $settings_file${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🔍 Validating hook configuration...${NC}"
    
    if command -v jq >/dev/null 2>&1; then
        # Count hooks
        local pre_hooks post_hooks stop_hooks notification_hooks
        pre_hooks=$(jq '.hooks.PreToolUse // [] | length' "$settings_file" 2>/dev/null || echo "0")
        post_hooks=$(jq '.hooks.PostToolUse // [] | length' "$settings_file" 2>/dev/null || echo "0")
        stop_hooks=$(jq '.hooks.Stop // [] | length' "$settings_file" 2>/dev/null || echo "0")
        notification_hooks=$(jq '.hooks.Notification // [] | length' "$settings_file" 2>/dev/null || echo "0")
        
        echo "  📌 PreToolUse hooks: $pre_hooks"
        echo "  📌 PostToolUse hooks: $post_hooks"
        echo "  📌 Stop hooks: $stop_hooks"
        echo "  📌 Notification hooks: $notification_hooks"
        
        local total=$((pre_hooks + post_hooks + stop_hooks + notification_hooks))
        echo -e "${GREEN}📊 Total hooks configured: $total${NC}"
    else
        echo -e "${YELLOW}⚠️  jq not available, cannot validate hook count${NC}"
        echo "Settings file exists and appears to be properly formatted."
    fi
    
    return 0
}

# Function to test environment availability
test_environments() {
    echo -e "${BLUE}🧪 Testing environment availability...${NC}"
    
    local environments=("python-env" "typescript-env" "rust-env" "go-env" "nushell-env")
    local available=0
    local total=${#environments[@]}
    
    for env in "${environments[@]}"; do
        if [[ -d "$env" ]] && [[ -f "$env/devbox.json" ]]; then
            echo -e "  ✅ $env: Ready"
            ((available++))
        else
            echo -e "  ❌ $env: Not found or incomplete"
        fi
    done
    
    echo -e "${GREEN}📊 Environments ready: $available/$total${NC}"
    
    if [[ $available -lt $total ]]; then
        echo -e "${YELLOW}⚠️  Some environments are missing. Hooks will work for available environments only.${NC}"
    fi
}

# Function to show hook features
show_features() {
    echo -e "${BLUE}🎯 Installed Hook Features:${NC}"
    echo ""
    echo "📝 Auto-formatting:"
    echo "  • Python files (.py) → ruff format"
    echo "  • TypeScript/JS files (.ts/.js/.tsx/.jsx) → prettier"
    echo "  • Rust files (.rs) → rustfmt"
    echo "  • Go files (.go) → goimports"
    echo "  • Nushell files (.nu) → nu format"
    echo ""
    echo "🧪 Auto-testing:"
    echo "  • Python test files → pytest"
    echo "  • TypeScript/JS test files → jest"
    echo "  • Rust test files → cargo test"
    echo "  • Go test files → go test"
    echo "  • Nushell test files → nu test"
    echo ""
    echo "🔍 Pre-commit validation:"
    echo "  • Runs linting before git commits"
    echo "  • Scans configuration files for secrets"
    echo ""
    echo "🎯 Task completion automation:"
    echo "  • Shows git status"
    echo "  • Runs cross-language validation"
    echo ""
    echo "🔔 Notification logging:"
    echo "  • Logs all Claude Code notifications"
    echo ""
}

# Main script logic
main() {
    # Parse command line arguments
    local scope="project"  # default
    local show_help=false
    local test_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --user)
                scope="user"
                shift
                ;;
            --project)
                scope="project"
                shift
                ;;
            --test)
                test_only=true
                shift
                ;;
            --help|-h)
                show_help=true
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help=true
                shift
                ;;
        esac
    done
    
    if [[ "$show_help" == true ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --project    Install hooks for this project only (default)"
        echo "  --user       Install hooks globally for all projects"
        echo "  --test       Test environment availability only"
        echo "  --help, -h   Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Install project-specific hooks"
        echo "  $0 --user             # Install user-global hooks"
        echo "  $0 --test             # Test environment setup"
        return 0
    fi
    
    if [[ "$test_only" == true ]]; then
        test_environments
        return 0
    fi
    
    # Main installation process
    test_environments
    echo ""
    
    if install_hooks "$scope"; then
        echo ""
        
        # Validate installation
        if [[ "$scope" == "project" ]]; then
            validate_installation ".claude/settings.json"
        else
            validate_installation "$HOME/.claude/settings.json"
        fi
        
        echo ""
        show_features
        
        echo ""
        echo -e "${GREEN}🎉 Claude Code hooks installation completed successfully!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Start Claude Code in this directory"
        echo "  2. Edit some files to see automatic formatting"
        echo "  3. Run a git commit to see pre-commit validation"
        echo "  4. Finish a task to see cross-language validation"
        echo ""
        echo "To manage hooks: nu nushell-env/scripts/hooks.nu status"
        
    else
        echo -e "${RED}❌ Installation failed${NC}"
        return 1
    fi
}

# Run main function with all arguments
main "$@"