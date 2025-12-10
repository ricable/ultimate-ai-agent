#!/bin/bash

set -e  # Exit on error

echo "Installing Fly.io CLI and setting up environment..."
echo "================================================"

# Function to prompt for configuration value
prompt_config() {
    local var_name=$1
    local description=$2
    local default=$3
    local current_value=${!var_name}
    
    if [ -n "$current_value" ]; then
        read -p "$description [$current_value]: " value
        value=${value:-$current_value}
    else
        read -p "$description ${default:+[$default]: }" value
        value=${value:-$default}
    fi
    echo "$value"
}

# Check if config exists
CONFIG_FILE="$(dirname "$0")/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found. Let's create one..."
    cp "$(dirname "$0")/config.template.sh" "$CONFIG_FILE"
fi

# Interactive configuration
echo
echo "SPARC CLI Fly.io Configuration"
echo "============================="
echo "Press Enter to keep existing/default values or input new ones."
echo

# Source existing config if it exists
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Function to sanitize app name
sanitize_app_name() {
    # Convert to lowercase, replace underscores and spaces with dashes, remove other special chars
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr ' ' '-' | sed 's/[^a-z0-9-]//g'
}

# Fly.io Configuration
while true; do
    FLY_APP_NAME_INPUT=$(prompt_config "FLY_APP_NAME" "Enter Fly.io application name" "sparc-cli")
    FLY_APP_NAME=$(sanitize_app_name "$FLY_APP_NAME_INPUT")
    if [ "$FLY_APP_NAME" != "$FLY_APP_NAME_INPUT" ]; then
        echo "Note: App name sanitized to: $FLY_APP_NAME (only lowercase letters, numbers, and dashes allowed)"
    fi
    if [[ "$FLY_APP_NAME" =~ ^[a-z0-9-]+$ ]]; then
        break
    fi
    echo "Error: Invalid app name. Please use only lowercase letters, numbers, and dashes."
done
FLY_REGION=$(prompt_config "FLY_REGION" "Enter Fly.io region (e.g., lax, sfo, iad)" "lax")
FLY_ORG=$(prompt_config "FLY_ORG" "Enter Fly.io organization" "personal")

# Machine Configuration
DEPLOY_MACHINE_SIZE=$(prompt_config "DEPLOY_MACHINE_SIZE" "Enter machine size" "shared-cpu-1x")
DEPLOY_MEMORY=$(prompt_config "DEPLOY_MEMORY" "Enter memory in MB" "256")

# API Keys
echo
echo "API Keys Configuration"
echo "====================="
echo "ANTHROPIC_API_KEY is required, others are optional."
echo

while true; do
    ANTHROPIC_API_KEY=$(prompt_config "ANTHROPIC_API_KEY" "Enter Anthropic API Key (Required)")
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        break
    fi
    echo "Error: ANTHROPIC_API_KEY is required"
done

OPENAI_API_KEY=$(prompt_config "OPENAI_API_KEY" "Enter OpenAI API Key (Optional)")
OPENROUTER_KEY=$(prompt_config "OPENROUTER_KEY" "Enter OpenRouter Key (Optional)")
ENCRYPTION_KEY=$(prompt_config "ENCRYPTION_KEY" "Enter Encryption Key (Optional)")
GEMINI_API_KEY=$(prompt_config "GEMINI_API_KEY" "Enter Gemini API Key (Optional)")
VERTEXAI_PROJECT=$(prompt_config "VERTEXAI_PROJECT" "Enter Vertex AI Project (Optional)")
VERTEXAI_LOCATION=$(prompt_config "VERTEXAI_LOCATION" "Enter Vertex AI Location (Optional)")

# Save configuration
cat > "$CONFIG_FILE" << EOL
#!/bin/bash

# Fly.io Configuration
export FLY_APP_NAME="$FLY_APP_NAME"
export FLY_REGION="$FLY_REGION"
export FLY_ORG="$FLY_ORG"

# Git Repository Configuration
export GIT_REPO_URL=""  # Will be prompted during deployment
export GIT_BRANCH="main"

# API Keys
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENROUTER_KEY="$OPENROUTER_KEY"
export ENCRYPTION_KEY="$ENCRYPTION_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"
export VERTEXAI_PROJECT="$VERTEXAI_PROJECT"
export VERTEXAI_LOCATION="$VERTEXAI_LOCATION"

# Deployment Configuration
export DEPLOY_MACHINE_SIZE="$DEPLOY_MACHINE_SIZE"
export DEPLOY_MEMORY="$DEPLOY_MEMORY"
EOL

chmod +x "$CONFIG_FILE"

echo
echo "Configuration saved to $CONFIG_FILE"
echo

# Source the new configuration
source "$CONFIG_FILE"

# Check for required configuration
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is required in config.sh"
    exit 1
fi

# Detect OS and install flyctl
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! command -v flyctl &> /dev/null; then
        curl -L https://fly.io/install.sh | sh
        
        # Export for current session
        export FLYCTL_INSTALL="/home/$USER/.fly"
        export PATH="$FLYCTL_INSTALL/bin:$PATH"

        # Verify flyctl is in PATH
        if [ ! -f "$FLYCTL_INSTALL/bin/flyctl" ]; then
            echo "Error: flyctl not found at $FLYCTL_INSTALL/bin/flyctl"
            exit 1
        fi
        
        # Add to various shell profiles if they exist
        for profile in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
            if [ -f "$profile" ]; then
                if ! grep -q "FLYCTL_INSTALL" "$profile"; then
                    echo "" >> "$profile"
                    echo "# Fly.io CLI" >> "$profile"
                    echo "export FLYCTL_INSTALL=\"$FLYCTL_INSTALL\"" >> "$profile"
                    echo "export PATH=\"\$FLYCTL_INSTALL/bin:\$PATH\"" >> "$profile"
                fi
            fi
        done
        
        echo "Added Fly.io CLI to PATH. Please restart your shell or run:"
        echo "export FLYCTL_INSTALL=\"$FLYCTL_INSTALL\""
        echo "export PATH=\"\$FLYCTL_INSTALL/bin:\$PATH\""
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v flyctl &> /dev/null; then
        if command -v brew &> /dev/null; then
            brew install flyctl
        else
            echo "Error: Homebrew is required for macOS installation"
            echo "Install from https://brew.sh and try again"
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Verify installation
if ! command -v flyctl &> /dev/null; then
    echo "Error: Failed to install flyctl"
    exit 1
fi

echo "Fly.io CLI installed successfully"
flyctl version

# Check if already logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "Please log in to Fly.io..."
    flyctl auth login
fi

# Create the app if it doesn't exist
if ! flyctl apps list | grep -q "^$FLY_APP_NAME"; then
    echo "Creating Fly.io application: $FLY_APP_NAME"
    flyctl apps create "$FLY_APP_NAME" --org "$FLY_ORG"
fi

# Set secrets
echo "Setting up application secrets..."
flyctl secrets set \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENROUTER_KEY="$OPENROUTER_KEY" \
    ENCRYPTION_KEY="$ENCRYPTION_KEY" \
    GEMINI_API_KEY="$GEMINI_API_KEY" \
    VERTEXAI_PROJECT="$VERTEXAI_PROJECT" \
    VERTEXAI_LOCATION="$VERTEXAI_LOCATION" \
    --app "$FLY_APP_NAME"

echo "Installation and setup complete!"
echo "You can now run ./deploy.sh to deploy your application"
