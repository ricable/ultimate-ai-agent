#!/bin/bash

# Function to print colorful messages
print_message() {
    local color=$1
    local message=$2
    echo -e "\033[${color}m${message}\033[0m"
}

# Check for Deno installation
print_message "1;34" "🔍 Checking for Deno installation..."
if ! command -v deno &> /dev/null; then
    print_message "1;33" "⚙️  Installing Deno..."
    curl -fsSL https://deno.land/x/install/install.sh | sh
    
    # Add Deno to PATH
    export DENO_INSTALL="$HOME/.deno"
    export PATH="$DENO_INSTALL/bin:$PATH"
    
    # Add to shell configuration
    if [ -f "$HOME/.bashrc" ]; then
        echo 'export DENO_INSTALL="$HOME/.deno"' >> "$HOME/.bashrc"
        echo 'export PATH="$DENO_INSTALL/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export DENO_INSTALL="$HOME/.deno"' >> "$HOME/.zshrc"
        echo 'export PATH="$DENO_INSTALL/bin:$PATH"' >> "$HOME/.zshrc"
    fi
fi

# Check for Docker installation
print_message "1;34" "🔍 Checking for Docker installation..."
if ! command -v docker &> /dev/null; then
    print_message "1;33" "⚠️  Docker is not installed. Please install Docker to use containerized features."
    print_message "1;37" "Visit: https://docs.docker.com/get-docker/"
fi

# Create necessary directories
print_message "1;34" "📁 Creating project directories..."
mkdir -p .env
mkdir -p logs

# Check for environment variables
print_message "1;34" "🔑 Checking environment configuration..."
if [ ! -f ".env/.env" ]; then
    print_message "1;33" "⚙️  Creating environment file..."
    cat > .env/.env << EOL
# Cloud Provider Credentials
SUPABASE_PROJECT_ID=
SUPABASE_ACCESS_TOKEN=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_ACCOUNT_ID=
FLY_API_TOKEN=
FLY_APP_NAME=

# Server Configuration
PORT=3000
HOST=localhost
LOG_LEVEL=info
EOL
fi

# Cache dependencies
print_message "1;34" "📦 Caching dependencies..."
deno cache src/apps/deno/server.ts

# Setup complete
print_message "1;32" "✅ Installation complete!"
print_message "1;37" "To start the server, run: ./start.sh"
