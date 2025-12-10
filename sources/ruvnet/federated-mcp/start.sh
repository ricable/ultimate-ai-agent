#!/bin/bash

# Function to print colorful messages
print_message() {
    local color=$1
    local message=$2
    echo -e "\033[${color}m${message}\033[0m"
}

# Check if running in Docker
if [ "$USE_DOCKER" = "true" ]; then
    print_message "1;34" "🐳 Starting server in Docker container..."
    docker-compose up
else
    # Load environment variables
    if [ -f ".env/.env" ]; then
        export $(cat .env/.env | grep -v '^#' | xargs)
    fi

    print_message "1;34" "🚀 Starting AI Federation Network..."
    
    # Run the server with all required permissions
    deno run \
        --allow-net \
        --allow-env \
        --allow-read \
        --allow-write \
        --allow-run \
        src/apps/deno/server.ts
fi
