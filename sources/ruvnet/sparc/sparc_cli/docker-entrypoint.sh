#!/bin/bash

# Function to check if a secret exists in GitHub Codespace secrets
check_codespace_secret() {
    local secret_name=$1
    if [ -n "${!secret_name}" ]; then
        echo "${!secret_name}"
        return 0
    fi
    return 1
}

# Function to check if a secret exists in .env file
check_env_file() {
    local secret_name=$1
    if [ -f .env ]; then
        local value=$(grep "^${secret_name}=" .env | cut -d '=' -f2)
        if [ -n "$value" ]; then
            echo "$value"
            return 0
        fi
    fi
    return 1
}

# Function to prompt for a secret if not found elsewhere
prompt_secret() {
    local secret_name=$1
    local required=$2
    echo "Enter value for ${secret_name}:"
    read -s value
    echo
    if [ -n "$value" ]; then
        echo "$value"
        return 0
    elif [ "$required" = "true" ]; then
        echo "Error: ${secret_name} is required" >&2
        exit 1
    fi
    return 1
}

# Function to get a secret from any available source
get_secret() {
    local secret_name=$1
    local required=$2
    local value

    # Try Codespace secrets first
    value=$(check_codespace_secret "$secret_name")
    if [ $? -eq 0 ]; then
        echo "$value"
        return 0
    fi

    # Try .env file next
    value=$(check_env_file "$secret_name")
    if [ $? -eq 0 ]; then
        echo "$value"
        return 0
    fi

    # Finally, prompt user if required
    if [ "$required" = "true" ] || [ "$secret_name" = "ANTHROPIC_API_KEY" ]; then
        value=$(prompt_secret "$secret_name" "$required")
        if [ $? -eq 0 ]; then
            echo "$value"
            return 0
        fi
    fi

    return 1
}

# Get all required secrets
ANTHROPIC_API_KEY=$(get_secret "ANTHROPIC_API_KEY" "true")
OPENAI_API_KEY=$(get_secret "OPENAI_API_KEY" "false")
OPENROUTER_KEY=$(get_secret "OPENROUTER_KEY" "false")
ENCRYPTION_KEY=$(get_secret "ENCRYPTION_KEY" "false")
GEMINI_API_KEY=$(get_secret "GEMINI_API_KEY" "false")
VERTEXAI_PROJECT=$(get_secret "VERTEXAI_PROJECT" "false")
VERTEXAI_LOCATION=$(get_secret "VERTEXAI_LOCATION" "false")

# Build Docker image from project root
cd "$(dirname "$0")/.." && docker build -t sparc-cli -f sparc_cli/Dockerfile .

# Execute the command passed to the script with environment variables
exec docker run -it \
    -e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" \
    -e "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    -e "OPENROUTER_KEY=${OPENROUTER_KEY}" \
    -e "ENCRYPTION_KEY=${ENCRYPTION_KEY}" \
    -e "GEMINI_API_KEY=${GEMINI_API_KEY}" \
    -e "VERTEXAI_PROJECT=${VERTEXAI_PROJECT}" \
    -e "VERTEXAI_LOCATION=${VERTEXAI_LOCATION}" \
    sparc-cli "$@"
