#!/bin/bash
#
# GCP Secret Manager Setup Script
# Creates and manages secrets for agentic-flow deployment
#

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-}"
ENV_FILE="${ENV_FILE:-../../.env}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set"
        exit 1
    fi

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found"
        exit 1
    fi

    log_info "Prerequisites OK"
}

create_secret() {
    local secret_name=$1
    local secret_value=$2

    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &> /dev/null; then
        log_warn "Secret $secret_name exists, adding new version..."
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" \
            --data-file=- \
            --project="$PROJECT_ID"
    else
        log_info "Creating secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets create "$secret_name" \
            --data-file=- \
            --replication-policy="automatic" \
            --project="$PROJECT_ID"
    fi
}

setup_secrets_from_env() {
    log_info "Setting up secrets from $ENV_FILE"

    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi

    # Read .env and create secrets
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z $key ]] && continue

        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)

        # Skip if value is empty
        [[ -z $value ]] && continue

        create_secret "$key" "$value"
    done < <(grep -v '^#' "$ENV_FILE" | grep '=')

    log_info "Secrets setup complete"
}

setup_secrets_interactive() {
    log_info "Interactive secret setup"

    # Anthropic API Key
    read -rsp "Enter Anthropic API Key (or press Enter to skip): " anthropic_key
    echo
    if [ -n "$anthropic_key" ]; then
        create_secret "ANTHROPIC_API_KEY" "$anthropic_key"
    fi

    # OpenRouter API Key
    read -rsp "Enter OpenRouter API Key (or press Enter to skip): " openrouter_key
    echo
    if [ -n "$openrouter_key" ]; then
        create_secret "OPENROUTER_API_KEY" "$openrouter_key"
    fi

    # E2B API Key
    read -rsp "Enter E2B API Key (or press Enter to skip): " e2b_key
    echo
    if [ -n "$e2b_key" ]; then
        create_secret "E2B_API_KEY" "$e2b_key"
    fi

    log_info "Interactive setup complete"
}

list_secrets() {
    log_info "Listing secrets for project: $PROJECT_ID"
    gcloud secrets list --project="$PROJECT_ID" --format="table(name,created)"
}

delete_secret() {
    local secret_name=$1
    log_warn "Deleting secret: $secret_name"
    gcloud secrets delete "$secret_name" --project="$PROJECT_ID" --quiet
    log_info "Secret deleted: $secret_name"
}

grant_access() {
    local secret_name=$1
    local service_account="${2:-${PROJECT_ID}@appspot.gserviceaccount.com}"

    log_info "Granting access to $service_account for secret $secret_name"

    gcloud secrets add-iam-policy-binding "$secret_name" \
        --member="serviceAccount:$service_account" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID"

    log_info "Access granted"
}

grant_all_access() {
    local service_account="${1:-${PROJECT_ID}@appspot.gserviceaccount.com}"

    log_info "Granting access to all secrets for $service_account"

    # Get all secret names
    secrets=$(gcloud secrets list --project="$PROJECT_ID" --format="value(name)")

    for secret in $secrets; do
        grant_access "$secret" "$service_account"
    done

    log_info "Access granted to all secrets"
}

show_usage() {
    cat <<EOF
Usage: $0 [command]

Commands:
    setup           Setup secrets from .env file
    interactive     Interactive secret setup
    list            List all secrets
    grant <secret>  Grant access to specific secret
    grant-all       Grant access to all secrets
    delete <secret> Delete a secret

Environment Variables:
    GCP_PROJECT_ID  GCP project ID (required)
    ENV_FILE        Path to .env file (default: ../../.env)

Examples:
    GCP_PROJECT_ID=my-project $0 setup
    GCP_PROJECT_ID=my-project $0 interactive
    GCP_PROJECT_ID=my-project $0 list
    GCP_PROJECT_ID=my-project $0 grant ANTHROPIC_API_KEY
    GCP_PROJECT_ID=my-project $0 grant-all
    GCP_PROJECT_ID=my-project $0 delete OLD_SECRET

EOF
}

main() {
    check_prerequisites

    case "${1:-setup}" in
        setup)
            setup_secrets_from_env
            ;;
        interactive)
            setup_secrets_interactive
            ;;
        list)
            list_secrets
            ;;
        grant)
            if [ -z "${2:-}" ]; then
                log_error "Secret name required"
                show_usage
                exit 1
            fi
            grant_access "$2" "${3:-}"
            ;;
        grant-all)
            grant_all_access "${2:-}"
            ;;
        delete)
            if [ -z "${2:-}" ]; then
                log_error "Secret name required"
                show_usage
                exit 1
            fi
            delete_secret "$2"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
