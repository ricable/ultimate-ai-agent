#!/bin/bash

# Ericsson RAN Intelligent Multi-Agent System - Repository Setup Script
# This script automates the initial GitHub repository setup with all configurations

set -e

echo "🚀 Setting up Ericsson RAN Intelligent Multi-Agent System repository..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    print_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
    exit 1
fi

# Get repository information
REPO_NAME=${1:-"ran-automation-agentdb"}
ORG_NAME=${2:-"ericsson-ran"}
DESCRIPTION="Ericsson RAN Intelligent Multi-Agent System with Cognitive Consciousness"

print_status "Creating repository: $ORG_NAME/$REPO_NAME"

# Create repository if it doesn't exist
if gh repo view "$ORG_NAME/$REPO_NAME" &> /dev/null; then
    print_warning "Repository $ORG_NAME/$REPO_NAME already exists"
else
    gh repo create "$ORG_NAME/$REPO_NAME" \
        --public \
        --description "$DESCRIPTION" \
        --clone=false \
        --disable-wiki=false \
        --disable-issues=false \
        --disable-projects=false
    print_success "Repository created successfully"
fi

# Setup repository settings
print_status "Configuring repository settings..."

# Enable required features
gh repo edit "$ORG_NAME/$REPO_NAME" \
    --enable-merge-commit=false \
    --enable-squash-merge=true \
    --enable-rebase-merge=true \
    --delete-branch-on-merge=true \
    --enable-issues=true \
    --enable-projects=true \
    --enable-wiki=true

# Setup branch protection rules
print_status "Setting up branch protection rules..."

# Protect main branch
gh api repos/$ORG_NAME/$REPO_NAME/branches/main/protection \
    --method PUT \
    --field required_status_checks='{"strict":true,"contexts":["quality-gate","performance-benchmarks","security-scan"]}' \
    --field enforce_admins=true \
    --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":false}' \
    --field restrictions=null \
    --field allow_force_pushes=false \
    --field allow_deletions=false || print_warning "Branch protection already exists or failed"

# Create default branches
print_status "Setting up development branches..."

for branch in develop staging; do
    if ! gh api repos/$ORG_NAME/$REPO_NAME/branches/$branch &> /dev/null; then
        gh api repos/$ORG_NAME/$REPO_NAME/git/refs \
            --method POST \
            --field ref="refs/heads/$branch" \
            --field sha="$(gh api repos/$ORG_NAME/$REPO_NAME/git/refs/heads/main --jq '.object.sha')" || print_warning "Branch $branch already exists"
    fi
done

# Setup project board
print_status "Creating project board..."

# Create GitHub project
PROJECT_ID=$(gh api orgs/$ORG_NAME/projects \
    --method POST \
    --field name="Phase 1 RAN Development" \
    --field body="Automated project board for Phase 1 development" \
    --jq '.id' 2>/dev/null || echo "")

if [ -n "$PROJECT_ID" ]; then
    print_success "Project board created with ID: $PROJECT_ID"

    # Create columns
    COLUMNS=("Backlog" "To Do" "In Progress" "PR Review" "Testing" "Done")
    for column in "${COLUMNS[@]}"; do
        gh api projects/$PROJECT_ID/columns \
            --method POST \
            --field name="$column" > /dev/null
    done
    print_success "Project board columns created"
else
    print_warning "Project board creation failed or already exists"
fi

# Setup labels
print_status "Creating repository labels..."

# Priority labels
gh label create "priority:critical" \
    --color "B91C1C" \
    --description "Critical priority - immediate attention required" \
    --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label already exists"

gh label create "priority:high" \
    --color "DC2626" \
    --description "High priority - address soon" \
    --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label already exists"

gh label create "priority:medium" \
    --color "F59E0B" \
    --description "Medium priority - normal workflow" \
    --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label already exists"

gh label create "priority:low" \
    --color "10B981" \
    --description "Low priority - can wait" \
    --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label already exists"

# Component labels
COMPONENTS=(
    "component:energy-optimizer:0891B2:Energy efficiency optimization components"
    "component:mobility-manager:7C3AED:Mobility management and handover optimization"
    "component:coverage-analyzer:059669:Coverage analysis and signal optimization"
    "component:performance-analyst:EA580C:Performance monitoring and analysis"
    "component:agentdb:DC2626:AgentDB integration and memory management"
    "component:ci-cd:4B5563:CI/CD pipelines and automation"
    "component:testing:7C2D12:Test automation and quality assurance"
    "component:documentation:1E40AF:Documentation and guides"
)

for component in "${COMPONENTS[@]}"; do
    IFS=':' read -r name color description <<< "$component"
    gh label create "$name" \
        --color "$color" \
        --description "$description" \
        --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label $name already exists"
done

# Type labels
TYPES=(
    "type:bug:DC2626:Bug fixes and error corrections"
    "type:feature:059669:New features and enhancements"
    "type:enhancement:0891B2:Improvements to existing features"
    "type:documentation:1E40AF:Documentation improvements"
    "type:test:7C2D12:Test-related tasks and improvements"
    "type:refactor:6B7280:Code refactoring and cleanup"
)

for type in "${TYPES[@]}"; do
    IFS=':' read -r name color description <<< "$type"
    gh label create "$name" \
        --color "$color" \
        --description "$description" \
        --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label $name already exists"
done

# Size labels
SIZES=(
    "size:small:10B981:Small task (1-2 hours)"
    "size:medium:F59E0B:Medium task (4-8 hours)"
    "size:large:DC2626:Large task (1-2 days)"
)

for size in "${SIZES[@]}"; do
    IFS=':' read -r name color description <<< "$size"
    gh label create "$name" \
        --color "$color" \
        --description "$description" \
        --repo "$ORG_NAME/$REPO_NAME" || print_warning "Label $name already exists"
done

# Setup milestones
print_status "Creating milestones..."

MILESTONES=(
    "Phase 1 Week 1: Foundation Setup:2025-11-07:Core infrastructure and cognitive consciousness setup"
    "Phase 1 Week 2: AgentDB Integration:2025-11-14:AgentDB integration and memory pattern optimization"
    "Phase 1 Week 3: Cognitive Features:2025-11-21:Temporal reasoning and strange-loop cognition"
    "Phase 1 Week 4: Optimization & Testing:2025-11-28:Performance optimization and comprehensive testing"
)

for milestone in "${MILESTONES[@]}"; do
    IFS=':' read -r title due_date description <<< "$milestone"
    gh api repos/$ORG_NAME/$REPO_NAME/milestones \
        --method POST \
        --field title="$title" \
        --field due_on="$due_date"T23:59:59Z \
        --field description="$description" > /dev/null || print_warning "Milestone $title already exists"
done

# Setup teams (if organization)
print_status "Setting up teams..."

TEAMS=(
    "core-team:Core Development Team:System architecture and core cognitive features"
    "energy-team:Energy Optimization Team:Energy efficiency and power consumption optimization"
    "mobility-team:Mobility Management Team:Handover optimization and mobility prediction"
    "coverage-team:Coverage Analysis Team:Signal strength optimization and coverage analysis"
    "performance-team:Performance Team:System performance monitoring and bottleneck identification"
    "agentdb-team:AgentDB Team:AgentDB integration and memory pattern optimization"
    "devops-team:DevOps Team:CI/CD pipelines and infrastructure automation"
    "qa-team:Quality Assurance Team:Test automation and quality gate enforcement"
    "docs-team:Documentation Team:API documentation and technical specifications"
    "senior-team:Senior Leadership Team:Technical direction and architecture decisions"
)

for team in "${TEAMS[@]}"; do
    IFS=':' read -r name description privacy <<< "$team"
    gh api orgs/$ORG_NAME/teams \
        --method POST \
        --field name="$name" \
        --field description="$description" \
        --field privacy="closed" > /dev/null || print_warning "Team $name already exists"

    # Add team to repository with appropriate permissions
    case $name in
        "senior-team"|"core-team")
            PERMISSION="admin"
            ;;
        "devops-team"|"qa-team")
            PERMISSION="maintain"
            ;;
        *)
            PERMISSION="write"
            ;;
    esac

    gh api orgs/$ORG_NAME/teams/$name/repos/$ORG_NAME/$REPO_NAME \
        --method PUT \
        --field permission="$PERMISSION" > /dev/null || print_warning "Failed to add team $name to repository"
done

# Setup secrets (template)
print_status "Creating secrets template..."

cat > .github/secrets-template.md << 'EOF'
# Required Secrets Configuration

## Essential Secrets
- `GITHUB_TOKEN`: GitHub token for API access (automatically provided)
- `AGENTDB_ENDPOINT`: AgentDB service endpoint
- `SONAR_HOST_URL`: SonarQube server URL
- `SONAR_TOKEN`: SonarQube authentication token
- `AWS_ACCESS_KEY_ID`: AWS access key for deployment
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for deployment
- `SLACK_WEBHOOK_URL`: Slack webhook for notifications

## Optional Secrets
- `DOCKER_REGISTRY_URL`: Custom Docker registry URL
- `DOCKER_USERNAME`: Docker registry username
- `DOCKER_PASSWORD`: Docker registry password
- `CODECOV_TOKEN`: Codecov token for coverage reports
- `NPM_TOKEN`: NPM token for package publishing

## Setup Instructions

1. Navigate to repository settings
2. Go to "Secrets and variables" > "Actions"
3. Add each secret with the appropriate value
4. Configure environment-specific secrets as needed

## AgentDB Integration

The AgentDB endpoint should be configured as:
```
https://your-agentdb-instance.com/api/v1
```

Ensure the endpoint is accessible from GitHub Actions runners.
EOF

print_success "Repository setup completed!"

# Final summary
echo ""
echo "🎉 Ericsson RAN Intelligent Multi-Agent System Repository Setup Complete!"
echo ""
echo "📋 Summary:"
echo "  • Repository: $ORG_NAME/$REPO_NAME"
echo "  • Project Board: Phase 1 RAN Development"
echo "  • Branches: main, develop, staging"
echo "  • Labels: $((${#COMPONENTS[@]} + ${#TYPES[@]} + ${#SIZES[@]} + 4)) total"
echo "  • Milestones: ${#MILESTONES[@]} created"
echo "  • Teams: ${#TEAMS[@]} configured"
echo ""
echo "🔧 Next Steps:"
echo "  1. Configure required secrets in repository settings"
echo "  2. Review and customize team permissions"
echo "  3. Setup SonarQube integration"
echo "  4. Configure AgentDB endpoint"
echo "  5. Test GitHub Actions workflows"
echo ""
echo "📚 Documentation:"
echo "  • See .github/secrets-template.md for required secrets"
echo "  • Review .github/workflows/ for automation workflows"
echo "  • Check config/github/project-config.json for project settings"
echo ""
print_success "Happy coding with Cognitive RAN Consciousness! 🧠✨"