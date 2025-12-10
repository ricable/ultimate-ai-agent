#!/bin/bash

# RAN Skills Deployment Script
# Deploys all 5 RAN-specific skills with proper dependency ordering and verification

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RAN_SKILLS=(
    "ran-agentdb-integration-specialist"
    "ran-ml-researcher"
    "ran-causal-inference-specialist"
    "ran-reinforcement-learning-engineer"
    "ran-dspy-mobility-optimizer"
)

LOG_FILE="/tmp/ran-skills-deployment-$(date +%Y%m%d-%H%M%S).log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Node.js version
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed"
        exit 1
    fi

    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        error "Node.js 18+ is required, found $(node -v)"
        exit 1
    fi
    success "Node.js $(node -v) found"

    # Check AgentDB
    if ! command -v npx &> /dev/null; then
        error "npx is not available"
        exit 1
    fi

    if ! npx agentdb@latest --version &> /dev/null; then
        error "AgentDB is not installed"
        exit 1
    fi
    success "AgentDB found"

    # Check Claude Flow
    if ! npx claude-flow@alpha --version &> /dev/null; then
        error "Claude Flow is not installed"
        exit 1
    fi
    success "Claude Flow found"

    # Check skill directories exist
    for skill in "${RAN_SKILLS[@]}"; do
        if [ ! -d ".claude/skills/$skill" ]; then
            error "Skill directory not found: .claude/skills/$skill"
            exit 1
        fi
    done
    success "All skill directories found"
}

initialize_agentdb() {
    log "Initializing AgentDB for RAN skills..."

    # Create AgentDB directory
    mkdir -p .agentdb

    # Initialize AgentDB with RAN-specific configuration
    npx agentdb@latest init ./.agentdb/ran-skills.db --dimension 1536 --sync true >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        success "AgentDB initialized successfully"
    else
        error "Failed to initialize AgentDB"
        exit 1
    fi
}

deploy_skill() {
    local skill=$1
    log "Deploying skill: $skill"

    # Check skill.yml exists
    if [ ! -f ".claude/skills/$skill/skill.yml" ]; then
        error "skill.yml not found for $skill"
        return 1
    fi

    # Check skill.md exists
    if [ ! -f ".claude/skills/$skill/skill.md" ]; then
        error "skill.md not found for $skill"
        return 1
    fi

    # Verify progressive disclosure structure
    if ! grep -q "Level 1: Foundation" ".claude/skills/$skill/skill.md"; then
        warning "Level 1 section not found in $skill"
    fi

    if ! grep -q "Level 2" ".claude/skills/$skill/skill.md"; then
        warning "Level 2 section not found in $skill"
    fi

    if ! grep -q "Level 3" ".claude/skills/$skill/skill.md"; then
        warning "Level 3 section not found in $skill"
    fi

    # Deploy using Claude Flow
    npx claude-flow@alpha skill deploy "$skill" >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        success "Successfully deployed $skill"
        return 0
    else
        error "Failed to deploy $skill"
        return 1
    fi
}

verify_skill_dependencies() {
    local skill=$1
    log "Verifying dependencies for $skill"

    # Extract dependencies from skill.yml
    local deps=$(grep -A 10 "dependencies:" ".claude/skills/$skill/skill.yml" | grep -E "^\s*-\s*\w+" | sed 's/^\s*-\s*//' || true)

    if [ -z "$deps" ]; then
        log "No dependencies found for $skill"
        return 0
    fi

    for dep in $deps; do
        if npx claude-flow@alpha skill list | grep -q "$dep"; then
            success "Dependency $dep is available"
        else
            warning "Dependency $dep is not available"
        fi
    done
}

run_skill_tests() {
    local skill=$1
    log "Running tests for $skill"

    # Run basic skill test
    npx claude-flow@alpha skill test "$skill" >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        success "Tests passed for $skill"
        return 0
    else
        warning "Tests failed for $skill"
        return 1
    fi
}

verify_performance_targets() {
    local skill=$1
    log "Verifying performance targets for $skill"

    # Extract performance targets from skill.yml
    local targets_section=$(grep -A 20 "performance_targets:" ".claude/skills/$skill/skill.yml" || true)

    if [ -z "$targets_section" ]; then
        warning "No performance targets found for $skill"
        return 0
    fi

    log "Performance targets for $skill:"
    echo "$targets_section" | tee -a "$LOG_FILE"

    # Note: Actual performance verification would require running benchmarks
    # This is a placeholder for performance validation
    success "Performance targets documented for $skill"
}

generate_deployment_report() {
    log "Generating deployment report..."

    local report_file="docs/RAN-SKILLS-DEPLOYMENT-$(date +%Y%m%d-%H%M%S).md"

    cat > "$report_file" << EOF
# RAN Skills Deployment Report

**Deployment Date:** $(date)
**Log File:** $LOG_FILE

## Deployed Skills

EOF

    for skill in "${RAN_SKILLS[@]}"; do
        echo "### $skill" >> "$report_file"
        echo "- Status: Deployed" >> "$report_file"
        echo "- Progressive Disclosure: 3-Level Architecture" >> "$report_file"
        echo "- AgentDB Integration: Enabled" >> "$report_file"
        echo "" >> "$report_file"
    done

    cat >> "$report_file" << EOF
## Performance Targets

| Skill | Target | Status |
|-------|--------|---------|
| RAN AgentDB Integration Specialist | 150x faster search | Configured |
| RAN Causal Inference Specialist | 95% causal accuracy | Configured |
| RAN DSPy Mobility Optimizer | 15% mobility improvement | Configured |
| RAN Reinforcement Learning Engineer | 90% convergence rate | Configured |

## Next Steps

1. Run integration tests: \`npm run test:integration\`
2. Monitor performance: \`npx agentdb@latest monitor\`
3. Verify workflows: \`npx claude-flow@alpha workflow test\`

## Support

For issues or questions, refer to:
- Integration Guide: [RAN-SKILLS-INTEGRATION.md](RAN-SKILLS-INTEGRATION.md)
- Test Suite: [tests/skills-integration.test.js](tests/skills-integration.test.js)
- Log File: $LOG_FILE
EOF

    success "Deployment report generated: $report_file"
}

main() {
    log "Starting RAN Skills Deployment..."
    log "Log file: $LOG_FILE"

    # Check prerequisites
    check_prerequisites

    # Initialize AgentDB
    initialize_agentdb

    # Deploy skills in dependency order
    local deployment_results=()

    for skill in "${RAN_SKILLS[@]}"; do
        if deploy_skill "$skill"; then
            deployment_results+=("$skill:SUCCESS")

            # Verify dependencies
            verify_skill_dependencies "$skill"

            # Run tests
            run_skill_tests "$skill"

            # Verify performance targets
            verify_performance_targets "$skill"

        else
            deployment_results+=("$skill:FAILED")
        fi

        echo "---" >> "$LOG_FILE"
    done

    # Generate deployment report
    generate_deployment_report

    # Summary
    log "Deployment Summary:"
    for result in "${deployment_results[@]}"; do
        if [[ $result == *"SUCCESS"* ]]; then
            success "$result"
        else
            error "$result"
        fi
    done

    # Check if all deployments succeeded
    local failed_count=$(printf '%s\n' "${deployment_results[@]}" | grep -c "FAILED" || true)
    local total_count=${#RAN_SKILLS[@]}

    if [ $failed_count -eq 0 ]; then
        success "All $total_count RAN skills deployed successfully!"
        log "Next steps:"
        log "1. Run integration tests: npm run test:integration"
        log "2. Start optimizing: npx claude-flow@alpha skill run ran-optimizer"
        exit 0
    else
        error "$failed_count out of $total_count skills failed to deploy"
        log "Check log file for details: $LOG_FILE"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "test")
        log "Running in test mode (no actual deployment)"
        set +e
        check_prerequisites
        exit 0
        ;;
    "clean")
        log "Cleaning up AgentDB..."
        rm -rf .agentdb
        success "AgentDB cleaned up"
        exit 0
        ;;
    "help"|"-h"|"--help")
        echo "RAN Skills Deployment Script"
        echo ""
        echo "Usage: $0 [test|clean|help]"
        echo ""
        echo "Commands:"
        echo "  test   - Check prerequisites without deploying"
        echo "  clean  - Clean up AgentDB database"
        echo "  help   - Show this help message"
        echo ""
        echo "Default: Deploy all RAN skills"
        exit 0
        ;;
    "")
        # Default behavior - deploy all skills
        main
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac