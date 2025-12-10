#!/bin/bash
set -e

VCS_TYPE=${1:-git}
TEST_SCENARIO=${TEST_SCENARIO:-concurrent}
AGENT_COUNT=${AGENT_COUNT:-10}
COMMIT_COUNT=${COMMIT_COUNT:-100}
CONFLICT_RATE=${CONFLICT_RATE:-10}
BENCHMARK_DURATION=${BENCHMARK_DURATION:-600}

RESULTS_DIR="/results/${VCS_TYPE}/${TEST_SCENARIO}"
mkdir -p "${RESULTS_DIR}"

echo "========================================="
echo "  Agentic-Jujutsu Benchmark Suite"
echo "========================================="
echo "VCS Type: ${VCS_TYPE}"
echo "Scenario: ${TEST_SCENARIO}"
echo "Agents: ${AGENT_COUNT}"
echo "Commits per agent: ${COMMIT_COUNT}"
echo "Conflict rate: ${CONFLICT_RATE}%"
echo "Max duration: ${BENCHMARK_DURATION}s"
echo "========================================="
echo ""

# Set up repository
bash /benchmark/scripts/setup-repos.sh "${VCS_TYPE}" "/repos/test-repo"

# Run appropriate benchmark scenario
case "${TEST_SCENARIO}" in
    concurrent)
        echo "Running Concurrent Commit Storm benchmark..."
        bash /benchmark/tests/concurrent-commits.sh "${VCS_TYPE}"
        ;;

    workspace-isolation)
        echo "Running Workspace Isolation benchmark..."
        bash /benchmark/tests/workspace-isolation.sh "${VCS_TYPE}"
        ;;

    conflict-resolution)
        echo "Running Conflict Resolution benchmark..."
        bash /benchmark/tests/conflict-resolution.sh "${VCS_TYPE}"
        ;;

    operation-log-sync)
        echo "Running Operation Log Synchronization benchmark..."
        bash /benchmark/tests/operation-log-sync.sh "${VCS_TYPE}"
        ;;

    real-world)
        echo "Running Real-World Development Workflow benchmark..."
        bash /benchmark/tests/real-world-workflow.sh "${VCS_TYPE}"
        ;;

    *)
        echo "Unknown scenario: ${TEST_SCENARIO}"
        exit 1
        ;;
esac

# Collect metrics
echo ""
echo "Collecting metrics..."
bash /benchmark/scripts/collect-metrics.sh "${VCS_TYPE}" "${TEST_SCENARIO}"

# Generate report
echo ""
echo "Generating report..."
python3 /benchmark/scripts/generate-report.py \
    --vcs "${VCS_TYPE}" \
    --scenario "${TEST_SCENARIO}" \
    --results-dir "${RESULTS_DIR}"

echo ""
echo "========================================="
echo "  Benchmark Complete!"
echo "========================================="
echo "Results saved to: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"
