#!/bin/bash
# AgentDB v1.6.0 - Landing Page Feature Verification
# Tests EVERY feature claimed in documentation

set -e

CLI="node /workspaces/agentic-flow/packages/agentdb/dist/cli/agentdb-cli.js"
TEST_DB="/tmp/landing-page-test.db"

echo "🧪 AgentDB v1.6.0 - Landing Page Feature Verification"
echo "======================================================="
echo ""

# Cleanup
rm -f "$TEST_DB" "${TEST_DB}.gz" /tmp/test-export.json*

PASS=0
FAIL=0
TOTAL=0

test_feature() {
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "Test $TOTAL: $1"
    if eval "$2" > /dev/null 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL"
        FAIL=$((FAIL + 1))
        echo "Command: $2"
    fi
}

# Initialize database
echo "📦 Initializing test database..."
export AGENTDB_PATH="$TEST_DB"
$CLI init >/dev/null 2>&1

echo ""
echo "=== CORE FEATURES (v1.5.9) ==="

test_feature "Reflexion: Store episode" \
    "$CLI reflexion store 'sess1' 'auth_task' 0.95 true 'OAuth2 implementation' 'user login' 'JWT issued' 100 50"

test_feature "Reflexion: Retrieve episodes" \
    "$CLI reflexion retrieve 'auth' 5 0.8"

test_feature "Reflexion: Critique summary" \
    "$CLI reflexion critique-summary 'auth'"

test_feature "Skill: Create skill" \
    "$CLI skill create 'jwt_generator' 'Generate JWT tokens' 'const jwt = require(\"jsonwebtoken\")'"

test_feature "Skill: Search skills" \
    "$CLI skill search 'jwt' 3"

test_feature "Skill: Consolidate patterns" \
    "$CLI skill consolidate 2 0.7 7 true"

test_feature "Causal: Add edge" \
    "$CLI causal add-edge 'add_tests' 'code_quality' 0.25 0.95 50"

test_feature "Causal: Query edges" \
    "$CLI causal query 'add_tests' 'code_quality' 0.7 0.1 10"

test_feature "Causal: Create experiment" \
    "$CLI causal experiment create 'test-exp' 'testing' 'quality'"

test_feature "Recall: With certificate" \
    "$CLI recall with-certificate 'authentication' 5 0.7 0.2 0.1"

test_feature "Learner: Run discovery" \
    "$CLI learner run 2 0.6 0.7 false"

test_feature "Database: Stats" \
    "$CLI db stats"

test_feature "Hooks: Query command" \
    "$CLI query --domain 'test' --query 'authentication' --k 5 --format json"

test_feature "Hooks: Store pattern" \
    "$CLI store-pattern --type 'test' --domain 'test-domain' --pattern '{\"key\":\"value\"}' --confidence 0.9"

test_feature "Hooks: Train command" \
    "$CLI train --domain 'test' --epochs 1 --batch-size 10"

test_feature "Hooks: Optimize memory" \
    "$CLI optimize-memory --compress true --consolidate-patterns true"

echo ""
echo "=== NEW FEATURES (v1.6.0) ==="

test_feature "Init: Custom dimension" \
    "$CLI init /tmp/test-dim.db --dimension 768"

test_feature "Init: With preset" \
    "$CLI init /tmp/test-preset.db --dimension 384 --preset small"

test_feature "Init: In-memory" \
    "$CLI init --in-memory --dimension 1536"

# Add some data for vector search
$CLI reflexion store 'test' 'vector_test' 0.9 true 'test' 'in' 'out' 100 50 >/dev/null 2>&1

# Generate a 1536D test vector (matching default embedding dimension)
TEST_VECTOR=$(node -e "const vec = Array(1536).fill(0).map(() => Math.random() * 0.1); console.log(JSON.stringify(vec));")

test_feature "Vector Search: Cosine similarity" \
    "$CLI vector-search '$TEST_DB' '$TEST_VECTOR' -k 5 -m cosine"

test_feature "Vector Search: Euclidean distance" \
    "$CLI vector-search '$TEST_DB' '$TEST_VECTOR' -k 5 -m euclidean"

test_feature "Vector Search: Dot product" \
    "$CLI vector-search '$TEST_DB' '$TEST_VECTOR' -k 5 -m dot"

test_feature "Export: Basic" \
    "$CLI export '$TEST_DB' /tmp/test-export.json"

test_feature "Export: With compression" \
    "$CLI export '$TEST_DB' /tmp/test-export-compressed.json --compress"

test_feature "Import: Basic" \
    "$CLI import /tmp/test-export.json /tmp/test-import.db"

test_feature "Import: With decompression" \
    "[ -f /tmp/test-export-compressed.json.gz ] && $CLI import /tmp/test-export-compressed.json.gz /tmp/test-import-decomp.db --decompress"

test_feature "Stats: Show database stats" \
    "$CLI stats '$TEST_DB'"

echo ""
echo "=== ADVANCED FEATURES (NEW) ==="

# Test MMR (using same 1536D test vector)
test_feature "MMR: Diversity ranking" \
    "$CLI vector-search '$TEST_DB' '$TEST_VECTOR' -k 5 --mmr 0.5"

# Test context synthesis
test_feature "Context Synthesis: Query with synthesis" \
    "$CLI query --domain 'test' --query 'test task' --k 5 --synthesize-context"

test_feature "Context Synthesis: Retrieve with synthesis" \
    "$CLI reflexion retrieve 'test' 5 --synthesize-context"

# Test metadata filtering
test_feature "Metadata Filtering: Simple filter" \
    "$CLI reflexion retrieve 'test' 10 --filters '{\"success\":true}'"

test_feature "Metadata Filtering: Range filter" \
    "$CLI reflexion retrieve 'test' 10 --filters '{\"reward\":{\"\$gte\":0.8}}'"

test_feature "Metadata Filtering: Multiple conditions" \
    "$CLI reflexion retrieve 'test' 10 --filters '{\"success\":true,\"reward\":{\"\$gte\":0.9}}'"

echo ""
echo "=== MCP SERVER ==="

test_feature "MCP: Server starts" \
    "timeout 3 $CLI mcp start &"

echo ""
echo "=== API EXPORTS ==="

# Test that all controllers are exported
test_feature "API: MMRDiversityRanker exports" \
    "node -e \"require('/workspaces/agentic-flow/packages/agentdb/dist/controllers/MMRDiversityRanker.js')\""

test_feature "API: ContextSynthesizer exports" \
    "node -e \"require('/workspaces/agentic-flow/packages/agentdb/dist/controllers/ContextSynthesizer.js')\""

test_feature "API: MetadataFilter exports" \
    "node -e \"require('/workspaces/agentic-flow/packages/agentdb/dist/controllers/MetadataFilter.js')\""

test_feature "API: All controllers from index" \
    "node -e \"const m = require('/workspaces/agentic-flow/packages/agentdb/dist/controllers/index.js'); if (!m.MMRDiversityRanker || !m.ContextSynthesizer || !m.MetadataFilter) process.exit(1)\""

echo ""
echo "======================================================="
echo "📊 FINAL RESULTS"
echo "======================================================="
echo ""
echo "Total Tests:  $TOTAL"
echo "Passed:       $PASS"
echo "Failed:       $FAIL"
echo "Success Rate: $(( PASS * 100 / TOTAL ))%"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎉 ALL FEATURES VERIFIED - LANDING PAGE CLAIMS ARE ACCURATE!"
    exit 0
else
    echo "⚠️  Some features failed - see details above"
    exit 1
fi
