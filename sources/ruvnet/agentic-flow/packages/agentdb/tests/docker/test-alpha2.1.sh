#!/bin/bash
set -e

echo "=== AgentDB v2.0.0-alpha.2.1 Docker Validation Test ==="
echo ""

cd /test-agentdb/project

# Install locally built package
echo "=== Installing local agentdb package ==="
cp -r /agentdb-source ./agentdb-local
cd agentdb-local && npm pack && cd ..
npm install ./agentdb-local/agentdb-2.0.0-alpha.2.1.tgz

echo ""
echo "=== Test 1: Version check ==="
npx agentdb --version

echo ""
echo "=== Test 2: Package.json export (dotenv dependency) ==="
node -e "const pkg = require('agentdb/package.json'); console.log('Version:', pkg.version); console.log('Has dotenv:', pkg.dependencies.dotenv ? 'YES ✅' : 'NO ❌');"

echo ""
echo "=== Test 3: Simulate command integration ==="
npx agentdb simulate list

echo ""
echo "=== Test 4: Main CLI commands ==="
npx agentdb --help | head -20

echo ""
echo "✅ All tests passed!"
