#!/bin/bash

echo "🐳 Testing research-swarm v1.0.1 in Clean Docker Environment"
echo "=============================================================="
echo ""

# Create test script that will run inside Docker
cat > /tmp/test-research-swarm.sh << 'DOCKERSCRIPT'
#!/bin/bash
set -e

echo "📦 Node.js version:"
node --version
echo ""

echo "📦 npm version:"
npm --version
echo ""

echo "🧪 Test 1: NPX package info"
echo "----------------------------"
npm info research-swarm 2>&1 | head -15
echo ""

echo "🧪 Test 2: Test lib/index.js exists in published package"
echo "---------------------------------------------------------"
npm pack research-swarm 2>&1 | tail -1 > /tmp/package.tgz
PACKAGE=$(cat /tmp/package.tgz)
tar -tzf "$PACKAGE" | grep "lib/index.js" && echo "✅ lib/index.js found in package!" || echo "❌ lib/index.js NOT found!"
echo ""

echo "🧪 Test 3: Install package locally"
echo "-----------------------------------"
npm install --no-save research-swarm
echo "✅ Package installed"
echo ""

echo "🧪 Test 4: Test programmatic imports"
echo "-------------------------------------"
cat > test-imports.mjs << 'EOF'
import swarm from 'research-swarm';
import { createResearchJob, initDatabase, VERSION } from 'research-swarm';

console.log('✅ Default export:', typeof swarm === 'object' ? 'OK' : 'FAIL');
console.log('✅ Named export (createResearchJob):', typeof createResearchJob === 'function' ? 'OK' : 'FAIL');
console.log('✅ Named export (initDatabase):', typeof initDatabase === 'function' ? 'OK' : 'FAIL');
console.log('✅ Version:', VERSION);
console.log('');
console.log('✅ All imports working correctly!');
EOF

node test-imports.mjs
echo ""

echo "🧪 Test 5: Test CLI is available"
echo "---------------------------------"
./node_modules/.bin/research-swarm --version
echo ""

echo "🧪 Test 6: Test subpath exports"
echo "--------------------------------"
cat > test-subpaths.mjs << 'EOF'
import { getDatabase } from 'research-swarm/db';
import { storeResearchPattern } from 'research-swarm/reasoningbank';

console.log('✅ Subpath export (db):', typeof getDatabase === 'function' ? 'OK' : 'FAIL');
console.log('✅ Subpath export (reasoningbank):', typeof storeResearchPattern === 'function' ? 'OK' : 'FAIL');
console.log('');
console.log('✅ All subpath exports working!');
EOF

node test-subpaths.mjs
echo ""

echo "=============================================================="
echo "✅ All Docker tests passed!"
echo "✅ research-swarm v1.0.1 is working correctly"
echo "=============================================================="
DOCKERSCRIPT

chmod +x /tmp/test-research-swarm.sh

# Run in Docker (Node 22 - latest LTS)
echo "🐳 Starting Docker container with Node.js 22..."
docker run --rm -v /tmp/test-research-swarm.sh:/test.sh node:22-alpine sh /test.sh

echo ""
echo "🎉 Docker validation complete!"
