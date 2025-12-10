#!/bin/bash
# Test database persistence

set -e

CLI="node /workspaces/agentic-flow/packages/agentdb/dist/cli/agentdb-cli.js"
TEST_DB="/tmp/test-persist-final.db"

rm -f "$TEST_DB"

echo "1. Init database"
export AGENTDB_PATH="$TEST_DB"
$CLI init

echo ""
echo "2. Store episode"
$CLI reflexion store "sess1" "task1" 0.9 true "critique" 2>&1 | tail -5

echo ""
echo "3. Check persistence"
node -e "
const Database = require('sql.js');
const fs = require('fs');
(async () => {
  const SQL = await Database();
  const buf = fs.readFileSync('$TEST_DB');
  const db = new SQL.Database(buf);
  const count = db.exec('SELECT COUNT(*) as count FROM episodes');
  console.log('Episodes in database:', count[0].values[0][0]);
  if (count[0].values[0][0] > 0) {
    console.log('✅ PERSISTENCE WORKING!');
    const episodes = db.exec('SELECT id, task, reward FROM episodes');
    console.log('Episodes:', JSON.stringify(episodes[0].values));
  } else {
    console.log('❌ PERSISTENCE FAILED - No episodes found');
  }
  db.close();
})();
"
