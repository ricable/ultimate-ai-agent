#!/bin/bash

# Simplified Docker Multi-Agent Collaboration Test
# Runs federation test without full TypeScript build

set -e

echo "🚀 Federation Multi-Agent Collaboration Test"
echo "=============================================="
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Create test directory structure
mkdir -p /tmp/federation-test
cd /tmp/federation-test

# Create simple hub server (Node.js standalone)
cat > hub.js << 'EOF'
const http = require('http');
const WebSocketServer = require('ws').WebSocketServer;
const Database = require('better-sqlite3');

const db = new Database(':memory:');
db.exec(`
  CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task TEXT NOT NULL,
    reward REAL NOT NULL,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX idx_tenant ON episodes(tenant_id);
`);

const server = http.createServer((req, res) => {
  if (req.url === '/stats') {
    const episodes = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
    const tenants = db.prepare('SELECT COUNT(DISTINCT tenant_id) as count FROM episodes').get();
    res.writeHead(200, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({
      connectedAgents: wss.clients.size,
      totalEpisodes: episodes.count,
      tenants: tenants.count
    }));
  } else {
    res.writeHead(200);
    res.end('Federation Hub');
  }
});

const wss = new WebSocketServer({ server });
console.log('✅ Hub server started on port 8443');

wss.on('connection', (ws) => {
  let agentId, tenantId;

  ws.on('message', (data) => {
    const msg = JSON.parse(data.toString());

    if (msg.type === 'auth') {
      agentId = msg.agentId;
      tenantId = msg.tenantId;
      console.log(`[Hub] Agent connected: ${agentId} (tenant: ${tenantId})`);
      ws.send(JSON.stringify({ type: 'ack', timestamp: Date.now() }));
    } else if (msg.type === 'push' && msg.data) {
      for (const ep of msg.data) {
        db.prepare('INSERT INTO episodes (tenant_id, agent_id, task, reward, created_at) VALUES (?, ?, ?, ?, ?)')
          .run(tenantId, agentId, ep.task, ep.reward, Date.now());
      }
      console.log(`[Hub] Stored ${msg.data.length} episodes from ${agentId}`);
      ws.send(JSON.stringify({ type: 'ack', timestamp: Date.now() }));
    } else if (msg.type === 'pull') {
      const episodes = db.prepare('SELECT * FROM episodes WHERE tenant_id = ? ORDER BY created_at DESC LIMIT 10')
        .all(tenantId);
      ws.send(JSON.stringify({ type: 'ack', data: episodes, timestamp: Date.now() }));
    }
  });

  ws.on('close', () => {
    console.log(`[Hub] Agent disconnected: ${agentId}`);
  });
});

server.listen(8443);
EOF

# Create agent script
cat > agent.js << 'EOF'
const WebSocket = require('ws');

const AGENT_TYPE = process.env.AGENT_TYPE || 'researcher';
const AGENT_ID = process.env.AGENT_ID || `${AGENT_TYPE}-${Date.now()}`;
const TENANT_ID = process.env.TENANT_ID || 'test-collaboration';

async function main() {
  console.log(`🤖 Agent ${AGENT_ID} starting...`);

  const ws = new WebSocket('ws://hub:8443');

  ws.on('open', () => {
    console.log(`✅ Connected to hub`);

    // Authenticate
    ws.send(JSON.stringify({
      type: 'auth',
      agentId: AGENT_ID,
      tenantId: TENANT_ID,
      timestamp: Date.now()
    }));

    // Simulate work for 30 seconds
    let iteration = 0;
    const interval = setInterval(() => {
      iteration++;

      // Create episode
      const episode = {
        task: `${AGENT_TYPE}-task-${iteration}`,
        input: `Input ${iteration}`,
        output: `Output ${iteration}`,
        reward: 0.75 + Math.random() * 0.2,
        success: true
      };

      // Push to hub
      ws.send(JSON.stringify({
        type: 'push',
        data: [episode],
        timestamp: Date.now()
      }));

      console.log(`[${AGENT_TYPE}] Iteration ${iteration} (reward: ${episode.reward.toFixed(2)})`);

      if (iteration >= 5) {
        clearInterval(interval);
        setTimeout(() => ws.close(), 1000);
      }
    }, 5000);
  });

  ws.on('close', () => {
    console.log('✅ Agent disconnected');
    process.exit(0);
  });
}

main().catch(console.error);
EOF

# Create docker-compose
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  hub:
    image: node:20-slim
    working_dir: /app
    command: sh -c "npm install ws better-sqlite3 && node hub.js"
    volumes:
      - ./hub.js:/app/hub.js
    ports:
      - "8443:8443"
    networks:
      - test-net

  researcher:
    image: node:20-slim
    working_dir: /app
    command: sh -c "sleep 3 && npm install ws && node agent.js"
    volumes:
      - ./agent.js:/app/agent.js
    environment:
      - AGENT_TYPE=researcher
      - AGENT_ID=researcher-001
      - TENANT_ID=test-collaboration
    depends_on:
      - hub
    networks:
      - test-net

  coder:
    image: node:20-slim
    working_dir: /app
    command: sh -c "sleep 5 && npm install ws && node agent.js"
    volumes:
      - ./agent.js:/app/agent.js
    environment:
      - AGENT_TYPE=coder
      - AGENT_ID=coder-001
      - TENANT_ID=test-collaboration
    depends_on:
      - hub
    networks:
      - test-net

  tester:
    image: node:20-slim
    working_dir: /app
    command: sh -c "sleep 7 && npm install ws && node agent.js"
    volumes:
      - ./agent.js:/app/agent.js
    environment:
      - AGENT_TYPE=tester
      - AGENT_ID=tester-001
      - TENANT_ID=test-collaboration
    depends_on:
      - hub
    networks:
      - test-net

  isolated:
    image: node:20-slim
    working_dir: /app
    command: sh -c "sleep 9 && npm install ws && node agent.js"
    volumes:
      - ./agent.js:/app/agent.js
    environment:
      - AGENT_TYPE=researcher
      - AGENT_ID=isolated-001
      - TENANT_ID=different-tenant
    depends_on:
      - hub
    networks:
      - test-net

networks:
  test-net:
    driver: bridge
EOF

echo "📦 Starting federation test..."
echo ""
docker-compose up --abort-on-container-exit

echo ""
echo "🔍 Checking results..."
sleep 2

# Query hub stats
STATS=$(curl -s http://localhost:8443/stats 2>/dev/null || echo '{}')
echo "Hub Stats: $STATS"

# Cleanup
docker-compose down -v

echo ""
echo "✅ Test complete!"
EOF

chmod +x /tmp/federation-test/test.sh
cd /tmp/federation-test
bash -c "docker-compose up --abort-on-container-exit; docker-compose down -v" 2>&1 | tail -100
