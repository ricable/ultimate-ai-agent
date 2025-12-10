/**
 * Run Federation Monitor Dashboard
 *
 * Provides real-time visualization of:
 * - Connected agents
 * - Memory sync events
 * - Tenant isolation
 * - Performance metrics
 */

import express from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';

const PORT = 3000;
const HUB_ENDPOINT = process.env.HUB_ENDPOINT || 'http://federation-hub:8443';

interface AgentStats {
  agentId: string;
  type: string;
  tenantId: string;
  connected: boolean;
  lastSync: number;
  episodeCount: number;
  avgReward: number;
}

class FederationMonitor {
  private app: express.Application;
  private server: http.Server;
  private wss: WebSocketServer;
  private clients: Set<WebSocket> = new Set();
  private agentStats: Map<string, AgentStats> = new Map();

  constructor() {
    this.app = express();
    this.server = http.createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });

    this.setupRoutes();
    this.setupWebSocket();
  }

  private setupRoutes(): void {
    // Serve dashboard HTML
    this.app.get('/', (req, res) => {
      res.send(this.getDashboardHTML());
    });

    // API endpoints
    this.app.get('/api/stats', async (req, res) => {
      try {
        const response = await fetch(`${HUB_ENDPOINT}/stats`);
        const stats = await response.json();
        res.json(stats);
      } catch (error: any) {
        res.status(500).json({ error: error.message });
      }
    });

    this.app.get('/api/agents', (req, res) => {
      const agents = Array.from(this.agentStats.values());
      res.json(agents);
    });
  }

  private setupWebSocket(): void {
    this.wss.on('connection', (ws: WebSocket) => {
      console.log('Dashboard client connected');
      this.clients.add(ws);

      ws.on('close', () => {
        this.clients.delete(ws);
        console.log('Dashboard client disconnected');
      });
    });
  }

  private broadcast(data: any): void {
    const message = JSON.stringify(data);
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
  }

  private getDashboardHTML(): string {
    return `
<!DOCTYPE html>
<html>
<head>
  <title>Federation Monitor</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      padding: 20px;
    }
    .header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 30px;
      border-radius: 10px;
      margin-bottom: 20px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1 { font-size: 2em; margin-bottom: 10px; }
    .subtitle { opacity: 0.9; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }
    .card {
      background: #1e293b;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .card h2 {
      font-size: 1.2em;
      margin-bottom: 15px;
      color: #60a5fa;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid #334155;
    }
    .metric:last-child { border-bottom: none; }
    .metric-label { opacity: 0.7; }
    .metric-value {
      font-weight: bold;
      font-size: 1.2em;
    }
    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
    }
    .status-dot.online { background: #10b981; }
    .status-dot.offline { background: #ef4444; }
    .agent-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .agent-item {
      background: #334155;
      padding: 15px;
      border-radius: 5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .agent-info { flex: 1; }
    .agent-type {
      display: inline-block;
      background: #3b82f6;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.8em;
      margin-left: 10px;
    }
    .tenant-badge {
      background: #8b5cf6;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.8em;
    }
    #log {
      background: #0f172a;
      padding: 15px;
      border-radius: 5px;
      font-family: 'Courier New', monospace;
      font-size: 0.9em;
      max-height: 300px;
      overflow-y: auto;
      border: 1px solid #334155;
    }
    .log-entry {
      padding: 5px 0;
      border-bottom: 1px solid #1e293b;
    }
    .log-time {
      color: #64748b;
      margin-right: 10px;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>🌐 Federation Monitor</h1>
    <div class="subtitle">Real-time agent collaboration dashboard</div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Hub Status</h2>
      <div class="metric">
        <span class="metric-label">Connected Agents</span>
        <span class="metric-value" id="agent-count">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Total Episodes</span>
        <span class="metric-value" id="episode-count">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Active Tenants</span>
        <span class="metric-value" id="tenant-count">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Uptime</span>
        <span class="metric-value" id="uptime">0s</span>
      </div>
    </div>

    <div class="card">
      <h2>Active Agents</h2>
      <div class="agent-list" id="agent-list">
        <div style="opacity: 0.5; text-align: center; padding: 20px;">
          No agents connected
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Activity Log</h2>
    <div id="log"></div>
  </div>

  <script>
    const ws = new WebSocket('ws://localhost:3000');
    const log = document.getElementById('log');

    function addLog(message) {
      const time = new Date().toLocaleTimeString();
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML = \`<span class="log-time">\${time}</span>\${message}\`;
      log.insertBefore(entry, log.firstChild);

      // Keep only last 50 entries
      while (log.children.length > 50) {
        log.removeChild(log.lastChild);
      }
    }

    async function updateStats() {
      try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        document.getElementById('agent-count').textContent = stats.connectedAgents || 0;
        document.getElementById('episode-count').textContent = stats.totalEpisodes || 0;
        document.getElementById('tenant-count').textContent = stats.tenants || 0;
        document.getElementById('uptime').textContent = Math.floor(stats.uptime || 0) + 's';
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      }
    }

    // Update every 2 seconds
    setInterval(updateStats, 2000);
    updateStats();

    // WebSocket messages
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      addLog(data.message || JSON.stringify(data));
    };

    ws.onopen = () => {
      addLog('✅ Connected to monitor');
    };

    ws.onclose = () => {
      addLog('❌ Disconnected from monitor');
    };

    addLog('🚀 Monitor started');
  </script>
</body>
</html>
    `;
  }

  start(): void {
    this.server.listen(PORT, () => {
      console.log(`📊 Federation Monitor running on http://localhost:${PORT}`);
    });

    // Poll hub stats
    setInterval(async () => {
      try {
        const response = await fetch(`${HUB_ENDPOINT}/stats`);
        const stats = await response.json();

        this.broadcast({
          type: 'stats',
          data: stats,
          timestamp: Date.now()
        });
      } catch (error) {
        // Hub not ready yet
      }
    }, 2000);
  }
}

const monitor = new FederationMonitor();
monitor.start();
