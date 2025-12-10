/**
 * Standalone Federation Hub Server for Docker
 * Uses built npm package
 */

import { FederationHubServer } from '../../dist/federation/FederationHubServer.js';
import express from 'express';

const PORT = parseInt(process.env.FEDERATION_HUB_PORT || process.env.HUB_PORT || '8443');
const DB_PATH = process.env.FEDERATION_DB_PATH || process.env.HUB_DB_PATH || '/data/hub.db';
const MAX_AGENTS = parseInt(process.env.FEDERATION_MAX_AGENTS || '1000');

async function main() {
  console.log('\n🌐 Starting Federation Hub Server...');
  console.log('═'.repeat(60));
  console.log(`📍 Port: ${PORT}`);
  console.log(`💾 Database: ${DB_PATH}`);
  console.log(`👥 Max Agents: ${MAX_AGENTS}`);
  console.log('🔒 Protocol: WebSocket (QUIC support planned)');
  console.log('');

  try {
    // Create hub server
    const hub = new FederationHubServer({
      port: PORT,
      dbPath: DB_PATH,
      maxAgents: MAX_AGENTS,
      syncInterval: 5000
    });

    // Start hub
    await hub.start();

    // Create health check endpoint
    const app = express();
    const healthPort = 8444;

    app.get('/health', (req, res) => {
      const stats = hub.getStats();
      res.json({
        status: 'healthy',
        ...stats,
        timestamp: Date.now()
      });
    });

    app.get('/stats', (req, res) => {
      const stats = hub.getStats();
      res.json(stats);
    });

    app.listen(healthPort, () => {
      console.log(`✅ Health check server running on port ${healthPort}`);
    });

    console.log('✅ Federation Hub Server is ready!');
    console.log('═'.repeat(60));
    console.log('');

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      console.log('\n🛑 Shutting down...');
      await hub.stop();
      process.exit(0);
    });

    process.on('SIGINT', async () => {
      console.log('\n🛑 Shutting down...');
      await hub.stop();
      process.exit(0);
    });

    // Keep alive and log stats
    setInterval(() => {
      const stats = hub.getStats();
      console.log(`[Hub] Agents: ${stats.connectedAgents}, Episodes: ${stats.totalEpisodes}, Tenants: ${stats.tenants}`);
    }, 10000);

  } catch (error) {
    console.error('❌ Hub server failed:', error);
    process.exit(1);
  }
}

main();
