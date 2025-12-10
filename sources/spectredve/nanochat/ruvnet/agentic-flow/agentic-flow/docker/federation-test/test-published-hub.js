import { FederationHubServer } from 'agentic-flow/dist/federation/FederationHubServer.js';
import express from 'express';

const PORT = parseInt(process.env.FEDERATION_HUB_PORT || '8443');
const DB_PATH = process.env.FEDERATION_DB_PATH || '/data/hub.db';

async function main() {
  console.log('\n🌐 Testing Published Package v1.8.13...');
  console.log('═'.repeat(60));
  console.log(`📍 Port: ${PORT}`);
  console.log(`💾 Database: ${DB_PATH}`);
  console.log('');

  try {
    const hub = new FederationHubServer({
      port: PORT,
      dbPath: DB_PATH,
      maxAgents: 100,
      syncInterval: 5000
    });

    await hub.start();

    const app = express();
    const healthPort = 8444;

    app.get('/health', (req, res) => {
      const stats = hub.getStats();
      res.json({ status: 'healthy', ...stats, timestamp: Date.now() });
    });

    app.get('/stats', (req, res) => {
      res.json(hub.getStats());
    });

    app.listen(healthPort, () => {
      console.log(`✅ Health check server running on port ${healthPort}`);
    });

    console.log('✅ Published package working correctly!');
    console.log('═'.repeat(60));

    setInterval(() => {
      const stats = hub.getStats();
      console.log(`[Hub] Agents: ${stats.connectedAgents}, Episodes: ${stats.totalEpisodes}`);
    }, 10000);

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
  } catch (error) {
    console.error('❌ Test failed:', error);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
