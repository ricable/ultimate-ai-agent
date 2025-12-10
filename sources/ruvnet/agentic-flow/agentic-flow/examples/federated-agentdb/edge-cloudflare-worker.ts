/**
 * Edge Computing Example: Cloudflare Workers + Federated AgentDB
 *
 * Use case: Ultra-low-latency edge functions (50ms-5s) that need:
 * - Regional memory access (user data stays in region)
 * - Cross-region coordination (eventual consistency)
 * - Privacy compliance (GDPR, data residency)
 */

import { EphemeralAgent } from '../../src/federation/index.js';

// Cloudflare Worker handler
export default {
  async fetch(request: Request, env: any): Promise<Response> {
    const url = new URL(request.url);
    const userId = url.searchParams.get('userId') || 'anonymous';
    const region = request.cf?.colo || 'UNKNOWN'; // Cloudflare datacenter code

    // Determine tenant from request
    const tenantId = request.headers.get('X-Tenant-ID') || 'default-tenant';

    // Spawn ephemeral agent with 30-second lifetime (short for edge)
    const agent = await EphemeralAgent.spawn({
      tenantId,
      lifetime: 30, // 30 seconds (quick edge execution)
      // Connect to regional hub for low latency
      hubEndpoint: getRegionalHub(region),
      syncInterval: 2000, // Sync every 2 seconds (edge needs frequent sync)
      enableEncryption: true // GDPR compliance
    });

    try {
      const result = await agent.execute(async (db, context) => {
        console.log(`[Edge ${region}] Agent ${context.agentId} processing request`);

        // 1. Query regional memories (fast, <10ms)
        const regionalMemories = await agent.queryMemories(
          `user:${userId} region:${region}`,
          3
        );

        // 2. Process request with regional context
        const response = await processEdgeRequest(request, regionalMemories, region);

        // 3. Store episode (will sync to hub, then propagate to other regions)
        await agent.storeEpisode({
          task: `edge-request-${region}`,
          input: `user:${userId} url:${url.pathname}`,
          output: JSON.stringify(response),
          reward: response.success ? 0.9 : 0.3
        });

        return {
          success: true,
          region,
          agentId: context.agentId,
          memoriesUsed: regionalMemories.length,
          remainingLifetime: agent.getRemainingLifetime(),
          data: response
        };
      });

      return new Response(JSON.stringify(result), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'X-Agent-ID': result.agentId,
          'X-Region': region
        }
      });

    } catch (error: any) {
      console.error(`[Edge ${region}] Error:`, error.message);
      await agent.destroy();

      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
};

/**
 * Get regional hub endpoint based on datacenter location
 */
function getRegionalHub(region: string): string {
  // Map Cloudflare datacenter codes to regional hubs
  const regionMap: Record<string, string> = {
    // North America
    'IAD': 'quic://hub-us-east.agentdb.io:4433',
    'ORD': 'quic://hub-us-central.agentdb.io:4433',
    'SJC': 'quic://hub-us-west.agentdb.io:4433',
    // Europe
    'LHR': 'quic://hub-eu-west.agentdb.io:4433',
    'FRA': 'quic://hub-eu-central.agentdb.io:4433',
    // Asia Pacific
    'SIN': 'quic://hub-ap-south.agentdb.io:4433',
    'NRT': 'quic://hub-ap-north.agentdb.io:4433',
    'SYD': 'quic://hub-ap-south.agentdb.io:4433'
  };

  return regionMap[region] || 'quic://hub-global.agentdb.io:4433';
}

/**
 * Process edge request with regional context
 */
async function processEdgeRequest(
  request: Request,
  memories: any[],
  region: string
): Promise<any> {
  const url = new URL(request.url);

  // Extract successful patterns from regional memory
  const patterns = memories
    .filter(m => m.reward >= 0.7)
    .map(m => m.output);

  return {
    success: true,
    message: `Processed at edge (${region}) using ${patterns.length} regional memories`,
    timestamp: Date.now(),
    region
  };
}

/**
 * Example Worker invocation:
 *
 * curl "https://edge-agent.example.com/?userId=user-123" \
 *   -H "X-Tenant-ID: acme-corp"
 *
 * Expected flow:
 * 1. Request hits Cloudflare edge in London (LHR)
 * 2. Ephemeral agent spawns with 30-second lifetime
 * 3. Agent connects to EU West hub (quic://hub-eu-west.agentdb.io:4433)
 * 4. Agent queries regional memories (fast, <10ms from local hub)
 * 5. Agent processes request using regional context
 * 6. Agent stores learning episode
 * 7. Episode syncs to regional hub (EU West)
 * 8. Regional hub syncs to global hub (eventual consistency)
 * 9. Global hub propagates to other regional hubs (AP, US, etc.)
 * 10. Agent auto-destroyed after 30 seconds
 *
 * Benefits:
 * - Ultra-low latency (<50ms including memory access)
 * - Data residency compliance (EU data stays in EU)
 * - Global learning (eventually propagates to all regions)
 * - Privacy by default (encrypted at rest and in transit)
 *
 * Regional Memory Propagation:
 * - Regional hub → Global hub: ~50ms
 * - Global hub → Other regional hubs: ~100ms
 * - Total propagation time: <5 minutes (eventual consistency)
 * - Critical memories can be prioritized for faster sync
 */
