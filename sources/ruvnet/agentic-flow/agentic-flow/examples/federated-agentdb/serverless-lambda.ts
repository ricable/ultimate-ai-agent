/**
 * Serverless Example: AWS Lambda + Federated AgentDB
 *
 * Use case: Short-lived Lambda functions (5-15 minutes) that need to:
 * - Access shared memory across invocations
 * - Learn from past executions
 * - Coordinate with other Lambda functions
 */

import { EphemeralAgent } from '../../src/federation/index.js';

// AWS Lambda handler
export async function handler(event: any) {
  const { userId, query, tenantId } = event;

  // Spawn ephemeral agent with 15-minute lifetime
  const agent = await EphemeralAgent.spawn({
    tenantId: tenantId || 'default-tenant',
    lifetime: 900, // 15 minutes (Lambda max)
    hubEndpoint: process.env.FEDERATION_HUB_ENDPOINT || 'quic://hub.agentdb.io:4433',
    syncInterval: 10000 // Sync every 10 seconds
  });

  try {
    // Execute task with access to federated memory
    const result = await agent.execute(async (db, context) => {
      console.log(`[Lambda] Agent ${context.agentId} processing query: ${query}`);

      // 1. Query past experiences from federated memory
      const memories = await agent.queryMemories(`user query: ${query}`, 5);

      if (memories.length > 0) {
        console.log(`[Lambda] Found ${memories.length} relevant memories from past executions`);
      }

      // 2. Process query using memories as context
      const response = await processUserQuery(query, memories);

      // 3. Store this execution as a learning episode
      await agent.storeEpisode({
        task: `user-query-${userId}`,
        input: query,
        output: response,
        reward: response.confidence, // Use confidence as reward signal
        critique: response.reasoning
      });

      console.log(`[Lambda] Stored learning episode (reward: ${response.confidence})`);

      return {
        userId,
        query,
        response: response.answer,
        confidence: response.confidence,
        agentId: context.agentId,
        remainingLifetime: agent.getRemainingLifetime()
      };
    });

    // Agent auto-destroyed after Lambda completes
    // Memory persists in federation hub for future invocations
    return {
      statusCode: 200,
      body: JSON.stringify(result)
    };

  } catch (error: any) {
    console.error('[Lambda] Error:', error.message);

    // Ensure cleanup even on error
    await agent.destroy();

    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
}

/**
 * Process user query with memory context
 */
async function processUserQuery(
  query: string,
  memories: any[]
): Promise<{ answer: string; confidence: number; reasoning: string }> {
  // Extract patterns from past experiences
  const successfulPatterns = memories
    .filter(m => m.reward >= 0.7)
    .map(m => ({
      input: m.input,
      output: m.output,
      reward: m.reward
    }));

  // Simulate query processing (in production, call LLM with memory context)
  const answer = `Processed "${query}" using ${successfulPatterns.length} past experiences`;
  const confidence = successfulPatterns.length > 0 ? 0.85 : 0.6;
  const reasoning = successfulPatterns.length > 0
    ? `Applied patterns from ${successfulPatterns.length} successful past executions`
    : 'No relevant past experiences found, using baseline approach';

  return { answer, confidence, reasoning };
}

/**
 * Example Lambda invocation:
 *
 * aws lambda invoke \
 *   --function-name user-query-processor \
 *   --payload '{"userId": "user-123", "query": "What is the weather?", "tenantId": "acme-corp"}' \
 *   response.json
 *
 * Expected flow:
 * 1. Lambda starts (cold start or warm)
 * 2. Ephemeral agent spawns and connects to federation hub
 * 3. Agent syncs memory from hub (gets memories from past Lambda invocations)
 * 4. Agent processes query using memory context
 * 5. Agent stores learning episode
 * 6. Agent syncs memory to hub (persists learning)
 * 7. Agent auto-destroyed when Lambda completes
 * 8. Memory persists in hub for next invocation
 *
 * Benefits:
 * - Each Lambda invocation learns from all past invocations
 * - Memory persists across cold starts
 * - Multi-tenant isolation (different customers use different memory spaces)
 * - Low latency (<50ms sync overhead)
 */
