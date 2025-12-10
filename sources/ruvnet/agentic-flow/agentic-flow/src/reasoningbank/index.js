/**
 * ReasoningBank Plugin for Agentic-Flow
 *
 * Main entry point and public API
 *
 * Paper: https://arxiv.org/html/2509.25140v1
 */
// Core algorithms
export { retrieveMemories, formatMemoriesForPrompt } from './core/retrieve.js';
export { judgeTrajectory } from './core/judge.js';
export { distillMemories } from './core/distill.js';
export { consolidate, shouldConsolidate } from './core/consolidate.js';
export { mattsParallel, mattsSequential } from './core/matts.js';
// Utilities
export { computeEmbedding, clearEmbeddingCache } from './utils/embeddings.js';
export { mmrSelection, cosineSimilarity } from './utils/mmr.js';
export { scrubPII, containsPII, scrubMemory } from './utils/pii-scrubber.js';
export { loadConfig } from './utils/config.js';
// Import for internal use
import { loadConfig } from './utils/config.js';
import * as db from './db/queries.js';
import { retrieveMemories } from './core/retrieve.js';
import { judgeTrajectory } from './core/judge.js';
import { distillMemories } from './core/distill.js';
import { shouldConsolidate, consolidate } from './core/consolidate.js';
// Database
export { db };
/**
 * Initialize ReasoningBank
 * Run migrations and check configuration
 */
export async function initialize() {
    const config = loadConfig();
    console.log('[ReasoningBank] Initializing...');
    console.log('[ReasoningBank] Enabled: true (initializing...)');
    console.log(`[ReasoningBank] Database: ${process.env.CLAUDE_FLOW_DB_PATH || '.swarm/memory.db'}`);
    console.log(`[ReasoningBank] Embeddings: ${config.embeddings.provider}`);
    console.log(`[ReasoningBank] Retrieval k: ${config.retrieve.k}`);
    // Run migrations to create database and tables
    try {
        await db.runMigrations();
        console.log(`[ReasoningBank] Database migrated successfully`);
    }
    catch (error) {
        console.error('[ReasoningBank] Migration error:', error);
        throw new Error('ReasoningBank initialization failed: could not run migrations');
    }
    // Check database connection
    try {
        const dbConn = db.getDb();
        const tables = dbConn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pattern%'").all();
        console.log(`[ReasoningBank] Database OK: ${tables.length} tables found`);
    }
    catch (error) {
        console.error('[ReasoningBank] Database error:', error);
        throw new Error('ReasoningBank initialization failed: database not accessible');
    }
    console.log('[ReasoningBank] Initialization complete');
}
/**
 * Full task execution with ReasoningBank
 * Combines retrieve → execute → judge → distill → consolidate
 */
export async function runTask(options) {
    console.log(`[ReasoningBank] Running task: ${options.taskId}`);
    // 1. Retrieve memories
    const memories = await retrieveMemories(options.query, {
        domain: options.domain,
        agent: options.agentId
    });
    console.log(`[ReasoningBank] Retrieved ${memories.length} memories`);
    // 2. Execute task with memories
    const trajectory = await options.executeFn(memories);
    // 3. Judge trajectory
    const verdict = await judgeTrajectory(trajectory, options.query);
    console.log(`[ReasoningBank] Verdict: ${verdict.label} (${verdict.confidence})`);
    // 4. Distill new memories
    const newMemories = await distillMemories(trajectory, verdict, options.query, {
        taskId: options.taskId,
        agentId: options.agentId,
        domain: options.domain
    });
    console.log(`[ReasoningBank] Distilled ${newMemories.length} new memories`);
    // 5. Consolidate if needed
    let consolidated = false;
    if (shouldConsolidate()) {
        console.log('[ReasoningBank] Running consolidation...');
        await consolidate();
        consolidated = true;
    }
    return {
        verdict,
        usedMemories: memories,
        newMemories,
        consolidated
    };
}
// Version info
export const VERSION = '1.0.0';
export const PAPER_URL = 'https://arxiv.org/html/2509.25140v1';
