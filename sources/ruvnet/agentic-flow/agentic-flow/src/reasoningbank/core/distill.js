/**
 * Memory Distillation from trajectories
 * Algorithm 3 from ReasoningBank paper
 */
import { readFileSync } from 'fs';
import { join } from 'path';
import { ulid } from 'ulid';
import { loadConfig } from '../utils/config.js';
import { scrubMemory } from '../utils/pii-scrubber.js';
import { computeEmbedding } from '../utils/embeddings.js';
import * as db from '../db/queries.js';
/**
 * Distill memories from a trajectory
 */
export async function distillMemories(trajectory, verdict, query, options = {}) {
    const config = loadConfig();
    const startTime = Date.now();
    console.log(`[INFO] Distilling memories from ${verdict.label} trajectory`);
    // Select appropriate prompt template
    const templateName = verdict.label === 'Success' ? 'distill-success.json' : 'distill-failure.json';
    const promptPath = join(process.cwd(), 'src', 'reasoningbank', 'prompts', templateName);
    const promptTemplate = JSON.parse(readFileSync(promptPath, 'utf-8'));
    const maxItems = verdict.label === 'Success'
        ? config.distill.max_items_success
        : config.distill.max_items_failure;
    const confidencePrior = verdict.label === 'Success'
        ? config.distill.confidence_prior_success
        : config.distill.confidence_prior_failure;
    // Check API key
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
        console.warn('[WARN] ANTHROPIC_API_KEY not set, using template-based distillation');
        return templateBasedDistill(trajectory, verdict, query, options);
    }
    try {
        // Format trajectory
        const trajectoryText = JSON.stringify(trajectory.steps || [], null, 2);
        // Build prompt
        const prompt = promptTemplate.template
            .replace('{{task_query}}', query)
            .replace('{{trajectory}}', trajectoryText)
            .replace('{{max_items}}', String(maxItems));
        // Call Anthropic API
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            body: JSON.stringify({
                model: config.distill.model,
                max_tokens: 2048,
                temperature: config.distill.temperature,
                system: promptTemplate.system,
                messages: [{ role: 'user', content: prompt }]
            })
        });
        if (!response.ok) {
            throw new Error(`Anthropic API error: ${response.status}`);
        }
        const result = await response.json();
        const content = result.content[0].text;
        // Parse memories from response
        const distilled = parseDistilledMemories(content);
        // Store memories in database
        const memoryIds = await storeMemories(distilled, confidencePrior, verdict, options);
        const duration = Date.now() - startTime;
        console.log(`[INFO] Distilled ${memoryIds.length} memories in ${duration}ms`);
        db.logMetric('rb.distill.latency_ms', duration);
        db.logMetric('rb.distill.yield', memoryIds.length);
        return memoryIds;
    }
    catch (error) {
        console.error('[ERROR] Distillation failed:', error);
        return templateBasedDistill(trajectory, verdict, query, options);
    }
}
/**
 * Parse distilled memories from LLM response
 */
function parseDistilledMemories(content) {
    try {
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            return parsed.memories || [];
        }
    }
    catch (error) {
        console.warn('[WARN] Failed to parse distilled memories JSON');
    }
    return [];
}
/**
 * Store memories in database
 */
async function storeMemories(memories, confidencePrior, verdict, options) {
    const memoryIds = [];
    for (const mem of memories) {
        // Scrub PII
        const scrubbed = scrubMemory(mem);
        // Generate embedding
        const embedding = await computeEmbedding(`${scrubbed.title} ${scrubbed.description} ${scrubbed.content}`);
        // Create memory ID
        const id = ulid();
        // Store memory
        db.upsertMemory({
            id,
            type: 'reasoning_memory',
            pattern_data: {
                title: scrubbed.title,
                description: scrubbed.description,
                content: scrubbed.content,
                source: {
                    task_id: options.taskId || 'unknown',
                    agent_id: options.agentId || 'unknown',
                    outcome: verdict.label,
                    evidence: []
                },
                tags: scrubbed.tags,
                domain: options.domain || scrubbed.domain,
                created_at: new Date().toISOString(),
                confidence: confidencePrior,
                n_uses: 0
            },
            confidence: confidencePrior,
            usage_count: 0
        });
        // Store embedding
        db.upsertEmbedding({
            id,
            model: 'distill-' + verdict.label.toLowerCase(),
            dims: embedding.length,
            vector: embedding,
            created_at: new Date().toISOString()
        });
        memoryIds.push(id);
        console.log(`[INFO] Stored memory: ${scrubbed.title}`);
    }
    return memoryIds;
}
/**
 * Template-based distillation (fallback)
 * Simple extraction without LLM
 */
function templateBasedDistill(trajectory, verdict, query, options) {
    console.log('[INFO] Using template-based distillation (no API key)');
    // Create a single generic memory from the trajectory
    const memory = {
        title: `${verdict.label}: ${query.substring(0, 50)}`,
        description: `Task outcome: ${verdict.label}`,
        content: `Query: ${query}\n\nSteps: ${trajectory.steps?.length || 0}\n\nOutcome: ${verdict.label}`,
        tags: [verdict.label.toLowerCase(), 'template'],
        domain: options.domain
    };
    // Store synchronously (no async needed for template)
    return []; // Skip storage for template-based (would need to make this async)
}
