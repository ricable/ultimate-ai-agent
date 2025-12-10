/**
 * Embedding generation for semantic similarity
 * Supports multiple providers: OpenAI, Claude (placeholder), local hashing
 */
import { loadConfig } from './config.js';
const embeddingCache = new Map();
/**
 * Compute embedding for text
 * Uses configured provider (openai, claude, or local)
 */
export async function computeEmbedding(text) {
    const config = loadConfig();
    // Check cache
    const cacheKey = `${config.embeddings.provider}:${text}`;
    if (embeddingCache.has(cacheKey)) {
        return embeddingCache.get(cacheKey);
    }
    let embedding;
    if (config.embeddings.provider === 'openai') {
        embedding = await openaiEmbed(text, config.embeddings.model);
    }
    else if (config.embeddings.provider === 'claude') {
        // Claude doesn't have native embeddings yet, use deterministic hash
        embedding = hashEmbed(text, config.embeddings.dimensions || 1024);
    }
    else {
        // Fallback to local hashing
        embedding = hashEmbed(text, config.embeddings.dimensions || 1024);
    }
    // Cache with TTL
    embeddingCache.set(cacheKey, embedding);
    setTimeout(() => embeddingCache.delete(cacheKey), config.embeddings.cache_ttl_seconds * 1000);
    return embedding;
}
/**
 * OpenAI embeddings API
 */
async function openaiEmbed(text, model) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
        console.warn('[WARN] OPENAI_API_KEY not set, falling back to hash embeddings');
        return hashEmbed(text, 1536); // OpenAI default dimension
    }
    try {
        const response = await fetch('https://api.openai.com/v1/embeddings', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: model || 'text-embedding-3-small',
                input: text
            })
        });
        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }
        const json = await response.json();
        return new Float32Array(json.data[0].embedding);
    }
    catch (error) {
        console.error('[ERROR] OpenAI embedding failed:', error);
        console.warn('[WARN] Falling back to hash embeddings');
        return hashEmbed(text, 1536);
    }
}
/**
 * Deterministic hash-based embedding
 * For testing and when API keys are unavailable
 */
function hashEmbed(text, dims) {
    const hash = simpleHash(text);
    const vec = new Float32Array(dims);
    // Generate deterministic pseudo-random vector from hash
    for (let i = 0; i < dims; i++) {
        vec[i] = Math.sin(hash * (i + 1) * 0.01) * 0.1 + Math.cos(hash * i * 0.02) * 0.05;
    }
    return normalize(vec);
}
/**
 * Simple string hash function
 */
function simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0; // Convert to 32-bit integer
    }
    return Math.abs(hash);
}
/**
 * Normalize vector to unit length
 */
function normalize(vec) {
    let mag = 0;
    for (let i = 0; i < vec.length; i++) {
        mag += vec[i] * vec[i];
    }
    mag = Math.sqrt(mag);
    if (mag === 0)
        return vec;
    for (let i = 0; i < vec.length; i++) {
        vec[i] /= mag;
    }
    return vec;
}
/**
 * Clear embedding cache (for testing)
 */
export function clearEmbeddingCache() {
    embeddingCache.clear();
}
