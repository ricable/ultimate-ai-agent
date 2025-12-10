#!/usr/bin/env node
/**
 * Query Cache Layer
 * Caches query results for faster repeated queries
 */

const { execSync } = require('child_process');
const crypto = require('crypto');

class QueryCache {
  constructor(ttl = 300000) { // 5 minutes default
    this.cache = new Map();
    this.ttl = ttl;
  }

  getCacheKey(query, namespace) {
    return crypto
      .createHash('md5')
      .update(`${query}:${namespace}`)
      .digest('hex');
  }

  get(query, namespace) {
    const key = this.getCacheKey(query, namespace);
    const cached = this.cache.get(key);

    if (!cached) return null;

    if (Date.now() - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.result;
  }

  set(query, namespace, result) {
    const key = this.getCacheKey(query, namespace);
    this.cache.set(key, {
      result,
      timestamp: Date.now()
    });
  }

  clear() {
    this.cache.clear();
  }

  stats() {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys())
    };
  }
}

// Cached query function
function cachedQuery(query, namespace = 'default') {
  const cache = global.queryCache || (global.queryCache = new QueryCache());

  // Check cache first
  const cached = cache.get(query, namespace);
  if (cached) {
    console.log('🔥 Cache hit!');
    return cached;
  }

  // Cache miss - execute query
  console.log('💫 Cache miss - fetching...');
  const result = execSync(
    `npx claude-flow@alpha memory query "${query}" --namespace ${namespace} --reasoningbank`,
    { encoding: 'utf8' }
  );

  cache.set(query, namespace, result);
  return result;
}

// CLI usage
if (require.main === module) {
  const query = process.argv[2];
  const namespace = process.argv[3] || 'default';

  if (!query) {
    console.log('Usage: node cached-query.js "your query" [namespace]');
    process.exit(1);
  }

  const result = cachedQuery(query, namespace);
  console.log(result);
}

module.exports = { QueryCache, cachedQuery };
