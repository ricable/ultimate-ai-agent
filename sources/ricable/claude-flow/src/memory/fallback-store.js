/**
 * Fallback memory store that tries SQLite first, then falls back to in-memory storage
 * Designed to handle npx environments where native modules may fail to load
 */

import { SqliteMemoryStore } from './sqlite-store.js';
import { InMemoryStore } from './in-memory-store.js';

class FallbackMemoryStore {
  constructor(options = {}) {
    this.options = options;
    this.primaryStore = null;
    this.fallbackStore = null;
    this.useFallback = false;
    this.initializationAttempted = false;
  }

  async initialize() {
    if (this.initializationAttempted) return;
    this.initializationAttempted = true;

    // First, try to initialize SQLite store
    try {
      this.primaryStore = new SqliteMemoryStore(this.options);
      await this.primaryStore.initialize();
      console.error(`[${new Date().toISOString()}] INFO [fallback-store] Successfully initialized SQLite store`);
      this.useFallback = false;
    } catch (error) {
      console.error(`[${new Date().toISOString()}] WARN [fallback-store] SQLite initialization failed, falling back to in-memory store:`, error.message);
      
      // Fall back to in-memory store
      this.fallbackStore = new InMemoryStore(this.options);
      await this.fallbackStore.initialize();
      this.useFallback = true;
      
      console.error(`[${new Date().toISOString()}] INFO [fallback-store] Using in-memory store (data will not persist across sessions)`);
      console.error(`[${new Date().toISOString()}] INFO [fallback-store] To enable persistent storage, install the package locally: npm install claude-flow@alpha`);
    }
  }

  get activeStore() {
    return this.useFallback ? this.fallbackStore : this.primaryStore;
  }

  async store(key, value, options = {}) {
    await this.initialize();
    return this.activeStore.store(key, value, options);
  }

  async retrieve(key, options = {}) {
    await this.initialize();
    return this.activeStore.retrieve(key, options);
  }

  async list(options = {}) {
    await this.initialize();
    return this.activeStore.list(options);
  }

  async delete(key, options = {}) {
    await this.initialize();
    return this.activeStore.delete(key, options);
  }

  async search(pattern, options = {}) {
    await this.initialize();
    return this.activeStore.search(pattern, options);
  }

  async cleanup() {
    await this.initialize();
    return this.activeStore.cleanup();
  }

  close() {
    if (this.primaryStore) {
      this.primaryStore.close();
    }
    if (this.fallbackStore) {
      this.fallbackStore.close();
    }
  }

  isUsingFallback() {
    return this.useFallback;
  }
}

// Export a singleton instance for MCP server
export const memoryStore = new FallbackMemoryStore();

export { FallbackMemoryStore };
export default FallbackMemoryStore;