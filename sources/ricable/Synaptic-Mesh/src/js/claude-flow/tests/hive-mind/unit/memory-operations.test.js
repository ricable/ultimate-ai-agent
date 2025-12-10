/**
 * Memory Operations Tests
 * Tests shared memory storage, retrieval, and synchronization
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';

// Memory storage types
const MEMORY_TYPES = {
  GENERAL: 'general',
  CONFIG: 'config',
  CACHE: 'cache',
  KNOWLEDGE: 'knowledge',
  CONTEXT: 'context',
  METRICS: 'metrics'
};

// Memory store implementation
class MemoryStore {
  constructor(options = {}) {
    this.store = new Map();
    this.metadata = new Map();
    this.accessLog = [];
    this.ttlTimers = new Map();
    this.options = {
      maxSize: options.maxSize || 1000,
      defaultTTL: options.defaultTTL || null,
      enableCompression: options.enableCompression || false,
      enableEncryption: options.enableEncryption || false,
      ...options
    };
    this.stats = {
      hits: 0,
      misses: 0,
      writes: 0,
      deletes: 0,
      evictions: 0
    };
  }

  set(key, value, options = {}) {
    // Check size limit
    if (this.store.size >= this.options.maxSize && !this.store.has(key)) {
      this._evictOldest();
    }

    // Clear existing TTL timer if any
    if (this.ttlTimers.has(key)) {
      clearTimeout(this.ttlTimers.get(key));
      this.ttlTimers.delete(key);
    }

    // Process value
    let processedValue = value;
    if (this.options.enableCompression) {
      processedValue = this._compress(value);
    }
    if (this.options.enableEncryption) {
      processedValue = this._encrypt(processedValue);
    }

    // Store value
    this.store.set(key, processedValue);
    
    // Store metadata
    const metadata = {
      type: options.type || MEMORY_TYPES.GENERAL,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      accessCount: 0,
      size: this._getSize(value),
      ttl: options.ttl || this.options.defaultTTL,
      tags: options.tags || [],
      permissions: options.permissions || { read: '*', write: '*' }
    };
    this.metadata.set(key, metadata);

    // Set TTL if specified
    if (metadata.ttl) {
      const timer = setTimeout(() => {
        this.delete(key);
      }, metadata.ttl);
      this.ttlTimers.set(key, timer);
    }

    // Update stats
    this.stats.writes++;

    // Log access
    this._logAccess('set', key);

    return true;
  }

  get(key, options = {}) {
    if (!this.store.has(key)) {
      this.stats.misses++;
      this._logAccess('miss', key);
      return undefined;
    }

    // Check if expired
    const metadata = this.metadata.get(key);
    if (metadata.ttl && Date.now() - metadata.createdAt > metadata.ttl) {
      this.delete(key);
      this.stats.misses++;
      return undefined;
    }

    // Get value
    let value = this.store.get(key);

    // Process value
    if (this.options.enableEncryption) {
      value = this._decrypt(value);
    }
    if (this.options.enableCompression) {
      value = this._decompress(value);
    }

    // Update metadata
    metadata.accessCount++;
    metadata.lastAccessed = Date.now();

    // Update stats
    this.stats.hits++;

    // Log access
    this._logAccess('get', key);

    return value;
  }

  delete(key) {
    if (!this.store.has(key)) {
      return false;
    }

    // Clear TTL timer
    if (this.ttlTimers.has(key)) {
      clearTimeout(this.ttlTimers.get(key));
      this.ttlTimers.delete(key);
    }

    // Delete entries
    this.store.delete(key);
    this.metadata.delete(key);

    // Update stats
    this.stats.deletes++;

    // Log access
    this._logAccess('delete', key);

    return true;
  }

  has(key) {
    return this.store.has(key);
  }

  clear() {
    // Clear all TTL timers
    for (const timer of this.ttlTimers.values()) {
      clearTimeout(timer);
    }
    this.ttlTimers.clear();

    // Clear store and metadata
    this.store.clear();
    this.metadata.clear();
    this.accessLog = [];

    // Reset stats
    this.stats = {
      hits: 0,
      misses: 0,
      writes: 0,
      deletes: 0,
      evictions: 0
    };
  }

  list(filter = {}) {
    const entries = [];

    for (const [key, metadata] of this.metadata.entries()) {
      // Apply filters
      if (filter.type && metadata.type !== filter.type) continue;
      if (filter.tags && !filter.tags.some(tag => metadata.tags.includes(tag))) continue;
      if (filter.since && metadata.updatedAt < filter.since) continue;

      entries.push({
        key,
        ...metadata,
        hasValue: this.store.has(key)
      });
    }

    // Sort by access count or update time
    if (filter.sortBy === 'access') {
      entries.sort((a, b) => b.accessCount - a.accessCount);
    } else if (filter.sortBy === 'updated') {
      entries.sort((a, b) => b.updatedAt - a.updatedAt);
    }

    return entries;
  }

  search(pattern) {
    const results = [];
    const regex = new RegExp(pattern, 'i');

    for (const [key, value] of this.store.entries()) {
      if (regex.test(key) || regex.test(JSON.stringify(value))) {
        results.push({
          key,
          metadata: this.metadata.get(key),
          matched: regex.test(key) ? 'key' : 'value'
        });
      }
    }

    return results;
  }

  getStats() {
    return {
      ...this.stats,
      size: this.store.size,
      maxSize: this.options.maxSize,
      hitRate: this.stats.hits / (this.stats.hits + this.stats.misses) || 0,
      totalAccess: this.accessLog.length
    };
  }

  export() {
    const data = {
      version: '1.0',
      timestamp: Date.now(),
      options: this.options,
      entries: []
    };

    for (const [key, value] of this.store.entries()) {
      data.entries.push({
        key,
        value,
        metadata: this.metadata.get(key)
      });
    }

    return data;
  }

  import(data) {
    this.clear();

    // Validate version
    if (data.version !== '1.0') {
      throw new Error(`Unsupported version: ${data.version}`);
    }

    // Import entries
    for (const entry of data.entries) {
      this.store.set(entry.key, entry.value);
      this.metadata.set(entry.key, entry.metadata);
    }

    return data.entries.length;
  }

  // Private helper methods
  _logAccess(operation, key) {
    this.accessLog.push({
      operation,
      key,
      timestamp: Date.now()
    });

    // Keep log size limited
    if (this.accessLog.length > 10000) {
      this.accessLog = this.accessLog.slice(-5000);
    }
  }

  _evictOldest() {
    // Find least recently used key
    let oldestKey = null;
    let oldestAccess = Infinity;

    for (const [key, metadata] of this.metadata.entries()) {
      const lastAccess = metadata.lastAccessed || metadata.createdAt;
      if (lastAccess < oldestAccess) {
        oldestAccess = lastAccess;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.delete(oldestKey);
      this.stats.evictions++;
    }
  }

  _getSize(value) {
    // Simple size estimation
    return JSON.stringify(value).length;
  }

  _compress(value) {
    // Mock compression
    return { compressed: true, data: JSON.stringify(value) };
  }

  _decompress(value) {
    // Mock decompression
    return value.compressed ? JSON.parse(value.data) : value;
  }

  _encrypt(value) {
    // Mock encryption
    return { encrypted: true, data: value };
  }

  _decrypt(value) {
    // Mock decryption
    return value.encrypted ? value.data : value;
  }
}

// Distributed memory synchronization
class DistributedMemory {
  constructor() {
    this.nodes = new Map();
    this.syncLog = [];
    this.conflictResolver = this._defaultConflictResolver;
  }

  addNode(nodeId, store) {
    this.nodes.set(nodeId, {
      store,
      lastSync: Date.now(),
      version: 0
    });
  }

  removeNode(nodeId) {
    this.nodes.delete(nodeId);
  }

  sync(sourceNode, targetNode) {
    const source = this.nodes.get(sourceNode);
    const target = this.nodes.get(targetNode);

    if (!source || !target) {
      throw new Error('Invalid nodes for sync');
    }

    const changes = this._detectChanges(source, target);
    this._applyChanges(target, changes);

    // Update sync metadata
    source.lastSync = Date.now();
    target.lastSync = Date.now();
    target.version = source.version;

    // Log sync
    this.syncLog.push({
      timestamp: Date.now(),
      source: sourceNode,
      target: targetNode,
      changes: changes.length
    });

    return changes.length;
  }

  broadcast(sourceNode, key, value) {
    const source = this.nodes.get(sourceNode);
    if (!source) {
      throw new Error(`Invalid source node: ${sourceNode}`);
    }

    let synced = 0;
    for (const [nodeId, node] of this.nodes.entries()) {
      if (nodeId !== sourceNode) {
        node.store.set(key, value);
        synced++;
      }
    }

    return synced;
  }

  _detectChanges(source, target) {
    const changes = [];
    const sourceData = source.store.export();
    const targetData = target.store.export();

    const targetKeys = new Set(targetData.entries.map(e => e.key));

    for (const entry of sourceData.entries) {
      const targetEntry = targetData.entries.find(e => e.key === entry.key);

      if (!targetEntry) {
        // New key
        changes.push({
          type: 'add',
          key: entry.key,
          value: entry.value,
          metadata: entry.metadata
        });
      } else if (entry.metadata.updatedAt > targetEntry.metadata.updatedAt) {
        // Updated key
        changes.push({
          type: 'update',
          key: entry.key,
          value: entry.value,
          metadata: entry.metadata,
          conflict: targetEntry.metadata.updatedAt > source.lastSync
        });
      }
    }

    return changes;
  }

  _applyChanges(target, changes) {
    for (const change of changes) {
      if (change.conflict) {
        const resolved = this.conflictResolver(change);
        if (resolved) {
          target.store.set(change.key, resolved.value, resolved.metadata);
        }
      } else {
        target.store.set(change.key, change.value, change.metadata);
      }
    }
  }

  _defaultConflictResolver(change) {
    // Default: latest update wins
    return change;
  }
}

describe('Memory Operations', () => {
  let store;

  beforeEach(() => {
    store = new MemoryStore();
  });

  describe('Basic Operations', () => {
    it('should store and retrieve values', () => {
      store.set('key1', 'value1');
      expect(store.get('key1')).toBe('value1');
    });

    it('should handle complex objects', () => {
      const obj = { name: 'test', data: [1, 2, 3], nested: { a: 1 } };
      store.set('complex', obj);
      expect(store.get('complex')).toEqual(obj);
    });

    it('should return undefined for missing keys', () => {
      expect(store.get('nonexistent')).toBeUndefined();
    });

    it('should check key existence', () => {
      store.set('exists', 'yes');
      expect(store.has('exists')).toBe(true);
      expect(store.has('notexists')).toBe(false);
    });

    it('should delete keys', () => {
      store.set('temp', 'data');
      expect(store.has('temp')).toBe(true);
      
      const deleted = store.delete('temp');
      expect(deleted).toBe(true);
      expect(store.has('temp')).toBe(false);
    });

    it('should clear all data', () => {
      store.set('key1', 'value1');
      store.set('key2', 'value2');
      
      store.clear();
      
      expect(store.has('key1')).toBe(false);
      expect(store.has('key2')).toBe(false);
      expect(store.getStats().size).toBe(0);
    });
  });

  describe('TTL (Time To Live)', () => {
    jest.useFakeTimers();

    it('should expire keys after TTL', () => {
      store.set('expire', 'soon', { ttl: 1000 });
      
      expect(store.get('expire')).toBe('soon');
      
      jest.advanceTimersByTime(1001);
      
      expect(store.has('expire')).toBe(false);
    });

    it('should handle default TTL', () => {
      const ttlStore = new MemoryStore({ defaultTTL: 500 });
      ttlStore.set('auto-expire', 'value');
      
      expect(ttlStore.get('auto-expire')).toBe('value');
      
      jest.advanceTimersByTime(501);
      
      expect(ttlStore.has('auto-expire')).toBe(false);
    });

    it('should clear TTL timer on delete', () => {
      store.set('temp', 'value', { ttl: 1000 });
      store.delete('temp');
      
      jest.advanceTimersByTime(1001);
      
      // Should not throw or cause issues
      expect(store.has('temp')).toBe(false);
    });

    jest.useRealTimers();
  });

  describe('Memory Types and Metadata', () => {
    it('should categorize memory by type', () => {
      store.set('api-key', 'secret', { type: MEMORY_TYPES.CONFIG });
      store.set('cache-1', 'data', { type: MEMORY_TYPES.CACHE });
      store.set('fact-1', 'info', { type: MEMORY_TYPES.KNOWLEDGE });

      const configs = store.list({ type: MEMORY_TYPES.CONFIG });
      expect(configs).toHaveLength(1);
      expect(configs[0].key).toBe('api-key');
    });

    it('should support tags', () => {
      store.set('doc1', 'content', { tags: ['important', 'api'] });
      store.set('doc2', 'content', { tags: ['api'] });
      store.set('doc3', 'content', { tags: ['temp'] });

      const apiDocs = store.list({ tags: ['api'] });
      expect(apiDocs).toHaveLength(2);
    });

    it('should track access count', () => {
      store.set('popular', 'data');
      
      store.get('popular');
      store.get('popular');
      store.get('popular');

      const entries = store.list();
      const popular = entries.find(e => e.key === 'popular');
      expect(popular.accessCount).toBe(3);
    });

    it('should track timestamps', () => {
      const before = Date.now();
      store.set('timed', 'value');
      const after = Date.now();

      const entries = store.list();
      const timed = entries.find(e => e.key === 'timed');
      
      expect(timed.createdAt).toBeGreaterThanOrEqual(before);
      expect(timed.createdAt).toBeLessThanOrEqual(after);
      expect(timed.updatedAt).toBe(timed.createdAt);
    });
  });

  describe('Size Management', () => {
    it('should enforce size limits', () => {
      const smallStore = new MemoryStore({ maxSize: 3 });
      
      smallStore.set('key1', 'value1');
      smallStore.set('key2', 'value2');
      smallStore.set('key3', 'value3');
      
      expect(smallStore.getStats().size).toBe(3);
      
      // Should evict oldest
      smallStore.set('key4', 'value4');
      
      expect(smallStore.getStats().size).toBe(3);
      expect(smallStore.has('key4')).toBe(true);
    });

    it('should track evictions', () => {
      const smallStore = new MemoryStore({ maxSize: 2 });
      
      smallStore.set('key1', 'value1');
      smallStore.set('key2', 'value2');
      smallStore.set('key3', 'value3');
      
      expect(smallStore.getStats().evictions).toBe(1);
    });

    it('should evict least recently used', () => {
      const smallStore = new MemoryStore({ maxSize: 3 });
      
      smallStore.set('key1', 'value1');
      smallStore.set('key2', 'value2');
      smallStore.set('key3', 'value3');
      
      // Access key1 and key3
      smallStore.get('key1');
      smallStore.get('key3');
      
      // Add new key - should evict key2
      smallStore.set('key4', 'value4');
      
      expect(smallStore.has('key1')).toBe(true);
      expect(smallStore.has('key2')).toBe(false);
      expect(smallStore.has('key3')).toBe(true);
      expect(smallStore.has('key4')).toBe(true);
    });
  });

  describe('Search and List', () => {
    beforeEach(() => {
      store.set('user:1', { name: 'Alice', role: 'admin' });
      store.set('user:2', { name: 'Bob', role: 'user' });
      store.set('config:api', { endpoint: 'https://api.example.com' });
      store.set('config:db', { host: 'localhost', port: 5432 });
    });

    it('should search by pattern', () => {
      const users = store.search('user:');
      expect(users).toHaveLength(2);
      expect(users.every(r => r.matched === 'key')).toBe(true);
    });

    it('should search in values', () => {
      const admins = store.search('admin');
      expect(admins).toHaveLength(1);
      expect(admins[0].key).toBe('user:1');
      expect(admins[0].matched).toBe('value');
    });

    it('should list with sorting', () => {
      // Access some keys to change access count
      store.get('user:1');
      store.get('user:1');
      store.get('config:api');

      const byAccess = store.list({ sortBy: 'access' });
      expect(byAccess[0].key).toBe('user:1');
      expect(byAccess[0].accessCount).toBe(2);
    });
  });

  describe('Statistics', () => {
    it('should track hit rate', () => {
      store.set('key1', 'value1');
      store.set('key2', 'value2');

      store.get('key1'); // hit
      store.get('key2'); // hit
      store.get('key3'); // miss
      store.get('key4'); // miss

      const stats = store.getStats();
      expect(stats.hits).toBe(2);
      expect(stats.misses).toBe(2);
      expect(stats.hitRate).toBe(0.5);
    });

    it('should track operations', () => {
      store.set('key1', 'value1');
      store.set('key2', 'value2');
      store.get('key1');
      store.delete('key2');

      const stats = store.getStats();
      expect(stats.writes).toBe(2);
      expect(stats.deletes).toBe(1);
      expect(stats.size).toBe(1);
    });
  });

  describe('Export/Import', () => {
    it('should export memory data', () => {
      store.set('key1', 'value1', { type: MEMORY_TYPES.CONFIG });
      store.set('key2', { data: 'complex' }, { tags: ['important'] });

      const exported = store.export();
      
      expect(exported.version).toBe('1.0');
      expect(exported.entries).toHaveLength(2);
      expect(exported.entries[0].key).toBe('key1');
      expect(exported.entries[0].metadata.type).toBe(MEMORY_TYPES.CONFIG);
    });

    it('should import memory data', () => {
      const data = {
        version: '1.0',
        timestamp: Date.now(),
        entries: [
          {
            key: 'imported1',
            value: 'data1',
            metadata: { type: MEMORY_TYPES.GENERAL, createdAt: Date.now() }
          },
          {
            key: 'imported2',
            value: { complex: true },
            metadata: { type: MEMORY_TYPES.CONFIG, createdAt: Date.now() }
          }
        ]
      };

      const count = store.import(data);
      
      expect(count).toBe(2);
      expect(store.get('imported1')).toBe('data1');
      expect(store.get('imported2')).toEqual({ complex: true });
    });
  });

  describe('Distributed Memory', () => {
    let distributed;
    let node1, node2, node3;

    beforeEach(() => {
      distributed = new DistributedMemory();
      node1 = new MemoryStore();
      node2 = new MemoryStore();
      node3 = new MemoryStore();

      distributed.addNode('node1', node1);
      distributed.addNode('node2', node2);
      distributed.addNode('node3', node3);
    });

    it('should sync between nodes', () => {
      node1.set('shared', 'data');
      node1.set('config', { port: 3000 });

      const synced = distributed.sync('node1', 'node2');
      
      expect(synced).toBe(2);
      expect(node2.get('shared')).toBe('data');
      expect(node2.get('config')).toEqual({ port: 3000 });
    });

    it('should broadcast to all nodes', () => {
      const count = distributed.broadcast('node1', 'announcement', 'Hello all!');
      
      expect(count).toBe(2);
      expect(node2.get('announcement')).toBe('Hello all!');
      expect(node3.get('announcement')).toBe('Hello all!');
      expect(node1.get('announcement')).toBeUndefined(); // Source doesn't get it
    });

    it('should detect conflicts', () => {
      node1.set('conflict', 'value1');
      node2.set('conflict', 'value2');

      // Mock to detect conflict
      const changes = distributed._detectChanges(
        distributed.nodes.get('node1'),
        distributed.nodes.get('node2')
      );

      const conflictChange = changes.find(c => c.key === 'conflict');
      expect(conflictChange).toBeDefined();
      expect(conflictChange.type).toBe('update');
    });
  });
});