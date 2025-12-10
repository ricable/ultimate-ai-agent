import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

/**
 * SyncCoordinator Unit Tests
 *
 * Tests for synchronization coordination including:
 * - Conflict detection and resolution
 * - Version vector management
 * - Bidirectional sync coordination
 * - Merge strategies
 * - Sync state tracking
 */

interface VersionVector {
  [nodeId: string]: number;
}

interface SyncItem {
  id: string;
  version: VersionVector;
  data: any;
  timestamp: number;
}

// Mock SyncCoordinator implementation for testing
class MockSyncCoordinator {
  private localVersion: VersionVector = {};
  private nodeId: string;
  private syncState: Map<string, any> = new Map();
  private conflictResolver: (local: any, remote: any) => any;

  constructor(nodeId: string, conflictResolver?: (local: any, remote: any) => any) {
    this.nodeId = nodeId;
    this.localVersion[nodeId] = 0;
    this.conflictResolver = conflictResolver || this.defaultConflictResolver;
  }

  private defaultConflictResolver(local: any, remote: any): any {
    // Last-write-wins by timestamp
    return local.timestamp > remote.timestamp ? local : remote;
  }

  incrementVersion(): void {
    this.localVersion[this.nodeId]++;
  }

  getVersion(): VersionVector {
    return { ...this.localVersion };
  }

  updateVersion(nodeId: string, version: number): void {
    this.localVersion[nodeId] = Math.max(this.localVersion[nodeId] || 0, version);
  }

  mergeVersions(remote: VersionVector): void {
    for (const [nodeId, version] of Object.entries(remote)) {
      this.updateVersion(nodeId, version);
    }
  }

  compareVersions(v1: VersionVector, v2: VersionVector): 'before' | 'after' | 'concurrent' {
    let v1Greater = false;
    let v2Greater = false;

    const allNodes = new Set([...Object.keys(v1), ...Object.keys(v2)]);

    for (const node of allNodes) {
      const val1 = v1[node] || 0;
      const val2 = v2[node] || 0;

      if (val1 > val2) v1Greater = true;
      if (val2 > val1) v2Greater = true;
    }

    if (v1Greater && v2Greater) return 'concurrent';
    if (v1Greater) return 'after';
    if (v2Greater) return 'before';
    return 'concurrent'; // Equal versions
  }

  async syncItem(item: SyncItem): Promise<{ action: string; result: any }> {
    const existing = this.syncState.get(item.id);

    if (!existing) {
      // New item
      this.syncState.set(item.id, item);
      this.mergeVersions(item.version);
      return { action: 'added', result: item };
    }

    const comparison = this.compareVersions(item.version, existing.version);

    switch (comparison) {
      case 'before':
        // Remote is older, keep local
        return { action: 'kept_local', result: existing };

      case 'after':
        // Remote is newer, update local
        this.syncState.set(item.id, item);
        this.mergeVersions(item.version);
        return { action: 'updated', result: item };

      case 'concurrent':
        // Conflict - use resolver
        const resolved = this.conflictResolver(existing, item);
        this.syncState.set(item.id, resolved);
        this.mergeVersions(item.version);
        return { action: 'resolved_conflict', result: resolved };
    }
  }

  async syncBatch(items: SyncItem[]): Promise<{ added: number; updated: number; conflicts: number }> {
    let added = 0;
    let updated = 0;
    let conflicts = 0;

    for (const item of items) {
      const result = await this.syncItem(item);

      switch (result.action) {
        case 'added':
          added++;
          break;
        case 'updated':
          updated++;
          break;
        case 'resolved_conflict':
          conflicts++;
          break;
      }
    }

    return { added, updated, conflicts };
  }

  getItemsAfterVersion(version: VersionVector): SyncItem[] {
    const items: SyncItem[] = [];

    for (const item of this.syncState.values()) {
      if (this.compareVersions(item.version, version) === 'after') {
        items.push(item);
      }
    }

    return items;
  }

  async bidirectionalSync(remoteItems: SyncItem[], remoteVersion: VersionVector): Promise<{
    toSend: SyncItem[];
    received: { added: number; updated: number; conflicts: number };
  }> {
    // Sync remote items to local
    const received = await this.syncBatch(remoteItems);

    // Find items to send to remote
    const toSend = this.getItemsAfterVersion(remoteVersion);

    return { toSend, received };
  }

  detectConflicts(items: SyncItem[]): SyncItem[] {
    const conflicts: SyncItem[] = [];

    for (const item of items) {
      const existing = this.syncState.get(item.id);

      if (existing) {
        const comparison = this.compareVersions(item.version, existing.version);
        if (comparison === 'concurrent') {
          conflicts.push(item);
        }
      }
    }

    return conflicts;
  }

  getSyncState(): Map<string, any> {
    return new Map(this.syncState);
  }

  clearSyncState(): void {
    this.syncState.clear();
  }

  getNodeId(): string {
    return this.nodeId;
  }
}

describe('SyncCoordinator', () => {
  let coordinator: MockSyncCoordinator;

  beforeEach(() => {
    coordinator = new MockSyncCoordinator('node-1');
  });

  afterEach(() => {
    coordinator.clearSyncState();
  });

  describe('Version Vector Management', () => {
    it('should initialize with node version', () => {
      const version = coordinator.getVersion();
      expect(version['node-1']).toBe(0);
    });

    it('should increment local version', () => {
      coordinator.incrementVersion();
      const version = coordinator.getVersion();
      expect(version['node-1']).toBe(1);
    });

    it('should update version for specific node', () => {
      coordinator.updateVersion('node-2', 5);
      const version = coordinator.getVersion();
      expect(version['node-2']).toBe(5);
    });

    it('should merge remote versions', () => {
      const remoteVersion = { 'node-2': 3, 'node-3': 2 };
      coordinator.mergeVersions(remoteVersion);

      const version = coordinator.getVersion();
      expect(version['node-2']).toBe(3);
      expect(version['node-3']).toBe(2);
    });

    it('should keep higher version when merging', () => {
      coordinator.updateVersion('node-2', 5);
      coordinator.mergeVersions({ 'node-2': 3 });

      const version = coordinator.getVersion();
      expect(version['node-2']).toBe(5);
    });
  });

  describe('Version Comparison', () => {
    it('should detect when v1 is before v2', () => {
      const v1 = { 'node-1': 1 };
      const v2 = { 'node-1': 2 };

      expect(coordinator.compareVersions(v1, v2)).toBe('before');
    });

    it('should detect when v1 is after v2', () => {
      const v1 = { 'node-1': 3 };
      const v2 = { 'node-1': 2 };

      expect(coordinator.compareVersions(v1, v2)).toBe('after');
    });

    it('should detect concurrent versions', () => {
      const v1 = { 'node-1': 2, 'node-2': 1 };
      const v2 = { 'node-1': 1, 'node-2': 2 };

      expect(coordinator.compareVersions(v1, v2)).toBe('concurrent');
    });

    it('should handle missing nodes in version vectors', () => {
      const v1 = { 'node-1': 1 };
      const v2 = { 'node-2': 1 };

      expect(coordinator.compareVersions(v1, v2)).toBe('concurrent');
    });
  });

  describe('Item Synchronization', () => {
    it('should add new item', async () => {
      const item: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 1 },
        data: { content: 'test' },
        timestamp: Date.now()
      };

      const result = await coordinator.syncItem(item);
      expect(result.action).toBe('added');
      expect(coordinator.getSyncState().size).toBe(1);
    });

    it('should update item when remote is newer', async () => {
      const oldItem: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 1 },
        data: { content: 'old' },
        timestamp: Date.now()
      };

      await coordinator.syncItem(oldItem);

      const newItem: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 2 },
        data: { content: 'new' },
        timestamp: Date.now()
      };

      const result = await coordinator.syncItem(newItem);
      expect(result.action).toBe('updated');
      expect(result.result.data.content).toBe('new');
    });

    it('should keep local when remote is older', async () => {
      const newItem: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 2 },
        data: { content: 'new' },
        timestamp: Date.now()
      };

      await coordinator.syncItem(newItem);

      const oldItem: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 1 },
        data: { content: 'old' },
        timestamp: Date.now()
      };

      const result = await coordinator.syncItem(oldItem);
      expect(result.action).toBe('kept_local');
      expect(result.result.data.content).toBe('new');
    });

    it('should resolve conflicts with custom resolver', async () => {
      const customResolver = (local: any, remote: any) => ({
        ...remote,
        data: { content: 'merged' }
      });

      const customCoordinator = new MockSyncCoordinator('node-1', customResolver);

      const item1: SyncItem = {
        id: 'item-1',
        version: { 'node-1': 1 },
        data: { content: 'local' },
        timestamp: Date.now()
      };

      await customCoordinator.syncItem(item1);

      const item2: SyncItem = {
        id: 'item-1',
        version: { 'node-2': 1 },
        data: { content: 'remote' },
        timestamp: Date.now() + 100
      };

      const result = await customCoordinator.syncItem(item2);
      expect(result.action).toBe('resolved_conflict');
      expect(result.result.data.content).toBe('merged');
    });
  });

  describe('Batch Synchronization', () => {
    it('should sync multiple items', async () => {
      const items: SyncItem[] = [
        {
          id: 'item-1',
          version: { 'node-2': 1 },
          data: { content: 'test1' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-2': 1 },
          data: { content: 'test2' },
          timestamp: Date.now()
        },
        {
          id: 'item-3',
          version: { 'node-2': 1 },
          data: { content: 'test3' },
          timestamp: Date.now()
        }
      ];

      const result = await coordinator.syncBatch(items);
      expect(result.added).toBe(3);
      expect(result.updated).toBe(0);
      expect(result.conflicts).toBe(0);
    });

    it('should track additions, updates, and conflicts separately', async () => {
      // Add initial items
      await coordinator.syncBatch([
        {
          id: 'item-1',
          version: { 'node-1': 1 },
          data: { content: 'v1' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-1': 1 },
          data: { content: 'v1' },
          timestamp: Date.now()
        }
      ]);

      // Batch with new, updated, and conflicting items
      const result = await coordinator.syncBatch([
        {
          id: 'item-1',
          version: { 'node-1': 2 },
          data: { content: 'v2' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-2': 1 },
          data: { content: 'conflict' },
          timestamp: Date.now()
        },
        {
          id: 'item-3',
          version: { 'node-2': 1 },
          data: { content: 'new' },
          timestamp: Date.now()
        }
      ]);

      expect(result.added).toBe(1);
      expect(result.updated).toBe(1);
      expect(result.conflicts).toBe(1);
    });

    it('should handle empty batch', async () => {
      const result = await coordinator.syncBatch([]);
      expect(result.added).toBe(0);
      expect(result.updated).toBe(0);
      expect(result.conflicts).toBe(0);
    });
  });

  describe('Bidirectional Sync', () => {
    it('should perform complete bidirectional sync', async () => {
      // Add local items
      await coordinator.syncBatch([
        {
          id: 'local-1',
          version: { 'node-1': 1 },
          data: { content: 'local' },
          timestamp: Date.now()
        }
      ]);

      // Remote items to sync
      const remoteItems: SyncItem[] = [
        {
          id: 'remote-1',
          version: { 'node-2': 1 },
          data: { content: 'remote' },
          timestamp: Date.now()
        }
      ];

      const remoteVersion = { 'node-2': 0 };

      const result = await coordinator.bidirectionalSync(remoteItems, remoteVersion);

      expect(result.received.added).toBe(1);
      expect(result.toSend.length).toBeGreaterThan(0);
    });

    it('should not send items already known to remote', async () => {
      await coordinator.syncBatch([
        {
          id: 'item-1',
          version: { 'node-1': 1 },
          data: { content: 'test' },
          timestamp: Date.now()
        }
      ]);

      const remoteVersion = { 'node-1': 1 };

      const result = await coordinator.bidirectionalSync([], remoteVersion);

      expect(result.toSend.length).toBe(0);
    });
  });

  describe('Conflict Detection', () => {
    it('should detect concurrent conflicts', async () => {
      const item: SyncItem = {
        id: 'item-1',
        version: { 'node-1': 1 },
        data: { content: 'local' },
        timestamp: Date.now()
      };

      await coordinator.syncItem(item);

      const conflictingItems: SyncItem[] = [
        {
          id: 'item-1',
          version: { 'node-2': 1 },
          data: { content: 'remote' },
          timestamp: Date.now()
        }
      ];

      const conflicts = coordinator.detectConflicts(conflictingItems);
      expect(conflicts.length).toBe(1);
      expect(conflicts[0].id).toBe('item-1');
    });

    it('should not detect conflicts for non-concurrent versions', async () => {
      const item: SyncItem = {
        id: 'item-1',
        version: { 'node-1': 1 },
        data: { content: 'local' },
        timestamp: Date.now()
      };

      await coordinator.syncItem(item);

      const nonConflictingItems: SyncItem[] = [
        {
          id: 'item-1',
          version: { 'node-1': 2 },
          data: { content: 'newer' },
          timestamp: Date.now()
        }
      ];

      const conflicts = coordinator.detectConflicts(nonConflictingItems);
      expect(conflicts.length).toBe(0);
    });

    it('should handle multiple conflicts', async () => {
      await coordinator.syncBatch([
        {
          id: 'item-1',
          version: { 'node-1': 1 },
          data: { content: 'local1' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-1': 1 },
          data: { content: 'local2' },
          timestamp: Date.now()
        }
      ]);

      const conflictingItems: SyncItem[] = [
        {
          id: 'item-1',
          version: { 'node-2': 1 },
          data: { content: 'remote1' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-2': 1 },
          data: { content: 'remote2' },
          timestamp: Date.now()
        }
      ];

      const conflicts = coordinator.detectConflicts(conflictingItems);
      expect(conflicts.length).toBe(2);
    });
  });

  describe('Incremental Sync', () => {
    it('should retrieve items after specific version', async () => {
      await coordinator.syncBatch([
        {
          id: 'item-1',
          version: { 'node-1': 1 },
          data: { content: 'v1' },
          timestamp: Date.now()
        },
        {
          id: 'item-2',
          version: { 'node-1': 2 },
          data: { content: 'v2' },
          timestamp: Date.now()
        },
        {
          id: 'item-3',
          version: { 'node-1': 3 },
          data: { content: 'v3' },
          timestamp: Date.now()
        }
      ]);

      const items = coordinator.getItemsAfterVersion({ 'node-1': 1 });
      expect(items.length).toBeGreaterThan(0);
    });
  });
});
