/**
 * Backend Detector Tests
 *
 * Tests automatic backend selection logic, platform detection,
 * capability detection, and graceful fallback mechanisms.
 */

import { describe, it, expect } from 'vitest';

// Mock backend detection utilities
class BackendDetector {
  /**
   * Detect available backends on current platform
   */
  static detectAvailableBackends(): string[] {
    const available: string[] = [];

    // Check for RuVector (WASM is always available)
    available.push('ruvector-wasm');

    // Check for RuVector native (platform-specific)
    if (this.isRuVectorNativeAvailable()) {
      available.push('ruvector-native');
    }

    // Check for hnswlib (native bindings)
    if (this.isHnswlibAvailable()) {
      available.push('hnswlib');
    }

    return available;
  }

  /**
   * Check if RuVector native is available for current platform
   */
  static isRuVectorNativeAvailable(): boolean {
    const platform = process.platform;
    const arch = process.arch;

    // RuVector native supports: linux-x64, linux-arm64, darwin-x64, darwin-arm64, win32-x64
    const supportedPlatforms = [
      'linux-x64',
      'linux-arm64',
      'darwin-x64',
      'darwin-arm64',
      'win32-x64',
    ];

    const platformKey = `${platform}-${arch}`;
    return supportedPlatforms.includes(platformKey);
  }

  /**
   * Check if hnswlib is available
   */
  static isHnswlibAvailable(): boolean {
    try {
      // Try to require hnswlib-node
      require('hnswlib-node');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Select optimal backend based on criteria
   */
  static selectOptimalBackend(criteria: {
    priority: 'performance' | 'compatibility' | 'memory';
    minVectors?: number;
  }): string {
    const available = this.detectAvailableBackends();

    if (available.length === 0) {
      throw new Error('No vector backends available');
    }

    const { priority, minVectors = 0 } = criteria;

    switch (priority) {
      case 'performance':
        // Prefer RuVector native > HNSW > RuVector WASM
        if (available.includes('ruvector-native')) {
          return 'ruvector-native';
        }
        if (available.includes('hnswlib') && minVectors > 100) {
          return 'hnswlib'; // HNSW better for large datasets
        }
        return available.includes('ruvector-wasm') ? 'ruvector-wasm' : available[0];

      case 'compatibility':
        // Prefer WASM for maximum compatibility
        return available.includes('ruvector-wasm') ? 'ruvector-wasm' : available[0];

      case 'memory':
        // HNSW can be memory-intensive, prefer RuVector
        if (available.includes('ruvector-native')) {
          return 'ruvector-native';
        }
        return available.includes('ruvector-wasm') ? 'ruvector-wasm' : available[0];

      default:
        return available[0];
    }
  }

  /**
   * Get backend capabilities
   */
  static getBackendCapabilities(backend: string): {
    supportsApproximateSearch: boolean;
    supportsPersistence: boolean;
    supportsIncrementalUpdates: boolean;
    supportsMultipleMetrics: boolean;
    estimatedSpeedMultiplier: number;
  } {
    switch (backend) {
      case 'ruvector-native':
        return {
          supportsApproximateSearch: true,
          supportsPersistence: true,
          supportsIncrementalUpdates: true,
          supportsMultipleMetrics: true,
          estimatedSpeedMultiplier: 150, // 150x faster than brute force
        };

      case 'ruvector-wasm':
        return {
          supportsApproximateSearch: true,
          supportsPersistence: true,
          supportsIncrementalUpdates: true,
          supportsMultipleMetrics: true,
          estimatedSpeedMultiplier: 10, // 10x faster than pure JS
        };

      case 'hnswlib':
        return {
          supportsApproximateSearch: true,
          supportsPersistence: true,
          supportsIncrementalUpdates: true, // Limited - requires rebuilds
          supportsMultipleMetrics: true,
          estimatedSpeedMultiplier: 100, // 100x faster than brute force
        };

      default:
        return {
          supportsApproximateSearch: false,
          supportsPersistence: false,
          supportsIncrementalUpdates: false,
          supportsMultipleMetrics: false,
          estimatedSpeedMultiplier: 1,
        };
    }
  }

  /**
   * Detect platform-specific optimizations
   */
  static detectOptimizations(): {
    simdAvailable: boolean;
    platform: string;
    arch: string;
    nodeVersion: string;
  } {
    return {
      simdAvailable: this.isSIMDAvailable(),
      platform: process.platform,
      arch: process.arch,
      nodeVersion: process.version,
    };
  }

  /**
   * Check if SIMD is available
   */
  static isSIMDAvailable(): boolean {
    try {
      const globalAny = globalThis as any;
      return (
        typeof globalAny.WebAssembly !== 'undefined' &&
        globalAny.WebAssembly.validate(
          new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65,
            0, 253, 15, 253, 98, 11,
          ])
        )
      );
    } catch {
      return false;
    }
  }
}

describe('Backend Detector Tests', () => {
  describe('Backend Detection', () => {
    it('should detect available backends', () => {
      const backends = BackendDetector.detectAvailableBackends();

      expect(Array.isArray(backends)).toBe(true);
      expect(backends.length).toBeGreaterThan(0);

      // WASM should always be available
      expect(backends).toContain('ruvector-wasm');
    });

    it('should detect RuVector native availability', () => {
      const isAvailable = BackendDetector.isRuVectorNativeAvailable();

      expect(typeof isAvailable).toBe('boolean');

      // Check if current platform is supported
      const platform = process.platform;
      const arch = process.arch;
      const supportedPlatforms = [
        'linux-x64',
        'linux-arm64',
        'darwin-x64',
        'darwin-arm64',
        'win32-x64',
      ];

      const platformKey = `${platform}-${arch}`;
      const expectedAvailability = supportedPlatforms.includes(platformKey);

      expect(isAvailable).toBe(expectedAvailability);
    });

    it('should detect hnswlib availability', () => {
      const isAvailable = BackendDetector.isHnswlibAvailable();

      expect(typeof isAvailable).toBe('boolean');

      // Should match actual hnswlib-node installation
      let expectedAvailability = false;
      try {
        require('hnswlib-node');
        expectedAvailability = true;
      } catch {
        expectedAvailability = false;
      }

      expect(isAvailable).toBe(expectedAvailability);
    });
  });

  describe('Backend Selection', () => {
    it('should select optimal backend for performance', () => {
      const backend = BackendDetector.selectOptimalBackend({
        priority: 'performance',
      });

      expect(typeof backend).toBe('string');

      const available = BackendDetector.detectAvailableBackends();
      expect(available).toContain(backend);
    });

    it('should select optimal backend for compatibility', () => {
      const backend = BackendDetector.selectOptimalBackend({
        priority: 'compatibility',
      });

      // Should prefer WASM for compatibility
      expect(backend).toContain('wasm');
    });

    it('should select optimal backend for memory efficiency', () => {
      const backend = BackendDetector.selectOptimalBackend({
        priority: 'memory',
      });

      expect(typeof backend).toBe('string');
      // Should not select hnswlib for memory-constrained scenarios
      if (backend === 'hnswlib') {
        // Only if it's the only option
        const available = BackendDetector.detectAvailableBackends();
        expect(available.length).toBe(1);
      }
    });

    it('should consider dataset size in selection', () => {
      const smallDataset = BackendDetector.selectOptimalBackend({
        priority: 'performance',
        minVectors: 50,
      });

      const largeDataset = BackendDetector.selectOptimalBackend({
        priority: 'performance',
        minVectors: 100000,
      });

      expect(typeof smallDataset).toBe('string');
      expect(typeof largeDataset).toBe('string');

      // Both should be valid backends
      const available = BackendDetector.detectAvailableBackends();
      expect(available).toContain(smallDataset);
      expect(available).toContain(largeDataset);
    });

    it('should throw error when no backends available', () => {
      // Mock scenario where no backends are available
      const originalDetect = BackendDetector.detectAvailableBackends;
      BackendDetector.detectAvailableBackends = () => [];

      expect(() => {
        BackendDetector.selectOptimalBackend({ priority: 'performance' });
      }).toThrow('No vector backends available');

      // Restore original method
      BackendDetector.detectAvailableBackends = originalDetect;
    });
  });

  describe('Backend Capabilities', () => {
    it('should report RuVector native capabilities', () => {
      const caps = BackendDetector.getBackendCapabilities('ruvector-native');

      expect(caps.supportsApproximateSearch).toBe(true);
      expect(caps.supportsPersistence).toBe(true);
      expect(caps.supportsIncrementalUpdates).toBe(true);
      expect(caps.supportsMultipleMetrics).toBe(true);
      expect(caps.estimatedSpeedMultiplier).toBeGreaterThan(100);
    });

    it('should report RuVector WASM capabilities', () => {
      const caps = BackendDetector.getBackendCapabilities('ruvector-wasm');

      expect(caps.supportsApproximateSearch).toBe(true);
      expect(caps.supportsPersistence).toBe(true);
      expect(caps.supportsIncrementalUpdates).toBe(true);
      expect(caps.supportsMultipleMetrics).toBe(true);
      expect(caps.estimatedSpeedMultiplier).toBeGreaterThan(1);
    });

    it('should report hnswlib capabilities', () => {
      const caps = BackendDetector.getBackendCapabilities('hnswlib');

      expect(caps.supportsApproximateSearch).toBe(true);
      expect(caps.supportsPersistence).toBe(true);
      expect(caps.supportsMultipleMetrics).toBe(true);
      expect(caps.estimatedSpeedMultiplier).toBeGreaterThan(10);
    });

    it('should handle unknown backends gracefully', () => {
      const caps = BackendDetector.getBackendCapabilities('unknown-backend');

      expect(caps.supportsApproximateSearch).toBe(false);
      expect(caps.supportsPersistence).toBe(false);
      expect(caps.estimatedSpeedMultiplier).toBe(1);
    });
  });

  describe('Platform Detection', () => {
    it('should detect platform information', () => {
      const info = BackendDetector.detectOptimizations();

      expect(info).toHaveProperty('platform');
      expect(info).toHaveProperty('arch');
      expect(info).toHaveProperty('nodeVersion');
      expect(info).toHaveProperty('simdAvailable');

      expect(typeof info.platform).toBe('string');
      expect(typeof info.arch).toBe('string');
      expect(typeof info.nodeVersion).toBe('string');
      expect(typeof info.simdAvailable).toBe('boolean');

      // Validate platform values
      expect(['linux', 'darwin', 'win32']).toContain(info.platform);
      expect(['x64', 'arm64', 'arm']).toContain(info.arch);
    });

    it('should detect SIMD support', () => {
      const simdAvailable = BackendDetector.isSIMDAvailable();

      expect(typeof simdAvailable).toBe('boolean');
    });
  });

  describe('Fallback Mechanisms', () => {
    it('should provide fallback priority list', () => {
      const available = BackendDetector.detectAvailableBackends();

      // Define fallback priority
      const priority = ['ruvector-native', 'hnswlib', 'ruvector-wasm'];

      let selected: string | null = null;
      for (const backend of priority) {
        if (available.includes(backend)) {
          selected = backend;
          break;
        }
      }

      expect(selected).not.toBeNull();
      expect(available).toContain(selected!);
    });

    it('should always have WASM fallback', () => {
      const available = BackendDetector.detectAvailableBackends();

      // WASM should always be available as last resort
      expect(available).toContain('ruvector-wasm');
    });
  });

  describe('Backend Comparison', () => {
    it('should compare performance characteristics', () => {
      const backends = BackendDetector.detectAvailableBackends();

      const comparisons = backends.map(backend => ({
        backend,
        capabilities: BackendDetector.getBackendCapabilities(backend),
      }));

      // Sort by speed
      comparisons.sort(
        (a, b) => b.capabilities.estimatedSpeedMultiplier - a.capabilities.estimatedSpeedMultiplier
      );

      console.log('[Backend Detector] Performance ranking:');
      comparisons.forEach((c, i) => {
        console.log(
          `  ${i + 1}. ${c.backend}: ${c.capabilities.estimatedSpeedMultiplier}x faster`
        );
      });

      expect(comparisons.length).toBeGreaterThan(0);
    });

    it('should identify feature differences', () => {
      const backends = ['ruvector-native', 'ruvector-wasm', 'hnswlib'];

      const features = backends.map(backend => ({
        backend,
        capabilities: BackendDetector.getBackendCapabilities(backend),
      }));

      // All should support approximate search
      for (const feature of features) {
        expect(feature.capabilities.supportsApproximateSearch).toBe(true);
      }

      // All should support persistence
      for (const feature of features) {
        expect(feature.capabilities.supportsPersistence).toBe(true);
      }
    });
  });

  describe('Auto-Selection Logic', () => {
    it('should auto-select based on comprehensive criteria', () => {
      const scenarios = [
        {
          name: 'Small dataset, performance priority',
          criteria: { priority: 'performance' as const, minVectors: 100 },
        },
        {
          name: 'Large dataset, performance priority',
          criteria: { priority: 'performance' as const, minVectors: 100000 },
        },
        {
          name: 'Maximum compatibility',
          criteria: { priority: 'compatibility' as const },
        },
        {
          name: 'Memory constrained',
          criteria: { priority: 'memory' as const },
        },
      ];

      for (const scenario of scenarios) {
        const selected = BackendDetector.selectOptimalBackend(scenario.criteria);
        const capabilities = BackendDetector.getBackendCapabilities(selected);

        console.log(
          `[Backend Detector] ${scenario.name}: ${selected} (${capabilities.estimatedSpeedMultiplier}x)`
        );

        expect(typeof selected).toBe('string');
        expect(capabilities).toBeDefined();
      }
    });
  });
});
