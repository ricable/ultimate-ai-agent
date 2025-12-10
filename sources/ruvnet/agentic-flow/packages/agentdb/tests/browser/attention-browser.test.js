/**
 * @test Browser WASM Attention Tests
 * @description Test attention mechanisms in browser environments with WASM fallback
 * @prerequisites
 *   - Browser environment (Chrome, Firefox, Safari)
 *   - WASM support
 * @coverage
 *   - WASM module loading
 *   - Lazy loading behavior
 *   - Fallback mechanisms
 *   - Cross-browser compatibility
 */

// Browser-compatible test setup
const { describe, it, expect, beforeAll, afterAll } = window.vitest || require('vitest');

describe('Attention Mechanism Browser Tests', () => {
  let AgentDB;
  let db;

  beforeAll(async () => {
    // Load AgentDB browser bundle
    if (typeof window !== 'undefined') {
      // Browser environment - load from bundle
      AgentDB = window.AgentDB;
    } else {
      // Node environment - skip browser tests
      console.log('⚠️  Browser tests require browser environment');
      return;
    }

    if (!AgentDB) {
      throw new Error('AgentDB not loaded in browser');
    }
  });

  afterAll(async () => {
    if (db) {
      await db.close();
    }
  });

  describe('WASM Module Loading', () => {
    it('should load WASM attention modules', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        runtime: 'wasm' // Force WASM runtime
      });

      await db.initialize();

      // Verify WASM modules are loaded
      const runtime = db.getRuntime();
      expect(runtime.type).toBe('wasm');
      expect(runtime.wasmLoaded).toBe(true);
    });

    it('should lazy-load WASM on first attention query', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        lazyLoadWASM: true
      });

      await db.initialize();

      const runtime = db.getRuntime();
      expect(runtime.wasmLoaded).toBe(false);

      // First attention query triggers loading
      const controller = db.getController('self-attention');
      await controller.computeAttention([0.1, 0.2, 0.3]);

      expect(runtime.wasmLoaded).toBe(true);
    });

    it('should handle WASM initialization errors gracefully', async () => {
      // Mock WASM load failure
      const originalWebAssembly = window.WebAssembly;
      window.WebAssembly = undefined;

      try {
        db = new AgentDB({
          dbPath: ':memory:',
          enableAttention: true,
          runtime: 'wasm'
        });

        await expect(db.initialize()).rejects.toThrow('WASM not supported');
      } finally {
        window.WebAssembly = originalWebAssembly;
      }
    });

    it('should verify WASM memory limits', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        wasmMemoryLimit: 256 * 1024 * 1024 // 256MB
      });

      await db.initialize();

      const runtime = db.getRuntime();
      expect(runtime.memoryLimit).toBe(256 * 1024 * 1024);
    });
  });

  describe('Fallback Behavior', () => {
    it('should fallback to JavaScript when WASM unavailable', async () => {
      const originalWebAssembly = window.WebAssembly;
      window.WebAssembly = undefined;

      try {
        db = new AgentDB({
          dbPath: ':memory:',
          enableAttention: true,
          fallbackToJS: true
        });

        await db.initialize();

        const runtime = db.getRuntime();
        expect(runtime.type).toBe('javascript');

        // Should still work with JS fallback
        const controller = db.getController('self-attention');
        const result = await controller.computeAttention([0.1, 0.2, 0.3]);

        expect(result).toBeDefined();
      } finally {
        window.WebAssembly = originalWebAssembly;
      }
    });

    it('should detect and use native NAPI if available', async () => {
      // In browser, NAPI not available - should use WASM
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        preferNative: true
      });

      await db.initialize();

      const runtime = db.getRuntime();
      expect(runtime.type).toBe('wasm'); // NAPI not available in browser
    });

    it('should handle partial WASM support', async () => {
      // Simulate limited WASM features
      const originalCompile = WebAssembly.compile;
      WebAssembly.compile = () => Promise.reject(new Error('SIMD not supported'));

      try {
        db = new AgentDB({
          dbPath: ':memory:',
          enableAttention: true,
          fallbackToJS: true
        });

        await db.initialize();

        const runtime = db.getRuntime();
        expect(runtime.type).toBe('javascript');
      } finally {
        WebAssembly.compile = originalCompile;
      }
    });
  });

  describe('Cross-Browser Compatibility', () => {
    const browsers = [
      { name: 'Chrome', userAgent: 'Chrome/120.0.0.0' },
      { name: 'Firefox', userAgent: 'Firefox/120.0' },
      { name: 'Safari', userAgent: 'Safari/17.0' },
      { name: 'Edge', userAgent: 'Edg/120.0.0.0' }
    ];

    for (const browser of browsers) {
      it(`should work in ${browser.name}`, async function() {
        if (!window.navigator.userAgent.includes(browser.userAgent.split('/')[0])) {
          this.skip();
          return;
        }

        db = new AgentDB({
          dbPath: ':memory:',
          enableAttention: true
        });

        await db.initialize();

        const memoryController = db.getController('memory');
        await memoryController.store({
          id: 'browser-test',
          embedding: [0.1, 0.2, 0.3]
        });

        const controller = db.getController('self-attention');
        const result = await controller.computeAttention([0.1, 0.2, 0.3]);

        expect(result).toBeDefined();
        expect(result.scores).toBeDefined();
      });
    }

    it('should detect IndexedDB support for persistence', async () => {
      const hasIndexedDB = 'indexedDB' in window;

      if (hasIndexedDB) {
        db = new AgentDB({
          dbPath: 'agentdb-browser',
          storage: 'indexeddb',
          enableAttention: true
        });

        await db.initialize();

        const runtime = db.getRuntime();
        expect(runtime.storage).toBe('indexeddb');

        await db.close();

        // Verify persistence
        const db2 = new AgentDB({
          dbPath: 'agentdb-browser',
          storage: 'indexeddb'
        });

        await db2.initialize();
        await db2.close();
      } else {
        console.log('IndexedDB not available, skipping persistence test');
      }
    });

    it('should use Web Workers for parallel processing', async () => {
      if (!window.Worker) {
        console.log('Web Workers not available');
        return;
      }

      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        useWorkers: true,
        maxWorkers: 4
      });

      await db.initialize();

      const runtime = db.getRuntime();
      expect(runtime.workers).toBeGreaterThan(0);

      // Process queries in parallel
      const controller = db.getController('multi-head-attention');
      const queries = Array(10).fill(null).map(() =>
        [Math.random(), Math.random(), Math.random()]
      );

      const results = await Promise.all(
        queries.map(q => controller.computeMultiHeadAttention(q))
      );

      expect(results).toHaveLength(10);
    });
  });

  describe('Browser Performance', () => {
    it('should process attention queries efficiently in browser', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true
      });

      await db.initialize();

      const memoryController = db.getController('memory');
      const controller = db.getController('self-attention');

      // Store test data
      for (let i = 0; i < 100; i++) {
        await memoryController.store({
          id: `perf-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const start = performance.now();

      for (let i = 0; i < 50; i++) {
        await controller.computeAttention([Math.random(), Math.random(), Math.random()]);
      }

      const duration = performance.now() - start;

      // Should process 50 queries in reasonable time (browser may be slower than Node)
      expect(duration).toBeLessThan(5000); // 5 seconds
    });

    it('should manage memory efficiently in browser', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true
      });

      await db.initialize();

      const memoryController = db.getController('memory');

      // Monitor memory if available
      const initialMemory = performance.memory?.usedJSHeapSize || 0;

      // Store large dataset
      for (let i = 0; i < 1000; i++) {
        await memoryController.store({
          id: `mem-${i}`,
          embedding: Array(128).fill(0).map(() => Math.random())
        });
      }

      const finalMemory = performance.memory?.usedJSHeapSize || 0;

      if (finalMemory > 0) {
        const increase = (finalMemory - initialMemory) / (1024 * 1024);
        expect(increase).toBeLessThan(50); // Less than 50MB increase
      }
    });

    it('should handle offline mode', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        offlineMode: true
      });

      await db.initialize();

      // Simulate offline
      const originalOnLine = navigator.onLine;
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: false
      });

      try {
        // Should still work offline
        const controller = db.getController('self-attention');
        const result = await controller.computeAttention([0.1, 0.2, 0.3]);

        expect(result).toBeDefined();
      } finally {
        Object.defineProperty(navigator, 'onLine', {
          writable: true,
          value: originalOnLine
        });
      }
    });
  });

  describe('Progressive Enhancement', () => {
    it('should enable advanced features when available', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        progressiveEnhancement: true
      });

      await db.initialize();

      const features = db.getAvailableFeatures();

      // Check which features are available
      if (features.simd) {
        expect(features.attentionOptimized).toBe(true);
      }

      if (features.sharedArrayBuffer) {
        expect(features.parallelAttention).toBe(true);
      }

      if (features.webgl) {
        expect(features.gpuAcceleration).toBe(true);
      }
    });

    it('should provide feature detection API', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true
      });

      await db.initialize();

      const capabilities = db.getCapabilities();

      expect(capabilities).toHaveProperty('wasm');
      expect(capabilities).toHaveProperty('simd');
      expect(capabilities).toHaveProperty('threads');
      expect(capabilities).toHaveProperty('sharedArrayBuffer');
      expect(capabilities).toHaveProperty('webgl');
      expect(capabilities).toHaveProperty('indexeddb');
    });
  });

  describe('Bundle Size and Loading', () => {
    it('should lazy-load attention modules to reduce initial bundle', async () => {
      const initialSize = window.performance.getEntriesByType('resource')
        .find(r => r.name.includes('agentdb'))?.encodedBodySize || 0;

      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        lazyLoadModules: true
      });

      await db.initialize();

      // Attention modules not loaded yet
      const loadedModules = db.getLoadedModules();
      expect(loadedModules).not.toContain('self-attention');

      // Trigger module load
      const controller = db.getController('self-attention');
      await controller.computeAttention([0.1, 0.2, 0.3]);

      const finalLoadedModules = db.getLoadedModules();
      expect(finalLoadedModules).toContain('self-attention');
    });

    it('should support code splitting for attention modules', async () => {
      db = new AgentDB({
        dbPath: ':memory:',
        enableAttention: true,
        codeSplitting: true
      });

      await db.initialize();

      // Load only what's needed
      const selfAttnController = await db.lazyLoadController('self-attention');
      expect(selfAttnController).toBeDefined();

      // Other controllers not loaded yet
      const loadedControllers = db.getLoadedControllers();
      expect(loadedControllers).toContain('self-attention');
      expect(loadedControllers).not.toContain('cross-attention');
    });
  });
});

// Browser-specific utilities
if (typeof window !== 'undefined') {
  window.runAttentionTests = async function() {
    console.log('🧪 Running browser attention tests...');

    const results = await window.vitest.run();

    console.log('✅ Tests complete:', results);
    return results;
  };
}
