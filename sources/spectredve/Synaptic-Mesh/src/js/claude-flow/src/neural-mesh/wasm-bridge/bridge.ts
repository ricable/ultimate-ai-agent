/**
 * WASM Integration Bridge
 * TypeScript bindings for Rust WASM modules with memory management
 */

import { EventEmitter } from 'events';

// Type definitions for WASM integration
export interface WasmModule {
  id: string;
  name: string;
  version: string;
  instance: WebAssembly.Instance;
  memory: WebAssembly.Memory;
  exports: Record<string, Function>;
  size: number;
  loaded: boolean;
  loadedAt: number;
}

export interface WasmBridgeConfig {
  memorySize: number;
  modules: string[];
  enableSIMD?: boolean;
  enableSharedMemory?: boolean;
  debugMode?: boolean;
}

export interface MemoryLayout {
  baseAddress: number;
  size: number;
  type: 'heap' | 'stack' | 'shared' | 'neural';
  owner?: string;
  allocated: boolean;
}

export interface NeuralData {
  type: 'f32' | 'f64' | 'i32' | 'i64';
  shape: number[];
  data: ArrayBuffer;
  stride?: number[];
}

export class WasmBridge extends EventEmitter {
  private modules: Map<string, WasmModule> = new Map();
  private memoryAllocations: Map<number, MemoryLayout> = new Map();
  private nextAllocationId: number = 1;
  private globalMemory: WebAssembly.Memory | null = null;
  private isInitialized: boolean = false;
  private config: WasmBridgeConfig | null = null;
  private debugMode: boolean = false;

  constructor() {
    super();
    this.setupErrorHandling();
  }

  /**
   * Initialize the WASM bridge
   */
  async initialize(config: WasmBridgeConfig): Promise<void> {
    console.log('🦀 Initializing WASM Bridge...');
    
    this.config = config;
    this.debugMode = config.debugMode || false;
    
    // Create shared memory if supported
    if (config.enableSharedMemory && this.isSharedMemorySupported()) {
      this.globalMemory = new WebAssembly.Memory({
        initial: Math.ceil(config.memorySize / 65536),
        maximum: Math.ceil(config.memorySize / 65536) * 2,
        shared: true
      });
      
      this.log('✅ Shared memory initialized');
    } else {
      this.globalMemory = new WebAssembly.Memory({
        initial: Math.ceil(config.memorySize / 65536),
        maximum: Math.ceil(config.memorySize / 65536) * 2
      });
      
      this.log('✅ Linear memory initialized');
    }

    // Load requested modules
    if (config.modules && config.modules.length > 0) {
      for (const modulePath of config.modules) {
        try {
          await this.loadModule(modulePath);
        } catch (error) {
          console.error(`Failed to load module ${modulePath}:`, error);
        }
      }
    }

    this.isInitialized = true;
    this.emit('initialized', { modules: this.modules.size, memorySize: config.memorySize });
    
    console.log('✅ WASM Bridge initialized successfully');
  }

  /**
   * Load a WASM module
   */
  async loadModule(modulePath: string, initParams?: any): Promise<WasmModule> {
    if (!this.isInitialized) {
      throw new Error('WASM Bridge not initialized');
    }

    this.log(`Loading WASM module: ${modulePath}`);

    try {
      // Fetch WASM module
      const response = await fetch(modulePath);
      if (!response.ok) {
        throw new Error(`Failed to fetch module: ${response.statusText}`);
      }

      const wasmBytes = await response.arrayBuffer();
      
      // Compile module
      const wasmModule = await WebAssembly.compile(wasmBytes);
      
      // Create import object with memory and required functions
      const importObject = this.createImportObject(initParams);
      
      // Instantiate module
      const instance = await WebAssembly.instantiate(wasmModule, importObject);
      
      // Create module descriptor
      const moduleId = this.generateModuleId(modulePath);
      const module: WasmModule = {
        id: moduleId,
        name: this.extractModuleName(modulePath),
        version: '1.0.0', // Could be extracted from module metadata
        instance,
        memory: this.globalMemory!,
        exports: instance.exports as Record<string, Function>,
        size: wasmBytes.byteLength,
        loaded: true,
        loadedAt: Date.now()
      };

      // Initialize module if it has an init function
      if ('init' in instance.exports && typeof instance.exports.init === 'function') {
        try {
          (instance.exports.init as Function)();
          this.log(`✅ Module ${moduleId} initialized`);
        } catch (error) {
          console.warn(`Module ${moduleId} init failed:`, error);
        }
      }

      this.modules.set(moduleId, module);
      this.emit('moduleLoaded', { module });
      
      this.log(`✅ Module ${moduleId} loaded successfully`);
      
      return module;

    } catch (error) {
      const errorMsg = `Failed to load WASM module ${modulePath}: ${error.message}`;
      this.log(`❌ ${errorMsg}`);
      throw new Error(errorMsg);
    }
  }

  /**
   * Execute a function in a WASM module
   */
  async executeWasmFunction(
    moduleId: string, 
    functionName: string, 
    args: any[] = []
  ): Promise<any> {
    const module = this.modules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not found`);
    }

    const func = module.exports[functionName];
    if (!func || typeof func !== 'function') {
      throw new Error(`Function ${functionName} not found in module ${moduleId}`);
    }

    try {
      this.log(`Executing ${moduleId}.${functionName} with args:`, args);
      
      const result = func(...args);
      
      this.log(`✅ Function ${functionName} completed`);
      this.emit('functionExecuted', { moduleId, functionName, args, result });
      
      return result;
    } catch (error) {
      const errorMsg = `Function execution failed: ${error.message}`;
      this.log(`❌ ${errorMsg}`);
      this.emit('functionError', { moduleId, functionName, error });
      throw new Error(errorMsg);
    }
  }

  /**
   * Allocate memory for neural data
   */
  allocateMemory(size: number, type: 'heap' | 'stack' | 'shared' | 'neural' = 'heap'): number {
    if (!this.globalMemory) {
      throw new Error('Memory not initialized');
    }

    // Find a suitable memory address
    const baseAddress = this.findAvailableMemorySlot(size);
    
    const layout: MemoryLayout = {
      baseAddress,
      size,
      type,
      allocated: true
    };

    const allocationId = this.nextAllocationId++;
    this.memoryAllocations.set(allocationId, layout);
    
    this.log(`📦 Allocated ${size} bytes at address ${baseAddress} (ID: ${allocationId})`);
    this.emit('memoryAllocated', { allocationId, layout });
    
    return allocationId;
  }

  /**
   * Deallocate memory
   */
  deallocateMemory(allocationId: number): void {
    const layout = this.memoryAllocations.get(allocationId);
    if (!layout) {
      throw new Error(`Memory allocation ${allocationId} not found`);
    }

    layout.allocated = false;
    this.memoryAllocations.delete(allocationId);
    
    this.log(`🗑️ Deallocated memory ID ${allocationId}`);
    this.emit('memoryDeallocated', { allocationId });
  }

  /**
   * Write neural data to WASM memory
   */
  writeNeuralData(allocationId: number, data: NeuralData): void {
    const layout = this.memoryAllocations.get(allocationId);
    if (!layout || !layout.allocated) {
      throw new Error(`Invalid memory allocation: ${allocationId}`);
    }

    if (data.data.byteLength > layout.size) {
      throw new Error(`Data size (${data.data.byteLength}) exceeds allocated memory (${layout.size})`);
    }

    const memoryView = new Uint8Array(this.globalMemory!.buffer);
    const dataView = new Uint8Array(data.data);
    
    memoryView.set(dataView, layout.baseAddress);
    
    this.log(`📝 Wrote ${data.data.byteLength} bytes of neural data to allocation ${allocationId}`);
    this.emit('dataWritten', { allocationId, data });
  }

  /**
   * Read neural data from WASM memory
   */
  readNeuralData(allocationId: number, size?: number): ArrayBuffer {
    const layout = this.memoryAllocations.get(allocationId);
    if (!layout || !layout.allocated) {
      throw new Error(`Invalid memory allocation: ${allocationId}`);
    }

    const readSize = size || layout.size;
    const memoryView = new Uint8Array(this.globalMemory!.buffer);
    const data = memoryView.slice(layout.baseAddress, layout.baseAddress + readSize);
    
    this.log(`📖 Read ${readSize} bytes from allocation ${allocationId}`);
    this.emit('dataRead', { allocationId, size: readSize });
    
    return data.buffer;
  }

  /**
   * Transfer data between JavaScript and WASM
   */
  async transferData(
    moduleId: string,
    operation: 'js_to_wasm' | 'wasm_to_js',
    data: any,
    format?: string
  ): Promise<any> {
    const module = this.modules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not found`);
    }

    switch (operation) {
      case 'js_to_wasm':
        return this.transferJSToWasm(module, data, format);
      case 'wasm_to_js':
        return this.transferWasmToJS(module, data, format);
      default:
        throw new Error(`Unknown transfer operation: ${operation}`);
    }
  }

  /**
   * Get loaded modules
   */
  async getLoadedModules(): Promise<Array<{name: string, version: string, size: number, loaded: boolean}>> {
    return Array.from(this.modules.values()).map(module => ({
      name: module.name,
      version: module.version,
      size: Math.round(module.size / 1024), // Size in KB
      loaded: module.loaded
    }));
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): any {
    const totalAllocations = this.memoryAllocations.size;
    const totalAllocatedBytes = Array.from(this.memoryAllocations.values())
      .reduce((sum, layout) => sum + layout.size, 0);
    
    const memoryByType = Array.from(this.memoryAllocations.values())
      .reduce((acc, layout) => {
        acc[layout.type] = (acc[layout.type] || 0) + layout.size;
        return acc;
      }, {} as Record<string, number>);

    return {
      totalAllocations,
      totalAllocatedBytes,
      memoryByType,
      memoryBuffer: this.globalMemory?.buffer.byteLength || 0,
      utilizationPercent: this.globalMemory ? 
        (totalAllocatedBytes / this.globalMemory.buffer.byteLength) * 100 : 0
    };
  }

  /**
   * Cleanup and shutdown
   */
  async cleanup(): Promise<void> {
    this.log('🧹 Cleaning up WASM Bridge...');
    
    // Cleanup all memory allocations
    for (const allocationId of this.memoryAllocations.keys()) {
      this.deallocateMemory(allocationId);
    }
    
    // Clear modules
    this.modules.clear();
    
    this.isInitialized = false;
    this.emit('cleanup');
    
    console.log('✅ WASM Bridge cleanup completed');
  }

  // Private helper methods
  private createImportObject(initParams?: any): any {
    return {
      env: {
        memory: this.globalMemory,
        // Add common WASM imports
        abort: (msg: number, file: number, line: number, column: number) => {
          console.error('WASM abort:', msg, file, line, column);
        },
        trace: (msg: number) => {
          this.log(`WASM trace: ${msg}`);
        },
        // Math functions
        Math_random: Math.random,
        Math_floor: Math.floor,
        Math_ceil: Math.ceil,
        Math_sqrt: Math.sqrt,
        // Console functions
        console_log: (ptr: number) => {
          const message = this.readStringFromMemory(ptr);
          console.log(`WASM: ${message}`);
        }
      },
      ...(initParams || {})
    };
  }

  private generateModuleId(modulePath: string): string {
    const name = this.extractModuleName(modulePath);
    const timestamp = Date.now().toString(36);
    return `${name}_${timestamp}`;
  }

  private extractModuleName(modulePath: string): string {
    return modulePath.split('/').pop()?.replace(/\.wasm$/, '') || 'unknown';
  }

  private findAvailableMemorySlot(size: number): number {
    if (!this.globalMemory) {
      throw new Error('Memory not initialized');
    }

    // Simple first-fit allocation strategy
    const allocations = Array.from(this.memoryAllocations.values())
      .sort((a, b) => a.baseAddress - b.baseAddress);
    
    let currentAddress = 0;
    
    for (const allocation of allocations) {
      if (allocation.baseAddress - currentAddress >= size) {
        return currentAddress;
      }
      currentAddress = allocation.baseAddress + allocation.size;
    }
    
    // Check if we have space at the end
    if (this.globalMemory.buffer.byteLength - currentAddress >= size) {
      return currentAddress;
    }
    
    throw new Error(`Insufficient memory: need ${size} bytes`);
  }

  private isSharedMemorySupported(): boolean {
    try {
      return typeof SharedArrayBuffer !== 'undefined' && 
             typeof Atomics !== 'undefined';
    } catch {
      return false;
    }
  }

  private transferJSToWasm(module: WasmModule, data: any, format?: string): any {
    // Implementation for JS to WASM data transfer
    // This would handle type conversion and memory management
    return data;
  }

  private transferWasmToJS(module: WasmModule, data: any, format?: string): any {
    // Implementation for WASM to JS data transfer
    // This would handle type conversion and memory management
    return data;
  }

  private readStringFromMemory(ptr: number): string {
    if (!this.globalMemory) return '';
    
    const memory = new Uint8Array(this.globalMemory.buffer);
    let length = 0;
    
    // Find string length (null-terminated)
    while (memory[ptr + length] !== 0) {
      length++;
    }
    
    const stringBytes = memory.slice(ptr, ptr + length);
    return new TextDecoder().decode(stringBytes);
  }

  private setupErrorHandling(): void {
    this.on('error', (error) => {
      console.error('WASM Bridge Error:', error);
    });
  }

  private log(message: string, ...args: any[]): void {
    if (this.debugMode) {
      console.log(`[WASM Bridge] ${message}`, ...args);
    }
  }
}

export default WasmBridge;