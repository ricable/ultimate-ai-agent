import { WASI } from '@wasmer/wasi';
import { WasmFs } from '@wasmer/wasmfs';
import WebAssembly from 'webassembly';
import { EventEmitter } from 'events';
import { Logger } from './utils/Logger';
import { EdgeStorage } from './storage/EdgeStorage';
import { MessageQueue } from './messaging/MessageQueue';
import { SecurityManager } from './security/SecurityManager';

export interface EdgeModuleConfig {
  name: string;
  wasmPath: string;
  memory?: {
    initial: number;
    maximum?: number;
  };
  env?: Record<string, string>;
  permissions?: {
    fileSystem: boolean;
    network: boolean;
    compute: boolean;
  };
}

export interface EdgeExecutionContext {
  moduleId: string;
  requestId: string;
  userId?: string;
  timeout: number;
  resources: {
    maxMemory: number;
    maxCpuTime: number;
  };
}

export class EdgeRuntime extends EventEmitter {
  private modules: Map<string, WebAssembly.Module> = new Map();
  private instances: Map<string, WebAssembly.Instance> = new Map();
  private wasmFs: WasmFs;
  private storage: EdgeStorage;
  private messageQueue: MessageQueue;
  private securityManager: SecurityManager;
  private isInitialized = false;

  constructor() {
    super();
    this.wasmFs = new WasmFs();
    this.storage = new EdgeStorage();
    this.messageQueue = new MessageQueue();
    this.securityManager = new SecurityManager();
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    Logger.info('EdgeRuntime', 'Initializing WebAssembly edge runtime');

    try {
      await this.storage.initialize();
      await this.messageQueue.initialize();
      await this.securityManager.initialize();
      
      this.isInitialized = true;
      this.emit('initialized');
      Logger.info('EdgeRuntime', 'Edge runtime initialized successfully');
    } catch (error) {
      Logger.error('EdgeRuntime', 'Failed to initialize edge runtime', error);
      throw error;
    }
  }

  async loadModule(config: EdgeModuleConfig): Promise<void> {
    Logger.info('EdgeRuntime', `Loading WebAssembly module: ${config.name}`);

    try {
      // Security validation
      await this.securityManager.validateModule(config);

      // Load WebAssembly binary
      const wasmBuffer = await this.loadWasmBinary(config.wasmPath);
      const module = await WebAssembly.compile(wasmBuffer);

      this.modules.set(config.name, module);
      Logger.info('EdgeRuntime', `Module ${config.name} loaded successfully`);
      
      this.emit('moduleLoaded', { name: config.name, module });
    } catch (error) {
      Logger.error('EdgeRuntime', `Failed to load module ${config.name}`, error);
      throw error;
    }
  }

  async createInstance(
    moduleName: string,
    context: EdgeExecutionContext
  ): Promise<string> {
    const module = this.modules.get(moduleName);
    if (!module) {
      throw new Error(`Module ${moduleName} not found`);
    }

    Logger.info('EdgeRuntime', `Creating instance for module: ${moduleName}`);

    try {
      // Create WASI instance
      const wasi = new WASI({
        env: context.userId ? { USER_ID: context.userId } : {},
        args: [],
        preopens: {
          '/tmp': '/tmp',
          '/data': '/data',
        },
      });

      // Create WebAssembly instance with imports
      const imports = {
        wasi_snapshot_preview1: wasi.wasiImport,
        env: {
          memory: new WebAssembly.Memory({
            initial: 256, // 16MB
            maximum: 1024, // 64MB
            shared: false,
          }),
          ...this.createRuntimeImports(context),
        },
      };

      const instance = await WebAssembly.instantiate(module, imports);
      const instanceId = `${moduleName}_${context.requestId}`;
      
      this.instances.set(instanceId, instance);
      
      // Initialize WASI
      wasi.start(instance);
      
      Logger.info('EdgeRuntime', `Instance created: ${instanceId}`);
      return instanceId;
    } catch (error) {
      Logger.error('EdgeRuntime', `Failed to create instance for ${moduleName}`, error);
      throw error;
    }
  }

  async executeFunction(
    instanceId: string,
    functionName: string,
    args: any[] = [],
    context: EdgeExecutionContext
  ): Promise<any> {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      throw new Error(`Instance ${instanceId} not found`);
    }

    Logger.info('EdgeRuntime', `Executing function: ${functionName} on ${instanceId}`);

    try {
      const startTime = Date.now();
      
      // Security check
      await this.securityManager.validateExecution(instanceId, functionName, context);

      // Get the exported function
      const exports = instance.exports as any;
      const func = exports[functionName];
      
      if (!func || typeof func !== 'function') {
        throw new Error(`Function ${functionName} not found or not callable`);
      }

      // Execute with timeout
      const result = await this.executeWithTimeout(
        () => func(...args),
        context.timeout
      );

      const executionTime = Date.now() - startTime;
      Logger.info('EdgeRuntime', `Function executed in ${executionTime}ms`);
      
      this.emit('functionExecuted', {
        instanceId,
        functionName,
        executionTime,
        success: true,
      });

      return result;
    } catch (error) {
      Logger.error('EdgeRuntime', `Function execution failed: ${functionName}`, error);
      
      this.emit('functionExecuted', {
        instanceId,
        functionName,
        success: false,
        error: error.message,
      });
      
      throw error;
    }
  }

  async destroyInstance(instanceId: string): Promise<void> {
    Logger.info('EdgeRuntime', `Destroying instance: ${instanceId}`);
    
    if (this.instances.has(instanceId)) {
      this.instances.delete(instanceId);
      this.emit('instanceDestroyed', { instanceId });
    }
  }

  async getInstanceStats(instanceId: string): Promise<any> {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      throw new Error(`Instance ${instanceId} not found`);
    }

    const exports = instance.exports as any;
    const memory = exports.memory as WebAssembly.Memory;
    
    return {
      instanceId,
      memorySize: memory ? memory.buffer.byteLength : 0,
      memoryPages: memory ? memory.buffer.byteLength / 65536 : 0,
      exports: Object.keys(exports),
    };
  }

  async listModules(): Promise<string[]> {
    return Array.from(this.modules.keys());
  }

  async listInstances(): Promise<string[]> {
    return Array.from(this.instances.keys());
  }

  private async loadWasmBinary(path: string): Promise<ArrayBuffer> {
    // In a real implementation, this would load from file system or URL
    // For now, we'll simulate loading
    Logger.info('EdgeRuntime', `Loading WASM binary from: ${path}`);
    
    // Placeholder - in production this would use fetch() or fs.readFile()
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to load WASM binary: ${response.statusText}`);
    }
    
    return response.arrayBuffer();
  }

  private createRuntimeImports(context: EdgeExecutionContext): Record<string, any> {
    return {
      // Logging functions
      log: (ptr: number, len: number) => {
        const memory = this.getMemoryFromContext(context);
        const message = this.readStringFromMemory(memory, ptr, len);
        Logger.info('WASM', message);
      },
      
      error: (ptr: number, len: number) => {
        const memory = this.getMemoryFromContext(context);
        const message = this.readStringFromMemory(memory, ptr, len);
        Logger.error('WASM', message);
      },

      // Time functions
      now: () => Date.now(),
      
      // Random number generation
      random: () => Math.random(),
      
      // Storage functions
      storage_get: async (keyPtr: number, keyLen: number) => {
        const memory = this.getMemoryFromContext(context);
        const key = this.readStringFromMemory(memory, keyPtr, keyLen);
        return this.storage.get(key);
      },
      
      storage_set: async (keyPtr: number, keyLen: number, valuePtr: number, valueLen: number) => {
        const memory = this.getMemoryFromContext(context);
        const key = this.readStringFromMemory(memory, keyPtr, keyLen);
        const value = this.readStringFromMemory(memory, valuePtr, valueLen);
        return this.storage.set(key, value);
      },
    };
  }

  private getMemoryFromContext(context: EdgeExecutionContext): WebAssembly.Memory {
    // This would get memory from the current execution context
    // Simplified implementation
    return new WebAssembly.Memory({ initial: 1 });
  }

  private readStringFromMemory(memory: WebAssembly.Memory, ptr: number, len: number): string {
    const buffer = new Uint8Array(memory.buffer, ptr, len);
    return new TextDecoder().decode(buffer);
  }

  private async executeWithTimeout<T>(fn: () => T | Promise<T>, timeout: number): Promise<T> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Function execution timed out after ${timeout}ms`));
      }, timeout);

      Promise.resolve(fn())
        .then(result => {
          clearTimeout(timer);
          resolve(result);
        })
        .catch(error => {
          clearTimeout(timer);
          reject(error);
        });
    });
  }

  async shutdown(): Promise<void> {
    Logger.info('EdgeRuntime', 'Shutting down edge runtime');
    
    // Destroy all instances
    for (const instanceId of this.instances.keys()) {
      await this.destroyInstance(instanceId);
    }
    
    // Clear modules
    this.modules.clear();
    
    // Shutdown services
    await this.storage.shutdown();
    await this.messageQueue.shutdown();
    await this.securityManager.shutdown();
    
    this.isInitialized = false;
    this.emit('shutdown');
    Logger.info('EdgeRuntime', 'Edge runtime shut down');
  }
}