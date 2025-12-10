declare global {
  interface KVNamespaceGetOptions<Type> {
    type: Type;
    cacheTtl?: number;
  }

  interface KVNamespacePutOptions {
    expiration?: number;
    expirationTtl?: number;
  }

  interface KVNamespace {
    get(key: string, options?: Partial<KVNamespaceGetOptions<undefined>>): Promise<string | null>;
    put(key: string, value: string | ReadableStream | ArrayBuffer, options?: KVNamespacePutOptions): Promise<void>;
    delete(key: string): Promise<void>;
  }

  interface DurableObjectNamespace {
    newUniqueId(): DurableObjectId;
    idFromName(name: string): DurableObjectId;
    get(id: DurableObjectId): DurableObjectInstance;
  }

  interface DurableObjectId {
    toString(): string;
    equals(other: DurableObjectId): boolean;
  }

  interface DurableObjectState {
    waitUntil(promise: Promise<any>): void;
    blockConcurrencyWhile<T>(callback: () => Promise<T>): Promise<T>;
    storage: DurableObjectStorage;
  }

  interface DurableObjectStorage {
    get<T = any>(key: string): Promise<T | undefined>;
    get<T = any>(keys: string[]): Promise<Map<string, T>>;
    put<T>(key: string, value: T): Promise<void>;
    put<T>(entries: Record<string, T>): Promise<void>;
    delete(key: string): Promise<boolean>;
    delete(keys: string[]): Promise<number>;
  }

  interface DurableObjectInstance {
    fetch(request: Request): Promise<Response>;
  }

  interface DurableObject {
    fetch(request: Request): Promise<Response>;
  }

  interface CloudflareWebSocket extends WebSocket {
    accept(): void;
  }

  interface ResponseInit {
    webSocket?: WebSocket;
  }
}

export type {
  KVNamespace,
  KVNamespaceGetOptions,
  KVNamespacePutOptions,
  DurableObjectNamespace,
  DurableObjectId,
  DurableObjectState,
  DurableObjectStorage,
  DurableObjectInstance,
  DurableObject,
  CloudflareWebSocket
};
