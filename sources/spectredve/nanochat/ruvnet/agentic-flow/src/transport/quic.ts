/**
 * QUIC Transport Layer for Agentic Flow
 *
 * High-performance QUIC protocol implementation using Rust/WASM
 * for ultra-low latency agent communication.
 *
 * Features:
 * - 0-RTT connection establishment (50-70% faster than TCP)
 * - Stream multiplexing (no head-of-line blocking)
 * - Connection pooling with automatic reuse
 * - TLS 1.3 encryption built-in
 *
 * @packageDocumentation
 */

import type { WasmQuicClient, ConnectionConfig, QuicMessage } from '../../crates/agentic-flow-quic/pkg';

/**
 * QUIC transport configuration
 */
export interface QuicTransportConfig {
  /** Server name for SNI (default: "localhost") */
  serverName?: string;
  /** Maximum idle timeout in milliseconds (min: 1000, default: 30000) */
  maxIdleTimeoutMs?: number;
  /** Maximum concurrent streams per connection (default: 100) */
  maxConcurrentStreams?: number;
  /** Enable 0-RTT connection resumption (default: true) */
  enable0Rtt?: boolean;
}

/**
 * Agent message structure
 */
export interface AgentMessage {
  /** Unique message ID */
  id: string;
  /** Message type */
  type: 'task' | 'result' | 'status' | 'coordination' | 'heartbeat' | string;
  /** Message payload (will be serialized) */
  payload: any;
  /** Optional metadata */
  metadata?: Record<string, any>;
}

/**
 * Connection pool statistics
 */
export interface PoolStatistics {
  /** Number of active connections */
  active: number;
  /** Number of idle connections */
  idle: number;
  /** Total connections created */
  created: number;
  /** Total connections closed */
  closed: number;
}

/**
 * QUIC Transport Client
 *
 * Provides high-level API for QUIC-based agent communication.
 *
 * @example
 * ```typescript
 * const transport = await QuicTransport.create({
 *   serverName: 'agent-proxy.local',
 *   enable0Rtt: true
 * });
 *
 * await transport.connect('127.0.0.1:4433');
 *
 * const message = {
 *   id: 'task-1',
 *   type: 'task',
 *   payload: { action: 'spawn', agentType: 'coder' }
 * };
 *
 * await transport.send('127.0.0.1:4433', message);
 * const response = await transport.receive('127.0.0.1:4433');
 *
 * console.log('Stats:', await transport.getStats());
 * ```
 */
export class QuicTransport {
  private wasmClient: WasmQuicClient | null = null;
  private config: Required<QuicTransportConfig>;

  private constructor(config: Required<QuicTransportConfig>) {
    this.config = config;
  }

  /**
   * Create new QUIC transport instance
   *
   * @param config - Transport configuration
   * @returns Promise resolving to QuicTransport instance
   */
  static async create(config: QuicTransportConfig = {}): Promise<QuicTransport> {
    // Load WASM module dynamically
    const wasmModule = await import('../../crates/agentic-flow-quic/pkg');

    // Default configuration
    const fullConfig: Required<QuicTransportConfig> = {
      serverName: config.serverName ?? 'localhost',
      maxIdleTimeoutMs: config.maxIdleTimeoutMs ?? 30000,
      maxConcurrentStreams: config.maxConcurrentStreams ?? 100,
      enable0Rtt: config.enable0Rtt ?? true,
    };

    // Validate configuration
    if (fullConfig.maxIdleTimeoutMs < 1000) {
      throw new Error('maxIdleTimeoutMs must be >= 1000ms');
    }

    if (fullConfig.maxConcurrentStreams === 0) {
      throw new Error('maxConcurrentStreams must be > 0');
    }

    const transport = new QuicTransport(fullConfig);

    // Create WASM config object
    const wasmConfig = wasmModule.defaultConfig();
    wasmConfig.server_name = fullConfig.serverName;
    wasmConfig.max_idle_timeout_ms = fullConfig.maxIdleTimeoutMs;
    wasmConfig.max_concurrent_streams = fullConfig.maxConcurrentStreams;
    wasmConfig.enable_0rtt = fullConfig.enable0Rtt;

    // Initialize WASM client
    transport.wasmClient = await wasmModule.WasmQuicClient.new(wasmConfig);

    return transport;
  }

  /**
   * Send message to server
   *
   * @param address - Server address (e.g., "127.0.0.1:4433")
   * @param message - Agent message to send
   */
  async send(address: string, message: AgentMessage): Promise<void> {
    if (!this.wasmClient) {
      throw new Error('Transport not initialized');
    }

    // Load WASM module for message creation
    const wasmModule = await import('../../crates/agentic-flow-quic/pkg');

    // Serialize payload
    const payloadBytes = new TextEncoder().encode(JSON.stringify(message.payload));

    // Create QUIC message
    const quicMessage = wasmModule.createQuicMessage(
      message.id,
      message.type,
      Array.from(payloadBytes),
      message.metadata ?? null
    );

    // Send via WASM client
    await this.wasmClient.sendMessage(address, quicMessage);
  }

  /**
   * Receive message from server
   *
   * @param address - Server address
   * @returns Promise resolving to received message
   */
  async receive(address: string): Promise<AgentMessage> {
    if (!this.wasmClient) {
      throw new Error('Transport not initialized');
    }

    const quicMessage = await this.wasmClient.recvMessage(address);

    // Deserialize payload
    const payloadStr = new TextDecoder().decode(
      new Uint8Array(quicMessage.payload)
    );
    const payload = JSON.parse(payloadStr);

    return {
      id: quicMessage.id,
      type: typeof quicMessage.msg_type === 'string'
        ? quicMessage.msg_type
        : Object.keys(quicMessage.msg_type)[0],
      payload,
      metadata: quicMessage.metadata,
    };
  }

  /**
   * Get connection pool statistics
   *
   * @returns Pool statistics
   */
  async getStats(): Promise<PoolStatistics> {
    if (!this.wasmClient) {
      throw new Error('Transport not initialized');
    }

    const stats = await this.wasmClient.poolStats();
    return {
      active: stats.active,
      idle: stats.idle,
      created: stats.created,
      closed: stats.closed,
    };
  }

  /**
   * Close all connections and cleanup resources
   */
  async close(): Promise<void> {
    if (this.wasmClient) {
      await this.wasmClient.close();
      this.wasmClient = null;
    }
  }

  /**
   * Convenience method: Connect and send message
   *
   * @param address - Server address
   * @param message - Message to send
   * @returns Response message
   */
  async request(address: string, message: AgentMessage): Promise<AgentMessage> {
    await this.send(address, message);
    return this.receive(address);
  }

  /**
   * Batch send multiple messages concurrently
   *
   * @param address - Server address
   * @param messages - Array of messages to send
   * @returns Promise resolving when all messages are sent
   */
  async sendBatch(address: string, messages: AgentMessage[]): Promise<void> {
    await Promise.all(messages.map(msg => this.send(address, msg)));
  }
}

/**
 * Create QUIC transport with default configuration
 *
 * @returns Promise resolving to QuicTransport instance
 */
export async function createQuicTransport(
  config?: QuicTransportConfig
): Promise<QuicTransport> {
  return QuicTransport.create(config);
}

export default QuicTransport;
