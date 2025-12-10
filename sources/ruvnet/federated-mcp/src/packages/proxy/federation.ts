import { FederationConfig } from '../core/types.ts';
import { AuthManager } from '../core/auth.ts';
import { JSONRPCMessage, JSONRPCRequest, JSONRPCResponse } from '../core/schema.ts';

export class FederationProxy {
  private servers: Map<string, FederationConfig>;
  private authManager: AuthManager;
  private connections: Map<string, WebSocket>;

  constructor(secret: string) {
    this.servers = new Map();
    this.connections = new Map();
    this.authManager = new AuthManager(secret);
  }

  async registerServer(config: FederationConfig): Promise<void> {
    this.servers.set(config.serverId, config);
    await this.establishConnection(config);
  }

  async removeServer(serverId: string): Promise<void> {
    const connection = this.connections.get(serverId);
    if (connection) {
      connection.close();
      this.connections.delete(serverId);
    }
    this.servers.delete(serverId);
  }

  private async establishConnection(config: FederationConfig): Promise<void> {
    try {
      const token = await this.authManager.createToken({
        serverId: config.serverId,
        type: 'federation'
      });

      // Append token as query parameter
      const wsUrl = new URL(config.endpoints.control);
      wsUrl.searchParams.set('token', token);
      
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(wsUrl.toString());

        ws.onopen = () => {
          console.log(`Connected to server ${config.serverId}`);
          this.connections.set(config.serverId, ws);
          resolve();
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as JSONRPCMessage;
            this.handleMessage(config.serverId, message);
          } catch (error) {
            console.error(`Failed to parse message from ${config.serverId}:`, error);
          }
        };

        ws.onclose = () => {
          console.log(`Disconnected from server ${config.serverId}`);
          this.connections.delete(config.serverId);
        };

        ws.onerror = (error) => {
          console.error(`Error with server ${config.serverId}:`, error);
          reject(error);
        };

        // Set a connection timeout
        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('Connection timeout'));
        }, 5000);

        // Clear timeout on successful connection
        ws.addEventListener('open', () => clearTimeout(timeout));
      });

    } catch (error) {
      console.error(`Failed to establish connection with ${config.serverId}:`, error);
      throw error;
    }
  }

  private handleMessage(serverId: string, message: JSONRPCMessage): void {
    // Implement message handling logic based on the JSON-RPC message type
    console.log(`Received message from ${serverId}:`, message);
  }

  getConnectedServers(): string[] {
    return Array.from(this.connections.keys());
  }
}
