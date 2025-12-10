import { ServerInfo } from '../packages/core/types.ts';
// Import types
import type {
  KVNamespace,
  DurableObjectNamespace,
  DurableObjectState,
  DurableObject,
  CloudflareWebSocket
} from './cloudflare.d.ts';
// Import value
import { WebSocketPair } from './cloudflare.d.ts';

interface Env {
  MCP_STORE: KVNamespace;
  CONNECTIONS: DurableObjectNamespace;
}

const serverInfo: ServerInfo = {
  name: "cloudflare-mcp-worker",
  version: "1.0.0",
  capabilities: {
    models: ["gpt-3.5-turbo", "gpt-4"],
    protocols: ["json-rpc"],
    features: ["task-execution", "federation"]
  }
};

export class ConnectionsStore implements DurableObject {
  private sessions: Map<string, CloudflareWebSocket>;
  private state: DurableObjectState;

  constructor(state: DurableObjectState) {
    this.state = state;
    this.sessions = new Map();
  }

  async fetch(request: Request): Promise<Response> {
    if (request.headers.get("Upgrade") === "websocket") {
      const pair = new WebSocketPair();
      const [client, server] = [pair[0], pair[1]] as [CloudflareWebSocket, CloudflareWebSocket];
      
      await this.handleSession(server);
      
      return new Response(null, {
        status: 101,
        webSocket: client,
      });
    }

    return new Response("Expected WebSocket", { status: 400 });
  }

  private async handleSession(ws: CloudflareWebSocket): Promise<void> {
    const sessionId = crypto.randomUUID();
    this.sessions.set(sessionId, ws);

    ws.accept();

    // Send initial server info
    ws.send(JSON.stringify({
      type: "info",
      data: serverInfo
    }));

    ws.addEventListener("message", async (msg: MessageEvent) => {
      try {
        const data = JSON.parse(msg.data as string);
        await this.handleMessage(sessionId, data);
      } catch (err) {
        ws.send(JSON.stringify({
          type: "error",
          error: "Invalid message format"
        }));
      }
    });

    ws.addEventListener("close", () => {
      this.sessions.delete(sessionId);
    });
  }

  private async handleMessage(sessionId: string, message: any): Promise<void> {
    const ws = this.sessions.get(sessionId);
    if (!ws) return;

    switch (message.type) {
      case "info":
        ws.send(JSON.stringify({
          type: "info",
          data: serverInfo
        }));
        break;

      case "capabilities":
        ws.send(JSON.stringify({
          type: "capabilities",
          data: serverInfo.capabilities
        }));
        break;

      case "task":
        try {
          const result = await this.processTask(message.data);
          ws.send(JSON.stringify({
            type: "task_result",
            data: result
          }));
        } catch (error) {
          ws.send(JSON.stringify({
            type: "error",
            error: error instanceof Error ? error.message : "Unknown error"
          }));
        }
        break;

      default:
        ws.send(JSON.stringify({
          type: "error",
          error: `Unknown message type: ${message.type}`
        }));
    }
  }

  private async processTask(task: any): Promise<Record<string, unknown>> {
    // Store task in Durable Object storage
    const taskId = crypto.randomUUID();
    await this.state.storage.put(`task:${taskId}`, JSON.stringify({
      ...task,
      timestamp: Date.now()
    }));

    return {
      status: "completed",
      taskId,
      result: "Task processed successfully"
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      // Handle WebSocket upgrade
      if (request.headers.get("Upgrade") === "websocket") {
        // Create or get the Durable Object for connection handling
        const id = env.CONNECTIONS.idFromName("default");
        const connectionStore = env.CONNECTIONS.get(id);
        return connectionStore.fetch(request);
      }

      // Handle HTTP requests
      const url = new URL(request.url);
      
      switch (url.pathname) {
        case "/info":
          return new Response(JSON.stringify(serverInfo), {
            headers: { "Content-Type": "application/json" }
          });

        case "/capabilities":
          return new Response(JSON.stringify(serverInfo.capabilities), {
            headers: { "Content-Type": "application/json" }
          });

        default:
          return new Response("Not Found", { status: 404 });
      }
    } catch (error) {
      return new Response(error instanceof Error ? error.message : "Internal Server Error", { 
        status: 500 
      });
    }
  }
};
