import { assertEquals } from "https://deno.land/std@0.224.0/testing/asserts.ts";
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { FederationProxy } from "../packages/proxy/federation.ts";
import { FederationConfig } from "../packages/core/types.ts";
import { JSONRPCRequest, JSONRPCResponse } from "../packages/core/schema.ts";

const TEST_SECRET = "test-secret-key";
const WS_PORT = 3002;

// Mock task data
const TASK_REQUEST: JSONRPCRequest = {
  jsonrpc: "2.0",
  method: "executeTask",
  params: {
    type: "calculation",
    input: {
      numbers: [1, 2, 3, 4, 5],
      operation: "sum"
    }
  },
  id: 1
};

const EXPECTED_RESPONSE: JSONRPCResponse = {
  jsonrpc: "2.0",
  result: {
    output: 15, // sum of [1,2,3,4,5]
    status: "completed"
  },
  id: 1
};

async function setupMockTaskServer() {
  const ac = new AbortController();
  const { signal } = ac;

  const handler = async (req: Request): Promise<Response> => {
    if (req.headers.get("upgrade") === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      
      socket.onopen = () => {
        console.log("Task Server WebSocket connected");
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.method === "executeTask") {
            // Simulate task execution and send response
            socket.send(JSON.stringify(EXPECTED_RESPONSE));
          }
        } catch (error) {
          console.error("Error handling message:", error);
        }
      };

      return response;
    }
    return new Response("Not a websocket request", { status: 400 });
  };

  const serverPromise = serve(handler, { port: WS_PORT, signal });
  await new Promise(resolve => setTimeout(resolve, 100));

  return {
    close: () => {
      ac.abort();
    }
  };
}

Deno.test({
  name: "Simple Task Workflow",
  async fn() {
    const mockServer = await setupMockTaskServer();
    try {
      const proxy = new FederationProxy(TEST_SECRET);
      
      // Configure and register a task execution server
      const config: FederationConfig = {
        serverId: "task-server",
        endpoints: {
          control: `ws://localhost:${WS_PORT}`,
          data: "http://localhost:3001",
        },
        auth: {
          type: "jwt",
          config: { secret: TEST_SECRET }
        }
      };

      // Register the server
      await proxy.registerServer(config);
      
      // Get the WebSocket connection
      const connection = proxy["connections"].get("task-server");
      assertEquals(!!connection, true, "WebSocket connection should be established");

      // Create a promise to wait for the response
      const responsePromise = new Promise<JSONRPCResponse>((resolve) => {
        if (connection) {
          connection.onmessage = (event) => {
            const response = JSON.parse(event.data);
            resolve(response);
          };
        }
      });

      // Send the task request
      if (connection) {
        connection.send(JSON.stringify(TASK_REQUEST));
      }

      // Wait for and verify the response
      const response = await responsePromise;
      assertEquals(response, EXPECTED_RESPONSE, "Task response should match expected output");

    } finally {
      mockServer.close();
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  },
  sanitizeResources: false,
  sanitizeOps: false
});
