import { assertEquals } from "https://deno.land/std@0.224.0/testing/asserts.ts";
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { FederationProxy } from "../packages/proxy/federation.ts";
import { FederationConfig } from "../packages/core/types.ts";

const TEST_SECRET = "test-secret-key";
const WS_PORT = 3000;

async function setupMockServer() {
  const ac = new AbortController();
  const { signal } = ac;

  const handler = async (req: Request): Promise<Response> => {
    if (req.headers.get("upgrade") === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.onopen = () => {
        console.log("WebSocket connected");
      };
      socket.onclose = () => {
        console.log("WebSocket closed");
      };
      return response;
    }
    return new Response("Not a websocket request", { status: 400 });
  };

  // Start the server and wait for it to be ready
  const serverPromise = serve(handler, { port: WS_PORT, signal });
  
  // Wait for the server to be ready
  await new Promise(resolve => setTimeout(resolve, 100));

  return {
    close: () => {
      ac.abort();
    }
  };
}

Deno.test({
  name: "Federation Proxy - Server Registration",
  async fn() {
    const mockServer = await setupMockServer();
    try {
      const proxy = new FederationProxy(TEST_SECRET);
      
      const config: FederationConfig = {
        serverId: "test-server",
        endpoints: {
          control: `ws://localhost:${WS_PORT}`,
          data: "http://localhost:3001",
        },
        auth: {
          type: "jwt",
          config: { secret: TEST_SECRET }
        }
      };

      await proxy.registerServer(config);
      
      const servers = proxy.getConnectedServers();
      assertEquals(servers.length, 1, "Should have one registered server");
      assertEquals(servers[0], "test-server", "Server ID should match");
    } finally {
      mockServer.close();
      // Wait for cleanup
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  },
  sanitizeResources: false,
  sanitizeOps: false
});

Deno.test({
  name: "Federation Proxy - Server Removal",
  async fn() {
    const mockServer = await setupMockServer();
    try {
      const proxy = new FederationProxy(TEST_SECRET);
      
      const config: FederationConfig = {
        serverId: "test-server",
        endpoints: {
          control: `ws://localhost:${WS_PORT}`,
          data: "http://localhost:3001",
        },
        auth: {
          type: "jwt",
          config: { secret: TEST_SECRET }
        }
      };

      await proxy.registerServer(config);
      await proxy.removeServer("test-server");
      
      const servers = proxy.getConnectedServers();
      assertEquals(servers.length, 0, "Should have no registered servers after removal");
    } finally {
      mockServer.close();
      // Wait for cleanup
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  },
  sanitizeResources: false,
  sanitizeOps: false
});
