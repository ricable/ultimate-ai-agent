import { edgeMCP } from './mcp.ts';

interface DeployConfig {
  port?: number;
  hostname?: string;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

async function startMCPServer(config: DeployConfig = {}) {
  const port = config.port || 8000;
  const hostname = config.hostname || 'localhost';

  // Display server information
  edgeMCP.printServerInfo();

  // Create HTTP server
  const server = Deno.serve({ port, hostname }, async (request) => {
    const url = new URL(request.url);

    // Handle WebSocket upgrade
    if (request.headers.get("upgrade") === "websocket") {
      try {
        const { socket, response } = Deno.upgradeWebSocket(request);
        await edgeMCP.handleWebSocket(socket);
        return response;
      } catch (err) {
        console.error('WebSocket upgrade failed:', err);
        return new Response('WebSocket upgrade failed', { status: 400 });
      }
    }

    // Handle HTTP requests
    try {
      switch (url.pathname) {
        case '/mcp':
          return await edgeMCP.handleHTTP(request);
        
        case '/intent-detection':
        case '/meeting-info':
        case '/webhook':
          // Forward to appropriate edge function handler
          const message = {
            type: url.pathname.slice(1),
            content: await request.json()
          };
          const response = await edgeMCP.handleHTTP(new Request(request.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message)
          }));
          return response;
        
        default:
          return new Response('Not Found', { status: 404 });
      }
    } catch (error) {
      console.error('Request handling error:', error);
      return new Response(JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error'
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  });

  console.log(`\nServer Endpoints:`);
  console.log(`- HTTP/WebSocket: http://${hostname}:${port}/mcp`);
  console.log(`- Intent Detection: http://${hostname}:${port}/intent-detection`);
  console.log(`- Meeting Info: http://${hostname}:${port}/meeting-info`);
  console.log(`- Webhook Handler: http://${hostname}:${port}/webhook\n`);

  return server;
}

if (import.meta.main) {
  const port = parseInt(Deno.env.get("PORT") || "8000");
  const hostname = Deno.env.get("HOSTNAME") || "localhost";
  const logLevel = (Deno.env.get("LOG_LEVEL") || "info") as 'debug' | 'info' | 'warn' | 'error';
  
  await startMCPServer({ port, hostname, logLevel });
}

export { startMCPServer };
