import { ServerInfo, Capabilities, Message, Response as MCPResponse } from './types.ts';

interface ConsoleOptions {
  showTimestamp?: boolean;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

export class MCPServer {
  protected info: ServerInfo;
  private consoleOptions: ConsoleOptions;

  constructor(info: ServerInfo, options: ConsoleOptions = {}) {
    this.info = info;
    this.consoleOptions = {
      showTimestamp: options.showTimestamp ?? true,
      logLevel: options.logLevel ?? 'info'
    };
  }

  protected log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown) {
    const levels = ['debug', 'info', 'warn', 'error'];
    if (levels.indexOf(level) < levels.indexOf(this.consoleOptions.logLevel!)) {
      return;
    }

    const timestamp = this.consoleOptions.showTimestamp 
      ? `[${new Date().toISOString()}] `
      : '';
    
    const prefix = `${timestamp}[${this.info.name}] ${level.toUpperCase()}: `;
    
    switch (level) {
      case 'debug':
        console.debug(prefix + message, data ?? '');
        break;
      case 'info':
        console.info(prefix + message, data ?? '');
        break;
      case 'warn':
        console.warn(prefix + message, data ?? '');
        break;
      case 'error':
        console.error(prefix + message, data ?? '');
        break;
    }
  }

  async handleWebSocket(socket: WebSocket): Promise<void> {
    this.log('info', 'WebSocket connection established');

    socket.onmessage = async (event) => {
      try {
        const message = JSON.parse(event.data);
        this.log('debug', 'Received WebSocket message', message);
        
        const response = await this.handleMessage(message);
        socket.send(JSON.stringify(response));
        
        this.log('debug', 'Sent WebSocket response', response);
      } catch (error) {
        this.log('error', 'WebSocket error', error);
        socket.send(JSON.stringify({ 
          success: false, 
          error: error instanceof Error ? error.message : 'Unknown error'
        }));
      }
    };

    socket.onclose = () => {
      this.log('info', 'WebSocket connection closed');
    };

    socket.onerror = (error) => {
      this.log('error', 'WebSocket error', error);
    };
  }

  async handleHTTP(request: Request): Promise<globalThis.Response> {
    try {
      const body = await request.json();
      this.log('debug', 'Received HTTP request', {
        method: request.method,
        url: request.url,
        body
      });

      const response = await this.handleMessage(body);
      this.log('debug', 'Sending HTTP response', response);

      return new Response(JSON.stringify(response), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      this.log('error', 'HTTP error', error);

      const errorResponse: MCPResponse = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      return new Response(JSON.stringify(errorResponse), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  protected async handleMessage(message: Message): Promise<MCPResponse> {
    this.log('debug', 'Processing message', message);

    try {
      switch (message.type) {
        case 'info':
          return {
            success: true,
            data: this.info
          };
        case 'capabilities':
          return {
            success: true,
            data: this.info.capabilities
          };
        default:
          this.log('warn', `Unknown message type: ${message.type}`);
          return {
            success: false,
            error: `Unknown message type: ${message.type}`
          };
      }
    } catch (error) {
      this.log('error', 'Message handling error', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  printServerInfo() {
    console.log('\n=== MCP Server Information ===');
    console.log(`Name: ${this.info.name}`);
    console.log(`Version: ${this.info.version}`);
    console.log('\nCapabilities:');
    if (this.info.capabilities.models?.length) {
      console.log('- Models:', this.info.capabilities.models.join(', '));
    }
    if (this.info.capabilities.protocols?.length) {
      console.log('- Protocols:', this.info.capabilities.protocols.join(', '));
    }
    if (this.info.capabilities.features?.length) {
      console.log('- Features:', this.info.capabilities.features.join(', '));
    }
    console.log('\nLogging:');
    console.log(`- Level: ${this.consoleOptions.logLevel}`);
    console.log(`- Timestamps: ${this.consoleOptions.showTimestamp ? 'enabled' : 'disabled'}`);
    console.log('===========================\n');
  }
}
