import { MCPServer } from '../core/server.ts';
import { ServerInfo, Capabilities, Message, Response as MCPResponse } from '../core/types.ts';
import { handleIntentDetection } from './intent-detection.ts';
import { handleMeetingInfo } from './meeting-info.ts';
import { handleWebhook } from './webhook-handler.ts';
import { EdgeResponse } from './types.ts';

const EDGE_CAPABILITIES: Capabilities = {
  models: ['gpt-4'],
  protocols: ['http', 'websocket'],
  features: [
    'intent-detection',
    'meeting-info',
    'webhook-handler'
  ]
};

export class EdgeMCPServer extends MCPServer {
  constructor() {
    const info: ServerInfo = {
      name: 'edge-mcp',
      version: '1.0.0',
      capabilities: EDGE_CAPABILITIES
    };
    super(info, {
      showTimestamp: true,
      logLevel: 'debug'
    });
  }

  // Public method for testing
  async processMessage(message: Message): Promise<MCPResponse> {
    return this.handleMessage(message);
  }

  protected override async handleMessage(message: Message): Promise<MCPResponse> {
    try {
      switch (message.type) {
        case 'intent-detection': {
          this.log('info', 'Processing intent detection request');
          const request = this.createRequest(message);
          this.log('debug', 'Created intent detection request', request);
          
          const response = await handleIntentDetection(request);
          const edgeResponse = await response.json() as EdgeResponse;
          
          this.log('debug', 'Intent detection response', edgeResponse);
          return {
            success: edgeResponse.success,
            data: edgeResponse
          };
        }

        case 'meeting-info': {
          this.log('info', 'Processing meeting info request');
          const request = this.createRequest(message);
          this.log('debug', 'Created meeting info request', request);
          
          const response = await handleMeetingInfo(request);
          const edgeResponse = await response.json() as EdgeResponse;
          
          this.log('debug', 'Meeting info response', edgeResponse);
          return {
            success: edgeResponse.success,
            data: edgeResponse
          };
        }

        case 'webhook': {
          this.log('info', 'Processing webhook request');
          const request = this.createRequest(message);
          this.log('debug', 'Created webhook request', request);
          
          const response = await handleWebhook(request);
          const edgeResponse = await response.json() as EdgeResponse;
          
          this.log('debug', 'Webhook response', edgeResponse);
          return {
            success: edgeResponse.success,
            data: edgeResponse
          };
        }

        case 'info':
        case 'capabilities':
          return await super.handleMessage(message);

        default:
          this.log('warn', `Unsupported message type: ${message.type}`);
          return {
            success: false,
            error: `Unsupported message type: ${message.type}`
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

  private createRequest(message: Message): Request {
    const headers = new Headers({
      'Content-Type': 'application/json'
    });

    // Add authorization if provided
    if (typeof message.content === 'object' && message.content !== null) {
      const content = message.content as Record<string, unknown>;
      if (content.authorization) {
        headers.set('Authorization', String(content.authorization));
      }
    }

    return new Request('http://edge-mcp', {
      method: 'POST',
      headers,
      body: JSON.stringify(message.content)
    });
  }
}

// Create and export server instance
export const edgeMCP = new EdgeMCPServer();
