import { handleIntentDetection } from './intent-detection.ts';
import { handleMeetingInfo } from './meeting-info.ts';
import { handleWebhook } from './webhook-handler.ts';
import { EdgeResponse } from './types.ts';

async function handleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);
  
  // Route requests to appropriate handlers
  switch (url.pathname) {
    case '/intent-detection':
      return handleIntentDetection(req);
    
    case '/meeting-info':
      return handleMeetingInfo(req);
    
    case '/webhook':
      return handleWebhook(req);
    
    default:
      const response: EdgeResponse = {
        success: false,
        message: 'Not found'
      };
      return new Response(JSON.stringify(response), {
        status: 404,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      });
  }
}

// Start the server if not in test environment
if (!Deno.env.get("DENO_TEST")) {
  Deno.serve(handleRequest);
}
