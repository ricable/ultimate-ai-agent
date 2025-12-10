import { WebhookPayload, EdgeResponse } from './types.ts';

export async function handleWebhook(req: Request): Promise<Response> {
  try {
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
      });
    }

    // Only accept POST requests
    if (req.method !== 'POST') {
      const response: EdgeResponse = {
        success: false,
        message: 'Method not allowed'
      };
      return new Response(JSON.stringify(response), { 
        status: 405,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Parse and validate webhook payload
    const payload: WebhookPayload = await req.json();

    if (!payload.meetingId || !payload.eventType) {
      const response: EdgeResponse = {
        success: false,
        message: 'Invalid webhook payload: missing required fields'
      };
      return new Response(JSON.stringify(response), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Only process completed transcriptions
    if (payload.eventType !== 'Transcription completed') {
      const response: EdgeResponse = {
        success: true,
        message: 'Event type not supported for processing',
      };
      return new Response(JSON.stringify(response), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Process the webhook
    const response: EdgeResponse<WebhookPayload> = {
      success: true,
      message: 'Webhook received successfully',
      data: payload
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });

  } catch (error: unknown) {
    const errorResponse: EdgeResponse = {
      success: false,
      message: error instanceof Error ? error.message : 'Internal server error',
    };
    return new Response(JSON.stringify(errorResponse), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}
