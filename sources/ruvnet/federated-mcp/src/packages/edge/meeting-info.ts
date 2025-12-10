import { TranscriptResponse, EdgeResponse } from './types.ts';

export async function handleMeetingInfo(req: Request): Promise<Response> {
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

    // Get API key from request header
    const apiKey = req.headers.get('Authorization')?.replace('Bearer ', '');
    if (!apiKey) {
      const response: EdgeResponse = {
        success: false,
        message: 'Missing API key'
      };
      return new Response(JSON.stringify(response), { 
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Parse request body
    const { meetingId } = await req.json();
    if (!meetingId) {
      const response: EdgeResponse = {
        success: false,
        message: 'Missing meetingId'
      };
      return new Response(JSON.stringify(response), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Query Fireflies GraphQL API
    const query = `
      query Transcript($transcriptId: String!) {
        transcript(id: $transcriptId) {
          summary {
            keywords
            action_items
            outline
            shorthand_bullet
            overview
            bullet_gist
            gist
            short_summary
          }
        }
      }
    `;

    const response = await fetch('https://api.fireflies.ai/graphql', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        query,
        variables: { transcriptId: meetingId },
      }),
    });

    const result = await response.json();

    if (result.errors) {
      const errorResponse: EdgeResponse = {
        success: false,
        message: result.errors[0].message
      };
      return new Response(JSON.stringify(errorResponse), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const successResponse: EdgeResponse<TranscriptResponse> = {
      success: true,
      message: 'Meeting info retrieved successfully',
      data: result.data
    };

    return new Response(JSON.stringify(successResponse), {
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
