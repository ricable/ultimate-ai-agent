import { assertEquals, assertExists } from "https://deno.land/std/testing/asserts.ts";
import { edgeMCP } from "../mcp.ts";
import { Message, Response as MCPResponse } from "../../core/types.ts";

// Mock external API calls
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
  const url = input.toString();
  
  if (url.includes('api.openai.com')) {
    return new Response(JSON.stringify({
      choices: [
        {
          message: {
            content: JSON.stringify([
              {
                type: "Task Assignment",
                confidence: 0.9,
                segment: {
                  text: "Let's schedule a follow-up meeting next week",
                  timestamp: 0,
                  speaker: "unknown"
                },
                metadata: {
                  context: ["meeting planning"],
                  entities: ["next week"]
                }
              }
            ])
          }
        }
      ]
    }));
  }
  
  if (url.includes('api.fireflies.ai')) {
    return new Response(JSON.stringify({
      data: {
        transcript: {
          summary: {
            keywords: ["meeting", "planning"],
            action_items: ["Schedule follow-up"],
            outline: ["Introduction", "Discussion"],
            shorthand_bullet: ["Meeting recap"],
            overview: "Team planning meeting",
            bullet_gist: ["Key points discussed"],
            gist: "Planning session",
            short_summary: "Team discussed upcoming plans"
          }
        }
      }
    }));
  }
  
  return originalFetch(input, init);
};

// Test MCP server capabilities
Deno.test("MCP Server - provides correct capabilities", async () => {
  const message: Message = {
    type: "capabilities",
    content: null
  };

  const response = await edgeMCP.processMessage(message);
  assertEquals(response.success, true);
  const capabilities = response.data as { features: string[] };
  assertEquals(Array.isArray(capabilities.features), true);
  assertEquals(capabilities.features.includes('intent-detection'), true);
  assertEquals(capabilities.features.includes('meeting-info'), true);
  assertEquals(capabilities.features.includes('webhook-handler'), true);
});

// Test intent detection through MCP
Deno.test("MCP Server - handles intent detection", async () => {
  const message: Message = {
    type: "intent-detection",
    content: {
      meetingId: "test-meeting-123",
      transcriptionText: "Let's schedule a follow-up meeting next week.",
      participants: ["John", "Alice"],
      metadata: {
        duration: 3600,
        date: "2024-01-15T10:00:00Z"
      }
    }
  };

  const response = await edgeMCP.processMessage(message);
  assertEquals(response.success, true);
  assertExists(response.data);
  const data = response.data as { data: { intents: unknown[] } };
  assertExists(data.data.intents);
  assertEquals(Array.isArray(data.data.intents), true);
});

// Test meeting info through MCP
Deno.test("MCP Server - handles meeting info", async () => {
  const message: Message = {
    type: "meeting-info",
    content: {
      meetingId: "test-meeting-123",
      authorization: "Bearer test-token"
    }
  };

  const response = await edgeMCP.processMessage(message);
  assertEquals(response.success, true);
  assertExists(response.data);
  const data = response.data as { data: { transcript: unknown } };
  assertExists(data.data.transcript);
});

// Test webhook through MCP
Deno.test("MCP Server - handles webhook", async () => {
  const message: Message = {
    type: "webhook",
    content: {
      meetingId: "test-meeting-123",
      eventType: "Transcription completed",
      clientReferenceId: "client-ref-123"
    }
  };

  const response = await edgeMCP.processMessage(message);
  assertEquals(response.success, true);
  assertExists(response.data);
  const data = response.data as { data: { meetingId: string } };
  assertEquals(data.data.meetingId, "test-meeting-123");
});

// Test invalid message type
Deno.test("MCP Server - handles invalid message type", async () => {
  const message: Message = {
    type: "invalid-type",
    content: {}
  };

  const response = await edgeMCP.processMessage(message);
  assertEquals(response.success, false);
  assertExists(response.error);
});
