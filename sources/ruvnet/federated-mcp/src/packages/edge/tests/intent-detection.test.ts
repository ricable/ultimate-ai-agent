import { assertEquals, assertExists } from "https://deno.land/std/testing/asserts.ts";
import { handleIntentDetection } from "../intent-detection.ts";

// Mock OpenAI API for testing
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
  if (input.toString().includes('api.openai.com')) {
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
  return originalFetch(input, init);
};

Deno.test("Intent Detection - handles valid request", async () => {
  const mockRequest = new Request("http://localhost/intent-detection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123",
      transcriptionText: "Let's schedule a follow-up meeting next week. @John will prepare the presentation.",
      participants: ["John", "Alice"],
      metadata: {
        duration: 3600,
        date: "2024-01-15T10:00:00Z"
      }
    })
  });

  const response = await handleIntentDetection(mockRequest);
  assertEquals(response.status, 200);

  const data = await response.json();
  assertExists(data.success);
  assertEquals(data.success, true);
  assertExists(data.data.intents);
  assertEquals(Array.isArray(data.data.intents), true);
});

Deno.test("Intent Detection - handles missing required fields", async () => {
  const mockRequest = new Request("http://localhost/intent-detection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123"
      // Missing transcriptionText
    })
  });

  const response = await handleIntentDetection(mockRequest);
  assertEquals(response.status, 400);

  const data = await response.json();
  assertEquals(data.success, false);
  assertExists(data.message);
});

Deno.test("Intent Detection - handles invalid method", async () => {
  const mockRequest = new Request("http://localhost/intent-detection", {
    method: "GET"
  });

  const response = await handleIntentDetection(mockRequest);
  assertEquals(response.status, 405);
});

Deno.test("Intent Detection - handles CORS preflight", async () => {
  const mockRequest = new Request("http://localhost/intent-detection", {
    method: "OPTIONS"
  });

  const response = await handleIntentDetection(mockRequest);
  assertEquals(response.status, 200);
  assertEquals(response.headers.get("Access-Control-Allow-Origin"), "*");
  assertEquals(response.headers.get("Access-Control-Allow-Methods"), "POST, OPTIONS");
});
