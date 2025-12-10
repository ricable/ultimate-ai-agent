import { assertEquals, assertExists } from "https://deno.land/std/testing/asserts.ts";
import { handleMeetingInfo } from "../meeting-info.ts";

// Mock Fireflies API for testing
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
  if (input.toString().includes('api.fireflies.ai')) {
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

Deno.test("Meeting Info - handles valid request", async () => {
  const mockRequest = new Request("http://localhost/meeting-info", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer test-api-key"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123"
    })
  });

  const response = await handleMeetingInfo(mockRequest);
  assertEquals(response.status, 200);

  const data = await response.json();
  assertExists(data.success);
  assertEquals(data.success, true);
  assertExists(data.data.transcript);
});

Deno.test("Meeting Info - handles missing API key", async () => {
  const mockRequest = new Request("http://localhost/meeting-info", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123"
    })
  });

  const response = await handleMeetingInfo(mockRequest);
  assertEquals(response.status, 401);

  const data = await response.json();
  assertEquals(data.success, false);
  assertEquals(data.message, "Missing API key");
});

Deno.test("Meeting Info - handles missing meetingId", async () => {
  const mockRequest = new Request("http://localhost/meeting-info", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer test-api-key"
    },
    body: JSON.stringify({})
  });

  const response = await handleMeetingInfo(mockRequest);
  assertEquals(response.status, 400);

  const data = await response.json();
  assertEquals(data.success, false);
  assertEquals(data.message, "Missing meetingId");
});

Deno.test("Meeting Info - handles invalid method", async () => {
  const mockRequest = new Request("http://localhost/meeting-info", {
    method: "GET"
  });

  const response = await handleMeetingInfo(mockRequest);
  assertEquals(response.status, 405);
});

Deno.test("Meeting Info - handles CORS preflight", async () => {
  const mockRequest = new Request("http://localhost/meeting-info", {
    method: "OPTIONS"
  });

  const response = await handleMeetingInfo(mockRequest);
  assertEquals(response.status, 200);
  assertEquals(response.headers.get("Access-Control-Allow-Origin"), "*");
  assertEquals(response.headers.get("Access-Control-Allow-Methods"), "POST, OPTIONS");
});
