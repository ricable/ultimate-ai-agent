import { assertEquals, assertExists } from "https://deno.land/std/testing/asserts.ts";
import { handleWebhook } from "../webhook-handler.ts";

Deno.test("Webhook Handler - handles valid transcription completed event", async () => {
  const mockRequest = new Request("http://localhost/webhook", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123",
      eventType: "Transcription completed",
      clientReferenceId: "client-ref-123"
    })
  });

  const response = await handleWebhook(mockRequest);
  assertEquals(response.status, 200);

  const data = await response.json();
  assertExists(data.success);
  assertEquals(data.success, true);
  assertExists(data.data);
  assertEquals(data.data.meetingId, "test-meeting-123");
  assertEquals(data.data.eventType, "Transcription completed");
});

Deno.test("Webhook Handler - handles unsupported event type", async () => {
  const mockRequest = new Request("http://localhost/webhook", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123",
      eventType: "Meeting started",
      clientReferenceId: "client-ref-123"
    })
  });

  const response = await handleWebhook(mockRequest);
  assertEquals(response.status, 200);

  const data = await response.json();
  assertEquals(data.success, true);
  assertEquals(data.message, "Event type not supported for processing");
});

Deno.test("Webhook Handler - handles missing required fields", async () => {
  const mockRequest = new Request("http://localhost/webhook", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      meetingId: "test-meeting-123"
      // Missing eventType
    })
  });

  const response = await handleWebhook(mockRequest);
  assertEquals(response.status, 400);

  const data = await response.json();
  assertEquals(data.success, false);
  assertExists(data.message);
});

Deno.test("Webhook Handler - handles invalid method", async () => {
  const mockRequest = new Request("http://localhost/webhook", {
    method: "GET"
  });

  const response = await handleWebhook(mockRequest);
  assertEquals(response.status, 405);
});

Deno.test("Webhook Handler - handles CORS preflight", async () => {
  const mockRequest = new Request("http://localhost/webhook", {
    method: "OPTIONS"
  });

  const response = await handleWebhook(mockRequest);
  assertEquals(response.status, 200);
  assertEquals(response.headers.get("Access-Control-Allow-Origin"), "*");
  assertEquals(response.headers.get("Access-Control-Allow-Methods"), "POST, OPTIONS");
});
