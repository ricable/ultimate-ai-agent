/**
 * Spin JS SDK Agent Handler
 * Runs ruvnet npm packages on Kubernetes via SpinKube/WasmEdge
 *
 * This component handles HTTP requests and orchestrates AI agent tasks
 * using the claude-flow and ruvector packages compiled to WebAssembly.
 */

import { ResponseBuilder } from "@fermyon/spin-sdk";

// Agent configuration from Spin variables
interface AgentConfig {
  agentId: string;
  model: string;
  anthropicApiKey: string;
  litellmEndpoint: string;
  ruvectorEndpoint: string;
}

// Request/Response types
interface AgentRequest {
  action: "execute" | "query" | "memory" | "status";
  task?: string;
  query?: string;
  memoryKey?: string;
  memoryValue?: any;
}

interface AgentResponse {
  success: boolean;
  agentId: string;
  result?: any;
  error?: string;
  metrics?: {
    executionTime: number;
    tokensUsed?: number;
  };
}

// Get configuration from Spin variables
function getConfig(): AgentConfig {
  const spinConfig = (globalThis as any).spinConfig || {};
  return {
    agentId: spinConfig.agent_id || "default-agent",
    model: spinConfig.agent_model || "claude-3-5-sonnet-20241022",
    anthropicApiKey: spinConfig.anthropic_api_key || "",
    litellmEndpoint: spinConfig.litellm_endpoint || "http://litellm:4000",
    ruvectorEndpoint: spinConfig.ruvector_endpoint || "http://ruvector:8765",
  };
}

// Vector memory operations using RuVector
async function vectorMemory(
  config: AgentConfig,
  operation: "store" | "query" | "delete",
  key?: string,
  value?: any
): Promise<any> {
  const endpoint = `${config.ruvectorEndpoint}/api/v1/vectors`;

  switch (operation) {
    case "store":
      const storeResponse = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: key,
          vector: value.embedding,
          metadata: value.metadata,
        }),
      });
      return storeResponse.json();

    case "query":
      const queryResponse = await fetch(`${endpoint}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_vector: value,
          top_k: 10,
        }),
      });
      return queryResponse.json();

    case "delete":
      const deleteResponse = await fetch(`${endpoint}/${key}`, {
        method: "DELETE",
      });
      return deleteResponse.json();
  }
}

// Execute AI task using LiteLLM gateway
async function executeTask(
  config: AgentConfig,
  task: string
): Promise<{ result: string; tokensUsed: number }> {
  const startTime = Date.now();

  const response = await fetch(`${config.litellmEndpoint}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.anthropicApiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      messages: [
        {
          role: "system",
          content: `You are an AI agent (ID: ${config.agentId}) running as a WebAssembly module on Kubernetes.
                    You have access to vector memory for knowledge retrieval and can execute structured tasks.
                    Respond concisely and focus on actionable outputs.`,
        },
        {
          role: "user",
          content: task,
        },
      ],
      max_tokens: 4096,
      temperature: 0.7,
    }),
  });

  const data = await response.json();

  return {
    result: data.choices?.[0]?.message?.content || "No response generated",
    tokensUsed: data.usage?.total_tokens || 0,
  };
}

// Main request handler
export async function handler(request: Request, res: ResponseBuilder): Promise<Response> {
  const config = getConfig();
  const startTime = Date.now();

  try {
    // Parse request body
    const body = await request.text();
    const agentRequest: AgentRequest = body ? JSON.parse(body) : { action: "status" };

    let result: any;

    switch (agentRequest.action) {
      case "execute":
        if (!agentRequest.task) {
          throw new Error("Task is required for execute action");
        }
        const taskResult = await executeTask(config, agentRequest.task);
        result = taskResult;
        break;

      case "query":
        if (!agentRequest.query) {
          throw new Error("Query is required for query action");
        }
        result = await vectorMemory(config, "query", undefined, agentRequest.query);
        break;

      case "memory":
        if (agentRequest.memoryKey && agentRequest.memoryValue) {
          result = await vectorMemory(config, "store", agentRequest.memoryKey, agentRequest.memoryValue);
        } else if (agentRequest.memoryKey) {
          result = await vectorMemory(config, "delete", agentRequest.memoryKey);
        } else {
          throw new Error("Memory key is required");
        }
        break;

      case "status":
      default:
        result = {
          agentId: config.agentId,
          model: config.model,
          runtime: "spin-wasm",
          status: "healthy",
          uptime: Date.now(),
        };
        break;
    }

    const response: AgentResponse = {
      success: true,
      agentId: config.agentId,
      result,
      metrics: {
        executionTime: Date.now() - startTime,
        tokensUsed: result?.tokensUsed,
      },
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "X-Agent-Id": config.agentId,
        "X-Execution-Time": String(Date.now() - startTime),
      },
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    const response: AgentResponse = {
      success: false,
      agentId: config.agentId,
      error: errorMessage,
      metrics: {
        executionTime: Date.now() - startTime,
      },
    };

    return new Response(JSON.stringify(response), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
