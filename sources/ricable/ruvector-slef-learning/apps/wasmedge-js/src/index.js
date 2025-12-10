/**
 * WasmEdge QuickJS Agent
 * Runs ruvnet npm packages on Kubernetes with WasmEdge runtime
 *
 * This module provides a lightweight AI agent that runs as a WebAssembly
 * module using WasmEdge's QuickJS engine for JavaScript execution.
 */

import * as http from "http";
import * as std from "std";

// Agent Configuration
const CONFIG = {
  agentId: std.getenv("AGENT_ID") || "wasmedge-agent-default",
  model: std.getenv("AGENT_MODEL") || "claude-3-5-sonnet-20241022",
  litellmEndpoint: std.getenv("LITELLM_ENDPOINT") || "http://litellm:4000",
  ruvectorEndpoint: std.getenv("RUVECTOR_ENDPOINT") || "http://ruvector:8765",
  agentdbEndpoint: std.getenv("AGENTDB_ENDPOINT") || "http://agentdb:8766",
  port: parseInt(std.getenv("PORT") || "8080"),
};

// Simple HTTP server using WasmEdge networking
class WasmEdgeAgent {
  constructor(config) {
    this.config = config;
    this.requestCount = 0;
    this.startTime = Date.now();
  }

  // Vector memory operations
  async storeVector(key, embedding, metadata = {}) {
    const response = await this.httpRequest(
      `${this.config.ruvectorEndpoint}/api/v1/vectors`,
      "POST",
      {
        id: key,
        vector: embedding,
        metadata: {
          ...metadata,
          agentId: this.config.agentId,
          timestamp: Date.now(),
        },
      }
    );
    return response;
  }

  async queryVectors(queryVector, topK = 10) {
    const response = await this.httpRequest(
      `${this.config.ruvectorEndpoint}/api/v1/vectors/search`,
      "POST",
      {
        query_vector: queryVector,
        top_k: topK,
        filter: { agentId: this.config.agentId },
      }
    );
    return response;
  }

  // Agent memory operations using AgentDB
  async storeMemory(key, value, memoryType = "episodic") {
    const response = await this.httpRequest(
      `${this.config.agentdbEndpoint}/api/v1/memory`,
      "POST",
      {
        agent_id: this.config.agentId,
        key,
        value,
        memory_type: memoryType,
        timestamp: Date.now(),
      }
    );
    return response;
  }

  async retrieveMemory(key) {
    const response = await this.httpRequest(
      `${this.config.agentdbEndpoint}/api/v1/memory/${this.config.agentId}/${key}`,
      "GET"
    );
    return response;
  }

  // Execute AI task using LiteLLM gateway
  async executeTask(task, context = []) {
    const messages = [
      {
        role: "system",
        content: `You are an AI agent (ID: ${this.config.agentId}) running as a WebAssembly module on Kubernetes using WasmEdge runtime.
You have access to vector memory (RuVector) and agent memory (AgentDB) for knowledge retrieval and state management.
Respond concisely and focus on actionable outputs. Current timestamp: ${new Date().toISOString()}`,
      },
      ...context,
      {
        role: "user",
        content: task,
      },
    ];

    const response = await this.httpRequest(
      `${this.config.litellmEndpoint}/v1/chat/completions`,
      "POST",
      {
        model: this.config.model,
        messages,
        max_tokens: 4096,
        temperature: 0.7,
      }
    );

    return {
      result: response.choices?.[0]?.message?.content || "No response",
      tokensUsed: response.usage?.total_tokens || 0,
      model: response.model,
    };
  }

  // HTTP request helper for WasmEdge networking
  async httpRequest(url, method, body = null) {
    return new Promise((resolve, reject) => {
      const options = {
        method,
        headers: {
          "Content-Type": "application/json",
          "X-Agent-Id": this.config.agentId,
        },
      };

      if (body) {
        options.body = JSON.stringify(body);
      }

      // Use WasmEdge's fetch API
      fetch(url, options)
        .then((res) => res.json())
        .then(resolve)
        .catch(reject);
    });
  }

  // Handle incoming HTTP request
  async handleRequest(request) {
    this.requestCount++;
    const startTime = Date.now();

    try {
      const { pathname } = new URL(request.url, `http://localhost:${this.config.port}`);
      const body = request.body ? JSON.parse(request.body) : {};

      let response;

      switch (pathname) {
        case "/health":
          response = this.healthCheck();
          break;

        case "/metrics":
          response = this.getMetrics();
          break;

        case "/execute":
          if (!body.task) {
            throw new Error("Task is required");
          }
          const taskResult = await this.executeTask(body.task, body.context || []);
          response = {
            success: true,
            agentId: this.config.agentId,
            ...taskResult,
          };
          break;

        case "/memory/store":
          if (!body.key || !body.value) {
            throw new Error("Key and value are required");
          }
          const storeResult = await this.storeMemory(body.key, body.value, body.type);
          response = { success: true, ...storeResult };
          break;

        case "/memory/retrieve":
          if (!body.key) {
            throw new Error("Key is required");
          }
          const retrieveResult = await this.retrieveMemory(body.key);
          response = { success: true, ...retrieveResult };
          break;

        case "/vectors/store":
          if (!body.key || !body.embedding) {
            throw new Error("Key and embedding are required");
          }
          const vectorStoreResult = await this.storeVector(body.key, body.embedding, body.metadata);
          response = { success: true, ...vectorStoreResult };
          break;

        case "/vectors/search":
          if (!body.query) {
            throw new Error("Query vector is required");
          }
          const searchResult = await this.queryVectors(body.query, body.topK);
          response = { success: true, ...searchResult };
          break;

        case "/status":
        default:
          response = {
            success: true,
            agentId: this.config.agentId,
            model: this.config.model,
            runtime: "wasmedge-quickjs",
            uptime: Date.now() - this.startTime,
            requestCount: this.requestCount,
          };
          break;
      }

      return {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "X-Agent-Id": this.config.agentId,
          "X-Execution-Time": String(Date.now() - startTime),
        },
        body: JSON.stringify(response),
      };

    } catch (error) {
      return {
        status: 500,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          success: false,
          agentId: this.config.agentId,
          error: error.message || "Unknown error",
          executionTime: Date.now() - startTime,
        }),
      };
    }
  }

  healthCheck() {
    return {
      status: "healthy",
      runtime: "wasmedge-quickjs",
      version: "0.1.0",
      agentId: this.config.agentId,
      uptime: Date.now() - this.startTime,
      timestamp: new Date().toISOString(),
    };
  }

  getMetrics() {
    return {
      requests_total: this.requestCount,
      uptime_seconds: (Date.now() - this.startTime) / 1000,
      agent_id: this.config.agentId,
      runtime: "wasmedge-quickjs",
    };
  }

  // Start HTTP server
  start() {
    console.log(`WasmEdge Agent ${this.config.agentId} starting on port ${this.config.port}...`);

    // WasmEdge HTTP server setup
    const server = http.createServer(async (req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", async () => {
        const response = await this.handleRequest({
          url: req.url,
          method: req.method,
          headers: req.headers,
          body,
        });

        res.writeHead(response.status, response.headers);
        res.end(response.body);
      });
    });

    server.listen(this.config.port, () => {
      console.log(`Agent listening on http://0.0.0.0:${this.config.port}`);
    });
  }
}

// Initialize and start agent
const agent = new WasmEdgeAgent(CONFIG);
agent.start();

export { WasmEdgeAgent, CONFIG };
