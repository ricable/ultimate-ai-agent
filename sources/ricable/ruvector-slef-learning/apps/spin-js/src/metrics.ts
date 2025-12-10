/**
 * Metrics Component for Spin Agent
 * Prometheus-compatible metrics endpoint
 */

import { ResponseBuilder } from "@fermyon/spin-sdk";

// Simple in-memory metrics (resets on cold start)
const metrics = {
  requests_total: 0,
  requests_success: 0,
  requests_error: 0,
  execution_time_sum: 0,
  tokens_used_total: 0,
};

// Increment metrics (called from main handler)
export function recordRequest(success: boolean, executionTime: number, tokensUsed: number = 0): void {
  metrics.requests_total++;
  if (success) {
    metrics.requests_success++;
  } else {
    metrics.requests_error++;
  }
  metrics.execution_time_sum += executionTime;
  metrics.tokens_used_total += tokensUsed;
}

export async function handler(request: Request, res: ResponseBuilder): Promise<Response> {
  // Generate Prometheus-compatible metrics
  const prometheusMetrics = `
# HELP spin_agent_requests_total Total number of requests processed
# TYPE spin_agent_requests_total counter
spin_agent_requests_total ${metrics.requests_total}

# HELP spin_agent_requests_success_total Total successful requests
# TYPE spin_agent_requests_success_total counter
spin_agent_requests_success_total ${metrics.requests_success}

# HELP spin_agent_requests_error_total Total failed requests
# TYPE spin_agent_requests_error_total counter
spin_agent_requests_error_total ${metrics.requests_error}

# HELP spin_agent_execution_time_seconds_sum Sum of execution times
# TYPE spin_agent_execution_time_seconds_sum counter
spin_agent_execution_time_seconds_sum ${metrics.execution_time_sum / 1000}

# HELP spin_agent_tokens_used_total Total tokens consumed
# TYPE spin_agent_tokens_used_total counter
spin_agent_tokens_used_total ${metrics.tokens_used_total}

# HELP spin_agent_info Agent information
# TYPE spin_agent_info gauge
spin_agent_info{runtime="spin-wasm",version="0.1.0"} 1
`.trim();

  return new Response(prometheusMetrics, {
    status: 200,
    headers: { "Content-Type": "text/plain; version=0.0.4" },
  });
}
