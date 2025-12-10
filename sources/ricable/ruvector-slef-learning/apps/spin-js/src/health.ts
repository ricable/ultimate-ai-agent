/**
 * Health Check Component for Spin Agent
 * Kubernetes readiness and liveness probe endpoint
 */

import { ResponseBuilder } from "@fermyon/spin-sdk";

interface HealthStatus {
  status: "healthy" | "unhealthy" | "degraded";
  runtime: string;
  version: string;
  timestamp: string;
  checks: {
    name: string;
    status: "pass" | "fail" | "warn";
    message?: string;
  }[];
}

export async function handler(request: Request, res: ResponseBuilder): Promise<Response> {
  const checks: HealthStatus["checks"] = [];

  // Runtime check
  checks.push({
    name: "runtime",
    status: "pass",
    message: "WebAssembly runtime operational",
  });

  // Memory check (simple allocation test)
  try {
    const testArray = new Array(1000).fill(0);
    checks.push({
      name: "memory",
      status: "pass",
      message: `Memory allocation successful (${testArray.length} elements)`,
    });
  } catch {
    checks.push({
      name: "memory",
      status: "fail",
      message: "Memory allocation failed",
    });
  }

  // Determine overall status
  const hasFailure = checks.some(c => c.status === "fail");
  const hasWarning = checks.some(c => c.status === "warn");

  const status: HealthStatus = {
    status: hasFailure ? "unhealthy" : hasWarning ? "degraded" : "healthy",
    runtime: "spin-wasm",
    version: "0.1.0",
    timestamp: new Date().toISOString(),
    checks,
  };

  return new Response(JSON.stringify(status), {
    status: hasFailure ? 503 : 200,
    headers: { "Content-Type": "application/json" },
  });
}
