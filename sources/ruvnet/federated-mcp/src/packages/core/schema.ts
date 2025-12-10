/* JSON-RPC types */

// Base interface for JSON-RPC messages
interface JSONRPCBase {
  jsonrpc: "2.0";
}

// Request message
export interface JSONRPCRequest extends JSONRPCBase {
  method: string;
  params?: unknown;
  id: number | string;
}

// Notification message (request without id)
export interface JSONRPCNotification extends JSONRPCBase {
  method: string;
  params?: unknown;
}

// Success response
export interface JSONRPCResponse extends JSONRPCBase {
  result: unknown;
  id: number | string;
}

// Error response
export interface JSONRPCError extends JSONRPCBase {
  error: {
    code: number;
    message: string;
    data?: unknown;
  };
  id: number | string | null;
}

// Union type for all JSON-RPC messages
export type JSONRPCMessage =
  | JSONRPCRequest
  | JSONRPCNotification
  | JSONRPCResponse
  | JSONRPCError;

// Protocol version constants
export const LATEST_PROTOCOL_VERSION = "2024-11-05";
export const JSONRPC_VERSION = "2.0";
