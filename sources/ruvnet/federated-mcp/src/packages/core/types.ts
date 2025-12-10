// Core protocol types
export interface Capabilities {
  models?: string[];
  protocols?: string[];
  features?: string[];
}

export interface ServerInfo {
  name: string;
  version: string;
  capabilities: Capabilities;
}

export interface Message {
  type: string;
  content: unknown;
}

export interface Response {
  success: boolean;
  data?: unknown;
  error?: string;
}

export interface FederationConfig {
  serverId: string;
  endpoints: {
    control: string;
    data: string;
  };
  auth: {
    type: 'jwt' | 'oauth2';
    config: Record<string, unknown>;
  };
}
