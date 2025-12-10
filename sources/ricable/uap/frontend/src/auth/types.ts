// Authentication types for frontend
export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_verified: boolean;
  roles: string[];
  created_at: string;
  last_login?: string;
  metadata: Record<string, any>;
}

export interface UserCreate {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  roles?: string[];
}

export interface UserLogin {
  username: string;
  password: string;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthResponse {
  tokens: Token;
  user: User;
  message: string;
}

export interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (credentials: UserLogin) => Promise<void>;
  register: (userData: UserCreate) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface Role {
  name: string;
  permissions: string[];
  description?: string;
}

// Permission-related types
export type Permission = 
  | "user:create" | "user:read" | "user:update" | "user:delete"
  | "agent:create" | "agent:read" | "agent:update" | "agent:delete"
  | "system:admin" | "system:manage" | "system:read"
  | "websocket:connect";

export interface AuthError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}