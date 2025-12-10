// Authentication API service
import { User, UserCreate, UserLogin, Token, AuthResponse, Role } from './types';
import { apiConfig, API_BASE_URL } from '../lib/api-config';

class AuthApiError extends Error {
  constructor(message: string, public status?: number, public details?: any) {
    super(message);
    this.name = 'AuthApiError';
  }
}

// Helper function to handle API responses
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
    throw new AuthApiError(
      errorData.detail || `HTTP error! status: ${response.status}`,
      response.status,
      errorData
    );
  }
  return response.json();
}

// Helper function to make authenticated requests (kept for backward compatibility)
function getAuthHeaders(token?: string): Record<string, string> {
  return apiConfig.createHeaders(token);
}

export const authApi = {
  // Register new user
  async register(userData: UserCreate): Promise<User> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/register'), {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(userData),
    });
    
    return handleResponse<User>(response);
  },

  // Login user
  async login(credentials: UserLogin): Promise<AuthResponse> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/login'), {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(credentials),
    });
    
    return handleResponse<AuthResponse>(response);
  },

  // Refresh access token
  async refreshToken(refreshToken: string): Promise<Token> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/refresh'), {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    
    return handleResponse<Token>(response);
  },

  // Logout user
  async logout(refreshToken: string, token: string): Promise<{ message: string }> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/logout'), {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    
    return handleResponse<{ message: string }>(response);
  },

  // Get current user info
  async getCurrentUser(token: string): Promise<User> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/me'), {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    
    return handleResponse<User>(response);
  },

  // Get available roles (admin only)
  async getRoles(token: string): Promise<Record<string, Role>> {
    const response = await fetch(apiConfig.getEndpoint('/api/auth/roles'), {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    
    return handleResponse<Record<string, Role>>(response);
  },

  // Test WebSocket authentication
  getWebSocketUrl(agentId: string, token?: string): string {
    return apiConfig.getAgentWebSocketUrl(agentId, token);
  },

  // Make authenticated API requests to other endpoints
  async makeAuthenticatedRequest<T>(
    endpoint: string,
    token: string,
    options: {
      method?: string;
      body?: any;
      headers?: Record<string, string>;
    } = {}
  ): Promise<T> {
    const { method = 'GET', body, headers = {} } = options;
    
    const requestHeaders = {
      ...getAuthHeaders(token),
      ...headers,
    };

    const requestOptions: RequestInit = {
      method,
      headers: requestHeaders,
    };

    if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
      requestOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
    }

    const response = await fetch(apiConfig.getEndpoint(endpoint), requestOptions);
    return handleResponse<T>(response);
  },
};

export { AuthApiError };