// Authentication context and provider
import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { User, UserLogin, UserCreate, AuthContextType, Token } from './types';
import { authApi, AuthApiError } from './api';

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Storage keys
const STORAGE_KEYS = {
  ACCESS_TOKEN: 'uap_access_token',
  REFRESH_TOKEN: 'uap_refresh_token',
  USER: 'uap_user',
} as const;

// Token utilities
class TokenManager {
  static setTokens(tokens: Token): void {
    localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, tokens.access_token);
    localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, tokens.refresh_token);
  }

  static getAccessToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN);
  }

  static getRefreshToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
  }

  static clearTokens(): void {
    localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
  }

  static setUser(user: User): void {
    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
  }

  static getUser(): User | null {
    const userStr = localStorage.getItem(STORAGE_KEYS.USER);
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  }

  static clearUser(): void {
    localStorage.removeItem(STORAGE_KEYS.USER);
  }

  static clearAll(): void {
    TokenManager.clearTokens();
    TokenManager.clearUser();
  }
}

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize auth state from localStorage
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const savedUser = TokenManager.getUser();
        const savedToken = TokenManager.getAccessToken();

        if (savedUser && savedToken) {
          // Verify token is still valid by fetching current user
          try {
            const currentUser = await authApi.getCurrentUser(savedToken);
            setUser(currentUser);
            setToken(savedToken);
            TokenManager.setUser(currentUser); // Update stored user info
          } catch (error) {
            // Token might be expired, try to refresh
            const refreshToken = TokenManager.getRefreshToken();
            if (refreshToken) {
              try {
                await refreshTokens(refreshToken);
              } catch {
                // Refresh failed, clear everything
                TokenManager.clearAll();
              }
            } else {
              TokenManager.clearAll();
            }
          }
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
        TokenManager.clearAll();
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Clear error when user changes
  useEffect(() => {
    if (error) {
      const timeout = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timeout);
    }
  }, [error]);

  const refreshTokens = useCallback(async (refreshToken?: string): Promise<void> => {
    const tokenToUse = refreshToken || TokenManager.getRefreshToken();
    
    if (!tokenToUse) {
      throw new Error('No refresh token available');
    }

    try {
      const newTokens = await authApi.refreshToken(tokenToUse);
      TokenManager.setTokens(newTokens);
      setToken(newTokens.access_token);

      // Fetch updated user info
      const currentUser = await authApi.getCurrentUser(newTokens.access_token);
      setUser(currentUser);
      TokenManager.setUser(currentUser);
    } catch (error) {
      // Refresh failed, clear everything
      TokenManager.clearAll();
      setUser(null);
      setToken(null);
      throw error;
    }
  }, []);

  const login = useCallback(async (credentials: UserLogin): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await authApi.login(credentials);
      
      // Store tokens and user
      TokenManager.setTokens(response.tokens);
      TokenManager.setUser(response.user);
      
      setUser(response.user);
      setToken(response.tokens.access_token);
    } catch (error) {
      if (error instanceof AuthApiError) {
        setError(error.message);
      } else {
        setError('Login failed. Please try again.');
      }
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const register = useCallback(async (userData: UserCreate): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      await authApi.register(userData);
      // After successful registration, automatically log in
      await login({ username: userData.username, password: userData.password });
    } catch (error) {
      if (error instanceof AuthApiError) {
        setError(error.message);
      } else {
        setError('Registration failed. Please try again.');
      }
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [login]);

  const logout = useCallback(async (): Promise<void> => {
    const refreshToken = TokenManager.getRefreshToken();
    const accessToken = TokenManager.getAccessToken();

    // Clear local state immediately
    setUser(null);
    setToken(null);
    TokenManager.clearAll();

    // Try to logout on server (don't block on failure)
    if (refreshToken && accessToken) {
      try {
        await authApi.logout(refreshToken, accessToken);
      } catch (error) {
        console.warn('Server logout failed:', error);
      }
    }
  }, []);

  const value: AuthContextType = {
    user,
    token,
    login,
    register,
    logout,
    refreshToken: () => refreshTokens(),
    isAuthenticated: !!user && !!token,
    isLoading,
    error,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Hook for making authenticated API requests
export const useAuthenticatedApi = () => {
  const { token, refreshToken: refresh } = useAuth();

  const makeRequest = useCallback(async <T,>(
    endpoint: string,
    options?: {
      method?: string;
      body?: any;
      headers?: Record<string, string>;
    }
  ): Promise<T> => {
    if (!token) {
      throw new Error('No authentication token available');
    }

    try {
      return await authApi.makeAuthenticatedRequest<T>(endpoint, token, options);
    } catch (error) {
      if (error instanceof AuthApiError && error.status === 401) {
        // Token might be expired, try to refresh
        try {
          await refresh();
          // Retry the request with new token
          const newToken = TokenManager.getAccessToken();
          if (newToken) {
            return await authApi.makeAuthenticatedRequest<T>(endpoint, newToken, options);
          }
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
        }
      }
      throw error;
    }
  }, [token, refresh]);

  return { makeRequest };
};