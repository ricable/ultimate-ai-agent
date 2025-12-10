/**
 * Unified API Configuration
 * 
 * This module provides centralized configuration for API endpoints
 * and handles different environments (development, Docker, production)
 */

// Get the API URL from environment or construct it
export function getApiUrl(): string {
  // Check if VITE_API_URL is provided (set by docker-compose)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  // For relative URLs in production (goes through proxy)
  if (import.meta.env.PROD) {
    return '';
  }

  // For development, construct from window location
  const protocol = window.location.protocol;
  const host = window.location.hostname;
  const port = import.meta.env.ARCHON_SERVER_PORT;
  
  if (!port) {
    throw new Error(
      'ARCHON_SERVER_PORT environment variable is required. ' +
      'Please set it in your environment variables. ' +
      'Default value: 8181'
    );
  }
  
  return `${protocol}//${host}:${port}`;
}

// Get the base path for API endpoints
export function getApiBasePath(): string {
  const apiUrl = getApiUrl();
  
  // If using relative URLs (empty string), just return /api
  if (!apiUrl) {
    return '/api';
  }
  
  // Otherwise, append /api to the base URL
  return `${apiUrl}/api`;
}

// Get WebSocket URL for real-time connections
export function getWebSocketUrl(): string {
  const apiUrl = getApiUrl();
  
  // If using relative URLs, construct from current location
  if (!apiUrl) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}`;
  }
  
  // Convert http/https to ws/wss
  return apiUrl.replace(/^http/, 'ws');
}

// Export commonly used values
export const API_BASE_URL = '/api';  // Always use relative URL for API calls
export const API_FULL_URL = getApiUrl();
export const WS_URL = getWebSocketUrl();