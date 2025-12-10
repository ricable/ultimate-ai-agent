/**
 * Vitest setup file for OpenCode client tests
 */

import { vi } from 'vitest';

// Mock global WebSocket
Object.defineProperty(global, 'WebSocket', {
  writable: true,
  value: vi.fn().mockImplementation(() => ({
    close: vi.fn(),
    send: vi.fn(),
    readyState: 1,
    onopen: null,
    onmessage: null,
    onerror: null,
    onclose: null,
  })),
});

// Mock localStorage
Object.defineProperty(global, 'localStorage', {
  value: {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
  },
  writable: true,
});

// Mock fetch
Object.defineProperty(global, 'fetch', {
  writable: true,
  value: vi.fn(),
});

// Mock environment variables
Object.defineProperty(process, 'env', {
  value: {
    NODE_ENV: 'test',
  },
});