/**
 * Authentication Middleware
 * Validates API requests and manages authentication
 */

import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

interface AuthRequest extends Request {
  userId?: string;
  sessionId?: string;
}

/**
 * Authentication middleware
 * In production, integrate with proper auth service (JWT, OAuth, etc.)
 */
export function authMiddleware(
  req: AuthRequest,
  res: Response,
  next: NextFunction
): void {
  try {
    // Skip auth for health check
    if (req.path === '/health') {
      return next();
    }

    // Check for API key in headers
    const apiKey = req.headers['x-api-key'] as string;
    const authHeader = req.headers['authorization'] as string;

    // Simple API key validation (in production, use proper JWT/OAuth)
    if (apiKey && validateApiKey(apiKey)) {
      req.userId = extractUserIdFromApiKey(apiKey);
      req.sessionId = uuidv4();
      return next();
    }

    // Bearer token validation
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      if (validateToken(token)) {
        req.userId = extractUserIdFromToken(token);
        req.sessionId = uuidv4();
        return next();
      }
    }

    // For development, allow unauthenticated access
    if (process.env.NODE_ENV === 'development') {
      req.userId = 'dev-user';
      req.sessionId = uuidv4();
      return next();
    }

    // No valid authentication found
    res.status(401).json({
      success: false,
      error: {
        code: 'UNAUTHORIZED',
        message: 'Authentication required',
        timestamp: new Date()
      },
      metadata: {
        requestId: uuidv4(),
        timestamp: new Date(),
        processingTimeMs: 0,
        version: '1.0.0'
      }
    });

  } catch (error) {
    console.error('Authentication error:', error);
    res.status(500).json({
      success: false,
      error: {
        code: 'AUTH_ERROR',
        message: 'Authentication failed',
        timestamp: new Date()
      },
      metadata: {
        requestId: uuidv4(),
        timestamp: new Date(),
        processingTimeMs: 0,
        version: '1.0.0'
      }
    });
  }
}

/**
 * Validate API key
 */
function validateApiKey(apiKey: string): boolean {
  // In production, validate against database or auth service
  return apiKey.startsWith('medai_') && apiKey.length >= 32;
}

/**
 * Validate bearer token
 */
function validateToken(token: string): boolean {
  // In production, verify JWT signature and expiration
  return token.length >= 20;
}

/**
 * Extract user ID from API key
 */
function extractUserIdFromApiKey(apiKey: string): string {
  // In production, lookup user from database
  return `user_${apiKey.substring(6, 14)}`;
}

/**
 * Extract user ID from token
 */
function extractUserIdFromToken(token: string): string {
  // In production, decode JWT and extract claims
  return `user_${token.substring(0, 8)}`;
}

/**
 * Optional middleware to require provider role
 */
export function requireProviderRole(
  req: AuthRequest,
  res: Response,
  next: NextFunction
): void {
  // In production, check user roles from database
  const userRole = 'provider'; // Mock role

  if (userRole !== 'provider' && userRole !== 'admin') {
    res.status(403).json({
      success: false,
      error: {
        code: 'FORBIDDEN',
        message: 'Provider role required',
        timestamp: new Date()
      },
      metadata: {
        requestId: uuidv4(),
        timestamp: new Date(),
        processingTimeMs: 0,
        version: '1.0.0'
      }
    });
    return;
  }

  next();
}
