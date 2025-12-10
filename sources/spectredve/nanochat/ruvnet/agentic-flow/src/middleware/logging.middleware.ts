/**
 * Logging Middleware
 * Request/response logging and monitoring
 */

import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

interface LogEntry {
  requestId: string;
  timestamp: Date;
  method: string;
  path: string;
  statusCode?: number;
  duration?: number;
  userId?: string;
  ip?: string;
  userAgent?: string;
  error?: string;
}

/**
 * Logging middleware
 */
export function loggingMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const requestId = uuidv4();
  const startTime = Date.now();

  // Attach request ID to request object
  (req as any).requestId = requestId;

  // Create log entry
  const logEntry: LogEntry = {
    requestId,
    timestamp: new Date(),
    method: req.method,
    path: req.path,
    userId: (req as any).userId,
    ip: req.ip || req.socket.remoteAddress,
    userAgent: req.headers['user-agent']
  };

  // Log request
  console.log(
    `[${logEntry.timestamp.toISOString()}] ${logEntry.method} ${logEntry.path} - RequestID: ${requestId}`
  );

  // Capture response
  const originalSend = res.send;
  res.send = function (data: any): Response {
    const duration = Date.now() - startTime;

    logEntry.statusCode = res.statusCode;
    logEntry.duration = duration;

    // Log response
    const statusColor = res.statusCode >= 500 ? '\x1b[31m' :
                       res.statusCode >= 400 ? '\x1b[33m' :
                       res.statusCode >= 300 ? '\x1b[36m' :
                       '\x1b[32m';
    const resetColor = '\x1b[0m';

    console.log(
      `[${new Date().toISOString()}] ${statusColor}${res.statusCode}${resetColor} ${logEntry.method} ${logEntry.path} - ${duration}ms`
    );

    // Store log entry (in production, send to logging service)
    storeLogEntry(logEntry);

    return originalSend.call(this, data);
  };

  // Capture errors
  res.on('error', (error: Error) => {
    logEntry.error = error.message;
    console.error(
      `[${new Date().toISOString()}] ERROR ${logEntry.method} ${logEntry.path}:`,
      error
    );
    storeLogEntry(logEntry);
  });

  next();
}

/**
 * Store log entry
 */
function storeLogEntry(entry: LogEntry): void {
  // In production, send to logging service (CloudWatch, Datadog, etc.)
  // For now, just track metrics

  if (entry.statusCode && entry.statusCode >= 400) {
    console.error(`Error log entry:`, JSON.stringify(entry, null, 2));
  }

  // Track performance metrics
  if (entry.duration && entry.duration > 5000) {
    console.warn(`Slow request detected: ${entry.path} took ${entry.duration}ms`);
  }
}

/**
 * Error logging utility
 */
export function logError(context: string, error: Error, metadata?: any): void {
  const errorLog = {
    timestamp: new Date().toISOString(),
    context,
    error: {
      name: error.name,
      message: error.message,
      stack: error.stack
    },
    metadata
  };

  console.error(`[ERROR] ${context}:`, JSON.stringify(errorLog, null, 2));

  // In production, send to error tracking service (Sentry, Rollbar, etc.)
}

/**
 * Performance logging utility
 */
export function logPerformance(operation: string, duration: number, metadata?: any): void {
  const perfLog = {
    timestamp: new Date().toISOString(),
    operation,
    duration,
    metadata
  };

  if (duration > 1000) {
    console.warn(`[PERFORMANCE] Slow operation: ${operation} took ${duration}ms`);
  } else {
    console.log(`[PERFORMANCE] ${operation}: ${duration}ms`);
  }

  // In production, send to APM service (New Relic, AppDynamics, etc.)
}
