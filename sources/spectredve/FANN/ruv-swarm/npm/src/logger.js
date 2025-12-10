/**
 * Enhanced Structured Logger for ruv-swarm
 * Provides comprehensive logging with connection lifecycle tracking,
 * performance metrics, and debug levels for MCP communication
 */

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { inspect } from 'util';

// Log levels with numeric values for comparison
const LOG_LEVELS = {
  TRACE: 0,
  DEBUG: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
  FATAL: 5,
};

// Reverse mapping for display
const LOG_LEVEL_NAMES = Object.entries(LOG_LEVELS).reduce((acc, [name, value]) => {
  acc[value] = name;
  return acc;
}, {});

// ANSI color codes for console output
const COLORS = {
  TRACE: '\x1b[90m',    // Gray
  DEBUG: '\x1b[36m',    // Cyan
  INFO: '\x1b[32m',     // Green
  WARN: '\x1b[33m',     // Yellow
  ERROR: '\x1b[31m',    // Red
  FATAL: '\x1b[35m',    // Magenta
  RESET: '\x1b[0m',
  DIM: '\x1b[2m',
  BRIGHT: '\x1b[1m',
};

// Icons for different log types
const ICONS = {
  TRACE: '🔍',
  DEBUG: '🐛',
  INFO: 'ℹ️ ',
  WARN: '⚠️ ',
  ERROR: '❌',
  FATAL: '💀',
  CONNECTION: '🔌',
  PERFORMANCE: '⚡',
  MEMORY: '💾',
  MCP: '🔧',
  SWARM: '🐝',
  AGENT: '🤖',
  NEURAL: '🧠',
};

/**
 * Correlation ID generator for tracking sessions/requests
 */
class CorrelationIdGenerator {
  constructor() {
    this.counter = 0;
    this.prefix = Date.now().toString(36);
  }

  generate(type = 'session') {
    this.counter++;
    return `${type}-${this.prefix}-${this.counter.toString(36).padStart(4, '0')}`;
  }
}

/**
 * Log rotation manager
 */
class LogRotation {
  constructor(options = {}) {
    this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB
    this.maxFiles = options.maxFiles || 5;
    this.logDir = options.logDir || path.join(process.cwd(), 'logs');
    this.baseFilename = options.baseFilename || 'ruv-swarm';
    
    // Ensure log directory exists
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  getCurrentLogPath() {
    return path.join(this.logDir, `${this.baseFilename}.log`);
  }

  shouldRotate() {
    const logPath = this.getCurrentLogPath();
    if (!fs.existsSync(logPath)) return false;
    
    const stats = fs.statSync(logPath);
    return stats.size >= this.maxFileSize;
  }

  rotate() {
    const currentPath = this.getCurrentLogPath();
    if (!fs.existsSync(currentPath)) return;

    // Shift existing rotated files
    for (let i = this.maxFiles - 1; i > 0; i--) {
      const oldPath = path.join(this.logDir, `${this.baseFilename}.${i}.log`);
      const newPath = path.join(this.logDir, `${this.baseFilename}.${i + 1}.log`);
      
      if (fs.existsSync(oldPath)) {
        if (i === this.maxFiles - 1) {
          fs.unlinkSync(oldPath); // Remove oldest
        } else {
          fs.renameSync(oldPath, newPath);
        }
      }
    }

    // Rotate current to .1
    fs.renameSync(currentPath, path.join(this.logDir, `${this.baseFilename}.1.log`));
  }

  write(content) {
    if (this.shouldRotate()) {
      this.rotate();
    }

    const logPath = this.getCurrentLogPath();
    fs.appendFileSync(logPath, content);
  }
}

/**
 * Performance tracker for monitoring operation times
 */
class PerformanceTracker {
  constructor() {
    this.operations = new Map();
    this.metrics = new Map();
  }

  startOperation(name, metadata = {}) {
    const id = `${name}-${Date.now()}`;
    this.operations.set(id, {
      name,
      startTime: performance.now(),
      metadata,
      memoryStart: process.memoryUsage(),
    });
    return id;
  }

  endOperation(id, success = true) {
    const operation = this.operations.get(id);
    if (!operation) return null;

    const endTime = performance.now();
    const duration = endTime - operation.startTime;
    const memoryEnd = process.memoryUsage();

    const metric = {
      name: operation.name,
      duration,
      success,
      memoryDelta: {
        heapUsed: memoryEnd.heapUsed - operation.memoryStart.heapUsed,
        external: memoryEnd.external - operation.memoryStart.external,
      },
      timestamp: new Date().toISOString(),
      ...operation.metadata,
    };

    // Store aggregated metrics
    if (!this.metrics.has(operation.name)) {
      this.metrics.set(operation.name, {
        count: 0,
        totalDuration: 0,
        avgDuration: 0,
        minDuration: Infinity,
        maxDuration: -Infinity,
        failures: 0,
      });
    }

    const stats = this.metrics.get(operation.name);
    stats.count++;
    stats.totalDuration += duration;
    stats.avgDuration = stats.totalDuration / stats.count;
    stats.minDuration = Math.min(stats.minDuration, duration);
    stats.maxDuration = Math.max(stats.maxDuration, duration);
    if (!success) stats.failures++;

    this.operations.delete(id);
    return metric;
  }

  getMetrics(name = null) {
    if (name) {
      return this.metrics.get(name);
    }
    return Object.fromEntries(this.metrics);
  }

  clearMetrics() {
    this.metrics.clear();
  }
}

/**
 * Main Logger class
 */
export class Logger {
  constructor(options = {}) {
    this.level = LOG_LEVELS[options.level?.toUpperCase()] ?? LOG_LEVELS.INFO;
    this.name = options.name || 'ruv-swarm';
    this.correlationId = options.correlationId || null;
    this.enableConsole = options.enableConsole ?? true;
    this.enableFile = options.enableFile ?? true;
    this.enableStderr = options.enableStderr ?? false; // For MCP stdio mode
    this.formatJson = options.formatJson ?? false;
    this.includeStackTrace = options.includeStackTrace ?? true;
    this.metadata = options.metadata || {};

    // Initialize components
    this.correlationIdGenerator = new CorrelationIdGenerator();
    this.performanceTracker = new PerformanceTracker();
    
    if (this.enableFile) {
      this.logRotation = new LogRotation({
        logDir: options.logDir,
        maxFileSize: options.maxFileSize,
        maxFiles: options.maxFiles,
        baseFilename: this.name,
      });
    }

    // Connection tracking
    this.connections = new Map();
    this.connectionMetrics = {
      total: 0,
      active: 0,
      failed: 0,
      avgDuration: 0,
    };
  }

  /**
   * Create child logger with inherited settings
   */
  child(options = {}) {
    return new Logger({
      level: LOG_LEVEL_NAMES[this.level],
      name: options.name || `${this.name}:${options.module || 'child'}`,
      correlationId: options.correlationId || this.correlationId,
      enableConsole: this.enableConsole,
      enableFile: this.enableFile,
      enableStderr: this.enableStderr,
      formatJson: this.formatJson,
      includeStackTrace: this.includeStackTrace,
      metadata: { ...this.metadata, ...options.metadata },
      logDir: this.logRotation?.logDir,
    });
  }

  /**
   * Set correlation ID for request tracking
   */
  setCorrelationId(id = null) {
    this.correlationId = id || this.correlationIdGenerator.generate();
    return this.correlationId;
  }

  /**
   * Core logging method
   */
  log(level, message, data = {}) {
    if (level < this.level) return;

    const timestamp = new Date().toISOString();
    const levelName = LOG_LEVEL_NAMES[level];

    // Build log entry
    const logEntry = {
      timestamp,
      level: levelName,
      logger: this.name,
      message,
      correlationId: this.correlationId,
      ...this.metadata,
      ...data,
    };

    // Add error details if present
    if (data.error instanceof Error) {
      logEntry.error = {
        name: data.error.name,
        message: data.error.message,
        code: data.error.code,
        stack: this.includeStackTrace ? data.error.stack : undefined,
      };
    }

    // Format output
    const formatted = this.formatJson
      ? JSON.stringify(logEntry)
      : this.formatConsoleOutput(levelName, message, logEntry);

    // Output to appropriate channels
    this.output(formatted, level);
  }

  /**
   * Format console output with colors and structure
   */
  formatConsoleOutput(levelName, message, data) {
    const color = COLORS[levelName] || COLORS.RESET;
    const icon = ICONS[levelName] || '';
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);

    let output = `${COLORS.DIM}[${timestamp}]${COLORS.RESET} `;
    output += `${color}${icon} ${levelName.padEnd(5)}${COLORS.RESET} `;
    output += `${COLORS.BRIGHT}[${this.name}]${COLORS.RESET} `;
    
    if (this.correlationId) {
      output += `${COLORS.DIM}(${this.correlationId})${COLORS.RESET} `;
    }

    output += message;

    // Add structured data if present
    const filteredData = { ...data };
    delete filteredData.timestamp;
    delete filteredData.level;
    delete filteredData.logger;
    delete filteredData.message;
    delete filteredData.correlationId;

    if (Object.keys(filteredData).length > 0) {
      output += '\n' + inspect(filteredData, {
        depth: 3,
        colors: true,
        compact: false,
        sorted: true,
      });
    }

    return output;
  }

  /**
   * Output to appropriate channels
   */
  output(formatted, level) {
    // In MCP stdio mode, use stderr for logs
    if (this.enableStderr || process.env.MCP_MODE === 'stdio') {
      process.stderr.write(formatted + '\n');
    } else if (this.enableConsole) {
      // Use appropriate console method
      if (level >= LOG_LEVELS.ERROR) {
        console.error(formatted);
      } else if (level >= LOG_LEVELS.WARN) {
        console.warn(formatted);
      } else {
        console.log(formatted);
      }
    }

    // Write to file
    if (this.enableFile && this.logRotation) {
      this.logRotation.write(formatted + '\n');
    }
  }

  // Convenience methods
  trace(message, data) { this.log(LOG_LEVELS.TRACE, message, data); }
  debug(message, data) { this.log(LOG_LEVELS.DEBUG, message, data); }
  info(message, data) { this.log(LOG_LEVELS.INFO, message, data); }
  warn(message, data) { this.log(LOG_LEVELS.WARN, message, data); }
  error(message, data) { this.log(LOG_LEVELS.ERROR, message, data); }
  fatal(message, data) { this.log(LOG_LEVELS.FATAL, message, data); }

  /**
   * Connection lifecycle logging
   */
  logConnection(event, connectionId, details = {}) {
    const icon = ICONS.CONNECTION;
    const message = `${icon} Connection ${event}: ${connectionId}`;
    
    switch (event) {
      case 'established':
        this.connections.set(connectionId, {
          id: connectionId,
          startTime: Date.now(),
          ...details,
        });
        this.connectionMetrics.total++;
        this.connectionMetrics.active++;
        this.info(message, { connection: details });
        break;

      case 'closed':
        const conn = this.connections.get(connectionId);
        if (conn) {
          const duration = Date.now() - conn.startTime;
          this.connectionMetrics.avgDuration = 
            (this.connectionMetrics.avgDuration * (this.connectionMetrics.total - 1) + duration) / 
            this.connectionMetrics.total;
          this.connections.delete(connectionId);
          this.connectionMetrics.active--;
          this.info(message, { duration, connection: details });
        }
        break;

      case 'failed':
        this.connectionMetrics.failed++;
        this.connectionMetrics.active = Math.max(0, this.connectionMetrics.active - 1);
        this.error(message, { connection: details, error: details.error });
        break;

      case 'retry':
        this.warn(message, { attempt: details.attempt, connection: details });
        break;

      default:
        this.debug(message, { connection: details });
    }
  }

  /**
   * MCP message logging
   */
  logMcp(direction, message, details = {}) {
    const icon = ICONS.MCP;
    const arrow = direction === 'in' ? '←' : '→';
    const level = details.error ? LOG_LEVELS.ERROR : LOG_LEVELS.DEBUG;
    
    this.log(level, `${icon} MCP ${arrow} ${message}`, {
      mcp: {
        direction,
        method: details.method,
        id: details.id,
        params: details.params,
        result: details.result,
        error: details.error,
        duration: details.duration,
      },
    });
  }

  /**
   * Performance logging
   */
  startOperation(name, metadata = {}) {
    const id = this.performanceTracker.startOperation(name, metadata);
    this.trace(`${ICONS.PERFORMANCE} Starting operation: ${name}`, { operationId: id, ...metadata });
    return id;
  }

  endOperation(id, success = true, data = {}) {
    const metric = this.performanceTracker.endOperation(id, success);
    if (metric) {
      const level = success ? LOG_LEVELS.DEBUG : LOG_LEVELS.WARN;
      const status = success ? 'completed' : 'failed';
      this.log(level, `${ICONS.PERFORMANCE} Operation ${status}: ${metric.name}`, {
        performance: metric,
        ...data,
      });
    }
    return metric;
  }

  /**
   * Memory usage logging
   */
  logMemoryUsage(context = '') {
    const usage = process.memoryUsage();
    const formatted = {
      heapUsed: `${(usage.heapUsed / 1024 / 1024).toFixed(2)} MB`,
      heapTotal: `${(usage.heapTotal / 1024 / 1024).toFixed(2)} MB`,
      external: `${(usage.external / 1024 / 1024).toFixed(2)} MB`,
      rss: `${(usage.rss / 1024 / 1024).toFixed(2)} MB`,
    };

    this.debug(`${ICONS.MEMORY} Memory usage${context ? ` (${context})` : ''}`, {
      memory: formatted,
      raw: usage,
    });
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(name = null) {
    return this.performanceTracker.getMetrics(name);
  }

  /**
   * Get connection metrics
   */
  getConnectionMetrics() {
    return {
      ...this.connectionMetrics,
      connections: Array.from(this.connections.values()),
    };
  }

  /**
   * Clear all metrics
   */
  clearMetrics() {
    this.performanceTracker.clearMetrics();
    this.connections.clear();
    this.connectionMetrics = {
      total: 0,
      active: 0,
      failed: 0,
      avgDuration: 0,
    };
  }
}

// Default logger instance
export const defaultLogger = new Logger({
  level: process.env.LOG_LEVEL || 'INFO',
  enableStderr: process.env.MCP_MODE === 'stdio',
  formatJson: process.env.LOG_FORMAT === 'json',
});

// Export static methods for convenience
export const trace = defaultLogger.trace.bind(defaultLogger);
export const debug = defaultLogger.debug.bind(defaultLogger);
export const info = defaultLogger.info.bind(defaultLogger);
export const warn = defaultLogger.warn.bind(defaultLogger);
export const error = defaultLogger.error.bind(defaultLogger);
export const fatal = defaultLogger.fatal.bind(defaultLogger);

export default Logger;