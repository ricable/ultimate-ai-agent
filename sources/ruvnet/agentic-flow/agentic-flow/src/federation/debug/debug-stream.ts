/**
 * Debug Streaming System for Federation
 *
 * Provides detailed, real-time visibility into agent operations
 * with multiple verbosity levels and customizable output formats.
 *
 * Features:
 * - Multiple debug levels (SILENT, BASIC, DETAILED, VERBOSE, TRACE)
 * - Real-time event streaming
 * - Performance metrics and timing
 * - Stack traces and context
 * - Customizable formatters
 * - File and console output
 * - JSON and human-readable formats
 */

import { EventEmitter } from 'events';
import { WriteStream, createWriteStream } from 'fs';

export enum DebugLevel {
  SILENT = 0,   // No output
  BASIC = 1,    // Major events only (init, shutdown, errors)
  DETAILED = 2, // Include operations (queries, inserts, updates)
  VERBOSE = 3,  // Include all events (presence, messages, tasks)
  TRACE = 4,    // Everything including internal state changes
}

export interface DebugEvent {
  timestamp: string;
  level: DebugLevel;
  category: string;
  operation: string;
  agentId?: string;
  tenantId?: string;
  duration?: number;
  data?: any;
  error?: Error;
  stackTrace?: string;
  metadata?: Record<string, any>;
}

export interface DebugStreamConfig {
  level: DebugLevel;
  output: 'console' | 'file' | 'both' | 'stream';
  format: 'human' | 'json' | 'compact';
  outputFile?: string;
  includeTimestamps?: boolean;
  includeStackTraces?: boolean;
  includeMetadata?: boolean;
  colorize?: boolean;
  filterCategories?: string[];
  filterAgents?: string[];
  customStream?: WriteStream;
}

export class DebugStream extends EventEmitter {
  private config: DebugStreamConfig;
  private fileStream?: WriteStream;
  private eventBuffer: DebugEvent[] = [];
  private metrics: Map<string, { count: number; totalDuration: number }> = new Map();

  constructor(config: Partial<DebugStreamConfig> = {}) {
    super();

    this.config = {
      level: config.level ?? DebugLevel.BASIC,
      output: config.output ?? 'console',
      format: config.format ?? 'human',
      includeTimestamps: config.includeTimestamps ?? true,
      includeStackTraces: config.includeStackTraces ?? false,
      includeMetadata: config.includeMetadata ?? true,
      colorize: config.colorize ?? true,
      filterCategories: config.filterCategories,
      filterAgents: config.filterAgents,
      outputFile: config.outputFile,
      customStream: config.customStream,
    };

    if (this.config.outputFile && (this.config.output === 'file' || this.config.output === 'both')) {
      this.fileStream = createWriteStream(this.config.outputFile, { flags: 'a' });
    }
  }

  /**
   * Log a debug event
   */
  log(event: Omit<DebugEvent, 'timestamp'>): void {
    // Check if event should be logged based on level
    if (event.level > this.config.level) {
      return;
    }

    // Apply filters
    if (this.config.filterCategories && !this.config.filterCategories.includes(event.category)) {
      return;
    }

    if (this.config.filterAgents && event.agentId && !this.config.filterAgents.includes(event.agentId)) {
      return;
    }

    const fullEvent: DebugEvent = {
      ...event,
      timestamp: new Date().toISOString(),
      stackTrace: this.config.includeStackTraces ? this.captureStackTrace() : undefined,
    };

    // Buffer event
    this.eventBuffer.push(fullEvent);

    // Update metrics
    if (event.duration !== undefined) {
      const key = `${event.category}:${event.operation}`;
      const existing = this.metrics.get(key) || { count: 0, totalDuration: 0 };
      this.metrics.set(key, {
        count: existing.count + 1,
        totalDuration: existing.totalDuration + event.duration,
      });
    }

    // Output event
    this.outputEvent(fullEvent);

    // Emit for external listeners
    this.emit('event', fullEvent);
  }

  /**
   * Log connection events
   */
  logConnection(operation: string, data?: any, error?: Error): void {
    this.log({
      level: DebugLevel.BASIC,
      category: 'connection',
      operation,
      data,
      error,
    });
  }

  /**
   * Log database operations
   */
  logDatabase(operation: string, data?: any, duration?: number, error?: Error): void {
    this.log({
      level: DebugLevel.DETAILED,
      category: 'database',
      operation,
      data,
      duration,
      error,
    });
  }

  /**
   * Log realtime events
   */
  logRealtime(operation: string, agentId?: string, data?: any, duration?: number): void {
    this.log({
      level: DebugLevel.VERBOSE,
      category: 'realtime',
      operation,
      agentId,
      data,
      duration,
    });
  }

  /**
   * Log memory operations
   */
  logMemory(operation: string, agentId?: string, tenantId?: string, data?: any, duration?: number): void {
    this.log({
      level: DebugLevel.DETAILED,
      category: 'memory',
      operation,
      agentId,
      tenantId,
      data,
      duration,
    });
  }

  /**
   * Log task operations
   */
  logTask(operation: string, agentId?: string, tenantId?: string, data?: any, duration?: number): void {
    this.log({
      level: DebugLevel.VERBOSE,
      category: 'task',
      operation,
      agentId,
      tenantId,
      data,
      duration,
    });
  }

  /**
   * Log internal state changes
   */
  logTrace(operation: string, data?: any): void {
    this.log({
      level: DebugLevel.TRACE,
      category: 'trace',
      operation,
      data,
    });
  }

  /**
   * Output event to configured destinations
   */
  private outputEvent(event: DebugEvent): void {
    const formatted = this.formatEvent(event);

    if (this.config.output === 'console' || this.config.output === 'both') {
      console.log(formatted);
    }

    if (this.config.output === 'file' || this.config.output === 'both') {
      if (this.fileStream) {
        this.fileStream.write(formatted + '\n');
      }
    }

    if (this.config.output === 'stream' && this.config.customStream) {
      this.config.customStream.write(formatted + '\n');
    }
  }

  /**
   * Format event for output
   */
  private formatEvent(event: DebugEvent): string {
    if (this.config.format === 'json') {
      return JSON.stringify(event);
    }

    if (this.config.format === 'compact') {
      return this.formatCompact(event);
    }

    return this.formatHuman(event);
  }

  /**
   * Format event in human-readable format
   */
  private formatHuman(event: DebugEvent): string {
    const parts: string[] = [];

    // Timestamp
    if (this.config.includeTimestamps) {
      const timestamp = this.colorize(event.timestamp, 'gray');
      parts.push(`[${timestamp}]`);
    }

    // Level
    const levelStr = this.getLevelString(event.level);
    parts.push(this.colorize(levelStr, this.getLevelColor(event.level)));

    // Category
    parts.push(this.colorize(event.category.toUpperCase(), 'cyan'));

    // Agent/Tenant
    if (event.agentId) {
      parts.push(this.colorize(`agent=${event.agentId}`, 'blue'));
    }
    if (event.tenantId) {
      parts.push(this.colorize(`tenant=${event.tenantId}`, 'blue'));
    }

    // Operation
    parts.push(this.colorize(event.operation, 'white'));

    // Duration
    if (event.duration !== undefined) {
      const durationStr = `${event.duration.toFixed(2)}ms`;
      parts.push(this.colorize(durationStr, 'yellow'));
    }

    let output = parts.join(' ');

    // Data
    if (event.data && this.config.includeMetadata) {
      const dataStr = typeof event.data === 'string'
        ? event.data
        : JSON.stringify(event.data, null, 2);
      output += '\n  ' + this.colorize('Data:', 'gray') + ' ' + dataStr;
    }

    // Metadata
    if (event.metadata && this.config.includeMetadata) {
      output += '\n  ' + this.colorize('Metadata:', 'gray') + ' ' + JSON.stringify(event.metadata);
    }

    // Error
    if (event.error) {
      output += '\n  ' + this.colorize('Error:', 'red') + ' ' + event.error.message;
      if (event.error.stack) {
        output += '\n  ' + this.colorize('Stack:', 'red') + '\n' + event.error.stack
          .split('\n')
          .map(line => '    ' + line)
          .join('\n');
      }
    }

    // Stack trace
    if (event.stackTrace && this.config.includeStackTraces) {
      output += '\n  ' + this.colorize('Trace:', 'gray') + '\n' + event.stackTrace
        .split('\n')
        .slice(0, 5)
        .map(line => '    ' + line)
        .join('\n');
    }

    return output;
  }

  /**
   * Format event in compact format
   */
  private formatCompact(event: DebugEvent): string {
    const parts: string[] = [];

    if (this.config.includeTimestamps) {
      parts.push(event.timestamp);
    }

    parts.push(this.getLevelString(event.level));
    parts.push(event.category);

    if (event.agentId) parts.push(`a=${event.agentId}`);
    if (event.tenantId) parts.push(`t=${event.tenantId}`);

    parts.push(event.operation);

    if (event.duration !== undefined) {
      parts.push(`${event.duration.toFixed(0)}ms`);
    }

    if (event.error) {
      parts.push(`ERROR: ${event.error.message}`);
    }

    return parts.join(' | ');
  }

  /**
   * Get level string
   */
  private getLevelString(level: DebugLevel): string {
    switch (level) {
      case DebugLevel.SILENT: return 'SILENT';
      case DebugLevel.BASIC: return 'BASIC ';
      case DebugLevel.DETAILED: return 'DETAIL';
      case DebugLevel.VERBOSE: return 'VERBOS';
      case DebugLevel.TRACE: return 'TRACE ';
      default: return 'UNKNOWN';
    }
  }

  /**
   * Get color for level
   */
  private getLevelColor(level: DebugLevel): string {
    switch (level) {
      case DebugLevel.BASIC: return 'green';
      case DebugLevel.DETAILED: return 'blue';
      case DebugLevel.VERBOSE: return 'magenta';
      case DebugLevel.TRACE: return 'gray';
      default: return 'white';
    }
  }

  /**
   * Colorize text
   */
  private colorize(text: string, color: string): string {
    if (!this.config.colorize) return text;

    const colors: Record<string, string> = {
      gray: '\x1b[90m',
      red: '\x1b[31m',
      green: '\x1b[32m',
      yellow: '\x1b[33m',
      blue: '\x1b[34m',
      magenta: '\x1b[35m',
      cyan: '\x1b[36m',
      white: '\x1b[37m',
    };

    const reset = '\x1b[0m';
    return (colors[color] || '') + text + reset;
  }

  /**
   * Capture stack trace
   */
  private captureStackTrace(): string {
    const stack = new Error().stack || '';
    return stack
      .split('\n')
      .slice(3) // Skip DebugStream internal frames
      .join('\n');
  }

  /**
   * Get metrics summary
   */
  getMetrics(): Record<string, { count: number; avgDuration: number }> {
    const summary: Record<string, { count: number; avgDuration: number }> = {};

    for (const [key, value] of this.metrics.entries()) {
      summary[key] = {
        count: value.count,
        avgDuration: value.totalDuration / value.count,
      };
    }

    return summary;
  }

  /**
   * Print metrics summary
   */
  printMetrics(): void {
    console.log('\n' + this.colorize('='.repeat(60), 'cyan'));
    console.log(this.colorize('Performance Metrics Summary', 'cyan'));
    console.log(this.colorize('='.repeat(60), 'cyan') + '\n');

    const metrics = this.getMetrics();
    const sorted = Object.entries(metrics).sort((a, b) => b[1].count - a[1].count);

    console.log(this.colorize('Operation'.padEnd(40) + 'Count'.padEnd(10) + 'Avg Duration', 'white'));
    console.log(this.colorize('-'.repeat(60), 'gray'));

    for (const [key, value] of sorted) {
      const countStr = value.count.toString().padEnd(10);
      const durationStr = value.avgDuration.toFixed(2) + 'ms';
      console.log(
        key.padEnd(40) +
        this.colorize(countStr, 'yellow') +
        this.colorize(durationStr, 'green')
      );
    }

    console.log('\n' + this.colorize('='.repeat(60), 'cyan') + '\n');
  }

  /**
   * Get event buffer
   */
  getEvents(filter?: { category?: string; agentId?: string; since?: Date }): DebugEvent[] {
    let events = [...this.eventBuffer];

    if (filter?.category) {
      events = events.filter(e => e.category === filter.category);
    }

    if (filter?.agentId) {
      events = events.filter(e => e.agentId === filter.agentId);
    }

    if (filter?.since) {
      events = events.filter(e => new Date(e.timestamp) >= filter.since!);
    }

    return events;
  }

  /**
   * Clear event buffer
   */
  clearEvents(): void {
    this.eventBuffer = [];
  }

  /**
   * Clear metrics
   */
  clearMetrics(): void {
    this.metrics.clear();
  }

  /**
   * Close file stream
   */
  close(): void {
    if (this.fileStream) {
      this.fileStream.end();
    }
  }
}

/**
 * Create debug stream with sensible defaults
 */
export function createDebugStream(config?: Partial<DebugStreamConfig>): DebugStream {
  return new DebugStream(config);
}

/**
 * Get debug level from environment variable
 */
export function getDebugLevelFromEnv(): DebugLevel {
  const level = process.env.DEBUG_LEVEL?.toUpperCase();

  switch (level) {
    case 'SILENT': return DebugLevel.SILENT;
    case 'BASIC': return DebugLevel.BASIC;
    case 'DETAILED': return DebugLevel.DETAILED;
    case 'VERBOSE': return DebugLevel.VERBOSE;
    case 'TRACE': return DebugLevel.TRACE;
    default: return DebugLevel.BASIC;
  }
}
