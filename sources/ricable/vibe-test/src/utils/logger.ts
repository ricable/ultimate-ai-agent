/**
 * High-performance structured logger for RAN operations
 */

export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
}

export interface LogEntry {
  timestamp: number;
  level: LogLevel;
  component: string;
  message: string;
  metadata?: Record<string, unknown>;
  traceId?: string;
}

const levelNames: Record<LogLevel, string> = {
  [LogLevel.TRACE]: 'TRACE',
  [LogLevel.DEBUG]: 'DEBUG',
  [LogLevel.INFO]: 'INFO',
  [LogLevel.WARN]: 'WARN',
  [LogLevel.ERROR]: 'ERROR',
  [LogLevel.FATAL]: 'FATAL',
};

const levelColors: Record<LogLevel, string> = {
  [LogLevel.TRACE]: '\x1b[90m',
  [LogLevel.DEBUG]: '\x1b[36m',
  [LogLevel.INFO]: '\x1b[32m',
  [LogLevel.WARN]: '\x1b[33m',
  [LogLevel.ERROR]: '\x1b[31m',
  [LogLevel.FATAL]: '\x1b[35m',
};

const RESET = '\x1b[0m';

export class Logger {
  private component: string;
  private minLevel: LogLevel;
  private traceId?: string;

  constructor(component: string, minLevel: LogLevel = LogLevel.INFO) {
    this.component = component;
    this.minLevel = minLevel;
  }

  setTraceId(traceId: string): void {
    this.traceId = traceId;
  }

  private log(level: LogLevel, message: string, metadata?: Record<string, unknown>): void {
    if (level < this.minLevel) return;

    const entry: LogEntry = {
      timestamp: Date.now(),
      level,
      component: this.component,
      message,
      metadata,
      traceId: this.traceId,
    };

    const color = levelColors[level];
    const levelStr = levelNames[level].padEnd(5);
    const time = new Date(entry.timestamp).toISOString();
    const comp = this.component.padEnd(20);
    const trace = this.traceId ? ` [${this.traceId.slice(0, 8)}]` : '';
    const meta = metadata ? ` ${JSON.stringify(metadata)}` : '';

    console.log(`${color}${time} ${levelStr}${RESET} [${comp}]${trace} ${message}${meta}`);
  }

  trace(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.TRACE, message, metadata);
  }

  debug(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.DEBUG, message, metadata);
  }

  info(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.INFO, message, metadata);
  }

  warn(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.WARN, message, metadata);
  }

  error(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.ERROR, message, metadata);
  }

  fatal(message: string, metadata?: Record<string, unknown>): void {
    this.log(LogLevel.FATAL, message, metadata);
  }

  child(component: string): Logger {
    const child = new Logger(`${this.component}:${component}`, this.minLevel);
    if (this.traceId) child.setTraceId(this.traceId);
    return child;
  }
}

export const createLogger = (component: string): Logger => new Logger(component);
