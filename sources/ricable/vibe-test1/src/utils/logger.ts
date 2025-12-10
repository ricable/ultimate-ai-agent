/**
 * Logging Utility for RAN Swarm Platform
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_COLORS = {
  debug: '\x1b[36m',  // Cyan
  info: '\x1b[32m',   // Green
  warn: '\x1b[33m',   // Yellow
  error: '\x1b[31m',  // Red
  reset: '\x1b[0m'
};

export class Logger {
  private context: string;
  private minLevel: LogLevel;

  private levelPriority: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3
  };

  constructor(context: string, minLevel: LogLevel = 'info') {
    this.context = context;
    this.minLevel = minLevel;
  }

  debug(message: string, data?: any): void {
    this.log('debug', message, data);
  }

  info(message: string, data?: any): void {
    this.log('info', message, data);
  }

  warn(message: string, data?: any): void {
    this.log('warn', message, data);
  }

  error(message: string, data?: any): void {
    this.log('error', message, data);
  }

  private log(level: LogLevel, message: string, data?: any): void {
    if (this.levelPriority[level] < this.levelPriority[this.minLevel]) {
      return;
    }

    const timestamp = new Date().toISOString();
    const color = LOG_COLORS[level];
    const reset = LOG_COLORS.reset;

    const logMessage = `${color}[${timestamp}] [${level.toUpperCase()}] [${this.context}]${reset} ${message}`;

    console.log(logMessage);
    if (data) {
      console.log(JSON.stringify(data, null, 2));
    }
  }
}

export const createLogger = (context: string, minLevel?: LogLevel) => new Logger(context, minLevel);
