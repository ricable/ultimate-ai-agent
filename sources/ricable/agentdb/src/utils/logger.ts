/**
 * Simple Logger Utility
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

export class Logger {
  private context: string;
  private static globalLevel: LogLevel = LogLevel.INFO;

  constructor(context: string) {
    this.context = context;
  }

  private shouldLog(level: LogLevel): boolean {
    return level >= Logger.globalLevel;
  }

  private formatMessage(level: string, message: string, meta?: any): string {
    const timestamp = new Date().toISOString();
    const metaStr = meta ? ` ${JSON.stringify(meta)}` : '';
    return `[${timestamp}] [${level}] [${this.context}] ${message}${metaStr}`;
  }

  debug(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.debug(this.formatMessage('DEBUG', message, meta));
    }
  }

  info(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.INFO)) {
      console.info(this.formatMessage('INFO', message, meta));
    }
  }

  warn(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.WARN)) {
      console.warn(this.formatMessage('WARN', message, meta));
    }
  }

  error(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.ERROR)) {
      console.error(this.formatMessage('ERROR', message, meta));
    }
  }

  static setGlobalLevel(level: LogLevel): void {
    Logger.globalLevel = level;
  }
}