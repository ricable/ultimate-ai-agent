/**
 * Simple in-memory rate limiter for proxy protection
 */

export interface RateLimiterConfig {
  points: number; // Number of requests
  duration: number; // Time window in seconds
  blockDuration: number; // Block duration in seconds when exceeded
}

interface ClientRecord {
  count: number;
  resetTime: number;
  blockedUntil?: number;
}

export class RateLimiter {
  private config: RateLimiterConfig;
  private clients: Map<string, ClientRecord> = new Map();
  private cleanupInterval: NodeJS.Timeout;

  constructor(config: RateLimiterConfig) {
    this.config = config;

    // Cleanup expired entries every minute
    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [key, record] of this.clients.entries()) {
        if (record.resetTime < now && (!record.blockedUntil || record.blockedUntil < now)) {
          this.clients.delete(key);
        }
      }
    }, 60000);
  }

  async consume(key: string): Promise<void> {
    const now = Date.now();
    const record = this.clients.get(key);

    // Check if client is blocked
    if (record?.blockedUntil && record.blockedUntil > now) {
      const remainingMs = record.blockedUntil - now;
      throw new Error(`Rate limit exceeded. Try again in ${Math.ceil(remainingMs / 1000)} seconds`);
    }

    // Initialize or reset record
    if (!record || record.resetTime < now) {
      this.clients.set(key, {
        count: 1,
        resetTime: now + this.config.duration * 1000
      });
      return;
    }

    // Increment count
    record.count++;

    // Check if limit exceeded
    if (record.count > this.config.points) {
      record.blockedUntil = now + this.config.blockDuration * 1000;
      throw new Error(`Rate limit exceeded (${this.config.points} requests per ${this.config.duration}s)`);
    }
  }

  destroy(): void {
    clearInterval(this.cleanupInterval);
    this.clients.clear();
  }
}
