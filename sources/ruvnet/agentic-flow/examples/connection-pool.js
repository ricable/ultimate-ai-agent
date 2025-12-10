#!/usr/bin/env node
/**
 * Connection Pool Manager
 * Reuses database connections for better performance
 */

class ConnectionPool {
  constructor(maxConnections = 5) {
    this.maxConnections = maxConnections;
    this.activeConnections = 0;
    this.queue = [];
  }

  async acquire() {
    return new Promise((resolve) => {
      if (this.activeConnections < this.maxConnections) {
        this.activeConnections++;
        resolve({
          id: this.activeConnections,
          release: () => this.release()
        });
      } else {
        this.queue.push(resolve);
      }
    });
  }

  release() {
    this.activeConnections--;

    if (this.queue.length > 0) {
      const next = this.queue.shift();
      this.activeConnections++;
      next({
        id: this.activeConnections,
        release: () => this.release()
      });
    }
  }

  stats() {
    return {
      active: this.activeConnections,
      queued: this.queue.length,
      max: this.maxConnections
    };
  }
}

module.exports = { ConnectionPool };
