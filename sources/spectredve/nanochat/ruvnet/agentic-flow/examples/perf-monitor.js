#!/usr/bin/env node
/**
 * Performance Monitor
 * Tracks and reports performance metrics
 */

class PerformanceMonitor {
  constructor() {
    this.metrics = {
      queries: [],
      stores: [],
      errors: []
    };
  }

  recordQuery(duration, query, results) {
    this.metrics.queries.push({
      timestamp: Date.now(),
      duration,
      query,
      resultCount: results
    });
  }

  recordStore(duration, key, size) {
    this.metrics.stores.push({
      timestamp: Date.now(),
      duration,
      key,
      size
    });
  }

  recordError(operation, error) {
    this.metrics.errors.push({
      timestamp: Date.now(),
      operation,
      error: error.message
    });
  }

  getReport() {
    const queries = this.metrics.queries;
    const stores = this.metrics.stores;

    const avgQueryTime = queries.length > 0
      ? queries.reduce((sum, q) => sum + q.duration, 0) / queries.length
      : 0;

    const avgStoreTime = stores.length > 0
      ? stores.reduce((sum, s) => sum + s.duration, 0) / stores.length
      : 0;

    return {
      summary: {
        totalQueries: queries.length,
        totalStores: stores.length,
        totalErrors: this.metrics.errors.length,
        avgQueryTime: avgQueryTime.toFixed(2) + 'ms',
        avgStoreTime: avgStoreTime.toFixed(2) + 'ms'
      },
      queries: queries.slice(-10), // Last 10
      stores: stores.slice(-10),
      errors: this.metrics.errors
    };
  }

  exportReport(filename = '/tmp/perf-report.json') {
    const fs = require('fs');
    fs.writeFileSync(filename, JSON.stringify(this.getReport(), null, 2));
    console.log(`📊 Performance report exported to: ${filename}`);
  }
}

module.exports = { PerformanceMonitor };
