/**
 * Usage Tracker
 * 
 * Monitors and enforces usage limits for Claude job execution
 * Provides detailed analytics and compliance reporting
 */

import fs from 'fs/promises';
import path from 'path';

export class UsageTracker {
  constructor() {
    this.dbPath = path.join(process.cwd(), '.claude-usage.json');
    this.data = {
      dailyUsage: {},
      totalUsage: {
        tasks: 0,
        tokens: 0,
        executionTime: 0,
        errors: 0
      },
      sessions: [],
      limits: {
        dailyTasks: 5,
        dailyTokens: 5000,
        maxTokensPerTask: 1000,
        maxExecutionTime: 300000
      }
    };
    
    this.init();
  }

  async init() {
    try {
      await this.loadData();
    } catch (error) {
      console.warn('Usage tracker initialization warning:', error.message);
      await this.saveData();
    }
  }

  /**
   * Check if usage is within limits
   */
  async checkLimits(requestedLimits = {}) {
    const today = this.getTodayKey();
    const todayUsage = this.data.dailyUsage[today] || { tasks: 0, tokens: 0, executionTime: 0 };
    
    const limits = { ...this.data.limits, ...requestedLimits };
    
    // Check daily task limit
    if (todayUsage.tasks >= limits.dailyTasks) {
      return {
        allowed: false,
        reason: `Daily task limit exceeded (${todayUsage.tasks}/${limits.dailyTasks})`,
        current: todayUsage,
        limits
      };
    }
    
    // Check daily token limit
    if (todayUsage.tokens >= limits.dailyTokens) {
      return {
        allowed: false,
        reason: `Daily token limit exceeded (${todayUsage.tokens}/${limits.dailyTokens})`,
        current: todayUsage,
        limits
      };
    }
    
    return {
      allowed: true,
      current: todayUsage,
      limits,
      remaining: {
        tasks: limits.dailyTasks - todayUsage.tasks,
        tokens: limits.dailyTokens - todayUsage.tokens
      }
    };
  }

  /**
   * Record usage after job execution
   */
  async recordUsage(usage) {
    const today = this.getTodayKey();
    const timestamp = new Date().toISOString();
    
    // Initialize today's usage if not exists
    if (!this.data.dailyUsage[today]) {
      this.data.dailyUsage[today] = {
        tasks: 0,
        tokens: 0,
        executionTime: 0,
        errors: 0,
        sessions: []
      };
    }
    
    // Update daily usage
    this.data.dailyUsage[today].tasks += 1;
    this.data.dailyUsage[today].tokens += usage.tokens || 0;
    this.data.dailyUsage[today].executionTime += usage.executionTime || 0;
    
    if (!usage.success) {
      this.data.dailyUsage[today].errors += 1;
    }
    
    // Update total usage
    this.data.totalUsage.tasks += 1;
    this.data.totalUsage.tokens += usage.tokens || 0;
    this.data.totalUsage.executionTime += usage.executionTime || 0;
    
    if (!usage.success) {
      this.data.totalUsage.errors += 1;
    }
    
    // Record session details
    const session = {
      timestamp,
      date: today,
      tokens: usage.tokens || 0,
      executionTime: usage.executionTime || 0,
      success: usage.success !== false,
      model: usage.model,
      jobId: usage.jobId
    };
    
    this.data.dailyUsage[today].sessions.push(session);
    this.data.sessions.push(session);
    
    // Limit session history to prevent excessive storage
    if (this.data.sessions.length > 1000) {
      this.data.sessions = this.data.sessions.slice(-500);
    }
    
    await this.saveData();
    return session;
  }

  /**
   * Get today's usage statistics
   */
  async getTodayUsage() {
    const today = this.getTodayKey();
    return this.data.dailyUsage[today] || {
      tasks: 0,
      tokens: 0,
      executionTime: 0,
      errors: 0,
      sessions: []
    };
  }

  /**
   * Get usage statistics for a specific date range
   */
  async getUsageRange(startDate, endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const results = [];
    
    for (const [dateKey, usage] of Object.entries(this.data.dailyUsage)) {
      const date = new Date(dateKey);
      if (date >= start && date <= end) {
        results.push({
          date: dateKey,
          ...usage
        });
      }
    }
    
    return results.sort((a, b) => new Date(a.date) - new Date(b.date));
  }

  /**
   * Get weekly usage summary
   */
  async getWeeklyUsage() {
    const endDate = new Date();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 7);
    
    const weeklyData = await this.getUsageRange(startDate, endDate);
    
    return {
      period: {
        start: startDate.toISOString().split('T')[0],
        end: endDate.toISOString().split('T')[0]
      },
      summary: weeklyData.reduce((sum, day) => ({
        tasks: sum.tasks + day.tasks,
        tokens: sum.tokens + day.tokens,
        executionTime: sum.executionTime + day.executionTime,
        errors: sum.errors + day.errors
      }), { tasks: 0, tokens: 0, executionTime: 0, errors: 0 }),
      dailyBreakdown: weeklyData
    };
  }

  /**
   * Get monthly usage summary
   */
  async getMonthlyUsage() {
    const endDate = new Date();
    const startDate = new Date(endDate);
    startDate.setDate(1); // First day of current month
    
    const monthlyData = await this.getUsageRange(startDate, endDate);
    
    return {
      period: {
        start: startDate.toISOString().split('T')[0],
        end: endDate.toISOString().split('T')[0]
      },
      summary: monthlyData.reduce((sum, day) => ({
        tasks: sum.tasks + day.tasks,
        tokens: sum.tokens + day.tokens,
        executionTime: sum.executionTime + day.executionTime,
        errors: sum.errors + day.errors
      }), { tasks: 0, tokens: 0, executionTime: 0, errors: 0 }),
      dailyBreakdown: monthlyData
    };
  }

  /**
   * Update usage limits
   */
  async setLimits(newLimits) {
    this.data.limits = { ...this.data.limits, ...newLimits };
    await this.saveData();
    return this.data.limits;
  }

  /**
   * Get current limits
   */
  getLimits() {
    return { ...this.data.limits };
  }

  /**
   * Reset daily usage (for testing or manual reset)
   */
  async resetDailyUsage() {
    const today = this.getTodayKey();
    this.data.dailyUsage[today] = {
      tasks: 0,
      tokens: 0,
      executionTime: 0,
      errors: 0,
      sessions: []
    };
    await this.saveData();
  }

  /**
   * Reset all usage data
   */
  async reset() {
    this.data = {
      dailyUsage: {},
      totalUsage: {
        tasks: 0,
        tokens: 0,
        executionTime: 0,
        errors: 0
      },
      sessions: [],
      limits: this.data.limits // Preserve limits
    };
    await this.saveData();
  }

  /**
   * Generate usage analytics report
   */
  async generateAnalytics() {
    const today = await this.getTodayUsage();
    const weekly = await this.getWeeklyUsage();
    const monthly = await this.getMonthlyUsage();
    
    // Calculate averages
    const totalDays = Object.keys(this.data.dailyUsage).length;
    const averages = {
      tasksPerDay: totalDays > 0 ? this.data.totalUsage.tasks / totalDays : 0,
      tokensPerDay: totalDays > 0 ? this.data.totalUsage.tokens / totalDays : 0,
      tokensPerTask: this.data.totalUsage.tasks > 0 ? this.data.totalUsage.tokens / this.data.totalUsage.tasks : 0,
      executionTimePerTask: this.data.totalUsage.tasks > 0 ? this.data.totalUsage.executionTime / this.data.totalUsage.tasks : 0
    };
    
    // Calculate success rate
    const successRate = this.data.totalUsage.tasks > 0 
      ? ((this.data.totalUsage.tasks - this.data.totalUsage.errors) / this.data.totalUsage.tasks) * 100 
      : 100;
    
    // Find peak usage days
    const peakDays = Object.entries(this.data.dailyUsage)
      .sort(([,a], [,b]) => b.tasks - a.tasks)
      .slice(0, 5)
      .map(([date, usage]) => ({ date, ...usage }));
    
    // Model usage distribution
    const modelUsage = {};
    this.data.sessions.forEach(session => {
      if (session.model) {
        modelUsage[session.model] = (modelUsage[session.model] || 0) + 1;
      }
    });
    
    return {
      timestamp: new Date().toISOString(),
      summary: {
        total: this.data.totalUsage,
        today,
        weekly: weekly.summary,
        monthly: monthly.summary
      },
      averages,
      successRate,
      peakDays,
      modelUsage,
      limits: this.data.limits,
      compliance: {
        withinDailyLimits: today.tasks <= this.data.limits.dailyTasks && today.tokens <= this.data.limits.dailyTokens,
        limitsRatio: {
          tasks: this.data.limits.dailyTasks > 0 ? (today.tasks / this.data.limits.dailyTasks) * 100 : 0,
          tokens: this.data.limits.dailyTokens > 0 ? (today.tokens / this.data.limits.dailyTokens) * 100 : 0
        }
      }
    };
  }

  /**
   * Export usage data for external analysis
   */
  async exportData(format = 'json') {
    const analytics = await this.generateAnalytics();
    
    if (format === 'csv') {
      return this.convertToCSV(analytics);
    }
    
    return analytics;
  }

  /**
   * Clean up old usage data
   */
  async cleanupOldData(retentionDays = 90) {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - retentionDays);
    const cutoffKey = this.dateToKey(cutoffDate);
    
    const keysToDelete = Object.keys(this.data.dailyUsage)
      .filter(key => key < cutoffKey);
    
    keysToDelete.forEach(key => {
      delete this.data.dailyUsage[key];
    });
    
    // Clean up sessions
    this.data.sessions = this.data.sessions.filter(session => {
      const sessionDate = new Date(session.timestamp);
      return sessionDate >= cutoffDate;
    });
    
    if (keysToDelete.length > 0) {
      await this.saveData();
    }
    
    return {
      deletedDays: keysToDelete.length,
      retentionDays,
      remainingDays: Object.keys(this.data.dailyUsage).length
    };
  }

  /**
   * Get usage trends and predictions
   */
  async getUsageTrends() {
    const last30Days = await this.getUsageRange(
      new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      new Date()
    );
    
    if (last30Days.length < 7) {
      return {
        trends: 'Insufficient data for trend analysis',
        prediction: null
      };
    }
    
    // Simple linear trend calculation
    const taskTrend = this.calculateTrend(last30Days.map(d => d.tasks));
    const tokenTrend = this.calculateTrend(last30Days.map(d => d.tokens));
    
    // Predict next week based on trend
    const avgTasks = taskTrend.average;
    const avgTokens = tokenTrend.average;
    
    return {
      trends: {
        tasks: {
          direction: taskTrend.slope > 0 ? 'increasing' : taskTrend.slope < 0 ? 'decreasing' : 'stable',
          slope: taskTrend.slope,
          average: avgTasks
        },
        tokens: {
          direction: tokenTrend.slope > 0 ? 'increasing' : tokenTrend.slope < 0 ? 'decreasing' : 'stable',
          slope: tokenTrend.slope,
          average: avgTokens
        }
      },
      prediction: {
        nextWeek: {
          estimatedTasks: Math.max(0, Math.round(avgTasks * 7 + taskTrend.slope * 3.5)),
          estimatedTokens: Math.max(0, Math.round(avgTokens * 7 + tokenTrend.slope * 3.5))
        }
      }
    };
  }

  // Private helper methods

  getTodayKey() {
    return new Date().toISOString().split('T')[0];
  }

  dateToKey(date) {
    return date.toISOString().split('T')[0];
  }

  async loadData() {
    try {
      const content = await fs.readFile(this.dbPath, 'utf8');
      const loadedData = JSON.parse(content);
      this.data = { ...this.data, ...loadedData };
    } catch (error) {
      // File doesn't exist yet, use defaults
    }
  }

  async saveData() {
    try {
      await fs.writeFile(this.dbPath, JSON.stringify(this.data, null, 2));
    } catch (error) {
      console.warn('Failed to save usage data:', error.message);
    }
  }

  convertToCSV(analytics) {
    const lines = ['Date,Tasks,Tokens,Execution Time,Errors,Success Rate'];
    
    // Add daily data
    Object.entries(this.data.dailyUsage).forEach(([date, usage]) => {
      const successRate = usage.tasks > 0 ? ((usage.tasks - usage.errors) / usage.tasks) * 100 : 100;
      lines.push(`${date},${usage.tasks},${usage.tokens},${usage.executionTime},${usage.errors},${successRate.toFixed(2)}`);
    });
    
    return lines.join('\n');
  }

  calculateTrend(values) {
    const n = values.length;
    const sumX = n * (n + 1) / 2;
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = values.reduce((sum, y, i) => sum + (i + 1) * y, 0);
    const sumXX = n * (n + 1) * (2 * n + 1) / 6;
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const average = sumY / n;
    
    return { slope, average };
  }
}