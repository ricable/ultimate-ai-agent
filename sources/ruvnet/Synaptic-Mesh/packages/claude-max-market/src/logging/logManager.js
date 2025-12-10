/**
 * Log Manager
 * 
 * Comprehensive logging system for compliance and audit trails
 */

import fs from 'fs/promises';
import path from 'path';

export class LogManager {
  constructor() {
    this.logDir = path.join(process.cwd(), '.claude-logs');
    this.maxLogSize = 10 * 1024 * 1024; // 10MB
    this.maxLogFiles = 10;
    this.logLevel = 'info';
    this.logs = [];
    this.auditLogs = [];
  }

  async init() {
    try {
      await fs.mkdir(this.logDir, { recursive: true });
    } catch (error) {
      console.warn('Failed to create log directory:', error.message);
    }
  }

  /**
   * Log information message
   */
  info(message, metadata = {}) {
    this.log('info', message, metadata);
  }

  /**
   * Log warning message
   */
  warn(message, metadata = {}) {
    this.log('warn', message, metadata);
  }

  /**
   * Log error message
   */
  error(message, metadata = {}) {
    this.log('error', message, metadata);
  }

  /**
   * Log debug message
   */
  debug(message, metadata = {}) {
    if (this.logLevel === 'debug') {
      this.log('debug', message, metadata);
    }
  }

  /**
   * Log audit event
   */
  audit(event, details = {}) {
    const auditEntry = {
      timestamp: new Date().toISOString(),
      event,
      details,
      sessionId: this.getCurrentSessionId(),
      userId: details.userId || 'system'
    };

    this.auditLogs.push(auditEntry);
    this.log('audit', `AUDIT: ${event}`, details);
    
    // Write to audit file immediately for compliance
    this.writeAuditLog(auditEntry);
  }

  /**
   * Generic log method
   */
  log(level, message, metadata = {}) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      metadata,
      sessionId: this.getCurrentSessionId()
    };

    this.logs.push(logEntry);
    
    // Console output with colors
    this.outputToConsole(logEntry);
    
    // Write to file
    this.writeToFile(logEntry);
    
    // Limit in-memory logs
    if (this.logs.length > 1000) {
      this.logs = this.logs.slice(-500);
    }
  }

  /**
   * Get recent logs
   */
  async getLogs(options = {}) {
    const { tail = 50, level = null, follow = false } = options;
    
    let filteredLogs = this.logs;
    
    if (level) {
      filteredLogs = filteredLogs.filter(log => log.level === level);
    }
    
    const result = filteredLogs.slice(-tail);
    
    if (follow) {
      // In a real implementation, this would set up a stream
      // For now, just return the logs
    }
    
    return result;
  }

  /**
   * Get audit trail
   */
  async getAuditTrail(options = {}) {
    const { startDate, endDate, event, userId } = options;
    
    let filteredAudits = this.auditLogs;
    
    if (startDate) {
      filteredAudits = filteredAudits.filter(audit => 
        new Date(audit.timestamp) >= new Date(startDate)
      );
    }
    
    if (endDate) {
      filteredAudits = filteredAudits.filter(audit => 
        new Date(audit.timestamp) <= new Date(endDate)
      );
    }
    
    if (event) {
      filteredAudits = filteredAudits.filter(audit => 
        audit.event.includes(event)
      );
    }
    
    if (userId) {
      filteredAudits = filteredAudits.filter(audit => 
        audit.userId === userId
      );
    }
    
    return filteredAudits.sort((a, b) => 
      new Date(b.timestamp) - new Date(a.timestamp)
    );
  }

  /**
   * Generate compliance report from logs
   */
  async generateComplianceReport() {
    const auditTrail = await this.getAuditTrail();
    const now = new Date();
    const last24Hours = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    
    const recentAudits = auditTrail.filter(audit => 
      new Date(audit.timestamp) >= last24Hours
    );
    
    // Categorize audit events
    const eventCategories = {
      userActions: recentAudits.filter(a => a.event.includes('user_')),
      systemEvents: recentAudits.filter(a => a.event.includes('system_')),
      jobExecutions: recentAudits.filter(a => a.event.includes('job_')),
      complianceChecks: recentAudits.filter(a => a.event.includes('compliance_'))
    };
    
    return {
      timestamp: now.toISOString(),
      period: {
        start: last24Hours.toISOString(),
        end: now.toISOString()
      },
      summary: {
        totalEvents: recentAudits.length,
        userActions: eventCategories.userActions.length,
        systemEvents: eventCategories.systemEvents.length,
        jobExecutions: eventCategories.jobExecutions.length,
        complianceChecks: eventCategories.complianceChecks.length
      },
      events: eventCategories,
      auditTrail: recentAudits
    };
  }

  /**
   * Search logs
   */
  async searchLogs(query, options = {}) {
    const { level, startDate, endDate, limit = 100 } = options;
    
    let searchLogs = this.logs;
    
    if (level) {
      searchLogs = searchLogs.filter(log => log.level === level);
    }
    
    if (startDate) {
      searchLogs = searchLogs.filter(log => 
        new Date(log.timestamp) >= new Date(startDate)
      );
    }
    
    if (endDate) {
      searchLogs = searchLogs.filter(log => 
        new Date(log.timestamp) <= new Date(endDate)
      );
    }
    
    // Simple text search in message and metadata
    const queryLower = query.toLowerCase();
    const results = searchLogs.filter(log => {
      const messageMatch = log.message.toLowerCase().includes(queryLower);
      const metadataMatch = JSON.stringify(log.metadata).toLowerCase().includes(queryLower);
      return messageMatch || metadataMatch;
    });
    
    return results.slice(0, limit);
  }

  /**
   * Export logs to file
   */
  async exportLogs(filePath, options = {}) {
    const { format = 'json', level, startDate, endDate } = options;
    
    let exportLogs = this.logs;
    
    if (level) {
      exportLogs = exportLogs.filter(log => log.level === level);
    }
    
    if (startDate) {
      exportLogs = exportLogs.filter(log => 
        new Date(log.timestamp) >= new Date(startDate)
      );
    }
    
    if (endDate) {
      exportLogs = exportLogs.filter(log => 
        new Date(log.timestamp) <= new Date(endDate)
      );
    }
    
    let content;
    if (format === 'csv') {
      content = this.convertToCSV(exportLogs);
    } else {
      content = JSON.stringify(exportLogs, null, 2);
    }
    
    await fs.writeFile(filePath, content);
    return exportLogs.length;
  }

  /**
   * Clean up old log files
   */
  async cleanupOldLogs(retentionDays = 30) {
    try {
      const files = await fs.readdir(this.logDir);
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - retentionDays);
      
      let deletedCount = 0;
      
      for (const file of files) {
        const filePath = path.join(this.logDir, file);
        const stats = await fs.stat(filePath);
        
        if (stats.mtime < cutoffDate) {
          await fs.unlink(filePath);
          deletedCount++;
        }
      }
      
      return { filesRemoved: deletedCount };
    } catch (error) {
      console.warn('Log cleanup failed:', error.message);
      return { filesRemoved: 0 };
    }
  }

  /**
   * Set log level
   */
  setLogLevel(level) {
    const validLevels = ['debug', 'info', 'warn', 'error'];
    if (validLevels.includes(level)) {
      this.logLevel = level;
      this.info(`Log level set to: ${level}`);
    } else {
      throw new Error(`Invalid log level: ${level}. Valid levels: ${validLevels.join(', ')}`);
    }
  }

  /**
   * Rotate log files
   */
  async rotateLogFiles() {
    try {
      const currentLogFile = path.join(this.logDir, 'claude-max.log');
      
      try {
        const stats = await fs.stat(currentLogFile);
        if (stats.size >= this.maxLogSize) {
          const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
          const rotatedFile = path.join(this.logDir, `claude-max.${timestamp}.log`);
          
          await fs.rename(currentLogFile, rotatedFile);
          this.info('Log file rotated', { rotatedTo: rotatedFile });
          
          // Clean up old rotated files
          await this.cleanupRotatedFiles();
        }
      } catch (error) {
        // Current log file doesn't exist yet
      }
    } catch (error) {
      console.warn('Log rotation failed:', error.message);
    }
  }

  // Private helper methods

  outputToConsole(logEntry) {
    const timestamp = new Date(logEntry.timestamp).toLocaleString();
    const level = logEntry.level.toUpperCase().padEnd(5);
    
    // Simple console output - in production, use a proper logging library
    console.log(`${timestamp} [${level}] ${logEntry.message}`);
    
    if (Object.keys(logEntry.metadata).length > 0) {
      console.log('  Metadata:', JSON.stringify(logEntry.metadata, null, 2));
    }
  }

  async writeToFile(logEntry) {
    try {
      const logFile = path.join(this.logDir, 'claude-max.log');
      const logLine = JSON.stringify(logEntry) + '\n';
      
      await fs.appendFile(logFile, logLine);
      
      // Check if rotation is needed
      this.rotateLogFiles();
    } catch (error) {
      console.warn('Failed to write to log file:', error.message);
    }
  }

  async writeAuditLog(auditEntry) {
    try {
      const auditFile = path.join(this.logDir, 'audit.log');
      const auditLine = JSON.stringify(auditEntry) + '\n';
      
      await fs.appendFile(auditFile, auditLine);
    } catch (error) {
      console.warn('Failed to write to audit log:', error.message);
    }
  }

  getCurrentSessionId() {
    // Simple session ID based on process start time
    return `session_${process.pid}_${Date.now()}`;
  }

  convertToCSV(logs) {
    const headers = ['Timestamp', 'Level', 'Message', 'Session ID', 'Metadata'];
    const lines = [headers.join(',')];
    
    logs.forEach(log => {
      const row = [
        log.timestamp,
        log.level,
        `"${log.message.replace(/"/g, '""')}"`,
        log.sessionId,
        `"${JSON.stringify(log.metadata).replace(/"/g, '""')}"`
      ];
      lines.push(row.join(','));
    });
    
    return lines.join('\n');
  }

  async cleanupRotatedFiles() {
    try {
      const files = await fs.readdir(this.logDir);
      const rotatedFiles = files
        .filter(file => file.startsWith('claude-max.') && file.endsWith('.log'))
        .map(file => ({
          name: file,
          path: path.join(this.logDir, file)
        }))
        .sort((a, b) => b.name.localeCompare(a.name)); // Sort by name (timestamp)
      
      // Keep only the most recent files
      if (rotatedFiles.length > this.maxLogFiles) {
        const filesToDelete = rotatedFiles.slice(this.maxLogFiles);
        
        for (const file of filesToDelete) {
          await fs.unlink(file.path);
        }
      }
    } catch (error) {
      console.warn('Failed to cleanup rotated files:', error.message);
    }
  }
}