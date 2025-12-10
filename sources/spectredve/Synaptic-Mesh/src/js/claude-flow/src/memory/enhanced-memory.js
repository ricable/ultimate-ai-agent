/**
 * Enhanced Memory Functions for Comprehensive Swarm Coordination
 */

import { SqliteMemoryStore } from './sqlite-store.js';

export class EnhancedMemory extends SqliteMemoryStore {
  constructor(options = {}) {
    super(options);
  }

  async initialize() {
    await super.initialize();
    // Apply enhanced schema
    try {
      const { readFileSync } = await import('fs');
      const schemaPath = new URL('./enhanced-schema.sql', import.meta.url);
      const schema = readFileSync(schemaPath, 'utf-8');
      this.db.exec(schema);
    } catch (error) {
      console.error(`[${new Date().toISOString()}] ERROR [enhanced-memory] Failed to apply enhanced schema:`, error);
      // Continue with basic schema only
    }
  }

  // === SESSION MANAGEMENT ===
  
  async saveSessionState(sessionId, state) {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO session_state 
      (session_id, user_id, project_path, active_branch, last_activity, state, context, environment)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    return stmt.run(
      sessionId,
      state.userId || process.env.USER,
      state.projectPath || process.cwd(),
      state.activeBranch || 'main',
      Date.now(),
      state.state || 'active',
      JSON.stringify(state.context || {}),
      JSON.stringify(state.environment || process.env)
    );
  }

  async resumeSession(sessionId) {
    const stmt = this.db.prepare('SELECT * FROM session_state WHERE session_id = ?');
    const session = stmt.get(sessionId);
    if (session) {
      session.context = JSON.parse(session.context);
      session.environment = JSON.parse(session.environment);
    }
    return session;
  }

  async getActiveSessions() {
    const stmt = this.db.prepare('SELECT * FROM active_sessions');
    return stmt.all();
  }

  // === MCP TOOL TRACKING ===
  
  async trackToolUsage(toolName, args, result, executionTime, success = true, error = null) {
    const stmt = this.db.prepare(`
      INSERT INTO mcp_tool_usage 
      (tool_name, session_id, arguments, result_summary, execution_time_ms, success, error_message)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    
    return stmt.run(
      toolName,
      this.currentSessionId,
      JSON.stringify(args),
      result ? JSON.stringify(result).substring(0, 500) : null,
      executionTime,
      success ? 1 : 0,
      error
    );
  }

  async getToolStats() {
    const stmt = this.db.prepare('SELECT * FROM tool_effectiveness');
    return stmt.all();
  }

  // === TRAINING DATA ===
  
  async recordTrainingExample(patternType, inputContext, actionTaken, outcome, score, feedback = null) {
    const stmt = this.db.prepare(`
      INSERT INTO training_data 
      (pattern_type, input_context, action_taken, outcome, success_score, model_version, feedback)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    
    return stmt.run(
      patternType,
      JSON.stringify(inputContext),
      JSON.stringify(actionTaken),
      JSON.stringify(outcome),
      score,
      this.modelVersion || '1.0',
      feedback
    );
  }

  async getTrainingData(patternType, limit = 100) {
    const stmt = this.db.prepare(`
      SELECT * FROM training_data 
      WHERE pattern_type = ? 
      ORDER BY success_score DESC, timestamp DESC 
      LIMIT ?
    `);
    return stmt.all(patternType, limit);
  }

  // === CODE PATTERNS ===
  
  async recordCodePattern(filePath, patternName, content, language) {
    const stmt = this.db.prepare(`
      INSERT INTO code_patterns (file_path, pattern_name, pattern_content, language, last_used)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(file_path, pattern_name) DO UPDATE SET
        frequency = frequency + 1,
        last_used = ?
    `);
    
    const now = Date.now();
    return stmt.run(filePath, patternName, content, language, now, now);
  }

  async findSimilarPatterns(language, limit = 10) {
    const stmt = this.db.prepare(`
      SELECT * FROM code_patterns 
      WHERE language = ? 
      ORDER BY effectiveness_score DESC, frequency DESC 
      LIMIT ?
    `);
    return stmt.all(language, limit);
  }

  // === AGENT COLLABORATION ===
  
  async recordAgentInteraction(sourceAgent, targetAgent, messageType, content, taskId) {
    const stmt = this.db.prepare(`
      INSERT INTO agent_interactions 
      (source_agent, target_agent, message_type, content, task_id, correlation_id)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    
    const correlationId = `${taskId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    return stmt.run(sourceAgent, targetAgent, messageType, JSON.stringify(content), taskId, correlationId);
  }

  async getAgentConversation(taskId) {
    const stmt = this.db.prepare(`
      SELECT * FROM agent_interactions 
      WHERE task_id = ? 
      ORDER BY timestamp ASC
    `);
    return stmt.all(taskId);
  }

  // === KNOWLEDGE GRAPH ===
  
  async addKnowledgeEntity(entityType, entityName, entityPath, relationships = [], metadata = {}) {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO knowledge_graph 
      (entity_type, entity_name, entity_path, relationships, metadata, updated_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    
    return stmt.run(
      entityType,
      entityName,
      entityPath,
      JSON.stringify(relationships),
      JSON.stringify(metadata),
      Date.now()
    );
  }

  async findRelatedEntities(entityName, depth = 1) {
    // Simple implementation - could be enhanced with recursive CTEs
    const stmt = this.db.prepare(`
      SELECT * FROM knowledge_graph 
      WHERE entity_name = ? OR relationships LIKE ?
    `);
    return stmt.all(entityName, `%"${entityName}"%`);
  }

  // === ERROR LEARNING ===
  
  async recordError(errorType, errorMessage, stackTrace, context, resolution = null) {
    const stmt = this.db.prepare(`
      INSERT INTO error_patterns 
      (error_type, error_message, stack_trace, context, resolution)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(error_type, error_message) DO UPDATE SET
        occurrence_count = occurrence_count + 1,
        last_seen = strftime('%s', 'now'),
        resolution = COALESCE(?, resolution)
    `);
    
    return stmt.run(
      errorType,
      errorMessage,
      stackTrace,
      JSON.stringify(context),
      resolution,
      resolution
    );
  }

  async getErrorSolutions(errorType) {
    const stmt = this.db.prepare(`
      SELECT * FROM error_patterns 
      WHERE error_type = ? AND resolution IS NOT NULL
      ORDER BY occurrence_count DESC
    `);
    return stmt.all(errorType);
  }

  // === TASK DEPENDENCIES ===
  
  async addTaskDependency(taskId, dependsOn, dependencyType = 'blocking') {
    const stmt = this.db.prepare(`
      INSERT INTO task_dependencies 
      (task_id, depends_on, dependency_type, status)
      VALUES (?, ?, ?, 'pending')
    `);
    
    return stmt.run(taskId, dependsOn, dependencyType);
  }

  async updateDependencyStatus(taskId, dependsOn, status) {
    const stmt = this.db.prepare(`
      UPDATE task_dependencies 
      SET status = ? 
      WHERE task_id = ? AND depends_on = ?
    `);
    
    return stmt.run(status, taskId, dependsOn);
  }

  async getTaskDependencies(taskId) {
    const stmt = this.db.prepare(`
      SELECT * FROM task_dependencies 
      WHERE task_id = ?
    `);
    return stmt.all(taskId);
  }

  // === PERFORMANCE TRACKING ===
  
  async recordPerformance(operationType, details, duration, memoryUsed, cpuPercent) {
    const stmt = this.db.prepare(`
      INSERT INTO performance_benchmarks 
      (operation_type, operation_details, duration_ms, memory_used_mb, cpu_percent, session_id)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    
    return stmt.run(
      operationType,
      JSON.stringify(details),
      duration,
      memoryUsed,
      cpuPercent,
      this.currentSessionId
    );
  }

  async getPerformanceTrends(operationType, days = 7) {
    const stmt = this.db.prepare(`
      SELECT DATE(timestamp, 'unixepoch') as date,
             AVG(duration_ms) as avg_duration,
             AVG(memory_used_mb) as avg_memory,
             COUNT(*) as operation_count
      FROM performance_benchmarks
      WHERE operation_type = ? 
        AND timestamp > strftime('%s', 'now', '-' || ? || ' days')
      GROUP BY date
      ORDER BY date DESC
    `);
    return stmt.all(operationType, days);
  }

  // === USER PREFERENCES ===
  
  async learnPreference(key, value, category, source = 'inferred', confidence = 0.8) {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO user_preferences 
      (preference_key, preference_value, category, learned_from, confidence_score)
      VALUES (?, ?, ?, ?, ?)
    `);
    
    return stmt.run(key, value, category, source, confidence);
  }

  async getPreferences(category = null) {
    if (category) {
      const stmt = this.db.prepare('SELECT * FROM user_preferences WHERE category = ?');
      return stmt.all(category);
    } else {
      const stmt = this.db.prepare('SELECT * FROM user_preferences');
      return stmt.all();
    }
  }

  // === UTILITY METHODS ===
  
  async getDatabaseStats() {
    const tables = [
      'memory_entries', 'session_state', 'mcp_tool_usage', 'training_data',
      'code_patterns', 'agent_interactions', 'knowledge_graph', 'error_patterns',
      'task_dependencies', 'performance_benchmarks', 'user_preferences'
    ];
    
    const stats = {};
    for (const table of tables) {
      try {
        const result = this.db.prepare(`SELECT COUNT(*) as count FROM ${table}`).get();
        stats[table] = result.count;
      } catch (e) {
        stats[table] = 0;
      }
    }
    
    return stats;
  }

  async exportSessionData(sessionId) {
    const data = {
      session: await this.resumeSession(sessionId),
      tools: this.db.prepare('SELECT * FROM mcp_tool_usage WHERE session_id = ?').all(sessionId),
      performance: this.db.prepare('SELECT * FROM performance_benchmarks WHERE session_id = ?').all(sessionId),
      interactions: this.db.prepare('SELECT * FROM agent_interactions WHERE correlation_id LIKE ?').all(`${sessionId}%`),
      timestamp: new Date().toISOString()
    };
    
    return data;
  }
}

// Singleton instance
export const enhancedMemory = new EnhancedMemory();

export default EnhancedMemory;