/**
 * SQLite Database Utilities for Research Jobs
 * Replaces Supabase client with local SQLite database
 */

import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Database file path
const DB_PATH = path.join(__dirname, '../data/research-jobs.db');
const SCHEMA_PATH = path.join(__dirname, '../schema/research-jobs.sql');

/**
 * Initialize database connection
 */
export function initDatabase() {
  const dataDir = path.dirname(DB_PATH);
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  const db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');

  const schema = fs.readFileSync(SCHEMA_PATH, 'utf8');
  db.exec(schema);

  console.log('✅ SQLite database initialized:', DB_PATH);
  return db;
}

/**
 * Get database connection
 */
export function getDatabase() {
  if (!fs.existsSync(DB_PATH)) {
    return initDatabase();
  }
  return new Database(DB_PATH);
}

/**
 * Create a new job record
 */
export function createJob(jobData) {
  const db = getDatabase();

  const stmt = db.prepare(`
    INSERT INTO research_jobs (
      id, agent, task, location, status, progress,
      current_message, created_at, started_at, metadata
    ) VALUES (
      @id, @agent, @task, @location, @status, @progress,
      @currentMessage, @createdAt, @startedAt, @metadata
    )
  `);

  const result = stmt.run({
    id: jobData.id,
    agent: jobData.agent,
    task: jobData.task,
    location: jobData.location || jobData.task,
    status: jobData.status || 'pending',
    progress: jobData.progress || 0,
    currentMessage: jobData.currentMessage || 'Job initialized',
    createdAt: jobData.createdAt || new Date().toISOString(),
    startedAt: jobData.startedAt || new Date().toISOString(),
    metadata: jobData.metadata ? JSON.stringify(jobData.metadata) : null
  });

  db.close();
  return result.changes > 0;
}

/**
 * Update job progress
 */
export function updateProgress(jobId, progress, message, additionalData = {}) {
  const db = getDatabase();

  const updates = {
    progress: Math.min(progress, 95),
    current_message: message,
    last_update: new Date().toISOString()
  };

  // Add additional data fields
  Object.keys(additionalData).forEach(key => {
    const snakeKey = key.replace(/([A-Z])/g, '_$1').toLowerCase();
    updates[snakeKey] = additionalData[key];
  });

  const fields = Object.keys(updates).map(key => `${key} = @${key}`).join(', ');
  const stmt = db.prepare(`UPDATE research_jobs SET ${fields} WHERE id = @id`);

  const result = stmt.run({ id: jobId, ...updates });
  db.close();

  return result.changes > 0;
}

/**
 * Mark job as complete
 */
export function markComplete(jobId, completionData) {
  const db = getDatabase();

  const stmt = db.prepare(`
    UPDATE research_jobs
    SET status = @status,
        progress = 100,
        current_message = @currentMessage,
        exit_code = @exitCode,
        execution_log = @executionLog,
        report_content = @reportContent,
        report_format = @reportFormat,
        report_path = @reportPath,
        duration_seconds = @durationSeconds,
        grounding_score = @groundingScore,
        tokens_used = @tokensUsed,
        memory_mb = @memoryMb,
        completed_at = @completedAt,
        last_update = @lastUpdate
    WHERE id = @id
  `);

  const result = stmt.run({
    id: jobId,
    status: completionData.exitCode === 0 ? 'completed' : 'failed',
    currentMessage: completionData.exitCode === 0
      ? 'Research completed successfully'
      : `Job failed with exit code ${completionData.exitCode}`,
    exitCode: completionData.exitCode,
    executionLog: completionData.executionLog || null,
    reportContent: completionData.reportContent || null,
    reportFormat: completionData.reportFormat || null,
    reportPath: completionData.reportPath || null,
    durationSeconds: completionData.durationSeconds || null,
    groundingScore: completionData.groundingScore || null,
    tokensUsed: completionData.tokensUsed || null,
    memoryMb: completionData.memoryMb || null,
    completedAt: new Date().toISOString(),
    lastUpdate: new Date().toISOString()
  });

  db.close();
  return result.changes > 0;
}

/**
 * Get job status
 */
export function getJobStatus(jobId) {
  const db = getDatabase();
  const stmt = db.prepare('SELECT * FROM research_jobs WHERE id = ?');
  const job = stmt.get(jobId);
  db.close();
  return job;
}

/**
 * Get all jobs with optional filtering
 */
export function getJobs(filters = {}) {
  const db = getDatabase();

  let query = 'SELECT * FROM research_jobs WHERE 1=1';
  const params = [];

  if (filters.status) {
    query += ' AND status = ?';
    params.push(filters.status);
  }

  if (filters.agent) {
    query += ' AND agent = ?';
    params.push(filters.agent);
  }

  query += ' ORDER BY created_at DESC';

  if (filters.limit) {
    query += ' LIMIT ?';
    params.push(filters.limit);
  }

  const stmt = db.prepare(query);
  const jobs = stmt.all(...params);

  db.close();
  return jobs;
}
