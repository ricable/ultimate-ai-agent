/**
 * AgentDB ReasoningBank Integration
 * Self-learning capabilities for research swarm with local SQLite storage
 */

import { getDatabase } from './db-utils.js';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';

/**
 * Store research pattern in ReasoningBank (local SQLite)
 */
export async function storeResearchPattern(jobId, pattern) {
  const db = getDatabase();

  try {
    const job = db.prepare('SELECT * FROM research_jobs WHERE id = ?').get(jobId);

    if (!job) {
      db.close();
      return false;
    }

    // Calculate reward based on research quality
    const reward = calculateReward(job);
    const patternId = uuidv4();

    // Store in reasoningbank_patterns table
    const stmt = db.prepare(`
      INSERT INTO reasoningbank_patterns (
        id, session_id, job_id, task, agent_type,
        input, output, reward, success, confidence,
        critique, latency_ms, tokens_used, memory_mb, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const config = job.metadata ? JSON.parse(job.metadata) : {};
    stmt.run(
      patternId,
      jobId,
      jobId,
      job.task,
      job.agent,
      JSON.stringify({ agent: job.agent, config }),
      job.report_content?.substring(0, 2000), // Summary
      reward,
      job.status === 'completed' ? 1 : 0,
      job.grounding_score || 0.5,
      generateCritique(job),
      (job.duration_seconds || 0) * 1000,
      job.tokens_used || 0,
      job.memory_mb || null,
      JSON.stringify({ pattern: pattern || {}, job_metrics: extractJobMetrics(job) })
    );

    // Create vector embedding for semantic search
    await createVectorEmbedding(db, patternId, 'pattern', job.task, job.report_content);

    // Create learning episode
    await createLearningEpisode(db, patternId, job, pattern);

    db.close();
    return true;

  } catch (error) {
    console.error('Failed to store reasoning pattern:', error.message);
    db.close();
    return false;
  }
}

/**
 * Calculate reward score based on research quality
 */
function calculateReward(job) {
  let reward = 0.5; // Base reward

  // Success bonus
  if (job.status === 'completed') reward += 0.3;
  
  // Quality metrics
  if (job.grounding_score) reward += (job.grounding_score / 100) * 0.2;
  
  // Efficiency bonus (completed faster than budget)
  if (job.duration_seconds) {
    const efficiency = Math.min(1, (120 * 60) / job.duration_seconds);
    reward += efficiency * 0.1;
  }

  // Has report content
  if (job.report_content && job.report_content.length > 1000) reward += 0.1;

  return Math.min(1, Math.max(0, reward));
}

/**
 * Generate critique for learning
 */
function generateCritique(job) {
  const critiques = [];

  if (job.status === 'failed') {
    critiques.push('Task failed - review error handling');
  }

  if (job.duration_seconds > 300) {
    critiques.push('Long execution time - optimize research depth');
  }

  if (!job.report_content || job.report_content.length < 500) {
    critiques.push('Short report - increase depth or broaden focus');
  }

  if (job.grounding_score && job.grounding_score < 80) {
    critiques.push('Low grounding score - strengthen citation requirements');
  }

  return critiques.length > 0 ? critiques.join('; ') : 'Good research execution';
}

/**
 * Extract job metrics for metadata
 */
function extractJobMetrics(job) {
  return {
    duration_seconds: job.duration_seconds,
    tokens_used: job.tokens_used,
    memory_mb: job.memory_mb,
    grounding_score: job.grounding_score,
    exit_code: job.exit_code,
    retry_count: job.retry_count
  };
}

/**
 * Create vector embedding for semantic search
 */
async function createVectorEmbedding(db, sourceId, sourceType, content_text, full_content) {
  const vectorId = uuidv4();
  const contentHash = crypto.createHash('sha256').update(content_text).digest('hex');

  const stmt = db.prepare(`
    INSERT INTO vector_embeddings (
      id, source_id, source_type, content_text, content_hash,
      content_length, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
  `);

  stmt.run(
    vectorId,
    sourceId,
    sourceType,
    content_text.substring(0, 5000),
    contentHash,
    content_text.length
  );

  return vectorId;
}

/**
 * Create learning episode from job
 */
async function createLearningEpisode(db, patternId, job, pattern) {
  const episodeId = uuidv4();

  // Get episode number for this pattern
  const prevEpisode = db.prepare(`
    SELECT MAX(episode_number) as max_episode
    FROM learning_episodes
    WHERE pattern_id = ?
  `).get(patternId);

  const episodeNumber = (prevEpisode?.max_episode || 0) + 1;

  // Calculate improvement rate
  const initialScore = 0.5; // Baseline score
  const finalScore = job.grounding_score || 0.5;
  const improvementRate = finalScore - initialScore;

  const stmt = db.prepare(`
    INSERT INTO learning_episodes (
      id, pattern_id, episode_number, task_category, difficulty_level,
      initial_score, final_score, improvement_rate,
      exploration_rate, exploitation_rate, learning_rate,
      verdict, judgment_score,
      started_at, completed_at, duration_seconds, metadata
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const taskCategory = extractTaskCategory(job.task);
  const difficultyLevel = pattern?.config?.depth || 5;
  const verdict = job.status === 'completed' ? 'success' : 'failure';

  // Calculate learning metrics
  const explorationRate = Math.min(1, (job.duration_seconds || 0) / 3600); // Time spent exploring
  const exploitationRate = finalScore; // How well we used existing knowledge
  const learningRate = improvementRate > 0 ? 0.1 : 0.05; // Adaptive learning rate

  stmt.run(
    episodeId,
    patternId,
    episodeNumber,
    taskCategory,
    difficultyLevel,
    initialScore,
    finalScore,
    improvementRate,
    explorationRate,
    exploitationRate,
    learningRate,
    verdict,
    finalScore, // judgment_score
    job.started_at,
    job.completed_at,
    job.duration_seconds,
    JSON.stringify({ pattern: pattern || {}, job_id: job.id })
  );

  return episodeId;
}

/**
 * Extract task category from task description
 */
function extractTaskCategory(task) {
  const keywords = {
    research: ['research', 'analyze', 'study', 'investigate', 'explore'],
    technical: ['implement', 'build', 'develop', 'code', 'program'],
    analysis: ['compare', 'evaluate', 'assess', 'review', 'examine'],
    documentation: ['document', 'write', 'explain', 'describe', 'summarize']
  };

  const taskLower = task.toLowerCase();
  for (const [category, words] of Object.entries(keywords)) {
    if (words.some(word => taskLower.includes(word))) {
      return category;
    }
  }

  return 'general';
}

/**
 * Search similar research patterns for learning (local SQLite)
 */
export async function searchSimilarPatterns(task, limit = 5) {
  const db = getDatabase();

  try {
    // Simple text-based similarity search
    const stmt = db.prepare(`
      SELECT
        p.*,
        LENGTH(p.task) as task_length,
        CASE
          WHEN p.task LIKE ? THEN 1.0
          ELSE 0.5
        END as similarity_score
      FROM reasoningbank_patterns p
      WHERE p.success = 1
      ORDER BY similarity_score DESC, p.reward DESC
      LIMIT ?
    `);

    const searchPattern = `%${task.substring(0, 50)}%`;
    const patterns = stmt.all(searchPattern, limit);

    db.close();
    return patterns;

  } catch (error) {
    console.error('Failed to search patterns:', error.message);
    db.close();
    return [];
  }
}

/**
 * Get learning statistics
 */
export function getLearningStats() {
  const db = getDatabase();

  try {
    const stats = db.prepare(`
      SELECT
        COUNT(*) as total_patterns,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_patterns,
        AVG(reward) as avg_reward,
        AVG(confidence) as avg_confidence,
        COUNT(DISTINCT task) as unique_tasks
      FROM reasoningbank_patterns
    `).get();

    db.close();
    return stats;

  } catch (error) {
    console.error('Failed to get learning stats:', error.message);
    db.close();
    return null;
  }
}
