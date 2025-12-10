/**
 * SkillLibrary - Lifelong Learning Skill Management
 *
 * Promotes high-reward trajectories into reusable skills.
 * Manages skill composition, relationships, and adaptive selection.
 *
 * Based on: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
 * https://arxiv.org/abs/2305.16291
 */

import { Database } from 'better-sqlite3';
import { EmbeddingService } from './EmbeddingService.js';

export interface Skill {
  id?: number;
  name: string;
  description?: string;
  signature: {
    inputs: Record<string, any>;
    outputs: Record<string, any>;
  };
  code?: string;
  successRate: number;
  uses: number;
  avgReward: number;
  avgLatencyMs: number;
  createdFromEpisode?: number;
  metadata?: Record<string, any>;
}

export interface SkillLink {
  parentSkillId: number;
  childSkillId: number;
  relationship: 'prerequisite' | 'alternative' | 'refinement' | 'composition';
  weight: number;
  metadata?: Record<string, any>;
}

export interface SkillQuery {
  task: string;
  k?: number;
  minSuccessRate?: number;
  preferRecent?: boolean;
}

export class SkillLibrary {
  private db: Database;
  private embedder: EmbeddingService;

  constructor(db: Database, embedder: EmbeddingService) {
    this.db = db;
    this.embedder = embedder;
  }

  /**
   * Create a new skill manually or from an episode
   */
  async createSkill(skill: Skill): Promise<number> {
    const stmt = this.db.prepare(`
      INSERT INTO skills (
        name, description, signature, code, success_rate, uses,
        avg_reward, avg_latency_ms, created_from_episode, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      skill.name,
      skill.description || null,
      JSON.stringify(skill.signature),
      skill.code || null,
      skill.successRate,
      skill.uses,
      skill.avgReward,
      skill.avgLatencyMs,
      skill.createdFromEpisode || null,
      skill.metadata ? JSON.stringify(skill.metadata) : null
    );

    const skillId = result.lastInsertRowid as number;

    // Generate and store embedding
    const text = this.buildSkillText(skill);
    const embedding = await this.embedder.embed(text);
    this.storeSkillEmbedding(skillId, embedding);

    return skillId;
  }

  /**
   * Update skill statistics after use
   */
  updateSkillStats(skillId: number, success: boolean, reward: number, latencyMs: number): void {
    const stmt = this.db.prepare(`
      UPDATE skills
      SET
        uses = uses + 1,
        success_rate = (success_rate * uses + ?) / (uses + 1),
        avg_reward = (avg_reward * uses + ?) / (uses + 1),
        avg_latency_ms = (avg_latency_ms * uses + ?) / (uses + 1)
      WHERE id = ?
    `);

    stmt.run(success ? 1 : 0, reward, latencyMs, skillId);
  }

  /**
   * Retrieve skills relevant to a task
   */
  async searchSkills(query: SkillQuery): Promise<Skill[]> {
    return this.retrieveSkills(query);
  }

  async retrieveSkills(query: SkillQuery): Promise<Skill[]> {
    const { task, k = 5, minSuccessRate = 0.5, preferRecent = true } = query;

    // Generate query embedding
    const queryEmbedding = await this.embedder.embed(task);

    // Build filters
    const filters = ['s.success_rate >= ?'];
    const params: any[] = [minSuccessRate];

    const stmt = this.db.prepare(`
      SELECT
        s.*,
        se.embedding
      FROM skills s
      JOIN skill_embeddings se ON s.id = se.skill_id
      WHERE ${filters.join(' AND ')}
      ORDER BY ${preferRecent ? 's.last_used_at DESC,' : ''} s.success_rate DESC
    `);

    const rows = stmt.all(...params) as any[];

    // Calculate similarities and rank
    const skills: (Skill & { similarity: number })[] = rows.map(row => {
      const embedding = this.deserializeEmbedding(row.embedding);
      const similarity = this.cosineSimilarity(queryEmbedding, embedding);

      return {
        id: row.id,
        name: row.name,
        description: row.description,
        signature: JSON.parse(row.signature),
        code: row.code,
        successRate: row.success_rate,
        uses: row.uses,
        avgReward: row.avg_reward,
        avgLatencyMs: row.avg_latency_ms,
        createdFromEpisode: row.created_from_episode,
        metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
        similarity
      };
    });

    // Compute composite scores
    skills.sort((a, b) => {
      const scoreA = this.computeSkillScore(a);
      const scoreB = this.computeSkillScore(b);
      return scoreB - scoreA;
    });

    return skills.slice(0, k);
  }

  /**
   * Link two skills with a relationship
   */
  linkSkills(link: SkillLink): void {
    const stmt = this.db.prepare(`
      INSERT INTO skill_links (parent_skill_id, child_skill_id, relationship, weight, metadata)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(parent_skill_id, child_skill_id, relationship)
      DO UPDATE SET weight = excluded.weight
    `);

    stmt.run(
      link.parentSkillId,
      link.childSkillId,
      link.relationship,
      link.weight,
      link.metadata ? JSON.stringify(link.metadata) : null
    );
  }

  /**
   * Get skill composition plan (prerequisites and alternatives)
   */
  getSkillPlan(skillId: number): {
    skill: Skill;
    prerequisites: Skill[];
    alternatives: Skill[];
    refinements: Skill[];
  } {
    // Get main skill
    const skill = this.getSkillById(skillId);

    // Get prerequisites
    const prereqStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'prerequisite'
      ORDER BY sl.weight DESC
    `);
    const prerequisites = prereqStmt.all(skillId).map(this.rowToSkill);

    // Get alternatives
    const altStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'alternative'
      ORDER BY sl.weight DESC, s.success_rate DESC
    `);
    const alternatives = altStmt.all(skillId).map(this.rowToSkill);

    // Get refinements
    const refStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'refinement'
      ORDER BY sl.weight DESC, s.created_at DESC
    `);
    const refinements = refStmt.all(skillId).map(this.rowToSkill);

    return { skill, prerequisites, alternatives, refinements };
  }

  /**
   * Consolidate high-reward episodes into skills
   * This is the core learning mechanism
   */
  consolidateEpisodesIntoSkills(config: {
    minAttempts?: number;
    minReward?: number;
    timeWindowDays?: number;
  }): number {
    const { minAttempts = 3, minReward = 0.7, timeWindowDays = 7 } = config;

    const stmt = this.db.prepare(`
      SELECT
        task,
        COUNT(*) as attempt_count,
        AVG(reward) as avg_reward,
        AVG(success) as success_rate,
        AVG(latency_ms) as avg_latency,
        MAX(id) as latest_episode_id,
        GROUP_CONCAT(id) as episode_ids
      FROM episodes
      WHERE ts > strftime('%s', 'now') - ?
        AND reward >= ?
      GROUP BY task
      HAVING attempt_count >= ?
    `);

    const candidates = stmt.all(timeWindowDays * 86400, minReward, minAttempts);
    let created = 0;

    for (const candidate of candidates as any[]) {
      // Check if skill already exists
      const existing = this.db.prepare('SELECT id FROM skills WHERE name = ?').get(candidate.task);

      if (!existing) {
        // Create new skill
        const skill: Skill = {
          name: candidate.task,
          description: `Auto-generated skill from successful episodes`,
          signature: {
            inputs: { task: 'string' },
            outputs: { result: 'any' }
          },
          successRate: candidate.success_rate,
          uses: candidate.attempt_count,
          avgReward: candidate.avg_reward,
          avgLatencyMs: candidate.avg_latency || 0,
          createdFromEpisode: candidate.latest_episode_id,
          metadata: {
            sourceEpisodes: candidate.episode_ids.split(',').map(Number),
            autoGenerated: true,
            consolidatedAt: Date.now()
          }
        };

        this.createSkill(skill).catch(err => {
          console.error('Error creating skill:', err);
        });
        created++;
      } else {
        // Update existing skill stats
        this.updateSkillStats(
          (existing as any).id,
          candidate.success_rate > 0.5,
          candidate.avg_reward,
          candidate.avg_latency || 0
        );
      }
    }

    return created;
  }

  /**
   * Prune underperforming skills
   */
  pruneSkills(config: {
    minUses?: number;
    minSuccessRate?: number;
    maxAgeDays?: number;
  }): number {
    const { minUses = 3, minSuccessRate = 0.4, maxAgeDays = 60 } = config;

    const stmt = this.db.prepare(`
      DELETE FROM skills
      WHERE uses < ?
        AND success_rate < ?
        AND created_at < strftime('%s', 'now') - ?
    `);

    const result = stmt.run(minUses, minSuccessRate, maxAgeDays * 86400);
    return result.changes;
  }

  // ========================================================================
  // Private Helper Methods
  // ========================================================================

  private getSkillById(id: number): Skill {
    const stmt = this.db.prepare('SELECT * FROM skills WHERE id = ?');
    const row = stmt.get(id);
    if (!row) throw new Error(`Skill ${id} not found`);
    return this.rowToSkill(row);
  }

  private rowToSkill(row: any): Skill {
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      signature: JSON.parse(row.signature),
      code: row.code,
      successRate: row.success_rate,
      uses: row.uses,
      avgReward: row.avg_reward,
      avgLatencyMs: row.avg_latency_ms,
      createdFromEpisode: row.created_from_episode,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined
    };
  }

  private buildSkillText(skill: Skill): string {
    const parts = [skill.name];
    if (skill.description) parts.push(skill.description);
    parts.push(JSON.stringify(skill.signature));
    return parts.join('\n');
  }

  private storeSkillEmbedding(skillId: number, embedding: Float32Array): void {
    const stmt = this.db.prepare(`
      INSERT INTO skill_embeddings (skill_id, embedding)
      VALUES (?, ?)
    `);
    stmt.run(skillId, Buffer.from(embedding.buffer));
  }

  private deserializeEmbedding(buffer: Buffer): Float32Array {
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private computeSkillScore(skill: Skill & { similarity: number }): number {
    // Composite score: similarity * 0.4 + success_rate * 0.3 + (uses/1000) * 0.1 + avg_reward * 0.2
    return (
      skill.similarity * 0.4 +
      skill.successRate * 0.3 +
      Math.min(skill.uses / 1000, 1.0) * 0.1 +
      skill.avgReward * 0.2
    );
  }
}
