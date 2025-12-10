/**
 * ReasoningBank-Enabled Agent
 *
 * Executes tasks with full ReasoningBank capabilities:
 * - RETRIEVE: Fetches relevant memories
 * - JUDGE: Evaluates success/failure
 * - DISTILL: Extracts learnings
 * - CONSOLIDATE: Deduplicates and prunes
 */

import Anthropic from '@anthropic-ai/sdk';
import {
  retrieveMemories,
  formatMemoriesForPrompt,
  judgeTrajectory,
  distillMemories,
  shouldConsolidate,
  consolidate,
  db
} from '../../agentic-flow/src/reasoningbank/index.js';
import { ulid } from 'ulid';
import type { Agent, Task, TaskResult } from '../lib/types.js';

export class ReasoningBankAgent implements Agent {
  readonly type = 'reasoningbank';
  private client: Anthropic;
  private model: string;
  private agentId: string;

  constructor(apiKey?: string, model: string = 'claude-sonnet-4-5-20250929') {
    this.client = new Anthropic({
      apiKey: apiKey || process.env.ANTHROPIC_API_KEY
    });
    this.model = model;
    this.agentId = 'benchmark-agent';
  }

  async executeTask(task: Task): Promise<TaskResult> {
    const startTime = Date.now();
    const taskId = ulid();

    try {
      // 1. RETRIEVE: Get relevant memories
      const memories = await retrieveMemories(task.description, {
        domain: task.domain,
        agent: this.agentId
      });

      const memoriesUsed = memories.length;

      // 2. EXECUTE: Run task with memory context
      const systemPrompt = memories.length > 0
        ? formatMemoriesForPrompt(memories)
        : '';

      const messages: Anthropic.Messages.MessageParam[] = [
        {
          role: 'user',
          content: `${task.description}\n\nInput:\n${task.input}`
        }
      ];

      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: 4096,
        system: systemPrompt || undefined,
        messages
      });

      const output = response.content[0].type === 'text' ? response.content[0].text : '';
      const tokens = response.usage.input_tokens + response.usage.output_tokens;

      // Build trajectory for judging
      const trajectory = {
        steps: [
          { action: 'retrieve', memoriesFound: memories.length },
          { action: 'execute', output, tokens }
        ],
        metadata: {
          task: task.description,
          input: task.input,
          output
        }
      };

      // 3. JUDGE: Evaluate success/failure
      const verdict = await judgeTrajectory(trajectory, task.description);

      const success = task.successCriteria({
        taskId: task.id,
        agentType: 'reasoningbank',
        success: false,
        output,
        tokens,
        latency: 0
      }) && verdict.label === 'Success';

      // 4. DISTILL: Extract learnings from this execution
      const newMemories = await distillMemories(trajectory, verdict, task.description, {
        taskId,
        agentId: this.agentId,
        domain: task.domain
      });

      const memoriesCreated = newMemories.length;

      // 5. CONSOLIDATE: Run if threshold reached
      if (shouldConsolidate()) {
        await consolidate();
      }

      const latency = Date.now() - startTime;

      return {
        taskId: task.id,
        agentType: 'reasoningbank',
        success,
        output,
        tokens,
        latency,
        memoriesUsed,
        memoriesCreated,
        confidence: verdict.confidence,
        trajectory
      };
    } catch (error) {
      const latency = Date.now() - startTime;
      console.error(`[ReasoningBankAgent] Error executing task ${task.id}:`, error);

      return {
        taskId: task.id,
        agentType: 'reasoningbank',
        success: false,
        output: '',
        tokens: 0,
        latency,
        memoriesUsed: 0,
        memoriesCreated: 0,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  async reset(): Promise<void> {
    // Clear all memories for this agent
    const dbConn = db.getDb();
    dbConn.prepare(`
      DELETE FROM patterns
      WHERE type = 'reasoning_memory'
      AND json_extract(pattern_data, '$.source.agent_id') = ?
    `).run(this.agentId);

    console.log(`[ReasoningBankAgent] Reset - cleared all memories`);
  }

  async getMemoryStats(): Promise<{ total: number; avgConfidence: number }> {
    const dbConn = db.getDb();

    const result = dbConn.prepare(`
      SELECT
        COUNT(*) as total,
        AVG(confidence) as avgConfidence
      FROM patterns
      WHERE type = 'reasoning_memory'
      AND json_extract(pattern_data, '$.source.agent_id') = ?
    `).get(this.agentId) as { total: number; avgConfidence: number };

    return {
      total: result.total || 0,
      avgConfidence: result.avgConfidence || 0
    };
  }
}
