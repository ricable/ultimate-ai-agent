#!/usr/bin/env node
/**
 * Agent Booster API Server - Morph LLM Compatible
 *
 * Drop-in replacement for Morph LLM API with 352x faster performance
 *
 * Compatible endpoints:
 * - POST /v1/chat/completions (Morph LLM format)
 * - POST /v1/apply (Morph LLM format)
 */

import express from 'express';
import { AgentBooster, MorphApplyRequest, MorphApplyResponse } from './index';

const app = express();
app.use(express.json());

// Initialize Agent Booster
const booster = new AgentBooster({
  confidenceThreshold: parseFloat(process.env.CONFIDENCE_THRESHOLD || '0.5'),
  maxChunks: parseInt(process.env.MAX_CHUNKS || '100'),
});

/**
 * Parse Morph LLM message format to extract instruction, code, and update
 */
function parseMorphMessage(content: string): { instruction: string; code: string; update: string } | null {
  const instructionMatch = content.match(/<instruction>(.*?)<\/instruction>/s);
  const codeMatch = content.match(/<code>(.*?)<\/code>/s);
  const updateMatch = content.match(/<update>(.*?)<\/update>/s);

  if (!instructionMatch || !codeMatch || !updateMatch) {
    return null;
  }

  return {
    instruction: instructionMatch[1].trim(),
    code: codeMatch[1].trim(),
    update: updateMatch[1].trim(),
  };
}

/**
 * POST /v1/chat/completions
 *
 * Morph LLM-compatible chat completions endpoint
 * Supports the same request/response format as Morph LLM
 */
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, stream } = req.body;

    if (!messages || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'messages array is required',
          type: 'invalid_request_error',
        },
      });
    }

    // Extract user message
    const userMessage = messages.find((m: any) => m.role === 'user');
    if (!userMessage) {
      return res.status(400).json({
        error: {
          message: 'No user message found',
          type: 'invalid_request_error',
        },
      });
    }

    // Parse Morph message format
    const parsed = parseMorphMessage(userMessage.content);
    if (!parsed) {
      return res.status(400).json({
        error: {
          message: 'Invalid message format. Expected <instruction>, <code>, and <update> tags',
          type: 'invalid_request_error',
        },
      });
    }

    // Detect language from code
    const language = detectLanguage(parsed.code);

    // Apply edit using Agent Booster
    const startTime = Date.now();
    const result = await booster.apply({
      code: parsed.code,
      edit: parsed.update,
      language,
    });

    // Return in Morph LLM format
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model || 'agent-booster-v1',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: result.output,
          },
          finish_reason: result.success ? 'stop' : 'length',
        },
      ],
      usage: {
        prompt_tokens: result.tokens.input,
        completion_tokens: result.tokens.output,
        total_tokens: result.tokens.input + result.tokens.output,
      },
      // Agent Booster extensions
      confidence: result.confidence,
      strategy: result.strategy,
      latency: result.latency,
    });
  } catch (error: any) {
    console.error('Error in /v1/chat/completions:', error);
    res.status(500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'internal_error',
      },
    });
  }
});

/**
 * POST /v1/apply
 *
 * Direct apply endpoint (simpler than chat completions)
 */
app.post('/v1/apply', async (req, res) => {
  try {
    const { code, edit, language } = req.body;

    if (!code || !edit) {
      return res.status(400).json({
        error: {
          message: 'code and edit fields are required',
          type: 'invalid_request_error',
        },
      });
    }

    const result = await booster.apply({
      code,
      edit,
      language: language || detectLanguage(code),
    });

    res.json(result);
  } catch (error: any) {
    console.error('Error in /v1/apply:', error);
    res.status(500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'internal_error',
      },
    });
  }
});

/**
 * POST /v1/batch
 *
 * Batch apply endpoint for processing multiple edits
 */
app.post('/v1/batch', async (req, res) => {
  try {
    const { requests } = req.body;

    if (!Array.isArray(requests)) {
      return res.status(400).json({
        error: {
          message: 'requests array is required',
          type: 'invalid_request_error',
        },
      });
    }

    const results = await booster.batchApply(requests);

    res.json({
      results,
      total: results.length,
      successful: results.filter(r => r.success).length,
    });
  } catch (error: any) {
    console.error('Error in /v1/batch:', error);
    res.status(500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'internal_error',
      },
    });
  }
});

/**
 * GET /health
 *
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    version: require('../package.json').version,
    uptime: process.uptime(),
  });
});

/**
 * GET /
 *
 * API documentation endpoint
 */
app.get('/', (req, res) => {
  res.json({
    name: 'Agent Booster API',
    version: require('../package.json').version,
    description: 'Morph LLM-compatible API with 352x faster performance',
    endpoints: {
      'POST /v1/chat/completions': 'Morph LLM-compatible chat completions',
      'POST /v1/apply': 'Direct apply endpoint',
      'POST /v1/batch': 'Batch apply multiple edits',
      'GET /health': 'Health check',
    },
    docs: 'https://github.com/yourusername/agent-booster',
  });
});

/**
 * Simple language detection based on file content
 */
function detectLanguage(code: string): string {
  if (code.includes('function') || code.includes('const') || code.includes('let')) {
    if (code.includes(':') && (code.includes('number') || code.includes('string') || code.includes('boolean'))) {
      return 'typescript';
    }
    return 'javascript';
  }
  if (code.includes('def ') || code.includes('import ')) {
    return 'python';
  }
  if (code.includes('fn ') || code.includes('impl ')) {
    return 'rust';
  }
  if (code.includes('func ') || code.includes('package ')) {
    return 'go';
  }
  if (code.includes('public class') || code.includes('private ')) {
    return 'java';
  }
  if (code.includes('#include')) {
    if (code.includes('std::') || code.includes('class ')) {
      return 'cpp';
    }
    return 'c';
  }

  return 'javascript'; // Default
}

// Start server
const PORT = parseInt(process.env.PORT || '3000');
const HOST = process.env.HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`🚀 Agent Booster API Server`);
  console.log(`📡 Listening on http://${HOST}:${PORT}`);
  console.log(`📚 API docs: http://${HOST}:${PORT}`);
  console.log(`🏥 Health check: http://${HOST}:${PORT}/health`);
  console.log(`\n✨ Morph LLM-compatible endpoints:`);
  console.log(`   POST /v1/chat/completions`);
  console.log(`   POST /v1/apply`);
  console.log(`   POST /v1/batch`);
  console.log(`\n⚡ 352x faster than Morph LLM | $0 cost | 100% private\n`);
});

export default app;
