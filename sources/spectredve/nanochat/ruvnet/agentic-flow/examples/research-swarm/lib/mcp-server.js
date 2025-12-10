#!/usr/bin/env node

/**
 * Research Swarm MCP Server
 * Model Context Protocol server for research swarm tools
 * 
 * Supports both stdio and HTTP/SSE streaming transports
 * Based on agentic-flow MCP architecture
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import express from 'express';
import { createJob, updateProgress, markComplete, getJobStatus, getJobs, initDatabase } from './db-utils.js';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment
dotenv.config({ path: path.join(__dirname, '../../.env') });

const VERSION = '1.0.0';
const SERVER_NAME = 'research-swarm';
const GITHUB_URL = 'https://github.com/ruvnet/agentic-flow';

// Initialize server
const server = new Server(
  {
    name: SERVER_NAME,
    version: VERSION
  },
  {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {}
    }
  }
);

/**
 * Research Swarm Tools
 */

// Initialize research swarm
server.setRequestHandler('tools/call', async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'research_swarm_init': {
        initDatabase();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              message: 'Research swarm initialized',
              version: VERSION
            })
          }]
        };
      }

      case 'research_swarm_create_job': {
        const { agent, task, config = {} } = args;
        
        if (!agent || !task) {
          throw new Error('Missing required arguments: agent, task');
        }

        const jobId = uuidv4();
        
        createJob({
          id: jobId,
          agent,
          task,
          status: 'pending',
          progress: 0,
          currentMessage: 'Job created',
          metadata: config
        });

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              jobId,
              agent,
              task,
              status: 'pending'
            })
          }]
        };
      }

      case 'research_swarm_start_job': {
        const { jobId, agent, task, config = {} } = args;
        
        if (!jobId || !agent || !task) {
          throw new Error('Missing required arguments: jobId, agent, task');
        }

        // Start research in background
        const runnerPath = path.join(__dirname, '../run-researcher-local.js');
        
        const env = {
          ...process.env,
          JOB_ID: jobId,
          RESEARCH_DEPTH: config.depth || '5',
          RESEARCH_TIME_BUDGET: config.timeBudget || '120',
          RESEARCH_FOCUS: config.focus || 'balanced',
          ANTI_HALLUCINATION_LEVEL: config.antiHallucination || 'high',
          CITATION_REQUIRED: config.citationRequired !== false ? 'true' : 'false',
          ED2551_MODE: config.ed2551Mode !== false ? 'true' : 'false'
        };

        const child = spawn('node', [runnerPath, agent, task], {
          env,
          detached: true,
          stdio: 'ignore'
        });

        child.unref();

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              jobId,
              status: 'started',
              pid: child.pid
            })
          }]
        };
      }

      case 'research_swarm_get_job': {
        const { jobId } = args;
        
        if (!jobId) {
          throw new Error('Missing required argument: jobId');
        }

        const job = getJobStatus(jobId);
        
        if (!job) {
          throw new Error(`Job not found: ${jobId}`);
        }

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              job: {
                id: job.id,
                agent: job.agent,
                task: job.task,
                status: job.status,
                progress: job.progress,
                currentMessage: job.current_message,
                durationSeconds: job.duration_seconds,
                reportPath: job.report_path,
                createdAt: job.created_at,
                completedAt: job.completed_at
              }
            })
          }]
        };
      }

      case 'research_swarm_list_jobs': {
        const { status, limit = 20 } = args;

        const jobs = getJobs({ status, limit });

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              count: jobs.length,
              jobs: jobs.map(j => ({
                id: j.id,
                agent: j.agent,
                task: j.task.substring(0, 100),
                status: j.status,
                progress: j.progress,
                createdAt: j.created_at
              }))
            })
          }]
        };
      }

      case 'research_swarm_update_progress': {
        const { jobId, progress, message } = args;
        
        if (!jobId || progress === undefined) {
          throw new Error('Missing required arguments: jobId, progress');
        }

        updateProgress(jobId, progress, message || 'Progress update');

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              jobId,
              progress,
              message
            })
          }]
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          success: false,
          error: error.message
        })
      }],
      isError: true
    };
  }
});

// List available tools
server.setRequestHandler('tools/list', async () => {
  return {
    tools: [
      {
        name: 'research_swarm_init',
        description: 'Initialize research swarm database and configuration',
        inputSchema: {
          type: 'object',
          properties: {},
          required: []
        }
      },
      {
        name: 'research_swarm_create_job',
        description: 'Create a new research job',
        inputSchema: {
          type: 'object',
          properties: {
            agent: { type: 'string', description: 'Agent name (e.g., researcher)' },
            task: { type: 'string', description: 'Research task description' },
            config: {
              type: 'object',
              description: 'Optional configuration',
              properties: {
                depth: { type: 'number', description: 'Research depth (1-10)' },
                timeBudget: { type: 'number', description: 'Time budget in minutes' },
                focus: { type: 'string', description: 'Focus mode (narrow|balanced|broad)' },
                antiHallucination: { type: 'string', description: 'Anti-hallucination level' },
                citationRequired: { type: 'boolean', description: 'Require citations' },
                ed2551Mode: { type: 'boolean', description: 'Enhanced research mode' }
              }
            }
          },
          required: ['agent', 'task']
        }
      },
      {
        name: 'research_swarm_start_job',
        description: 'Start a created research job',
        inputSchema: {
          type: 'object',
          properties: {
            jobId: { type: 'string', description: 'Job ID to start' },
            agent: { type: 'string', description: 'Agent name' },
            task: { type: 'string', description: 'Task description' },
            config: { type: 'object', description: 'Optional configuration' }
          },
          required: ['jobId', 'agent', 'task']
        }
      },
      {
        name: 'research_swarm_get_job',
        description: 'Get job status and details',
        inputSchema: {
          type: 'object',
          properties: {
            jobId: { type: 'string', description: 'Job ID to query' }
          },
          required: ['jobId']
        }
      },
      {
        name: 'research_swarm_list_jobs',
        description: 'List research jobs with optional filtering',
        inputSchema: {
          type: 'object',
          properties: {
            status: { type: 'string', description: 'Filter by status (pending|running|completed|failed)' },
            limit: { type: 'number', description: 'Maximum results to return' }
          },
          required: []
        }
      },
      {
        name: 'research_swarm_update_progress',
        description: 'Update job progress',
        inputSchema: {
          type: 'object',
          properties: {
            jobId: { type: 'string', description: 'Job ID' },
            progress: { type: 'number', description: 'Progress percentage (0-100)' },
            message: { type: 'string', description: 'Status message' }
          },
          required: ['jobId', 'progress']
        }
      }
    ]
  };
});

/**
 * Start server based on transport mode
 */
async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || 'stdio';

  if (mode === 'http' || mode === 'sse') {
    // HTTP/SSE server mode
    const port = parseInt(args[2]) || 3000;
    const app = express();

    app.use(express.json());

    // Health check
    app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        server: SERVER_NAME,
        version: VERSION,
        github: GITHUB_URL
      });
    });

    // SSE endpoint
    app.get('/sse', async (req, res) => {
      const transport = new SSEServerTransport('/message', res);
      await server.connect(transport);
    });

    // Message endpoint
    app.post('/message', async (req, res) => {
      // Handle MCP messages
      res.json({ success: true });
    });

    app.listen(port, () => {
      console.log(`🌐 Research Swarm MCP Server (HTTP/SSE) listening on port ${port}`);
      console.log(`   Health: http://localhost:${port}/health`);
      console.log(`   SSE:    http://localhost:${port}/sse`);
      console.log(`   GitHub: ${GITHUB_URL}`);
      console.log(`   Created by: rUv <https://ruv.io>`);
    });

  } else {
    // stdio mode (default)
    console.error('🚀 Research Swarm MCP Server (stdio) started');
    console.error(`   Version: ${VERSION}`);
    console.error(`   GitHub: ${GITHUB_URL}`);
    console.error(`   Created by: rUv <https://ruv.io>`);

    const transport = new StdioServerTransport();
    await server.connect(transport);
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
