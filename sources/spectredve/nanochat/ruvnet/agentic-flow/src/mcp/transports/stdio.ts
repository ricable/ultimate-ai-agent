#!/usr/bin/env node
/**
 * Medical MCP Server - STDIO Transport
 * Provides command-line interface for medical analysis via standard input/output
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { MedicalAnalyzeTool } from '../tools/medical-analyze';
import { MedicalVerifyTool } from '../tools/medical-verify';
import { ProviderNotifyTool } from '../tools/provider-notify';
import { ConfidenceScoreTool } from '../tools/confidence-score';
import { CitationVerifyTool } from '../tools/citation-verify';
import { KnowledgeSearchTool } from '../tools/knowledge-search';

// Initialize server
const server = new Server(
  {
    name: 'medical-mcp-stdio',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Initialize tools
const medicalAnalyze = new MedicalAnalyzeTool();
const medicalVerify = new MedicalVerifyTool();
const providerNotify = new ProviderNotifyTool();
const confidenceScore = new ConfidenceScoreTool();
const citationVerify = new CitationVerifyTool();
const knowledgeSearch = new KnowledgeSearchTool();

// Tool definitions
const tools = [
  {
    name: 'medical_analyze',
    description: 'Analyze medical symptoms and conditions with anti-hallucination safeguards. Provides confidence scoring, citation verification, and emergency escalation.',
    inputSchema: {
      type: 'object',
      properties: {
        symptoms: {
          type: 'array',
          items: { type: 'string' },
          description: 'List of symptoms to analyze',
        },
        patientHistory: {
          type: 'string',
          description: 'Relevant patient medical history',
        },
        vitalSigns: {
          type: 'object',
          description: 'Vital signs (temperature, heartRate, systolicBP, oxygenSaturation, etc.)',
        },
        includeRecommendations: {
          type: 'boolean',
          default: true,
          description: 'Include treatment recommendations',
        },
      },
      required: ['symptoms'],
    },
  },
  {
    name: 'medical_verify',
    description: 'Verify medical analysis quality and accuracy with comprehensive checks for confidence, citations, consistency, completeness, and safety.',
    inputSchema: {
      type: 'object',
      properties: {
        analysisId: {
          type: 'string',
          description: 'Analysis ID to verify',
        },
        analysis: {
          type: 'object',
          description: 'Medical analysis object to verify',
        },
        strictMode: {
          type: 'boolean',
          default: true,
          description: 'Enable strict validation mode',
        },
      },
      required: ['analysisId', 'analysis'],
    },
  },
  {
    name: 'provider_notify',
    description: 'Notify healthcare providers about medical analyses requiring review. Supports multiple channels (email, SMS, pager, app) based on urgency.',
    inputSchema: {
      type: 'object',
      properties: {
        analysisId: {
          type: 'string',
          description: 'Analysis ID to notify about',
        },
        analysis: {
          type: 'object',
          description: 'Medical analysis object',
        },
        urgency: {
          type: 'string',
          enum: ['low', 'medium', 'high', 'critical'],
          description: 'Notification urgency level',
        },
        recipient: {
          type: 'string',
          description: 'Provider email or phone number',
        },
        channel: {
          type: 'string',
          enum: ['email', 'sms', 'pager', 'app'],
          description: 'Notification channel',
        },
        message: {
          type: 'string',
          description: 'Custom notification message',
        },
      },
      required: ['analysisId'],
    },
  },
  {
    name: 'confidence_score',
    description: 'Calculate comprehensive confidence scores for medical analyses with component breakdown (diagnosis, treatment, prognosis) and quality metrics.',
    inputSchema: {
      type: 'object',
      properties: {
        analysis: {
          type: 'object',
          description: 'Medical analysis object to score',
        },
        detailedBreakdown: {
          type: 'boolean',
          default: false,
          description: 'Include detailed component breakdown',
        },
      },
      required: ['analysis'],
    },
  },
  {
    name: 'citation_verify',
    description: 'Verify medical citations against trusted sources (PubMed, Cochrane, NICE, etc.). Validates citation quality, relevance, and source reliability.',
    inputSchema: {
      type: 'object',
      properties: {
        citations: {
          type: 'array',
          items: { type: 'object' },
          description: 'Array of citation objects to verify',
        },
        strictMode: {
          type: 'boolean',
          default: true,
          description: 'Enable strict validation mode',
        },
      },
      required: ['citations'],
    },
  },
  {
    name: 'knowledge_search',
    description: 'Search medical knowledge bases (PubMed, Cochrane, UpToDate, NICE) for relevant clinical information, guidelines, and research.',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query for medical knowledge',
        },
        filters: {
          type: 'object',
          properties: {
            sourceTypes: {
              type: 'array',
              items: { type: 'string' },
              description: 'Filter by source types',
            },
            minRelevance: {
              type: 'number',
              description: 'Minimum relevance score (0-1)',
            },
            dateRange: {
              type: 'object',
              properties: {
                start: { type: 'string', description: 'Start date (ISO format)' },
                end: { type: 'string', description: 'End date (ISO format)' },
              },
            },
            specialties: {
              type: 'array',
              items: { type: 'string' },
              description: 'Filter by medical specialties',
            },
          },
        },
        maxResults: {
          type: 'number',
          default: 10,
          description: 'Maximum number of results',
        },
      },
      required: ['query'],
    },
  },
];

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let result;

    switch (name) {
      case 'medical_analyze':
        result = await medicalAnalyze.execute(args as any);
        break;

      case 'medical_verify':
        result = await medicalVerify.execute(args as any);
        break;

      case 'provider_notify':
        result = await providerNotify.execute(args as any);
        break;

      case 'confidence_score':
        result = await confidenceScore.execute(args as any);
        break;

      case 'citation_verify':
        result = await citationVerify.execute(args as any);
        break;

      case 'knowledge_search':
        // Convert string dates to Date objects
        if (args?.filters?.dateRange) {
          const dr = (args.filters as any).dateRange;
          (args.filters as any).dateRange = {
            start: dr.start ? new Date(dr.start) : undefined,
            end: dr.end ? new Date(dr.end) : undefined,
          };
        }
        result = await knowledgeSearch.execute(args as any);
        break;

      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    return result;
  } catch (error: any) {
    return {
      content: [
        {
          type: 'text',
          text: `❌ Error executing ${name}: ${error.message}\n${error.stack || ''}`,
        },
      ],
      isError: true,
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('🚀 Medical MCP Server v1.0.0 running on stdio');
  console.error('📦 6 medical tools available with anti-hallucination features');
  console.error('🏥 Tools: medical_analyze, medical_verify, provider_notify, confidence_score, citation_verify, knowledge_search');
  console.error('✨ Features: Real-time confidence monitoring, citation validation, emergency escalation');

  // Keep process alive
  const keepAlive = setInterval(() => {}, 1000 * 60 * 60);

  // Handle graceful shutdown
  const shutdown = () => {
    clearInterval(keepAlive);
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  return new Promise(() => {});
}

main().catch(console.error);
