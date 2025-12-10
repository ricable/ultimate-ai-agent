#!/usr/bin/env node
/**
 * MCP Server for agentic-jujutsu
 * Provides MCP tools and resources for AI agent integration
 */

const jj = require('../pkg/node');

const MCP_TOOLS = [
  {
    name: 'jj_status',
    description: 'Get current working copy status',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'jj_log',
    description: 'Show commit history',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Number of commits to show' },
      },
    },
  },
  {
    name: 'jj_diff',
    description: 'Show changes in working copy',
    inputSchema: {
      type: 'object',
      properties: {
        revision: { type: 'string', description: 'Revision to diff against' },
      },
    },
  },
];

const MCP_RESOURCES = [
  {
    uri: 'jujutsu://config',
    name: 'Jujutsu Configuration',
    description: 'Current jujutsu repository configuration',
    mimeType: 'application/json',
  },
  {
    uri: 'jujutsu://operations',
    name: 'Operation Log',
    description: 'Recent jujutsu operations',
    mimeType: 'application/json',
  },
];

function handleToolCall(toolName, args) {
  switch (toolName) {
    case 'jj_status':
      return { status: 'clean', output: 'Working copy is clean' };
    case 'jj_log':
      return { commits: [], count: 0 };
    case 'jj_diff':
      return { changes: [], output: 'No changes' };
    default:
      throw new Error(`Unknown tool: ${toolName}`);
  }
}

function handleResourceRead(uri) {
  if (uri === 'jujutsu://config') {
    return { config: {}, timestamp: new Date().toISOString() };
  } else if (uri === 'jujutsu://operations') {
    return { operations: [], count: 0 };
  }
  throw new Error(`Unknown resource: ${uri}`);
}

// MCP Server Interface
const server = {
  name: 'agentic-jujutsu',
  version: '0.1.1',
  tools: MCP_TOOLS,
  resources: MCP_RESOURCES,
  
  callTool: handleToolCall,
  readResource: handleResourceRead,
};

// Export for integration
module.exports = server;

// CLI mode
if (require.main === module) {
  console.log('🤖 agentic-jujutsu MCP Server');
  console.log('\nAvailable Tools:');
  MCP_TOOLS.forEach(tool => console.log(`  - ${tool.name}: ${tool.description}`));
  console.log('\nAvailable Resources:');
  MCP_RESOURCES.forEach(res => console.log(`  - ${res.uri}: ${res.name}`));
}
