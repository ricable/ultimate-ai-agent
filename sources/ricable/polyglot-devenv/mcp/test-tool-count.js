#!/usr/bin/env node

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Start the MCP server
const server = spawn('node', [join(__dirname, 'dist/index.js')], {
  stdio: ['pipe', 'pipe', 'pipe'],
});

let toolCount = 0;

// Send a list tools request
setTimeout(() => {
  const request = {
    jsonrpc: "2.0",
    id: 1,
    method: "tools/list"
  };
  
  server.stdin.write(JSON.stringify(request) + '\n');
}, 1000);

// Listen for response
server.stdout.on('data', (data) => {
  const lines = data.toString().split('\n').filter(line => line.trim());
  
  for (const line of lines) {
    try {
      const response = JSON.parse(line);
      if (response.id === 1 && response.result && response.result.tools) {
        toolCount = response.result.tools.length;
        console.log(`✅ MCP Server successfully loaded ${toolCount} tools`);
        
        // Log the tool categories
        const toolsByCategory = {};
        response.result.tools.forEach(tool => {
          const category = getToolCategory(tool.name);
          if (!toolsByCategory[category]) {
            toolsByCategory[category] = [];
          }
          toolsByCategory[category].push(tool.name);
        });
        
        console.log('\n📊 Tool Distribution by Category:');
        Object.entries(toolsByCategory).forEach(([category, tools]) => {
          console.log(`  ${category}: ${tools.length} tools`);
        });
        
        console.log(`\n🎯 Target: 74 tools | Actual: ${toolCount} tools`);
        console.log(toolCount >= 74 ? '✅ SUCCESS: All tools loaded!' : '❌ MISSING: Some tools not loaded');
        
        server.kill();
        process.exit(0);
      }
    } catch (e) {
      // Ignore JSON parse errors for non-JSON output
    }
  }
});

server.stderr.on('data', (data) => {
  console.error('Server error:', data.toString());
});

server.on('exit', (code) => {
  if (toolCount === 0) {
    console.log('❌ Failed to get tool count from server');
    process.exit(1);
  }
});

// Timeout after 10 seconds
setTimeout(() => {
  console.log('❌ Timeout waiting for server response');
  server.kill();
  process.exit(1);
}, 10000);

function getToolCategory(toolName) {
  if (toolName.startsWith('claude_flow_')) return '🚀 Claude-Flow Tools';
  if (toolName.startsWith('enhanced_hook_')) return '🚀 Enhanced AI Hooks';
  if (toolName.startsWith('docker_mcp_')) return '🚀 Docker MCP Tools';
  if (toolName.startsWith('host_') || toolName.startsWith('container_') || toolName.startsWith('security_boundary')) return '📦 Host/Container Tools';
  if (toolName.startsWith('nushell_')) return '🐚 Nushell Automation';
  if (toolName.startsWith('config_')) return '⚙️ Configuration Management';
  if (toolName.includes('analytics') || toolName.includes('monitoring') || toolName.includes('intelligence') || toolName.includes('anomaly') || toolName.includes('predictive') || toolName.includes('business_intelligence')) return '📈 Advanced Analytics';
  if (toolName.startsWith('agui_')) return '🤖 AG-UI Tools';
  if (toolName.startsWith('environment_')) return '🌍 Environment Tools';
  if (toolName.startsWith('devbox_')) return '📦 DevBox Tools';
  if (toolName.startsWith('devpod_')) return '🐳 DevPod Tools';
  if (toolName.startsWith('polyglot_')) return '🔄 Cross-Language Tools';
  if (toolName.startsWith('performance_')) return '⚡ Performance Tools';
  if (toolName.startsWith('security_')) return '🔒 Security Tools';
  if (toolName.startsWith('hook_')) return '🔗 Hook Tools';
  if (toolName.startsWith('prp_')) return '📋 PRP Tools';
  return '🔧 Other Tools';
}