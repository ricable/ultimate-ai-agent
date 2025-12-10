#!/usr/bin/env node
/**
 * Test script to validate streaming output fix for Issue #40
 *
 * This script tests that:
 * 1. Tool calls are logged to stderr in real-time
 * 2. Tool completions are logged to stderr
 * 3. Assistant text content goes to stdout
 * 4. Timestamps are included for debugging
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🧪 Testing Streaming Output Fix (Issue #40)\n');
console.log('═'.repeat(60));

// Test configuration
const testTask = 'List all files in the current directory using the Bash tool, then count how many files were found';

// Track what we receive
let receivedToolCalls = 0;
let receivedToolCompletions = 0;
let receivedTimestamps = 0;
let stderrOutput = '';
let stdoutOutput = '';

// Spawn the agent with streaming enabled
console.log(`\n📝 Task: ${testTask}\n`);
console.log('⏳ Running agent with streaming enabled...\n');

const agentProcess = spawn('node', [
  join(__dirname, '../dist/index.js'),
  '--agent', 'coder',
  '--task', testTask,
  '--stream'
], {
  cwd: join(__dirname, '..'),
  env: {
    ...process.env,
    ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || 'test-key',
    ENABLE_STREAMING: 'true',
    // Disable MCP servers for faster testing
    ENABLE_CLAUDE_FLOW_MCP: 'false',
    ENABLE_FLOW_NEXUS_MCP: 'false',
    ENABLE_AGENTIC_PAYMENTS_MCP: 'false'
  }
});

// Monitor stderr for progress indicators
agentProcess.stderr.on('data', (data) => {
  const text = data.toString();
  stderrOutput += text;

  // Look for tool call indicators
  if (text.includes('🔍 Tool call')) {
    receivedToolCalls++;
    console.error(`✅ Received tool call notification (#${receivedToolCalls})`);
  }

  // Look for tool completion indicators
  if (text.includes('✅ Tool completed')) {
    receivedToolCompletions++;
    console.error(`✅ Received tool completion notification (#${receivedToolCompletions})`);
  }

  // Look for timestamps
  if (text.match(/\[\d{2}:\d{2}:\d{2}\]/)) {
    receivedTimestamps++;
  }

  // Echo progress to console
  process.stderr.write(text);
});

// Monitor stdout for assistant content
agentProcess.stdout.on('data', (data) => {
  const text = data.toString();
  stdoutOutput += text;
  process.stdout.write(text);
});

// Handle completion
agentProcess.on('close', (code) => {
  console.log('\n' + '═'.repeat(60));
  console.log('\n📊 Test Results:\n');

  let passCount = 0;
  let totalTests = 4;

  // Test 1: Tool calls were logged
  if (receivedToolCalls > 0) {
    console.log(`✅ PASS: Tool calls logged (${receivedToolCalls} calls)`);
    passCount++;
  } else {
    console.log('❌ FAIL: No tool calls logged to stderr');
  }

  // Test 2: Tool completions were logged
  if (receivedToolCompletions > 0) {
    console.log(`✅ PASS: Tool completions logged (${receivedToolCompletions} completions)`);
    passCount++;
  } else {
    console.log('❌ FAIL: No tool completions logged to stderr');
  }

  // Test 3: Timestamps were included
  if (receivedTimestamps > 0) {
    console.log(`✅ PASS: Timestamps included (${receivedTimestamps} timestamps)`);
    passCount++;
  } else {
    console.log('❌ FAIL: No timestamps found in stderr');
  }

  // Test 4: Assistant content went to stdout
  if (stdoutOutput.length > 100) {
    console.log(`✅ PASS: Assistant content sent to stdout (${stdoutOutput.length} chars)`);
    passCount++;
  } else {
    console.log('❌ FAIL: No substantial output on stdout');
  }

  console.log(`\n📈 Score: ${passCount}/${totalTests} tests passed`);

  // Additional diagnostics
  console.log('\n📋 Diagnostics:');
  console.log(`   Stderr length: ${stderrOutput.length} chars`);
  console.log(`   Stdout length: ${stdoutOutput.length} chars`);
  console.log(`   Tool calls detected: ${receivedToolCalls}`);
  console.log(`   Tool completions detected: ${receivedToolCompletions}`);
  console.log(`   Timestamps detected: ${receivedTimestamps}`);

  if (passCount === totalTests) {
    console.log('\n✅ All tests passed! Streaming output is working correctly.\n');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Streaming output may need further fixes.\n');
    process.exit(1);
  }
});

// Handle errors
agentProcess.on('error', (error) => {
  console.error('\n❌ Error spawning agent process:', error);
  process.exit(1);
});

// Timeout after 60 seconds
setTimeout(() => {
  console.error('\n⏰ Test timeout (60s) - killing agent process');
  agentProcess.kill();
  process.exit(1);
}, 60000);
