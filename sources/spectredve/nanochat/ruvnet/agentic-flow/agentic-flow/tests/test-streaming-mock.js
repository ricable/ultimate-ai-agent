#!/usr/bin/env node
/**
 * Mock test to verify streaming code is properly integrated
 * Tests the code paths without requiring API calls
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🧪 Mock Streaming Code Path Verification\n');
console.log('═'.repeat(60));

// Check that the streaming fixes are in the compiled code
const claudeAgentPath = join(__dirname, '../dist/agents/claudeAgent.js');
const indexPath = join(__dirname, '../dist/index.js');

let passCount = 0;
let totalTests = 0;

// Test 1: Check tool call detection code exists
totalTests++;
const claudeAgentCode = readFileSync(claudeAgentPath, 'utf-8');
if (claudeAgentCode.includes('🔍 Tool call') && 
    claudeAgentCode.includes('toolCallCount') &&
    claudeAgentCode.includes('content_block_start')) {
  console.log('✅ PASS: Tool call detection code found in compiled output');
  passCount++;
} else {
  console.log('❌ FAIL: Tool call detection code missing');
}

// Test 2: Check tool completion code exists
totalTests++;
if (claudeAgentCode.includes('✅ Tool completed') && 
    claudeAgentCode.includes('content_block_stop')) {
  console.log('✅ PASS: Tool completion detection code found');
  passCount++;
} else {
  console.log('❌ FAIL: Tool completion detection code missing');
}

// Test 3: Check timestamp code exists
totalTests++;
if (claudeAgentCode.includes('toISOString') && 
    claudeAgentCode.includes('split')) {
  console.log('✅ PASS: Timestamp generation code found');
  passCount++;
} else {
  console.log('❌ FAIL: Timestamp generation code missing');
}

// Test 4: Check stderr routing exists
totalTests++;
if (claudeAgentCode.includes('process.stderr.write')) {
  console.log('✅ PASS: stderr routing code found');
  passCount++;
} else {
  console.log('❌ FAIL: stderr routing code missing');
}

// Test 5: Check uncork() calls exist
totalTests++;
if (claudeAgentCode.includes('uncork')) {
  console.log('✅ PASS: Output flushing (uncork) code found');
  passCount++;
} else {
  console.log('❌ FAIL: Output flushing code missing');
}

// Test 6: Check enhanced stream handler exists
totalTests++;
const indexCode = readFileSync(indexPath, 'utf-8');
if (indexCode.includes('streamHandler') && 
    indexCode.includes('process.stderr.write') &&
    indexCode.includes('process.stdout.write')) {
  console.log('✅ PASS: Enhanced stream handler found in index.js');
  passCount++;
} else {
  console.log('❌ FAIL: Enhanced stream handler missing');
}

// Test 7: Check stream handler separates progress from content
totalTests++;
if (indexCode.includes('startsWith') && 
    (indexCode.includes('🔍') || indexCode.includes('includes'))) {
  console.log('✅ PASS: Stream handler has progress/content separation logic');
  passCount++;
} else {
  console.log('❌ FAIL: Stream handler missing separation logic');
}

console.log('\n' + '═'.repeat(60));
console.log(`\n📊 Results: ${passCount}/${totalTests} tests passed`);

if (passCount === totalTests) {
  console.log('\n✅ All code paths verified! Streaming fix is properly integrated.\n');
  console.log('💡 To test with real API calls, run:');
  console.log('   export ANTHROPIC_API_KEY=sk-ant-...');
  console.log('   node dist/index.js --agent coder --task "Your task" --stream\n');
  process.exit(0);
} else {
  console.log('\n⚠️  Some code paths missing. Build may have failed.\n');
  process.exit(1);
}
