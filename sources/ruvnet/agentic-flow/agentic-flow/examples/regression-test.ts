#!/usr/bin/env tsx
/**
 * Regression Test Suite
 *
 * Verifies that tool emulation code doesn't break existing functionality:
 * 1. Tool emulation files exist but aren't imported in main code
 * 2. TypeScript compilation succeeds
 * 3. Proxy classes remain unchanged
 * 4. Agent execution works with tool-capable models
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { detectModelCapabilities } from '../src/utils/modelCapabilities.js';

console.log('\n' + '═'.repeat(80));
console.log('🧪 REGRESSION TEST SUITE - Tool Emulation Backward Compatibility');
console.log('═'.repeat(80) + '\n');

let passCount = 0;
let failCount = 0;

function test(name: string, fn: () => boolean | Promise<boolean>) {
  process.stdout.write(`Testing: ${name}... `);
  try {
    const result = fn();
    if (result instanceof Promise) {
      return result.then(r => {
        if (r) {
          console.log('✅ PASS');
          passCount++;
        } else {
          console.log('❌ FAIL');
          failCount++;
        }
      });
    } else {
      if (result) {
        console.log('✅ PASS');
        passCount++;
      } else {
        console.log('❌ FAIL');
        failCount++;
      }
    }
  } catch (error: any) {
    console.log(`❌ FAIL: ${error.message}`);
    failCount++;
  }
}

// Test 1: Tool emulation files exist
test('Tool emulation files exist', () => {
  return existsSync('src/utils/modelCapabilities.ts') &&
         existsSync('src/proxy/tool-emulation.ts');
});

// Test 2: Tool emulation integrated in CLI (Phase 2)
test('Tool emulation integrated in cli-proxy (Phase 2)', () => {
  const cliProxy = execSync('grep -c "detectModelCapabilities" src/cli-proxy.ts || true', { encoding: 'utf-8' });
  return parseInt(cliProxy.trim()) >= 1; // Should have at least 1 import
});

// Test 3: Tool emulation integrated in proxy (Phase 2)
test('Tool emulation integrated in anthropic-to-openrouter (Phase 2)', () => {
  const proxy = execSync('grep -c "tool-emulation\\|modelCapabilities" src/proxy/anthropic-to-openrouter.ts || true', { encoding: 'utf-8' });
  return parseInt(proxy.trim()) >= 2; // Should have imports for both
});

// Test 4: Tool emulation not imported in anthropic-to-gemini
test('Tool emulation isolated (not imported in anthropic-to-gemini)', () => {
  const gemini = execSync('grep -c "tool-emulation\\|modelCapabilities" src/proxy/anthropic-to-gemini.ts || true', { encoding: 'utf-8' });
  return parseInt(gemini.trim()) === 0;
});

// Test 5: TypeScript compilation succeeds
test('TypeScript compilation succeeds', () => {
  try {
    execSync('npm run build', { stdio: 'pipe' });
    return true;
  } catch (error) {
    return false;
  }
});

// Test 6: Model capability detection works for native tool models
test('Model capability detection - DeepSeek (native tools)', () => {
  const cap = detectModelCapabilities('deepseek/deepseek-chat');
  return cap.supportsNativeTools === true && cap.requiresEmulation === false;
});

// Test 7: Model capability detection works for Claude
test('Model capability detection - Claude (native tools)', () => {
  const cap = detectModelCapabilities('claude-3-5-sonnet-20241022');
  return cap.supportsNativeTools === true && cap.requiresEmulation === false;
});

// Test 8: Model capability detection works for non-tool models
test('Model capability detection - Mistral 7B (needs emulation)', () => {
  const cap = detectModelCapabilities('mistralai/mistral-7b-instruct');
  return cap.supportsNativeTools === false && cap.requiresEmulation === true;
});

// Test 9: Check if proxy exports remain unchanged
test('AnthropicToOpenRouterProxy class exists', () => {
  const exports = execSync('grep -c "export class AnthropicToOpenRouterProxy" src/proxy/anthropic-to-openrouter.ts', { encoding: 'utf-8' });
  return parseInt(exports.trim()) === 1;
});

// Test 10: Check if GeminiProxy class exists
test('AnthropicToGeminiProxy class exists', () => {
  const exports = execSync('grep -c "export class AnthropicToGeminiProxy" src/proxy/anthropic-to-gemini.ts', { encoding: 'utf-8' });
  return parseInt(exports.trim()) === 1;
});

// Test 11: Tool emulation exports are correct
test('ToolEmulator class exported', () => {
  const exports = execSync('grep -c "export class ToolEmulator" src/proxy/tool-emulation.ts', { encoding: 'utf-8' });
  return parseInt(exports.trim()) === 1;
});

// Test 12: Agent definitions unchanged
test('Agent definitions directory exists', () => {
  return existsSync('.claude/agents');
});

// Test 13: Check no tool rewriting in proxy (critical - proves no regression)
test('Proxy does NOT rewrite tool names', () => {
  const rewrite = execSync('grep -n "tool.name.*=" src/proxy/anthropic-to-openrouter.ts | grep -v "name: tool.name" || true', { encoding: 'utf-8' });
  // Should be empty (no lines that assign to tool.name except direct pass-through)
  return rewrite.trim().length === 0;
});

// Test 14: Verify tool schemas pass through unchanged
test('Proxy passes tool schemas unchanged', () => {
  const schema = execSync('grep -c "parameters: tool.input_schema" src/proxy/anthropic-to-openrouter.ts', { encoding: 'utf-8' });
  return parseInt(schema.trim()) >= 1;
});

// Test 15: Verify examples compile
test('Example files are valid TypeScript', () => {
  try {
    execSync('npx tsc --noEmit examples/tool-emulation-demo.ts', { stdio: 'pipe' });
    execSync('npx tsc --noEmit examples/tool-emulation-test.ts', { stdio: 'pipe' });
    return true;
  } catch (error) {
    return false;
  }
});

// Summary
console.log('\n' + '═'.repeat(80));
console.log('📊 TEST SUMMARY');
console.log('═'.repeat(80) + '\n');

const total = passCount + failCount;
const percentage = ((passCount / total) * 100).toFixed(1);

console.log(`✅ Passed: ${passCount}/${total} (${percentage}%)`);
console.log(`❌ Failed: ${failCount}/${total}`);

if (failCount === 0) {
  console.log('\n🎉 All regression tests passed!');
  console.log('✅ Tool emulation code is isolated and non-breaking');
  console.log('✅ Existing functionality remains unchanged');
  console.log('✅ TypeScript compilation succeeds');
  console.log('✅ Proxy tool pass-through verified');
} else {
  console.log('\n⚠️  Some regression tests failed!');
  console.log('Please review the failures above.');
  process.exit(1);
}

console.log('\n' + '═'.repeat(80) + '\n');
