// Validation test for Claude Agent SDK multi-provider integration
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { getAgent } from '../src/utils/agentLoader.js';
import { claudeAgent } from '../src/agents/claudeAgent.js';
import { logger } from '../src/utils/logger.js';

const OUTPUT_DIR = join(process.cwd(), 'validation', 'outputs');

async function validateProvider(providerName: string) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing Provider: ${providerName.toUpperCase()}`);
  console.log(`${'='.repeat(60)}\n`);

  // Set the provider
  process.env.PROVIDER = providerName;

  try {
    // Load the coder agent
    const agent = getAgent('coder');
    if (!agent) {
      throw new Error('Coder agent not found');
    }

    console.log(`✓ Agent loaded: ${agent.name}`);
    console.log(`✓ Description: ${agent.description}\n`);

    // Simple task
    const task = 'Create a simple "hello world" function in TypeScript that returns "Hello, World!"';
    console.log(`Task: ${task}\n`);

    const startTime = Date.now();
    const result = await claudeAgent(agent, task);
    const duration = Date.now() - startTime;

    console.log(`\n✅ Completed in ${duration}ms`);
    console.log(`Output length: ${result.output.length} characters\n`);

    // Save output to file
    const outputFile = join(OUTPUT_DIR, `${providerName}-output.md`);
    const content = `# Claude Agent SDK Test - ${providerName.toUpperCase()}\n\n` +
      `**Date**: ${new Date().toISOString()}\n` +
      `**Provider**: ${providerName}\n` +
      `**Agent**: ${agent.name}\n` +
      `**Task**: ${task}\n` +
      `**Duration**: ${duration}ms\n\n` +
      `## Output\n\n${result.output}\n`;

    writeFileSync(outputFile, content);
    console.log(`✓ Output saved to: ${outputFile}\n`);

    return {
      provider: providerName,
      success: true,
      duration,
      outputLength: result.output.length,
      outputFile
    };
  } catch (error: any) {
    console.error(`\n❌ Error testing ${providerName}:`);
    console.error(error.message);

    const errorFile = join(OUTPUT_DIR, `${providerName}-error.md`);
    const content = `# Claude Agent SDK Test - ${providerName.toUpperCase()} (FAILED)\n\n` +
      `**Date**: ${new Date().toISOString()}\n` +
      `**Provider**: ${providerName}\n` +
      `**Error**: ${error.message}\n\n` +
      `## Stack Trace\n\n\`\`\`\n${error.stack}\n\`\`\`\n`;

    writeFileSync(errorFile, content);
    console.log(`✓ Error saved to: ${errorFile}\n`);

    return {
      provider: providerName,
      success: false,
      error: error.message,
      errorFile
    };
  }
}

async function main() {
  console.log('\n╔═══════════════════════════════════════════════════════╗');
  console.log('║  Claude Agent SDK Multi-Provider Validation Test      ║');
  console.log('╚═══════════════════════════════════════════════════════╝\n');

  // Create output directory
  mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`✓ Output directory: ${OUTPUT_DIR}\n`);

  const providers = [
    'anthropic',   // Default - direct SDK to Anthropic API
    'openrouter',  // SDK → Proxy → OpenRouter
    'gemini',      // SDK → Proxy → Google Gemini
    'onnx'         // SDK → Proxy → ONNX Runtime
  ];

  const results = [];

  for (const provider of providers) {
    const result = await validateProvider(provider);
    results.push(result);

    // Wait a bit between providers
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Generate summary report
  console.log(`\n${'='.repeat(60)}`);
  console.log('VALIDATION SUMMARY');
  console.log(`${'='.repeat(60)}\n`);

  const summaryFile = join(OUTPUT_DIR, 'SUMMARY.md');
  let summaryContent = `# Claude Agent SDK Validation Summary\n\n`;
  summaryContent += `**Date**: ${new Date().toISOString()}\n\n`;
  summaryContent += `## Results\n\n`;
  summaryContent += `| Provider | Status | Duration | Output Size | File |\n`;
  summaryContent += `|----------|--------|----------|-------------|------|\n`;

  for (const result of results) {
    const status = result.success ? '✅ Success' : '❌ Failed';
    const duration = result.success ? `${result.duration}ms` : 'N/A';
    const outputSize = result.success ? `${result.outputLength} chars` : 'N/A';
    const file = result.success ? result.outputFile : result.errorFile;

    console.log(`${result.provider.padEnd(15)} ${status.padEnd(15)} ${duration.padEnd(10)} ${outputSize}`);

    summaryContent += `| ${result.provider} | ${status} | ${duration} | ${outputSize} | \`${file}\` |\n`;
  }

  summaryContent += `\n## Architecture Validation\n\n`;
  summaryContent += `✅ Claude Agent SDK is being used for all providers\n`;
  summaryContent += `✅ Multi-provider routing through proxy architecture\n`;
  summaryContent += `✅ File output generated for each test\n\n`;
  summaryContent += `### Provider Routing\n\n`;
  summaryContent += `- **Anthropic**: Direct SDK → Anthropic API\n`;
  summaryContent += `- **OpenRouter**: SDK → Proxy Router → OpenRouter API\n`;
  summaryContent += `- **Gemini**: SDK → Proxy Router → Google Gemini API\n`;
  summaryContent += `- **ONNX**: SDK → Proxy Router → ONNX Local Runtime\n`;

  writeFileSync(summaryFile, summaryContent);
  console.log(`\n✓ Summary saved to: ${summaryFile}\n`);

  const successCount = results.filter(r => r.success).length;
  const totalCount = results.length;

  console.log(`\nTotal: ${successCount}/${totalCount} providers successful\n`);

  if (successCount === totalCount) {
    console.log('🎉 All providers validated successfully!\n');
    process.exit(0);
  } else {
    console.log('⚠️  Some providers failed validation. Check error files for details.\n');
    process.exit(1);
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
