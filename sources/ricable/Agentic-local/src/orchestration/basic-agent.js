/**
 * Basic Agent Example
 * Demonstrates simple agentic-flow setup with local GaiaNet node
 */

import 'dotenv/config';
import { AgenticFlow } from 'agentic-flow';
import { DockerSandbox } from '../sandbox/docker-sandbox.js';

// Initialize sandbox for code execution
const sandbox = new DockerSandbox({
  memoryLimit: process.env.SANDBOX_MEMORY_LIMIT || '2g',
  cpuLimit: process.env.SANDBOX_CPU_LIMIT || '2',
  timeout: parseInt(process.env.SANDBOX_TIMEOUT) || 300000,
  network: process.env.SANDBOX_NETWORK || 'none'
});

// Configure agent with local GaiaNet node
const agent = new AgenticFlow({
  provider: process.env.LLM_PROVIDER || 'local',
  baseURL: process.env.GAIANET_ENDPOINT || 'http://localhost:8080/v1',
  model: process.env.GAIANET_MODEL || 'Qwen2.5-Coder-32B-Instruct',
  temperature: parseFloat(process.env.TEMPERATURE) || 0.2,
  maxTokens: 4096,

  // Enable Agent Booster for 352x faster code operations
  enableBooster: process.env.ENABLE_AGENT_BOOSTER === 'true',

  // Callback for code execution
  onCodeGenerated: async (code, language) => {
    console.log(`\n🔧 Executing ${language} code in sandbox...`);

    const result = await sandbox.execute(code, language);

    if (result.success) {
      console.log('✅ Execution successful');
      console.log('Output:', result.stdout);
      return { success: true, output: result.stdout };
    } else {
      console.log('❌ Execution failed');
      console.log('Error:', result.stderr);
      return { success: false, error: result.stderr };
    }
  }
});

/**
 * Example 1: Simple code generation
 */
async function simpleCodeGeneration() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 1: Simple Code Generation');
  console.log('='.repeat(60));

  const prompt = `
    Create a JavaScript function that:
    1. Takes an array of numbers as input
    2. Returns an object with: sum, average, min, max
    3. Include example usage
  `;

  const result = await agent.run(prompt);
  console.log('\n📝 Generated Code:\n', result.code);
  console.log('\n🎯 Test Results:\n', result.testOutput);
}

/**
 * Example 2: Iterative development with feedback loop
 */
async function iterativeDevelopment() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 2: Iterative Development');
  console.log('='.repeat(60));

  const prompt = `
    Create a REST API endpoint calculator in Express.js with:
    - POST /calculate endpoint
    - Accepts { a, b, operation } in JSON
    - Supports: add, subtract, multiply, divide
    - Returns { result } or { error }
    - Include proper error handling
  `;

  let iteration = 0;
  let success = false;

  while (!success && iteration < 3) {
    iteration++;
    console.log(`\n🔄 Iteration ${iteration}`);

    const result = await agent.run(prompt);

    if (result.success) {
      console.log('✅ Code validated successfully!');
      success = true;
    } else {
      console.log('❌ Validation failed, agent will retry...');
      // Agent automatically learns from error via ReasoningBank
    }
  }
}

/**
 * Example 3: Multi-step task with reasoning
 */
async function multiStepTask() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 3: Multi-Step Task');
  console.log('='.repeat(60));

  const task = `
    Create a data analysis pipeline:

    Step 1: Generate sample CSV data (sales records with: date, product, quantity, price)
    Step 2: Write a function to parse the CSV
    Step 3: Calculate total revenue per product
    Step 4: Find the best-selling product
    Step 5: Generate a summary report

    Execute each step and verify the output.
  `;

  const result = await agent.run(task, {
    enableReasoning: true, // Use ReasoningBank for complex reasoning
    maxIterations: parseInt(process.env.MAX_ITERATIONS) || 10
  });

  console.log('\n📊 Analysis Results:');
  console.log(result.output);
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Sovereign Agentic Stack - Basic Agent Examples');
  console.log('Using local GaiaNet node:', process.env.GAIANET_ENDPOINT);
  console.log('Model:', process.env.GAIANET_MODEL);

  try {
    // Run examples
    await simpleCodeGeneration();
    await iterativeDevelopment();
    await multiStepTask();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All examples completed successfully!');
    console.log('='.repeat(60));

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { agent, simpleCodeGeneration, iterativeDevelopment, multiStepTask };
