/**
 * Agent Booster - Usage Examples
 * Demonstrates various API usage patterns
 */

const agentBooster = require('./index');

// Example 1: Basic optimization
async function example1() {
  console.log('=== Example 1: Basic Optimization ===\n');

  const prompt = `
    Please help me create a comprehensive function that will process user input data.
    The function should handle various types of input including strings, numbers, and objects.
    It needs to validate the input, perform necessary transformations, and return the processed result.
    Make sure to include proper error handling and edge case management.
  `;

  try {
    const optimized = await agentBooster.apply(prompt);
    console.log('Original:', prompt);
    console.log('\nOptimized:', optimized);
    console.log('\nRuntime:', agentBooster.getRuntime());
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Example 2: With optimization options
async function example2() {
  console.log('\n=== Example 2: Custom Options ===\n');

  const prompt = 'Write a detailed function to parse JSON data with error handling';

  try {
    const optimized = await agentBooster.apply(prompt, {
      strategy: 'aggressive',
      maxTokens: 50,
      preserve: ['JSON', 'error'],
      targetModel: 'claude-3'
    });

    console.log('Optimized:', optimized);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Example 3: Prompt analysis
async function example3() {
  console.log('\n=== Example 3: Prompt Analysis ===\n');

  const prompt = `
    I need you to create a very detailed and comprehensive solution for managing
    user authentication in our application. This should include login functionality,
    password reset features, session management, and security best practices.
    Please make sure to cover all edge cases and provide thorough documentation.
  `;

  try {
    const analysis = await agentBooster.analyze(prompt);
    console.log('Analysis Results:');
    console.log('  Original Tokens:', analysis.originalTokens);
    console.log('  Optimized Tokens:', analysis.optimizedTokens);
    console.log('  Savings:', analysis.savings, `(${analysis.savingsPercent.toFixed(1)}%)`);
    console.log('  Complexity:', analysis.complexity);
    console.log('  Patterns:', analysis.patterns);
    console.log('  Suggestions:', analysis.suggestions);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Example 4: Batch processing
async function example4() {
  console.log('\n=== Example 4: Batch Processing ===\n');

  const prompts = [
    'Create a user registration form',
    'Implement password validation logic',
    'Build a login authentication system',
    'Design a password reset workflow'
  ];

  try {
    const optimized = await agentBooster.batchApply(prompts, {
      strategy: 'balanced'
    });

    console.log('Batch Results:');
    optimized.forEach((opt, idx) => {
      console.log(`\n[${idx + 1}] Original: ${prompts[idx]}`);
      console.log(`    Optimized: ${opt}`);
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Example 5: Version information
async function example5() {
  console.log('\n=== Example 5: Version Info ===\n');

  try {
    const info = agentBooster.getVersion();
    console.log('Version Information:');
    console.log('  Package Version:', info.version);
    console.log('  Runtime:', info.runtime);
    console.log('  Core Version:', info.coreVersion);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Example 6: Error handling
async function example6() {
  console.log('\n=== Example 6: Error Handling ===\n');

  try {
    // This will work or throw an error if no runtime is available
    await agentBooster.apply('test prompt');
    console.log('✓ Runtime available');
  } catch (error) {
    if (error.message.includes('No runtime available')) {
      console.log('⚠ No runtime available. Install WASM: npm install @agent-booster/wasm');
    } else {
      console.error('Unexpected error:', error.message);
    }
  }
}

// Run all examples
async function runExamples() {
  console.log('Agent Booster - Usage Examples\n');
  console.log('='.repeat(50));

  await example1();
  await example2();
  await example3();
  await example4();
  await example5();
  await example6();

  console.log('\n' + '='.repeat(50));
  console.log('Examples complete!\n');
}

// Run if executed directly
if (require.main === module) {
  runExamples().catch(console.error);
}

module.exports = { runExamples };
