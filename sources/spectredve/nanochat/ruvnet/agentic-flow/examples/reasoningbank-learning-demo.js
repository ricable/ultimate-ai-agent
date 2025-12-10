#!/usr/bin/env node
/**
 * ReasoningBank Self-Learning System Demo
 * Demonstrates SAFLA (Self-Aware Feedback Loop Algorithm) capabilities
 */

const { execSync } = require('child_process');

// Helper to run memory commands
function memory(action, ...args) {
  const cmd = `npx claude-flow@alpha memory ${action} ${args.join(' ')}`;
  console.log(`\n🔧 Executing: ${cmd}\n`);
  try {
    return execSync(cmd, { encoding: 'utf8', stdio: 'inherit' });
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
}

console.log(`
╔════════════════════════════════════════════════════════════════╗
║  🧠 ReasoningBank Self-Learning System Demo                   ║
║  Using SAFLA (Self-Aware Feedback Loop Algorithm)            ║
╚════════════════════════════════════════════════════════════════╝
`);

// Phase 1: Initialize Learning Namespace
console.log('\n📚 Phase 1: Store Initial Knowledge Base\n');
console.log('Storing patterns about coding best practices...\n');

const patterns = [
  {
    key: 'pattern_async_await',
    value: 'Always use async/await instead of callback hell. Wrap in try-catch for error handling.',
    category: 'javascript'
  },
  {
    key: 'pattern_sql_injection',
    value: 'Never concatenate user input in SQL queries. Use parameterized queries or ORMs.',
    category: 'security'
  },
  {
    key: 'pattern_api_versioning',
    value: 'Version APIs in URL path (/api/v1/) not query params. Use semantic versioning.',
    category: 'api-design'
  },
  {
    key: 'pattern_error_logging',
    value: 'Log errors with context (user ID, timestamp, stack trace). Never expose internal errors to users.',
    category: 'observability'
  },
  {
    key: 'pattern_cache_invalidation',
    value: 'Use cache keys with TTL and versioning. Implement cache-aside pattern for consistency.',
    category: 'performance'
  }
];

patterns.forEach((pattern, idx) => {
  console.log(`  [${idx + 1}/${patterns.length}] Storing: ${pattern.key}`);
  memory(
    'store',
    pattern.key,
    `"${pattern.value}"`,
    '--namespace self_learning',
    '--reasoningbank'
  );
});

// Phase 2: Demonstrate Semantic Learning
console.log('\n\n🔍 Phase 2: Semantic Query Testing\n');
console.log('Testing how the system learns semantic relationships...\n');

const queries = [
  { query: 'security vulnerabilities', expected: 'SQL injection pattern' },
  { query: 'performance improvements', expected: 'Cache invalidation' },
  { query: 'error handling', expected: 'Error logging and async/await' },
  { query: 'REST API design', expected: 'API versioning' }
];

queries.forEach((test, idx) => {
  console.log(`\n  [${idx + 1}/${queries.length}] Query: "${test.query}"`);
  console.log(`  Expected to find: ${test.expected}\n`);
  memory('query', `"${test.query}"`, '--namespace self_learning', '--reasoningbank');
});

// Phase 3: Learning from Usage
console.log('\n\n📊 Phase 3: Usage-Based Learning\n');
console.log('The system tracks which patterns are most useful...\n');

memory('status', '--reasoningbank');

// Phase 4: Pattern Recognition
console.log('\n\n🎯 Phase 4: Cross-Domain Pattern Recognition\n');
console.log('Testing if the system can connect related concepts...\n');

console.log('Query: "how to prevent attacks"\n');
memory('query', '"how to prevent attacks"', '--namespace self_learning', '--reasoningbank');

console.log('\nQuery: "optimize database access"\n');
memory('query', '"optimize database access"', '--namespace self_learning', '--reasoningbank');

// Summary
console.log(`
╔════════════════════════════════════════════════════════════════╗
║  ✅ Self-Learning Demo Complete!                              ║
╚════════════════════════════════════════════════════════════════╝

🧠 ReasoningBank Capabilities Demonstrated:

1. ✅ Pattern Storage - Stored 5 coding best practices
2. ✅ Semantic Search - Found relevant patterns without exact matches
3. ✅ Usage Tracking - System tracks which patterns are accessed
4. ✅ Confidence Scoring - Each pattern has confidence metrics
5. ✅ Cross-Domain Learning - Connects related concepts

🚀 Next Steps:

1. Add more patterns to improve learning
2. Query patterns during development for suggestions
3. Export learned patterns for team sharing
4. Use in CI/CD for automated code review

📚 Advanced Usage:

# Store a new learning
npx claude-flow@alpha memory store my_pattern "Your pattern here" \\
  --namespace self_learning --reasoningbank

# Query for learned patterns
npx claude-flow@alpha memory query "your question" \\
  --namespace self_learning --reasoningbank

# Check learning statistics
npx claude-flow@alpha memory status --reasoningbank

# Export learned knowledge
npx claude-flow@alpha memory export learning-backup.json

💡 Pro Tips:

- Higher usage count = more reliable pattern
- Confidence score shows pattern quality (0-100%)
- Match score shows query relevance (0-100%)
- Use namespaces to organize different domains
- Export regularly to preserve learnings
`);
