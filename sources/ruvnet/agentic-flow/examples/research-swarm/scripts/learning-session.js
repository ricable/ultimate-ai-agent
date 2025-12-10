#!/usr/bin/env node

/**
 * Learning Session - Memory Distillation and Pattern Association
 *
 * This script processes multiple research patterns to create:
 * - Memory distillations (compressed knowledge)
 * - Pattern associations (similarity mappings)
 *
 * Usage:
 *   node scripts/learning-session.js
 *   node scripts/learning-session.js --min-patterns 3
 */

import { getDatabase } from '../lib/db-utils.js';
import { v4 as uuidv4 } from 'uuid';

const MIN_PATTERNS = parseInt(process.argv.find(arg => arg.startsWith('--min-patterns='))?.split('=')[1] || '2');

console.log(`
╔═══════════════════════════════════════════════════════════════════╗
║              AGENTDB LEARNING SESSION                              ║
╚═══════════════════════════════════════════════════════════════════╝

🧠 Processing patterns for memory distillation and association...
📊 Minimum patterns required: ${MIN_PATTERNS}
`);

const db = getDatabase();

// Get all successful patterns
const patterns = db.prepare(`
  SELECT * FROM reasoningbank_patterns
  WHERE success = 1
  ORDER BY reward DESC, created_at DESC
`).all();

console.log(`✅ Found ${patterns.length} successful patterns\n`);

if (patterns.length < MIN_PATTERNS) {
  console.log(`⚠️  Need at least ${MIN_PATTERNS} patterns for distillation`);
  console.log(`   Run more research tasks first!\n`);
  db.close();
  process.exit(0);
}

// Group patterns by task category
const categories = {};
patterns.forEach(p => {
  const category = extractCategory(p.task);
  if (!categories[category]) categories[category] = [];
  categories[category].push(p);
});

console.log(`📂 Categories found:`);
Object.keys(categories).forEach(cat => {
  console.log(`   - ${cat}: ${categories[cat].length} patterns`);
});
console.log('');

// Generate memory distillations for each category
let distillationCount = 0;
for (const [category, categoryPatterns] of Object.entries(categories)) {
  if (categoryPatterns.length < 2) continue;

  console.log(`🔬 Distilling knowledge for category: ${category}`);

  const distillationId = uuidv4();
  const keyInsights = extractInsights(categoryPatterns);
  const successFactors = extractSuccessFactors(categoryPatterns);
  const failurePatterns = extractFailurePatterns(categoryPatterns);
  const bestPractices = extractBestPractices(categoryPatterns);

  db.prepare(`
    INSERT INTO memory_distillations (
      id, source_pattern_ids, task_category,
      key_insights, success_factors, failure_patterns, best_practices,
      confidence_score, usage_count, success_rate, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
  `).run(
    distillationId,
    JSON.stringify(categoryPatterns.map(p => p.id)),
    category,
    JSON.stringify(keyInsights),
    JSON.stringify(successFactors),
    JSON.stringify(failurePatterns),
    JSON.stringify(bestPractices),
    calculateConfidence(categoryPatterns),
    0,
    categoryPatterns.filter(p => p.success).length / categoryPatterns.length
  );

  distillationCount++;
  console.log(`   ✅ Created distillation: ${category} (${categoryPatterns.length} patterns)\n`);
}

// Generate pattern associations (similarity mappings)
console.log(`🔗 Generating pattern associations...`);
let associationCount = 0;

for (let i = 0; i < patterns.length; i++) {
  for (let j = i + 1; j < patterns.length; j++) {
    const similarity = calculateSimilarity(patterns[i], patterns[j]);

    if (similarity > 0.3) { // Only store meaningful associations
      const associationId = uuidv4();
      const associationType = determineAssociationType(patterns[i], patterns[j], similarity);

      db.prepare(`
        INSERT INTO pattern_associations (
          id, pattern_id_a, pattern_id_b,
          similarity_score, association_type,
          learning_value, usage_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
      `).run(
        associationId,
        patterns[i].id,
        patterns[j].id,
        similarity,
        associationType,
        similarity * 0.8, // Learning value based on similarity
        0
      );

      associationCount++;
    }
  }
}

console.log(`   ✅ Created ${associationCount} pattern associations\n`);

// Generate summary statistics
const stats = db.prepare(`
  SELECT
    (SELECT COUNT(*) FROM reasoningbank_patterns) as total_patterns,
    (SELECT COUNT(*) FROM learning_episodes) as total_episodes,
    (SELECT COUNT(*) FROM memory_distillations) as total_distillations,
    (SELECT COUNT(*) FROM pattern_associations) as total_associations,
    (SELECT COUNT(*) FROM vector_embeddings) as total_vectors
`).get();

db.close();

console.log(`
╔═══════════════════════════════════════════════════════════════════╗
║              LEARNING SESSION COMPLETE                             ║
╚═══════════════════════════════════════════════════════════════════╝

📊 Database Statistics:
   Patterns:       ${stats.total_patterns}
   Episodes:       ${stats.total_episodes}
   Distillations:  ${stats.total_distillations}
   Associations:   ${stats.total_associations}
   Vectors:        ${stats.total_vectors}

🎓 New Learning:
   Memory Distillations: +${distillationCount}
   Pattern Associations: +${associationCount}

✅ Learning session completed successfully!
`);

// Helper functions
function extractCategory(task) {
  const keywords = {
    'AI/ML': ['machine learning', 'deep learning', 'AI', 'neural', 'model'],
    'Cloud': ['cloud', 'AWS', 'Azure', 'infrastructure', 'deployment'],
    'Technology': ['technology', 'software', 'system', 'platform'],
    'Research': ['research', 'analyze', 'study', 'investigate']
  };

  const taskLower = task.toLowerCase();
  for (const [category, words] of Object.entries(keywords)) {
    if (words.some(word => taskLower.includes(word))) {
      return category;
    }
  }
  return 'General';
}

function extractInsights(patterns) {
  return patterns.slice(0, 5).map(p => ({
    pattern_id: p.id,
    task: p.task.substring(0, 100),
    reward: p.reward,
    key_finding: `High reward (${p.reward}) achieved through structured approach`
  }));
}

function extractSuccessFactors(patterns) {
  const successful = patterns.filter(p => p.success && p.reward > 0.7);
  return [
    `High reward patterns (n=${successful.length})`,
    `Average confidence: ${(patterns.reduce((sum, p) => sum + (p.confidence || 0), 0) / patterns.length).toFixed(2)}`,
    `Consistent execution time: ~${Math.round(patterns.reduce((sum, p) => sum + (p.latency_ms || 0), 0) / patterns.length / 1000)}s`
  ];
}

function extractFailurePatterns(patterns) {
  const failed = patterns.filter(p => !p.success || p.reward < 0.3);
  if (failed.length === 0) return ['No significant failure patterns detected'];

  return [
    `${failed.length} patterns with suboptimal outcomes`,
    `Common issue: Low reward scores`,
    `Recommendation: Review task complexity and time allocation`
  ];
}

function extractBestPractices(patterns) {
  return [
    'Structured task decomposition leads to higher reward',
    'Maintain execution time under 30 seconds for optimal performance',
    'Clear, specific tasks produce better confidence scores',
    'Regular pattern review improves learning efficacy'
  ];
}

function calculateConfidence(patterns) {
  const avgReward = patterns.reduce((sum, p) => sum + p.reward, 0) / patterns.length;
  const consistencyScore = 1 - (Math.max(...patterns.map(p => p.reward)) - Math.min(...patterns.map(p => p.reward)));
  return Math.min(0.95, (avgReward + consistencyScore) / 2);
}

function calculateSimilarity(p1, p2) {
  const task1 = p1.task.toLowerCase();
  const task2 = p2.task.toLowerCase();

  // Simple word overlap similarity
  const words1 = new Set(task1.split(/\s+/));
  const words2 = new Set(task2.split(/\s+/));

  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);

  const jaccardSimilarity = intersection.size / union.size;

  // Boost similarity if both are successful
  const successBoost = (p1.success && p2.success) ? 0.2 : 0;

  // Boost if rewards are similar
  const rewardSimilarity = 1 - Math.abs(p1.reward - p2.reward);

  return Math.min(0.95, jaccardSimilarity + (successBoost * 0.5) + (rewardSimilarity * 0.3));
}

function determineAssociationType(p1, p2, similarity) {
  if (similarity > 0.8) return 'similar';
  if (Math.abs(p1.reward - p2.reward) > 0.5) return 'contrasting';
  if (p1.agent_type !== p2.agent_type) return 'complementary';
  return 'sequential';
}
