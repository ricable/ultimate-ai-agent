#!/usr/bin/env node

/**
 * Database Optimization Script
 *
 * Applies performance optimizations to the SQLite database:
 * - Creates indexes for frequently queried columns
 * - Enables WAL mode for better concurrency
 * - Optimizes cache settings
 * - Analyzes tables for query planner
 *
 * Usage:
 *   node scripts/optimize-db.js
 */

import { getDatabase } from '../lib/db-utils.js';
import chalk from 'chalk';

console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║              DATABASE OPTIMIZATION                                 ║
╚═══════════════════════════════════════════════════════════════════╝
`));

const db = getDatabase();

// Track optimization status
const optimizations = [];

try {
  // 1. Enable WAL mode for better concurrency
  console.log(chalk.cyan('📝 Enabling WAL mode...'));
  const walResult = db.prepare('PRAGMA journal_mode=WAL').get();
  optimizations.push({ name: 'WAL Mode', status: walResult.journal_mode === 'wal' ? 'enabled' : 'already active' });
  console.log(chalk.green(`   ✅ WAL mode: ${walResult.journal_mode}\n`));

  // 2. Increase cache size to 64MB
  console.log(chalk.cyan('💾 Setting cache size...'));
  db.prepare('PRAGMA cache_size=-64000').run(); // 64MB
  const cacheSize = db.prepare('PRAGMA cache_size').get();
  optimizations.push({ name: 'Cache Size', status: `${Math.abs(cacheSize.cache_size / 1000)}MB` });
  console.log(chalk.green(`   ✅ Cache size: ${Math.abs(cacheSize.cache_size / 1000)}MB\n`));

  // 3. Enable memory-mapped I/O
  console.log(chalk.cyan('🗺️  Enabling memory-mapped I/O...'));
  db.prepare('PRAGMA mmap_size=134217728').run(); // 128MB
  const mmapSize = db.prepare('PRAGMA mmap_size').get();
  optimizations.push({ name: 'Memory Mapping', status: `${mmapSize.mmap_size / 1024 / 1024}MB` });
  console.log(chalk.green(`   ✅ Memory mapping: ${mmapSize.mmap_size / 1024 / 1024}MB\n`));

  // 4. Set synchronous mode for performance
  console.log(chalk.cyan('⚡ Setting synchronous mode...'));
  db.prepare('PRAGMA synchronous=NORMAL').run();
  const syncMode = db.prepare('PRAGMA synchronous').get();
  optimizations.push({ name: 'Synchronous Mode', status: syncMode.synchronous === 1 ? 'NORMAL' : 'FULL' });
  console.log(chalk.green(`   ✅ Synchronous mode: ${syncMode.synchronous === 1 ? 'NORMAL' : 'FULL'}\n`));

  // 5. Create indexes for frequently queried columns
  console.log(chalk.cyan('📇 Creating performance indexes...\n'));

  const indexes = [
    // Research jobs indexes
    {
      name: 'idx_jobs_status',
      table: 'research_jobs',
      sql: 'CREATE INDEX IF NOT EXISTS idx_jobs_status ON research_jobs(status, created_at DESC)'
    },
    {
      name: 'idx_jobs_agent',
      table: 'research_jobs',
      sql: 'CREATE INDEX IF NOT EXISTS idx_jobs_agent ON research_jobs(agent, created_at DESC)'
    },
    {
      name: 'idx_jobs_completed',
      table: 'research_jobs',
      sql: 'CREATE INDEX IF NOT EXISTS idx_jobs_completed ON research_jobs(completed_at DESC) WHERE status = "completed"'
    },

    // ReasoningBank pattern indexes
    {
      name: 'idx_patterns_session',
      table: 'reasoningbank_patterns',
      sql: 'CREATE INDEX IF NOT EXISTS idx_patterns_session ON reasoningbank_patterns(session_id, created_at DESC)'
    },
    {
      name: 'idx_patterns_reward',
      table: 'reasoningbank_patterns',
      sql: 'CREATE INDEX IF NOT EXISTS idx_patterns_reward ON reasoningbank_patterns(reward DESC, success DESC)'
    },
    {
      name: 'idx_patterns_success',
      table: 'reasoningbank_patterns',
      sql: 'CREATE INDEX IF NOT EXISTS idx_patterns_success ON reasoningbank_patterns(success, reward DESC)'
    },
    {
      name: 'idx_patterns_task',
      table: 'reasoningbank_patterns',
      sql: 'CREATE INDEX IF NOT EXISTS idx_patterns_task ON reasoningbank_patterns(task)'
    },

    // Learning episodes indexes
    {
      name: 'idx_episodes_pattern',
      table: 'learning_episodes',
      sql: 'CREATE INDEX IF NOT EXISTS idx_episodes_pattern ON learning_episodes(pattern_id, created_at DESC)'
    },
    {
      name: 'idx_episodes_verdict',
      table: 'learning_episodes',
      sql: 'CREATE INDEX IF NOT EXISTS idx_episodes_verdict ON learning_episodes(verdict, judgment_score DESC)'
    },

    // Vector embeddings indexes
    {
      name: 'idx_vectors_source',
      table: 'vector_embeddings',
      sql: 'CREATE INDEX IF NOT EXISTS idx_vectors_source ON vector_embeddings(source_id, source_type)'
    },
    {
      name: 'idx_vectors_hash',
      table: 'vector_embeddings',
      sql: 'CREATE INDEX IF NOT EXISTS idx_vectors_hash ON vector_embeddings(content_hash)'
    },

    // Memory distillations indexes
    {
      name: 'idx_distillations_category',
      table: 'memory_distillations',
      sql: 'CREATE INDEX IF NOT EXISTS idx_distillations_category ON memory_distillations(task_category, confidence_score DESC)'
    },
    {
      name: 'idx_distillations_usage',
      table: 'memory_distillations',
      sql: 'CREATE INDEX IF NOT EXISTS idx_distillations_usage ON memory_distillations(usage_count DESC, last_used_at DESC)'
    },

    // Pattern associations indexes
    {
      name: 'idx_associations_pattern_a',
      table: 'pattern_associations',
      sql: 'CREATE INDEX IF NOT EXISTS idx_associations_pattern_a ON pattern_associations(pattern_id_a, similarity_score DESC)'
    },
    {
      name: 'idx_associations_pattern_b',
      table: 'pattern_associations',
      sql: 'CREATE INDEX IF NOT EXISTS idx_associations_pattern_b ON pattern_associations(pattern_id_b, similarity_score DESC)'
    },
    {
      name: 'idx_associations_similarity',
      table: 'pattern_associations',
      sql: 'CREATE INDEX IF NOT EXISTS idx_associations_similarity ON pattern_associations(similarity_score DESC, association_type)'
    }
  ];

  let createdCount = 0;
  indexes.forEach(index => {
    try {
      db.prepare(index.sql).run();
      console.log(chalk.green(`   ✅ ${index.name} on ${index.table}`));
      createdCount++;
    } catch (error) {
      if (error.message.includes('already exists')) {
        console.log(chalk.gray(`   ⏭️  ${index.name} (already exists)`));
      } else {
        console.log(chalk.red(`   ❌ ${index.name}: ${error.message}`));
      }
    }
  });

  optimizations.push({ name: 'Indexes Created', status: `${createdCount}/${indexes.length}` });
  console.log(chalk.green(`\n   Created ${createdCount} indexes\n`));

  // 6. Run ANALYZE to update query planner statistics
  console.log(chalk.cyan('📊 Analyzing tables for query optimizer...'));
  db.prepare('ANALYZE').run();
  optimizations.push({ name: 'Query Analysis', status: 'completed' });
  console.log(chalk.green('   ✅ Analysis completed\n'));

  // 7. Check database integrity
  console.log(chalk.cyan('🔍 Checking database integrity...'));
  const integrityResult = db.prepare('PRAGMA integrity_check').get();
  const integrityOk = integrityResult.integrity_check === 'ok';
  optimizations.push({ name: 'Integrity Check', status: integrityOk ? 'passed' : 'failed' });
  console.log(integrityOk ? chalk.green('   ✅ Integrity check passed\n') : chalk.red('   ❌ Integrity check failed\n'));

  // 8. Get database statistics
  console.log(chalk.cyan('📈 Database Statistics:\n'));

  const tables = ['research_jobs', 'reasoningbank_patterns', 'learning_episodes',
                  'vector_embeddings', 'memory_distillations', 'pattern_associations'];

  tables.forEach(table => {
    try {
      const count = db.prepare(`SELECT COUNT(*) as count FROM ${table}`).get();
      console.log(`   ${table}: ${chalk.cyan(count.count)} rows`);
    } catch (error) {
      console.log(`   ${table}: ${chalk.red('error')}`);
    }
  });

  console.log('');

  // Summary
  console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║              OPTIMIZATION SUMMARY                                  ║
╚═══════════════════════════════════════════════════════════════════╝
`));

  optimizations.forEach(opt => {
    console.log(`   ${chalk.bold(opt.name)}: ${chalk.cyan(opt.status)}`);
  });

  console.log(chalk.bold.green(`
\n✅ Database optimization completed successfully!

📌 Performance Tips:
   - WAL mode improves concurrent access
   - Indexes speed up queries by 10-100x
   - Memory mapping reduces disk I/O
   - Run ANALYZE periodically after bulk inserts

💡 Maintenance:
   - Run this script after major data imports
   - Consider VACUUM if database grows significantly
   - Monitor query performance with EXPLAIN QUERY PLAN
`));

} catch (error) {
  console.error(chalk.red(`\n❌ Optimization failed: ${error.message}`));
  process.exit(1);
} finally {
  db.close();
}
