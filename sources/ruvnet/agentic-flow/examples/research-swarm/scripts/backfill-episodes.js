#!/usr/bin/env node

/**
 * Backfill Learning Episodes
 *
 * Populates missing episode_number and improvement_rate fields
 * in existing learning_episodes records.
 *
 * Usage:
 *   node scripts/backfill-episodes.js
 */

import { getDatabase } from '../lib/db-utils.js';
import chalk from 'chalk';

console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║              BACKFILL LEARNING EPISODES                            ║
╚═══════════════════════════════════════════════════════════════════╝
`));

const db = getDatabase();

try {
  // Get all learning episodes that need backfilling
  const episodes = db.prepare(`
    SELECT * FROM learning_episodes
    WHERE episode_number IS NULL OR improvement_rate IS NULL
  `).all();

  console.log(chalk.cyan(`📊 Found ${episodes.length} episodes needing backfill\n`));

  if (episodes.length === 0) {
    console.log(chalk.green('✅ All episodes are already up to date!\n'));
    db.close();
    process.exit(0);
  }

  // Group episodes by pattern_id
  const byPattern = {};
  episodes.forEach(episode => {
    if (!byPattern[episode.pattern_id]) {
      byPattern[episode.pattern_id] = [];
    }
    byPattern[episode.pattern_id].push(episode);
  });

  console.log(chalk.cyan(`📂 Processing ${Object.keys(byPattern).length} patterns...\n`));

  let updatedCount = 0;

  // Process each pattern's episodes
  for (const [patternId, patternEpisodes] of Object.entries(byPattern)) {
    console.log(chalk.dim(`   Pattern: ${patternId.substring(0, 8)}...`));

    // Sort by created_at to assign sequential episode numbers
    patternEpisodes.sort((a, b) => {
      const dateA = new Date(a.started_at || a.created_at);
      const dateB = new Date(b.started_at || b.created_at);
      return dateA - dateB;
    });

    // Update each episode
    patternEpisodes.forEach((episode, index) => {
      const episodeNumber = index + 1;

      // Calculate improvement rate
      const initialScore = episode.initial_score || 0.5;
      const finalScore = episode.final_score || 0.5;
      const improvementRate = finalScore - initialScore;

      // Calculate learning metrics if missing
      const explorationRate = episode.exploration_rate || Math.min(1, (episode.duration_seconds || 0) / 3600);
      const exploitationRate = episode.exploitation_rate || finalScore;
      const learningRate = episode.learning_rate || (improvementRate > 0 ? 0.1 : 0.05);

      // Update the episode
      db.prepare(`
        UPDATE learning_episodes
        SET episode_number = ?,
            improvement_rate = ?,
            exploration_rate = ?,
            exploitation_rate = ?,
            learning_rate = ?
        WHERE id = ?
      `).run(
        episodeNumber,
        improvementRate,
        explorationRate,
        exploitationRate,
        learningRate,
        episode.id
      );

      updatedCount++;
      process.stdout.write(chalk.dim(`\r   Updated ${updatedCount}/${episodes.length} episodes`));
    });
  }

  console.log(chalk.green(`\n\n✅ Successfully backfilled ${updatedCount} learning episodes!\n`));

  // Show summary statistics
  const stats = db.prepare(`
    SELECT
      COUNT(*) as total,
      COUNT(DISTINCT pattern_id) as unique_patterns,
      AVG(episode_number) as avg_episode_number,
      AVG(improvement_rate) as avg_improvement_rate,
      AVG(exploration_rate) as avg_exploration_rate,
      AVG(exploitation_rate) as avg_exploitation_rate,
      AVG(learning_rate) as avg_learning_rate
    FROM learning_episodes
  `).get();

  console.log(chalk.bold('📊 Learning Episodes Summary:'));
  console.log(chalk.dim(`   Total Episodes:          ${stats.total}`));
  console.log(chalk.dim(`   Unique Patterns:         ${stats.unique_patterns}`));
  console.log(chalk.dim(`   Avg Episode Number:      ${stats.avg_episode_number?.toFixed(2) || 'N/A'}`));
  console.log(chalk.dim(`   Avg Improvement Rate:    ${stats.avg_improvement_rate?.toFixed(3) || 'N/A'}`));
  console.log(chalk.dim(`   Avg Exploration Rate:    ${stats.avg_exploration_rate?.toFixed(3) || 'N/A'}`));
  console.log(chalk.dim(`   Avg Exploitation Rate:   ${stats.avg_exploitation_rate?.toFixed(3) || 'N/A'}`));
  console.log(chalk.dim(`   Avg Learning Rate:       ${stats.avg_learning_rate?.toFixed(3) || 'N/A'}`));
  console.log('');

} catch (error) {
  console.error(chalk.red(`\n❌ Backfill failed: ${error.message}`));
  console.error(error.stack);
  process.exit(1);
} finally {
  db.close();
}
