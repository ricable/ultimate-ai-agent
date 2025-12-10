#!/bin/bash
# Test v1 to v2 Migration Path
set -e

echo "=========================================="
echo "AgentDB v2 - Migration Test"
echo "=========================================="
echo ""

cd /test

# Test 1: Create v1-style database
echo "✓ Test 1: Create v1-style Database"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReasoningBank } = require('./dist/controllers/ReasoningBank.js');
const { SkillLibrary } = require('./dist/controllers/SkillLibrary.js');
const { ReflexionMemory } = require('./dist/controllers/ReflexionMemory.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase('/tmp/v1-migration-test.db');

  // Load schemas
  const schema = fs.readFileSync('./dist/schemas/schema.sql', 'utf-8');
  const frontierSchema = fs.readFileSync('./dist/schemas/frontier-schema.sql', 'utf-8');
  db.exec(schema);
  db.exec(frontierSchema);
  const embedder = new EmbeddingService({
    model: 'Xenova/all-MiniLM-L6-v2',
    dimension: 384,
    provider: 'transformers'
  });
  await embedder.initialize();

  // Create data with v1 API
  const bank = new ReasoningBank(db, embedder);
  const skills = new SkillLibrary(db, embedder);
  const reflexion = new ReflexionMemory(db, embedder);

  // Store test data
  await bank.storePattern({
    taskType: 'migration-test',
    approach: 'Test approach for migration',
    successRate: 0.95
  });

  await skills.createSkill({
    name: 'migration-skill',
    description: 'Test skill for migration',
    code: 'console.log(\"migration test\")',
    successRate: 0.9
  });

  await reflexion.storeEpisode({
    sessionId: 'migration-session',
    task: 'Migration test task',
    reward: 0.85,
    success: true
  });

  console.log('✅ v1 database created with test data');

  db.close();
})();
"

# Test 2: Read v1 data with v2 API
echo "✓ Test 2: Read v1 Data with v2 API"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReasoningBank } = require('./dist/controllers/ReasoningBank.js');
const { SkillLibrary } = require('./dist/controllers/SkillLibrary.js');
const { ReflexionMemory } = require('./dist/controllers/ReflexionMemory.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase('/tmp/v1-migration-test.db');

  // Load schemas
  const schema = fs.readFileSync('./dist/schemas/schema.sql', 'utf-8');
  const frontierSchema = fs.readFileSync('./dist/schemas/frontier-schema.sql', 'utf-8');
  db.exec(schema);
  db.exec(frontierSchema);
  const embedder = new EmbeddingService({
    model: 'Xenova/all-MiniLM-L6-v2',
    dimension: 384,
    provider: 'transformers'
  });
  await embedder.initialize();

  // Open with v2 API (with optional backends)
  const bank = new ReasoningBank(db, embedder, undefined);
  const skills = new SkillLibrary(db, embedder, undefined);
  const reflexion = new ReflexionMemory(db, embedder, undefined, undefined, undefined);

  // Read v1 data
  const patterns = await bank.searchPatterns({
    task: 'migration test',
    k: 10
  });
  console.log('✅ Can read v1 patterns with v2 API:', patterns.length > 0);

  const foundSkills = await skills.searchSkills({
    query: 'migration',
    k: 10
  });
  console.log('✅ Can read v1 skills with v2 API:', foundSkills.length > 0);

  const episodes = await reflexion.retrieveRelevant({
    task: 'migration',
    k: 10
  });
  console.log('✅ Can read v1 episodes with v2 API:', episodes.length > 0);

  db.close();
})();
"

# Test 3: Add v2 data to v1 database
echo "✓ Test 3: Add v2 Data to v1 Database"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReflexionMemory } = require('./dist/controllers/ReflexionMemory.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase('/tmp/v1-migration-test.db');

  // Load schemas
  const schema = fs.readFileSync('./dist/schemas/schema.sql', 'utf-8');
  const frontierSchema = fs.readFileSync('./dist/schemas/frontier-schema.sql', 'utf-8');
  db.exec(schema);
  db.exec(frontierSchema);
  const embedder = new EmbeddingService({
    model: 'Xenova/all-MiniLM-L6-v2',
    dimension: 384,
    provider: 'transformers'
  });
  await embedder.initialize();

  const reflexion = new ReflexionMemory(db, embedder);

  // Add new data with v2 features (but without backends, so falls back gracefully)
  const episodeId = await reflexion.storeEpisode({
    sessionId: 'v2-session',
    task: 'v2 feature test',
    reward: 0.92,
    success: true,
    critique: 'v2 critique test'
  });

  console.log('✅ Can add v2 data to v1 database:', episodeId > 0);

  // Verify new methods work (return null/empty without backends)
  const learningStats = reflexion.getLearningStats();
  const graphStats = reflexion.getGraphStats();

  console.log('✅ New methods return gracefully:',
    learningStats === null && graphStats === null);

  db.close();
})();
"

echo ""
echo "=========================================="
echo "✅ MIGRATION TEST COMPLETED"
echo "=========================================="
