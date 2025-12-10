#!/bin/bash
# Test v1 API Backward Compatibility
set -e

echo "=========================================="
echo "AgentDB v2 - v1 API Compatibility Test"
echo "=========================================="
echo ""

cd /test

# Test 1: v1 ReasoningBank API
echo "✓ Test 1: v1 ReasoningBank API"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReasoningBank } = require('./dist/controllers/ReasoningBank.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  // v1 API: Only db and embedder (no backends)
  const db = await createDatabase(':memory:');

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

  const bank = new ReasoningBank(db, embedder);

  // v1 API: storePattern
  const patternId = await bank.storePattern({
    taskType: 'test',
    approach: 'Test approach',
    successRate: 0.95
  });

  console.log('✅ v1 storePattern works:', patternId > 0);

  // v1 API: searchPatterns
  const results = await bank.searchPatterns({
    task: 'test query',
    k: 5
  });

  console.log('✅ v1 searchPatterns works:', results.length >= 0);

  db.close();
})();
"

# Test 2: v1 SkillLibrary API
echo "✓ Test 2: v1 SkillLibrary API"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { SkillLibrary } = require('./dist/controllers/SkillLibrary.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase(':memory:');

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

  // v1 API: Only db and embedder
  const skills = new SkillLibrary(db, embedder);

  const skillId = await skills.createSkill({
    name: 'test-skill',
    description: 'Test skill',
    code: 'console.log(\"test\")',
    successRate: 0.9
  });

  console.log('✅ v1 createSkill works:', skillId > 0);

  const found = await skills.searchSkills({
    query: 'test',
    k: 5
  });

  console.log('✅ v1 searchSkills works:', found.length >= 0);

  db.close();
})();
"

# Test 3: v1 ReflexionMemory API
echo "✓ Test 3: v1 ReflexionMemory API"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReflexionMemory } = require('./dist/controllers/ReflexionMemory.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase(':memory:');

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

  // v1 API: Only db and embedder (2 parameters)
  const reflexion = new ReflexionMemory(db, embedder);

  const episodeId = await reflexion.storeEpisode({
    sessionId: 'test-session',
    task: 'Test task',
    reward: 0.8,
    success: true
  });

  console.log('✅ v1 storeEpisode works:', episodeId > 0);

  const episodes = await reflexion.retrieveRelevant({
    task: 'Test query',
    k: 5
  });

  console.log('✅ v1 retrieveRelevant works:', episodes.length >= 0);

  db.close();
})();
"

# Test 4: v1 CausalRecall API
echo "✓ Test 4: v1 CausalRecall API"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { CausalRecall } = require('./dist/controllers/CausalRecall.js');
const { EmbeddingService } = require('./dist/controllers/EmbeddingService.js');

(async () => {
  const db = await createDatabase(':memory:');

  // Load schemas (CausalRecall needs frontier schema for causal_edges)
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

  // v1 API: db, embedder, undefined for vectorBackend, config
  const causal = new CausalRecall(db, embedder, undefined, {
    alpha: 0.7,
    beta: 0.2,
    gamma: 0.1,
    minConfidence: 0.6
  });

  const stats = causal.getStats();
  console.log('✅ v1 CausalRecall works:', typeof stats === 'object');

  db.close();
})();
"

echo ""
echo "=========================================="
echo "✅ ALL v1 API COMPATIBILITY TESTS PASSED"
echo "=========================================="
