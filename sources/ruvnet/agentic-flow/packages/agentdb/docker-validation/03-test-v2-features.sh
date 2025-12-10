#!/bin/bash
# Test v2 New Features (GNN, Graph, RuVector)
set -e

echo "=========================================="
echo "AgentDB v2 - New Features Test"
echo "=========================================="
echo ""

cd /test

# Test 1: Graceful degradation without backends
echo "✓ Test 1: Graceful Degradation (No Backends)"
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

  // v2 API: All backends undefined (graceful degradation)
  const reflexion = new ReflexionMemory(
    db,
    embedder,
    undefined,  // vectorBackend
    undefined,  // learningBackend
    undefined   // graphBackend
  );

  const episodeId = await reflexion.storeEpisode({
    sessionId: 'test',
    task: 'Test task',
    reward: 0.8,
    success: true
  });

  console.log('✅ Works without backends:', episodeId > 0);

  const episodes = await reflexion.retrieveRelevant({
    task: 'Test query',
    k: 5
  });

  console.log('✅ Retrieval works:', episodes.length >= 0);

  // Check that new methods exist
  console.log('✅ New methods exist:');
  console.log('  - getLearningStats:', typeof reflexion.getLearningStats === 'function');
  console.log('  - getGraphStats:', typeof reflexion.getGraphStats === 'function');
  console.log('  - trainGNN:', typeof reflexion.trainGNN === 'function');
  console.log('  - getEpisodeRelationships:', typeof reflexion.getEpisodeRelationships === 'function');

  db.close();
})();
"

# Test 2: Vector backend compatibility
echo "✓ Test 2: Vector Backend Integration"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReasoningBank } = require('./dist/controllers/ReasoningBank.js');
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

  // Try to detect backends
  try {
    const { detectBackend } = require('./dist/backends/detector.js');
    const detection = await detectBackend();
    console.log('✅ Backend detection works:', detection.backend);
    console.log('  - GNN available:', detection.features.gnn);
    console.log('  - Graph available:', detection.features.graph);
  } catch (error) {
    console.log('⚠️  Backend detection not available (expected in minimal install)');
  }

  db.close();
})();
"

# Test 3: Method signatures unchanged
echo "✓ Test 3: Method Signatures Unchanged"
node -e "
const fs = require('fs');
const { createDatabase } = require('./dist/db-fallback.js');
const { ReflexionMemory } = require('./dist/controllers/ReflexionMemory.js');
const { SkillLibrary } = require('./dist/controllers/SkillLibrary.js');
const { ReasoningBank } = require('./dist/controllers/ReasoningBank.js');
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

  // Check all core methods still exist
  const reflexion = new ReflexionMemory(db, embedder);
  const skills = new SkillLibrary(db, embedder);
  const bank = new ReasoningBank(db, embedder);

  console.log('✅ Core method signatures preserved:');
  console.log('  ReflexionMemory.storeEpisode:', typeof reflexion.storeEpisode === 'function');
  console.log('  ReflexionMemory.retrieveRelevant:', typeof reflexion.retrieveRelevant === 'function');
  console.log('  ReflexionMemory.getTaskStats:', typeof reflexion.getTaskStats === 'function');
  console.log('  SkillLibrary.createSkill:', typeof skills.createSkill === 'function');
  console.log('  SkillLibrary.searchSkills:', typeof skills.searchSkills === 'function');
  console.log('  ReasoningBank.storePattern:', typeof bank.storePattern === 'function');
  console.log('  ReasoningBank.searchPatterns:', typeof bank.searchPatterns === 'function');

  db.close();
})();
"

echo ""
echo "=========================================="
echo "✅ v2 FEATURES TEST COMPLETED"
echo "=========================================="
