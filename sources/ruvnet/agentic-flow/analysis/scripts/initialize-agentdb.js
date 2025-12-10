#!/usr/bin/env node

/**
 * AgentDB Initialization for Maternal Life-History Trade-Off Analysis
 * Uses reasoning bank, vector search, and self-learning capabilities
 */

const { AgentDB } = require('agentdb');
const fs = require('fs');
const path = require('path');

async function initializeAgentDB() {
  console.log('🧠 Initializing AgentDB with Reasoning Bank...');

  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, '../config/agentdb-config.json'), 'utf8')
  );

  // Initialize database with vector and reasoning capabilities
  const db = new AgentDB({
    name: config.database.name,
    vector: {
      dimension: config.database.vectorDimension,
      quantization: config.database.quantization,
      indexType: config.database.indexType
    },
    learning: {
      enabled: true,
      algorithms: config.database.reasoningBank.algorithms,
      persistence: config.database.reasoningBank.sessionPersistence
    }
  });

  await db.connect();

  // Create collections for different data aspects
  console.log('📦 Creating maternal_health collection...');
  await db.createCollection('maternal_health', {
    schema: {
      motherID: 'string',
      birthYear: 'number',
      offspringCount: 'number',
      longevity: 'number',
      reproductiveCessationAge: 'number',
      interbirthIntervals: 'array',
      environmentalStress: 'number',
      physiologicalMarkers: 'object',
      dataset: 'string',
      location: 'string',
      embedding: 'vector'
    },
    indexes: ['motherID', 'birthYear', 'dataset', 'location']
  });

  console.log('📦 Creating environmental_stress collection...');
  await db.createCollection('environmental_stress', {
    schema: {
      eventID: 'string',
      eventName: 'string',
      location: 'string',
      startYear: 'number',
      endYear: 'number',
      severity: 'number',
      mortalityRate: 'number',
      affectedPopulation: 'number',
      duration: 'number',
      description: 'string',
      embedding: 'vector'
    },
    indexes: ['eventID', 'location', 'startYear', 'endYear']
  });

  console.log('📦 Creating physiological_markers collection...');
  await db.createCollection('physiological_markers', {
    schema: {
      markerID: 'string',
      motherID: 'string',
      telomereLength: 'number',
      immuneMarkers: 'object',
      epigeneticMarkers: 'object',
      hormonalProfiles: 'object',
      metabolicMarkers: 'object',
      measurementDate: 'number',
      dataset: 'string',
      embedding: 'vector'
    },
    indexes: ['markerID', 'motherID', 'dataset', 'measurementDate']
  });

  console.log('📦 Creating causal_relationships collection...');
  await db.createCollection('causal_relationships', {
    schema: {
      relationshipID: 'string',
      factor1: 'string',
      factor2: 'string',
      correlationStrength: 'number',
      causalDirection: 'string',
      confidence: 'number',
      pValue: 'number',
      sampleSize: 'number',
      evidence: 'array',
      methodology: 'string',
      embedding: 'vector'
    },
    indexes: ['relationshipID', 'factor1', 'factor2']
  });

  console.log('📦 Creating novel_patterns collection...');
  await db.createCollection('novel_patterns', {
    schema: {
      patternID: 'string',
      patternType: 'string',
      description: 'string',
      evidence: 'array',
      statisticalSignificance: 'number',
      effectSize: 'number',
      discoveryMethod: 'string',
      discoveryDate: 'number',
      validatedAcrossDatasets: 'boolean',
      embedding: 'vector'
    },
    indexes: ['patternID', 'patternType', 'discoveryMethod']
  });

  // Initialize learning plugins
  console.log('🎓 Configuring learning algorithms...');
  await db.initializeLearning({
    algorithms: config.database.reasoningBank.algorithms,
    memoryPatterns: config.database.reasoningBank.memoryPatterns
  });

  console.log('✅ AgentDB initialized successfully');
  console.log(`📊 Collections: ${await db.listCollections()}`);

  return db;
}

// Export for use in analysis scripts
module.exports = { initializeAgentDB };

if (require.main === module) {
  initializeAgentDB()
    .then(() => process.exit(0))
    .catch(err => {
      console.error('❌ Initialization failed:', err);
      process.exit(1);
    });
}
