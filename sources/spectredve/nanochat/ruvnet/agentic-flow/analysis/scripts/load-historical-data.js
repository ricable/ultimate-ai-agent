#!/usr/bin/env node

/**
 * Historical Dataset Loader for Maternal Life-History Trade-Off Analysis
 * Loads data from multiple famine and population studies
 */

const { AgentDB } = require('agentdb');
const fs = require('fs');
const path = require('path');

/**
 * Sample historical datasets (representative examples)
 * In production, these would be loaded from actual CSV/JSON files
 */

const FINNISH_FAMINE_DATA = [
  {
    motherID: 'FF_1866_001',
    birthYear: 1840,
    offspringCount: 8,
    longevity: 67,
    reproductiveCessationAge: 42,
    interbirthIntervals: [1.5, 2.0, 1.8, 2.2, 1.9, 2.5, 2.1],
    environmentalStress: 8.5,
    physiologicalMarkers: {
      baseline_health: 'fair',
      famine_exposure: 'severe',
      postfamine_recovery: 'partial'
    },
    dataset: 'finnish_famine_1866_1868',
    location: 'Finland'
  },
  {
    motherID: 'FF_1866_002',
    birthYear: 1835,
    offspringCount: 6,
    longevity: 71,
    reproductiveCessationAge: 40,
    interbirthIntervals: [2.0, 2.3, 2.5, 2.8, 3.0],
    environmentalStress: 7.0,
    physiologicalMarkers: {
      baseline_health: 'good',
      famine_exposure: 'moderate',
      postfamine_recovery: 'good'
    },
    dataset: 'finnish_famine_1866_1868',
    location: 'Finland'
  }
];

const QUEBEC_POPULATION_DATA = [
  {
    motherID: 'QC_1700_001',
    birthYear: 1680,
    offspringCount: 12,
    longevity: 62,
    reproductiveCessationAge: 45,
    interbirthIntervals: [1.2, 1.5, 1.3, 1.8, 1.6, 2.0, 1.7, 1.9, 2.2, 2.5, 2.8],
    environmentalStress: 3.0,
    physiologicalMarkers: {
      baseline_health: 'good',
      environmental_conditions: 'stable',
      nutritional_status: 'adequate'
    },
    dataset: 'quebec_population_1621_1800',
    location: 'Quebec'
  }
];

const DUTCH_HUNGER_WINTER_DATA = [
  {
    motherID: 'DHW_1944_001',
    birthYear: 1920,
    offspringCount: 3,
    longevity: 78,
    reproductiveCessationAge: 38,
    interbirthIntervals: [3.5, 4.0],
    environmentalStress: 9.0,
    physiologicalMarkers: {
      baseline_health: 'good',
      famine_exposure: 'severe',
      long_term_effects: 'metabolic_changes',
      epigenetic_modifications: 'documented'
    },
    dataset: 'dutch_hunger_winter_1944_1945',
    location: 'Netherlands'
  }
];

const ENVIRONMENTAL_STRESS_EVENTS = [
  {
    eventID: 'FF_1866',
    eventName: 'Finnish Famine',
    location: 'Finland',
    startYear: 1866,
    endYear: 1868,
    severity: 8.5,
    mortalityRate: 0.15,
    affectedPopulation: 300000,
    duration: 2,
    description: 'Severe famine in Finland caused by crop failures and harsh winters. Approximately 15% of the population died.'
  },
  {
    eventID: 'DHW_1944',
    eventName: 'Dutch Hunger Winter',
    location: 'Netherlands',
    startYear: 1944,
    endYear: 1945,
    severity: 9.0,
    mortalityRate: 0.08,
    affectedPopulation: 4500000,
    duration: 0.5,
    description: 'Acute famine during World War II caused by German blockade. Well-documented epigenetic effects on offspring.'
  },
  {
    eventID: 'SOL_1941',
    eventName: 'Siege of Leningrad',
    location: 'Soviet Union',
    startYear: 1941,
    endYear: 1944,
    severity: 9.5,
    mortalityRate: 0.40,
    affectedPopulation: 2500000,
    duration: 3,
    description: 'Prolonged siege during World War II resulting in severe starvation. Extreme mortality and documented long-term health effects.'
  },
  {
    eventID: 'BF_1974',
    eventName: 'Bangladesh Famine',
    location: 'Bangladesh',
    startYear: 1974,
    endYear: 1975,
    severity: 8.0,
    mortalityRate: 0.10,
    affectedPopulation: 30000000,
    duration: 1,
    description: 'Famine caused by flooding and political instability. Significant impact on maternal and child health.'
  }
];

const PHYSIOLOGICAL_MARKERS = [
  {
    markerID: 'PM_FF_001',
    motherID: 'FF_1866_001',
    telomereLength: 6.2,
    immuneMarkers: {
      il6: 3.5,
      crp: 4.2,
      tnf_alpha: 2.8
    },
    epigeneticMarkers: {
      dna_methylation_age: 72,
      clock_acceleration: 5
    },
    hormonalProfiles: {
      cortisol: 'elevated',
      estrogen: 'low'
    },
    metabolicMarkers: {
      glucose: 105,
      insulin_resistance: 'moderate'
    },
    measurementDate: 1900,
    dataset: 'finnish_famine_1866_1868'
  }
];

const CAUSAL_RELATIONSHIPS = [
  {
    relationshipID: 'CR_001',
    factor1: 'environmental_stress',
    factor2: 'offspring_count',
    correlationStrength: -0.72,
    causalDirection: 'environmental_stress -> offspring_count',
    confidence: 0.95,
    pValue: 0.001,
    sampleSize: 1247,
    evidence: [
      'Finnish Famine cohort analysis',
      'Dutch Hunger Winter longitudinal study',
      'Cross-cultural validation studies'
    ],
    methodology: 'Structural Equation Modeling with instrumental variables'
  },
  {
    relationshipID: 'CR_002',
    factor1: 'offspring_count',
    factor2: 'maternal_longevity',
    correlationStrength: -0.58,
    causalDirection: 'offspring_count -> maternal_longevity',
    confidence: 0.89,
    pValue: 0.005,
    sampleSize: 892,
    evidence: [
      'Quebec population registry',
      'Finnish demographic records',
      'Modern epidemiological studies'
    ],
    methodology: 'Cox proportional hazards regression with competing risks'
  },
  {
    relationshipID: 'CR_003',
    factor1: 'environmental_stress',
    factor2: 'telomere_length',
    correlationStrength: -0.64,
    causalDirection: 'environmental_stress -> telomere_length',
    confidence: 0.92,
    pValue: 0.002,
    sampleSize: 456,
    evidence: [
      'Dutch Hunger Winter biomarker study',
      'Contemporary famine cohort measurements'
    ],
    methodology: 'Linear mixed-effects models with random intercepts'
  }
];

async function loadHistoricalData(db) {
  console.log('📊 Loading historical datasets into AgentDB...');

  // Load maternal health records
  console.log('  → Loading maternal health records...');
  const maternalData = [
    ...FINNISH_FAMINE_DATA,
    ...QUEBEC_POPULATION_DATA,
    ...DUTCH_HUNGER_WINTER_DATA
  ];

  for (const record of maternalData) {
    await db.insert('maternal_health', record);
  }
  console.log(`    ✓ Loaded ${maternalData.length} maternal health records`);

  // Load environmental stress events
  console.log('  → Loading environmental stress events...');
  for (const event of ENVIRONMENTAL_STRESS_EVENTS) {
    await db.insert('environmental_stress', event);
  }
  console.log(`    ✓ Loaded ${ENVIRONMENTAL_STRESS_EVENTS.length} environmental stress events`);

  // Load physiological markers
  console.log('  → Loading physiological markers...');
  for (const marker of PHYSIOLOGICAL_MARKERS) {
    await db.insert('physiological_markers', marker);
  }
  console.log(`    ✓ Loaded ${PHYSIOLOGICAL_MARKERS.length} physiological marker records`);

  // Load causal relationships
  console.log('  → Loading causal relationships...');
  for (const relationship of CAUSAL_RELATIONSHIPS) {
    await db.insert('causal_relationships', relationship);
  }
  console.log(`    ✓ Loaded ${CAUSAL_RELATIONSHIPS.length} causal relationship records`);

  console.log('✅ Historical data loading complete');
}

async function generateEmbeddings(db) {
  console.log('🧠 Generating vector embeddings for semantic search...');

  // Generate embeddings for environmental stress descriptions
  console.log('  → Generating embeddings for environmental stress events...');
  const events = await db.query('SELECT * FROM environmental_stress');
  for (const event of events) {
    const embedding = await db.generateEmbedding(event.description);
    await db.update('environmental_stress', event.eventID, {
      ...event,
      embedding
    });
  }

  // Generate embeddings for causal relationships
  console.log('  → Generating embeddings for causal relationships...');
  const relationships = await db.query('SELECT * FROM causal_relationships');
  for (const rel of relationships) {
    const text = `${rel.factor1} affects ${rel.factor2} with correlation ${rel.correlationStrength}. Evidence: ${rel.evidence.join(', ')}`;
    const embedding = await db.generateEmbedding(text);
    await db.update('causal_relationships', rel.relationshipID, {
      ...rel,
      embedding
    });
  }

  console.log('✅ Vector embeddings generated successfully');
}

// Main execution
async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, '../config/agentdb-config.json'), 'utf8')
  );

  const db = new AgentDB({
    name: config.database.name,
    path: config.database.path
  });

  await db.connect();
  await loadHistoricalData(db);
  await generateEmbeddings(db);
  await db.close();

  console.log('🎉 Data loading complete!');
}

module.exports = { loadHistoricalData, generateEmbeddings };

if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch(err => {
      console.error('❌ Data loading failed:', err);
      process.exit(1);
    });
}
