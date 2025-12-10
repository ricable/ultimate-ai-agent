#!/usr/bin/env node

/**
 * Load and prepare maternal life-history trade-off study data
 * Sources: Finnish famine (1866-1868), Quebec population (1621-1800)
 */

const { initializeAgentDB } = require('./initialize-agentdb');

// Sample data structure based on research literature
const finnishFamineData = {
  dataset: 'finnish_famine_1866_1868',
  description: 'Finnish famine cohort analysis',
  citation: 'Helle et al. (2005), Proceedings of the Royal Society B',
  keyFindings: {
    longevityDecrease: 6, // months per child during famine
    normalPeriod: 4, // months per child in normal conditions
    environmentalStressor: 'severe_famine'
  },
  samples: [
    // Structure for actual data points
    {
      motherID: 'FIN_001',
      birthYear: 1850,
      famineExposure: true,
      offspringCount: 6,
      longevity: 68,
      longevityDecreasePerChild: 6,
      socioeconomicStatus: 'lower',
      physiologicalMarkers: {
        estimatedNutritionalStress: 'high',
        reproductiveAge: '15-45'
      }
    }
    // Additional samples would be loaded from actual dataset
  ]
};

const quebecData = {
  dataset: 'quebec_population_1621_1800',
  description: 'Quebec historical population records',
  citation: 'Evolutionary demography studies',
  keyFindings: {
    tradeOffPattern: 'confirmed',
    populationType: 'pre_industrial',
    environmentalVariability: 'moderate'
  },
  samples: [
    {
      motherID: 'QUE_001',
      birthYear: 1700,
      offspringCount: 8,
      longevity: 65,
      environmentalConditions: 'stable',
      physiologicalMarkers: {
        reproductiveSuccess: 'high',
        survivorship: 'moderate'
      }
    }
    // Additional samples would be loaded from actual dataset
  ]
};

async function loadStudyData() {
  console.log('📥 Loading maternal life-history trade-off study data...');

  const db = await initializeAgentDB();

  // Load Finnish famine data
  console.log('📊 Loading Finnish famine dataset...');
  for (const sample of finnishFamineData.samples) {
    await db.insert('maternal_health', {
      ...sample,
      dataset: finnishFamineData.dataset,
      citation: finnishFamineData.citation
    });
  }

  // Load Quebec historical data
  console.log('📊 Loading Quebec historical dataset...');
  for (const sample of quebecData.samples) {
    await db.insert('maternal_health', {
      ...sample,
      dataset: quebecData.dataset,
      citation: quebecData.citation
    });
  }

  // Store metadata
  await db.setMetadata('datasets', {
    finnish_famine: finnishFamineData,
    quebec_population: quebecData
  });

  console.log('✅ Study data loaded successfully');

  // Generate vector embeddings for semantic search
  console.log('🔍 Generating vector embeddings...');
  await db.generateEmbeddings('maternal_health');

  return db;
}

module.exports = { loadStudyData, finnishFamineData, quebecData };

if (require.main === module) {
  loadStudyData()
    .then(() => process.exit(0))
    .catch(err => {
      console.error('❌ Data loading failed:', err);
      process.exit(1);
    });
}
