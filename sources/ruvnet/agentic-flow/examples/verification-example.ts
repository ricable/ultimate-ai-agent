/**
 * Verification System Usage Examples
 * Demonstrates complete anti-hallucination workflow
 */

import {
  VerificationSystem,
  ConfidenceScorer,
  VerificationPipeline,
  LeanAgenticIntegration,
  StrangeLoopsDetector,
  AgentDBIntegration,
  MedicalCitation,
  ProviderFeedback,
  FeatureVector,
  CausalModel,
} from '../src/verification';

/**
 * Example 1: Basic Confidence Scoring
 */
async function example1_confidenceScoring() {
  console.log('\n=== Example 1: Confidence Scoring ===\n');

  const scorer = new ConfidenceScorer();

  const citations: MedicalCitation[] = [
    {
      id: 'cit1',
      type: 'meta-analysis',
      title: 'Meta-analysis of ACE inhibitors in heart failure',
      year: 2023,
      citationCount: 850,
      impactFactor: 18.5,
      evidenceLevel: 'A',
      doi: '10.1001/jama.2023.12345',
    },
    {
      id: 'cit2',
      type: 'clinical-trial',
      title: 'Randomized trial of ACE inhibitors',
      year: 2022,
      citationCount: 420,
      impactFactor: 12.3,
      evidenceLevel: 'A',
      doi: '10.1056/NEJMoa2022001',
    },
  ];

  const score = await scorer.calculateConfidence(
    'ACE inhibitors demonstrate mortality benefit in patients with heart failure',
    citations,
    { sampleSize: 5000, confidenceInterval: [0.15, 0.35] }
  );

  console.log('Confidence Score:', score.overall.toFixed(3));
  console.log('Citation Strength:', score.citationStrength.toFixed(3));
  console.log('Medical Agreement:', score.medicalAgreement.toFixed(3));
  console.log('Expert Consensus:', score.expertConsensus.toFixed(3));
  console.log('Is Confident?', scorer.isConfident(score) ? 'YES' : 'NO');
  console.log('Confidence Level:', scorer.getConfidenceLevel(score.overall));

  if (score.contradictions.length > 0) {
    console.log('\nContradictions Detected:');
    score.contradictions.forEach(c => console.log(`  - ${c}`));
  }
}

/**
 * Example 2: Verification Pipeline with Hallucination Detection
 */
async function example2_verificationPipeline() {
  console.log('\n=== Example 2: Verification Pipeline ===\n');

  const pipeline = new VerificationPipeline();

  // Test Case 1: Good claim with citations
  const goodInput = {
    claim: 'Evidence from multiple randomized controlled trials suggests that statins may reduce cardiovascular events in high-risk patients',
    citations: [
      {
        id: 'cit1',
        type: 'meta-analysis',
        title: 'Statin efficacy meta-analysis',
        year: 2023,
        citationCount: 1200,
        impactFactor: 20,
        evidenceLevel: 'A',
      },
    ],
  };

  console.log('Testing GOOD claim...');
  const goodResult = await pipeline.preOutputVerification(goodInput);
  console.log('Verified:', goodResult.verified);
  console.log('Confidence:', goodResult.confidence.overall.toFixed(3));
  console.log('Warnings:', goodResult.warnings.length);
  console.log('Hallucinations:', goodResult.hallucinations.length);

  // Test Case 2: Problematic claim
  const badInput = {
    claim: 'This treatment always cures cancer permanently with 150% success rate and absolutely zero side effects',
    citations: [],
  };

  console.log('\nTesting BAD claim...');
  const badResult = await pipeline.preOutputVerification(badInput);
  console.log('Verified:', badResult.verified);
  console.log('Confidence:', badResult.confidence.overall.toFixed(3));
  console.log('Requires Review:', badResult.requiresReview);

  console.log('\nHallucinations Detected:');
  badResult.hallucinations.forEach(h => {
    console.log(`  - [${h.severity.toUpperCase()}] ${h.type}: ${h.description}`);
    if (h.suggestion) console.log(`    Suggestion: ${h.suggestion}`);
  });

  console.log('\nSuggestions:');
  badResult.suggestions.forEach(s => console.log(`  - ${s}`));
}

/**
 * Example 3: Causal Inference Validation
 */
async function example3_causalInference() {
  console.log('\n=== Example 3: Causal Inference ===\n');

  const leanAgentic = new LeanAgenticIntegration();

  const causalModel: CausalModel = {
    variables: [
      { name: 'medication', type: 'treatment', observed: true, distribution: 'binary' },
      { name: 'blood_pressure', type: 'outcome', observed: true, distribution: 'continuous' },
      { name: 'age', type: 'confounder', observed: true, distribution: 'continuous' },
      { name: 'bmi', type: 'confounder', observed: true, distribution: 'continuous' },
    ],
    relationships: [
      { from: 'medication', to: 'blood_pressure', type: 'direct', strength: 0.75 },
      { from: 'age', to: 'blood_pressure', type: 'confounded', strength: 0.45 },
      { from: 'bmi', to: 'blood_pressure', type: 'confounded', strength: 0.35 },
    ],
    assumptions: ['SUTVA', 'Ignorability', 'Positivity', 'Consistency'],
    confounders: ['age', 'bmi'],
  };

  const data = {
    effectEstimate: -12.5, // mmHg reduction
    standardError: 2.1,
    sampleSize: 1500,
    randomized: true,
  };

  const result = await leanAgentic.validateCausalInference(
    'Medication reduces blood pressure',
    data,
    causalModel
  );

  console.log('Causal Effect:', result.effect.toFixed(2), 'mmHg');
  console.log('95% CI:', `[${result.confidence[0].toFixed(2)}, ${result.confidence[1].toFixed(2)}]`);
  console.log('P-Value:', result.pValue.toFixed(4));
  console.log('Statistically Significant:', result.significant ? 'YES' : 'NO');
  console.log('Study Method:', result.method);

  console.log('\nAssumptions:');
  result.assumptions.forEach(a => {
    const status = a.satisfied ? '✓' : '✗';
    console.log(`  ${status} ${a.assumption} [${a.risk} risk]`);
    console.log(`    ${a.evidence}`);
  });

  if (result.threats.length > 0) {
    console.log('\nBias Threats:');
    result.threats.forEach(t => {
      console.log(`  - [${t.severity.toUpperCase()}] ${t.type}: ${t.description}`);
      if (t.mitigation) console.log(`    Mitigation: ${t.mitigation}`);
    });
  }
}

/**
 * Example 4: Circular Reasoning and Contradiction Detection
 */
async function example4_logicalPatterns() {
  console.log('\n=== Example 4: Logical Pattern Detection ===\n');

  const detector = new StrangeLoopsDetector();

  // Test circular reasoning
  console.log('Testing Circular Reasoning...');
  const circularText = 'A is effective because it works well. It works well because it is effective. Therefore, A is effective.';
  const circularPatterns = await detector.detectCircularReasoning(circularText);

  if (circularPatterns.length > 0) {
    console.log('Circular patterns detected:', circularPatterns.length);
    circularPatterns.forEach(p => {
      console.log(`  - [${p.severity.toUpperCase()}] ${p.description}`);
      console.log(`    Chain: ${p.chain.join(' → ')}`);
    });
  }

  // Test contradictions
  console.log('\nTesting Contradictions...');
  const contradictoryText = 'The medication always works in all patients. The medication never works in any patient.';
  const contradictions = await detector.detectContradictions(contradictoryText);

  if (contradictions.length > 0) {
    console.log('Contradictions detected:', contradictions.length);
    contradictions.forEach(p => {
      console.log(`  - [${p.severity.toUpperCase()}] ${p.description}`);
    });
  }

  // Test causal chain
  console.log('\nTesting Causal Chain Validation...');
  const chain = {
    nodes: [
      { id: 'smoking', claim: 'Smoking occurs', evidence: ['study1'], confidence: 0.9 },
      { id: 'inflammation', claim: 'Inflammation increases', evidence: ['study2'], confidence: 0.85 },
      { id: 'cancer', claim: 'Cancer develops', evidence: ['study3'], confidence: 0.8 },
    ],
    edges: [
      { from: 'smoking', to: 'inflammation', type: 'causes' as const, strength: 0.85 },
      { from: 'inflammation', to: 'cancer', type: 'causes' as const, strength: 0.75 },
    ],
    valid: true,
    cycles: [],
    contradictions: [],
  };

  const chainResult = await detector.validateCausalChain(chain);
  console.log('Chain Valid:', chainResult.valid);
  console.log('Chain Strength:', chainResult.strength.toFixed(3));
  console.log('Issues:', chainResult.issues.length);
}

/**
 * Example 5: Learning from Provider Corrections
 */
async function example5_learningSystem() {
  console.log('\n=== Example 5: Learning System ===\n');

  const agentDB = new AgentDBIntegration();

  // Simulate provider corrections
  const corrections = [
    {
      claim: 'Treatment cures all cases',
      originalConfidence: 0.85,
      feedback: {
        reviewerId: 'provider_001',
        approved: false,
        corrections: [
          {
            type: 'factual' as const,
            original: 'cures all cases',
            corrected: 'shows efficacy in many cases',
            importance: 'high' as const,
          },
        ],
        confidenceAssessment: 0.35,
        reasoning: 'Overstated efficacy - no treatment cures all cases',
        categories: ['oncology', 'evidence-overstatement'],
      },
      features: {
        citationCount: 2,
        peerReviewedRatio: 0.5,
        recencyScore: 0.7,
        evidenceLevelScore: 0.6,
        contradictionCount: 0,
        hallucinationFlags: 3,
        textLength: 50,
        quantitativeClaims: 0,
      },
    },
    {
      claim: 'Meta-analyses demonstrate consistent benefit',
      originalConfidence: 0.75,
      feedback: {
        reviewerId: 'provider_002',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.88,
        reasoning: 'Well-supported claim with appropriate language',
        categories: ['cardiology', 'evidence-based'],
      },
      features: {
        citationCount: 5,
        peerReviewedRatio: 1.0,
        recencyScore: 0.9,
        evidenceLevelScore: 0.95,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 80,
        quantitativeClaims: 0,
      },
    },
  ];

  // Learn from corrections
  console.log('Training on provider corrections...');
  for (const correction of corrections) {
    await agentDB.learnFromCorrection(
      correction.claim,
      correction.originalConfidence,
      correction.feedback,
      correction.features
    );
  }

  // Check model statistics
  const modelStats = agentDB.getModelStatistics();
  console.log('\nModel Statistics:');
  console.log('Training Examples:', modelStats.trainingExamples);
  console.log('Model Accuracy:', modelStats.accuracy.toFixed(3));

  // Check pattern statistics
  const patternStats = agentDB.getPatternStatistics();
  console.log('\nPattern Statistics:');
  console.log('Total Patterns:', patternStats.totalPatterns);
  console.log('Reliable Patterns:', patternStats.reliablePatterns);
  console.log('Average Reliability:', patternStats.averageReliability.toFixed(3));

  // Test prediction
  console.log('\nTesting Predictions...');
  const testFeatures: FeatureVector = {
    citationCount: 8,
    peerReviewedRatio: 0.9,
    recencyScore: 0.85,
    evidenceLevelScore: 0.9,
    contradictionCount: 0,
    hallucinationFlags: 0,
    textLength: 120,
    quantitativeClaims: 1,
  };

  const predicted = agentDB.predictConfidence(testFeatures);
  console.log('Predicted Confidence:', predicted.toFixed(3));

  const adjustment = await agentDB.getConfidenceAdjustment(testFeatures, ['cardiology']);
  console.log('Learning Adjustment:', adjustment.adjustment.toFixed(3));
  console.log('Reason:', adjustment.reason);
}

/**
 * Example 6: Complete End-to-End Verification
 */
async function example6_completeSystem() {
  console.log('\n=== Example 6: Complete Verification System ===\n');

  const system = new VerificationSystem();

  const input = {
    claim: 'Recent systematic reviews and meta-analyses suggest that aspirin therapy may provide cardiovascular benefit in selected high-risk populations',
    citations: [
      {
        id: 'cit1',
        type: 'meta-analysis',
        title: 'Aspirin for primary prevention of cardiovascular events',
        year: 2023,
        citationCount: 750,
        impactFactor: 16.5,
        evidenceLevel: 'A',
      },
      {
        id: 'cit2',
        type: 'guideline',
        title: 'ACC/AHA Guideline on aspirin use',
        year: 2022,
        citationCount: 1200,
        evidenceLevel: 'A',
      },
    ],
    features: {
      citationCount: 2,
      peerReviewedRatio: 1.0,
      recencyScore: 0.95,
      evidenceLevelScore: 1.0,
      contradictionCount: 0,
      hallucinationFlags: 0,
      textLength: 140,
      quantitativeClaims: 0,
    },
    context: ['cardiology', 'evidence-based-medicine'],
    metadata: {
      timestamp: Date.now(),
      source: 'medical-agent-v1',
    },
  };

  console.log('Running complete verification...\n');
  const result = await system.verify(input);

  console.log('VERIFICATION RESULT:');
  console.log('==================');
  console.log('Verified:', result.verified ? '✓ PASS' : '✗ FAIL');
  console.log('Overall Confidence:', result.confidence.overall.toFixed(3));
  console.log('Requires Review:', result.requiresReview ? 'YES' : 'NO');

  console.log('\nConfidence Breakdown:');
  console.log('  Statistical:', result.confidence.statistical.toFixed(3));
  console.log('  Citation Strength:', result.confidence.citationStrength.toFixed(3));
  console.log('  Medical Agreement:', result.confidence.medicalAgreement.toFixed(3));
  console.log('  Expert Consensus:', result.confidence.expertConsensus.toFixed(3));

  if (result.hallucinations.length > 0) {
    console.log('\nHallucinations:', result.hallucinations.length);
    result.hallucinations.forEach(h => {
      console.log(`  - [${h.severity}] ${h.type}: ${h.description}`);
    });
  } else {
    console.log('\nHallucinations: None detected ✓');
  }

  if (result.logicalPatterns.length > 0) {
    console.log('\nLogical Patterns:', result.logicalPatterns.length);
  } else {
    console.log('\nLogical Patterns: None detected ✓');
  }

  if (result.contradictions.length > 0) {
    console.log('\nContradictions:', result.contradictions.length);
  } else {
    console.log('\nContradictions: None detected ✓');
  }

  console.log('\nLearning Adjustment:', result.learningAdjustment.adjustment.toFixed(3));
  if (result.learningAdjustment.reason) {
    console.log('Reason:', result.learningAdjustment.reason);
  }

  // System statistics
  const stats = system.getStatistics();
  console.log('\nSystem Statistics:');
  console.log('  Model Training Examples:', stats.model.trainingExamples);
  console.log('  Model Accuracy:', stats.model.accuracy.toFixed(3));
  console.log('  Total Patterns:', stats.patterns.totalPatterns);
  console.log('  Reliable Patterns:', stats.patterns.reliablePatterns);
}

/**
 * Run all examples
 */
async function runAllExamples() {
  console.log('╔═══════════════════════════════════════════════════╗');
  console.log('║  VERIFICATION SYSTEM EXAMPLES                     ║');
  console.log('║  Anti-Hallucination & Medical Accuracy System     ║');
  console.log('╚═══════════════════════════════════════════════════╝');

  try {
    await example1_confidenceScoring();
    await example2_verificationPipeline();
    await example3_causalInference();
    await example4_logicalPatterns();
    await example5_learningSystem();
    await example6_completeSystem();

    console.log('\n╔═══════════════════════════════════════════════════╗');
    console.log('║  ALL EXAMPLES COMPLETED SUCCESSFULLY              ║');
    console.log('╚═══════════════════════════════════════════════════╝\n');
  } catch (error) {
    console.error('\n❌ Error running examples:', error);
  }
}

// Run examples if executed directly
if (require.main === module) {
  runAllExamples();
}

export {
  example1_confidenceScoring,
  example2_verificationPipeline,
  example3_causalInference,
  example4_logicalPatterns,
  example5_learningSystem,
  example6_completeSystem,
};
