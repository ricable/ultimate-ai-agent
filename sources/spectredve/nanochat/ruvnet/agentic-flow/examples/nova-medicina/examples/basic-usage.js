/**
 * Basic Usage Examples for Nova Medicina
 *
 * This file demonstrates simple symptom analysis,
 * reading confidence scores, and understanding results.
 */

const { MedicalAnalysisService } = require('../src/services/medical-analysis.service');
const { AntiHallucinationService } = require('../src/services/anti-hallucination.service');

// ============================================
// Example 1: Simple Symptom Analysis
// ============================================

async function simpleSymptomAnalysis() {
  console.log('=== Example 1: Simple Symptom Analysis ===\n');

  try {
    const medicalService = new MedicalAnalysisService();

    // Create a basic medical query
    const query = {
      condition: 'Type 2 Diabetes',
      symptoms: [
        'increased thirst',
        'frequent urination',
        'unexplained weight loss',
        'fatigue'
      ],
      patientContext: {
        age: 45,
        gender: 'male',
        medicalHistory: ['hypertension'],
        medications: ['lisinopril']
      }
    };

    console.log('Analyzing symptoms:', query.symptoms.join(', '));
    console.log('Patient age:', query.patientContext.age);
    console.log('\nGenerating analysis...\n');

    // Perform analysis
    const analysis = await medicalService.analyzeCondition(query);

    // Display results
    console.log('Analysis Complete!');
    console.log('================\n');
    console.log('Primary Diagnosis:', analysis.diagnosis.condition);
    console.log('ICD-10 Code:', analysis.diagnosis.icdCode);
    console.log('Confidence:', (analysis.diagnosis.confidence * 100).toFixed(1) + '%');
    console.log('\nDescription:', analysis.diagnosis.description);

    console.log('\nKey Recommendations:');
    analysis.recommendations.forEach((rec, idx) => {
      console.log(`  ${idx + 1}. ${rec.description}`);
      console.log(`     Priority: ${rec.priority}`);
    });

    console.log('\nSuccess! ✓');

    return analysis;

  } catch (error) {
    console.error('Error during analysis:', error.message);
    throw error;
  }
}

// ============================================
// Example 2: Reading Confidence Scores
// ============================================

async function understandingConfidenceScores() {
  console.log('\n\n=== Example 2: Understanding Confidence Scores ===\n');

  try {
    const medicalService = new MedicalAnalysisService();
    const antiHallucinationService = new AntiHallucinationService();

    // Analyze a condition
    const query = {
      condition: 'Influenza',
      symptoms: ['fever', 'cough', 'body aches', 'headache'],
      patientContext: {
        age: 28,
        gender: 'female'
      }
    };

    const analysis = await medicalService.analyzeCondition(query);

    // Calculate confidence score
    const confidenceResult = await antiHallucinationService.calculateConfidenceScore(analysis);

    console.log('Overall Confidence Score:', (confidenceResult.overall * 100).toFixed(1) + '%');
    console.log('\nBreakdown by Component:');
    console.log('========================\n');

    // Display breakdown
    console.log('1. Diagnosis Confidence:', (confidenceResult.breakdown.diagnosisConfidence * 100).toFixed(1) + '%');
    console.log('   Weight: 30%');
    console.log('   Impact: Based on evidence strength and contraindications\n');

    console.log('2. Citation Quality:', (confidenceResult.breakdown.citationQuality * 100).toFixed(1) + '%');
    console.log('   Weight: 25%');
    console.log('   Impact: Verified DOI/PubMed citations\n');

    console.log('3. Knowledge Base Validation:', (confidenceResult.breakdown.knowledgeBaseValidation * 100).toFixed(1) + '%');
    console.log('   Weight: 20%');
    console.log('   Impact: Matches diagnostic criteria\n');

    console.log('4. Contradiction Check:', (confidenceResult.breakdown.contradictionCheck * 100).toFixed(1) + '%');
    console.log('   Weight: 15%');
    console.log('   Impact: No conflicting statements\n');

    console.log('5. Provider Alignment:', (confidenceResult.breakdown.providerAlignment * 100).toFixed(1) + '%');
    console.log('   Weight: 10%');
    console.log('   Impact: Historical provider agreement\n');

    // Interpretation
    console.log('Interpretation:');
    console.log('===============\n');

    if (confidenceResult.overall >= 0.90) {
      console.log('✓ HIGH CONFIDENCE - Analysis can be auto-approved');
    } else if (confidenceResult.overall >= 0.75) {
      console.log('⚠ MEDIUM CONFIDENCE - Standard review recommended');
    } else if (confidenceResult.overall >= 0.50) {
      console.log('⚠ LOW CONFIDENCE - Provider review required');
    } else {
      console.log('✗ VERY LOW CONFIDENCE - Urgent provider review needed');
    }

    // Display warnings if any
    if (confidenceResult.warnings && confidenceResult.warnings.length > 0) {
      console.log('\n⚠ Warnings:');
      confidenceResult.warnings.forEach((warning, idx) => {
        console.log(`  ${idx + 1}. [${warning.severity.toUpperCase()}] ${warning.message}`);
      });
    }

    return confidenceResult;

  } catch (error) {
    console.error('Error calculating confidence:', error.message);
    throw error;
  }
}

// ============================================
// Example 3: Understanding Analysis Results
// ============================================

async function interpretingResults() {
  console.log('\n\n=== Example 3: Interpreting Analysis Results ===\n');

  try {
    const medicalService = new MedicalAnalysisService();

    const query = {
      condition: 'Pneumonia',
      symptoms: ['fever', 'productive cough', 'chest pain', 'difficulty breathing'],
      patientContext: {
        age: 65,
        gender: 'male',
        medicalHistory: ['COPD', 'smoking history']
      },
      isEmergency: false
    };

    const analysis = await medicalService.analyzeCondition(query);

    console.log('Complete Analysis Structure:');
    console.log('============================\n');

    // 1. Primary Diagnosis
    console.log('1. PRIMARY DIAGNOSIS');
    console.log('   Condition:', analysis.diagnosis.condition);
    console.log('   ICD-10:', analysis.diagnosis.icdCode);
    console.log('   Confidence:', (analysis.diagnosis.confidence * 100).toFixed(1) + '%');
    console.log('   Evidence:', analysis.diagnosis.evidence.join(', '));
    console.log();

    // 2. Differential Diagnoses
    if (analysis.differentialDiagnoses && analysis.differentialDiagnoses.length > 0) {
      console.log('2. DIFFERENTIAL DIAGNOSES');
      analysis.differentialDiagnoses.forEach((diff, idx) => {
        console.log(`   ${idx + 1}. ${diff.condition} (${diff.icdCode})`);
        console.log(`      Probability: ${(diff.confidence * 100).toFixed(1)}%`);
      });
      console.log();
    }

    // 3. Recommendations
    console.log('3. RECOMMENDATIONS');
    const priorityGroups = {
      urgent: [],
      high: [],
      medium: [],
      low: []
    };

    analysis.recommendations.forEach(rec => {
      priorityGroups[rec.priority].push(rec);
    });

    Object.entries(priorityGroups).forEach(([priority, recs]) => {
      if (recs.length > 0) {
        console.log(`   ${priority.toUpperCase()}:`);
        recs.forEach(rec => {
          console.log(`   - ${rec.type}: ${rec.description}`);
        });
      }
    });
    console.log();

    // 4. Required Tests
    if (analysis.requiredTests && analysis.requiredTests.length > 0) {
      console.log('4. DIAGNOSTIC TESTS');
      analysis.requiredTests.forEach(test => {
        console.log(`   - ${test.name}`);
        console.log(`     Purpose: ${test.purpose}`);
        console.log(`     Priority: ${test.priority}`);
      });
      console.log();
    }

    // 5. Follow-up
    if (analysis.followUp) {
      console.log('5. FOLLOW-UP');
      console.log(`   Timeframe: ${analysis.followUp.timeframe}`);
      console.log(`   Reason: ${analysis.followUp.reason}`);
      console.log();
    }

    // 6. Citations
    if (analysis.citations && analysis.citations.length > 0) {
      console.log('6. MEDICAL CITATIONS');
      analysis.citations.slice(0, 3).forEach((citation, idx) => {
        console.log(`   ${idx + 1}. ${citation.title}`);
        console.log(`      Source: ${citation.source}`);
        if (citation.doi) console.log(`      DOI: ${citation.doi}`);
      });
      console.log();
    }

    console.log('Result interpretation complete! ✓');

    return analysis;

  } catch (error) {
    console.error('Error interpreting results:', error.message);
    throw error;
  }
}

// ============================================
// Example 4: Error Handling
// ============================================

async function errorHandlingExample() {
  console.log('\n\n=== Example 4: Error Handling ===\n');

  try {
    const medicalService = new MedicalAnalysisService();

    // Example with invalid input
    const invalidQuery = {
      condition: '', // Empty condition
      symptoms: [], // No symptoms
      patientContext: {} // No context
    };

    console.log('Attempting analysis with invalid input...\n');

    await medicalService.analyzeCondition(invalidQuery);

  } catch (error) {
    console.log('✓ Error caught successfully!');
    console.log('Error type:', error.constructor.name);
    console.log('Error message:', error.message);
    console.log('\nProper error handling demonstrated:');
    console.log('1. Validate input before analysis');
    console.log('2. Use try-catch blocks');
    console.log('3. Provide meaningful error messages');
    console.log('4. Log errors for debugging');
  }
}

// ============================================
// Run All Examples
// ============================================

async function runAllExamples() {
  console.log('Nova Medicina - Basic Usage Examples');
  console.log('====================================\n');

  try {
    // Run each example
    await simpleSymptomAnalysis();
    await understandingConfidenceScores();
    await interpretingResults();
    await errorHandlingExample();

    console.log('\n\n✓ All examples completed successfully!');
    console.log('\nNext Steps:');
    console.log('- Check cli-examples.sh for command-line usage');
    console.log('- See api-client.js for REST API integration');
    console.log('- Review advanced-workflows.js for complex scenarios');

  } catch (error) {
    console.error('\n✗ Error running examples:', error.message);
    process.exit(1);
  }
}

// Export functions for use in other scripts
module.exports = {
  simpleSymptomAnalysis,
  understandingConfidenceScores,
  interpretingResults,
  errorHandlingExample,
  runAllExamples
};

// Run examples if executed directly
if (require.main === module) {
  runAllExamples();
}
