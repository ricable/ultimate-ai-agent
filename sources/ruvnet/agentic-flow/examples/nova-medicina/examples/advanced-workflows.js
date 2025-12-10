/**
 * Advanced Workflows for Nova Medicina
 *
 * Demonstrates complex scenarios including multi-symptom analysis,
 * provider escalation, and learning from feedback.
 */

const { NovaMedicinaClient } = require('./api-client');
const { MedicalAnalysisService } = require('../src/services/medical-analysis.service');
const { AgentDBLearningService } = require('../src/services/agentdb-learning.service');
const { ProviderService } = require('../src/services/provider.service');

// ============================================
// Example 1: Complex Multi-Symptom Analysis
// ============================================

async function example1_MultiSymptomAnalysis() {
  console.log('=== Example 1: Complex Multi-Symptom Analysis ===\n');

  const client = new NovaMedicinaClient();
  const learningService = new AgentDBLearningService();

  try {
    // Patient with multiple overlapping conditions
    console.log('Analyzing complex multi-system presentation...\n');

    const complexCase = {
      patientId: 'PATIENT_12345',
      condition: 'Multi-System Disease Complex',
      symptoms: [
        // Cardiovascular
        'chest pain',
        'shortness of breath',
        'irregular heartbeat',

        // Respiratory
        'persistent cough',
        'wheezing',

        // Systemic
        'fever',
        'fatigue',
        'weight loss',

        // Musculoskeletal
        'joint pain',
        'muscle weakness',

        // Neurological
        'headache',
        'dizziness',
        'confusion'
      ],
      patientContext: {
        age: 58,
        gender: 'male',
        medicalHistory: [
          'hypertension',
          'type 2 diabetes',
          'COPD',
          'previous MI (2 years ago)'
        ],
        medications: [
          'metformin 1000mg BID',
          'lisinopril 20mg daily',
          'atorvastatin 40mg daily',
          'aspirin 81mg daily',
          'tiotropium inhaler'
        ],
        allergies: ['penicillin', 'sulfa drugs'],
        socialHistory: {
          smoking: 'former smoker (quit 2 years ago)',
          alcohol: 'occasional',
          occupation: 'construction worker'
        },
        familyHistory: [
          'father: heart disease, diabetes',
          'mother: autoimmune disorder'
        ],
        vitalSigns: {
          temperature: 38.2,
          heartRate: 108,
          bloodPressure: '145/92',
          respiratoryRate: 24,
          oxygenSaturation: 91,
          weight: 82,
          height: 175
        },
        recentLabs: {
          glucose: 156,
          creatinine: 1.4,
          BNP: 450,
          troponin: 0.02,
          WBC: 14.2,
          CRP: 45
        }
      },
      isEmergency: false
    };

    console.log('Patient Profile:');
    console.log('  Age:', complexCase.patientContext.age);
    console.log('  Symptom Count:', complexCase.symptoms.length);
    console.log('  Conditions:', complexCase.patientContext.medicalHistory.length);
    console.log('  Medications:', complexCase.patientContext.medications.length);
    console.log();

    // Step 1: Initial analysis
    console.log('Step 1: Performing initial multi-system analysis...');
    const initialAnalysis = await client.analyze(complexCase);

    console.log('  Primary Diagnosis:', initialAnalysis.diagnosis.condition);
    console.log('  ICD-10:', initialAnalysis.diagnosis.icdCode);
    console.log('  Confidence:', (initialAnalysis.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log();

    // Step 2: Differential diagnoses
    console.log('Step 2: Evaluating differential diagnoses...');
    if (initialAnalysis.differentialDiagnoses && initialAnalysis.differentialDiagnoses.length > 0) {
      console.log('  Differential Diagnoses:');
      initialAnalysis.differentialDiagnoses.slice(0, 5).forEach((diff, idx) => {
        console.log(`    ${idx + 1}. ${diff.condition} (${diff.icdCode})`);
        console.log(`       Confidence: ${(diff.confidence * 100).toFixed(1)}%`);
        console.log(`       Key factors: ${diff.supportingEvidence.join(', ')}`);
      });
    }
    console.log();

    // Step 3: Search for similar cases
    console.log('Step 3: Searching for similar historical cases...');
    const similarCases = await learningService.findSimilarCases({
      symptoms: complexCase.symptoms.slice(0, 5), // Use top symptoms
      patientAge: complexCase.patientContext.age,
      medicalHistory: complexCase.patientContext.medicalHistory
    });

    console.log(`  Found ${similarCases.length} similar cases`);
    if (similarCases.length > 0) {
      console.log('  Top matches:');
      similarCases.slice(0, 3).forEach((match, idx) => {
        console.log(`    ${idx + 1}. Similarity: ${(match.similarity * 100).toFixed(1)}%`);
        console.log(`       Diagnosis: ${match.diagnosis}`);
        console.log(`       Outcome: ${match.outcome}`);
        console.log(`       Provider approved: ${match.providerApproved ? 'Yes' : 'No'}`);
      });
    }
    console.log();

    // Step 4: Risk stratification
    console.log('Step 4: Risk stratification analysis...');
    const riskAssessment = {
      cardiovascular: {
        risk: 'HIGH',
        factors: ['chest pain', 'history of MI', 'elevated BNP', 'diabetes'],
        score: 0.78
      },
      respiratory: {
        risk: 'MODERATE',
        factors: ['COPD', 'smoking history', 'low O2 sat'],
        score: 0.62
      },
      infectious: {
        risk: 'MODERATE',
        factors: ['fever', 'elevated WBC', 'elevated CRP'],
        score: 0.58
      },
      metabolic: {
        risk: 'MODERATE',
        factors: ['diabetes', 'elevated glucose', 'elevated creatinine'],
        score: 0.54
      }
    };

    Object.entries(riskAssessment).forEach(([system, assessment]) => {
      console.log(`  ${system.toUpperCase()}: ${assessment.risk}`);
      console.log(`    Risk score: ${(assessment.score * 100).toFixed(0)}%`);
      console.log(`    Factors: ${assessment.factors.join(', ')}`);
    });
    console.log();

    // Step 5: Prioritized action plan
    console.log('Step 5: Generating prioritized action plan...');
    const actionPlan = {
      immediate: [
        'ECG to rule out acute coronary syndrome',
        'Chest X-ray for pulmonary assessment',
        'Repeat troponin in 3 hours',
        'Continuous cardiac monitoring'
      ],
      urgent: [
        'Blood cultures if infection suspected',
        'CT chest if PE suspected',
        'Echocardiogram for heart failure assessment',
        'Comprehensive metabolic panel'
      ],
      scheduled: [
        'Cardiology consultation within 24 hours',
        'Pulmonology follow-up',
        'Diabetes management optimization',
        'Medication review and reconciliation'
      ]
    };

    Object.entries(actionPlan).forEach(([priority, actions]) => {
      console.log(`  ${priority.toUpperCase()} ACTIONS:`);
      actions.forEach((action, idx) => {
        console.log(`    ${idx + 1}. ${action}`);
      });
    });
    console.log();

    // Step 6: Clinical decision support
    console.log('Step 6: Clinical decision support recommendations...');
    const decisionSupport = {
      redFlags: [
        'Chest pain with cardiac risk factors - consider ACS',
        'Elevated troponin - serial measurements needed',
        'Low oxygen saturation - may need supplemental O2'
      ],
      considerations: [
        'Multiple comorbidities increase complexity',
        'Drug interactions possible with current medications',
        'Renal function impaired - adjust medication doses'
      ],
      guidelines: [
        'ACC/AHA guidelines for chest pain evaluation',
        'GOLD guidelines for COPD management',
        'ADA guidelines for diabetes in hospitalized patients'
      ]
    };

    console.log('  RED FLAGS:');
    decisionSupport.redFlags.forEach(flag => console.log(`    ⚠ ${flag}`));
    console.log();
    console.log('  CONSIDERATIONS:');
    decisionSupport.considerations.forEach(con => console.log(`    • ${con}`));
    console.log();
    console.log('  RELEVANT GUIDELINES:');
    decisionSupport.guidelines.forEach(guide => console.log(`    📋 ${guide}`));

    console.log('\n✓ Complex multi-symptom analysis completed!\n');

    return {
      analysis: initialAnalysis,
      similarCases,
      riskAssessment,
      actionPlan,
      decisionSupport
    };

  } catch (error) {
    console.error('Error in multi-symptom analysis:', error.message);
    throw error;
  }
}

// ============================================
// Example 2: Intelligent Provider Escalation
// ============================================

async function example2_ProviderEscalation() {
  console.log('=== Example 2: Intelligent Provider Escalation ===\n');

  const client = new NovaMedicinaClient();
  const providerService = new ProviderService();

  try {
    // Analyze case with uncertain diagnosis
    console.log('Analyzing case with diagnostic uncertainty...\n');

    const uncertainCase = {
      condition: 'Unclear Neurological Symptoms',
      symptoms: [
        'progressive weakness',
        'numbness in extremities',
        'difficulty swallowing',
        'double vision',
        'facial drooping'
      ],
      patientContext: {
        age: 42,
        gender: 'female',
        symptomOnset: '3 weeks ago',
        progression: 'gradual worsening'
      }
    };

    const analysis = await client.analyze(uncertainCase);

    console.log('Analysis Results:');
    console.log('  Diagnosis:', analysis.diagnosis.condition);
    console.log('  Confidence:', (analysis.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log('  Requires Review:', analysis.requiresProviderReview);
    console.log();

    // Intelligent escalation logic
    console.log('Evaluating escalation criteria...\n');

    const escalationCriteria = {
      lowConfidence: analysis.confidenceScore.overall < 0.70,
      multipleDifferentials: analysis.differentialDiagnoses.length > 3,
      seriousCondition: analysis.diagnosis.severity === 'serious',
      rapidProgression: uncertainCase.patientContext.progression === 'rapid' ||
                       uncertainCase.patientContext.progression === 'gradual worsening',
      neurologicalSymptoms: uncertainCase.symptoms.some(s =>
        s.includes('weakness') || s.includes('numbness') || s.includes('vision')
      )
    };

    console.log('Escalation Criteria:');
    Object.entries(escalationCriteria).forEach(([criterion, met]) => {
      console.log(`  ${met ? '✓' : '✗'} ${criterion}`);
    });
    console.log();

    // Determine escalation path
    const urgencyLevel = Object.values(escalationCriteria).filter(v => v).length;
    let escalationPath;

    if (urgencyLevel >= 4) {
      escalationPath = {
        level: 'URGENT',
        specialists: ['neurologist', 'critical-care'],
        timeframe: 'within 2 hours',
        channels: ['phone', 'pager', 'sms'],
        message: 'Urgent neurological consultation needed - possible progressive neuromuscular disorder'
      };
    } else if (urgencyLevel >= 2) {
      escalationPath = {
        level: 'HIGH',
        specialists: ['neurologist'],
        timeframe: 'within 24 hours',
        channels: ['email', 'sms'],
        message: 'Neurological evaluation requested for diagnostic clarification'
      };
    } else {
      escalationPath = {
        level: 'ROUTINE',
        specialists: ['primary-care'],
        timeframe: 'within 1 week',
        channels: ['email'],
        message: 'Follow-up recommended for symptom monitoring'
      };
    }

    console.log('Escalation Path Determined:');
    console.log(`  Level: ${escalationPath.level}`);
    console.log(`  Specialists: ${escalationPath.specialists.join(', ')}`);
    console.log(`  Timeframe: ${escalationPath.timeframe}`);
    console.log(`  Channels: ${escalationPath.channels.join(', ')}`);
    console.log(`  Message: ${escalationPath.message}`);
    console.log();

    // Execute escalation
    console.log('Executing escalation workflow...');

    const notifications = await Promise.all(
      escalationPath.specialists.map(async (specialization) => {
        // Find available specialist
        const specialist = await providerService.findAvailableSpecialist({
          specialization,
          urgency: escalationPath.level,
          maxResponseTime: escalationPath.timeframe
        });

        if (specialist) {
          // Send notification
          const notification = await providerService.notifyProvider({
            analysisId: analysis.id,
            providerId: specialist.id,
            priority: escalationPath.level.toLowerCase(),
            channels: escalationPath.channels,
            message: escalationPath.message,
            metadata: {
              escalationReason: Object.keys(escalationCriteria)
                .filter(k => escalationCriteria[k]),
              confidence: analysis.confidenceScore.overall,
              differentialCount: analysis.differentialDiagnoses.length
            }
          });

          console.log(`  ✓ Notified ${specialist.name} (${specialization})`);
          return notification;
        } else {
          console.log(`  ⚠ No available ${specialization} specialist`);
          return null;
        }
      })
    );

    console.log();
    console.log('Escalation Summary:');
    console.log(`  Notifications sent: ${notifications.filter(n => n !== null).length}`);
    console.log(`  Specialists notified: ${escalationPath.specialists.join(', ')}`);
    console.log(`  Expected response: ${escalationPath.timeframe}`);

    console.log('\n✓ Provider escalation completed!\n');

    return { analysis, escalationPath, notifications };

  } catch (error) {
    console.error('Error in provider escalation:', error.message);
    throw error;
  }
}

// ============================================
// Example 3: Continuous Learning from Feedback
// ============================================

async function example3_LearningFromFeedback() {
  console.log('=== Example 3: Continuous Learning from Feedback ===\n');

  const client = new NovaMedicinaClient();
  const learningService = new AgentDBLearningService();

  try {
    // Simulate a week of analyses and provider feedback
    console.log('Simulating continuous learning cycle...\n');

    const weekOfCases = [
      {
        day: 1,
        case: {
          condition: 'pneumonia',
          symptoms: ['fever', 'cough', 'chest pain'],
          patientContext: { age: 65 }
        },
        providerFeedback: {
          decision: 'approved',
          confidence: 0.95,
          comments: 'Excellent diagnosis, confirmed by imaging'
        }
      },
      {
        day: 2,
        case: {
          condition: 'asthma exacerbation',
          symptoms: ['wheezing', 'shortness of breath'],
          patientContext: { age: 32 }
        },
        providerFeedback: {
          decision: 'modified',
          confidence: 0.88,
          comments: 'Correct diagnosis, but missed severity assessment',
          modifications: {
            addedRecommendations: ['Spirometry testing', 'Allergist referral']
          }
        }
      },
      {
        day: 3,
        case: {
          condition: 'viral gastroenteritis',
          symptoms: ['nausea', 'vomiting', 'diarrhea'],
          patientContext: { age: 8 }
        },
        providerFeedback: {
          decision: 'approved',
          confidence: 0.92,
          comments: 'Good call on viral vs bacterial'
        }
      },
      {
        day: 4,
        case: {
          condition: 'migraine',
          symptoms: ['severe headache', 'photophobia', 'nausea'],
          patientContext: { age: 38 }
        },
        providerFeedback: {
          decision: 'rejected',
          confidence: 0.45,
          comments: 'Missed red flags for subarachnoid hemorrhage',
          correctDiagnosis: {
            condition: 'Subarachnoid Hemorrhage',
            reason: 'Sudden onset, worst headache of life'
          }
        }
      },
      {
        day: 5,
        case: {
          condition: 'type 2 diabetes',
          symptoms: ['increased thirst', 'frequent urination', 'fatigue'],
          patientContext: { age: 52 }
        },
        providerFeedback: {
          decision: 'approved',
          confidence: 0.94,
          comments: 'Clear diagnosis, appropriate recommendations'
        }
      }
    ];

    // Process each case
    console.log('Processing weekly cases and feedback...\n');

    const learningMetrics = {
      totalCases: weekOfCases.length,
      approved: 0,
      modified: 0,
      rejected: 0,
      patternsLearned: [],
      confidenceImprovements: []
    };

    for (const dayCase of weekOfCases) {
      console.log(`Day ${dayCase.day}:`);

      // Analyze case
      const analysis = await client.analyze(dayCase.case);
      console.log(`  Initial diagnosis: ${analysis.diagnosis.condition}`);
      console.log(`  AI confidence: ${(analysis.confidenceScore.overall * 100).toFixed(1)}%`);

      // Apply provider feedback
      const feedback = dayCase.providerFeedback;
      console.log(`  Provider decision: ${feedback.decision.toUpperCase()}`);
      console.log(`  Provider confidence: ${(feedback.confidence * 100).toFixed(1)}%`);

      // Update metrics
      learningMetrics[feedback.decision]++;

      // Learn from feedback
      if (feedback.decision === 'approved') {
        console.log('  ✓ Learning: Reinforcing successful pattern');
        await learningService.reinforcePattern({
          analysisId: analysis.id,
          symptoms: dayCase.case.symptoms,
          diagnosis: analysis.diagnosis.condition,
          confidence: feedback.confidence,
          outcome: 'success'
        });

        learningMetrics.patternsLearned.push({
          condition: analysis.diagnosis.condition,
          strength: 'reinforced'
        });

      } else if (feedback.decision === 'modified') {
        console.log('  ⚠ Learning: Adjusting diagnostic approach');
        await learningService.learnModification({
          analysisId: analysis.id,
          originalDiagnosis: analysis.diagnosis.condition,
          modifications: feedback.modifications,
          reason: feedback.comments
        });

        learningMetrics.patternsLearned.push({
          condition: analysis.diagnosis.condition,
          strength: 'adjusted'
        });

      } else if (feedback.decision === 'rejected') {
        console.log('  ✗ Learning: Critical correction needed');
        await learningService.learnFromError({
          analysisId: analysis.id,
          incorrectDiagnosis: analysis.diagnosis.condition,
          correctDiagnosis: feedback.correctDiagnosis.condition,
          symptoms: dayCase.case.symptoms,
          criticalFactors: feedback.correctDiagnosis.reason,
          severity: 'high'
        });

        learningMetrics.patternsLearned.push({
          condition: feedback.correctDiagnosis.condition,
          strength: 'critical_learning'
        });

        // Identify confidence gap
        const confidenceGap = analysis.confidenceScore.overall - feedback.confidence;
        learningMetrics.confidenceImprovements.push({
          condition: analysis.diagnosis.condition,
          gap: confidenceGap,
          correction: feedback.correctDiagnosis.condition
        });
      }

      console.log(`  Comments: ${feedback.comments}`);
      console.log();
    }

    // Analyze learning progress
    console.log('Weekly Learning Summary:');
    console.log('========================\n');
    console.log('Case Distribution:');
    console.log(`  Total cases: ${learningMetrics.totalCases}`);
    console.log(`  Approved: ${learningMetrics.approved} (${(learningMetrics.approved / learningMetrics.totalCases * 100).toFixed(1)}%)`);
    console.log(`  Modified: ${learningMetrics.modified} (${(learningMetrics.modified / learningMetrics.totalCases * 100).toFixed(1)}%)`);
    console.log(`  Rejected: ${learningMetrics.rejected} (${(learningMetrics.rejected / learningMetrics.totalCases * 100).toFixed(1)}%)`);
    console.log();

    console.log('Patterns Learned:');
    learningMetrics.patternsLearned.forEach((pattern, idx) => {
      console.log(`  ${idx + 1}. ${pattern.condition} - ${pattern.strength}`);
    });
    console.log();

    if (learningMetrics.confidenceImprovements.length > 0) {
      console.log('Confidence Calibration:');
      learningMetrics.confidenceImprovements.forEach((improvement, idx) => {
        console.log(`  ${idx + 1}. Over-confident in: ${improvement.condition}`);
        console.log(`     Gap: ${(improvement.gap * 100).toFixed(1)}%`);
        console.log(`     Correct diagnosis: ${improvement.correction}`);
      });
      console.log();
    }

    // Get updated system metrics
    console.log('Retrieving updated system performance...');
    const updatedMetrics = await client.getMetrics();

    console.log('System Performance After Learning:');
    console.log(`  Total analyses: ${updatedMetrics.totalAnalyses}`);
    console.log(`  Success rate: ${(updatedMetrics.successRate * 100).toFixed(1)}%`);
    console.log(`  Average confidence: ${(updatedMetrics.avgConfidence * 100).toFixed(1)}%`);
    console.log(`  Patterns learned: ${updatedMetrics.patternsLearned}`);
    console.log(`  Provider approval rate: ${(updatedMetrics.providerReviews.approvalRate * 100).toFixed(1)}%`);

    console.log('\n✓ Continuous learning cycle completed!\n');

    return { learningMetrics, updatedMetrics };

  } catch (error) {
    console.error('Error in learning from feedback:', error.message);
    throw error;
  }
}

// ============================================
// Run All Examples
// ============================================

async function runAllExamples() {
  console.log('Nova Medicina Advanced Workflows');
  console.log('================================\n');

  try {
    await example1_MultiSymptomAnalysis();
    await example2_ProviderEscalation();
    await example3_LearningFromFeedback();

    console.log('\n✓ All advanced workflow examples completed!');
    console.log('\nKey Takeaways:');
    console.log('- Multi-symptom analysis handles complex cases');
    console.log('- Intelligent escalation ensures appropriate care');
    console.log('- Continuous learning improves system performance');
    console.log('\nFor more information:');
    console.log('- See docs/TUTORIALS.md for comprehensive guides');
    console.log('- Review mcp-integration.md for Claude Desktop setup');
    console.log('- Check basic-usage.js for fundamentals');

  } catch (error) {
    console.error('\n✗ Error running examples:', error.message);
    process.exit(1);
  }
}

// Export functions
module.exports = {
  example1_MultiSymptomAnalysis,
  example2_ProviderEscalation,
  example3_LearningFromFeedback,
  runAllExamples
};

// Run if executed directly
if (require.main === module) {
  runAllExamples();
}
