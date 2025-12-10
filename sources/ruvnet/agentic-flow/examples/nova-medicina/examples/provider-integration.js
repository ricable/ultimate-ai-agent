/**
 * Provider Integration Examples
 *
 * Demonstrates healthcare provider dashboard setup,
 * notification configuration, and review workflow implementation.
 */

const { ProviderService } = require('../src/services/provider.service');
const { NovaMedicinaClient } = require('./api-client');

// ============================================
// Example 1: Provider Dashboard Setup
// ============================================

async function example1_ProviderDashboard() {
  console.log('=== Example 1: Provider Dashboard Setup ===\n');

  const providerService = new ProviderService();

  try {
    // Register a new provider
    console.log('Registering new healthcare provider...');
    const provider = {
      id: 'PROVIDER_001',
      name: 'Dr. Sarah Johnson',
      email: 'sarah.johnson@hospital.com',
      phone: '+1-555-0123',
      specialization: 'Internal Medicine',
      licenseNumber: 'MD-12345',
      hospital: 'General Hospital',
      department: 'Emergency Medicine',
      notificationPreferences: {
        channels: ['email', 'sms', 'push'],
        urgentOnly: false,
        quietHours: {
          enabled: true,
          start: '22:00',
          end: '07:00'
        },
        minimumConfidenceThreshold: 0.75
      }
    };

    const registeredProvider = await providerService.registerProvider(provider);
    console.log('✓ Provider registered:', registeredProvider.name);
    console.log('  Provider ID:', registeredProvider.id);
    console.log('  Specialization:', registeredProvider.specialization);
    console.log('  Notification channels:', registeredProvider.notificationPreferences.channels.join(', '));
    console.log();

    // Get provider dashboard data
    console.log('Loading provider dashboard...');
    const dashboard = await providerService.getProviderDashboard(registeredProvider.id);

    console.log('Dashboard Overview:');
    console.log('  Pending Reviews:', dashboard.pendingReviews);
    console.log('  Completed Today:', dashboard.completedToday);
    console.log('  Urgent Cases:', dashboard.urgentCases);
    console.log('  Average Response Time:', dashboard.avgResponseTime);
    console.log();

    // Get pending reviews
    console.log('Pending Reviews:');
    const pendingReviews = await providerService.getPendingReviews(registeredProvider.id);

    pendingReviews.slice(0, 3).forEach((review, idx) => {
      console.log(`  ${idx + 1}. Analysis ID: ${review.analysisId}`);
      console.log(`     Condition: ${review.diagnosis.condition}`);
      console.log(`     Confidence: ${(review.confidenceScore * 100).toFixed(1)}%`);
      console.log(`     Priority: ${review.priority}`);
      console.log(`     Submitted: ${new Date(review.submittedAt).toLocaleString()}`);
      console.log();
    });

    console.log('✓ Provider dashboard setup completed!\n');

    return { provider: registeredProvider, dashboard };

  } catch (error) {
    console.error('Error setting up provider dashboard:', error.message);
    throw error;
  }
}

// ============================================
// Example 2: Notification Configuration
// ============================================

async function example2_NotificationConfig() {
  console.log('=== Example 2: Notification Configuration ===\n');

  const providerService = new ProviderService();

  try {
    const providerId = 'PROVIDER_001';

    // Configure notification preferences
    console.log('Configuring notification preferences...\n');

    // Scenario 1: Emergency notifications only
    console.log('Scenario 1: Emergency-only notifications');
    const emergencyConfig = {
      channels: ['sms', 'push'],
      urgentOnly: true,
      minimumConfidenceThreshold: 0.0, // All urgent cases
      quietHours: {
        enabled: false // Always notify for emergencies
      }
    };

    await providerService.updateNotificationPreferences(providerId, emergencyConfig);
    console.log('✓ Emergency-only mode activated');
    console.log('  Channels: SMS, Push');
    console.log('  Quiet hours: Disabled for emergencies');
    console.log();

    // Scenario 2: Business hours with quiet time
    console.log('Scenario 2: Business hours with quiet time');
    const businessHoursConfig = {
      channels: ['email', 'sms'],
      urgentOnly: false,
      minimumConfidenceThreshold: 0.70,
      quietHours: {
        enabled: true,
        start: '20:00',
        end: '08:00',
        timezone: 'America/New_York'
      },
      workingHours: {
        enabled: true,
        monday: { start: '09:00', end: '17:00' },
        tuesday: { start: '09:00', end: '17:00' },
        wednesday: { start: '09:00', end: '17:00' },
        thursday: { start: '09:00', end: '17:00' },
        friday: { start: '09:00', end: '17:00' }
      }
    };

    await providerService.updateNotificationPreferences(providerId, businessHoursConfig);
    console.log('✓ Business hours mode configured');
    console.log('  Active hours: Mon-Fri 9 AM - 5 PM');
    console.log('  Quiet hours: 8 PM - 8 AM');
    console.log('  Minimum confidence: 70%');
    console.log();

    // Scenario 3: High-confidence cases only
    console.log('Scenario 3: High-confidence review cases only');
    const highConfidenceConfig = {
      channels: ['email'],
      urgentOnly: false,
      minimumConfidenceThreshold: 0.85,
      notificationTypes: ['provider_review', 'analysis_complete'],
      quietHours: {
        enabled: true,
        start: '18:00',
        end: '09:00'
      }
    };

    await providerService.updateNotificationPreferences(providerId, highConfidenceConfig);
    console.log('✓ High-confidence mode configured');
    console.log('  Only cases with confidence > 85%');
    console.log('  Email notifications only');
    console.log();

    // Test notification routing
    console.log('Testing notification routing...\n');

    const testCases = [
      { confidence: 0.95, priority: 'medium', expected: 'ROUTE TO PROVIDER' },
      { confidence: 0.80, priority: 'high', expected: 'ROUTE TO PROVIDER' },
      { confidence: 0.65, priority: 'medium', expected: 'FILTERED OUT' },
      { confidence: 0.40, priority: 'urgent', expected: 'ROUTE TO PROVIDER (URGENT)' }
    ];

    testCases.forEach((testCase, idx) => {
      const shouldNotify = providerService.shouldNotifyProvider(
        providerId,
        testCase.confidence,
        testCase.priority
      );

      console.log(`Test ${idx + 1}:`);
      console.log(`  Confidence: ${(testCase.confidence * 100).toFixed(0)}%`);
      console.log(`  Priority: ${testCase.priority}`);
      console.log(`  Result: ${shouldNotify ? '✓ NOTIFY' : '✗ SKIP'}`);
      console.log(`  Expected: ${testCase.expected}`);
      console.log();
    });

    console.log('✓ Notification configuration completed!\n');

  } catch (error) {
    console.error('Error configuring notifications:', error.message);
    throw error;
  }
}

// ============================================
// Example 3: Review Workflow Implementation
// ============================================

async function example3_ReviewWorkflow() {
  console.log('=== Example 3: Review Workflow Implementation ===\n');

  const client = new NovaMedicinaClient();
  const providerService = new ProviderService();

  try {
    const providerId = 'PROVIDER_001';

    // Step 1: Receive analysis requiring review
    console.log('Step 1: Analysis submitted for review');
    const query = {
      condition: 'Complex Multi-System Disease',
      symptoms: ['fever', 'joint pain', 'rash', 'fatigue'],
      patientContext: {
        age: 42,
        gender: 'female',
        medicalHistory: ['lupus', 'rheumatoid arthritis']
      }
    };

    const analysis = await client.analyze(query);
    console.log('  Analysis ID:', analysis.id);
    console.log('  Confidence:', (analysis.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log('  Requires Review:', analysis.requiresProviderReview);
    console.log();

    // Step 2: Notify provider
    console.log('Step 2: Notifying provider');
    const notification = await providerService.notifyProvider({
      analysisId: analysis.id,
      providerId: providerId,
      priority: analysis.confidenceScore.overall < 0.5 ? 'urgent' : 'high',
      channels: ['email', 'sms']
    });
    console.log('  ✓ Provider notified via:', notification.channels.join(', '));
    console.log('  Notification ID:', notification.id);
    console.log();

    // Step 3: Provider reviews analysis
    console.log('Step 3: Provider reviewing analysis');
    console.log('  (Provider accesses dashboard and reviews details)');
    console.log('  - Reviews symptoms and diagnosis');
    console.log('  - Checks differential diagnoses');
    console.log('  - Evaluates confidence scores');
    console.log('  - Reviews citations and evidence');
    console.log();

    // Step 4a: Approve analysis
    console.log('Step 4a: Provider approves analysis');
    const approvedReview = {
      analysisId: analysis.id,
      providerId: providerId,
      decision: 'approved',
      comments: 'Diagnosis confirmed. Treatment plan appropriate.',
      timestamp: new Date().toISOString()
    };

    const approvalResult = await client.submitReview(approvedReview);
    console.log('  ✓ Analysis approved');
    console.log('  Status:', approvalResult.status);
    console.log('  Learning updated: Pattern saved for future reference');
    console.log();

    // Alternative Step 4b: Modify analysis
    console.log('Step 4b (Alternative): Provider modifies analysis');
    const modifiedReview = {
      analysisId: analysis.id,
      providerId: providerId,
      decision: 'modified',
      comments: 'Updated based on recent lab results',
      modifications: {
        diagnosis: {
          condition: 'Systemic Lupus Erythematosus with Nephritis',
          icdCode: 'M32.14',
          confidence: 0.92,
          evidence: [
            'Positive ANA test',
            'Elevated anti-dsDNA',
            'Proteinuria present',
            'Complement levels decreased'
          ]
        },
        recommendations: [
          {
            type: 'medication',
            description: 'Start hydroxychloroquine 400mg daily',
            priority: 'high'
          },
          {
            type: 'referral',
            description: 'Nephrology consultation within 1 week',
            priority: 'urgent'
          },
          {
            type: 'monitoring',
            description: 'Monitor kidney function weekly',
            priority: 'high'
          }
        ],
        additionalTests: [
          {
            name: 'Kidney biopsy',
            purpose: 'Assess extent of nephritis',
            priority: 'urgent'
          }
        ]
      }
    };

    console.log('  Modified diagnosis:', modifiedReview.modifications.diagnosis.condition);
    console.log('  New confidence:', (modifiedReview.modifications.diagnosis.confidence * 100).toFixed(1) + '%');
    console.log('  Additional recommendations:', modifiedReview.modifications.recommendations.length);
    console.log();

    // Alternative Step 4c: Reject analysis
    console.log('Step 4c (Alternative): Provider rejects analysis');
    const rejectedReview = {
      analysisId: analysis.id,
      providerId: providerId,
      decision: 'rejected',
      comments: 'Diagnosis does not match clinical presentation. Recommend alternative workup.',
      reasoning: [
        'Symptom timeline inconsistent with suggested diagnosis',
        'Key diagnostic criteria not met',
        'Alternative diagnosis more likely based on physical exam'
      ],
      suggestedActions: [
        'Order comprehensive metabolic panel',
        'Consider infectious disease consultation',
        'Repeat imaging studies'
      ]
    };

    console.log('  ✗ Analysis rejected');
    console.log('  Reason:', rejectedReview.reasoning[0]);
    console.log('  Suggested actions:', rejectedReview.suggestedActions.length);
    console.log();

    // Step 5: Learning system update
    console.log('Step 5: System learning from provider feedback');
    await providerService.updateLearningFromReview({
      analysisId: analysis.id,
      providerId: providerId,
      decision: approvedReview.decision,
      modifications: modifiedReview.modifications
    });
    console.log('  ✓ Pattern recognition updated');
    console.log('  ✓ Confidence scoring improved');
    console.log('  ✓ Provider preferences learned');
    console.log();

    // Step 6: Analytics and metrics
    console.log('Step 6: Review analytics');
    const providerMetrics = await providerService.getProviderMetrics(providerId);
    console.log('  Total reviews:', providerMetrics.totalReviews);
    console.log('  Approval rate:', (providerMetrics.approvalRate * 100).toFixed(1) + '%');
    console.log('  Avg response time:', providerMetrics.avgResponseTime);
    console.log('  Modification rate:', (providerMetrics.modificationRate * 100).toFixed(1) + '%');

    console.log('\n✓ Review workflow completed!\n');

    return { analysis, approvalResult, providerMetrics };

  } catch (error) {
    console.error('Error in review workflow:', error.message);
    throw error;
  }
}

// ============================================
// Example 4: Multi-Provider Coordination
// ============================================

async function example4_MultiProviderCoordination() {
  console.log('=== Example 4: Multi-Provider Coordination ===\n');

  const client = new NovaMedicinaClient();
  const providerService = new ProviderService();

  try {
    // Complex case requiring multiple specialists
    console.log('Submitting complex multi-system case...');
    const complexQuery = {
      condition: 'Suspected Sepsis with Multi-Organ Involvement',
      symptoms: [
        'high fever',
        'rapid heart rate',
        'confusion',
        'decreased urine output',
        'difficulty breathing'
      ],
      patientContext: {
        age: 72,
        gender: 'male',
        medicalHistory: ['diabetes', 'CKD', 'COPD'],
        medications: ['metformin', 'enalapril', 'tiotropium'],
        vitalSigns: {
          temperature: 39.5,
          heartRate: 125,
          bloodPressure: '85/50',
          respiratoryRate: 28,
          oxygenSaturation: 88
        }
      },
      isEmergency: true
    };

    const analysis = await client.analyze(complexQuery);
    console.log('Analysis ID:', analysis.id);
    console.log('Emergency flag: YES');
    console.log('Confidence:', (analysis.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log();

    // Notify multiple specialists
    console.log('Coordinating multi-specialty review...');

    const specialists = [
      {
        id: 'PROVIDER_ER_001',
        name: 'Dr. Emergency Medicine',
        specialization: 'Emergency Medicine',
        priority: 'urgent'
      },
      {
        id: 'PROVIDER_CRITICAL_001',
        name: 'Dr. Critical Care',
        specialization: 'Critical Care',
        priority: 'urgent'
      },
      {
        id: 'PROVIDER_NEPHRO_001',
        name: 'Dr. Nephrology',
        specialization: 'Nephrology',
        priority: 'high'
      },
      {
        id: 'PROVIDER_PULM_001',
        name: 'Dr. Pulmonology',
        specialization: 'Pulmonology',
        priority: 'high'
      }
    ];

    // Notify all specialists in parallel
    const notifications = await Promise.all(
      specialists.map(specialist =>
        providerService.notifyProvider({
          analysisId: analysis.id,
          providerId: specialist.id,
          priority: specialist.priority,
          specialization: specialist.specialization,
          channels: ['push', 'sms', 'pager']
        })
      )
    );

    console.log(`✓ Notified ${specialists.length} specialists:`);
    specialists.forEach((specialist, idx) => {
      console.log(`  ${idx + 1}. ${specialist.name} (${specialist.specialization})`);
      console.log(`     Priority: ${specialist.priority}`);
    });
    console.log();

    // Track review status
    console.log('Tracking multi-provider review status...');
    const reviewStatus = await providerService.getMultiProviderReviewStatus(analysis.id);

    console.log('Review Status:');
    console.log(`  Total providers: ${reviewStatus.totalProviders}`);
    console.log(`  Reviewed: ${reviewStatus.reviewedCount}`);
    console.log(`  Pending: ${reviewStatus.pendingCount}`);
    console.log(`  Consensus reached: ${reviewStatus.consensusReached ? 'YES' : 'NO'}`);
    console.log();

    // Aggregate recommendations
    if (reviewStatus.consensusReached) {
      console.log('Consensus Recommendations:');
      reviewStatus.consensusRecommendations.forEach((rec, idx) => {
        console.log(`  ${idx + 1}. ${rec.description}`);
        console.log(`     Agreed by: ${rec.agreementCount}/${reviewStatus.totalProviders} providers`);
      });
    }

    console.log('\n✓ Multi-provider coordination completed!\n');

    return { analysis, notifications, reviewStatus };

  } catch (error) {
    console.error('Error in multi-provider coordination:', error.message);
    throw error;
  }
}

// ============================================
// Example 5: Provider Analytics Dashboard
// ============================================

async function example5_ProviderAnalytics() {
  console.log('=== Example 5: Provider Analytics Dashboard ===\n');

  const providerService = new ProviderService();

  try {
    const providerId = 'PROVIDER_001';

    // Get comprehensive analytics
    console.log('Loading provider analytics...\n');
    const analytics = await providerService.getProviderAnalytics(providerId, {
      timeframe: 'last_30_days'
    });

    // Performance metrics
    console.log('Performance Metrics:');
    console.log('===================');
    console.log('  Total Reviews:', analytics.totalReviews);
    console.log('  Approved:', analytics.approved, `(${(analytics.approvalRate * 100).toFixed(1)}%)`);
    console.log('  Modified:', analytics.modified, `(${(analytics.modificationRate * 100).toFixed(1)}%)`);
    console.log('  Rejected:', analytics.rejected, `(${(analytics.rejectionRate * 100).toFixed(1)}%)`);
    console.log();

    // Response time metrics
    console.log('Response Time Metrics:');
    console.log('  Average:', analytics.avgResponseTime);
    console.log('  Median:', analytics.medianResponseTime);
    console.log('  Fastest:', analytics.fastestResponse);
    console.log('  Slowest:', analytics.slowestResponse);
    console.log();

    // Quality metrics
    console.log('Quality Metrics:');
    console.log('  Agreement with AI:', (analytics.aiAgreementRate * 100).toFixed(1) + '%');
    console.log('  Patient outcomes tracked:', analytics.outcomesTracked);
    console.log('  Positive outcomes:', (analytics.positiveOutcomeRate * 100).toFixed(1) + '%');
    console.log('  Learning contribution:', analytics.learningContributions, 'patterns');
    console.log();

    // Specialization insights
    console.log('Specialization Insights:');
    console.log('  Primary conditions reviewed:');
    analytics.topConditions.slice(0, 5).forEach((condition, idx) => {
      console.log(`    ${idx + 1}. ${condition.name} (${condition.count} cases)`);
    });
    console.log();

    // Recommendations
    console.log('System Recommendations:');
    analytics.recommendations.forEach((rec, idx) => {
      console.log(`  ${idx + 1}. ${rec.message}`);
      console.log(`     Impact: ${rec.impact}`);
    });

    console.log('\n✓ Provider analytics completed!\n');

    return analytics;

  } catch (error) {
    console.error('Error loading provider analytics:', error.message);
    throw error;
  }
}

// ============================================
// Run All Examples
// ============================================

async function runAllExamples() {
  console.log('Provider Integration Examples');
  console.log('==============================\n');

  try {
    await example1_ProviderDashboard();
    await example2_NotificationConfig();
    await example3_ReviewWorkflow();
    await example4_MultiProviderCoordination();
    await example5_ProviderAnalytics();

    console.log('\n✓ All provider integration examples completed!');
    console.log('\nNext Steps:');
    console.log('- Check advanced-workflows.js for complex scenarios');
    console.log('- Review mcp-integration.md for Claude Desktop setup');
    console.log('- See docs/TUTORIALS.md for comprehensive guides');

  } catch (error) {
    console.error('\n✗ Error running examples:', error.message);
    process.exit(1);
  }
}

// Export functions
module.exports = {
  example1_ProviderDashboard,
  example2_NotificationConfig,
  example3_ReviewWorkflow,
  example4_MultiProviderCoordination,
  example5_ProviderAnalytics,
  runAllExamples
};

// Run if executed directly
if (require.main === module) {
  runAllExamples();
}
