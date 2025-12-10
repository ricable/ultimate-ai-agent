/**
 * Nova Medicina API Client Examples
 *
 * Demonstrates REST API integration, WebSocket real-time updates,
 * and comprehensive error handling for the medical analysis system.
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');

// ============================================
// Configuration
// ============================================

const API_CONFIG = {
  baseURL: process.env.MEDAI_API_URL || 'http://localhost:3000',
  apiKey: process.env.MEDAI_API_KEY || 'medai_demo_key_12345',
  timeout: 30000,
  wsURL: process.env.MEDAI_WS_URL || 'ws://localhost:3000'
};

// ============================================
// API Client Class
// ============================================

class NovaMedicinaClient extends EventEmitter {
  constructor(config = API_CONFIG) {
    super();
    this.config = config;
    this.axios = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': config.apiKey
      }
    });

    // Add response interceptor for error handling
    this.axios.interceptors.response.use(
      response => response,
      error => this.handleError(error)
    );
  }

  // Error handling
  handleError(error) {
    if (error.response) {
      // Server responded with error status
      const apiError = new Error(error.response.data.message || 'API Error');
      apiError.status = error.response.status;
      apiError.code = error.response.data.code;
      apiError.details = error.response.data.details;
      throw apiError;
    } else if (error.request) {
      // No response received
      throw new Error('No response from server. Check your connection.');
    } else {
      // Request setup error
      throw new Error(`Request error: ${error.message}`);
    }
  }

  // Submit analysis
  async analyze(query) {
    try {
      const response = await this.axios.post('/api/analyze', query);
      return response.data;
    } catch (error) {
      console.error('Analysis failed:', error.message);
      throw error;
    }
  }

  // Get analysis result
  async getAnalysis(analysisId) {
    try {
      const response = await this.axios.get(`/api/analysis/${analysisId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get analysis:', error.message);
      throw error;
    }
  }

  // Submit provider review
  async submitReview(review) {
    try {
      const response = await this.axios.post('/api/provider/review', review);
      return response.data;
    } catch (error) {
      console.error('Review submission failed:', error.message);
      throw error;
    }
  }

  // Notify provider
  async notifyProvider(notification) {
    try {
      const response = await this.axios.post('/api/provider/notify', notification);
      return response.data;
    } catch (error) {
      console.error('Provider notification failed:', error.message);
      throw error;
    }
  }

  // Get system metrics
  async getMetrics() {
    try {
      const response = await this.axios.get('/api/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to get metrics:', error.message);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    try {
      const response = await this.axios.get('/api/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error.message);
      throw error;
    }
  }

  // WebSocket connection for real-time updates
  connectWebSocket(analysisId) {
    const ws = new WebSocket(`${this.config.wsURL}?analysisId=${analysisId}`);

    ws.on('open', () => {
      console.log('WebSocket connected');
      this.emit('connected');
    });

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        this.emit(message.type, message.payload);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    });

    ws.on('close', () => {
      console.log('WebSocket disconnected');
      this.emit('disconnected');
    });

    return ws;
  }
}

// ============================================
// Example 1: Basic API Usage
// ============================================

async function example1_BasicAnalysis() {
  console.log('=== Example 1: Basic API Analysis ===\n');

  const client = new NovaMedicinaClient();

  try {
    // Check API health first
    console.log('Checking API health...');
    const health = await client.healthCheck();
    console.log('API Status:', health.status);
    console.log('Version:', health.version);
    console.log();

    // Submit analysis
    console.log('Submitting analysis request...');
    const query = {
      condition: 'Type 2 Diabetes',
      symptoms: [
        'increased thirst',
        'frequent urination',
        'unexplained weight loss'
      ],
      patientContext: {
        age: 45,
        gender: 'male',
        medicalHistory: ['hypertension'],
        medications: ['lisinopril']
      }
    };

    const result = await client.analyze(query);

    console.log('Analysis ID:', result.id);
    console.log('Status:', result.status);
    console.log('Diagnosis:', result.diagnosis.condition);
    console.log('Confidence:', (result.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log('Requires Provider Review:', result.requiresProviderReview);
    console.log();

    // Retrieve full analysis
    console.log('Retrieving full analysis...');
    const fullAnalysis = await client.getAnalysis(result.id);

    console.log('Recommendations:', fullAnalysis.recommendations.length);
    console.log('Citations:', fullAnalysis.citations.length);
    console.log('Warnings:', fullAnalysis.warnings ? fullAnalysis.warnings.length : 0);

    console.log('\n✓ Basic analysis completed successfully!\n');

    return result;

  } catch (error) {
    console.error('Error:', error.message);
    if (error.status) console.error('Status Code:', error.status);
    if (error.code) console.error('Error Code:', error.code);
    throw error;
  }
}

// ============================================
// Example 2: Real-Time Updates with WebSocket
// ============================================

async function example2_RealtimeUpdates() {
  console.log('=== Example 2: Real-Time Updates ===\n');

  const client = new NovaMedicinaClient();

  return new Promise(async (resolve, reject) => {
    try {
      // Submit analysis
      console.log('Submitting analysis...');
      const query = {
        condition: 'Pneumonia',
        symptoms: ['fever', 'cough', 'chest pain', 'difficulty breathing'],
        patientContext: {
          age: 65,
          gender: 'male'
        }
      };

      const result = await client.analyze(query);
      console.log('Analysis ID:', result.id);
      console.log('Connecting to WebSocket for real-time updates...\n');

      // Connect WebSocket
      const ws = client.connectWebSocket(result.id);

      // Listen for events
      client.on('connected', () => {
        console.log('✓ Connected to real-time updates');
      });

      client.on('analysis_progress', (data) => {
        console.log(`Progress: ${data.progress}% - ${data.stage}`);
      });

      client.on('analysis_update', (data) => {
        console.log('Update:', data.message);
        if (data.confidence) {
          console.log('Current Confidence:', (data.confidence * 100).toFixed(1) + '%');
        }
      });

      client.on('analysis_warning', (data) => {
        console.log('⚠ Warning:', data.message);
        console.log('   Severity:', data.severity);
      });

      client.on('analysis_complete', (data) => {
        console.log('\n✓ Analysis complete!');
        console.log('Final Status:', data.status);
        console.log('Confidence:', (data.confidence * 100).toFixed(1) + '%');
        ws.close();
        resolve(data);
      });

      client.on('error', (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      });

      // Set timeout
      setTimeout(() => {
        console.log('Demo timeout reached');
        ws.close();
        resolve(result);
      }, 10000);

    } catch (error) {
      console.error('Error:', error.message);
      reject(error);
    }
  });
}

// ============================================
// Example 3: Provider Review Workflow
// ============================================

async function example3_ProviderReview() {
  console.log('=== Example 3: Provider Review Workflow ===\n');

  const client = new NovaMedicinaClient();

  try {
    // Submit analysis that requires review
    console.log('Submitting analysis with moderate confidence...');
    const query = {
      condition: 'Rare Autoimmune Disorder',
      symptoms: ['joint pain', 'fatigue', 'rash'],
      patientContext: {
        age: 35,
        gender: 'female'
      }
    };

    const analysis = await client.analyze(query);
    console.log('Analysis ID:', analysis.id);
    console.log('Confidence:', (analysis.confidenceScore.overall * 100).toFixed(1) + '%');
    console.log('Requires Review:', analysis.requiresProviderReview);
    console.log();

    // Notify provider
    if (analysis.requiresProviderReview) {
      console.log('Notifying healthcare provider...');
      const notification = {
        analysisId: analysis.id,
        providerId: 'PROVIDER_001',
        priority: analysis.confidenceScore.overall < 0.5 ? 'urgent' : 'high',
        channels: ['email', 'sms']
      };

      const notificationResult = await client.notifyProvider(notification);
      console.log('Notification sent:', notificationResult.status);
      console.log('Channels used:', notificationResult.channels.join(', '));
      console.log();
    }

    // Simulate provider review
    console.log('Simulating provider review...');
    console.log('(In production, provider would review through their dashboard)');
    console.log();

    // Submit provider review
    console.log('Submitting provider review...');
    const review = {
      analysisId: analysis.id,
      providerId: 'PROVIDER_001',
      decision: 'modified',
      comments: 'Updated diagnosis based on additional lab results',
      modifications: {
        diagnosis: {
          condition: 'Systemic Lupus Erythematosus',
          confidence: 0.92,
          evidence: ['ANA positive', 'Anti-dsDNA antibodies']
        },
        additionalTests: [
          'Complement levels',
          'Renal function tests'
        ]
      }
    };

    const reviewResult = await client.submitReview(review);
    console.log('Review submitted:', reviewResult.status);
    console.log('Updated Diagnosis:', reviewResult.updatedAnalysis.diagnosis.condition);
    console.log('New Confidence:', (reviewResult.updatedAnalysis.diagnosis.confidence * 100).toFixed(1) + '%');

    console.log('\n✓ Provider review workflow completed!\n');

    return reviewResult;

  } catch (error) {
    console.error('Error:', error.message);
    throw error;
  }
}

// ============================================
// Example 4: Error Handling
// ============================================

async function example4_ErrorHandling() {
  console.log('=== Example 4: Error Handling ===\n');

  const client = new NovaMedicinaClient();

  // Test 1: Invalid API key
  console.log('Test 1: Invalid API key');
  try {
    const invalidClient = new NovaMedicinaClient({
      ...API_CONFIG,
      apiKey: 'invalid_key'
    });
    await invalidClient.healthCheck();
  } catch (error) {
    console.log('✓ Caught authentication error');
    console.log('  Status:', error.status);
    console.log('  Message:', error.message);
  }
  console.log();

  // Test 2: Invalid input
  console.log('Test 2: Invalid analysis input');
  try {
    await client.analyze({
      condition: '', // Empty condition
      symptoms: [] // No symptoms
    });
  } catch (error) {
    console.log('✓ Caught validation error');
    console.log('  Message:', error.message);
    if (error.details) {
      console.log('  Details:', JSON.stringify(error.details, null, 2));
    }
  }
  console.log();

  // Test 3: Network timeout
  console.log('Test 3: Request timeout handling');
  try {
    const timeoutClient = new NovaMedicinaClient({
      ...API_CONFIG,
      timeout: 1 // Very short timeout
    });
    await timeoutClient.analyze({
      condition: 'Test',
      symptoms: ['symptom1']
    });
  } catch (error) {
    console.log('✓ Caught timeout error');
    console.log('  Message:', error.message);
  }
  console.log();

  // Test 4: Non-existent analysis
  console.log('Test 4: Retrieve non-existent analysis');
  try {
    await client.getAnalysis('non_existent_id_12345');
  } catch (error) {
    console.log('✓ Caught not found error');
    console.log('  Status:', error.status);
    console.log('  Message:', error.message);
  }

  console.log('\n✓ Error handling tests completed!\n');
}

// ============================================
// Example 5: Batch Processing
// ============================================

async function example5_BatchProcessing() {
  console.log('=== Example 5: Batch Processing ===\n');

  const client = new NovaMedicinaClient();

  try {
    // Multiple patients to analyze
    const patients = [
      {
        condition: 'Diabetes',
        symptoms: ['increased thirst', 'frequent urination'],
        patientContext: { age: 45, gender: 'male' }
      },
      {
        condition: 'Hypertension',
        symptoms: ['headache', 'dizziness'],
        patientContext: { age: 60, gender: 'female' }
      },
      {
        condition: 'Asthma',
        symptoms: ['wheezing', 'shortness of breath'],
        patientContext: { age: 32, gender: 'male' }
      }
    ];

    console.log(`Processing ${patients.length} patients in batch...\n`);

    // Process all analyses in parallel
    const results = await Promise.all(
      patients.map((patient, idx) => {
        console.log(`Submitting analysis ${idx + 1}/${patients.length}...`);
        return client.analyze(patient)
          .then(result => {
            console.log(`✓ Analysis ${idx + 1} complete: ${result.diagnosis.condition}`);
            return result;
          })
          .catch(error => {
            console.error(`✗ Analysis ${idx + 1} failed: ${error.message}`);
            return null;
          });
      })
    );

    // Filter successful results
    const successful = results.filter(r => r !== null);

    console.log(`\nBatch complete: ${successful.length}/${patients.length} successful`);

    // Aggregate statistics
    const avgConfidence = successful.reduce((sum, r) =>
      sum + r.confidenceScore.overall, 0) / successful.length;

    const needsReview = successful.filter(r => r.requiresProviderReview).length;

    console.log('\nBatch Statistics:');
    console.log('  Average Confidence:', (avgConfidence * 100).toFixed(1) + '%');
    console.log('  Require Review:', needsReview);
    console.log('  Auto-Approved:', successful.length - needsReview);

    console.log('\n✓ Batch processing completed!\n');

    return results;

  } catch (error) {
    console.error('Batch processing error:', error.message);
    throw error;
  }
}

// ============================================
// Example 6: System Metrics
// ============================================

async function example6_SystemMetrics() {
  console.log('=== Example 6: System Metrics ===\n');

  const client = new NovaMedicinaClient();

  try {
    console.log('Fetching system metrics...\n');

    const metrics = await client.getMetrics();

    console.log('Learning System Metrics:');
    console.log('========================');
    console.log('  Total Analyses:', metrics.totalAnalyses);
    console.log('  Successful:', metrics.successfulAnalyses);
    console.log('  Failed:', metrics.failedAnalyses);
    console.log('  Success Rate:', (metrics.successRate * 100).toFixed(1) + '%');
    console.log();

    console.log('Pattern Recognition:');
    console.log('  Patterns Learned:', metrics.patternsLearned);
    console.log('  Pattern Matches:', metrics.patternMatches);
    console.log('  Match Accuracy:', (metrics.patternAccuracy * 100).toFixed(1) + '%');
    console.log();

    console.log('Confidence Scoring:');
    console.log('  Average Confidence:', (metrics.avgConfidence * 100).toFixed(1) + '%');
    console.log('  High Confidence (>0.90):', metrics.highConfidenceCount);
    console.log('  Medium Confidence (0.75-0.90):', metrics.mediumConfidenceCount);
    console.log('  Low Confidence (<0.75):', metrics.lowConfidenceCount);
    console.log();

    console.log('Provider Reviews:');
    console.log('  Total Reviews:', metrics.totalReviews);
    console.log('  Approved:', metrics.approvedReviews);
    console.log('  Modified:', metrics.modifiedReviews);
    console.log('  Rejected:', metrics.rejectedReviews);
    console.log('  Approval Rate:', (metrics.approvalRate * 100).toFixed(1) + '%');

    console.log('\n✓ System metrics retrieved!\n');

    return metrics;

  } catch (error) {
    console.error('Error fetching metrics:', error.message);
    throw error;
  }
}

// ============================================
// Run All Examples
// ============================================

async function runAllExamples() {
  console.log('Nova Medicina API Client Examples');
  console.log('==================================\n');

  try {
    await example1_BasicAnalysis();
    await example2_RealtimeUpdates();
    await example3_ProviderReview();
    await example4_ErrorHandling();
    await example5_BatchProcessing();
    await example6_SystemMetrics();

    console.log('\n✓ All API examples completed successfully!');
    console.log('\nNext Steps:');
    console.log('- Review provider-integration.js for provider workflows');
    console.log('- Check advanced-workflows.js for complex scenarios');
    console.log('- See mcp-integration.md for Claude Desktop integration');

  } catch (error) {
    console.error('\n✗ Error running examples:', error.message);
    process.exit(1);
  }
}

// Export client and examples
module.exports = {
  NovaMedicinaClient,
  example1_BasicAnalysis,
  example2_RealtimeUpdates,
  example3_ProviderReview,
  example4_ErrorHandling,
  example5_BatchProcessing,
  example6_SystemMetrics,
  runAllExamples
};

// Run if executed directly
if (require.main === module) {
  runAllExamples();
}
