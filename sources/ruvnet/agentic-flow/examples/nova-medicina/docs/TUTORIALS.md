# Nova Medicina Tutorials

Comprehensive step-by-step tutorials for using the Nova Medicina medical analysis system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Tutorials](#basic-tutorials)
3. [Intermediate Tutorials](#intermediate-tutorials)
4. [Advanced Tutorials](#advanced-tutorials)
5. [Common Use Cases](#common-use-cases)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

Before starting these tutorials, ensure you have:

- **Node.js** 18.0.0 or higher
- **npm** 9.0.0 or higher
- **Nova Medicina** installed (global or local)
- Basic understanding of medical terminology
- API key (for REST API access)

### Installation

```bash
# Install Nova Medicina globally
npm install -g nova-medicina

# Or install locally in your project
npm install nova-medicina

# Verify installation
medai --version
```

### Initial Configuration

```bash
# Set up basic configuration
medai config set api.key YOUR_API_KEY_HERE
medai config set provider.id YOUR_PROVIDER_ID

# Verify configuration
medai config show
```

---

## Basic Tutorials

### Tutorial 1: Your First Medical Analysis

**Goal**: Analyze a simple set of symptoms and understand the results.

**Steps**:

1. **Choose a Condition**: Let's analyze diabetes symptoms.

```bash
medai analyze "diabetes"
```

You'll be prompted to enter symptoms interactively:
- Type: `increased thirst`
- Type: `frequent urination`
- Type: `unexplained weight loss`
- Press Enter with empty line when done

2. **Enter Patient Context** (optional but recommended):
```
Age: 45
Gender: male
```

3. **Review Results**:

The output will show:
- **Primary Diagnosis**: Type 2 Diabetes Mellitus
- **ICD-10 Code**: E11.9
- **Confidence Score**: e.g., 87%
- **Recommendations**: Diagnostic tests, lifestyle changes
- **Follow-up**: Recommended timeframe

**Understanding the Results**:

- **Confidence Score >90%**: High confidence, can likely proceed with treatment
- **Confidence Score 75-90%**: Moderate confidence, standard review process
- **Confidence Score <75%**: Low confidence, provider review required

### Tutorial 2: Using Command-Line Arguments

**Goal**: Learn to use CLI efficiently without interactive mode.

**Basic Syntax**:
```bash
medai analyze <condition> -s <symptom1> <symptom2> ... [options]
```

**Example 1**: Simple analysis
```bash
medai analyze "hypertension" -s "headache" "dizziness" --age 55
```

**Example 2**: With medical history
```bash
medai analyze "asthma" \
  -s "wheezing" "shortness of breath" "chest tightness" \
  --age 32 \
  --gender female \
  --history "allergies,hay fever"
```

**Example 3**: Emergency case
```bash
medai analyze "chest pain" \
  -s "severe chest pain" "shortness of breath" "sweating" \
  --age 58 \
  --emergency
```

**Example 4**: JSON output for automation
```bash
medai analyze "flu" -s "fever" "cough" "body aches" --output json
```

### Tutorial 3: Verifying Confidence Scores

**Goal**: Understand how to interpret and verify confidence scores.

**Step 1**: Analyze a condition and save the ID
```bash
medai analyze "pneumonia" -s "fever" "cough" "chest pain" --output json > result.json
ANALYSIS_ID=$(cat result.json | jq -r '.id')
```

**Step 2**: Verify confidence score
```bash
medai verify $ANALYSIS_ID
```

**Output Explanation**:
```
Overall Confidence: 85%

Breakdown:
  Diagnosis Confidence: 92% (30% weight)
  Citation Quality: 85% (25% weight)
  Knowledge Base Validation: 88% (20% weight)
  Contradiction Check: 100% (15% weight)
  Provider Alignment: 80% (10% weight)
```

**Step 3**: Get detailed breakdown
```bash
medai verify $ANALYSIS_ID --detailed
```

This shows:
- Individual scoring factors
- Warnings detected
- Citation verification status
- Contradiction details (if any)

### Tutorial 4: Working with Provider Reviews

**Goal**: Submit and manage provider reviews.

**Step 1**: List pending reviews
```bash
medai provider list --status pending
```

**Step 2**: Review a specific analysis
```bash
medai verify ANALYSIS_ID --detailed
```

**Step 3**: Submit approval
```bash
medai provider review ANALYSIS_ID \
  --decision approved \
  --comments "Diagnosis confirmed by lab results"
```

**Step 4**: Submit with modifications
```bash
medai provider review ANALYSIS_ID \
  --decision modified \
  --comments "Updated based on imaging" \
  --modifications '{"additionalTests": ["CT scan"]}'
```

**Step 5**: Reject analysis
```bash
medai provider review ANALYSIS_ID \
  --decision rejected \
  --comments "Diagnosis inconsistent with clinical presentation"
```

---

## Intermediate Tutorials

### Tutorial 5: Using the REST API

**Goal**: Integrate Nova Medicina with your application via REST API.

**Prerequisites**:
```bash
# Start the API server
medai serve --port 3000

# Or use an existing server
export MEDAI_API_URL=http://localhost:3000
export MEDAI_API_KEY=your_api_key_here
```

**Step 1**: Basic API request using cURL

```bash
curl -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "condition": "diabetes",
    "symptoms": ["increased thirst", "frequent urination"],
    "patientContext": {
      "age": 45,
      "gender": "male"
    }
  }'
```

**Step 2**: Using JavaScript/Node.js

```javascript
const axios = require('axios');

async function analyzePatient() {
  try {
    const response = await axios.post('http://localhost:3000/api/analyze', {
      condition: 'diabetes',
      symptoms: ['increased thirst', 'frequent urination'],
      patientContext: {
        age: 45,
        gender: 'male'
      }
    }, {
      headers: {
        'X-API-Key': 'YOUR_API_KEY'
      }
    });

    console.log('Analysis ID:', response.data.id);
    console.log('Diagnosis:', response.data.diagnosis.condition);
    console.log('Confidence:', response.data.confidenceScore.overall);

    return response.data;
  } catch (error) {
    console.error('Analysis failed:', error.message);
  }
}

analyzePatient();
```

**Step 3**: Using Python

```python
import requests

def analyze_patient():
    url = 'http://localhost:3000/api/analyze'
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': 'YOUR_API_KEY'
    }
    data = {
        'condition': 'diabetes',
        'symptoms': ['increased thirst', 'frequent urination'],
        'patientContext': {
            'age': 45,
            'gender': 'male'
        }
    }

    response = requests.post(url, json=data, headers=headers)
    result = response.json()

    print(f"Analysis ID: {result['id']}")
    print(f"Diagnosis: {result['diagnosis']['condition']}")
    print(f"Confidence: {result['confidenceScore']['overall']}")

    return result

analyze_patient()
```

### Tutorial 6: Real-Time Updates with WebSocket

**Goal**: Receive real-time progress updates during analysis.

**Step 1**: Connect to WebSocket server

```javascript
const WebSocket = require('ws');

// Connect to WebSocket
const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  console.log('Connected to Nova Medicina');
});

ws.on('message', (data) => {
  const message = JSON.parse(data);

  switch (message.type) {
    case 'analysis_progress':
      console.log(`Progress: ${message.payload.progress}%`);
      console.log(`Stage: ${message.payload.stage}`);
      break;

    case 'analysis_update':
      console.log(`Update: ${message.payload.message}`);
      if (message.payload.confidence) {
        console.log(`Confidence: ${message.payload.confidence}`);
      }
      break;

    case 'analysis_warning':
      console.warn(`Warning: ${message.payload.message}`);
      console.warn(`Severity: ${message.payload.severity}`);
      break;

    case 'analysis_complete':
      console.log('Analysis complete!');
      console.log(`Status: ${message.payload.status}`);
      console.log(`Confidence: ${message.payload.confidence}`);
      ws.close();
      break;
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});

ws.on('close', () => {
  console.log('Disconnected from Nova Medicina');
});
```

**Step 2**: Submitting analysis with WebSocket tracking

```javascript
async function analyzeWithRealTimeUpdates(query) {
  // Connect WebSocket first
  const ws = new WebSocket('ws://localhost:3000');

  // Wait for connection
  await new Promise(resolve => ws.on('open', resolve));

  // Submit analysis
  const response = await axios.post('http://localhost:3000/api/analyze', query, {
    headers: { 'X-API-Key': 'YOUR_API_KEY' }
  });

  // Subscribe to updates for this analysis
  ws.send(JSON.stringify({
    type: 'subscribe',
    analysisId: response.data.id
  }));

  // Listen for updates (code from Step 1)
  // ...

  return response.data;
}
```

### Tutorial 7: Batch Processing Multiple Patients

**Goal**: Analyze multiple patients efficiently.

**Step 1**: Create patients data file

**patients.json**:
```json
[
  {
    "condition": "diabetes",
    "symptoms": ["increased thirst", "frequent urination"],
    "patientContext": { "age": 45, "gender": "male" }
  },
  {
    "condition": "hypertension",
    "symptoms": ["headache", "dizziness"],
    "patientContext": { "age": 60, "gender": "female" }
  },
  {
    "condition": "asthma",
    "symptoms": ["wheezing", "shortness of breath"],
    "patientContext": { "age": 32, "gender": "male" }
  }
]
```

**Step 2**: Batch process using CLI

```bash
# Sequential processing
cat patients.json | medai analyze --batch

# Parallel processing (5 at a time)
cat patients.json | medai analyze --batch --parallel 5
```

**Step 3**: Batch process using API

```javascript
const fs = require('fs');
const axios = require('axios');

async function batchAnalyze() {
  const patients = JSON.parse(fs.readFileSync('patients.json', 'utf8'));

  // Process all in parallel
  const results = await Promise.all(
    patients.map(patient =>
      axios.post('http://localhost:3000/api/analyze', patient, {
        headers: { 'X-API-Key': 'YOUR_API_KEY' }
      })
      .then(response => ({
        success: true,
        data: response.data
      }))
      .catch(error => ({
        success: false,
        error: error.message
      }))
    )
  );

  // Analyze results
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(`Successful: ${successful.length}/${results.length}`);
  console.log(`Failed: ${failed.length}/${results.length}`);

  // Calculate statistics
  const avgConfidence = successful.reduce((sum, r) =>
    sum + r.data.confidenceScore.overall, 0) / successful.length;

  console.log(`Average Confidence: ${(avgConfidence * 100).toFixed(1)}%`);

  return results;
}

batchAnalyze();
```

### Tutorial 8: Configuring Notification Preferences

**Goal**: Set up provider notification preferences.

**Step 1**: View current preferences

```bash
medai config show provider
```

**Step 2**: Configure notification channels

```bash
# Enable multiple channels
medai config set provider.notificationChannels email,sms,push

# Set quiet hours
medai config set provider.quietHours.enabled true
medai config set provider.quietHours.start "22:00"
medai config set provider.quietHours.end "07:00"
```

**Step 3**: Set confidence thresholds

```bash
# Minimum confidence to trigger notification
medai config set provider.minimumConfidenceThreshold 0.75

# Only notify for urgent cases
medai config set provider.urgentOnly false
```

**Step 4**: Test notification

```bash
# Notify provider about specific analysis
medai provider notify ANALYSIS_ID --provider-id YOUR_PROVIDER_ID

# Urgent notification
medai provider notify ANALYSIS_ID --provider-id YOUR_PROVIDER_ID --urgent
```

---

## Advanced Tutorials

### Tutorial 9: Multi-System Analysis

**Goal**: Analyze complex cases with multiple organ systems involved.

**Scenario**: Patient with cardiovascular, respiratory, and metabolic symptoms.

**Step 1**: Prepare comprehensive patient data

```json
{
  "condition": "Multi-System Disease Complex",
  "symptoms": [
    "chest pain",
    "shortness of breath",
    "irregular heartbeat",
    "persistent cough",
    "wheezing",
    "fever",
    "fatigue",
    "joint pain",
    "confusion"
  ],
  "patientContext": {
    "age": 58,
    "gender": "male",
    "medicalHistory": [
      "hypertension",
      "type 2 diabetes",
      "COPD",
      "previous MI"
    ],
    "medications": [
      "metformin 1000mg BID",
      "lisinopril 20mg daily",
      "atorvastatin 40mg daily"
    ],
    "vitalSigns": {
      "temperature": 38.2,
      "heartRate": 108,
      "bloodPressure": "145/92",
      "oxygenSaturation": 91
    },
    "recentLabs": {
      "glucose": 156,
      "creatinine": 1.4,
      "troponin": 0.02
    }
  },
  "isEmergency": false
}
```

**Step 2**: Submit for analysis

```bash
medai analyze --file complex-case.json --detailed
```

Or via API:
```javascript
const result = await axios.post('http://localhost:3000/api/analyze',
  complexCase, { headers: { 'X-API-Key': 'YOUR_API_KEY' } });
```

**Step 3**: Review differential diagnoses

The system will provide:
- Primary diagnosis with confidence
- 5-10 differential diagnoses ranked by probability
- Risk stratification by organ system
- Prioritized action plan

**Step 4**: Search for similar historical cases

```bash
medai patterns search --symptoms "chest pain,shortness of breath,fever" \
  --age-range "50-65" \
  --min-similarity 0.80
```

### Tutorial 10: Provider Dashboard Integration

**Goal**: Build a provider dashboard that displays pending reviews and analytics.

**Step 1**: Create dashboard backend

```javascript
const express = require('express');
const { ProviderService } = require('nova-medicina');

const app = express();
const providerService = new ProviderService();

// Get dashboard data
app.get('/dashboard/:providerId', async (req, res) => {
  try {
    const { providerId } = req.params;

    // Get dashboard overview
    const dashboard = await providerService.getProviderDashboard(providerId);

    // Get pending reviews
    const pendingReviews = await providerService.getPendingReviews(providerId);

    // Get analytics
    const analytics = await providerService.getProviderAnalytics(providerId, {
      timeframe: 'last_30_days'
    });

    res.json({
      dashboard,
      pendingReviews,
      analytics
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Submit review
app.post('/review', async (req, res) => {
  try {
    const { analysisId, providerId, decision, comments } = req.body;

    const result = await providerService.submitReview({
      analysisId,
      providerId,
      decision,
      comments
    });

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(4000, () => {
  console.log('Provider dashboard running on port 4000');
});
```

**Step 2**: Create frontend dashboard (React example)

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ProviderDashboard({ providerId }) {
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboard();
  }, [providerId]);

  async function loadDashboard() {
    const response = await axios.get(`/dashboard/${providerId}`);
    setDashboard(response.data);
    setLoading(false);
  }

  async function submitReview(analysisId, decision, comments) {
    await axios.post('/review', {
      analysisId,
      providerId,
      decision,
      comments
    });
    loadDashboard(); // Refresh
  }

  if (loading) return <div>Loading...</div>;

  return (
    <div className="provider-dashboard">
      <h1>Provider Dashboard</h1>

      <div className="stats">
        <div className="stat">
          <h3>Pending Reviews</h3>
          <p>{dashboard.dashboard.pendingReviews}</p>
        </div>
        <div className="stat">
          <h3>Completed Today</h3>
          <p>{dashboard.dashboard.completedToday}</p>
        </div>
        <div className="stat">
          <h3>Urgent Cases</h3>
          <p>{dashboard.dashboard.urgentCases}</p>
        </div>
      </div>

      <div className="pending-reviews">
        <h2>Pending Reviews</h2>
        {dashboard.pendingReviews.map(review => (
          <div key={review.analysisId} className="review-card">
            <h3>{review.diagnosis.condition}</h3>
            <p>Confidence: {(review.confidenceScore * 100).toFixed(1)}%</p>
            <p>Priority: {review.priority}</p>
            <button onClick={() => submitReview(review.analysisId, 'approved', '')}>
              Approve
            </button>
            <button onClick={() => submitReview(review.analysisId, 'rejected', '')}>
              Reject
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Tutorial 11: Claude Desktop MCP Integration

**Goal**: Use Nova Medicina directly from Claude Desktop conversations.

**Step 1**: Install and configure MCP server

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nova-medicina": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start"],
      "env": {
        "MEDAI_API_KEY": "your_api_key_here",
        "MEDAI_PROVIDER_ID": "your_provider_id"
      }
    }
  }
}
```

**Step 2**: Restart Claude Desktop

**Step 3**: Use Nova Medicina tools in conversations

Example conversation:

> **User**: Analyze a patient with symptoms: increased thirst, frequent urination, unexplained weight loss. Patient is 45 years old, male.

Claude will automatically invoke `medai_analyze` tool and provide results.

> **User**: What's the confidence score for that analysis?

Claude will invoke `medai_verify` with the analysis ID.

> **User**: Approve the analysis

Claude will invoke `medai_review` to submit approval.

**Step 4**: Advanced MCP usage

```
# Search for similar cases
User: Find similar cases with fever, cough, and chest pain

# Get system metrics
User: Show Nova Medicina system performance metrics

# Notify provider
User: Notify the cardiology team about the last analysis
```

### Tutorial 12: Continuous Learning and Improvement

**Goal**: Implement continuous learning from provider feedback.

**Step 1**: Set up learning pipeline

```javascript
const { AgentDBLearningService } = require('nova-medicina');

const learningService = new AgentDBLearningService();

// After provider approval
async function handleApproval(analysisId, feedback) {
  await learningService.reinforcePattern({
    analysisId,
    outcome: 'success',
    confidence: feedback.confidence,
    providerComments: feedback.comments
  });
}

// After provider modification
async function handleModification(analysisId, modifications) {
  await learningService.learnModification({
    analysisId,
    modifications,
    reason: modifications.reason
  });
}

// After provider rejection
async function handleRejection(analysisId, correction) {
  await learningService.learnFromError({
    analysisId,
    incorrectDiagnosis: correction.original,
    correctDiagnosis: correction.correct,
    criticalFactors: correction.factors,
    severity: 'high'
  });
}
```

**Step 2**: Track learning metrics

```javascript
async function getLearningMetrics() {
  const metrics = await learningService.getMetrics();

  console.log('Learning Performance:');
  console.log('  Patterns learned:', metrics.patternsLearned);
  console.log('  Pattern match accuracy:', metrics.patternAccuracy);
  console.log('  Confidence improvement:', metrics.confidenceImprovement);
  console.log('  Provider agreement rate:', metrics.providerAgreementRate);

  return metrics;
}
```

**Step 3**: Implement feedback loop

```javascript
// Weekly learning review
setInterval(async () => {
  const weeklyMetrics = await learningService.getMetrics({
    timeframe: 'last_7_days'
  });

  // Identify improvement areas
  const improvementAreas = await learningService.identifyImprovementAreas();

  // Update confidence thresholds
  if (weeklyMetrics.providerAgreementRate > 0.90) {
    // Increase auto-approval threshold
    await updateConfig('antiHallucination.autoApproveThreshold', 0.92);
  }

  // Retrain patterns
  await learningService.retrainPatterns(improvementAreas);

}, 7 * 24 * 60 * 60 * 1000); // Weekly
```

---

## Common Use Cases

### Use Case 1: Emergency Room Triage

**Scenario**: Quick assessment of emergency patients for triage prioritization.

**Implementation**:

```bash
# Quick triage
medai analyze "trauma" \
  -s "severe bleeding" "altered consciousness" "rapid heart rate" \
  --age 35 \
  --emergency \
  --auto-notify
```

**Workflow**:
1. Patient arrives at ER
2. Triage nurse enters symptoms
3. System provides instant preliminary assessment
4. Confidence score determines urgency
5. If critical, immediately notifies ER physician
6. Results guide initial treatment

### Use Case 2: Telemedicine Consultation

**Scenario**: Remote patient consultation with AI-assisted diagnosis.

**Implementation**:

```javascript
// Telemedicine platform integration
async function conductTelemedicineConsultation(patientData) {
  // Analyze symptoms
  const analysis = await analyzeSymptoms(patientData);

  if (analysis.confidenceScore.overall >= 0.85) {
    // High confidence - provide self-care recommendations
    return {
      type: 'self-care',
      recommendations: analysis.recommendations,
      followUp: analysis.followUp
    };
  } else {
    // Low confidence - schedule video consultation
    await scheduleVideoConsult(patientData.patientId);
    await notifyProvider(analysis.id, { urgent: true });

    return {
      type: 'video-consult-required',
      reason: 'Further evaluation needed',
      analysis: analysis
    };
  }
}
```

### Use Case 3: Clinical Decision Support

**Scenario**: Assist physicians with differential diagnosis during patient encounters.

**Implementation**:

```javascript
// Clinical decision support system
async function clinicalDecisionSupport(symptoms, patientContext) {
  // Get AI analysis
  const analysis = await medicalService.analyzeCondition({
    condition: 'differential diagnosis',
    symptoms,
    patientContext
  });

  // Search similar cases
  const similarCases = await learningService.findSimilarCases({
    symptoms,
    patientAge: patientContext.age
  });

  // Get evidence-based recommendations
  const guidelines = await getRelevantGuidelines(
    analysis.diagnosis.condition
  );

  return {
    primaryDiagnosis: analysis.diagnosis,
    differentials: analysis.differentialDiagnoses,
    confidence: analysis.confidenceScore,
    similarCases,
    guidelines,
    recommendedTests: analysis.requiredTests,
    warnings: analysis.warnings
  };
}
```

### Use Case 4: Quality Assurance

**Scenario**: Review and validate diagnoses for quality assurance.

**Implementation**:

```javascript
// Quality assurance workflow
async function qualityAssuranceReview(analysisId) {
  // Get analysis
  const analysis = await getAnalysis(analysisId);

  // Verify confidence scores
  const verification = await verifyConfidence(analysisId);

  // Check against guidelines
  const guidelineCompliance = await checkGuidelines(
    analysis.diagnosis,
    analysis.recommendations
  );

  // Review citations
  const citationValidity = await verifyCitations(analysis.citations);

  // Generate QA report
  return {
    analysisId,
    overallScore: verification.confidenceScore.overall,
    guidelineCompliance,
    citationValidity,
    recommendations: generateQARecommendations(
      verification,
      guidelineCompliance,
      citationValidity
    )
  };
}
```

### Use Case 5: Medical Education

**Scenario**: Training medical students with real-world case studies.

**Implementation**:

```javascript
// Educational case studies
async function generateEducationalCase(topic) {
  // Create case scenario
  const caseStudy = await createCaseStudy(topic);

  // Get AI analysis (hidden from students initially)
  const expertAnalysis = await analyzeCondition(caseStudy);

  // Find similar real cases
  const realCases = await findSimilarCases(caseStudy.symptoms);

  // Create learning objectives
  return {
    case: caseStudy,
    learningObjectives: generateObjectives(topic),
    studentTask: 'Diagnose and create treatment plan',
    expertAnalysis, // Revealed after student submission
    realCases,
    guidelines: getRelevantGuidelines(topic)
  };
}
```

---

## Troubleshooting Guide

### Problem 1: Low Confidence Scores

**Symptoms**: Consistently receiving low confidence scores (<70%)

**Possible Causes**:
- Insufficient symptom information
- Vague or contradictory symptoms
- Missing patient context
- Rare or complex condition

**Solutions**:

1. **Provide more detailed symptoms**:
```bash
# Instead of:
medai analyze "disease" -s "pain"

# Use:
medai analyze "disease" -s "sharp pain in lower right abdomen" "nausea" "fever" --age 25
```

2. **Add patient context**:
```bash
medai analyze "condition" -s "symptom1" "symptom2" \
  --age 45 \
  --gender female \
  --history "diabetes,hypertension" \
  --medications "metformin,lisinopril"
```

3. **Use emergency flag for urgent cases**:
```bash
medai analyze "chest pain" -s "crushing chest pain" "shortness of breath" --emergency
```

4. **Request provider review**:
```bash
medai provider notify ANALYSIS_ID --urgent
```

### Problem 2: API Connection Errors

**Symptoms**: "Connection refused" or "Timeout" errors

**Solutions**:

1. **Check if API server is running**:
```bash
# Check health
curl http://localhost:3000/api/health

# If not running, start server
medai serve --port 3000
```

2. **Verify configuration**:
```bash
medai config show api
```

3. **Check network connectivity**:
```bash
ping localhost
telnet localhost 3000
```

4. **Review logs**:
```bash
tail -f ~/.medai/logs/api-server.log
```

### Problem 3: Authentication Failures

**Symptoms**: "Invalid API key" or "401 Unauthorized"

**Solutions**:

1. **Verify API key**:
```bash
medai config show api.key
```

2. **Regenerate API key**:
```bash
medai config set api.key $(medai auth generate-key)
```

3. **Check environment variables**:
```bash
echo $MEDAI_API_KEY
```

4. **Update configuration**:
```bash
export MEDAI_API_KEY=your_new_key
medai config set api.key $MEDAI_API_KEY
```

### Problem 4: MCP Server Not Starting

**Symptoms**: Claude Desktop shows "nova-medicina unavailable"

**Solutions**:

1. **Check Node.js version**:
```bash
node --version  # Should be 18+
```

2. **Test MCP server manually**:
```bash
npx nova-medicina mcp start --debug
```

3. **Review Claude Desktop configuration**:
Check `~/Library/Application Support/Claude/claude_desktop_config.json`

4. **Check logs**:
```bash
tail -f ~/Library/Logs/Claude/mcp-server-nova-medicina.log
```

5. **Restart Claude Desktop** after configuration changes

### Problem 5: Slow Performance

**Symptoms**: Analysis takes too long (>30 seconds)

**Solutions**:

1. **Increase timeout**:
```bash
medai config set api.timeout 60000
```

2. **Check system resources**:
```bash
# CPU usage
top

# Memory usage
free -h

# Disk space
df -h
```

3. **Enable caching**:
```bash
medai config set cache.enabled true
medai config set cache.ttl 3600
```

4. **Use parallel processing for batch**:
```bash
medai analyze --batch --parallel 5 < patients.json
```

---

## Best Practices

### 1. Data Input Best Practices

**DO**:
- ✓ Provide specific, detailed symptoms
- ✓ Include relevant patient context (age, gender, history)
- ✓ Use medical terminology when possible
- ✓ Specify symptom onset and progression
- ✓ Include vital signs and lab results when available

**DON'T**:
- ✗ Use vague descriptions ("feeling unwell")
- ✗ Omit patient age and gender
- ✗ Ignore medication history
- ✗ Skip emergency flag for critical cases
- ✗ Mix multiple conditions in one query

### 2. Confidence Score Interpretation

**Confidence Ranges**:
- **90-100%**: High confidence - Consider for auto-approval
- **75-89%**: Moderate confidence - Standard review process
- **50-74%**: Low confidence - Provider review required
- **<50%**: Very low confidence - Urgent provider review

**Actions by Confidence**:
```javascript
if (confidence >= 0.90) {
  // High confidence - proceed with treatment
  action = 'auto-approve';
} else if (confidence >= 0.75) {
  // Moderate - standard review
  action = 'schedule-review';
} else if (confidence >= 0.50) {
  // Low - provider review needed
  action = 'notify-provider';
} else {
  // Very low - urgent review
  action = 'urgent-escalation';
}
```

### 3. Security Best Practices

**API Keys**:
- Never commit API keys to version control
- Rotate API keys regularly (monthly recommended)
- Use environment variables for keys
- Restrict API key permissions

**Data Protection**:
- Enable HTTPS for production
- Encrypt patient data at rest
- Implement access logging
- Follow HIPAA compliance guidelines

**Example secure configuration**:
```bash
# Use environment variables
export MEDAI_API_KEY="$(cat ~/.medai/api-key)"
export MEDAI_ENCRYPTION_KEY="$(cat ~/.medai/encryption-key)"

# Set restrictive permissions
chmod 600 ~/.medai/api-key
chmod 600 ~/.medai/encryption-key
```

### 4. Performance Optimization

**Caching**:
```bash
# Enable caching for repeated queries
medai config set cache.enabled true
medai config set cache.ttl 3600  # 1 hour
```

**Batch Processing**:
```bash
# Use parallel processing for multiple patients
medai analyze --batch --parallel 10 < patients.json
```

**Database Optimization**:
```bash
# Optimize AgentDB
medai db optimize

# Clear old analyses
medai db cleanup --older-than 90days
```

### 5. Provider Workflow Best Practices

**Review Process**:
1. Always verify confidence scores before approval
2. Check differential diagnoses
3. Review citations and evidence
4. Consider patient context
5. Document reasoning in comments

**Notification Management**:
- Set appropriate quiet hours
- Configure priority levels correctly
- Use urgent notifications sparingly
- Review pending cases daily

**Learning Feedback**:
- Provide detailed comments in reviews
- Document modifications clearly
- Explain rejections thoroughly
- Contribute to pattern improvement

### 6. Integration Best Practices

**Error Handling**:
```javascript
try {
  const analysis = await medaiClient.analyze(query);
  return analysis;
} catch (error) {
  if (error.status === 401) {
    // Authentication error - refresh token
    await refreshAuthToken();
    return medaiClient.analyze(query);
  } else if (error.status === 429) {
    // Rate limit - wait and retry
    await delay(5000);
    return medaiClient.analyze(query);
  } else {
    // Log and handle error
    logger.error('Analysis failed', error);
    throw error;
  }
}
```

**Logging**:
```javascript
// Comprehensive logging
logger.info('Analysis started', {
  analysisId: analysis.id,
  condition: query.condition,
  symptomCount: query.symptoms.length
});

logger.info('Analysis completed', {
  analysisId: analysis.id,
  diagnosis: analysis.diagnosis.condition,
  confidence: analysis.confidenceScore.overall,
  requiresReview: analysis.requiresProviderReview
});
```

**Testing**:
- Write unit tests for integrations
- Test error scenarios
- Validate data before submission
- Monitor API response times

---

## Additional Resources

### Documentation
- [API Reference](./API.md)
- [MCP Integration Guide](../examples/mcp-integration.md)
- [CLI Reference](./CLI.md)

### Examples
- [Basic Usage Examples](../examples/basic-usage.js)
- [API Client Examples](../examples/api-client.js)
- [Provider Integration Examples](../examples/provider-integration.js)
- [Advanced Workflows](../examples/advanced-workflows.js)

### Support
- GitHub Issues: https://github.com/nova-medicina/support
- Documentation: https://docs.nova-medicina.com
- Community Forum: https://community.nova-medicina.com

---

**Note**: Nova Medicina is designed to assist healthcare professionals. Always consult with qualified medical practitioners for diagnosis and treatment decisions. This system is a clinical decision support tool, not a replacement for professional medical judgment.
