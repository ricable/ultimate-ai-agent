# Medical AI Analysis System - Backend

A comprehensive medical analysis system with CLI and REST API interfaces, featuring anti-hallucination detection, provider review workflows, and AgentDB-powered pattern learning.

## Features

### 🛡️ Anti-Hallucination System
- **Confidence Scoring**: Multi-factor confidence assessment for all analyses
- **Citation Verification**: Automatic verification of medical literature citations
- **Knowledge Base Validation**: Cross-reference with trusted medical sources
- **Contradiction Detection**: Identify contradictory statements and recommendations
- **Provider Approval**: Automatic routing to healthcare providers for low-confidence results

### 🏥 CLI Interface
```bash
# Analyze medical condition
medai analyze "diabetes" -s "increased thirst" "frequent urination" --age 45

# Interactive mode
medai analyze --interactive

# Verify analysis confidence
medai verify <analysisId> --detailed

# Provider review
medai provider review <analysisId> --decision approved --comments "Diagnosis confirmed"

# Notify provider
medai provider notify <analysisId> --urgent

# Configuration
medai config show
medai config set minimumConfidence 0.75
```

### 🌐 REST API

#### Endpoints

**POST /api/analyze** - Submit medical query
```json
{
  "condition": "Type 2 Diabetes",
  "symptoms": ["increased thirst", "frequent urination"],
  "patientContext": {
    "age": 45,
    "gender": "male",
    "medicalHistory": ["hypertension"],
    "currentMedications": ["lisinopril"]
  },
  "options": {
    "requireHighConfidence": true,
    "includeEmergencyCheck": true
  }
}
```

**GET /api/analysis/:id** - Get analysis results

**POST /api/provider/review** - Submit provider review
```json
{
  "analysisId": "123e4567-e89b-12d3-a456-426614174000",
  "decision": "approved",
  "comments": "Diagnosis confirmed through lab results"
}
```

**POST /api/provider/notify** - Notify provider
```json
{
  "analysisId": "123e4567-e89b-12d3-a456-426614174000",
  "urgent": true
}
```

**GET /api/metrics** - Get learning metrics

### 📡 WebSocket Support
Real-time updates for analysis progress:
```javascript
const ws = new WebSocket('ws://localhost:3000');

ws.on('message', (data) => {
  const message = JSON.parse(data);
  if (message.type === 'analysis_update') {
    console.log(`Progress: ${message.payload.progress}%`);
    console.log(`Status: ${message.payload.status}`);
  }
});
```

### 🧠 AgentDB Integration
- **Pattern Recognition**: Learn from successful analyses
- **Skill Library**: Build reusable diagnostic skills
- **Reflexion Learning**: Improve from provider feedback
- **Embedding Search**: Find similar historical cases

## Architecture

```
src/
├── cli/                    # CLI interface
│   └── index.ts           # Command definitions
├── api/                    # REST API
│   └── index.ts           # Express server & WebSocket
├── services/              # Business logic
│   ├── medical-analysis.service.ts
│   ├── anti-hallucination.service.ts
│   ├── agentdb-learning.service.ts
│   └── provider.service.ts
├── middleware/            # Express middleware
│   ├── auth.middleware.ts
│   └── logging.middleware.ts
├── types/                 # TypeScript types
│   └── medical.types.ts
└── utils/                 # Utilities
    └── config.ts
```

## Installation

```bash
# Install dependencies
npm install

# Build
npm run build

# Run API server
npm start

# Run CLI
npm run cli -- analyze "condition" -s "symptom1" "symptom2"
```

## Configuration

### Environment Variables
```bash
PORT=3000
AGENTDB_PATH=./data/medical-learning.db
MIN_CONFIDENCE=0.70
NODE_ENV=production
ALLOWED_ORIGINS=http://localhost:3000,https://app.example.com
```

### Configuration File
```json
{
  "antiHallucination": {
    "minimumConfidence": 0.70,
    "requireProviderReviewThreshold": 0.75,
    "autoApproveThreshold": 0.90,
    "enableContradictionDetection": true,
    "enableCitationVerification": true
  },
  "providers": {
    "notificationEnabled": true,
    "autoAssignProvider": true,
    "requiredForHighRisk": true
  },
  "api": {
    "port": 3000,
    "enableWebSocket": true,
    "rateLimitPerMinute": 100
  },
  "learning": {
    "enablePatternLearning": true,
    "agentDbPath": "./data/medical-learning.db",
    "minimumPatternFrequency": 3
  }
}
```

## Anti-Hallucination Pipeline

1. **Analysis Generation** - AI generates medical analysis
2. **Confidence Scoring** - Multi-factor confidence calculation
   - Diagnosis confidence (0.3 weight)
   - Citation verification (0.25 weight)
   - Knowledge base validation (0.2 weight)
   - Contradiction detection (0.15 weight)
   - Provider alignment (0.1 weight)
3. **Warning Generation** - Identify potential issues
   - Low confidence
   - Hallucination detection
   - Contradictions
   - Emergency situations
4. **Provider Review** - Route to provider if needed
5. **Learning Update** - Update AgentDB with outcomes

## Security

- **Authentication**: API key and Bearer token support
- **Rate Limiting**: 100 requests/minute per IP
- **CORS**: Configurable origin whitelist
- **Helmet**: Security headers
- **Input Validation**: Request validation
- **Logging**: Comprehensive request/response logging

## Provider Workflow

1. **Analysis Triggers Review** - Low confidence or high-risk
2. **Provider Notification** - Email, SMS, or push notification
3. **Provider Reviews** - Approve, reject, or modify
4. **Learning Update** - System learns from provider decisions
5. **Pattern Recognition** - Apply learnings to future cases

## Development

```bash
# Run in development mode
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint

# Tests
npm run test
```

## Integration Examples

### Express App Integration
```typescript
import { MedicalAnalysisAPI } from './src/api';

const api = new MedicalAnalysisAPI(3000);
api.start();
```

### CLI in Node.js
```typescript
import { exec } from 'child_process';

exec('medai analyze "diabetes" -s "thirst" --output json', (error, stdout) => {
  const result = JSON.parse(stdout);
  console.log(result);
});
```

### WebSocket Client
```typescript
import WebSocket from 'ws';

const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  ws.send(JSON.stringify({ type: 'subscribe', analysisId: '123' }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data.toString());
  console.log('Update:', message);
});
```

## License

MIT

## Contributors

Built with the SPARC methodology and AgentDB integration.
