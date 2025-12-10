# Medical Analysis System - Quick Start Guide

## 🚀 Fast Track Setup

### Prerequisites
- Node.js 18+
- npm or yarn
- Git

### Installation

```bash
# Navigate to project directory
cd /home/user/agentic-flow/src

# Install dependencies
npm install

# Build TypeScript
npm run build
```

## 🏥 Using the CLI

### Quick Analysis
```bash
# Interactive mode (guided prompts)
npm run cli -- analyze --interactive

# Direct analysis
npm run cli -- analyze "diabetes" \
  -s "increased thirst" "frequent urination" \
  --age 45 \
  --gender male

# With full context
npm run cli -- analyze "hypertension" \
  -s "headache" "dizziness" "chest pain" \
  --age 55 \
  --history "diabetes" "high cholesterol" \
  --medications "metformin" "atorvastatin" \
  --emergency-check
```

### Verify Analysis
```bash
# Get confidence breakdown
npm run cli -- verify <analysisId> --detailed
```

### Provider Commands
```bash
# Submit review
npm run cli -- provider review <analysisId> \
  --decision approved \
  --comments "Diagnosis confirmed with lab results"

# Notify provider
npm run cli -- provider notify <analysisId> --urgent

# List pending reviews
npm run cli -- provider list
```

### Configuration
```bash
# Show current config
npm run cli -- config show

# Update settings
npm run cli -- config set minimumConfidence 0.80
```

## 🌐 Using the REST API

### Start the Server
```bash
# Production mode
npm start

# Development mode (with auto-reload)
npm run dev
```

### API Endpoints

#### Submit Analysis
```bash
curl -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: medai_dev_key_12345678" \
  -d '{
    "condition": "Type 2 Diabetes",
    "symptoms": [
      "increased thirst",
      "frequent urination",
      "unexplained weight loss"
    ],
    "patientContext": {
      "age": 45,
      "gender": "male",
      "medicalHistory": ["hypertension"],
      "currentMedications": ["lisinopril"]
    },
    "options": {
      "requireHighConfidence": true,
      "includeEmergencyCheck": true,
      "includeDifferentials": true
    }
  }'
```

#### Get Analysis
```bash
curl http://localhost:3000/api/analysis/<analysisId> \
  -H "X-API-Key: medai_dev_key_12345678"
```

#### Provider Review
```bash
curl -X POST http://localhost:3000/api/provider/review \
  -H "Content-Type: application/json" \
  -H "X-API-Key: medai_dev_key_12345678" \
  -d '{
    "analysisId": "<analysisId>",
    "decision": "approved",
    "comments": "Diagnosis confirmed through diagnostic tests"
  }'
```

#### Health Check
```bash
curl http://localhost:3000/api/health
```

## 📡 WebSocket Connection

### JavaScript/Node.js
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  console.log('Connected to MedAI Analysis System');
});

ws.on('message', (data) => {
  const message = JSON.parse(data);

  switch(message.type) {
    case 'analysis_update':
      console.log(`Progress: ${message.payload.progress}%`);
      console.log(`Status: ${message.payload.status}`);
      break;

    case 'warning':
      console.warn(`Warning: ${message.payload.message}`);
      break;

    case 'confidence_update':
      console.log(`Confidence: ${message.payload.score}`);
      break;
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

### Browser
```javascript
const ws = new WebSocket('ws://localhost:3000');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Update:', message);
};
```

## 🔧 Configuration

### Environment Variables
```bash
# Create .env file
cat > /home/user/agentic-flow/src/.env << EOF
PORT=3000
NODE_ENV=development
AGENTDB_PATH=./data/medical-learning.db
MIN_CONFIDENCE=0.70
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
EOF
```

### Configuration File
```bash
# Create config.json
cat > /home/user/agentic-flow/src/config.json << EOF
{
  "antiHallucination": {
    "minimumConfidence": 0.70,
    "requireProviderReviewThreshold": 0.75,
    "autoApproveThreshold": 0.90
  },
  "api": {
    "port": 3000,
    "enableWebSocket": true,
    "rateLimitPerMinute": 100
  },
  "learning": {
    "enablePatternLearning": true,
    "agentDbPath": "./data/medical-learning.db"
  }
}
EOF
```

## 📊 Example Workflows

### Basic Analysis Workflow
```bash
# 1. Analyze condition
ANALYSIS_ID=$(npm run cli -- analyze "diabetes" \
  -s "thirst" "urination" \
  --output json | jq -r '.data.id')

# 2. Verify confidence
npm run cli -- verify $ANALYSIS_ID --detailed

# 3. If low confidence, notify provider
npm run cli -- provider notify $ANALYSIS_ID

# 4. Provider submits review
npm run cli -- provider review $ANALYSIS_ID \
  --decision approved \
  --comments "Confirmed with HbA1c test"
```

### API Workflow
```bash
# 1. Start server
npm start &
SERVER_PID=$!

# 2. Wait for server
sleep 3

# 3. Submit analysis
RESPONSE=$(curl -s -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: medai_dev_key" \
  -d '{"condition":"diabetes","symptoms":["thirst"]}')

# 4. Extract analysis ID
ANALYSIS_ID=$(echo $RESPONSE | jq -r '.data.id')

# 5. Get results
curl http://localhost:3000/api/analysis/$ANALYSIS_ID \
  -H "X-API-Key: medai_dev_key" | jq .

# 6. Cleanup
kill $SERVER_PID
```

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Manual Testing
```bash
# Test CLI
./scripts/start-medical-system.sh

# Test API
curl http://localhost:3000/api/health

# Test WebSocket
wscat -c ws://localhost:3000
```

## 🔍 Monitoring

### View Logs
```bash
# API server logs
npm start 2>&1 | tee logs/api.log

# CLI operation logs
npm run cli -- analyze "condition" -s "symptom" 2>&1 | tee logs/cli.log
```

### Check Metrics
```bash
curl http://localhost:3000/api/metrics \
  -H "X-API-Key: medai_dev_key" | jq .
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>

# Or use different port
PORT=3001 npm start
```

### Database Issues
```bash
# Reset AgentDB
rm -rf ./data/medical-learning.db
mkdir -p ./data
```

### Dependencies Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 🎓 Learning Resources

- **Type Definitions**: `/home/user/agentic-flow/src/types/medical.types.ts`
- **API Documentation**: `/home/user/agentic-flow/src/README.md`
- **Implementation Report**: `/home/user/agentic-flow/docs/medical-analysis-backend-implementation.md`

## 🔐 Security Notes

### Development
- Default API key: `medai_dev_key_12345678`
- No authentication required in dev mode

### Production
1. Set `NODE_ENV=production`
2. Generate secure API keys
3. Enable HTTPS
4. Configure CORS whitelist
5. Use environment variables for secrets
6. Enable rate limiting
7. Implement proper JWT authentication

## 📞 Support

For issues or questions:
1. Check implementation report in `/docs/medical-analysis-backend-implementation.md`
2. Review type definitions for API contracts
3. Check logs for error messages
4. Verify configuration settings

## 🚀 Next Steps

1. **Customize Configuration**
   - Adjust confidence thresholds
   - Configure notification channels
   - Set up provider preferences

2. **Integrate with Frontend**
   - Use REST API endpoints
   - Connect WebSocket for real-time updates
   - Display confidence scores

3. **Deploy to Production**
   - Containerize with Docker
   - Set up CI/CD pipeline
   - Configure production database
   - Enable monitoring and alerting

4. **Enhance Features**
   - Add more medical knowledge bases
   - Integrate with EHR systems
   - Implement FHIR compliance
   - Add multi-language support
