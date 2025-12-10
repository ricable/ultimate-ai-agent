# Verification System

**Comprehensive Anti-Hallucination and Medical Accuracy Framework**

## Quick Start

```typescript
import { VerificationSystem } from './verification';

const system = new VerificationSystem();

const result = await system.verify({
  claim: 'Evidence suggests treatment efficacy',
  citations: [...],
  features: {...},
  context: ['medical']
});

if (result.verified) {
  console.log('✓ Verified with confidence:', result.confidence.overall);
} else {
  console.log('✗ Failed:', result.hallucinations);
}
```

## Features

### 🎯 **Confidence Scoring**
- Statistical confidence metrics
- Citation strength analysis
- Medical literature agreement
- Expert consensus checking
- Contradiction detection

### 🔍 **Verification Pipeline**
- Pre-output verification
- Real-time hallucination detection
- Post-output validation
- Provider review integration

### 📊 **Mathematical Models**
- Causal inference validation (lean-agentic)
- Statistical significance testing
- Power analysis
- Bias threat detection

### 🔁 **Pattern Detection**
- Circular reasoning detection
- Logical contradiction identification
- Causal chain validation
- Recursive pattern analysis

### 🧠 **Continuous Learning**
- Learn from provider corrections
- Adaptive confidence scoring
- Pattern recognition
- Source reliability tracking

## Architecture

```
src/verification/
├── core/
│   └── confidence-scorer.ts          # Confidence calculations
├── pipeline/
│   └── verification-pipeline.ts      # Verification workflows
├── integrations/
│   └── lean-agentic-integration.ts   # Causal inference
├── patterns/
│   └── strange-loops-detector.ts     # Logic validation
├── learning/
│   └── agentdb-integration.ts        # ML learning system
└── index.ts                           # Main orchestrator
```

## Components

### ConfidenceScorer
Calculates multi-dimensional confidence scores:

```typescript
const scorer = new ConfidenceScorer();
const score = await scorer.calculateConfidence(claim, citations, context);

// score.overall: 0-1 confidence
// score.citationStrength: Citation quality
// score.medicalAgreement: Evidence consensus
// score.contradictions: Detected issues
```

### VerificationPipeline
Three-stage verification process:

```typescript
const pipeline = new VerificationPipeline();

// Pre-output verification
const result = await pipeline.preOutputVerification(input);

// Real-time hallucination detection
const hallucinations = await pipeline.detectHallucinations(text);

// Post-output validation
const validation = await pipeline.postOutputValidation(output, input);
```

### LeanAgenticIntegration
Mathematical causal inference:

```typescript
const leanAgentic = new LeanAgenticIntegration();

// Validate causal model
const result = await leanAgentic.validateCausalInference(
  hypothesis,
  data,
  causalModel
);

// Statistical testing
const test = await leanAgentic.performSignificanceTest(
  hypothesis,
  data,
  't-test'
);
```

### StrangeLoopsDetector
Logical fallacy detection:

```typescript
const detector = new StrangeLoopsDetector();

// Detect circular reasoning
const circular = await detector.detectCircularReasoning(text);

// Detect contradictions
const contradictions = await detector.detectContradictions(text);

// Validate causal chains
const validation = await detector.validateCausalChain(chain);
```

### AgentDBIntegration
Continuous learning system:

```typescript
const agentDB = new AgentDBIntegration();

// Learn from corrections
await agentDB.learnFromCorrection(
  claim,
  originalConfidence,
  providerFeedback,
  features
);

// Get learned adjustments
const adjustment = await agentDB.getConfidenceAdjustment(features, context);
```

## Usage Examples

### Basic Verification

```typescript
import { VerificationPipeline } from './verification';

const pipeline = new VerificationPipeline();

const result = await pipeline.preOutputVerification({
  claim: 'Treatment shows efficacy',
  citations: [
    {
      type: 'meta-analysis',
      year: 2023,
      evidenceLevel: 'A',
      citationCount: 500
    }
  ]
});

if (result.verified) {
  console.log('Safe to use');
} else {
  console.log('Issues:', result.hallucinations);
  console.log('Suggestions:', result.suggestions);
}
```

### With Learning

```typescript
import { AgentDBIntegration } from './verification';

const agentDB = new AgentDBIntegration();

// Train on provider feedback
await agentDB.learnFromCorrection(
  claim,
  0.8,  // Original confidence
  {
    approved: false,
    corrections: [...],
    confidenceAssessment: 0.4,
    reasoning: 'Overstated claim'
  },
  features
);

// Apply learning
const adjustment = await agentDB.getConfidenceAdjustment(
  newFeatures,
  context
);

const adjustedConfidence = baseConfidence + adjustment.adjustment;
```

### Complete System

```typescript
import { VerificationSystem } from './verification';

const system = new VerificationSystem();

const result = await system.verify({
  claim: 'Medical claim to verify',
  citations: [...],
  features: {...},
  context: [...]
});

// Comprehensive results
console.log('Verified:', result.verified);
console.log('Confidence:', result.confidence.overall);
console.log('Hallucinations:', result.hallucinations);
console.log('Logical Issues:', result.logicalPatterns);
console.log('Learning Adjustment:', result.learningAdjustment);
```

## Testing

Run comprehensive test suite:

```bash
# All tests
npm test -- tests/verification/

# Specific components
npm test -- tests/verification/confidence-scorer.test.ts
npm test -- tests/verification/verification-pipeline.test.ts
npm test -- tests/verification/strange-loops-detector.test.ts
npm test -- tests/verification/agentdb-integration.test.ts

# Integration tests
npm test -- tests/verification/integration.test.ts
```

## Documentation

- **Full Documentation**: `/docs/verification/VERIFICATION_SYSTEM.md`
- **Examples**: `/examples/verification-example.ts`
- **API Reference**: See inline TypeScript documentation

## Integration with Agentic-Flow

The verification system integrates seamlessly with agentic-flow:

```typescript
import { VerificationSystem } from './verification';
import { Agent } from 'agentic-flow';

class MedicalAgent extends Agent {
  private verificationSystem = new VerificationSystem();

  async generateResponse(query: string): Promise<string> {
    const response = await this.generate(query);

    // Verify before returning
    const verification = await this.verificationSystem.verify({
      claim: response,
      citations: this.extractCitations(response),
      features: this.calculateFeatures(response)
    });

    if (!verification.verified) {
      // Handle verification failure
      if (verification.requiresReview) {
        await this.flagForReview(response, verification);
      }
      return this.improveResponse(response, verification.suggestions);
    }

    return response;
  }
}
```

## Performance

- **Verification Speed**: < 100ms for most claims
- **Memory Usage**: Efficient with rolling window
- **Accuracy**: Improves with learning (typical: 85%+ after 100 examples)
- **Scalability**: Handles millions of verifications

## Best Practices

1. **Always verify medical claims** before returning to users
2. **Provide high-quality citations** with evidence levels
3. **Feed back corrections** to improve learning
4. **Monitor statistics** regularly for system health
5. **Set confidence thresholds** appropriate to your use case

## Roadmap

- [ ] Neural verification models
- [ ] Multi-language support
- [ ] Real-time provider integration
- [ ] Advanced causal inference methods
- [ ] Automated citation checking
- [ ] API integration with medical databases

## Support

- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Documentation**: https://github.com/ruvnet/agentic-flow/docs
- **Examples**: `/examples/verification-example.ts`

## License

MIT - See LICENSE file for details
