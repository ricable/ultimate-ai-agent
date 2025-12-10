# Nova Medicina API Documentation

> Complete API reference for Nova Medicina

## Table of Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Core Classes](#core-classes)
- [CLI Commands](#cli-commands)
- [MCP Integration](#mcp-integration)

## Installation

```bash
npm install nova-medicina
```

## Authentication

Configure API keys via environment variables or config file:

```bash
export NOVA_MEDICINA_OPENAI_KEY="sk-..."
export NOVA_MEDICINA_ANTHROPIC_KEY="sk-ant-..."
```

Or use the config command:

```bash
nova-medicina config --api-key "your-key-here"
```

## Core Classes

### Analyzer

Symptom analysis with multi-model consensus.

#### Constructor

```javascript
import { Analyzer } from 'nova-medicina';

const analyzer = new Analyzer({
  minConfidenceScore: 0.95,
  verificationLevel: 'moderate' // 'strict' | 'moderate' | 'basic'
});
```

#### Methods

**analyze(options)**

Analyze symptoms with anti-hallucination verification.

```javascript
const result = await analyzer.analyze({
  symptoms: 'fever, cough, fatigue',
  duration: '3 days',
  age: 35,
  gender: 'female',
  medicalHistory: ['asthma']
});
```

Returns:
```javascript
{
  symptoms: string,
  confidence: number,
  urgency: 'emergency' | 'urgent' | 'routine' | 'self-care',
  recommendations: string[],
  citations: Array<Citation>,
  disclaimer: string
}
```

**assessUrgency(analysis)**

Assess urgency level of symptoms.

```javascript
const urgency = analyzer.assessUrgency(analysis);
// Returns: 'emergency' | 'urgent' | 'routine' | 'self-care'
```

**verify(analysis)**

Verify analysis against medical databases.

```javascript
const verification = await analyzer.verify(analysis);
```

Returns:
```javascript
{
  verified: boolean,
  confidence: number,
  sources: string[]
}
```

### Verifier

Medical information verification system.

#### Constructor

```javascript
import { Verifier } from 'nova-medicina';

const verifier = new Verifier({
  confidenceThreshold: 0.95,
  sources: ['pubmed', 'cochrane', 'uptodate']
});
```

#### Methods

**verify(options)**

Verify medical diagnosis or treatment.

```javascript
const result = await verifier.verify({
  diagnosis: 'type 2 diabetes',
  treatment: 'metformin',
  evidence: {
    fastingGlucose: 140,
    hba1c: 7.2
  }
});
```

Returns:
```javascript
{
  diagnosis: string,
  verified: boolean,
  confidence: number,
  contradictions: string[],
  sources: string[],
  citations: Array<Citation>
}
```

**checkContradictions(claim)**

Check for contradictory evidence.

```javascript
const contradictions = await verifier.checkContradictions({
  claim: 'aspirin for fever in children',
  evidence: {...}
});
```

**getCitations(query)**

Retrieve medical literature citations.

```javascript
const citations = await verifier.getCitations('diabetes treatment');
```

### ProviderSearch

Healthcare provider search and verification.

#### Constructor

```javascript
import { ProviderSearch } from 'nova-medicina';

const search = new ProviderSearch({
  verifyCredentials: true,
  minRating: 4.0,
  maxDistance: 25
});
```

#### Methods

**search(options)**

Search for healthcare providers.

```javascript
const providers = await search.search({
  specialty: 'cardiology',
  location: 'New York, NY',
  insurance: 'Blue Cross',
  acceptingPatients: true
});
```

**verifyCredentials(provider)**

Verify provider medical license and certifications.

```javascript
const credentials = await search.verifyCredentials(provider);
```

### ConfigManager

Configuration management.

#### Methods

**set(key, value)**

Set configuration value.

```javascript
import { ConfigManager } from 'nova-medicina';

const config = new ConfigManager();
await config.set('verification-level', 'strict');
```

**get(key)**

Get configuration value.

```javascript
const level = await config.get('verification-level');
```

**list()**

List all configuration settings.

```javascript
const settings = await config.list();
```

**reset()**

Reset to default configuration.

```javascript
await config.reset();
```

## CLI Commands

### analyze

Analyze symptoms with AI-powered consensus.

```bash
nova-medicina analyze "symptoms" [options]
```

**Options:**
- `-s, --symptoms <symptoms>` - Symptoms (required)
- `-d, --duration <duration>` - Duration
- `-a, --age <age>` - Patient age
- `-g, --gender <gender>` - Patient gender
- `-v, --verification-level <level>` - Verification level
- `-o, --output <file>` - Output file
- `--include-references` - Include citations

**Example:**
```bash
nova-medicina analyze \
  --symptoms "chest pain, shortness of breath" \
  --age 55 \
  --verification-level strict
```

### verify

Verify medical information.

```bash
nova-medicina verify [options]
```

**Options:**
- `-d, --diagnosis <diagnosis>` - Diagnosis to verify
- `-t, --treatment <treatment>` - Treatment to verify
- `-e, --evidence <file>` - Evidence file (JSON)
- `--show-contradictions` - Show contradictions
- `--literature-review` - Include literature review

### provider

Search for healthcare providers.

```bash
nova-medicina provider [options]
```

**Options:**
- `-s, --specialty <specialty>` - Medical specialty
- `-l, --location <location>` - Location
- `-i, --insurance <insurance>` - Insurance provider
- `--verify-credentials` - Verify credentials
- `--accepting-patients` - Only accepting new patients

### config

Manage configuration.

```bash
nova-medicina config [options]
```

**Options:**
- `--set <key=value>` - Set value
- `--get <key>` - Get value
- `--list` - List all settings
- `--reset` - Reset to defaults

## MCP Integration

Nova Medicina can be used as an MCP server.

### Add to Claude Desktop

```bash
claude mcp add nova-medicina npx nova-medicina mcp start
```

### Start MCP Server

```bash
nova-medicina mcp start --port 3000
```

### Available MCP Tools

**nova_medicina_analyze**

Analyze symptoms with full safety checks.

```json
{
  "tool": "nova_medicina_analyze",
  "parameters": {
    "symptoms": "fever and sore throat",
    "age": 28,
    "duration": "3 days"
  }
}
```

**nova_medicina_verify**

Verify medical information.

```json
{
  "tool": "nova_medicina_verify",
  "parameters": {
    "diagnosis": "type 2 diabetes",
    "evidence": {...}
  }
}
```

---

For more information, visit: https://github.com/ruvnet/nova-medicina
