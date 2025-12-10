# Nova Medicina MCP Integration Guide

Complete guide for integrating Nova Medicina with Claude Desktop via Model Context Protocol (MCP).

## Table of Contents

1. [MCP Overview](#mcp-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [MCP Tool Examples](#mcp-tool-examples)
5. [SSE vs STDIO](#sse-vs-stdio)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## MCP Overview

The Model Context Protocol (MCP) enables Claude Desktop to directly interact with Nova Medicina's medical analysis capabilities as native tools. This integration provides:

- **Real-time medical analysis** from within Claude conversations
- **Confidence scoring** and anti-hallucination detection
- **Provider review workflows** integrated with chat
- **Pattern recognition** from historical cases
- **Citation verification** for medical claims

### Available MCP Tools

Nova Medicina exposes the following tools via MCP:

| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `medai_analyze` | Analyze symptoms and diagnose conditions | condition, symptoms |
| `medai_verify` | Verify confidence scores | analysisId |
| `medai_review` | Submit provider review | analysisId, decision |
| `medai_notify` | Notify healthcare provider | analysisId, providerId |
| `medai_metrics` | Get system learning metrics | - |
| `medai_patterns` | Search similar cases | symptoms |

---

## Installation

### Prerequisites

- **Claude Desktop** (version 1.0.0+)
- **Node.js** (version 18.0.0+)
- **Nova Medicina** installed globally or locally

### Step 1: Install Nova Medicina

```bash
# Option A: Global installation
npm install -g nova-medicina

# Option B: Local installation
npm install nova-medicina
```

### Step 2: Configure MCP Server

Add Nova Medicina to your Claude Desktop MCP configuration:

**Location**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

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

### Step 3: Restart Claude Desktop

After configuration, restart Claude Desktop to load the MCP server.

---

## Configuration

### Environment Variables

Configure Nova Medicina MCP server with these environment variables:

```bash
# Required
MEDAI_API_KEY=your_api_key_here        # API authentication key
MEDAI_PROVIDER_ID=your_provider_id     # Healthcare provider ID

# Optional
MEDAI_API_URL=http://localhost:3000    # API server URL
MEDAI_MIN_CONFIDENCE=0.75              # Minimum confidence threshold
MEDAI_AUTO_NOTIFY=true                 # Auto-notify on low confidence
MEDAI_LOG_LEVEL=info                   # Logging level (debug, info, warn, error)
```

### Configuration File

Alternatively, use a configuration file:

**Location**: `~/.config/nova-medicina/config.json`

```json
{
  "api": {
    "url": "http://localhost:3000",
    "key": "your_api_key_here",
    "timeout": 30000
  },
  "provider": {
    "id": "your_provider_id",
    "autoNotify": true,
    "notificationChannels": ["email", "sms"]
  },
  "antiHallucination": {
    "minConfidence": 0.75,
    "providerReviewThreshold": 0.75,
    "autoApproveThreshold": 0.90
  },
  "learning": {
    "enablePatternRecognition": true,
    "minPatternConfidence": 0.70
  }
}
```

---

## MCP Tool Examples

### Example 1: Medical Analysis

Ask Claude to analyze symptoms using the MCP tool:

**User**: Analyze a patient with diabetes symptoms: increased thirst, frequent urination, unexplained weight loss. Patient is 45 years old, male.

**Claude** will automatically invoke:

```json
{
  "tool": "medai_analyze",
  "parameters": {
    "condition": "diabetes",
    "symptoms": [
      "increased thirst",
      "frequent urination",
      "unexplained weight loss"
    ],
    "patientContext": {
      "age": 45,
      "gender": "male"
    }
  }
}
```

**Response**:

```json
{
  "id": "analysis_abc123",
  "diagnosis": {
    "condition": "Type 2 Diabetes Mellitus",
    "icdCode": "E11.9",
    "confidence": 0.87
  },
  "confidenceScore": {
    "overall": 0.87,
    "requiresProviderReview": false
  },
  "recommendations": [
    {
      "type": "diagnostic_test",
      "description": "Fasting blood glucose test",
      "priority": "high"
    },
    {
      "type": "diagnostic_test",
      "description": "HbA1c test",
      "priority": "high"
    }
  ]
}
```

### Example 2: Confidence Verification

**User**: Verify the confidence score for analysis_abc123

**Claude** will invoke:

```json
{
  "tool": "medai_verify",
  "parameters": {
    "analysisId": "analysis_abc123"
  }
}
```

**Response**:

```json
{
  "analysisId": "analysis_abc123",
  "confidenceScore": {
    "overall": 0.87,
    "breakdown": {
      "diagnosisConfidence": 0.92,
      "citationQuality": 0.85,
      "knowledgeBaseValidation": 0.88,
      "contradictionCheck": 1.0,
      "providerAlignment": 0.80
    }
  },
  "warnings": [],
  "requiresProviderReview": false
}
```

### Example 3: Provider Review

**User**: Submit provider review for analysis_abc123, decision: approved, comments: "Diagnosis confirmed by lab results"

**Claude** will invoke:

```json
{
  "tool": "medai_review",
  "parameters": {
    "analysisId": "analysis_abc123",
    "decision": "approved",
    "comments": "Diagnosis confirmed by lab results"
  }
}
```

**Response**:

```json
{
  "status": "success",
  "reviewId": "review_xyz789",
  "learningUpdated": true,
  "message": "Review submitted successfully. Pattern recognition updated."
}
```

### Example 4: Pattern Recognition

**User**: Find similar cases with symptoms: fever, cough, chest pain

**Claude** will invoke:

```json
{
  "tool": "medai_patterns",
  "parameters": {
    "symptoms": ["fever", "cough", "chest pain"]
  }
}
```

**Response**:

```json
{
  "similarCases": [
    {
      "analysisId": "analysis_def456",
      "diagnosis": "Bacterial Pneumonia",
      "similarity": 0.92,
      "outcome": "successful_treatment",
      "providerApproved": true
    },
    {
      "analysisId": "analysis_ghi789",
      "diagnosis": "Community-Acquired Pneumonia",
      "similarity": 0.88,
      "outcome": "successful_treatment",
      "providerApproved": true
    }
  ],
  "recognizedPatterns": [
    {
      "pattern": "respiratory_infection_elderly",
      "confidence": 0.85,
      "frequency": 127
    }
  ]
}
```

### Example 5: System Metrics

**User**: Show Nova Medicina system metrics

**Claude** will invoke:

```json
{
  "tool": "medai_metrics",
  "parameters": {}
}
```

**Response**:

```json
{
  "totalAnalyses": 1247,
  "successRate": 0.94,
  "avgConfidence": 0.83,
  "patternsLearned": 342,
  "providerReviews": {
    "total": 234,
    "approved": 198,
    "modified": 28,
    "rejected": 8,
    "approvalRate": 0.85
  }
}
```

---

## SSE vs STDIO

Nova Medicina MCP server supports two transport protocols:

### STDIO (Standard Input/Output)

**Default mode**. Best for local Claude Desktop usage.

**Configuration**:
```json
{
  "mcpServers": {
    "nova-medicina": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start"],
      "transport": "stdio"
    }
  }
}
```

**Pros**:
- Simple setup
- Low latency
- Bidirectional communication
- Automatic lifecycle management

**Cons**:
- Local only
- No remote access

### SSE (Server-Sent Events)

**Alternative mode**. Best for remote/networked scenarios.

**Configuration**:
```json
{
  "mcpServers": {
    "nova-medicina": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start", "--transport", "sse", "--port", "3001"],
      "transport": "sse",
      "url": "http://localhost:3001/sse"
    }
  }
}
```

**Pros**:
- Remote access
- Network-based
- Multiple clients
- Web-compatible

**Cons**:
- Higher latency
- Requires network configuration
- Unidirectional (server → client)

### Choosing Between SSE and STDIO

| Use Case | Recommended |
|----------|-------------|
| Local Claude Desktop | STDIO |
| Remote server setup | SSE |
| Multi-client access | SSE |
| Low-latency required | STDIO |
| Web integration | SSE |
| Simple deployment | STDIO |

---

## Advanced Usage

### Custom Tool Configuration

Create specialized MCP tools for specific workflows:

**Example**: Emergency triage tool

```json
{
  "mcpServers": {
    "nova-medicina-emergency": {
      "command": "npx",
      "args": [
        "nova-medicina",
        "mcp",
        "start",
        "--mode", "emergency",
        "--auto-notify",
        "--priority", "urgent"
      ],
      "env": {
        "MEDAI_EMERGENCY_MODE": "true",
        "MEDAI_AUTO_ESCALATE": "true",
        "MEDAI_NOTIFY_THRESHOLD": "0.0"
      }
    }
  }
}
```

### Workflow Integration

Combine MCP tools with Claude workflows:

**Example**: Complete diagnostic workflow

```
User: I have a patient with chest pain, shortness of breath, and sweating. Age 58, male, history of hypertension.

Claude (automated workflow):
1. Invoke medai_analyze with emergency flag
2. Check confidence with medai_verify
3. If low confidence, invoke medai_notify for urgent review
4. Search similar cases with medai_patterns
5. Provide comprehensive analysis with citations
```

### Multi-Provider Setup

Configure multiple MCP servers for different specializations:

```json
{
  "mcpServers": {
    "medai-cardiology": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start"],
      "env": {
        "MEDAI_PROVIDER_ID": "CARDIO_PROVIDER_001",
        "MEDAI_SPECIALIZATION": "cardiology"
      }
    },
    "medai-neurology": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start"],
      "env": {
        "MEDAI_PROVIDER_ID": "NEURO_PROVIDER_001",
        "MEDAI_SPECIALIZATION": "neurology"
      }
    }
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. MCP Server Not Starting

**Symptom**: Claude shows "nova-medicina MCP server unavailable"

**Solutions**:
```bash
# Check Node.js version
node --version  # Should be 18+

# Test MCP server manually
npx nova-medicina mcp start

# Check Claude Desktop logs
tail -f ~/Library/Logs/Claude/mcp-server-nova-medicina.log
```

#### 2. Authentication Errors

**Symptom**: "Invalid API key" or "Authentication failed"

**Solutions**:
- Verify `MEDAI_API_KEY` in configuration
- Check API key validity: `medai config show`
- Regenerate API key: `medai config set api.key new_key`

#### 3. Tool Not Found

**Symptom**: Claude says "I don't have access to that tool"

**Solutions**:
- Restart Claude Desktop
- Verify MCP configuration syntax
- Check tool names match exactly
- Review Claude Desktop MCP logs

#### 4. Slow Response Times

**Symptom**: Long delays when invoking tools

**Solutions**:
```bash
# Use STDIO instead of SSE for local use
# Increase timeout in configuration
{
  "api": {
    "timeout": 60000  // Increase to 60s
  }
}

# Check API server health
medai health

# Monitor performance
medai metrics --live
```

#### 5. Connection Timeouts

**Symptom**: "Connection timeout" or "Server not responding"

**Solutions**:
```bash
# Check API server is running
curl http://localhost:3000/api/health

# Start API server if needed
medai serve --port 3000

# Verify network connectivity
ping localhost
```

### Debug Mode

Enable detailed logging:

```json
{
  "mcpServers": {
    "nova-medicina": {
      "command": "npx",
      "args": ["nova-medicina", "mcp", "start", "--debug"],
      "env": {
        "MEDAI_LOG_LEVEL": "debug",
        "MEDAI_MCP_DEBUG": "true"
      }
    }
  }
}
```

View debug logs:
```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp-server-nova-medicina.log

# Linux
tail -f ~/.local/share/Claude/logs/mcp-server-nova-medicina.log

# Windows
type %APPDATA%\Claude\logs\mcp-server-nova-medicina.log
```

### Getting Help

If issues persist:

1. **Check documentation**: `medai docs`
2. **View examples**: `medai examples`
3. **Community support**: https://github.com/nova-medicina/support
4. **Report bugs**: https://github.com/nova-medicina/issues

---

## Best Practices

1. **Always verify confidence scores** for critical medical decisions
2. **Use emergency flag** for urgent cases
3. **Enable provider review** for low confidence analyses
4. **Leverage pattern recognition** for similar cases
5. **Monitor system metrics** regularly
6. **Keep API key secure** (never commit to version control)
7. **Update regularly** for latest features and fixes

## Next Steps

- Review [advanced-workflows.js](./advanced-workflows.js) for complex scenarios
- Check [docs/TUTORIALS.md](../docs/TUTORIALS.md) for comprehensive guides
- Explore [api-client.js](./api-client.js) for programmatic integration
- See [provider-integration.js](./provider-integration.js) for healthcare workflows

---

**Note**: Nova Medicina is a medical analysis tool designed to assist healthcare professionals. Always consult with qualified medical practitioners for diagnosis and treatment decisions.
