# Nova Medicina Examples

Comprehensive examples demonstrating the Nova Medicina medical analysis system.

## Overview

This directory contains fully functional, well-commented examples covering all aspects of Nova Medicina:

## Examples

### 1. **basic-usage.js** (327 lines)
Simple symptom analysis, reading confidence scores, and understanding results.

**Features**:
- Simple symptom analysis
- Confidence score interpretation
- Result structure explanation
- Error handling patterns

**Run**: `node basic-usage.js`

### 2. **cli-examples.sh** (360 lines)
Complete CLI command reference with real-world examples.

**Features**:
- All CLI commands demonstrated
- Batch processing examples
- Configuration management
- Piping and chaining workflows
- Real-world scenarios

**Run**: `bash cli-examples.sh`

### 3. **api-client.js** (565 lines)
REST API integration with error handling and WebSocket support.

**Features**:
- API client implementation
- Real-time WebSocket updates
- Batch processing
- Error handling
- System metrics

**Run**: `node api-client.js`

### 4. **provider-integration.js** (625 lines)
Healthcare provider dashboard and review workflows.

**Features**:
- Provider dashboard setup
- Notification configuration
- Review workflow implementation
- Multi-provider coordination
- Provider analytics

**Run**: `node provider-integration.js`

### 5. **mcp-integration.md** (573 lines)
Claude Desktop MCP integration guide.

**Features**:
- MCP configuration
- Tool usage examples
- SSE vs STDIO comparison
- Troubleshooting guide
- Advanced usage patterns

**Read**: View in any markdown viewer

### 6. **advanced-workflows.js** (690 lines)
Complex multi-symptom analysis and learning workflows.

**Features**:
- Multi-symptom analysis
- Intelligent provider escalation
- Continuous learning from feedback
- Risk stratification
- Clinical decision support

**Run**: `node advanced-workflows.js`

## Quick Start

1. **Install dependencies**:
```bash
npm install
```

2. **Configure API**:
```bash
export MEDAI_API_KEY=your_api_key_here
export MEDAI_PROVIDER_ID=your_provider_id
```

3. **Run examples**:
```bash
# Basic usage
node basic-usage.js

# API client
node api-client.js

# Provider integration
node provider-integration.js

# Advanced workflows
node advanced-workflows.js
```

## Documentation

For comprehensive tutorials and guides, see:
- **[TUTORIALS.md](../docs/TUTORIALS.md)** - Step-by-step tutorials
- **[MCP Integration](./mcp-integration.md)** - Claude Desktop integration
- **[CLI Examples](./cli-examples.sh)** - Command-line usage

## Example Categories

### Basic Examples
- Simple symptom analysis
- Confidence score interpretation
- CLI command usage

### Intermediate Examples
- REST API integration
- WebSocket real-time updates
- Batch processing
- Provider workflows

### Advanced Examples
- Multi-system analysis
- Intelligent escalation
- Continuous learning
- Pattern recognition

## Key Features Demonstrated

✓ **Medical Analysis** - Symptom-based diagnosis
✓ **Confidence Scoring** - Multi-factor assessment
✓ **Anti-Hallucination** - Citation verification
✓ **Provider Integration** - Review workflows
✓ **Real-Time Updates** - WebSocket support
✓ **Batch Processing** - Multiple patients
✓ **Pattern Learning** - AgentDB integration
✓ **MCP Integration** - Claude Desktop
✓ **Error Handling** - Comprehensive coverage
✓ **Best Practices** - Production-ready code

## Testing

Each example includes:
- Fully functional code
- Comprehensive comments
- Error handling
- Expected output documentation

## Support

- **Documentation**: [../docs/TUTORIALS.md](../docs/TUTORIALS.md)
- **API Reference**: [../docs/API.md](../docs/API.md)
- **Issues**: https://github.com/nova-medicina/issues

## Statistics

- **Total Examples**: 6 files
- **Total Lines**: ~4,700+ lines
- **Code Coverage**: All major features
- **Documentation**: Comprehensive

---

**Note**: These examples are for demonstration and educational purposes. Always consult qualified healthcare professionals for medical decisions.
