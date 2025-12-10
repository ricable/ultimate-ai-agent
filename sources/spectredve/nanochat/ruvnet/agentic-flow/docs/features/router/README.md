# Multi-Model Router - Intelligent Model Selection

> Automatic model selection based on task requirements and optimization priorities

## 📚 Documentation

- [Router User Guide](ROUTER_USER_GUIDE.md) - Complete user guide
- [Router Config Reference](ROUTER_CONFIG_REFERENCE.md) - Configuration options
- [Top 20 Models Matrix](TOP20_MODELS_MATRIX.md) - Model comparison

## 🚀 Quick Start

### Basic Usage
```bash
# Auto-select optimal model
npx claude-flow agent \
  --agent coder \
  --task "Implement authentication" \
  --priority quality

# Cost-optimized model
npx claude-flow agent \
  --agent coder \
  --task "Fix typo" \
  --priority cost
```

## 🎯 Optimization Priorities

- **quality** - Best results (Sonnet 4.5, GPT-4)
- **balanced** - Cost/quality balance (GPT-4-mini, Gemini Pro)
- **cost** - Cheapest options (OpenRouter, Gemini Flash)
- **speed** - Fastest response (Haiku, GPT-3.5)
- **privacy** - Local only (ONNX models)

---

**Back to**: [Features](../README.md) | [Main Documentation](../../README.md)
