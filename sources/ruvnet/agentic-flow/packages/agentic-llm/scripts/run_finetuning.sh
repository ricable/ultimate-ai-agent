#!/bin/bash
# Complete MCP Fine-Tuning Pipeline
# Trains, validates, and benchmarks Phi-4 for Claude Agent SDK

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   MCP Tool Fine-Tuning for Claude Agent SDK              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check GPU
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
    echo "✅ GPU detected"
else
    echo "⚠️  No GPU - training will be slow"
fi

echo ""
echo "📊 Step 1: Generating MCP Tool Dataset..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 training/mcp_dataset.py

echo ""
echo "🎓 Step 2: Fine-Tuning Phi-4..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "  • LoRA rank: 32 (optimal for tool calling)"
echo "  • Learning rate: 5e-5"
echo "  • Epochs: 5"
echo "  • Batch size: 4 (gradient accumulation: 4)"
echo ""
python3 training/finetune_mcp.py

echo ""
echo "✅ Step 3: Validating MCP Tool Accuracy..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 validation/mcp_validator.py \
    --model-path /app/checkpoints/mcp_finetuned/final_model \
    --output-dir /app/validation/results

echo ""
echo "📈 Step 4: Running Before/After Benchmarks..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 benchmarks/finetune_comparison.py \
    --finetuned /app/checkpoints/mcp_finetuned/final_model \
    --output-dir /app/benchmarks/finetune_results

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              FINE-TUNING COMPLETE!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results:"
echo "  • Model: /app/checkpoints/mcp_finetuned/final_model"
echo "  • Validation: /app/validation/results/mcp_validation_results.json"
echo "  • Benchmarks: /app/benchmarks/finetune_results/finetune_comparison.json"
echo ""

# Display summary
if [ -f "/app/benchmarks/finetune_results/finetune_comparison.json" ]; then
    echo "📊 Quick Summary:"
    python3 << 'EOF'
import json
try:
    with open('/app/benchmarks/finetune_results/finetune_comparison.json', 'r') as f:
        data = json.load(f)
        ft = data['finetuned']
        print(f"\n  Tool Accuracy: {ft['accuracy']:.1f}%")
        print(f"  Parameter Accuracy: {ft['param_accuracy']:.1f}%")
        print(f"  Tool Detection: {ft['tool_found']}/{ft['total']}")

        if ft['accuracy'] >= 95:
            print("\n  ✅ EXCELLENT - Production ready!")
        elif ft['accuracy'] >= 85:
            print("\n  ✓ GOOD - Ready for deployment")
        else:
            print("\n  ⚠ Needs improvement - more training recommended")
except Exception as e:
    print(f"  Could not load summary: {e}")
EOF
fi

echo ""
echo "🚀 Next Steps:"
echo "  1. Review validation results"
echo "  2. Export to ONNX: python3 quantization/quantize.py"
echo "  3. Deploy with Claude Agent SDK"
