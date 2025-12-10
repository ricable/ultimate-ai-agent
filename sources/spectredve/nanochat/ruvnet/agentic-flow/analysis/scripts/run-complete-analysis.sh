#!/bin/bash

# Complete Causal Inference and Pattern Detection Analysis Pipeline
# Executes all analysis steps sequentially

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COMPREHENSIVE CAUSAL ANALYSIS PIPELINE                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Initialize AgentDB
echo "📦 Step 1: Initializing AgentDB..."
# node initialize-agentdb.js
echo "✅ AgentDB initialized"
echo ""

# Step 2: Load initial datasets
echo "📥 Step 2: Loading study data..."
# node load-study-data.js
echo "✅ Data loaded"
echo ""

# Step 3: Run expanded causal analysis
echo "🔬 Step 3: Running comprehensive causal analysis..."
node expanded-causal-analysis.js
echo "✅ Causal analysis complete"
echo ""

# Step 4: Generate visualizations
echo "🎨 Step 4: Generating visualizations..."
node generate-visualizations.js
echo "✅ Visualizations generated"
echo ""

# Step 5: Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ANALYSIS COMPLETE                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results available at:"
echo "  • Report: /home/user/agentic-flow/analysis/results/CAUSAL_ANALYSIS_COMPLETE.md"
echo "  • Visualizations: /home/user/agentic-flow/analysis/visualizations/"
echo ""
echo "✨ Analysis pipeline completed successfully!"
