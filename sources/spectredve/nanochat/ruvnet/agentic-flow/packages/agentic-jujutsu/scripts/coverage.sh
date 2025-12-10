#!/bin/bash
# Coverage report generation script for agentic-jujutsu

set -e

echo "🔍 Generating test coverage report for agentic-jujutsu..."

# Check if cargo-tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo "📦 Installing cargo-tarpaulin..."
    cargo install cargo-tarpaulin
fi

# Clean previous coverage data
echo "🧹 Cleaning previous coverage data..."
cargo clean

# Run tests with coverage
echo "🧪 Running tests with coverage tracking..."
cargo tarpaulin \
    --out Html \
    --output-dir coverage \
    --timeout 300 \
    --all-features \
    --exclude-files "benches/*" \
    --exclude-files "tests/*" \
    --skip-clean \
    --verbose

# Generate coverage badge
if [ -f coverage/index.html ]; then
    echo "✅ Coverage report generated successfully!"
    echo "📊 Report location: coverage/index.html"

    # Extract coverage percentage
    if command -v grep &> /dev/null; then
        COVERAGE=$(grep -oP '\d+\.\d+%' coverage/index.html | head -1 || echo "N/A")
        echo "📈 Total Coverage: $COVERAGE"
    fi
else
    echo "❌ Failed to generate coverage report"
    exit 1
fi

# Optional: Generate JSON report for CI/CD
echo "📄 Generating JSON report..."
cargo tarpaulin \
    --out Json \
    --output-dir coverage \
    --timeout 300 \
    --all-features \
    --skip-clean \
    --quiet || true

echo "🎉 Coverage generation complete!"
echo ""
echo "To view the report, open: file://$(pwd)/coverage/index.html"
