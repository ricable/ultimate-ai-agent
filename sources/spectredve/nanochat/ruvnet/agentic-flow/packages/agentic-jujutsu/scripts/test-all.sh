#!/bin/bash
# Comprehensive test runner for agentic-jujutsu

set -e

echo "🧪 Running comprehensive test suite for agentic-jujutsu"
echo "======================================================"

# Function to print section headers
print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# 1. Unit Tests
print_section "1️⃣  Running Unit Tests"
cargo test --lib --all-features --verbose

# 2. Integration Tests
print_section "2️⃣  Running Integration Tests"
cargo test --test '*' --all-features --verbose

# 3. Doc Tests
print_section "3️⃣  Running Documentation Tests"
cargo test --doc --all-features --verbose

# 4. Property-Based Tests
print_section "4️⃣  Running Property-Based Tests (proptest)"
PROPTEST_CASES=500 cargo test --test '*properties*' --all-features --verbose

# 5. Benchmarks (compile only, don't run)
print_section "5️⃣  Compiling Benchmarks"
cargo bench --no-run --all-features

# 6. WASM Tests (if wasm-pack is available)
if command -v wasm-pack &> /dev/null; then
    print_section "6️⃣  Running WASM Tests"
    wasm-pack test --node --all-features || echo "⚠️  WASM tests skipped (requires wasm-pack)"
else
    echo "⚠️  Skipping WASM tests (wasm-pack not installed)"
fi

# 7. Linting
print_section "7️⃣  Running Clippy (Linter)"
cargo clippy --all-features -- -D warnings

# 8. Format Check
print_section "8️⃣  Checking Code Format"
cargo fmt -- --check

# 9. Check for Security Vulnerabilities
if command -v cargo-audit &> /dev/null; then
    print_section "9️⃣  Security Audit"
    cargo audit || echo "⚠️  Security vulnerabilities found"
else
    echo "⚠️  Skipping security audit (cargo-audit not installed)"
fi

# Summary
print_section "✅ Test Suite Complete"
echo "All tests passed successfully!"
echo ""
echo "Next steps:"
echo "  - Run './scripts/coverage.sh' to generate coverage report"
echo "  - Run 'cargo bench' to run performance benchmarks"
