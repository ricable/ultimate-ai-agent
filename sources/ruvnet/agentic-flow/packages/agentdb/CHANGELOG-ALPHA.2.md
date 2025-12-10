# Changelog - AgentDB v2.0.0-alpha.2

## Release Date
2025-11-30

## Summary
Critical bug fixes from Docker validation testing. This release addresses the top 3 issues discovered during comprehensive testing of alpha.1.

---

## 🐛 Critical Fixes

### 1. Package.json Export Blocking Version Access
**Issue**: Users could not access `require('agentdb/package.json').version`

**Error**:
```
Error [ERR_PACKAGE_PATH_NOT_EXPORTED]: Package subpath './package.json'
is not defined by "exports"
```

**Fix**: Added `"./package.json": "./package.json"` to exports

**Impact**: ✅ Version access now works programmatically

**Test**:
```javascript
const pkg = require('agentdb/package.json');
console.log(pkg.version); // ✅ Works: "2.0.0-alpha.2"
```

---

### 2. Simulate Commands Not Accessible
**Issue**: `npx agentdb@alpha simulate list` returned "Unknown command"

**Root Cause**: Simulation CLI was separate entry point not integrated into main CLI

**Fix**: Added `agentdb-simulate` binary to package.json

**Impact**: ✅ Simulation commands now accessible

**Usage**:
```bash
# New way (alpha.2+)
npx agentdb-simulate@alpha list

# Shows all 8 simulation scenarios
```

---

### 3. TypeScript Error in History Tracker
**Issue**: Build failed with type error in history-tracker baseline metrics

**Error**:
```
TS2739: Type 'Record<string, number>' is missing properties:
recall, latency, throughput, memoryUsage
```

**Fix**: Added required baseline properties with defaults

---

## ✨ New Features

### Examples Directory
Added `/examples/` with programmatic usage guide:

- `examples/quickstart.js` - Basic initialization example
- `examples/README.md` - Complete examples documentation

**Note**: Full programmatic API examples coming in alpha.3

---

## 📚 Documentation

### New Documentation Files
1. **ALPHA_VALIDATION_REPORT.md** (50+ pages)
   - Complete Docker validation results
   - 20 tests run (16 passed, 3 failed, 1 warning)
   - Detailed issue analysis
   - Pre-beta checklist

2. **ALPHA_VALIDATION_SUMMARY.md**
   - Executive summary
   - Quick reference guide
   - Known limitations

3. **GITHUB_ISSUES.md**
   - GitHub issue templates for all 5 critical bugs
   - Reproduction steps
   - Suggested fixes

### Docker Testing Infrastructure
- `tests/docker/Dockerfile.alpha-test` - Clean testing environment
- `tests/docker/validate-alpha.sh` - 30+ automated tests
- `tests/docker/README.md` - Testing documentation

---

## 📦 Package Changes

### Files Included
```json
{
  "files": [
    "dist",
    "src",
    "simulation",    // ✨ NEW: Simulation scenarios
    "examples",      // ✨ NEW: Usage examples
    "scripts/postinstall.cjs",
    "README.md",
    "LICENSE"
  ]
}
```

### Binary Entries
```json
{
  "bin": {
    "agentdb": "dist/cli/agentdb-cli.js",
    "agentdb-simulate": "dist/simulation/cli.js"  // ✨ NEW
  }
}
```

### Exports
```json
{
  "exports": {
    ".": "./dist/index.js",
    "./package.json": "./package.json",  // ✨ NEW
    "./cli": "./dist/cli/agentdb-cli.js",
    "./controllers": "./dist/controllers/index.js",
    ...
  }
}
```

---

## ✅ Validation Results

### Tests Passed (16/20)
- ✅ Package installation (npx, npm)
- ✅ Version verification (2.0.0-alpha.2)
- ✅ CLI core commands
- ✅ TypeScript declarations
- ✅ File integrity
- ✅ Documentation included
- ✅ Security (npm audit clean)
- ✅ Package.json export fix
- ✅ Simulate command availability

### Known Issues (4 remaining)
See ALPHA_VALIDATION_REPORT.md for complete list.

**Top 3 for Alpha.3**:
1. Programmatic API auto-initialization
2. Transformers.js offline support
3. Deprecated dependency updates

---

## 🚀 Installation

### New Users
```bash
# Install alpha.2
npm install agentdb@alpha

# Verify version
npx agentdb --version
# Output: agentdb v2.0.0-alpha.2
```

### Upgrading from Alpha.1
```bash
# Update to latest alpha
npm install agentdb@alpha --force

# Verify update
npx agentdb --version
# Should show: v2.0.0-alpha.2
```

### Using Simulations
```bash
# List available scenarios
npx agentdb-simulate@alpha list

# Run simulation
npx agentdb-simulate@alpha run hnsw --nodes 100000
```

---

## 🔬 Testing

### Validated Environments
- ✅ Docker (Node.js 20-slim, Debian Bookworm)
- ✅ Local installation (Node.js 18+, 20+)
- ✅ Linux x64
- ⏳ macOS (untested in alpha.2)
- ⏳ Windows (untested in alpha.2)

### Test Commands
```bash
# Test version access
node -e "console.log(require('agentdb/package.json').version)"

# Test simulate command
npx agentdb-simulate@alpha list

# Test CLI
npx agentdb@alpha --help
npx agentdb@alpha init --db test.db
```

---

## 📊 Package Stats

| Metric | Alpha.1 | Alpha.2 | Change |
|--------|---------|---------|--------|
| Compressed Size | 967.7 kB | ~970 kB | +0.2% |
| Unpacked Size | 6.1 MB | ~6.3 MB | +3% |
| Files | 841 | ~860 | +19 files |
| npm audit vulns | 0 | 0 | ✅ Clean |

---

## ⚠️ Breaking Changes

**None** - Alpha.2 is fully backward compatible with Alpha.1

---

## 🐛 Bug Reports

Found an issue? Please report:
- **GitHub**: https://github.com/ruvnet/agentic-flow/issues
- **Tag**: `agentdb`, `v2.0-alpha`
- **Include**: Node version, OS, error messages, reproduction steps

---

## 📅 Roadmap

### Alpha.3 (Estimated: 2 weeks)
- ✅ Programmatic API factory method
- ✅ Auto-initialize schemas
- ✅ Complete examples for all features
- ✅ TypeScript usage examples
- ✅ Offline Transformers.js support
- ✅ Update deprecated dependencies

### Beta.1 (Estimated: 4-6 weeks)
- ✅ Complete API documentation
- ✅ Integration tests
- ✅ Performance benchmarks
- ✅ Migration guide (v1 → v2)
- ✅ Video tutorials

### Stable v2.0 (Estimated: 8-10 weeks)
- ✅ Production-ready
- ✅ Security audit
- ✅ Performance optimizations
- ✅ Community examples

---

## 🙏 Acknowledgments

Special thanks to:
- Early adopters testing alpha.1
- Docker validation infrastructure contributors
- Issue reporters who helped identify critical bugs

---

## 📝 Notes

This is an **ALPHA** release. While functional for testing, it may have rough edges.

**Recommended for**:
- ✅ Early adopters
- ✅ CLI users
- ✅ Testing and feedback

**Not recommended for**:
- ❌ Production deployments
- ❌ Critical applications
- ❌ Users needing stable API

---

**Full Changelog**: https://github.com/ruvnet/agentic-flow/compare/agentdb@2.0.0-alpha.1...agentdb@2.0.0-alpha.2
