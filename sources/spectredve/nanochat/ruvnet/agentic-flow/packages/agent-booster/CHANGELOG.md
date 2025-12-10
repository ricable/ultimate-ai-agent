# Changelog

All notable changes to Agent Booster will be documented in this file.

## [0.2.1] - 2025-10-08

### Fixed
- **CRITICAL FIX: Strategy Selection for Replacements**
  - Lowered `FuzzyReplace` threshold from 80% → 50%
  - var→const now uses `fuzzy_replace` (was `insert_after` causing duplicates)
  - Type annotations now use `fuzzy_replace` (was `insert_after` causing duplicates)
  - Confidence scores improved from ~57% to ~64% for simple substitutions
- **WASM files missing from npm package**
  - Removed blocking `wasm/.gitignore` file
  - Package now includes full 1.3MB WASM module
  - Package size increased from 28KB → 469KB (correct)

### Changed
- Strategy selection thresholds updated in `crates/agent-booster/src/merge.rs`:
  - `ExactReplace`: 95% → 90% (slightly more lenient)
  - `FuzzyReplace`: 80% → 50% (much more lenient, fixes duplicates)
  - `InsertAfter`: 60% → 30% (fallback for low similarity)

### Test Results
| Operation | v0.1.2 | v0.2.1 | Status |
|-----------|--------|--------|--------|
| var → const | ❌ insert_after (duplicate) | ✅ fuzzy_replace | Fixed |
| Add types | ❌ insert_after (duplicate) | ✅ fuzzy_replace | Fixed |
| Error handling | ✅ exact_replace | ✅ exact_replace | Same |
| Confidence | 57% | 64% | +7% |

## [0.2.0] - 2025-10-08

### ⚠️ YANKED - Missing WASM files in package

Published without WASM files due to blocking .gitignore. Use 0.2.1 instead.

## [0.1.2] - 2025-10-08

### Fixed
- **CLI JSON parsing error**: Fixed stdin detection to check for file arguments first
- Previously: `if (!process.stdin.isTTY)` → always tried to read stdin
- Now: `if (!process.stdin.isTTY && !args.file)` → checks for file arg first
- Resolves: `{"success":false,"error":"Unexpected end of JSON input"}`

### Test Results
- ✅ JSON stdin mode works
- ✅ File argument mode works
- ⚠️ Insert-only behavior (fixed in 0.2.1)

## [0.1.1] - 2025-10-08

### Fixed
- WASM files not included in npm package
- Removed blocking `wasm/.gitignore` file

### Changed
- `prepublishOnly` script: `npm run build` → `npm run build:js`
- No longer requires Rust/cargo for publishing
- Repository URL updated to `ruvnet/agentic-flow`
- Homepage updated to `https://ruv.io`

## [0.1.0] - 2025-10-08

### Added
- Initial release
- Rust/WASM pattern matching engine
- Template-based code transformations
- Tree-sitter AST parsing
- Merge strategies: ExactReplace, FuzzyReplace, InsertAfter, InsertBefore, Append
- Confidence scoring system
- CLI interface (`agent-booster apply`)
- MCP server mode (`agent-booster-server`)
- Validation suite

### Known Issues
- Insert-only behavior (see KNOWN-ISSUES.md)
- Strategy thresholds too conservative (fixed in 0.2.1)

---

**Repository**: https://github.com/ruvnet/agentic-flow/tree/main/agent-booster
**Homepage**: https://ruv.io
**License**: MIT OR Apache-2.0
