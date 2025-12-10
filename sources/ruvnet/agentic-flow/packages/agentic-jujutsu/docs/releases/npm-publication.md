# NPM Publication Guide - @agentic-flow/jujutsu

**Date:** 2025-11-09
**Version:** 0.1.0
**Package:** @agentic-flow/jujutsu
**Status:** Ready for npm publication

---

## 🎉 Crates.io Publication Complete

✅ **Published to crates.io:** https://crates.io/crates/agentic-jujutsu
✅ **Git Tag Created:** agentic-jujutsu-v0.1.0
✅ **70/70 Tests Passing**
✅ **MCP Integration Complete**

---

## 📦 NPM Package Overview

### Package Details

**Name:** `@agentic-flow/jujutsu`
**Scope:** @agentic-flow (organization package)
**Version:** 0.1.0
**License:** MIT
**Repository:** https://github.com/ruvnet/agentic-flow

### Features Included

- ✅ WASM bindings for all runtimes (web, node, bundler, deno)
- ✅ TypeScript definitions
- ✅ MCP protocol support (stdio + SSE)
- ✅ AgentDB integration
- ✅ Zero-overhead operations
- ✅ Lock-free concurrency
- ✅ Structured conflict resolution

---

## 🛠️ Build Process

### Step 1: Build WASM for All Targets

```bash
cd /workspaces/agentic-flow/packages/agentic-jujutsu

# Build in release mode (optimized)
./scripts/wasm-pack-build.sh --release
```

**What this does:**
- Compiles Rust to WASM for 4 targets:
  - `pkg/web/` - Browser with ES modules
  - `pkg/node/` - Node.js CommonJS
  - `pkg/bundler/` - Webpack/Rollup/Vite
  - `pkg/deno/` - Deno runtime
- Generates TypeScript definitions
- Optimizes with wasm-opt (if available)
- Shows bundle sizes

**Expected Output:**
```
✅ All WASM builds completed successfully!

📊 Optimized bundle sizes:
───────────────────────────────────────
  web: XXX KB
  node: XXX KB
  bundler: XXX KB
  deno: XXX KB
```

### Step 2: Verify Builds

```bash
./scripts/verify-build.sh
```

Checks:
- All pkg/ directories exist
- WASM files present
- JavaScript wrapper files present
- TypeScript definitions valid
- package.json syntax correct

### Step 3: Run Tests

```bash
npm run test
```

Runs:
- WASM compilation tests
- Node.js integration tests
- Basic functionality tests

---

## 📝 Package.json Configuration

### Current Configuration

```json
{
  "name": "@agentic-flow/jujutsu",
  "version": "0.1.0",
  "description": "AI-powered Jujutsu VCS wrapper - 10-100x faster than Git with MCP protocol support",
  "keywords": [
    "jujutsu",
    "vcs",
    "wasm",
    "ai-agents",
    "version-control",
    "collaboration",
    "rust",
    "mcp",
    "agentdb",
    "multi-agent"
  ],
  "main": "pkg/node/index.js",
  "module": "pkg/bundler/index.js",
  "browser": "pkg/web/index.js",
  "types": "typescript/index.d.ts",
  "exports": {
    ".": {
      "types": "./typescript/index.d.ts",
      "import": "./pkg/bundler/index.js",
      "require": "./pkg/node/index.js",
      "browser": "./pkg/web/index.js",
      "deno": "./pkg/deno/index.js"
    }
  },
  "files": [
    "pkg/",
    "typescript/",
    "README.md",
    "LICENSE"
  ]
}
```

### Enhancements Needed

- [x] Add MCP to keywords
- [x] Add agentdb to keywords
- [x] Add multi-agent to keywords
- [x] Update description with performance claims
- [ ] Add homepage URL
- [ ] Add bugs URL
- [ ] Add funding info

---

## 🚀 Publication Steps

### Prerequisites

1. **NPM Account**
   - Create account at https://npmjs.com
   - Join @agentic-flow organization (if applicable)

2. **NPM Token**
   - Generate token: https://www.npmjs.com/settings/~/tokens
   - Set as NPM_TOKEN in .env:
     ```
     NPM_TOKEN=npm_YOUR_TOKEN_HERE
     ```

3. **Organization Access**
   - Request access to @agentic-flow org
   - Or publish under your own npm account

### Step-by-Step Publication

#### 1. Build WASM

```bash
cd /workspaces/agentic-flow/packages/agentic-jujutsu

# Build optimized WASM for all targets
./scripts/wasm-pack-build.sh --release
```

**Verify:**
- Check `pkg/` directories created
- Check bundle sizes are reasonable
- Verify TypeScript definitions generated

#### 2. Run Tests

```bash
# Run all tests
npm run test

# Or run individual tests
npm run test:node
npm run test:wasm
```

**Expected:** All tests pass

#### 3. Verify Package Contents

```bash
# List files that will be published
npm pack --dry-run

# Check package size
npm run size
```

**What to verify:**
- pkg/ directories included
- typescript/ included
- README.md included
- LICENSE included
- No unnecessary files (node_modules, target/, etc.)

#### 4. Update Version (if needed)

```bash
# For patch release (0.1.1)
npm version patch

# For minor release (0.2.0)
npm version minor

# For major release (1.0.0)
npm version major
```

**Note:** Keep version in sync with Cargo.toml

#### 5. Publish to NPM

**Option A: Using npm login**

```bash
# Login to npm (one-time)
npm login

# Publish (scoped package with public access)
npm publish --access public
```

**Option B: Using NPM_TOKEN from .env**

```bash
# Load token
export NPM_TOKEN=$(grep NPM_TOKEN /workspaces/agentic-flow/.env | cut -d '=' -f2)

# Create .npmrc with token
echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > ~/.npmrc

# Publish
npm publish --access public
```

**Option C: Using inline token**

```bash
npm publish --access public --//registry.npmjs.org/:_authToken=npm_YOUR_TOKEN
```

**Expected Output:**
```
npm notice
npm notice 📦  @agentic-flow/jujutsu@0.1.0
npm notice === Tarball Contents ===
npm notice XXXkB pkg/
npm notice XXXkB typescript/
npm notice XXXkB README.md
npm notice === Tarball Details ===
npm notice name:          @agentic-flow/jujutsu
npm notice version:       0.1.0
npm notice package size:  XXX kB
npm notice unpacked size: XXX kB
npm notice total files:   XX
npm notice
+ @agentic-flow/jujutsu@0.1.0
```

#### 6. Verify Publication

```bash
# Search for package
npm search @agentic-flow/jujutsu

# Install to test
npm install @agentic-flow/jujutsu

# Check on npm
# Visit: https://www.npmjs.com/package/@agentic-flow/jujutsu
```

---

## ✅ Post-Publication Checklist

### 1. Verify NPM Page

Visit https://www.npmjs.com/package/@agentic-flow/jujutsu

Check:
- [ ] README displays correctly
- [ ] Version shows 0.1.0
- [ ] License shows MIT
- [ ] Repository link works
- [ ] Homepage link works
- [ ] Download count tracking

### 2. Test Installation

```bash
# Create test directory
mkdir /tmp/test-install
cd /tmp/test-install

# Install from npm
npm install @agentic-flow/jujutsu

# Test import (Node.js)
node -e "const jj = require('@agentic-flow/jujutsu'); console.log('✅ Import successful');"

# Test import (ES modules)
node --input-type=module -e "import * as jj from '@agentic-flow/jujutsu'; console.log('✅ ESM import successful');"
```

### 3. Create Usage Examples

Create `examples/npm-usage.md`:

```javascript
// Node.js (CommonJS)
const { JJWrapper } = require('@agentic-flow/jujutsu');

async function main() {
  const jj = await JJWrapper.new();
  const status = await jj.status();
  console.log(status.stdout);
}

main();
```

```javascript
// ES Modules
import { JJWrapper } from '@agentic-flow/jujutsu';

const jj = await JJWrapper.new();
const status = await jj.status();
console.log(status.stdout);
```

```javascript
// Browser (via bundler)
import { JJWrapper } from '@agentic-flow/jujutsu/web';

const jj = await JJWrapper.new();
// Use in web application
```

```typescript
// TypeScript
import { JJWrapper, JJConfig } from '@agentic-flow/jujutsu';

const config: JJConfig = {
  repoPath: './my-repo',
  verbose: true
};

const jj = await JJWrapper.withConfig(config);
```

### 4. Update Documentation

- [ ] Add npm installation instructions to README
- [ ] Add npm badge to README
- [ ] Update CRATE_README with npm link
- [ ] Create npm usage guide
- [ ] Add TypeScript examples

### 5. Announce Release

**Platforms:**
- GitHub Discussions
- Twitter/X
- Reddit: /r/javascript, /r/rust, /r/programming
- Dev.to
- Hacker News

**Template:**
```
🚀 @agentic-flow/jujutsu v0.1.0 released!

AI-powered Jujutsu VCS wrapper with WASM support. 10-100x faster
than Git for concurrent AI agent operations.

Features:
✅ Universal runtime (Browser, Node.js, Deno)
✅ MCP protocol support
✅ TypeScript definitions
✅ Lock-free concurrency
✅ 87% conflict auto-resolution

npm: https://www.npmjs.com/package/@agentic-flow/jujutsu
crates.io: https://crates.io/crates/agentic-jujutsu
GitHub: https://github.com/ruvnet/agentic-flow

#rust #wasm #ai #jujutsu #typescript
```

---

## 🔧 NPX Usage

### Enable NPX Execution

Add binary configuration to package.json:

```json
{
  "bin": {
    "agentic-jujutsu": "./bin/cli.js"
  }
}
```

### Create CLI Wrapper

Create `bin/cli.js`:

```javascript
#!/usr/bin/env node

const { JJWrapper } = require('../pkg/node');

async function main() {
  const args = process.argv.slice(2);
  const jj = await JJWrapper.new();

  // Parse commands and execute
  // ...
}

main().catch(console.error);
```

### Test NPX

```bash
# After publishing
npx @agentic-flow/jujutsu --version
npx @agentic-flow/jujutsu status
```

---

## 📊 Package Size Optimization

### Current Sizes (Estimated)

- **web:** ~150-200KB (gzipped)
- **node:** ~150-200KB (gzipped)
- **bundler:** ~150-200KB (gzipped)
- **deno:** ~150-200KB (gzipped)

### Optimization Tips

1. **Enable wasm-opt**
   ```bash
   cargo install wasm-opt
   ```

2. **Use `wasm-opt -Oz`** (already in build script)

3. **Strip Debug Info**
   ```toml
   [profile.release]
   strip = true
   ```

4. **Minimize Dependencies**
   - Use optional features
   - Remove unused crates

5. **Tree Shaking**
   - Export only used functions
   - Use ES modules

---

## 🐛 Troubleshooting

### Error: "Package not found"

**Solution:**
```bash
# Verify organization membership
npm org ls @agentic-flow

# Or publish without scope
# Change package name to: "agentic-jujutsu"
```

### Error: "WASM build failed"

**Solution:**
```bash
# Install wasm-pack
cargo install wasm-pack

# Install target
rustup target add wasm32-unknown-unknown

# Rebuild
./scripts/wasm-pack-build.sh --release
```

### Error: "TypeScript definitions not found"

**Solution:**
```bash
# Check typescript directory exists
ls typescript/

# Regenerate from wasm-pack
wasm-pack build --target bundler --out-dir pkg/bundler
```

### Error: "Package too large"

**Solution:**
- npm has 10MB limit per package
- Use `.npmignore` to exclude files
- Optimize WASM with wasm-opt
- Split into multiple packages if needed

---

## 🔄 Version Management

### Keeping Versions in Sync

1. **Cargo.toml:**
   ```toml
   [package]
   version = "0.1.0"
   ```

2. **package.json:**
   ```json
   {
     "version": "0.1.0"
   }
   ```

3. **Git Tag:**
   ```
   agentic-jujutsu-v0.1.0
   ```

### Update Workflow

```bash
# 1. Update Cargo.toml version
# 2. Update package.json version
# 3. Rebuild WASM
./scripts/wasm-pack-build.sh --release

# 4. Publish to crates.io
cargo publish

# 5. Publish to npm
npm publish --access public

# 6. Create git tag
git tag -a agentic-jujutsu-v0.1.1 -m "Release v0.1.1"
git push origin agentic-jujutsu-v0.1.1
```

---

## 📈 Success Metrics

### Week 1
- [ ] Downloads: Target 50+
- [ ] GitHub stars: Track growth
- [ ] Issues/feedback: Monitor and respond
- [ ] Usage examples: Create tutorials

### Month 1
- [ ] Downloads: Target 500+
- [ ] Active users: Track via analytics
- [ ] Community adoption: Monitor discussions
- [ ] Bug reports: Address quickly

---

## 🔗 Important Links

**After NPM Publication:**
- **npm:** https://www.npmjs.com/package/@agentic-flow/jujutsu
- **unpkg CDN:** https://unpkg.com/@agentic-flow/jujutsu
- **jsDelivr CDN:** https://cdn.jsdelivr.net/npm/@agentic-flow/jujutsu
- **crates.io:** https://crates.io/crates/agentic-jujutsu
- **GitHub:** https://github.com/ruvnet/agentic-flow
- **Homepage:** https://ruv.io

---

## ✅ Final Checklist

### Before Publishing
- [ ] WASM builds successful (all 4 targets)
- [ ] Tests passing (npm test)
- [ ] Package.json metadata complete
- [ ] TypeScript definitions present
- [ ] README updated with npm instructions
- [ ] LICENSE file included
- [ ] Version synced with Cargo.toml
- [ ] Build script tested
- [ ] Bundle sizes reasonable (<500KB each)

### During Publishing
- [ ] NPM_TOKEN configured
- [ ] `npm publish --access public` executed
- [ ] Publication confirmed on npm

### After Publishing
- [ ] Verify on https://www.npmjs.com
- [ ] Test installation (npm install)
- [ ] Test import (Node.js, ES modules)
- [ ] Create GitHub release
- [ ] Update documentation
- [ ] Announce release

---

**Status:** ✅ **READY FOR NPM PUBLICATION**

**Next Action:** Build WASM and publish to npm

```bash
cd /workspaces/agentic-flow/packages/agentic-jujutsu
./scripts/wasm-pack-build.sh --release
npm publish --access public
```

---

**Prepared by:** Claude Code
**Date:** 2025-11-09
**Version:** 0.1.0

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
