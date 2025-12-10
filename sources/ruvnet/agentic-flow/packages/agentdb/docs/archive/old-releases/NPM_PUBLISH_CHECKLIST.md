# AgentDB v2.0.0-alpha.1 - NPM Publish Checklist

**Date:** 2025-11-28
**Version:** 2.0.0-alpha.1
**Status:** 🟡 Pre-Release (Alpha)

---

## ✅ Pre-Publish Validation (Required)

### 1. **Code Quality** ✅

- [x] TypeScript compilation successful (`npm run build`)
- [x] All linting rules pass
- [x] No console.log or debug statements in production code
- [x] Source maps generated

**Command:**
```bash
npm run build
```

**Expected:** ✅ Clean build, no errors

---

### 2. **Test Suite** ✅

- [x] Unit tests pass (`npm run test:unit`)
- [x] Backend tests pass (`npm run test:backend`)
- [x] API compatibility tests pass
- [x] Browser bundle tests pass
- [x] Test coverage > 80%

**Command:**
```bash
npm run test:unit
npm run test:backend
```

**Expected:** ✅ All tests passing

---

### 3. **Docker Validation** 🟡 IN PROGRESS

- [ ] Docker build succeeds (`npm run docker:build`)
- [ ] Docker test suite passes (`npm run docker:test`)
- [ ] Package validation passes (`npm run docker:package`)
- [ ] CLI validation passes (`npm run docker:cli`)
- [ ] MCP server validation passes (`npm run docker:mcp`)
- [ ] Production image builds (`npm run docker:prod`)

**Command:**
```bash
npm run docker:test
```

**Expected:** ✅ All 8 stages pass

---

### 4. **Package Contents** 🔴 TODO

- [ ] package.json version updated to `2.0.0-alpha.1`
- [ ] README.md is comprehensive and up-to-date
- [ ] LICENSE file present (MIT)
- [ ] CHANGELOG.md documents v2 changes
- [ ] `files` field includes all necessary artifacts
- [ ] No sensitive files included (.env, credentials, etc.)

**Command:**
```bash
npm pack
tar -tzf agentdb-*.tgz | less
```

**Expected:** Only dist/, src/, scripts/, README, LICENSE

---

### 5. **Dependencies** ✅

- [x] Core dependencies locked in package.json
- [x] Optional dependencies (@ruvector/core, @ruvector/gnn, better-sqlite3)
- [x] DevDependencies only for development
- [x] No unnecessary dependencies
- [x] Security vulnerabilities checked (`npm audit`)

**Command:**
```bash
npm audit
npm ls --depth=0
```

**Expected:** No critical vulnerabilities

---

### 6. **Documentation** 🔴 TODO

- [ ] README.md includes:
  - [ ] Installation instructions
  - [ ] Quick start guide
  - [ ] Multi-backend usage examples
  - [ ] GNN feature documentation
  - [ ] Docker deployment guide
  - [ ] MCP server setup
  - [ ] Migration guide from v1
- [ ] API documentation complete
- [ ] Changelog updated with v2 features
- [ ] Examples directory with working demos

**Location:** `/workspaces/agentic-flow/packages/agentdb/README.md`

---

### 7. **Performance Benchmarks** 🔴 TODO

- [ ] Benchmark suite runs successfully
- [ ] Performance metrics documented
- [ ] Comparison with v1 documented
- [ ] Backend comparison (SQLite vs HNSWLib vs RuVector)

**Command:**
```bash
npm run benchmark:full
npm run benchmark:backends
```

**Expected:** Documented results in benchmarks/results/

---

### 8. **CLI Functionality** ✅

- [x] `agentdb init` creates database
- [x] `agentdb mcp start` launches MCP server
- [x] `agentdb --version` shows correct version
- [x] `agentdb --help` displays help

**Command:**
```bash
node dist/cli/agentdb-cli.js --version
node dist/cli/agentdb-cli.js --help
```

**Expected:** v2.0.0-alpha.1, help text displayed

---

### 9. **Browser Compatibility** ✅

- [x] Browser bundle builds (`npm run build:browser`)
- [x] Bundle size acceptable (< 100KB minified)
- [x] WASM modules load correctly
- [x] sql.js integration works

**Command:**
```bash
npm run build:browser
ls -lh dist/browser/
```

**Expected:** agentdb-browser.min.js created

---

### 10. **MCP Integration** ✅

- [x] MCP server starts without errors
- [x] MCP tools exposed correctly
- [x] Authentication working
- [x] Real-time subscriptions functional

**Command:**
```bash
timeout 5 node dist/cli/agentdb-cli.js mcp start
```

**Expected:** Server starts, timeout after 5s (normal)

---

## 🚀 NPM Publish Steps

### Step 1: Dry Run (Required)

```bash
npm publish --dry-run
```

**Expected:** Package preview, no errors

---

### Step 2: Tag Creation (Alpha Release)

```bash
git tag -a v2.0.0-alpha.1 -m "AgentDB v2.0.0-alpha.1 - Multi-Backend Architecture"
git push origin v2.0.0-alpha.1
```

---

### Step 3: Publish to NPM (Alpha Tag)

```bash
npm publish --tag alpha --access public
```

**Expected:** Published to https://www.npmjs.com/package/agentdb/v/2.0.0-alpha.1

---

### Step 4: Verify Installation

```bash
npm install agentdb@alpha
node -e "const agentdb = require('agentdb'); console.log('✅ Package works')"
```

---

### Step 5: Update Documentation

- [ ] Update GitHub README with alpha notice
- [ ] Add migration guide link
- [ ] Update examples repository
- [ ] Announce on GitHub Discussions

---

## 📋 Post-Publish Validation

### 1. **NPM Package Page**

- [ ] Visit https://www.npmjs.com/package/agentdb
- [ ] README renders correctly
- [ ] Version shows `2.0.0-alpha.1`
- [ ] Tags show `alpha` and `latest` (if appropriate)

---

### 2. **Installation Test**

```bash
mkdir /tmp/test-agentdb-alpha
cd /tmp/test-agentdb-alpha
npm init -y
npm install agentdb@alpha
node -e "const agentdb = require('agentdb'); console.log('Version:', agentdb.version)"
```

**Expected:** v2.0.0-alpha.1

---

### 3. **Docker Hub (Optional)**

If publishing Docker images:

```bash
docker tag agentdb-production:latest ruvnet/agentdb:2.0.0-alpha.1
docker tag agentdb-production:latest ruvnet/agentdb:alpha
docker push ruvnet/agentdb:2.0.0-alpha.1
docker push ruvnet/agentdb:alpha
```

---

### 4. **GitHub Release**

- [ ] Create GitHub release for v2.0.0-alpha.1
- [ ] Attach tarball (agentdb-2.0.0-alpha.1.tgz)
- [ ] Include release notes
- [ ] Mark as "pre-release"

---

## ⚠️ Known Issues (Alpha Release)

### Critical Gaps

1. **No public benchmarks** - ann-benchmarks.com submission pending
2. **GNN performance claims unvalidated** - 150x speedup needs verification
3. **Optional dependencies** - @ruvector packages may not be publicly available
4. **Limited production testing** - Alpha quality, use with caution

### Workarounds

- Use SQLite or HNSWLib backends (stable)
- Optional GNN features require manual @ruvector installation
- Expect breaking changes before stable v2.0.0

---

## 🎯 Success Criteria

**Minimum Requirements for Alpha:**

- [x] TypeScript builds successfully
- [x] Core tests pass
- [ ] Docker tests pass ← **CURRENT TASK**
- [ ] Package installs via npm
- [ ] Basic CLI functionality works
- [ ] Documentation covers new features

**Recommended Before Stable v2.0.0:**

- [ ] Public benchmarks published
- [ ] 10+ production deployments
- [ ] Community feedback incorporated
- [ ] Security audit complete
- [ ] Performance claims validated

---

## 📞 Support & Feedback

**Issues:** https://github.com/ruvnet/agentic-flow/issues
**Discussions:** https://github.com/ruvnet/agentic-flow/discussions
**Email:** support@agentdb.ruv.io
**Discord:** https://discord.gg/agentdb (if available)

---

## 🔐 Publishing Credentials

**NPM Account:** (Verify 2FA enabled)
**NPM Token:** (Stored in GitHub Secrets: `NPM_TOKEN`)
**GitHub Token:** (For releases)
**Docker Hub:** (Optional, for container images)

---

## ✅ Final Checklist

**Before running `npm publish --tag alpha`:**

- [ ] All tests pass (Docker + npm)
- [ ] Version bumped to 2.0.0-alpha.1
- [ ] CHANGELOG.md updated
- [ ] README.md comprehensive
- [ ] Git tag created
- [ ] No uncommitted changes
- [ ] Branch: `release/v2.0.0-alpha`
- [ ] Dry run successful

**Ready to publish:** 🔴 NO (Docker tests pending)

---

**Last Updated:** 2025-11-28
**Next Review:** After Docker tests complete
