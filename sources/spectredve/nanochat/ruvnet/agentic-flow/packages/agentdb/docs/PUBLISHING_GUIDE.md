# AgentDB Publishing Guide

## Publishing Strategy: Alpha → Beta → Stable

AgentDB v2.0 uses npm distribution tags to safely roll out major updates while keeping existing users on stable versions.

## Version Tags Explained

- **`latest`** - Stable production version (current users get this by default)
- **`alpha`** - Early preview for testing new features (v2.0.0-alpha.x)
- **`beta`** - Release candidate, feature-complete (v2.0.0-beta.x)

## Publishing Workflow

### 1. Publish Alpha Release

```bash
# Ensure you're on the correct branch
git checkout claude/review-ruvector-integration-01RCeorCdAUbXFnwS4BX4dZ5

# Build the package
npm run build

# Test locally first
npm pack
# This creates agentdb-2.0.0-alpha.1.tgz - test this in another project

# Publish to npm with 'alpha' tag
npm publish --tag alpha

# Verify it worked
npm view agentdb dist-tags
# Should show: { latest: '1.x.x', alpha: '2.0.0-alpha.1' }
```

**Who gets it:**
- Explicitly install: `npm install agentdb@alpha`
- Default installs (`npm install agentdb`) still get the stable version

### 2. Iterate on Alpha

For bug fixes or improvements during alpha testing:

```bash
# Update version in package.json
# 2.0.0-alpha.1 → 2.0.0-alpha.2

npm run build
npm publish --tag alpha
```

### 3. Promote to Beta

When alpha is stable and feature-complete:

```bash
# Update version in package.json
# 2.0.0-alpha.2 → 2.0.0-beta.1

npm run build
npm publish --tag beta
```

**Who gets it:**
- Explicitly install: `npm install agentdb@beta`
- Can also still get alpha: `npm install agentdb@alpha`

### 4. Promote to Stable (Latest)

When beta testing is complete and you're ready for general availability:

```bash
# Update version in package.json
# 2.0.0-beta.1 → 2.0.0

npm run build

# Publish with 'latest' tag (makes it the default)
npm publish --tag latest

# Verify
npm view agentdb dist-tags
# Should show: { latest: '2.0.0', alpha: '2.0.0-alpha.2', beta: '2.0.0-beta.1' }
```

**Who gets it:**
- Everyone! Default installs now get v2.0.0

## Testing Published Packages

### Test Alpha Locally

```bash
# In a test project
npm install agentdb@alpha

# Verify version
npm list agentdb
# Should show: agentdb@2.0.0-alpha.1

# Test the package
node -e "import('agentdb').then(m => console.log(m.createDatabase))"
```

### Test MCP Integration

```bash
# Test MCP server with alpha version
npx agentdb@alpha mcp start

# Or add to Claude config:
{
  "mcpServers": {
    "agentdb-alpha": {
      "command": "npx",
      "args": ["agentdb@alpha", "mcp", "start"]
    }
  }
}
```

## Rollback Strategy

If you need to rollback or fix the 'latest' tag:

```bash
# Point 'latest' back to previous stable version
npm dist-tag add agentdb@1.3.0 latest

# Verify
npm view agentdb dist-tags
```

## Version Naming Conventions

- **Alpha**: `2.0.0-alpha.1`, `2.0.0-alpha.2`, etc.
  - For initial testing, may have breaking changes
  - Not recommended for production

- **Beta**: `2.0.0-beta.1`, `2.0.0-beta.2`, etc.
  - Feature-complete, API stable
  - Testing for production readiness

- **Release Candidate**: `2.0.0-rc.1`, `2.0.0-rc.2`, etc. (optional)
  - Final testing before stable release
  - Only critical bug fixes

- **Stable**: `2.0.0`
  - Production-ready
  - Becomes the new 'latest'

## Communicating with Users

### Alpha Announcement

```markdown
🚀 **AgentDB v2.0.0-alpha.1 is now available for testing!**

Early adopters can try the new features:
- 150x faster vector search with RuVector
- 25 latent space simulations
- Graph Neural Networks with 8-head attention
- 97.9% self-healing

Install: `npm install agentdb@alpha`

⚠️ Alpha version - not recommended for production
📖 Feedback: https://github.com/ruvnet/agentic-flow/issues
```

### Beta Announcement

```markdown
🎉 **AgentDB v2.0.0-beta.1 - Release Candidate**

Feature-complete and API-stable. Testing for production readiness.

Install: `npm install agentdb@beta`

📊 All features validated through 24 simulation iterations
✅ Zero regressions, 100% backward compatibility
📖 Migration guide: docs/MIGRATION_v2.0.md
```

### Stable Release Announcement

```markdown
🎊 **AgentDB v2.0.0 is now STABLE!**

Install: `npm install agentdb@latest` (or just `npm install agentdb`)

🚀 150x faster with RuVector
🧠 Self-learning with GNN attention
🔄 97.9% self-healing over 30 days
🎮 25 empirically validated simulations

📖 Full changelog: CHANGELOG.md
📚 Migration guide: docs/MIGRATION_v2.0.md
```

## Pre-Publication Checklist

Before publishing any version:

- [ ] All tests passing (`npm test`)
- [ ] TypeScript compiles without errors (`npm run build`)
- [ ] Documentation updated (README.md, CHANGELOG.md)
- [ ] Version bumped in package.json
- [ ] Git committed and tagged
- [ ] Tested locally with `npm pack`
- [ ] Reviewed breaking changes (if any)
- [ ] Migration guide written (for major versions)

## Current Status

**Version:** 2.0.0-alpha.1
**Tag:** alpha
**Status:** Ready for early adopter testing

**Next Steps:**
1. Publish alpha: `npm publish --tag alpha`
2. Gather feedback from early adopters
3. Fix critical issues in alpha.2, alpha.3, etc.
4. Promote to beta when stable
5. Final testing in beta
6. Promote to latest when production-ready

## Useful Commands

```bash
# View all published versions
npm view agentdb versions

# View distribution tags
npm view agentdb dist-tags

# Add/change a dist-tag
npm dist-tag add agentdb@2.0.0-alpha.1 alpha

# Remove a dist-tag
npm dist-tag rm agentdb alpha

# Deprecate a version
npm deprecate agentdb@2.0.0-alpha.1 "Use 2.0.0-alpha.2 instead"

# Unpublish (only within 72 hours, use with caution!)
npm unpublish agentdb@2.0.0-alpha.1
```

## Resources

- [npm dist-tag documentation](https://docs.npmjs.com/cli/v8/commands/npm-dist-tag)
- [Semantic Versioning](https://semver.org/)
- [npm Publishing Guide](https://docs.npmjs.com/packages-and-modules/contributing-packages-to-the-registry)

---

**Remember:** The `latest` tag is sacred. Only promote to `latest` when you're confident the release is production-ready and won't break existing users' workflows.
