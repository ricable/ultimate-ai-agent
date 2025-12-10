# TITAN Scripts Directory

Automation scripts for TITAN RAN project setup, validation, and deployment.

## 🔐 Authentication & Setup

### `setup-oauth.sh`
Interactive wizard for configuring OAuth-based authentication.

```bash
# Run interactive setup
npm run auth:setup
# or
./scripts/setup-oauth.sh
```

**Features:**
- Guides through Claude Code login
- Generates and saves OAuth tokens
- Configures Google AI Pro OAuth
- Sets AI strategy preferences
- Updates `.env` automatically

**Requirements:**
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- Active subscriptions (Claude Pro/Team, Google AI Pro/Ultra)

---

### `validate-auth.sh`
Validates authentication configuration and checks for issues.

```bash
# Validate configuration
npm run auth:validate
# or
./scripts/validate-auth.sh
```

**Checks:**
- ✅ Claude Code OAuth token
- ✅ Google OAuth credentials
- ✅ Latest models configured
- ✅ AI strategy settings
- ✅ E2B and other services

**Exit Codes:**
- `0` - All checks passed
- `1` - Errors found (setup incomplete)

---

## 🚀 Deployment Scripts

### `start-local.sh`
Start TITAN in local development mode.

```bash
npm run start:local
# or
./scripts/start-local.sh
```

**Environment:** Mac Silicon, direct execution

---

### `start-devpod.sh`
Start TITAN in DevPod containerized environment.

```bash
npm run start:devpod
# or
./scripts/start-devpod.sh
```

**Environment:** Docker containers, isolated workspace

---

### `test-integration.sh`
Run integration tests for all AI providers.

```bash
npm run test:integration
# or
./scripts/test-integration.sh
```

**Tests:**
- Claude Code integration
- Google Gemini integration
- E2B sandbox connectivity
- Multi-provider consensus

---

## 🔍 Pre-Flight Checks

### Recommended Workflow

```bash
# 1. First-time setup
npm run auth:setup

# 2. Validate configuration
npm run auth:validate

# 3. Pre-flight (validate + build)
npm run auth:preflight

# 4. Start TITAN
npm run start:local

# 5. Test integration
npm run test:integration
```

---

## 📝 Script Maintenance

### Adding New Scripts

1. Create script in `scripts/` directory
2. Make executable: `chmod +x scripts/your-script.sh`
3. Add to `package.json` scripts section
4. Document in this README
5. Test in both local and DevPod modes

### Best Practices

```bash
# Use absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set error handling
set -e  # Exit on error

# Use colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}✓${NC} Success message"
echo -e "${RED}✗${NC} Error message"
```

---

## 🛠️ Troubleshooting

### Script Permissions

```bash
# If script won't execute
chmod +x scripts/*.sh

# Check permissions
ls -la scripts/
```

### Environment Loading

```bash
# If .env not loading
source config/.env

# Verify variables
echo $CLAUDE_CODE_OAUTH_TOKEN
echo $GOOGLE_OAUTH_CLIENT_ID
```

### macOS Specific Issues

```bash
# If sed errors (BSD vs GNU)
# Scripts use sed -i.bak for compatibility

# Clean up backup files
find config -name "*.bak" -delete
```

---

## 📚 Related Documentation

- [Authentication Guide](../docs/AUTH.md) - Complete authentication setup (OAuth, API keys, free tier)
- [Quick Start](../docs/QUICK-START.md) - 5-minute setup guide
- [Multi-Provider Setup](../docs/MULTI-PROVIDER-SETUP.md) - AI provider configuration

---

## 🔗 Quick Links

**Setup Commands:**
```bash
npm run auth:setup      # Interactive OAuth setup
npm run auth:validate   # Validate configuration
npm run auth:preflight  # Pre-flight check
```

**Deployment Commands:**
```bash
npm run start:local     # Start local mode
npm run start:devpod    # Start DevPod mode
npm run docker:up       # Start Docker Compose
```

**Testing Commands:**
```bash
npm run test:integration  # Integration tests
npm run ui:integration    # UI integration demo
npm run env:validate      # Environment validation
```
