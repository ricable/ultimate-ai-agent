# TITAN Quick Start Guide 🚀

**RECOMMENDED: See [Authentication Guide](AUTH.md) for all setup options:**
- OAuth Subscriptions (75-83% cost savings)
- Free Tier ($0/month with Google AI Studio)
- API Keys (legacy, pay-per-use)

**Get TITAN running in under 5 minutes with multi-provider AI support.**

## 🎯 Choose Your Setup Method

### Option A: OAuth Subscriptions or Free Tier (Recommended)

**See the complete [Authentication Guide](AUTH.md) for:**
- OAuth Subscriptions (75-83% savings)
- Free Tier ($0/month)
- API Keys (legacy)

**Quick OAuth Setup:**
```bash
npm run auth:setup
npm run auth:validate
npm run start:local
```

---

### Option B: Quick API Key Setup (Legacy)

## Prerequisites Checklist

- [ ] Mac Silicon (M1/M2/M3/M4)
- [ ] Node.js 18+ installed
- [ ] API keys ready (Claude, Gemini, E2B, OpenRouter)
- [ ] Docker Desktop running (for DevPod mode only)

## Step 1: Get API Keys (2 minutes)

### Required Keys

1. **Claude Code PRO MAX** (Primary AI)
   - Visit: https://console.anthropic.com/settings/keys
   - Click "Create Key"
   - Copy key starting with `sk-ant-`

2. **Google AI Studio** (Secondary AI)
   - Visit: https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy key starting with `AIza`

3. **E2B Sandboxes** (Safety validation)
   - Visit: https://e2b.dev/docs
   - Sign up and get key
   - Copy key starting with `e2b_`

4. **OpenRouter** (Optional fallback)
   - Visit: https://openrouter.ai/keys
   - Create account and get key
   - Copy key starting with `sk-or-`

## Step 2: Configure Environment (1 minute)

```bash
# Navigate to project
cd /Users/cedric/dev/ultimate-ran-1

# Copy environment template
cp config/.env.template config/.env

# Edit with your keys
nano config/.env
```

**Paste your keys:**

```bash
ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE
GOOGLE_AI_API_KEY=AIzaSy-YOUR-KEY-HERE
E2B_API_KEY=e2b_YOUR-KEY-HERE
OPENROUTER_API_KEY=sk-or-YOUR-KEY-HERE  # Optional

# Recommended strategy
AGENTIC_FLOW_STRATEGY=consensus
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

## Step 3: Install Dependencies (1 minute)

```bash
# Install all packages
npm install

# Build TypeScript
npm run build
```

## Step 4: Test Integration (30 seconds)

```bash
# Verify all API connections
npm run test:integration
```

**Expected output:**
```
✓ Claude Code PRO MAX working
✓ Google AI Pro (Gemini) working
✓ E2B Sandboxes working
✓ OpenRouter working
✓ agentic-flow installed
✓ claude-flow installed
✓ Docker running

Passed: 7 / 7 tests
🎉 All tests passed!
```

## Step 5A: Run Locally (30 seconds)

**Recommended for development:**

```bash
npm run start:local
```

You should see:
```
✓ TITAN is running!
  UI Dashboard:  http://localhost:3000
  AG-UI Server:  http://localhost:3001
  QUIC Port:     4433
```

**Open in browser:**
```bash
open http://localhost:3000
```

## Step 5B: Run in DevPod (2 minutes)

**Recommended for production:**

```bash
# Install DevPod (first time only)
brew install devpod

# Start DevPod environment
npm run start:devpod
```

This will:
1. Create Docker workspace
2. Install dependencies
3. Start all services
4. Open VS Code

**Access the workspace:**
```bash
devpod ssh titan-ran
```

## 🎯 What's Running?

Once started, you'll have:

| Service | Port | Description |
|---------|------|-------------|
| **UI Dashboard** | 3000 | Main control interface |
| **AG-UI Server** | 3001 | Real-time agent monitoring |
| **QUIC Transport** | 4433 | Agent communication |
| **PostgreSQL** | 5432 | AgentDB memory (DevPod only) |
| **Redis** | 6379 | Caching (DevPod only) |

## 🧪 Try It Out

### Example 1: Optimize a Cell

```bash
# Open dashboard
open http://localhost:3000

# Or use CLI
npx claude-flow@alpha swarm spawn \
  --intent "Optimize CELL_001 SINR"
```

### Example 2: Run AI Consensus

```bash
# Test consensus mode with both AIs
npm run ui:integration
```

### Example 3: Monitor Agents

```bash
# Real-time monitoring
open http://localhost:3001

# Or CLI
npx claude-flow@alpha swarm status
```

## 🔍 Verify Everything Works

Run the integration examples:

```bash
# All examples
npm run ui:integration

# Specific examples
EXAMPLE=1 npm run ui:integration  # Claude only
EXAMPLE=2 npm run ui:integration  # Gemini only
EXAMPLE=3 npm run ui:integration  # Consensus (recommended)
EXAMPLE=4 npm run ui:integration  # Full dashboard
```

## 🛑 Stop TITAN

### Local Mode
```bash
# Press Ctrl+C in the terminal
```

### DevPod Mode
```bash
devpod stop titan-ran
```

### Docker Compose Mode
```bash
npm run docker:down
```

## ⚡ Quick Commands Reference

```bash
# Start
npm run start:local          # Local Mac Silicon
npm run start:devpod         # DevPod environment
npm run docker:up            # Docker Compose

# Test
npm run test:integration     # Test all APIs
npm run ui:integration       # Test AI integration
npm test                     # Run test suite

# Monitor
npm run agui:frontend        # Open monitoring dashboard
npm run swarm:spawn          # Spawn new swarm
npm run hive:status          # Check hive mind status

# Stop
Ctrl+C                       # Local mode
devpod stop titan-ran        # DevPod mode
npm run docker:down          # Docker Compose
```

## 🎨 AI Strategies

Change strategy in `config/.env`:

```bash
# Highest confidence (both AIs must agree)
AGENTIC_FLOW_STRATEGY=consensus

# Fast & reliable (Claude primary)
AGENTIC_FLOW_STRATEGY=claude_primary

# Multimodal analysis (Gemini primary)
AGENTIC_FLOW_STRATEGY=gemini_primary

# Maximum speed (parallel execution)
AGENTIC_FLOW_STRATEGY=parallel
```

## 🆘 Troubleshooting

### API Key Errors

```bash
# Verify keys are set
echo $ANTHROPIC_API_KEY
echo $GOOGLE_AI_API_KEY

# Or check .env file
cat config/.env
```

### Port Already in Use

```bash
# Find what's using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Can't Connect to API

```bash
# Test connection manually
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages
```

### Docker Not Running

```bash
# Check Docker status
docker info

# Start Docker Desktop
open /Applications/Docker.app
```

## 📚 Next Steps

1. ✅ **Setup Complete** - TITAN is running!
2. 📖 **Read Full Guide** - [docs/MULTI-PROVIDER-SETUP.md](MULTI-PROVIDER-SETUP.md)
3. 🎯 **Try Examples** - [src/ui/integrations/README.md](../src/ui/integrations/README.md)
4. 🏗️ **Build Features** - Start optimizing your RAN!
5. 📊 **Monitor Performance** - Use AG-UI dashboard

## 🎉 Success Indicators

You know it's working when you see:

- ✅ All integration tests pass
- ✅ Dashboard loads at http://localhost:3000
- ✅ AG-UI shows active agents
- ✅ Consensus mode returns recommendations
- ✅ No error messages in logs

**Congratulations! TITAN is ready to optimize your RAN! 🚀**

---

**Need Help?**
- Full documentation: [docs/MULTI-PROVIDER-SETUP.md](MULTI-PROVIDER-SETUP.md)
- Integration guide: [src/ui/integrations/README.md](../src/ui/integrations/README.md)
- GitHub issues: https://github.com/ricable/ultimate-ran/issues
