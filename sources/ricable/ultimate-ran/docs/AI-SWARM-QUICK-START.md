# AI Swarm Quick Start Guide
## Get Your Consensus-Based Agent Swarm Running in 5 Minutes

---

## ✅ What's Been Implemented

Your TITAN AG-UI now has **full AI Swarm Coordinator** integration with:

✨ **Both AI SDKs Available:**
- ✅ **Claude Agent SDK** - Advanced reasoning with tool use
- ✅ **Google Generative AI (Gemini)** - Multimodal analysis

🤖 **Multi-Agent Consensus:**
- ✅ 4-6 agents working together (mix of Claude + Gemini)
- ✅ Voting-based decisions with configurable thresholds
- ✅ Real-time agent status visualization
- ✅ Consensus history tracking

🎯 **Three Swarm Topologies:**
- ✅ **Consensus** - Democratic voting (recommended)
- ✅ **Hierarchical** - Leader-based coordination
- ✅ **Mesh** - Peer-to-peer collaboration

📊 **AG-UI Dashboard:**
- ✅ Live agent cards with status (Idle/Busy/Error)
- ✅ Consensus reasoning history with vote details
- ✅ One-click swarm initialization
- ✅ Real-time task monitoring

---

## 🚀 Quick Setup (5 Steps)

### **Step 1: Get API Keys**

**Claude API Key:**
```bash
# Sign up at: https://console.anthropic.com/
# Requires: Claude Code PRO MAX subscription ($20-40/month)
# Get your key: Settings → API Keys
```

**Google Gemini API Key:**
```bash
# Sign up at: https://aistudio.google.com/app/apikey
# Free tier: 15 requests/minute
# Click "Create API Key" → Copy
```

### **Step 2: Configure Environment**

```bash
# Create/edit config file
nano config/.env

# Add your keys:
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_AI_API_KEY=AIzaSy...
```

### **Step 3: Start TITAN Dashboard**

```bash
# Build the project
npm run build

# Start AG-UI server
npm run agui:start

# In a new terminal, open the dashboard
npm run agui:frontend
```

The dashboard will open at: **http://localhost:8080**

### **Step 4: Initialize the Swarm**

In the AG-UI Dashboard:

1. Scroll to **"🤖 AI Agent Swarm"** section
2. Select topology: **"Consensus (Recommended)"**
3. Click **"Initialize Swarm"**
4. Wait 3-5 seconds for agents to spawn

You should see:
```
✅ Total Agents: 4
✅ Active: 4
✅ Busy: 0
✅ Topology: CONSENSUS
```

### **Step 5: Watch Consensus in Action**

The swarm is now ready! It will automatically participate in:

- **Approval Validations** - Multi-agent voting on parameter changes
- **Cell Analysis** - Consensus on performance issues
- **Optimization Requests** - Collaborative parameter tuning

---

## 🎯 Test Consensus Voting

### Trigger a Consensus Decision

The swarm will vote on approval requests. To see it in action:

1. Navigate to **"Pending Approvals"** section
2. Click **"APPROVE"** or **"REJECT"** on any request
3. The swarm will:
   - Assign 4 validator agents (2 Claude + 2 Gemini)
   - Each agent analyzes the request independently
   - Agents cast votes with confidence scores
   - Final decision based on 75% threshold

4. Check **"Consensus Reasoning History"** section to see:
   - Which agents voted APPROVE/REJECT
   - Individual confidence levels
   - Final decision and average confidence

---

## 📊 Agent Card Explanation

Each agent card shows:

```
┌──────────────────────┐
│ claude-1      [IDLE] │ ← Agent ID & Status
│ ANALYZER             │ ← Specialized Role
│ Confidence: 85%      │ ← Current confidence
│ Tasks: 12            │ ← Completed tasks
└──────────────────────┘
```

**Status Colors:**
- 🟢 **IDLE** - Ready for tasks
- 🟡 **BUSY** - Currently processing
- 🔴 **ERROR** - API failure, click Reset

**Border Colors:**
- 🟣 **Purple** - Claude agent
- 🔵 **Blue** - Gemini agent

---

## 🔧 Available Topologies

### Consensus (Default)
```
Best for: Production deployments
Agents: 2 Claude + 2 Gemini
Threshold: 75% agreement required
Speed: Moderate (1.5-2.5s)
```

### Hierarchical
```
Best for: Fast prototyping
Agents: 1 Coordinator + 3 Workers
Threshold: Coordinator has veto power
Speed: Fast (800ms-1.5s)
```

### Mesh
```
Best for: Maximum redundancy
Agents: Up to 6 agents peer-to-peer
Threshold: Simple majority (50%)
Speed: Slower (2-3.5s)
```

---

## 💡 Understanding Consensus Results

Example consensus card:

```
┌────────────────────────────────────┐
│ APPROVED           3:42:15 PM      │
│ Confidence: 88.5% | Votes: 4 agents│
├────────────────────────────────────┤
│ claude-1    APPROVE (92%)          │
│ claude-2    APPROVE (85%)          │
│ gemini-1    APPROVE (87%)          │
│ gemini-2    REJECT (88%)           │
└────────────────────────────────────┘
```

**Decision Logic:**
- 3 out of 4 agents approved = 75%
- Meets consensus threshold → **APPROVED**
- Average confidence: 88.5%

---

## 🐛 Quick Troubleshooting

### "Swarm initialization failed"
```bash
# Check API keys are set
echo $ANTHROPIC_API_KEY
echo $GOOGLE_AI_API_KEY

# If empty, edit config/.env and restart server
```

### "All agents show ERROR status"
```
Possible causes:
1. Invalid API keys → Check console.anthropic.com
2. Rate limit exceeded → Wait 1 minute, click Reset
3. Network timeout → Check internet connection
```

### "No consensus history showing"
```
Trigger a decision by:
1. Approving/rejecting a pending approval
2. Or wait for automatic optimization tasks
3. History updates every 5 seconds
```

---

## 📖 Next Steps

Now that your swarm is running, explore:

1. **[Full AI Swarm Documentation](./AI-SWARM-INTEGRATION.md)** - Deep dive into consensus algorithms
2. **[Multi-Provider Setup](./MULTI-PROVIDER-SETUP.md)** - Advanced configuration options
3. **[Architecture Guide](./architecture-status-report.md)** - How TITAN agents work together

---

## 🎉 You're Ready!

Your AI swarm is now orchestrating consensus-based reasoning across multiple AI agents for robust network optimization decisions.

**Key Benefits:**
- ✅ Higher confidence through multi-agent validation
- ✅ Reduced hallucination risk (agents check each other)
- ✅ Transparent reasoning (see all votes)
- ✅ Fault tolerance (swarm continues if 1 agent fails)

**Happy Optimizing! 🚀**

---

## 📞 Support

Questions or issues?
- GitHub Issues: https://github.com/your-repo/issues
- Documentation: /docs/AI-SWARM-INTEGRATION.md

---

**TITAN RAN Platform v7.0.0-alpha.1**
*Powered by Claude 3.5 Sonnet + Google Gemini 2.0*
