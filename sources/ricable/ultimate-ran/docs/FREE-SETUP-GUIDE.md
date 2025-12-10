# FREE AI Setup Guide - No Costs, Use Your Email

Since you have **Claude MAX** and **Google AI Pro** subscriptions, here's what you need to know and how to get TITAN running **completely FREE**.

---

## 🚨 Important: Your Subscriptions vs API Access

### What You Currently Have:

✅ **Claude MAX subscription** ($20-40/month)
- Access to: claude.ai web interface
- Benefits: Unlimited messages, priority access, Claude 3.5 Sonnet
- **Does NOT include**: Programmatic API access

✅ **Google AI Pro subscription** (Gemini Advanced)
- Access to: gemini.google.com web interface
- Benefits: Advanced Gemini model, Google Workspace integration
- **Does NOT include**: Programmatic API access

### What TITAN Needs:

❌ Your subscriptions **cannot** be used by TITAN directly
- TITAN needs to call AI APIs programmatically (not web chat)
- Web subscriptions ≠ API access (different products)

✅ **But there's GREAT news**: You can get everything **FREE**!

---

## 💰 FREE Setup - $0/Month

You can run TITAN with **zero additional costs** using free API tiers:

### Option 1: Google AI Studio (100% FREE)

**Google AI Studio API is FREE** - you don't need Gemini Advanced!

✅ **Completely free** (no credit card required)
✅ **15 requests/minute** (enough for development)
✅ **1,500 requests/day** (45,000/month)
✅ **Same models** as Gemini Advanced

**Setup Steps:**

1. **Get FREE Google AI API Key** (using your cedricable@gmail.com):
   ```
   → Visit: https://aistudio.google.com/app/apikey
   → Sign in with: cedricable@gmail.com
   → Click "Create API Key"
   → Copy the key (starts with AIzaSy...)
   ```

2. **Add to your config**:
   ```bash
   cd /Users/cedric/dev/ultimate-ran-1
   cp config/.env.template config/.env
   nano config/.env
   ```

3. **Paste your FREE API key**:
   ```bash
   GOOGLE_AI_API_KEY=AIzaSy-your-actual-key-here
   GOOGLE_AI_MODEL=gemini-2.0-flash-exp
   ```

4. **Run TITAN**:
   ```bash
   npm run build
   npm run agui:start
   npm run agui:frontend
   ```

**Cost: $0/month** ✅

---

### Option 2: Add Claude via OpenRouter ($1 Free Credit)

If you want both Claude + Gemini for consensus:

1. **Get OpenRouter Account** (using cedricable@gmail.com):
   ```
   → Visit: https://openrouter.ai/keys
   → Sign in with Google: cedricable@gmail.com
   → Get $1 FREE credit automatically
   → Create API key
   ```

2. **Add to your config/.env**:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

   # Use both for consensus
   AGENTIC_FLOW_PRIMARY_PROVIDER=google
   AGENTIC_FLOW_SECONDARY_PROVIDER=openrouter
   AGENTIC_FLOW_STRATEGY=consensus
   ```

**Cost: $0 for first ~300 requests** ✅

---

## 🎯 Recommended FREE Setup (Zero Cost)

Here's your complete FREE configuration using just Google AI:

```bash
# config/.env

# ==================== FREE GOOGLE AI ====================
GOOGLE_AI_API_KEY=AIzaSy-your-FREE-key-from-aistudio
GOOGLE_AI_MODEL=gemini-2.0-flash-exp

# Use only Google AI (no secondary provider needed)
AGENTIC_FLOW_PRIMARY_PROVIDER=google
AGENTIC_FLOW_SECONDARY_PROVIDER=none
AGENTIC_FLOW_STRATEGY=single

# E2B Sandboxes (FREE 100 hours/month)
E2B_API_KEY=e2b_your-FREE-key

# Platform config
PLATFORM_ARCH=arm64
RUNTIME_MODE=local
```

**Total Monthly Cost: $0** 🎉

---

## 🔄 If You Want to Use Your Existing Subscriptions

Since you're already paying for Claude MAX and Gemini Advanced, here's an experimental way to use them (not recommended for production):

### Browser Automation Approach

```bash
# config/.env

USE_WEB_AUTOMATION=true
CLAUDE_EMAIL=cedricable@gmail.com
GEMINI_EMAIL=cedricable@gmail.com

# You'll need to extract session cookies:
# 1. Login to claude.ai
# 2. Press F12 → Application → Cookies → claude.ai
# 3. Copy 'sessionKey' value
CLAUDE_SESSION_COOKIE=your-session-cookie-here

# Repeat for gemini.google.com
GEMINI_SESSION_COOKIE=your-session-cookie-here
```

**Pros:**
- ✅ Uses your existing subscriptions (no extra cost)
- ✅ Access to Claude MAX features

**Cons:**
- ❌ Session cookies expire (need to refresh weekly)
- ❌ Rate limited and slower
- ❌ Not suitable for production
- ❌ Requires browser automation setup

**Verdict:** Not worth the hassle. Use FREE Google AI Studio instead!

---

## 📊 Cost Comparison

| Setup | Monthly Cost | Requests/Month | Best For |
|-------|--------------|----------------|----------|
| **Google AI Studio (FREE)** | **$0** | **45,000** | Development & Testing ✅ |
| OpenRouter ($1 credit) | $0 (one-time) | ~300 | Initial testing |
| Google AI + Anthropic API | ~$3-5 | Unlimited | Production |
| Web Automation (your subscriptions) | $0 | Limited | Not recommended ❌ |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Get FREE Google AI Key

```bash
# Open in browser
open https://aistudio.google.com/app/apikey

# Sign in with: cedricable@gmail.com
# Click "Create API Key"
# Copy the key
```

### Step 2: Configure TITAN

```bash
cd /Users/cedric/dev/ultimate-ran-1
cp config/.env.template config/.env

# Edit config/.env and add your FREE key
nano config/.env
```

Paste this:
```bash
GOOGLE_AI_API_KEY=AIzaSy-YOUR-ACTUAL-KEY-HERE
GOOGLE_AI_MODEL=gemini-2.0-flash-exp
AGENTIC_FLOW_PRIMARY_PROVIDER=google
AGENTIC_FLOW_STRATEGY=single
```

### Step 3: Build & Run

```bash
npm run build
npm run agui:start

# In new terminal:
npm run agui:frontend
```

### Step 4: Initialize AI Swarm

1. Navigate to http://localhost:8080
2. Find "🤖 AI Agent Swarm" section
3. Click "Initialize Swarm"
4. Watch your FREE agents spawn!

---

## ❓ FAQ

### Q: Why can't I use my Claude MAX subscription?

**A:** Claude MAX gives you unlimited access to **claude.ai website**, not the **Anthropic API**. These are separate products with separate billing. Think of it like:
- Netflix subscription = streaming on netflix.com
- Netflix API = programmatic access for apps (different product)

### Q: Why can't I use my Gemini Advanced subscription?

**A:** Same reason! Gemini Advanced is for **gemini.google.com website**. But good news: Google AI Studio API is **completely FREE** and gives you the same models!

### Q: Is the FREE tier good enough?

**A:** Absolutely!
- ✅ 1,500 requests/day = 45,000/month
- ✅ Same Gemini 2.0 Flash model
- ✅ 15 requests/minute
- ✅ Perfect for development and small deployments

### Q: When should I pay for Anthropic API?

**A:** Only if you need:
- High request volume (>50 req/min)
- Claude-specific features (tool use, advanced reasoning)
- Production-grade SLAs

For most development, FREE Google AI is perfect.

### Q: Can I get a refund on my subscriptions?

**A:** Your subscriptions are still useful:
- **Claude MAX**: Great for interactive development, asking questions
- **Gemini Advanced**: Access to latest Gemini models in browser

Keep them for your own use, run TITAN on FREE APIs!

---

## 📞 Need Help?

If you have issues getting your FREE setup working:

1. Check API key is correct (starts with `AIzaSy`)
2. Verify it's in `config/.env` (not `.env.template`)
3. Restart the server after changing config
4. Check logs: `npm run agui:start` (look for connection errors)

**You should NOT need to pay anything extra to run TITAN!** 🎉

---

## ✅ Summary

**What You Should Do:**

1. ✅ Get FREE Google AI API key from AI Studio
2. ✅ Use it in TITAN ($0 cost)
3. ✅ Keep your Claude MAX & Gemini Advanced for personal use
4. ✅ Enjoy consensus-based AI swarm at zero cost!

**What You Should NOT Do:**

1. ❌ Try to extract subscription credentials (complex, not worth it)
2. ❌ Pay for Anthropic API unless you need it
3. ❌ Cancel your subscriptions (they're great for interactive work!)

**Your Setup:**
- TITAN: FREE Google AI Studio API
- Your work: Claude MAX & Gemini Advanced web interfaces
- Total new cost: **$0** ✨

---

**Happy Optimizing with FREE AI! 🚀**
