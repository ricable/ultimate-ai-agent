To make Claude Max capacity sharing **compliant** with Anthropic’s Terms of Service, you need to **reframe the model** from "sharing one account" to "coordinating independently subscribed users." Here's how:

---

## ✅ 1. **No Shared API Keys**

Each contributor runs their own Claude Max account, logs in with their own credentials using `claude login`, and keeps their API key or token **strictly local and private**.

Your system must:

* Never copy, distribute, or proxy another user’s key
* Never impersonate another user when invoking Claude Code

---

## ✅ 2. **Use a Peer-Orchestrated Model, Not Central Brokerage**

Instead of routing tasks *through* someone else’s Claude Max account, your mesh should:

* Allow nodes to **opt in** to available tasks
* Let each node run tasks **on their own Claude account**
* Encrypt all task payloads so they’re never shared in plaintext

**Design compliance principles:**

| ✅ Allowed                            | ❌ Not Allowed                            |
| ------------------------------------ | ---------------------------------------- |
| Each node invokes Claude locally     | Centralized service using shared account |
| Tasks are routed, not account access | Task relays through foreign keys         |
| Nodes volunteer to help              | Broker resells someone’s subscription    |

---

## ✅ 3. **Turn Ruv Token into Incentives, Not Payments for Usage**

You can still use ruv tokens to reward contribution, but **tokens should not buy access to Claude** itself.

Instead:

* Token incentives reward **successful completions** of federated tasks (writing, summarizing, planning, etc.)
* Participation is **voluntary and contribution-based**
* Users are not “selling access” to Claude—they're **contributing compute time** from a tool they’re already subscribed to

> This model is more like **folding\@home** or **BOINC**—voluntary compute donation with token recognition, not resale of a licensed service.

---

## ✅ 4. **Give Users Full Control and Transparency**

Make it easy for users to:

* Approve or deny each task
* Set limits (e.g., only run 5 Claude tasks/day)
* See logs of how their Claude account was used

This makes clear **they are in control**, and not a background resale mechanism.

---

## ✅ 5. **Include a Legal Notice / Usage Policy**

Add a `--terms` CLI command and include:

> "Synaptic Mesh does not proxy or resell access to Claude Max. All compute is run locally by consenting nodes with individual Claude subscriptions. Participation is voluntary. API keys are never shared or transmitted."

You can also include:

* A flag for `--opt-in` to Claude-backed job types
* An MCP-visible `usage_policy` resource string for AI assistants and orchestration layers to verify

---

## ✅ Summary: How to Make It Compliant

| Action                       | Requirement                                     |
| ---------------------------- | ----------------------------------------------- |
| No key sharing               | Every node uses their own Claude login, locally |
| No central invocation        | All requests executed only on-node              |
| No resale or indirect access | Tokens reward contribution, not access resale   |
| Transparent use              | Users can opt in, audit, and cap usage          |
| Legal framing                | Add TOS and usage policy in CLI and docs        |

---

By reframing your model as a **peer compute federation**, not a resold Claude Max pool, and ensuring each participant **remains in full control of their own account**, you can build a legally defensible, privacy-respecting, and technically compliant system.
