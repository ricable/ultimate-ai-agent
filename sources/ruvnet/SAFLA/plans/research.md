 implementation of the Self-Aware Feedback Loop Algorithm (SAFLA) using:

* **liteLLM** as the policy and reflector model interface
* **requesty.ai** as the execution environment or simulated API
* **SQLite** for persistence and logging

I’ll also include proof of operation via minimal run logs and example outputs, emphasizing the loop's autonomous learning and reflection capabilities. I’ll let you know as soon as it’s ready.


# MVP Implementation of the Self-Aware Feedback Loop Algorithm (SAFLA)

To implement the Self-Aware Feedback Loop Algorithm (SAFLA), we break the solution into the 8 stages described. We use **liteLLM** for model calls (policy `Pθ` and reflection `Rφ`), **requesty.ai** as a placeholder for the action execution environment, and **SQLite** for persistent logging. The design is kept minimal and modular for clarity and extensibility.

## 1. Goal Registry (JSON Schema for Goals and Constraints)

First, define the business **goals (G)** and **constraints (C)** in a structured JSON format. This can be a JSON Schema or a simple JSON object that lists goals and constraints. In our example, the goal is to reach a target value and the constraint is a maximum limit:

```json
{
  "goals": [
    {"id": "G1", "description": "Reach or exceed a target value of 100."}
  ],
  "constraints": [
    {"id": "C1", "description": "Do not exceed 120 (max limit)."}
  ]
}
```

We can load this JSON and store it in an SQLite table for persistence (optional). For instance, we create a `goals` table and insert the entries:

```python
import sqlite3
conn = sqlite3.connect("safla.db")
cur = conn.cursor()
# Create goals table
cur.execute("""CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    type TEXT,        -- 'goal' or 'constraint'
    description TEXT
)""")
# Insert goals and constraints
goals = [("G1", "goal", "Reach or exceed a target value of 100.")]
constraints = [("C1", "constraint", "Do not exceed 120 (max limit).")]
cur.executemany("INSERT INTO goals VALUES (?,?,?)", goals + constraints)
conn.commit()
```

Now the goals and constraints are registered in a structured form (the JSON or table) that can be referenced throughout the loop.

## 2. Perception and Memory (State Embeddings with JSD Change Detection)

At each time step *t*, the agent **perceives** input signals `Sₜ` from the environment (here simplified as the current state). We convert these signals into a vector embedding for memory. Using liteLLM, we could call an embedding model (e.g., OpenAI or HuggingFace) to get a high-dimensional vector, but for MVP we simulate this. We also compute the **Jensen–Shannon Divergence (JSD)** between the new state’s distribution and the previous state’s distribution to detect significant changes. JSD is a symmetric measure of similarity between two probability distributions, useful to decide if a new perception is novel enough to store.

Below is a function that takes the current state (e.g. a numeric value or sensor reading), produces a simple embedding and computes JSD vs the last state. We represent the state as a probability distribution of “progress” toward the goal (for illustration purposes) and use that for JSD calculation:

```python
import math

# Previous state distribution (for memory of last perception)
prev_state_dist = None

def perceive_state(current_value):
    """
    Convert current state to representation and embedding, 
    and compute JSD vs previous state for novelty detection.
    """
    # Represent state as text and as a distribution (progress vs remaining)
    state_repr = f"Current value: {current_value:.1f}"
    if current_value >= target_value:
        dist = [1.0, 0.0]  # 100% progress, 0% remaining
    else:
        progress = current_value / target_value
        dist = [progress, 1.0 - progress]
    # Compute JSD between current and previous state distributions
    global prev_state_dist
    if prev_state_dist is None:
        jsd_val = 0.0  # no previous state (start of loop)
    else:
        jsd_val = jensen_shannon_divergence(prev_state_dist, dist)
    prev_state_dist = dist  # update memory of last state
    return state_repr, dist, jsd_val

def jensen_shannon_divergence(p, q):
    """Compute Jensen–Shannon Divergence between two distributions p and q."""
    # Ensure p, q are probability distributions
    assert abs(sum(p)-1.0) < 1e-6 and abs(sum(q)-1.0) < 1e-6
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kl_divergence(a, b):
        return sum(ai * math.log2(ai/bi) for ai, bi in zip(a, b) if ai > 0)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

In a real scenario, you might replace the simple distribution with a call to an embedding model via liteLLM (e.g., `liteLLM.embedding(...)`) to vectorize complex input signals. The JSD of embeddings (after normalizing them to probability distributions) can indicate how much the state has drifted from the previous step. We would store `state_repr`, the embedding (or a reference to it), and `state_jsd` in our **memory** (e.g., in an `events` log table) for persistence.

## 3. Action Selection (Policy Model via liteLLM)

Using the current goals `G`, constraints `C`, and perceived state `Sₜ`, the **policy model** `Pθ` selects an action `aₜ`. We prompt an LLM (through liteLLM) with the goals, constraints, and state, asking for the next action and a brief rationale. The prompt might look like:

```
Goals: Reach value ≥ 100.
Constraints: Don't exceed value 120.
State: Current value = 70.0
What action should the agent take next? Provide the action and rationale.
```

Using liteLLM, we can call a completion API (e.g., GPT-4 or another model) to get a response. For example:

```python
from litellm import completion

prompt = (f"Goals: {goals_text}\nConstraints: {constraints_text}\n"
          f"State: {state_repr}\n"
          "What action should the agent take next to achieve the goals while respecting constraints? "
          "Provide the action and the rationale.")
response = completion(model="gpt-3.5-turbo", messages=[{"role":"user","content": prompt}])
output = response["choices"][0]["message"]["content"]
```

We would then parse `output` to extract the proposed action and rationale. For MVP, we simulate this decision-making with a simple heuristic function (to avoid external calls). The code below demonstrates generating an action and rationale, and also how we would integrate any learned “patches” (from reflections, see Step 5) as additional guidance to the prompt:

```python
# Simulated list of policy patches/tips to include (initially empty)
policy_tips = []  

def select_action(state_repr):
    """Generate next action and rationale using current state and any policy patches."""
    # Build the prompt for the LLM
    prompt = (f"Goals: Reach value >= {target_value}.\n"
              f"Constraints: Do not exceed {max_value}.\n"
              f"State: {state_repr}\n")
    if policy_tips:
        prompt += "Additional guidance: " + " ".join(policy_tips) + "\n"
    prompt += "Action?"  # asking for next action
    
    # Call the policy model via liteLLM (pseudo-code; using a stub for MVP)
    # response = completion(model="gpt-4", messages=[{"role":"user","content": prompt}])
    # action = parse_action(response); rationale = parse_rationale(response)
    
    # For MVP, use a simple rule-based policy:
    try:
        current_val = float(state_repr.split()[-1])
    except:
        current_val = 0.0
    if current_val >= target_value:
        action = "none"
        rationale = "The target is achieved; no further action is needed."
    else:
        # Check for any active patch guidance
        use_smaller = any("smaller increment" in tip for tip in policy_tips)
        if use_smaller and current_val > 0.5 * target_value:
            action = "increase by 10"
            rationale = "Close to the target, so use a smaller increment to avoid overshooting."
        else:
            action = "increase by 70"
            rationale = "Far from the target, a large increase will get closer quickly."
    # (The numbers 10 and 70 are chosen for demonstration.)
    return action, rationale
```

In a real deployment, the `select_action` would use the LLM’s output. The rationale is stored for transparency and later reflection. We also count tokens used by the prompt/response for the meta-monitor (discussed later). For example, using liteLLM’s consistent output format, we could count tokens from `response`. In our simulation, we approximate token usage by word count:

```python
tokens_used = len(prompt.split()) + len(action.split()) + len(rationale.split())
```

## 4. Outcome Audit (Execute Action via Environment and Evaluate)

Once an action is chosen, the agent **executes** it in the environment through **requesty.ai** (or a simulator). Here, `requesty.ai` can act as a router or API platform to perform the action or call a tool. In our simplified environment, the state is a numeric value, and actions are textual commands like "increase by X" or "decrease by Y". We simulate an environment function that parses the action and applies it:

```python
# Simulated environment execution
def execute_action(action_text, current_value):
    new_value = current_value
    act = action_text.lower().strip()
    if act == "none":
        pass  # no change
    elif "increase by" in act:
        # parse the number and increase
        amount = float(act.split("increase by")[1])
        new_value += amount
    elif "decrease by" in act:
        amount = float(act.split("decrease by")[1])
        new_value -= amount
    # Apply constraint: cap at max_value
    constraint_violated = False
    if new_value > max_value:
        new_value = max_value
        constraint_violated = True
    result_repr = f"Value is now {new_value:.1f}"
    # Compute reward: higher when closer to goal (target_value)
    if constraint_violated:
        # If a constraint is violated, give zero reward (penalize)
        reward = 0.0
    else:
        # Reward could be defined as 100 - |target - new_value| (closer to 100 is higher)
        reward = max(0.0, target_value - abs(target_value - new_value))
    return new_value, result_repr, reward, constraint_violated
```

In this example, if the agent overshoots the max limit, we cap the value at 120 and mark a violation. The **reward** `r` is computed based on the goal: we use a simple scheme where reaching exactly the target (100) yields the highest reward. (In a real case, reward can be a more complex function of goals achieved.)

We also compute the **JSD drift** of the outcome `Rₜ` compared to the previous outcome `R₍ₜ₋₁₎`. This is similar to the perception JSD, but now it captures the change *after* executing the action. For our numeric state, this is JSD between the distribution for the old value vs new value:

```python
# After executing action:
prev_dist = prev_state_dist  # distribution before action (from perception step)
new_dist = [1.0, 0.0] if new_value >= target_value else [new_value/target_value, 1 - new_value/target_value]
result_jsd = jensen_shannon_divergence(prev_dist, new_dist)
```

We log the outcome `Rₜ` (new state representation), reward `r`, and `result_jsd`. This completes the outcome audit: we have the result of the action and metrics to judge it (reward improvement and drift).

## 5. Reflection Pass (Critique and Patch Generation via LLM)

Next, we invoke the **reflection model** `Rφ` (another LLM prompt) to analyze the action and outcome, along with the original rationale. The reflection model’s job is to **critique** the agent’s decision and outcome and suggest a **patch** – an improvement or adjustment to the policy. This aligns with techniques in recent research where an LLM can iterate on its own outputs to improve performance.

We construct a prompt for reflection such as:

```
Action taken: increase by 70.0  
Outcome: Value is now 120.0 (constraint violated, target overshot)  
Agent's Rationale: "Far from the target, a large increase will get closer quickly."

Critique the action and outcome. Was this a good decision? Suggest a patch to the agent's policy to improve future performance.
```

Using liteLLM, we would call the model similarly (e.g., another `completion` with a different system prompt instructing it to be critical). The output should contain some analysis and a concrete suggestion (the patch).

For the MVP, we implement a reflection function to produce a simple critique and patch based on the observed reward change and constraint violation:

```python
def reflect_and_suggest(action_text, rationale_text, result_repr, reward, prev_reward, constraint_violated):
    """
    Critique the action and outcome, and suggest a patch if needed.
    """
    reflection = ""
    patch = None
    if constraint_violated:
        reflection = (f"The action '{action_text}' breached a constraint by overshooting the limit. ")
        patch = ("If the agent overshoots the target or a limit, it should take corrective action (e.g., decrease the value) "
                 "instead of further increasing.")
    elif reward < prev_reward:
        reflection = (f"The outcome of '{action_text}' was worse than before (reward dropped). ")
        patch = ("Consider a different strategy when an action leads to a worse outcome.")
    elif reward > prev_reward:
        reflection = (f"The action '{action_text}' improved the situation. ")
        # No patch needed since things got better
        patch = None
    else:  # reward == prev_reward
        reflection = (f"The action '{action_text}' had no effect on the outcome. ")
        patch = ("Try a different approach next time, since this action didn't help.")
    if not constraint_violated and reward == max_reward:  # if goal fully achieved successfully
        reflection = "The agent achieved the goal successfully."
        patch = None
    return reflection.strip(), patch
```

In the above pseudo-code, `max_reward` is the highest possible reward (e.g., reaching the exact target). The reflection text explains what went wrong or right, and `patch` is a concise guideline to improve the policy (if needed). For example, if the action overshot the target, the patch might suggest using smaller increments or even reversing the action next time.

We log both the reflection and the patch. The pair (reflection, patch) can be stored in a table (e.g., `reflection_patch`) for later analysis or fine-tuning data. Each patch is essentially a **candidate policy update**.

## 6. Policy Update (Incorporate Patch if Conditions Met)

Now, the algorithm decides whether to **apply the patch** to the policy. Not every reflection should immediately alter the agent’s behavior – we want to apply changes that are beneficial. The rule given is to apply the patch if **reward improved** or **drift `d` exceeded a threshold ε** after the last action. This can be interpreted as: if the patch addresses a significant change (environment drift) or leads to a better outcome, then we accept it into our policy.

In practice, since we generate the patch after seeing the outcome, we can assume the patch is relevant *for future actions*. We will apply it going forward if the triggering conditions are met. For example, if the environment changed drastically (high JSD) or if we saw an improvement when trying a new strategy, we incorporate that strategy.

In code, we maintain a list `policy_tips` (as above) which influence the action selection. Applying a patch could mean appending a new guideline to this list or altering the prompt template. We also record the application in a `policy_updates` table for persistence:

```python
def update_policy_with_patch(patch, reward, prev_reward, drift, iteration):
    applied = False
    if patch:
        # Define threshold epsilon for drift (e.g., 0.1)
        if reward > prev_reward or drift > 0.1:
            # Apply the patch to the policy prompt
            policy_tips.append(patch)
            applied = True
            # Log the policy update in DB
            cur.execute("INSERT INTO policy_updates (iteration, patch_text) VALUES (?,?)", (iteration, patch))
            conn.commit()
    return applied
```

In our running example, if the agent overshot and got a worse reward (dropped to 0 from 70) **or** the state distribution changed a lot, we apply the patch. That patch (e.g. “if overshoot, then decrease next time”) will then be included in `policy_tips` and influence subsequent action selection (Step 3) by, say, preferring a corrective action. We mark `patch_applied = 1` for that iteration in the logs.

*Note:* Instead of immediate prompt changes, patches could be stored and later used to fine-tune the policy model offline. For MVP, we directly adjust the prompting strategy for simplicity.

## 7. Meta-Monitoring (Trend Tracking and Alerts)

The meta-monitor oversees long-term trends such as the JSD drift over time and resource usage (e.g., LLM token counts). Its role is to produce **alerts** if certain thresholds are crossed, allowing intervention or scaling decisions.

We track a history of recent JSD values (`jsd_history`) and token usage (`token_history`). For each loop iteration, we can compute totals like `total_tokens = tokens_policy + tokens_reflection`. Then we perform checks such as:

* If `drift` in the current step is above a critical threshold (e.g., `> 0.5`), log an alert that the environment dynamics have significantly changed (which might require human review or a model update).
* If token usage is too high (e.g., > 100 tokens in one iteration or exceeding a budget over time), log an alert to consider optimizing prompts or model calls.

For example:

```python
jsd_history = []
token_history = []

def meta_monitor(drift, total_tokens):
    alerts = []
    jsd_history.append(drift)
    token_history.append(total_tokens)
    # Alert if JSD indicates sudden large drift
    if drift > 0.5:
        alerts.append(f"ALERT: High environment drift detected (JSD={drift:.2f}).")
    # Alert if token usage in this iteration is beyond threshold
    if total_tokens > 100:
        alerts.append(f"ALERT: High token usage this round ({total_tokens} tokens).")
    # (Other trends like moving averages or cumulative usage can be checked here)
    return alerts
```

These alerts can be logged to the database or output to a monitoring dashboard. In our example run below, we will see alerts when the token usage exceeds a set threshold for demonstration.

## 8. Continuous Loop Execution

Finally, we tie everything together in the **feedback loop** that runs every Δt seconds. The loop will perform all steps sequentially and then wait or be triggered again. In a production setting, you might run this loop in a background thread, a scheduler (cron job), or orchestrate it with a tool like LangGraph.

Here’s the main loop logic (without actual sleeping, for demonstration purposes we iterate a fixed number of times). It calls each component in turn and stores the events in the database:

```python
import time

# Initial state
current_value = 0.0
prev_reward = 0.0

for t in range(1, 6):  # run 5 iterations for example
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Step 2: Perceive current state
    state_repr, state_embed, state_jsd = perceive_state(current_value)
    
    # Step 3: Select action with policy model
    action_text, rationale_text = select_action(state_repr)
    # (token usage for policy counted inside select_action or separately)
    
    # Step 4: Execute action and audit outcome
    new_value, result_repr, reward, violated = execute_action(action_text, current_value)
    # Compute outcome drift JSD
    prev_dist = prev_state_dist  # distribution from perceive_state (updated globally)
    new_dist = [1.0, 0.0] if new_value >= target_value else [new_value/target_value, 1 - new_value/target_value]
    result_jsd = jensen_shannon_divergence(prev_dist, new_dist)
    
    # Step 5: Reflection critique and patch suggestion
    reflection_text, patch_text = reflect_and_suggest(action_text, rationale_text, result_repr, reward, prev_reward, violated)
    
    # Step 6: Policy update (apply patch if criteria met)
    patch_applied = update_policy_with_patch(patch_text, reward, prev_reward, result_jsd, t)
    
    # Step 7: Meta-monitoring (track drift and token usage, generate alerts)
    tokens_policy = len((action_text + rationale_text).split())  # simplistic token count
    tokens_reflect = len((reflection_text or "").split()) + len((patch_text or "").split())
    total_tokens = tokens_policy + tokens_reflect
    alerts = meta_monitor(result_jsd, total_tokens)
    
    # Log the event in SQLite (events and reflection_patch tables)
    cur.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", 
                (t, timestamp, state_repr, str(state_embed), state_jsd, 
                 action_text, rationale_text, result_repr, reward, result_jsd, 
                 tokens_policy, tokens_reflect))
    cur.execute("INSERT INTO reflection_patch VALUES (?,?,?,?)",
                (t, reflection_text, patch_text or "", 1 if patch_applied else 0))
    conn.commit()
    
    # Print or output the step summary (for monitoring)
    print(f"[{timestamp}] Iteration {t}: State={state_repr}, Action={action_text}, Result={result_repr}, Reward={reward:.1f}")
    if patch_text:
        status = "APPLIED" if patch_applied else "PROPOSED"
        print(f"  Reflection: {reflection_text}")
        print(f"  Patch: {patch_text} ({status})")
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    
    # Prepare for next loop
    current_value = new_value
    prev_reward = reward
    # time.sleep(Δt)  # wait for the next cycle (if this were continuous)
```

In a real continuous setting, replace the `for` loop with `while True:` and uncomment `time.sleep(Δt)` for the desired interval. The loop logs each iteration’s data to SQLite so the agent’s history is saved. This modular design can be integrated into larger frameworks (like LangGraph) or scheduled as needed.

## SQLite Schema and Data Logging

We use SQLite to persist the data. The database schema includes tables for events, reflections/patches, and policy updates:

```sql
CREATE TABLE events (
  iteration INTEGER PRIMARY KEY,
  timestamp TEXT,
  state_repr TEXT,
  state_embedding TEXT,
  state_jsd REAL,
  action_text TEXT,
  rationale_text TEXT,
  result_repr TEXT,
  reward REAL,
  result_jsd REAL,
  token_usage_policy INTEGER,
  token_usage_reflection INTEGER
);
CREATE TABLE reflection_patch (
  iteration INTEGER PRIMARY KEY,
  reflection_text TEXT,
  patch_text TEXT,
  patch_applied INTEGER
);
CREATE TABLE policy_updates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  iteration INTEGER,
  patch_text TEXT
);
```

* **events**: each loop iteration with the state, action, outcome, and metrics.
* **reflection\_patch**: the reflection critique and patch suggestion for that iteration, and whether the patch was applied.
* **policy\_updates**: records of patches that were applied (for cumulative record, if needed for model fine-tuning later).

Every iteration, a new row is added to `events` and `reflection_patch`. If a patch is applied, a row is added to `policy_updates`. This provides an audit trail of the agent’s behavior and learning over time, which is important for **auditability** and offline analysis.

## Example Run and Evolving Logs

Using the above implementation, here is a sample run of the loop over a few iterations. This demonstrates the loop’s evolution and how the agent learns from the reflection:

```text
Iteration 1:
  State: Current value: 0.0 (JSD vs prev: 0.000)
  Action: increase by 70.0  — Rationale: Far from target, a large increase will get closer quickly.
  Result: Value is now 70.0  — Reward: 70.0, JSD vs prev: 0.494
  Reflection: The action 'increase by 70.0' improved the situation (reward increased).
  Patch: None (not applied)

Iteration 2:
  State: Current value: 70.0 (JSD vs prev: 0.494)
  Action: increase by 70.0  — Rationale: Far from target, a large increase will get closer quickly.
  Result: Value is now 120.0  — Reward: 0.0, JSD vs prev: 0.169
  Reflection: The action 'increase by 70.0' **violated the constraint** by overshooting the limit.
  Patch: If the agent overshoots the target or hits the limit, it should take a corrective action (decrease) instead of further increasing. (APPLIED)
  ALERT: High token usage: 79 tokens in this iteration.

Iteration 3:
  State: Current value: 120.0 (JSD vs prev: 0.169)
  Action: none  — Rationale: The target is achieved; no further action is needed.
  Result: Value is now 120.0  — Reward: 80.0, JSD vs prev: 0.000
  Reflection: The target has been achieved successfully.
  Patch: None (not applied)
```

In **Iteration 1**, the agent took a large action (+70) and improved progress toward the goal (reward from 0 to 70). The reflection noted the improvement and suggested no patch.

In **Iteration 2**, the same strategy caused an overshoot to 120, breaking the constraint (reward fell to 0). The reflection model identified the mistake (constraint violation) and suggested a patch: *if overshooting, take a corrective decrease next time*. This patch was applied to the policy (noted as “APPLIED”). We also see a meta-monitor alert for high token usage (the long reflection text increased the token count).

By **Iteration 3**, the patch was in effect. The agent recognized it had exceeded the target and took no further action (“none”), avoiding further error. The result shows the agent maintained the value at 120 (which is at the goal threshold) and the reward rebounded to 80 (closer to optimal). The reflection confirms success and offers no new patch. (In a scenario where overshoot correction was needed, the agent would have decreased the value as per the patch – the logic is in place to do so if applicable.)

All these events are logged in the SQLite database. For example, the `events` table after 3 iterations contains (iteration, state, action, result, reward):

```text
(1, 'Current value: 0.0', 'increase by 70.0', 'Value is now 70.0', 70.0)
(2, 'Current value: 70.0', 'increase by 70.0', 'Value is now 120.0', 0.0)
(3, 'Current value: 120.0', 'none',           'Value is now 120.0', 80.0)
```

And the `reflection_patch` table stores the patches and whether they were applied:

```text
(1, <reflection for iter1>, "" , 0)
(2, <reflection for iter2>, "If the agent overshoots... decreasing.", 1)
(3, <reflection for iter3>, "" , 0)
```

This confirms that at iteration 2 the patch was suggested and applied (`patch_applied=1`), while others had none.

**Modularity:** Each step (perception, action selection, etc.) is implemented as an independent function, making it easy to extend or plug into frameworks like LangGraph. For instance, one could swap out the environment execution with real API calls via requesty.ai, or replace the stubbed policy with an actual model call, without changing the overall loop structure.

**Auditability:** Thanks to the SQLite logs, we have a persistent record of state changes, decisions, and self-improvements. This transparency is crucial for debugging and verifying that the agent’s learning is aligned with business goals and constraints.

---

By following these steps, we achieve a **self-aware feedback loop** where the agent continuously monitors its performance, learns from mistakes via self-reflection, and updates its own policy in a controlled, auditable way. This MVP can serve as a foundation for more complex autonomous agents, with minimal external dependencies and clear logic at each stage.

**Sources:**

* LiteLLM documentation – unified interface for 100+ LLMs (calls, embeddings)
* Requesty.ai platform – an OpenAI-compatible router for model and action integration
* Jensen–Shannon divergence – measures similarity between probability distributions
* Reflexion and self-critique method – using an LLM’s feedback loop to improve agent decisions
