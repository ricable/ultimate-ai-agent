# safla_main.py
import time
import sqlite3
from safla_modules import (
    perceive_state, select_action, execute_action,
    reflect_and_suggest, update_policy_with_patch, meta_monitor,
    init_db, insert_goal_schema, insert_event, insert_reflection_patch, insert_policy_update
)

# Configuration
target_value = 100.0
max_value = 120.0
prev_state_dist = None
current_value = 0.0
prev_reward = 0.0
policy_tips = []

# Initialize database
conn, cur = init_db()
insert_goal_schema(cur)
conn.commit()

# Run feedback loop
for iteration in range(1, 6):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Step 2: Perception
    state_repr, state_dist, state_jsd = perceive_state(current_value, target_value)

    # Step 3: Action Selection
    action_text, rationale_text = select_action(state_repr, target_value, max_value, policy_tips)

    # Step 4: Outcome Audit
    new_value, result_repr, reward, violated = execute_action(action_text, current_value, max_value, target_value)
    new_dist = [1.0, 0.0] if new_value >= target_value else [new_value/target_value, 1 - new_value/target_value]
    result_jsd = meta_monitor.jsd(state_dist, new_dist)

    # Step 5: Reflection
    reflection_text, patch_text = reflect_and_suggest(action_text, rationale_text, result_repr, reward, prev_reward, violated)

    # Step 6: Policy Update
    patch_applied = update_policy_with_patch(patch_text, reward, prev_reward, result_jsd, iteration, policy_tips, cur)

    # Step 7: Meta-Monitor
    tokens_policy = len((action_text + rationale_text).split())
    tokens_reflect = len((reflection_text or "").split()) + len((patch_text or "").split())
    total_tokens = tokens_policy + tokens_reflect
    alerts = meta_monitor.check(result_jsd, total_tokens)

    # Step 8: Log and Iterate
    insert_event(cur, iteration, timestamp, state_repr, str(state_dist), state_jsd,
                 action_text, rationale_text, result_repr, reward, result_jsd,
                 tokens_policy, tokens_reflect)
    insert_reflection_patch(cur, iteration, reflection_text, patch_text, patch_applied)
    if patch_applied:
        insert_policy_update(cur, iteration, patch_text)
    conn.commit()

    print(f"[{timestamp}] Iteration {iteration}: State={state_repr}, Action={action_text}, Result={result_repr}, Reward={reward:.1f}")
    if patch_text:
        status = "APPLIED" if patch_applied else "PROPOSED"
        print(f"  Reflection: {reflection_text}\n  Patch: {patch_text} ({status})")
    if alerts:
        for alert in alerts:
            print(f"  {alert}")

    current_value = new_value
    prev_reward = reward
