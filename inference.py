import sys
from env.environment import SupportOpsEnv
from env.models import Action


# -------------------------
# ✅ STRICT RULE POLICY
# -------------------------
def strict_policy(obs):
    try:
        text = " ".join([getattr(m, "content", "") for m in (obs.conversation or [])]).lower()
    except Exception:
        text = ""

    category = getattr(obs, "category", None)
    priority = getattr(obs, "priority", None)
    status = getattr(obs, "status", None)
    tool_result = getattr(obs, "tool_result", None)

    # STEP 1: CLASSIFY
    if category is None:
        if any(k in text for k in ["unauthorized", "fraud", "hacked"]):
            return Action(action_type="classify", content="security")

        if any(k in text for k in ["charged", "payment", "refund"]):
            return Action(action_type="classify", content="billing")

        return Action(action_type="classify", content="technical")

    # STEP 2: PRIORITY
    if priority is None:
        if category == "security":
            return Action(action_type="prioritize", content="urgent")
        if category == "technical":
            return Action(action_type="prioritize", content="high")
        return Action(action_type="prioritize", content="medium")

    # STEP 3: ESCALATE
    if category == "security" and status != "escalated":
        return Action(action_type="escalate")

    # STEP 4: TOOL
    if tool_result is None:
        if category == "billing":
            return Action(action_type="refund_api")
        if category == "technical":
            return Action(action_type="db_lookup")

    # STEP 5: RESPOND
    if tool_result is not None or status == "escalated":
        return Action(
            action_type="respond",
            content="Your issue has been processed. We are working on it."
        )

    # STEP 6: RESOLVE
    return Action(action_type="resolve")

# -------------------------
# 🚀 RUN (FULLY SAFE)
# -------------------------
def run():
    try:
        env = SupportOpsEnv(seed=42)
        scores = []

        for i in range(3):
            obs = env.reset()
            final_score = 0
            steps = 0
            task_name = f"task_{i+1}"

            print(f"[START] task={task_name}", flush=True)

            for step in range(6):
                try:
                    action = strict_policy(obs)
                    obs, reward, done, _ = env.step(action)

                    steps += 1
                    final_score = reward.score

                    print(f"[STEP] step={steps} reward={final_score:.4f}", flush=True)

                    if done:
                        break

                except Exception:
                    break

            scores.append(final_score)

            print(f"[END] task={task_name} score={final_score:.4f} steps={steps}", flush=True)

        return sum(scores) / len(scores)

    except Exception:
        # FAIL-SAFE OUTPUT (never crash)
        print("[START] task=error", flush=True)
        print("[STEP] step=1 reward=0.0", flush=True)
        print("[END] task=error score=0.0 steps=1", flush=True)
        return 0.0


if __name__ == "__main__":
    run()
