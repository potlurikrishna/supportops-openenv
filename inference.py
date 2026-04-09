import sys
from env.environment import SupportOpsEnv
from env.models import Action


# -------------------------
# ✅ STRICT RULE POLICY
# -------------------------
def strict_policy(obs):
    text = " ".join([m.content for m in obs.conversation]).lower()

    # STEP 1: CLASSIFY
    if obs.category is None:
        if any(k in text for k in ["unauthorized", "fraud", "hacked"]):
            return Action(action_type="classify", content="security")

        if any(k in text for k in ["charged", "payment", "refund"]):
            return Action(action_type="classify", content="billing")

        return Action(action_type="classify", content="technical")

    # STEP 2: PRIORITY
    if obs.priority is None:
        if obs.category == "security":
            return Action(action_type="prioritize", content="urgent")
        if obs.category == "technical":
            return Action(action_type="prioritize", content="high")
        return Action(action_type="prioritize", content="medium")

    # STEP 3: ESCALATE
    if obs.category == "security" and obs.status != "escalated":
        return Action(action_type="escalate")

    # STEP 4: TOOL
    if obs.tool_result is None:
        if obs.category == "billing":
            return Action(action_type="refund_api")
        if obs.category == "technical":
            return Action(action_type="db_lookup")

    # STEP 5: RESPOND
    if obs.tool_result is not None or obs.status == "escalated":
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
