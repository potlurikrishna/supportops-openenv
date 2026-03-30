import os
import json
from env.environment import SupportOpsEnv
from env.models import Action
from openai import OpenAI

# client = OpenAI(
#     base_url=os.getenv("API_BASE_URL"),
#     api_key=os.getenv("HF_TOKEN")
# )


def smart_policy(obs):
    text = " ".join([m.content for m in obs.conversation]).lower()

    # 1. CLASSIFY
    if obs.category is None:
        if any(k in text for k in ["charged", "payment", "refund"]):
            return Action(action_type="classify", content="billing")
        if any(k in text for k in ["crash", "bug", "error"]):
            return Action(action_type="classify", content="technical")
        if any(k in text for k in ["unauthorized", "hacked", "fraud"]):
            return Action(action_type="classify", content="security")
        return Action(action_type="classify", content="billing")

    # 2. PRIORITY
    if obs.priority is None:
        if obs.category == "security":
            return Action(action_type="prioritize", content="urgent")
        if obs.category == "technical":
            return Action(action_type="prioritize", content="high")
        return Action(action_type="prioritize", content="medium")

    # 3. ESCALATION (MANDATORY FOR SECURITY)
    if obs.category == "security" and not getattr(obs, "status", None) == "escalated":
        return Action(action_type="escalate")

    # 4. TOOL (MANDATORY BEFORE RESOLVE)
    if obs.tool_result is None:
        if obs.category == "billing":
            return Action(action_type="refund_api")
        if obs.category == "technical":
            return Action(action_type="db_lookup")

    # 5. CONVERSATION STEP
    if len(obs.conversation) < 2:
        return Action(
            action_type="respond",
            content="We are analyzing your issue."
        )

    # 6. FINAL CHECK BEFORE RESOLVE
    # DO NOT resolve unless all conditions are satisfied

    if obs.category == "security" and not getattr(obs, "status", None) == "escalated":
        return Action(action_type="escalate")

    if obs.tool_result is None:
        return Action(action_type="db_lookup" if obs.category == "technical" else "refund_api")

    # 7. ONLY NOW RESOLVE
    return Action(action_type="resolve")

def run():
    env = SupportOpsEnv(seed=42)
    scores = []
    report = []

    for i in range(3):
        obs = env.reset()
        final_score = 0
        steps_log = []

        for step in range(6):
            action = smart_policy(obs)
            obs, reward, done, _ = env.step(action)

            final_score = reward.score

            steps_log.append({
                "step": step + 1,
                "action": action.action_type,
                "score": reward.score,
                "breakdown": reward.breakdown
            })

            if done:
                break

        scores.append(final_score)

        report.append({
            "task_id": obs.ticket_id,
            "final_score": final_score,
            "steps": steps_log
        })

        print(f"Task {i+1} Score: {final_score:.2f}")

    avg = sum(scores) / len(scores)
    print("Baseline Score:", avg)

    # Save report
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return avg


if __name__ == "__main__":
    run()