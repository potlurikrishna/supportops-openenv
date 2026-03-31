import os
import json
from env.environment import SupportOpsEnv
from env.models import Action
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

MODEL_NAME = os.getenv("MODEL_NAME")


# -------------------------
# 🧠 STRICT RULE ENGINE (PRIMARY)
# -------------------------

def smart_policy(obs):
    text = " ".join([m.content for m in obs.conversation]).lower()

    # STEP 1: CLASSIFY
    if obs.category is None:
        if "unauthorized" in text or "fraud" in text or "hacked" in text:
            return Action(action_type="classify", content="security")

        if "charged" in text or "payment" in text or "refund" in text:
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
    if len(obs.conversation) < 2:
        return Action(
            action_type="respond",
            content="We are resolving your issue."
        )

    # STEP 6: RESOLVE
    return Action(action_type="resolve")
# -------------------------
# 🤖 OPTIONAL LLM (SAFE USE)
# -------------------------
def llm_suggestion(obs):
    try:
        prompt = f"""
You are a support agent.

State:
category={obs.category}
priority={obs.priority}
status={obs.status}

Return ONLY JSON:
{{"action": "...", "content": "..."}}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        output = response.choices[0].message.content.strip()
        data = json.loads(output)

        return Action(
            action_type=data["action"],
            content=data.get("content")
        )

    except Exception:
        return None


# -------------------------
# 🚀 FINAL POLICY (SAFE HYBRID)
# -------------------------
def smart_policy(obs):
    # ALWAYS get correct action first
    correct_action = strict_policy(obs)

    # Try LLM (optional improvement)
    llm_action = llm_suggestion(obs)

    # If LLM gives SAME valid action → use it
    if llm_action and llm_action.action_type == correct_action.action_type:
        return llm_action

    # Otherwise → ALWAYS fallback to correct logic
    return correct_action


# -------------------------
# 🧪 RUN
# -------------------------
def run():
    env = SupportOpsEnv(seed=42)
    scores = []

    for i in range(3):
        obs = env.reset()
        final_score = 0

        print(f"\n===== TASK {i+1} =====")

        for step in range(6):
            action = smart_policy(obs)
            obs, reward, done, _ = env.step(action)

            print(f"Step {step+1} | Action: {action.action_type} | Score: {reward.score:.2f}")

            final_score = reward.score

            if done:
                break

        print(f"Final Task Score: {final_score:.2f}")
        scores.append(final_score)

    avg = sum(scores) / len(scores)
    print(f"\n🔥 Average Score: {avg:.2f}")

    return avg


if __name__ == "__main__":
    run()