import os
import json
from env.environment import SupportOpsEnv
from env.models import Action
from openai import OpenAI

# ✅ Load from HF secrets (DO NOT hardcode)
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

MODEL_NAME = os.getenv("MODEL_NAME")


# -------------------------
# 🔥 LLM DECISION FUNCTION
# -------------------------
def llm_decide(obs):
    text = " ".join([m.content for m in obs.conversation])

    prompt = f"""
You are an AI customer support agent.

STRICT RULES:
- Follow EXACT order:
  1. classify
  2. prioritize
  3. escalate (ONLY if security)
  4. tool (billing → refund_api, technical → db_lookup)
  5. respond
  6. resolve

- NEVER repeat same step twice
- NEVER skip order
- NEVER loop actions

CURRENT STATE:
category: {obs.category}
priority: {obs.priority}
status: {obs.status}
tool_result: {obs.tool_result}

USER ISSUE:
{text}

Return ONLY JSON:
{{
  "action": "...",
  "content": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
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
# 🧠 SAFETY FALLBACK (CRITICAL)
# -------------------------
def rule_fallback(obs):
    text = " ".join([m.content for m in obs.conversation]).lower()

    # CLASSIFY
    if obs.category is None:
        if "unauthorized" in text or "fraud" in text:
            return Action(action_type="classify", content="security")
        if "charged" in text or "payment" in text:
            return Action(action_type="classify", content="billing")
        return Action(action_type="classify", content="technical")

    # PRIORITY
    if obs.priority is None:
        if obs.category == "security":
            return Action(action_type="prioritize", content="urgent")
        if obs.category == "technical":
            return Action(action_type="prioritize", content="high")
        return Action(action_type="prioritize", content="medium")

    # ESCALATE
    if obs.category == "security" and obs.status != "escalated":
        return Action(action_type="escalate")

    # TOOL
    if obs.tool_result is None:
        if obs.category == "billing":
            return Action(action_type="refund_api")
        if obs.category == "technical":
            return Action(action_type="db_lookup")

    # RESPOND
    if len(obs.conversation) < 2:
        return Action(action_type="respond", content="We are resolving your issue.")

    return Action(action_type="resolve")


# -------------------------
# 🚀 FINAL POLICY (LLM + GUARDRAILS)
# -------------------------
def smart_policy(obs):
    action = llm_decide(obs)

    # ❌ If LLM fails → fallback
    if action is None:
        return rule_fallback(obs)

    # ❌ Prevent loops / invalid actions
    if obs.category is not None and action.action_type == "classify":
        return rule_fallback(obs)

    if obs.priority is not None and action.action_type == "prioritize":
        return rule_fallback(obs)

    if obs.category != "security" and action.action_type == "escalate":
        return rule_fallback(obs)

    if obs.tool_result is not None and action.action_type in ["refund_api", "db_lookup"]:
        return rule_fallback(obs)

    return action


# -------------------------
# 🧪 RUN (BASELINE)
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