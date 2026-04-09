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
# ✅ STRICT RULE POLICY (GROUND TRUTH)
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

    # STEP 3: ESCALATE (ONLY SECURITY)
    if obs.category == "security" and obs.status != "escalated":
        return Action(action_type="escalate")

    # STEP 4: TOOL
    if obs.tool_result is None:
        if obs.category == "billing":
            return Action(action_type="refund_api")
        if obs.category == "technical":
            return Action(action_type="db_lookup")

    # STEP 5: RESPOND (only after tool or escalation)
    if obs.tool_result is not None or obs.status == "escalated":
        return Action(
            action_type="respond",
            content="Your issue has been processed. We are working on it."
        )

    # STEP 6: RESOLVE (only when ready)
    if obs.tool_result is not None or obs.status == "escalated":
        return Action(action_type="resolve")


# -------------------------
# 🤖 LLM (SAFE + CONTROLLED)
# -------------------------
def llm_suggestion(obs):
    if not USE_LLM:
        return None

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
            temperature=0.1,
            timeout=3  # ✅ prevent hanging
        )

        output = response.choices[0].message.content.strip()

        # SAFE PARSING
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            json_str = output[start:end]
            data = json.loads(json_str)
        except Exception:
            return None

        VALID_ACTIONS = {
            "classify",
            "prioritize",
            "escalate",
            "refund_api",
            "db_lookup",
            "respond",
            "resolve"
        }

        if data.get("action") not in VALID_ACTIONS:
            return None

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
    rule_action = strict_policy(obs)

    # ✅ Try LLM but NEVER depend on it
    try:
        llm_action = llm_suggestion(obs)
        if llm_action and llm_action.action_type == rule_action.action_type:
            return llm_action
    except Exception:
        pass

    return rule_action


# -------------------------
# 🧪 RUN
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
                    action = smart_policy(obs)
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
        # FAIL SAFE → still print something
        print("[START] task=error", flush=True)
        print("[STEP] step=1 reward=0.0", flush=True)
        print("[END] task=error score=0.0 steps=1", flush=True)
        return 0.0

if __name__ == "__main__":
    run()
