import os
import json
from openai import OpenAI
from env.environment import SupportOpsEnv
from env.models import Action

# ---------------------------
# ENV VARIABLES
# ---------------------------
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ---------------------------
# BASELINE POLICY (LLM)
# ---------------------------
def smart_policy(obs):
    conversation_text = "\n".join([m.content for m in obs.conversation])

    prompt = f"""
You are a customer support AI agent.

Steps:
1. classify → billing / technical / security
2. prioritize → low / medium / high / urgent
3. use tool if needed
4. escalate only for security
5. respond if needed
6. resolve

State:
Category: {obs.category}
Priority: {obs.priority}
Status: {obs.status}
SLA Remaining: {obs.sla_remaining}

Conversation:
{conversation_text}

Actions:
- classify:<category>
- prioritize:<level>
- escalate
- refund_api
- db_lookup
- respond:<message>
- resolve

Return ONLY one action.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )

        output = response.choices[0].message.content.strip().lower()

    except Exception as e:
        print("LLM Error:", e)
        return Action(action_type="resolve")

    # Parse output
    try:
        if output.startswith("classify"):
            return Action(action_type="classify", content=output.split(":")[1].strip())

        if output.startswith("prioritize"):
            return Action(action_type="prioritize", content=output.split(":")[1].strip())

        if output.startswith("respond"):
            return Action(action_type="respond", content=output.split(":", 1)[1].strip())

        if "refund_api" in output:
            return Action(action_type="refund_api")

        if "db_lookup" in output:
            return Action(action_type="db_lookup")

        if "escalate" in output:
            return Action(action_type="escalate")

        if "resolve" in output:
            return Action(action_type="resolve")

    except:
        pass

    return Action(action_type="resolve")


# ---------------------------
# RUN BASELINE
# ---------------------------
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

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return avg


if __name__ == "__main__":
    run()