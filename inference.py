import os
import gradio as gr
from openai import OpenAI
from env.environment import SupportOpsEnv
from env.models import Action

# ---------------------------
# ENV VARIABLES (MUST SET)
# ---------------------------
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ---------------------------
# LLM AGENT
# ---------------------------
def llm_policy(obs):
    conversation_text = "\n".join([m.content for m in obs.conversation])

    prompt = f"""
You are a customer support AI agent.

Your job:
- Classify issue into: billing / technical / security
- Set priority: low / medium / high / urgent
- Use tools if needed
- Escalate ONLY for security
- Resolve when ready

Current State:
Category: {obs.category}
Priority: {obs.priority}
Status: {obs.status}
SLA Remaining: {obs.sla_remaining}

Conversation:
{conversation_text}

Available Actions:
- classify:<category>
- prioritize:<level>
- escalate
- refund_api
- db_lookup
- respond:<message>
- resolve

Respond with ONLY ONE action.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict decision-making agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )

        output = response.choices[0].message.content.strip().lower()

    except Exception as e:
        print("LLM Error:", e)
        return Action(action_type="resolve")

    # ---------------------------
    # PARSE OUTPUT → ACTION
    # ---------------------------
    try:
        if output.startswith("classify"):
            return Action(action_type="classify", content=output.split(":")[1])

        if output.startswith("prioritize"):
            return Action(action_type="prioritize", content=output.split(":")[1])

        if output.startswith("respond"):
            return Action(action_type="respond", content=output.split(":", 1)[1])

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

    # fallback
    return Action(action_type="resolve")


# ---------------------------
# RUN SIMULATION
# ---------------------------
def run_simulation():
    env = SupportOpsEnv(seed=42)
    logs = []
    scores = []

    for i in range(3):
        obs = env.reset()
        logs.append(f"\n===== TASK {i+1} =====")

        final_score = 0

        for step in range(6):
            action = llm_policy(obs)
            obs, reward, done, _ = env.step(action)

            logs.append(
                f"Step {step+1} | Action: {action.action_type} | Score: {reward.score:.2f}"
            )

            final_score = reward.score

            if done:
                break

        logs.append(f"Final Task Score: {final_score:.2f}")
        scores.append(final_score)

    avg = sum(scores) / len(scores)
    logs.append(f"\n🔥 Average Score: {avg:.2f}")

    return "\n".join(logs)


# ---------------------------
# GRADIO UI
# ---------------------------
demo = gr.Interface(
    fn=run_simulation,
    inputs=[],
    outputs="text",
    title="SupportOps OpenEnv (LLM Agent)",
    description="AI agent solving real-world customer support tickets using tools, escalation, and SLA."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)